"""
Memory management utilities for efficient training.
"""

import os
import gc
import psutil
import numpy as np
import logging
import time
import weakref
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class WeakRefCache:
    """
    Cache implementation using weak references to allow garbage collection
    of cached objects when memory pressure increases.
    """

    def __init__(self):
        self._cache = weakref.WeakValueDictionary()

    def get(self, key):
        """Get item from cache if it exists"""
        return self._cache.get(key)

    def set(self, key, value):
        """Store item in cache with weak reference"""
        self._cache[key] = value
        return value

    def clear(self):
        """Clear all items from cache"""
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics
    """
    # Get process memory info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    return {
        'process_rss_gb': memory_info.rss / 1e9,  # Resident Set Size in GB
        'process_vms_gb': memory_info.vms / 1e9,  # Virtual Memory Size in GB
        'system_used_percent': system_memory.percent,
        'system_available_gb': system_memory.available / 1e9,
        'system_total_gb': system_memory.total / 1e9
    }


def adaptive_memory_clean(level: str = "medium") -> Dict[str, float]:
    """
    Clean memory adaptively based on specified level.
    
    Args:
        level: Cleaning level ("small", "medium", "large")
        
    Returns:
        Dictionary with memory statistics before and after cleaning
    """
    # Get memory usage before cleaning
    before = get_memory_usage()
    
    # Always run garbage collection
    gc.collect()
    
    if level in ["medium", "large"]:
        # Clear Numpy cache
        np.clear_cffi_cache()
    
    if level == "large":
        # Try to release TensorFlow memory
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
            # Reset GPU memory if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.reset_memory_stats(gpu)
        except:
            pass
        
        # Try to release PyTorch memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    # Get memory usage after cleaning
    after = get_memory_usage()
    
    # Calculate savings
    savings = {
        'rss_gb': before['process_rss_gb'] - after['process_rss_gb'],
        'system_available_gb': after['system_available_gb'] - before['system_available_gb']
    }
    
    logger.debug(f"Memory cleaning ({level}) - Saved: {savings['rss_gb']:.2f} GB process RSS")
    
    return {
        'before': before,
        'after': after,
        'savings': savings
    }


def limit_memory_growth(max_memory_percent: float = 90.0) -> bool:
    """
    Monitor memory usage and pause execution if it exceeds threshold.
    
    Args:
        max_memory_percent: Maximum memory usage percentage
        
    Returns:
        Whether action was taken
    """
    system_memory = psutil.virtual_memory()
    
    if system_memory.percent > max_memory_percent:
        logger.warning(f"Memory usage critical: {system_memory.percent:.1f}% - Cleaning and pausing")
        adaptive_memory_clean("large")
        
        # If still high, wait for memory to be released
        system_memory = psutil.virtual_memory()
        if system_memory.percent > max_memory_percent:
            wait_time = 5.0  # seconds
            logger.warning(f"Still high memory usage, waiting {wait_time}s")
            time.sleep(wait_time)
            return True
    
    return False


def memory_efficient_operation(func):
    """
    Decorator to make functions memory-efficient.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function that cleans memory before and after execution
    """
    def wrapper(*args, **kwargs):
        # Clean before execution
        adaptive_memory_clean("small")
        
        # Monitor memory growth during execution
        result = func(*args, **kwargs)
        
        # Clean after execution
        adaptive_memory_clean("small")
        
        return result
    
    return wrapper


def cleanup_tf_session():
    """
    Clean up TensorFlow session resources.
    More selective than just clear_session().
    """
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    gc.collect()


def selective_cleanup(variables_dict, exclude_keys=None):
    """
    Selectively clean up variables to free memory.

    Args:
        variables_dict: Dictionary of variables (usually locals() or globals())
        exclude_keys: List of keys to exclude from cleanup
    """
    exclude_keys = exclude_keys or []
    exclude_keys.extend(["self", "__builtins__", "__name__", "__file__", "__doc__"])

    for key in list(variables_dict.keys()):
        if key not in exclude_keys:
            var = variables_dict[key]
            if (
                isinstance(var, (np.ndarray, list))
                and hasattr(var, "__len__")
                and len(var) > 1000
            ):
                variables_dict[key] = None

    gc.collect()


def memory_status():
    """
    Get current memory usage status.

    Returns:
        Dict with memory usage information
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    status = {
        "rss": mem_info.rss / (1024 * 1024),  # MB
        "vms": mem_info.vms / (1024 * 1024),  # MB
        "percent": process.memory_percent(),
        "system_available": psutil.virtual_memory().available / (1024 * 1024),  # MB
        "system_percent": psutil.virtual_memory().percent,
    }

    return status


def log_memory_usage(where=""):
    """Log the current memory usage"""
    status = memory_status()
    logger.info(
        f"Memory usage {where}: "
        f"RSS={status['rss']:.1f}MB ({status['percent']:.1f}%), "
        f"System: {status['system_percent']:.1f}% used"
    )

    # Return True if memory usage is high, which can trigger cleanup
    return status["system_percent"] > 90
