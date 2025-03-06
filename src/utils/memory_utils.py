"""
Utilities for memory management and resource cleanup.
"""

import gc
import logging
import os
import weakref

import numpy as np
import psutil
import tensorflow as tf

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


def cleanup_tf_session():
    """
    Clean up TensorFlow session resources.
    More selective than just clear_session().
    """
    tf.keras.backend.clear_session()
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
