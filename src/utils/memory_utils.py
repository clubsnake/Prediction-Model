"""
Memory management utilities for efficient model training.
This module provides a consolidated approach to memory management by
delegating to TrainingOptimizer methods where appropriate.
"""

import logging
import weakref


# Configure logging
logger = logging.getLogger(__name__)

# Get the global optimizer instance
from src.utils.training_optimizer import get_training_optimizer


# Keep WeakRefCache for backward compatibility
class WeakRefCache:
    """
    A cache that uses weak references to avoid memory leaks.
    Objects will be automatically removed from the cache when they are no longer
    referenced elsewhere in the code.
    """

    def __init__(self):
        self._cache = weakref.WeakValueDictionary()

    def get(self, key):
        """Get an item from the cache"""
        return self._cache.get(key)

    def set(self, key, value):
        """Store an item in the cache"""
        self._cache[key] = value
        return value

    def clear(self):
        """Clear all items from the cache"""
        self._cache.clear()

    def __len__(self):
        """Return the number of items in the cache"""
        return len(self._cache)


def cleanup_tf_session():
    """
    Clear TensorFlow session to free memory.
    Now delegates to TrainingOptimizer for a more comprehensive approach.
    """
    optimizer = get_training_optimizer()
    return optimizer.cleanup_memory(level="medium")


def log_memory_usage(tag=""):
    """
    Log current memory usage.
    Now delegates to TrainingOptimizer for unified memory tracking.

    Args:
        tag: Identifier for the memory usage log entry

    Returns:
        bool: True if memory usage is high
    """
    optimizer = get_training_optimizer()
    return optimizer.log_memory_usage(tag)


def adaptive_memory_clean(level="medium"):
    """
    Clean memory based on the specified level of cleanup intensity.

    Args:
        level: Cleanup intensity ("light", "medium", "heavy", "critical")

    Returns:
        Dict: Results of the cleanup operation
    """
    optimizer = get_training_optimizer()

    # Map "critical" to "heavy" for compatibility
    if level == "critical":
        level = "heavy"

    return optimizer.cleanup_memory(level=level)


def clear_model_cache():
    """Clear the model cache in TrainingOptimizer"""
    optimizer = get_training_optimizer()
    return optimizer.clear_model_cache()


def clear_prediction_cache():
    """Clear the prediction cache in TrainingOptimizer"""
    optimizer = get_training_optimizer()
    return optimizer.clear_prediction_cache()


def get_gpu_memory_info():
    """Get information about GPU memory usage"""
    optimizer = get_training_optimizer()
    memory_info = optimizer._get_memory_info()

    # Extract just the GPU information
    gpu_info = {k: v for k, v in memory_info.items() if k.startswith("gpu")}
    return gpu_info


def get_system_memory_info():
    """Get information about system memory usage"""
    optimizer = get_training_optimizer()
    memory_info = optimizer._get_memory_info()

    # Extract just the system information
    system_info = {
        k: v
        for k, v in memory_info.items()
        if k.startswith("system") or k in ["ram_gb", "ram_percent"]
    }
    return system_info
