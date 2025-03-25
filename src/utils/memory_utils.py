"""
Memory management utilities for efficient model training.
This module provides a consolidated approach to memory management by
delegating to TrainingOptimizer methods where appropriate.
"""

import logging
import weakref
import os


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


def cleanup_tf_session(preserve_tf_session=True):
    """
    Clear TensorFlow session to free memory.
    Now delegates to TrainingOptimizer for a more comprehensive approach.
    
    Args:
        preserve_tf_session: If True, avoids resetting the TensorFlow session
                            completely, which could interfere with TensorFlow's
                            own session management. Default: True
    """
    optimizer = get_training_optimizer()
    level = "light" if preserve_tf_session else "medium"
    return optimizer.cleanup_memory(level=level, preserve_tf_session=preserve_tf_session)


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


def adaptive_memory_clean(level="medium", preserve_tf_session=True):
    """
    Clean memory based on the specified level of cleanup intensity.

    Args:
        level: Cleanup intensity ("light", "medium", "heavy", "critical")
        preserve_tf_session: If True, avoids operations that might interfere
                           with TensorFlow's session management. Default: True

    Returns:
        Dict: Results of the cleanup operation
    """
    optimizer = get_training_optimizer()

    # Map "critical" to "heavy" for compatibility
    if level == "critical":
        level = "heavy"

    return optimizer.cleanup_memory(level=level, preserve_tf_session=preserve_tf_session)


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


def is_memory_critical():
    """
    Check if system memory usage is at a critical level.
    
    Returns:
        bool: True if memory usage is critical (above 90%)
    """
    optimizer = get_training_optimizer()
    memory_info = optimizer._get_memory_info()
    return memory_info.get("ram_percent", 0) > 90


def run_gpu_warmup(duration=10, intensity=0.5):
    """Run a GPU warmup to improve performance during initial operations"""
    try:
        # Check for DirectML environment
        is_directml = ("TENSORFLOW_USE_DIRECTML" in os.environ or 
                      "DML_VISIBLE_DEVICES" in os.environ)
        
        # Import TensorFlow to check version and GPU type
        import tensorflow as tf
        tf_version = tf.__version__
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        logger.info(f"Running GPU warmup with TF {tf_version}, DirectML: {is_directml}")
        logger.info(f"GPU devices: {gpu_devices}")
        
        # If no GPU devices found, log warning and exit
        if not gpu_devices:
            logger.warning("No GPU devices found for warmup")
            return False
        
        from src.utils.gpu_memory_manager import GPUMemoryManager
        manager = GPUMemoryManager()
        
        # Force DirectML detection if we've detected it here
        if is_directml:
            manager.is_directml = True
            logger.info("Forcing DirectML mode for GPU warmup")
        
        # Call the warmup function with appropriate parameters
        peak_util = manager.warmup_gpu(intensity=intensity)
        
        # Notify training optimizer about the warmup
        try:
            optimizer = get_training_optimizer()
            if hasattr(optimizer, "log_gpu_activity"):
                optimizer.log_gpu_activity("warmup", peak_util)
        except Exception as e:
            logger.debug(f"Could not log GPU activity: {e}")
            # No need to re-raise this exception as it's non-critical
            
        return peak_util
    except Exception as e:
        logger.error(f"Error running GPU warmup: {e}", exc_info=True)
        # Try a simple DirectML fallback if the manager approach fails
        try:
            import tensorflow as tf
            import time
            
            logger.info("Using simple DirectML fallback for GPU warmup")
            with tf.device('/GPU:0'):
                # Simple matrix operations that should work with DirectML
                for i in range(3):
                    a = tf.random.normal([5000, 5000])
                    b = tf.random.normal([5000, 5000])
                    c = tf.matmul(a, b)
                    _ = c.numpy()  # Force execution
                    logger.info(f"DirectML fallback: completed iteration {i+1}/3")
                    time.sleep(1)
                
            logger.info("Simple DirectML warmup completed")
            return 10  # Return a nominal utilization value
        except Exception as e2:
            logger.error(f"DirectML fallback also failed: {e2}", exc_info=True)
            # Return False instead of silently failing
            return False
