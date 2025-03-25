"""
TensorFlow environment setup and optimization utilities.

This module handles the configuration of TensorFlow runtime environment variables
and settings to optimize hardware resource utilization. It works in conjunction
with the training_optimizer to ensure consistent memory management.

This refactored version:
1. Delegates GPU memory management to training_optimizer
2. Focuses on environment variable setup while avoiding conflicts
3. Allows training_optimizer settings to take precedence
4. Maintains platform-specific optimizations

The module is designed to be imported early in the application startup process
to ensure TensorFlow is configured correctly before any models are loaded.
"""

import logging
import multiprocessing
import os
import platform
import sys
from typing import Dict

# Import training_optimizer for unified memory management
from src.utils.training_optimizer import get_training_optimizer

# Import gpu_memory_management for configuration
from src.utils.gpu_memory_management import configure_gpu_memory

try:
    # Import config settings
    from config.config_loader import get_config, get_value
except ImportError:
    # Fall back to stub if needed
    logging.warning("Could not import config.config_loader, using stub")

    def get_value(path, default=None):
        return default

    def get_config():
        return {}


logger = logging.getLogger(__name__)

def setup_tf_environment(
    cpu_threads=None, memory_growth=True, mixed_precision=None, use_training_optimizer=True
) -> Dict[str, str]:
    """
    Setup environment variables and TensorFlow configurations for optimal hardware utilization.
    
    This function cooperates with training_optimizer for memory management while focusing on
    environment variable configuration.

    Args:
        cpu_threads: Number of threads to use (None = auto-detect)
        memory_growth: Enable dynamic GPU memory growth
        mixed_precision: Enable mixed precision training (None = use config setting)
        use_training_optimizer: Whether to delegate GPU configuration to training_optimizer

    Returns:
        Dictionary of set environment variables
    """
    env_vars = {}

    # Auto-detect optimal CPU thread count if not specified
    if cpu_threads is None:
        cpu_count = multiprocessing.cpu_count()
        # Use 80% of available cores, but at least 2
        cpu_threads = max(2, int(cpu_count * 0.8))

    # Set OpenMP and TensorFlow thread settings
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(cpu_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(max(2, int(cpu_threads / 2)))

    # Record settings
    env_vars["OMP_NUM_THREADS"] = str(cpu_threads)
    env_vars["TF_NUM_INTRAOP_THREADS"] = str(cpu_threads)
    env_vars["TF_NUM_INTEROP_THREADS"] = str(max(2, int(cpu_threads / 2)))

    # Prevent TensorFlow from allocating all GPU memory at once
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env_vars["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Log memory usage
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # INFO + WARNING
    env_vars["TF_CPP_MIN_LOG_LEVEL"] = "1"

    # Ensure we're using the GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_vars["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Avoid TensorFlow aggressive CPU optimization on Windows
    if platform.system() == "Windows":
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env_vars["TF_ENABLE_ONEDNN_OPTS"] = "0"

    logger.info(f"Environment configured with {cpu_threads} threads")

    # Use training_optimizer for GPU configuration if requested
    if use_training_optimizer:
        optimizer = get_training_optimizer()
        
        # Let training_optimizer handle GPU memory growth
        if memory_growth:
            # Just log this - don't configure directly
            logger.info("Delegating GPU memory growth to training_optimizer")
        
        # Let training_optimizer handle mixed precision
        if mixed_precision is not None:
            # Just log this - don't configure directly
            logger.info(f"Delegating mixed precision setting ({mixed_precision}) to training_optimizer")
    else:
        # Use direct configuration only if explicitly requested
        # This path should be avoided in most cases
        try:
            # Import TensorFlow here instead of at the module level
            import tensorflow as tf

            # Dynamic memory growth for GPUs
            if memory_growth:
                gpus = tf.config.experimental.list_physical_devices("GPU")
                if gpus:
                    logger.info(f"Found {len(gpus)} GPU(s)")
                    for gpu in gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                            logger.info(f"Enabled dynamic memory growth for {gpu}")
                        except RuntimeError as e:
                            logger.warning(f"Error setting memory growth: {e}")
                else:
                    logger.info("No GPUs detected - using CPU only")

            # If mixed_precision is None, try to get from config
            if mixed_precision is None:
                try:
                    mixed_precision = get_value("hardware.use_mixed_precision", False)
                    logger.info(f"Using mixed_precision={mixed_precision} from user_config.yaml")
                except ImportError:
                    mixed_precision = False
                    logger.warning(
                        "Could not import config.config_loader, defaulting mixed_precision to False"
                    )

            # Mixed precision for faster training - only if explicitly enabled
            if mixed_precision:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled (float16)")
            else:
                policy = tf.keras.mixed_precision.Policy("float32")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision disabled (using float32)")

            # Check what policy was actually applied
            active_policy = tf.keras.mixed_precision.global_policy()
            logger.info(f"Active precision policy: {active_policy.name}")

            # Log TensorFlow configuration
            logger.info(f"TensorFlow version: {tf.__version__}")
            logger.info(f"Eager execution: {tf.executing_eagerly()}")

        except ImportError:
            logger.warning(
                "TensorFlow not imported yet - will apply settings when imported"
            )
        except Exception as e:
            logger.error(f"Error configuring TensorFlow: {e}")
    
    # Apply general performance settings that don't conflict with training_optimizer
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    env_vars["TF_GPU_THREAD_MODE"] = "gpu_private"
    
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
    env_vars["TF_CUDNN_USE_AUTOTUNE"] = "1"
    
    # Don't set GPU allocator directly - let training_optimizer handle it
    # This avoids conflicts with training_optimizer's memory management
    
    return env_vars

def init_environment():
    """Initialize environment with training_optimizer taking precedence."""
    # First initialize training_optimizer
    optimizer = get_training_optimizer()
    
    # Then set up environment variables
    env_vars = setup_tf_environment(use_training_optimizer=True)
    
    return {
        "environment_variables": env_vars,
        "training_optimizer_initialized": True
    }

if __name__ == "__main__":
    # Run initialization when module is executed directly
    init_result = init_environment()
    print(f"Environment initialized: {len(init_result['environment_variables'])} variables set")