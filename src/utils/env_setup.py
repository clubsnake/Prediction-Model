"""
TensorFlow environment setup and optimization utilities.

This module handles the configuration of TensorFlow runtime environment variables
and settings to optimize hardware resource utilization. It provides:

1. Function to setup optimal TensorFlow environment based on available hardware
2. Memory management utilities to clean up TensorFlow resources
3. Proper handling of GPU memory allocation and growth settings

The module is designed to be imported early in the application startup process
to ensure TensorFlow is configured correctly before any models are loaded.
Circular imports are avoided by moving TensorFlow imports inside functions.
"""

import logging
import multiprocessing
import os
import platform
from typing import Dict
import threading

# Fix incorrect imports
try:
    # Try relative import first
    from ..utils.gpu_memory_management import configure_gpu_memory
except ImportError:
    # Fall back to absolute import if needed
    try:
        from src.utils.gpu_memory_management import configure_gpu_memory
    except ImportError:
        # Define a no-op function as fallback
        def configure_gpu_memory(*args, **kwargs):
            pass


try:
    # Fix import to use correct path
    from config.config_loader import get_config, get_value
except ImportError:
    # Fall back to stub if needed
    logging.warning("Could not import config.config_loader, using stub")

    def get_value(path, default=None):
        return default

    def get_config():
        return {}


logger = logging.getLogger(__name__)

# Move TensorFlow import inside functions to avoid initialization issues
# This helps prevent circular imports when other modules import this one


def setup_tf_environment(
    cpu_threads=None, memory_growth=True, mixed_precision=None
) -> Dict[str, str]:
    """
    Setup environment variables and TensorFlow configurations for optimal hardware utilization.

    This function configures various environment variables and TensorFlow settings to
    optimize performance based on available hardware. It handles:
    - CPU thread allocation
    - GPU memory growth settings
    - Mixed precision configuration
    - Platform-specific optimizations

    Args:
        cpu_threads: Number of threads to use (None = auto-detect)
        memory_growth: Enable dynamic GPU memory growth
        mixed_precision: Enable mixed precision training (None = use config setting)

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

    # Configure TensorFlow if imported
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
                from config.config_loader import get_value

                mixed_precision = get_value("hardware.use_mixed_precision", False)
                logger.info(
                    f"Using mixed_precision={mixed_precision} from user_config.yaml"
                )
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
    
    # Apply settings for maximum performance by default
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    env_vars["TF_GPU_THREAD_MODE"] = "gpu_private"
    
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
    env_vars["TF_CUDNN_USE_AUTOTUNE"] = "1"
    
    # Apply mixed precision automatically if supported
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1" if mixed_precision else "0"
    env_vars["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1" if mixed_precision else "0"
    
    # Set faster GPU allocator
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    env_vars["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    
    return env_vars

