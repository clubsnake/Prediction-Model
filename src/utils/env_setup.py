r"""
Sets up environment variables for optimal performance.
This module should be imported before any other ML libraries.
"""

import logging
import multiprocessing
import os
import platform
import sys
from typing import Dict

# Import from existing modules rather than creating new code
try:
    from gpu_memory_management import configure_gpu_memory
except ImportError:
    # Handle case when imported from different directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.gpu_memory_management import configure_gpu_memory

logger = logging.getLogger(__name__)


def setup_tf_environment(cpu_threads=None, memory_growth=True, mixed_precision=True) -> Dict[str, str]:
    """
    Setup environment variables and TensorFlow configurations for optimal hardware utilization.
    
    Args:
        cpu_threads: Number of threads to use (None = auto-detect)
        memory_growth: Enable dynamic GPU memory growth
        mixed_precision: Enable mixed precision training
        
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
    os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(cpu_threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(max(2, int(cpu_threads/2)))
    
    # Record settings
    env_vars['OMP_NUM_THREADS'] = str(cpu_threads)
    env_vars['TF_NUM_INTRAOP_THREADS'] = str(cpu_threads)
    env_vars['TF_NUM_INTEROP_THREADS'] = str(max(2, int(cpu_threads/2)))
    
    # Prevent TensorFlow from allocating all GPU memory at once
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    env_vars['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Log memory usage
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # INFO + WARNING
    env_vars['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
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
        import tensorflow as tf
        
        # Dynamic memory growth for GPUs
        if memory_growth:
            gpus = tf.config.experimental.list_physical_devices('GPU')
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
        
        # Mixed precision for faster training
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled (float16)")
        
        # Log TensorFlow configuration
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"Eager execution: {tf.executing_eagerly()}")
        
    except ImportError:
        logger.warning("TensorFlow not imported yet - will apply settings when imported")
    except Exception as e:
        logger.error(f"Error configuring TensorFlow: {e}")
    
    return env_vars


def cleanup_tf_session():
    """
    Clean up TensorFlow session resources.
    Call this function when you're done with TensorFlow operations
    to free up memory.
    """
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        # For older TF versions
        if hasattr(tf, "reset_default_graph"):
            tf.reset_default_graph()
        import gc

        gc.collect()
        return True
    except Exception as e:
        logger.error(f"Error cleaning up TensorFlow session: {e}")
        return False
