"""
Sets up environment variables for optimal performance.
This module should be imported before any other ML libraries.
"""

import os
import platform
import logging
import sys
import multiprocessing
from typing import Dict

# Import from existing modules rather than creating new code
try:
    from Scripts.gpu_memory_management import configure_gpu_memory, clean_gpu_memory
except ImportError:
    # Handle case when imported from different directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Scripts.gpu_memory_management import configure_gpu_memory, clean_gpu_memory

logger = logging.getLogger(__name__)

def setup_tf_environment() -> Dict[str, str]:
    """
    Configure environment variables for optimal TensorFlow and ML library performance.
    Should be called before importing any ML libraries.
    
    Returns:
        Dictionary of set environment variables
    """
    env_vars = {}
    
    try:
        # Detect number of CPU cores for optimal threading
        cpu_count = multiprocessing.cpu_count()
        
        # Threading optimizations
        # Use half of available logical cores for intra-op parallelism
        intra_threads = max(1, cpu_count // 2)
        env_vars['TF_NUM_INTRAOP_THREADS'] = str(intra_threads)
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(intra_threads)
        
        # Use quarter of available logical cores for inter-op parallelism
        inter_threads = max(1, cpu_count // 4)
        env_vars['TF_NUM_INTEROP_THREADS'] = str(inter_threads)
        os.environ['TF_NUM_INTEROP_THREADS'] = str(inter_threads)
        
        # Set OpenMP thread settings 
        env_vars['OMP_NUM_THREADS'] = str(intra_threads)
        os.environ['OMP_NUM_THREADS'] = str(intra_threads)
        
        # Reduce TensorFlow logging verbosity (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
        env_vars['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
        # Windows-specific optimizations
        if platform.system() == 'Windows':
            # Needed for some versions of TensorFlow on Windows
            env_vars['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Configure GPU memory using the centralized system
        gpu_config = {
            'allow_growth': True,
            'memory_limit_mb': None,  # No specific limit
            'use_xla': True,          # Enable XLA JIT compilation
            'mixed_precision': True   # Enable mixed precision
        }
        
        # Configure GPU memory through existing management system
        config_result = configure_gpu_memory(gpu_config)
        
        # Log the configuration result
        logger.info(f"Environment setup complete: {env_vars}")
        logger.info(f"GPU configuration: {config_result}")
        
        return env_vars
    
    except Exception as e:
        logger.error(f"Error setting up environment variables: {str(e)}")
        return {}

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
        if hasattr(tf, 'reset_default_graph'):
            tf.reset_default_graph()
        import gc
        gc.collect()
        return True
    except Exception as e:
        logger.error(f"Error cleaning up TensorFlow session: {e}")
        return False