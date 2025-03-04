# resource_config.py
"""
Configures and optimizes GPU/CPU resources with support for both NVIDIA and AMD GPUs.
"""

import os
import platform
import tensorflow as tf
import logging
from config import USE_GPU, GPU_MEMORY_FRACTION, OMP_NUM_THREADS, USE_XLA, USE_MIXED_PRECISION, USE_DIRECTML, TF_ENABLE_ONEDNN_OPTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure threading environment variables BEFORE importing tensorflow
if OMP_NUM_THREADS:
    import multiprocessing
    num_threads = OMP_NUM_THREADS if OMP_NUM_THREADS else multiprocessing.cpu_count()
    # Set environment variables for threading (this works before TF initialization)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(num_threads)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_threads)
    print(f"Set TensorFlow threading environment to {num_threads} threads")

# Enable additional TensorFlow optimizations via environment variables
if TF_ENABLE_ONEDNN_OPTS:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable Intel MKL-DNN optimizations if available

# Import centralized GPU memory management
from Scripts.gpu_memory_management import configure_gpu_memory, clean_gpu_memory, get_memory_info

# Configure GPU with settings from config
memory_config = {
    'use_gpu': USE_GPU,
    'allow_growth': True,
    'memory_limit_mb': int(GPU_MEMORY_FRACTION * 10000) if GPU_MEMORY_FRACTION < 1.0 else None,
    'use_xla': USE_XLA,
    'mixed_precision': USE_MIXED_PRECISION,
    'directml_enabled': USE_DIRECTML
}

# Initialize GPU - This will be ignored if already configured by gpu_memory_management
gpu_config_result = configure_gpu_memory(memory_config)

def list_available_gpus():
    """
    Simple utility to list all available GPUs detected by TensorFlow.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        print("\n=== GPU DETECTION ===")
        print(f"TensorFlow version: {tf.__version__}")
        
        backend = gpu_config_result.get('backend', 'unknown')
        print(f"Using {backend} backend")
            
        print(f"GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            
        # Additional device info that's helpful for debugging
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Logical GPUs: {len(logical_gpus)}")
        
        if not gpus:
            print("WARNING: No GPUs detected - using CPU for computation")
            
        print("====================\n")
        return gpus
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return []

def test_gpu_availability():
    """
    Run a simple test to verify which device TensorFlow is using and print device placement info.
    """
    try:
        import tensorflow as tf
        
        # Enable device placement logging
        tf.debugging.set_log_device_placement(True)
        
        print("\n===== GPU TEST =====")
        print("Testing TensorFlow device placement...")
        
        # Create some simple operations to test device placement
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        
        # Reset device placement logging (so it doesn't affect other operations)
        tf.debugging.set_log_device_placement(False)
        
        print("\nGPU test completed successfully")
        print("===== TEST COMPLETE =====\n")
        
        return c  # Return the result of the simple operation
    except Exception as e:
        print(f"Error testing GPU: {e}")
        return None

def run_gpu_benchmark(size=1000, dtype=None):
    """
    Quick benchmark to verify GPU acceleration and measure performance.
    
    Args:
        size: Size of matrices to multiply (larger = more intensive)
        dtype: Data type to use (tf.float32 or tf.float16)
    
    Returns:
        Execution time in milliseconds
    """
    try:
        import tensorflow as tf
        import time
        
        if dtype is None:
            dtype = tf.float32
        
        print(f"\n=== RUNNING GPU BENCHMARK (size={size}) ===")
        # Create two random matrices
        a = tf.random.normal((size, size), dtype=dtype)
        b = tf.random.normal((size, size), dtype=dtype)
        
        # First run - warm up (often slower due to initialization)
        _ = tf.matmul(a, b)
        
        # Timed run
        start = time.time()
        c = tf.matmul(a, b)
        # Force evaluation
        _ = c.numpy()
        end = time.time()
        
        execution_ms = (end - start) * 1000
        print(f"Matrix multiplication ({size}x{size}): {execution_ms:.2f} ms")
        
        # Clean up memory after benchmark
        clean_gpu_memory(force_gc=True)
        
        return execution_ms
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None

# List GPUs at import time for quick feedback
list_available_gpus()
