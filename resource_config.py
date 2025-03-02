# resource_config.py
"""
Configures and optimizes GPU/CPU resources with support for both NVIDIA and AMD GPUs.
"""

import os
import platform
from config import USE_GPU, GPU_MEMORY_FRACTION, OMP_NUM_THREADS, USE_XLA, USE_MIXED_PRECISION, USE_DIRECTML, TF_ENABLE_ONEDNN_OPTS

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

# Try to detect if we're on AMD hardware to use DirectML automatically
IS_AMD_GPU = False
try:
    if platform.system() == "Windows":
        import subprocess
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
        IS_AMD_GPU = any(gpu in output.lower() for gpu in ["amd", "radeon", "rx"])
except:
    pass

# Function to configure DirectML
def setup_directml():
    try:
        print("Attempting to use TensorFlow DirectML...")
        os.environ["TF_DIRECTML_KERNEL_CACHE"] = "1"  # Enable kernel caching for better performance
        import tensorflow as tf
        if hasattr(tf, 'experimental') and hasattr(tf.experimental, 'directml'):
            print("DirectML backend available. Configuring...")
            physical_devices = tf.experimental.directml.devices()
            if physical_devices:
                print(f"Found {len(physical_devices)} DirectML device(s)")
                # Use first device
                tf.experimental.directml.enable_device(0)
                print(f"Using DirectML device: {physical_devices[0]}")
                return tf
            else:
                print("No DirectML devices found")
                return None
        else:
            print("DirectML not available in this TensorFlow build")
            return None
    except ImportError:
        print("TensorFlow DirectML not installed. Install with: pip install tensorflow-directml")
        return None
    except Exception as e:
        print(f"DirectML setup failed: {str(e)}")
        return None

# First check for DirectML if AMD GPU is detected
if IS_AMD_GPU and USE_GPU and USE_DIRECTML:
    tf = setup_directml()
    if tf is None:
        IS_AMD_GPU = False  # Fall back to standard config if DirectML setup failed

# Standard TensorFlow configuration (for NVIDIA GPUs or CPU)
if not IS_AMD_GPU:
    tf = None
    cuda_failed = False
    
    try:
        if not USE_GPU:
            # Force CPU-only
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("Running on CPU only as per configuration.")
            
        import tensorflow as tf
        
        if tf is not None and USE_GPU:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth for each GPU
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU(s) detected. Memory growth enabled on {len(gpus)} GPU(s).")
                    
                    # Enable XLA optimization for faster execution
                    if USE_XLA:
                        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
                        print("XLA (Accelerated Linear Algebra) optimization enabled.")
                    
                    # Try to enable mixed precision for faster computation on GPUs that support it
                    if USE_MIXED_PRECISION:
                        try:
                            from tensorflow import keras
                            mixed_precision = keras.mixed_precision
                            policy = 'mixed_float16' if len(tf.config.list_physical_devices('GPU')) > 0 else 'float32'
                            mixed_precision.set_global_policy(policy)
                            print(f"Mixed precision policy set to: {policy}")
                        except Exception as e:
                            print(f"Could not enable mixed precision: {e}")
                    
                    # Set memory limits if configured
                    if GPU_MEMORY_FRACTION < 1.0:
                        memory_limit = int(GPU_MEMORY_FRACTION * 10000)
                        for gpu in gpus:
                            tf.config.experimental.set_virtual_device_configuration(
                                gpu,
                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                            )
                        print(f"GPU memory limited to {memory_limit}MB per device")
                        
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
                    cuda_failed = True  # Mark that CUDA initialization failed
            else:
                print("No NVIDIA GPUs detected with CUDA; will try DirectML as fallback.")
                cuda_failed = True  # No GPUs found, try DirectML
    except ImportError:
        print("TensorFlow not installed; cannot configure GPU settings.")
        cuda_failed = True  # Mark that CUDA initialization failed
    except Exception as e:
        print(f"Error initializing TensorFlow with CUDA: {str(e)}")
        cuda_failed = True  # Mark that CUDA initialization failed
    
    # Try DirectML as fallback if CUDA failed and DirectML is enabled
    if cuda_failed and USE_GPU and USE_DIRECTML:
        print("CUDA initialization failed or no CUDA GPUs detected. Trying DirectML as fallback...")
        tf_directml = setup_directml()
        if tf_directml is not None:
            print("Successfully initialized DirectML as fallback")
            tf = tf_directml  # Replace the tf object with DirectML version
            IS_AMD_GPU = True  # Set this flag to true since we're using DirectML now

def optimize_gpu_memory(limit_mb: int = 4096) -> None:
    """
    Optionally limit or adjust GPU memory usage.
    """
    if IS_AMD_GPU:
        print("Memory limiting not directly supported with DirectML backend")
        return
    
    if 'tf' in globals():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)]
                    )
                    print(f"Set GPU memory limit to {limit_mb} MB.")
                except RuntimeError as e:
                    print(f"GPU memory config error: {e}")

def test_gpu_availability():
    """
    Run a simple test to verify which device TensorFlow is using and print device placement info.
    Call this function when you need to debug GPU issues.
    """
    if 'tf' not in globals():
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow not installed; cannot test GPU.")
            return
            
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
    
    # Print additional device info
    print("\nAvailable devices:")
    devices = tf.config.list_physical_devices()
    for i, device in enumerate(devices):
        print(f"  {i}: {device.name} ({device.device_type})")
    
    # For NVIDIA GPUs, try to print CUDA info
    if not IS_AMD_GPU and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        try:
            gpu_details = tf.config.experimental.get_device_details(devices[0]) if devices else {}
            print("\nGPU Details:", gpu_details)
        except:
            pass
            
    print("===== TEST COMPLETE =====\n")
    
    return c  # Return the result of the simple operation

def list_available_gpus():
    """
    Simple utility to list all available GPUs detected by TensorFlow.
    """
    if 'tf' not in globals():
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow not installed; cannot detect GPUs.")
            return []
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print("\n=== GPU DETECTION ===")
        print(f"TensorFlow version: {tf.__version__}")
        
        if IS_AMD_GPU:
            print("Using DirectML backend")
        else:
            print("Using CUDA backend")
            
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

def run_gpu_benchmark(size=1000, dtype=None):
    """
    Quick benchmark to verify GPU acceleration and measure performance.
    
    Args:
        size: Size of matrices to multiply (larger = more intensive)
        dtype: Data type to use (tf.float32 or tf.float16)
    
    Returns:
        Execution time in milliseconds
    """
    if 'tf' not in globals():
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow not installed; cannot run benchmark.")
            return None
            
    if dtype is None:
        dtype = tf.float32
    
    try:
        import time
        
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
        
        # Benchmark interpretation guide
        if execution_ms < 50:
            print("✓ Excellent performance - GPU acceleration working well")
        elif execution_ms < 200:
            print("✓ Good performance - GPU acceleration appears to be working")
        elif execution_ms < 1000:
            print("⚠ Moderate performance - GPU may be limited or operations running on CPU")
        else:
            print("⚠ Slow performance - GPU acceleration may not be working correctly")
            
        print("=======================================\n")
        return execution_ms
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None

def configure_tensorflow_threading():
    """
    Report the current thread configuration - only use this for display purposes.
    All actual thread configuration should be done via environment variables
    before TensorFlow is imported.
    """
    if 'tf' not in globals():
        return
    
    try:
        threads_info = f"TensorFlow is using:"
        try:
            threads_info += f" {tf.config.threading.get_inter_op_parallelism_threads()} inter-op threads,"
        except:
            threads_info += " unknown inter-op threads,"
            
        try:
            threads_info += f" {tf.config.threading.get_intra_op_parallelism_threads()} intra-op threads"
        except:
            threads_info += " unknown intra-op threads"
            
        print(threads_info)
    except Exception as e:
        print(f"Could not determine thread configuration: {e}")

def manage_gpu_growth():
    """Configure GPU memory growth to prevent OOM issues"""
    if 'tf' not in globals():
        return
        
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
        
    # Allow memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"Could not set memory growth for GPU: {e}")
    
# Call functions when the module is imported
list_available_gpus()
