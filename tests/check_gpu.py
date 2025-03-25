# Save as check_gpu.py in your project root
import tensorflow as tf
import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Check for DirectML environment
is_directml = ("TENSORFLOW_USE_DIRECTML" in os.environ or 
              "DML_VISIBLE_DEVICES" in os.environ)

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("DirectML mode:", "Enabled" if is_directml else "Disabled")

# Try to allocate memory on GPU
try:
    with tf.device('/GPU:0'):
        # Start with simple operations that should work on all GPUs
        print("Running basic matrix operations...")
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("GPU Matrix multiplication successful")
        
        # Now warm up the GPU with more intensive operations
        try:
            print("Attempting to import GPU warmup module...")
            
            # If using DirectML, set environment variable to signal this to the warmup
            if is_directml and "TENSORFLOW_USE_DIRECTML" not in os.environ:
                os.environ["TENSORFLOW_USE_DIRECTML"] = "1"
                print("Set TENSORFLOW_USE_DIRECTML=1 for warmup")
                
            from src.utils.memory_utils import run_gpu_warmup
            print("Successfully imported GPU warmup module")
            print("Running GPU warmup to get those fans spinning...")
            peak_util = run_gpu_warmup(intensity=0.5)
            print(f"GPU warmup complete! Reached {peak_util}% utilization")
            
            # Simple additional DirectML-safe test if peak utilization is very low
            if peak_util < 5 and is_directml:
                print("Running additional DirectML-safe operations...")
                for i in range(3):
                    with tf.device('/GPU:0'):
                        x = tf.random.normal([2000, 2000])
                        y = tf.random.normal([2000, 2000])
                        z = tf.matmul(x, y)
                        _ = z.numpy()  # Force execution
                    print(f"Additional operation {i+1}/3 completed")
                
        except ImportError as e:
            print(f"GPU warmup module not found: {e}")
            print(f"Python path: {sys.path}")
            print("Skipping warmup.")
        except Exception as e:
            print(f"Error during GPU warmup: {e}")
            
            # If using DirectML, try a very simple backup warmup
            if is_directml:
                print("Trying simple DirectML warmup...")
                try:
                    for i in range(3):
                        x = tf.random.normal([3000, 3000])
                        y = tf.random.normal([3000, 3000])
                        z = tf.matmul(x, y)
                        _ = z.numpy()
                        print(f"Simple warmup {i+1}/3 completed")
                except Exception as e2:
                    print(f"Simple DirectML warmup also failed: {e2}")
except Exception as e:
    print("Error using GPU:", e)