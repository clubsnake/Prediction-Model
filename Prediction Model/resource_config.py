# resource_config.py
"""
Configures and optimizes GPU/CPU resources based on config flags.
"""

import os
from config import USE_GPU, GPU_MEMORY_FRACTION

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed; cannot configure GPU settings.")
    tf = None

if not USE_GPU:
    # Force CPU-only
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Running on CPU only as per configuration.")
else:
    if tf is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU(s) detected. Memory growth enabled on {len(gpus)} GPU(s).")
                # For limiting GPU memory usage, uncomment & adjust:
                # memory_limit = int(GPU_MEMORY_FRACTION * 10000)
                # for gpu in gpus:
                #     tf.config.experimental.set_virtual_device_configuration(
                #         gpu,
                #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                #     )
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPUs detected; running on CPU.")
    else:
        print("Cannot configure GPU since TensorFlow is not installed.")

def optimize_gpu_memory(limit_mb: int = 4096) -> None:
    """
    Optionally limit or adjust GPU memory usage.
    """
    if tf is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)]
                    )
                except RuntimeError as e:
                    print(f"GPU memory config error: {e}")
