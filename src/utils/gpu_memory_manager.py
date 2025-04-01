import logging
import os
import time
import sys
import numpy as np
import tensorflow as tf
import threading
from contextlib import contextmanager
import gc

logger = logging.getLogger("GPU_Memory_Manager")

# Import centralized GPU memory management
from src.utils.gpu_memory_management import (
    clean_gpu_memory,
    configure_gpu_memory,
    get_memory_info,
    get_gpu_utilization,   
    stress_test_gpu,       
)

# Import adaptive_memory_clean from memory_utils instead of duplicating it
from src.utils.memory_utils import adaptive_memory_clean

# Add thread lock to prevent concurrent GPU memory modifications
_manager_lock = threading.RLock()

# Thread-local storage for device contexts
_thread_local_context = threading.local()

class DeviceContextScope:
    """
    Context manager for TensorFlow device operations.
    Ensures operations run on the specified device and cleans up properly.
    
    Example:
        with DeviceContextScope('/GPU:0', memory_manager):
            # All TensorFlow operations here will run on GPU:0
            model = tf.keras.models.Sequential(...)
            model.fit(...)
    """
    
    def __init__(self, device_path, memory_manager=None):
        """
        Initialize with device path and optional memory manager
        
        Args:
            device_path: TensorFlow device path (e.g., '/GPU:0', '/CPU:0')
            memory_manager: Optional GPUMemoryManager instance for optimization
        """
        self.device_path = device_path
        self.memory_manager = memory_manager
        self.device_context = None
        
    def __enter__(self):
        """Enter the device context"""
        # Import tensorflow here to avoid circular imports
        import tensorflow as tf
        
        # Create device context
        self.device_context = tf.device(self.device_path)
        
        # Store old context if needed for nesting
        if not hasattr(_thread_local_context, 'stack'):
            _thread_local_context.stack = []
            
        # Save previous context
        _thread_local_context.stack.append(self.device_path)
        
        # Store in thread-local for proper nesting
        _thread_local_context.current_device = self.device_path
        
        # Enter TF device context
        self.device_context.__enter__()
        
        # Optimize settings if memory manager provided
        if self.memory_manager:
            self.memory_manager.optimize_for_context(self.device_path)
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the device context and clean up"""
        # Exit TF device context
        self.device_context.__exit__(exc_type, exc_val, exc_tb)
        
        # Restore previous context if there was one
        if hasattr(_thread_local_context, 'stack') and _thread_local_context.stack:
            _thread_local_context.stack.pop()
            if _thread_local_context.stack:
                _thread_local_context.current_device = _thread_local_context.stack[-1]
            else:
                if hasattr(_thread_local_context, 'current_device'):
                    delattr(_thread_local_context, 'current_device')
        
        # Clean up memory if requested and no exception occurred
        if self.memory_manager and exc_type is None:
            self.memory_manager.post_operation_cleanup(aggressive=False)
        
        # Return False to propagate any exceptions
        return False

class GPUMemoryManager:

    def __init__(
        self,
        allow_growth=True,
        visible_devices=None,
    ):
        """
        Initialize the GPU memory manager.

        Args:
            allow_growth: Whether to allow GPU memory growth
            visible_devices: List of GPU indices to use or None for all
        """
        self.allow_growth = allow_growth
        self.visible_devices = visible_devices
        self.logger = logging.getLogger("GPUMemoryManager")
        self.original_gpus = []
        self.logical_gpus = []
        self.initialized = False
        self.memory_usage_log = []
        self.batch_size_log = []
        self.performance_mode = False  # Default to standard mode
        self.gpu_utilization_history = []  # Track GPU utilization
        self.has_gpu = False  # Default to False until initialized

        # Enhanced DirectML detection
        self.is_directml = (
            "TENSORFLOW_USE_DIRECTML" in os.environ
            or "DML_VISIBLE_DEVICES" in os.environ
        )
        
        # Additional check for DirectML
        try:
            import tensorflow as tf
            if hasattr(tf, 'experimental') and hasattr(tf.experimental, 'get_device_policy'):
                self.is_directml = self.is_directml or 'DirectML' in str(tf.config.list_physical_devices())
            
            # Check if DirectML dll is loaded
            if hasattr(tf, 'version'):
                self.is_directml = self.is_directml or 'directml.dll' in str(tf.sysconfig.get_build_info()).lower()
            
            # NEW: Check if any GPU device name contains "DML"
            gpu_devices = tf.config.list_physical_devices('GPU')
            if any("DML" in str(device) for device in gpu_devices):
                self.is_directml = True
                self.logger.info("Detected DirectML GPU device based on device name")
        except Exception:
            pass
            
        if self.is_directml:
            self.logger.info("DirectML environment detected in GPUMemoryManager")
            
        # Add thread-local context for GPU operations
        self.thread_local = threading.local()

    def _configure_directml_compatibility(self):
        """
        Configure TensorFlow for DirectML compatibility
        
        This method sets necessary options to make LSTM and other layers work with DirectML
        """
        try:
            import tensorflow as tf
            
            # Force LSTM implementation to standard (avoid CuDNN implementation)
            if hasattr(tf.keras.layers, 'LSTM'):
                tf.keras.layers.LSTM._use_implementation = 1
                self.logger.info("Configured LSTM for DirectML compatibility")
                
            # Set GPU memory growth to prevent OOM errors
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    self.logger.warning(f"Could not set memory growth for DirectML: {e}")
                    
            # Disable XLA compilation which may not be compatible
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
            self.logger.info("Disabled XLA compilation for DirectML compatibility")
            
        except Exception as e:
            self.logger.error(f"Error configuring DirectML compatibility: {e}")

    def initialize(self):
        """
        Initialize the GPU memory manager and configure the GPUs using the centralized system.

        Returns:
            List of logical devices available after configuration
        """
        # Use lock to prevent concurrent initialization
        with _manager_lock:
            config = {
                "allow_growth": self.allow_growth,
                "visible_gpus": self.visible_devices,
                "directml_enabled": self.is_directml,  # Pass DirectML flag to configuration
            }
            configure_gpu_memory(config)
            self.initialized = True
    
            # Get list of GPUs after configuration
            try:
                import tensorflow as tf
    
                self.original_gpus = tf.config.list_physical_devices("GPU")
                self.logical_gpus = tf.config.list_logical_devices("GPU")
                self.has_gpu = len(self.logical_gpus) > 0  # Set has_gpu based on detected GPUs
                self.logger.info(f"Available logical GPUs: {len(self.logical_gpus)}")
    
                # Handle LSTM implementation for DirectML to avoid CudnnRNN errors
                if self.is_directml:
                    self._configure_directml_compatibility()
                    
                # Store device contexts in thread-local storage
                if not hasattr(self.thread_local, 'device_contexts'):
                    self.thread_local.device_contexts = {}
                    for i, gpu in enumerate(self.logical_gpus):
                        # Create device context for each GPU
                        self.thread_local.device_contexts[f'/GPU:{i}'] = tf.device(f'/GPU:{i}')
                    # Create CPU device context
                    self.thread_local.device_contexts['/CPU:0'] = tf.device('/CPU:0')
                    
            except Exception as e:
                self.logger.error(f"Error getting GPU list: {e}")
    
            return self.logical_gpus

    def run_gpu_stress_test(self, duration=10, intensity=0.9):
        """Run a GPU stress test to check performance and get those fans spinning!"""
        # Initialize if needed
        if not self.initialized:
            self.initialize()
            
        self.logger.info("Running GPU stress test to get fans spinning...")
        results = stress_test_gpu(duration_seconds=duration, intensity=intensity)
        
        if "error" in results:
            self.logger.error(f"GPU stress test failed: {results['error']}")
            return False
            
        self.logger.info("GPU stress test results:")
        self.logger.info(f"- Max utilization: {results['max_utilization']}%")
        self.logger.info(f"- Average utilization: {results['avg_utilization']:.1f}%")
        
        if "initial_temp" in results and "final_temp" in results:
            self.logger.info(f"- Temperature increased: {results['initial_temp']}°C → {results['final_temp']}°C")
            
        self.logger.info(f"- Fans spinning: {'YES!' if results['fans_spinning'] else 'No'}")
        
        return results

    # New method for rapid warmup to get GPU going
    def warmup_gpu(self, intensity=0.7, unique_prefix=None):
        """
        Quickly warm up the GPU to get it ready for compute-intensive operations.
        This helps "get those fans spinning" by raising GPU temperature to operating levels.
        
        Args:
            intensity: How hard to push the GPU during warmup (0.1 to 1.0)
            unique_prefix: Optional prefix to make layer names unique
        
        Returns:
            Peak GPU utilization achieved
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize()
            
        try:
            import tensorflow as tf
            import time
            
            # Use lock to ensure exclusive GPU access during warmup
            with _manager_lock:
                self.logger.info(f"Warming up GPU with {intensity:.0%} intensity...")
                
                # Set default unique prefix if not provided
                if unique_prefix is None:
                    unique_prefix = f"warmup_{int(time.time())}__"
                
                # Check if we're using DirectML
                if self.is_directml:
                    self.logger.info("Using DirectML-compatible GPU warmup method")
                    return self._directml_warmup(intensity, unique_prefix)
                    
                # Create a compute-intensive model using a large convolution
                input_shape = (256, 256, 3)
                input_tensor = tf.keras.layers.Input(shape=input_shape)
                x = input_tensor
                
                # Add compute-intensive layers with unique names
                for i in range(5):  # Multiple layers increase compute intensity
                    x = tf.keras.layers.Conv2D(
                        filters=64 * (i+1), 
                        kernel_size=(3, 3),
                        activation='relu',
                        padding='same',
                        name=f"{unique_prefix}conv2d_{i}"  # Add unique prefix and index
                    )(x)
                    if i % 2 == 0:
                        x = tf.keras.layers.MaxPooling2D(
                            name=f"{unique_prefix}maxpool_{i}"
                        )(x)
                
                # Add a final dense layer
                x = tf.keras.layers.GlobalAveragePooling2D(
                    name=f"{unique_prefix}global_avg_pool"
                )(x)
                x = tf.keras.layers.Dense(
                    1000, 
                    activation='relu',
                    name=f"{unique_prefix}dense_1000"
                )(x)
                output = tf.keras.layers.Dense(
                    10, 
                    activation='softmax',
                    name=f"{unique_prefix}dense_output"
                )(x)
                
                model = tf.keras.Model(inputs=input_tensor, outputs=output)
                
                # IMPORTANT: DO NOT use jit_compile=True with DirectML
                # Even though we're in the non-DirectML path, remove it to be safe
                # since DirectML detection might be incorrect in some environments
                @tf.function
                def training_step(x):
                    with tf.GradientTape() as tape:
                        pred = model(x, training=True)
                        loss = tf.reduce_mean(pred)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    # Apply or store gradients to prevent unused variable warning
                    if any(g is not None for g in gradients):
                        logger.debug(f"Computed gradients for {len(gradients)} variables")
                    return gradients
                
                # Create random input data
                batch_size = max(1, int(32 * intensity))
                input_data = tf.random.normal([batch_size, *input_shape])
                
                # Track utilization
                peak_util = 0
                start_util = get_gpu_utilization(0)
                
                self.logger.info(f"Starting GPU warmup (initial utilization: {start_util}%)...")
                
                # Run the warmup
                start_time = time.time()
                iterations = int(10 * intensity)  # Scale iterations by intensity
                
                for i in range(iterations):
                    _ = training_step(input_data)
                    
                    # Check utilization every other iteration
                    if i % 2 == 0:
                        current_util = get_gpu_utilization(0)
                        peak_util = max(peak_util, current_util)
                        self.logger.info(f"Iteration {i+1}/{iterations}: GPU at {current_util}%")
                        
                        # Store in history
                        self.gpu_utilization_history.append({
                            "timestamp": time.time(),
                            "utilization": current_util,
                            "operation": "warmup",
                            "iteration": i
                        })
                
                duration = time.time() - start_time
                
                self.logger.info(f"GPU warmup complete! Reached {peak_util}% utilization")
                self.logger.info(f"Took {duration:.1f} seconds, fans should be spinning now!")
                
                # Clean up
                self.clean_memory()
                
                return peak_util
            
        except Exception as e:
            self.logger.error(f"Error warming up GPU: {e}")
            return 0

    def _directml_warmup(self, intensity=0.7, unique_prefix=None):
        """
        DirectML-compatible GPU warmup that uses simpler operations.
        
        Args:
            intensity: Warmup intensity (0.1 to 1.0)
            unique_prefix: Optional prefix to make layer names unique
            
        Returns:
            Peak GPU utilization achieved
        """
        try:
            import tensorflow as tf
            import time
            import numpy as np
            
            self.logger.info("Running DirectML-compatible GPU warmup...")
            
            # Get initial utilization
            try:
                start_util = get_gpu_utilization(0)
                self.logger.info(f"Initial GPU utilization: {start_util}%")
                peak_util = start_util
            except Exception as e:
                self.logger.warning(f"Could not get initial GPU utilization: {e}")
                start_util = 0
                peak_util = 0
            
            # Scale matrix size based on intensity (smaller for DirectML)
            matrix_size = min(4096, int(1000 * intensity))  # Reduced size for DirectML
            iterations = max(10, int(5 * intensity))
            
            # Simple matrix operations that should work on DirectML
            for i in range(iterations):
                try:
                    self.logger.info(f"Warmup iteration {i+1}/{iterations}")
                    
                    # Create tensors directly on the GPU
                    with tf.device('/GPU:0'):
                        # Simple matrix multiplication
                        a = tf.random.normal([matrix_size, matrix_size])
                        b = tf.random.normal([matrix_size, matrix_size])
                        
                        # Execute operations that should be compatible with DirectML
                        c = tf.matmul(a, b)
                        
                        # Force execution
                        result = c.numpy()
                        
                        # Add some basic math operations that should work
                        d = tf.reduce_mean(c)
                        e = tf.square(d)
                        
                        # Force execution again
                        _ = e.numpy()
                    
                    # Check utilization if available
                    try:
                        current_util = get_gpu_utilization(0)
                        peak_util = max(peak_util, current_util)
                        self.logger.info(f"Current GPU utilization: {current_util}%")
                        
                        # Store in history
                        self.gpu_utilization_history.append({
                            "timestamp": time.time(),
                            "utilization": current_util,
                            "operation": "directml_warmup",
                            "iteration": i
                        })
                    except Exception:
                        pass
                    
                    # Short delay
                    time.sleep(0.5)
                
                except Exception as e:
                    self.logger.warning(f"Error in warmup iteration {i+1}: {e}")
                    # Continue with next iteration
            
            self.logger.info(f"DirectML warmup complete! Peak utilization: {peak_util}%")
            
            # Attempt to clean memory
            try:
                self.clean_memory()
            except Exception as e:
                self.logger.warning(f"Error cleaning memory: {e}")
                
            return peak_util
            
        except Exception as e:
            self.logger.error(f"Error in DirectML warmup: {e}")
            return 0

    def get_available_memory(self, device_idx=0) -> float:
        """
        Get the available memory on a GPU in MB.

        Args:
            device_idx: Index of the GPU to query

        Returns:
            Available memory in MB or -1 if error
        """
        if not self.initialized:
            self.initialize()

        if not self.original_gpus:
            return -1  # No GPUs available

        memory_info = get_memory_info(device_idx)
        if "current_mb" in memory_info:
            return memory_info["current_mb"]
        return -1

    def log_memory_usage(self, tag="", device_idx=0):
        """
        Log current memory usage for a GPU.

        Args:
            tag: A string tag to identify this log entry
            device_idx: Index of the GPU to query
        """
        memory_mb = self.get_available_memory(device_idx)
        timestamp = time.time()

        log_entry = {
            "timestamp": timestamp,
            "tag": tag,
            "device_idx": device_idx,
            "memory_mb": memory_mb,
        }

        self.memory_usage_log.append(log_entry)
        self.logger.info(f"Memory usage [{tag}]: GPU {device_idx} - {memory_mb:.2f} MB")

    def estimate_optimal_batch_size(
        self,
        model,
        sample_input_shape,
        min_batch=1,
        max_batch=8192,
        device_idx=0,
        safety_factor=0.8,
        respect_optuna=True,
    ):
        """
        Estimate the optimal batch size for a model based on available GPU memory.
        Uses a binary search approach to find the largest batch size that fits.

        Args:
            model: TensorFlow model
            sample_input_shape: Shape of a single input sample (without batch dimension)
            min_batch: Minimum batch size to consider
            max_batch: Maximum batch size to consider
            device_idx: Index of the GPU to use
            safety_factor: Factor to apply to the final batch size (0-1)
        Returns:
            Optimal batch size
        """
        if not self.initialized:
            self.initialize()

        if not self.original_gpus:
            # No GPUs, return a conservative batch size
            self.logger.warning("No GPUs available, returning conservative batch size")
            return min(64, max_batch)

        # Clean memory before estimation
        self.clean_memory()

        # Create a strategy for this device
        try:
            with tf.device(f"/device:GPU:{device_idx}"):
                # Binary search for optimal batch size
                low, high = min_batch, max_batch
                optimal_batch = min_batch

                while low <= high:
                    mid = (low + high) // 2
                    try:
                        # Create batch of current size
                        batch_shape = (mid,) + tuple(sample_input_shape)
                        test_input = tf.random.normal(batch_shape)

                        # Run a forward and backward pass
                        with tf.GradientTape():
                            _ = model(test_input, training=True)

                        # If successful, try a larger batch
                        optimal_batch = mid
                        low = mid + 1

                        # Clean up to avoid fragmentation
                        del test_input
                        self.clean_memory(force_gc=False)

                    except (
                        tf.errors.ResourceExhaustedError,
                        tf.errors.InternalError,
                        tf.errors.UnknownError,
                        tf.errors.OOMError,
                    ) as e:
                        # Out of memory, try smaller batch
                        self.logger.info(f"Batch size {mid} is too large: {e}")
                        high = mid - 1

                        # Need a more thorough cleanup after OOM
                        self.clean_memory(force_gc=True)

                # Apply safety factor to avoid edge cases
                safe_batch = max(min_batch, int(optimal_batch * safety_factor))

                # Log the result
                log_entry = {
                    "timestamp": time.time(),
                    "device_idx": device_idx,
                    "optimal_batch": optimal_batch,
                    "safe_batch": safe_batch,
                    "input_shape": sample_input_shape,
                }
                self.batch_size_log.append(log_entry)

                self.logger.info(
                    f"Optimal batch size: {optimal_batch}, "
                    f"Safe batch size: {safe_batch}"
                )
                return safe_batch

        except Exception as e:
            self.logger.error(f"Error estimating batch size: {e}")
            return min(32, max_batch)  # Fallback to a conservative value

    def get_gradient_accumulation_steps(self, desired_batch, actual_batch):
        """
        Calculate the number of gradient accumulation steps needed to
        simulate a larger batch size.

        Args:
            desired_batch: The desired effective batch size
            actual_batch: The actual batch size that fits in memory

        Returns:
            Number of gradient accumulation steps
        """
        if actual_batch <= 0:
            return 1

        steps = max(1, desired_batch // actual_batch)
        if desired_batch % actual_batch != 0:
            steps += 1

        return steps

    def clean_memory(self, force_gc=True):
        """
        Clean up GPU memory by clearing TensorFlow caches and running garbage collection.

        Args:
            force_gc: Whether to force Python garbage collection
        """
        # Use the manager lock to prevent concurrent cleanup
        with _manager_lock:
            return clean_gpu_memory(force_gc)
            
    def optimize_for_context(self, device_path):
        """
        Optimize GPU settings for the current operation context
        
        Args:
            device_path: Device path (e.g. '/GPU:0')
        """
        if not device_path.startswith('/GPU:'):
            # Nothing to optimize for CPU
            return
            
        try:
            # Extract GPU index from device path
            gpu_idx = int(device_path.split(':')[1])
            
            # Ensure memory growth is enabled
            gpus = tf.config.list_physical_devices('GPU')
            if gpu_idx < len(gpus):
                try:
                    tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
                except Exception as e:
                    self.logger.debug(f"Could not set memory growth for GPU:{gpu_idx}: {e}")
                    
            # Set thread-local device context
            if not hasattr(self.thread_local, 'current_device'):
                self.thread_local.current_device = device_path
        except Exception as e:
            self.logger.warning(f"Error optimizing for device context {device_path}: {e}")
            
    def post_operation_cleanup(self, aggressive=False):
        """
        Perform cleanup after an operation completes
        
        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        if aggressive:
            self.clean_memory(force_gc=True)
        else:
            # Lighter cleanup - just garbage collection
            gc.collect()

    def enable_mixed_precision(self):
        """
        Enable mixed precision only if configured in user settings and not using DirectML.

        Returns:
            True if mixed precision is enabled, False otherwise
        """
        try:
            # If using DirectML, always use float32 for compatibility
            if self.is_directml:
                self.logger.info(
                    "Using DirectML - forcing float32 precision for compatibility"
                )
                tf.keras.mixed_precision.set_global_policy("float32")
                return False

            # Check user configuration first
            try:
                # Avoid circular imports
                sys.path.append(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
                )
                from config.config_loader import get_value

                use_mixed_precision = get_value("hardware.use_mixed_precision", False)

                if not use_mixed_precision:
                    self.logger.info(
                        "Mixed precision disabled in configuration - keeping float32 precision"
                    )
                    tf.keras.mixed_precision.set_global_policy("float32")
                    return False
            except ImportError:
                # If we can't import, check TF_FORCE_FLOAT32 environment variable
                if os.environ.get("TF_FORCE_FLOAT32") == "1":
                    self.logger.info(
                        "TF_FORCE_FLOAT32 is set - keeping float32 precision"
                    )
                    tf.keras.mixed_precision.set_global_policy("float32")
                    return False

            # Only enable mixed precision if config allows it
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            self.logger.info("Mixed precision (float16) enabled")
            return True
        except Exception as e:
            self.logger.error(f"Could not set precision policy: {e}")
            return False

    def create_training_strategy(self, devices=None, use_mirrored=True):
        """
        Create a distribution strategy for training across multiple GPUs.

        Args:
            devices: List of device indices to use (None for all)
            use_mirrored: Whether to use MirroredStrategy or OneDeviceStrategy

        Returns:
            TensorFlow distribution strategy
        """
        if not self.initialized:
            self.initialize()

        if not self.logical_gpus:
            # No GPUs, use default strategy
            self.logger.warning("No GPUs available, using default strategy")
            return tf.distribute.get_strategy()

        try:
            if len(self.logical_gpus) > 1 and use_mirrored:
                # Use MirroredStrategy for multiple GPUs
                if devices is None:
                    # Use all available GPUs
                    strategy = tf.distribute.MirroredStrategy()
                    self.logger.info(
                        f"Using MirroredStrategy with {len(self.logical_gpus)} GPUs"
                    )
                else:
                    # Use specified GPUs
                    device_names = [f"/device:GPU:{d}" for d in devices]
                    strategy = tf.distribute.MirroredStrategy(devices=device_names)
                    self.logger.info(f"Using MirroredStrategy with GPUs: {devices}")
            else:
                # Use OneDeviceStrategy for single GPU
                device_idx = devices[0] if devices else 0
                device_name = f"/device:GPU:{device_idx}"
                strategy = tf.distribute.OneDeviceStrategy(device=device_name)
                self.logger.info(f"Using OneDeviceStrategy with GPU {device_idx}")

            return strategy

        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            return tf.distribute.get_strategy()  # Fallback to default

    def model_memory_profile(self, model, sample_batch_size=1, detailed=False):
        """
        Profile a model's memory usage.

        Args:
            model: TensorFlow model to profile
            sample_batch_size: Batch size to use for profiling
            detailed: Whether to print detailed layer-by-layer breakdown

        Returns:
            Dictionary with memory usage statistics
        """
        if not hasattr(model, "inputs") or not model.inputs:
            self.logger.error("Model doesn't have defined inputs")
            return {}

        try:
            # Get input shapes
            input_shapes = [
                (shape[0] if shape[0] is None else sample_batch_size, *shape[1:])
                for shape in [input.shape for input in model.inputs]
            ]

            # Create dummy inputs
            dummy_inputs = [tf.ones(shape) for shape in input_shapes]

            # Log initial memory
            self.log_memory_usage("before_profile")

            # Run a forward pass
            _ = model(dummy_inputs)

            # Log after loading
            self.log_memory_usage("after_forward_pass")

            # Calculate parameter count
            trainable_params = np.sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            )
            non_trainable_params = np.sum(
                [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
            )
            total_params = trainable_params + non_trainable_params

            # Estimate memory usage (in bytes)
            param_memory = total_params * 4  # 4 bytes per float32 parameter
            gradient_memory = trainable_params * 4  # 4 bytes per gradient
            optimizer_memory = trainable_params * 8  # 8 bytes per parameter for Adam

            # Calculate activation memory (rough estimate)
            activation_memory = 0
            # Basic calculation - estimate 4 bytes for each activation value
            for layer in model.layers:
                if hasattr(layer, "output_shape") and layer.output_shape:
                    # Calculate size based on output shape
                    if isinstance(layer.output_shape, tuple):
                        # Single output
                        shape_list = list(layer.output_shape)
                        if shape_list[0] is None:  # Batch dimension
                            shape_list[0] = sample_batch_size
                        # Calculate number of elements
                        num_elements = 1
                        for dim in shape_list:
                            if dim is not None:
                                num_elements *= dim
                        # 4 bytes per float32 element
                        activation_memory += num_elements * 4
                    elif isinstance(layer.output_shape, list):
                        # Multiple outputs
                        for out_shape in layer.output_shape:
                            if isinstance(out_shape, tuple):
                                shape_list = list(out_shape)
                                if shape_list[0] is None:
                                    shape_list[0] = sample_batch_size
                                num_elements = 1
                                for dim in shape_list:
                                    if dim is not None:
                                        num_elements *= dim
                                activation_memory += num_elements * 4

            # Print detailed breakdown if requested
            if detailed:
                self.logger.info("Layer-by-layer breakdown:")
                self.logger.info(
                    f"{'Layer':<30} {'Output Shape':<20} {'Params':<10}"
                )
                self.logger.info("-" * 60)
                
                for layer in model.layers:
                    params = layer.count_params()
                    shape_str = str(layer.output_shape)
                    # Truncate long shape strings
                    if len(shape_str) > 20:
                        shape_str = shape_str[:17] + "..."
                    
                    self.logger.info(
                        f"{layer.name:<30} {shape_str:<20} {params:<10,}"
                    )

            # Convert all memory values from bytes to MB for easier reading
            param_memory_mb = param_memory / (1024 * 1024)
            gradient_memory_mb = gradient_memory / (1024 * 1024)
            optimizer_memory_mb = optimizer_memory / (1024 * 1024)
            activation_memory_mb = activation_memory / (1024 * 1024)
            total_memory_mb = param_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_memory_mb

            # Log final results
            self.logger.info(f"Model memory profile for {model.name}:")
            self.logger.info(f"- Total parameters: {total_params:,} ({trainable_params:,} trainable)")
            self.logger.info(f"- Parameter memory: {param_memory_mb:.2f} MB")
            self.logger.info(f"- Gradient memory: {gradient_memory_mb:.2f} MB")
            self.logger.info(f"- Optimizer memory: {optimizer_memory_mb:.2f} MB")
            self.logger.info(f"- Activation memory: {activation_memory_mb:.2f} MB")
            self.logger.info(f"- Total estimated memory: {total_memory_mb:.2f} MB")

            # Return results as a dictionary
            return {
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
                "total_params": total_params,
                "param_memory_mb": param_memory_mb,
                "gradient_memory_mb": gradient_memory_mb,
                "optimizer_memory_mb": optimizer_memory_mb,
                "activation_memory_mb": activation_memory_mb,
                "total_memory_mb": total_memory_mb,
            }
            
        except Exception as e:
            self.logger.error(f"Error profiling model memory: {e}")
            return {
                "error": str(e),
                "trainable_params": 0,
                "total_memory_mb": 0,
            }
    
    def get_torch_device(self):
        """
        Get the appropriate PyTorch device (CUDA/GPU if available, otherwise CPU).
        Also handles DirectML detection for proper device placement.
        
        Returns:
            torch.device: PyTorch device object, or string 'cpu' as fallback if PyTorch is not available
        """
        # Ensure initialization before checking GPU availability
        if not self.initialized:
            self.initialize()
            
        try:
            import torch
            
            # Debug info to help identify the problem
            cuda_available = torch.cuda.is_available()
            self.logger.info(f"PyTorch CUDA available: {cuda_available}, TensorFlow GPUs detected: {self.has_gpu}")
            
            # Check for DirectML GPU
            directml_available = False
            torch_directml_device = None
            
            # Try to detect if torch-directml is installed
            try:
                import torch_directml
                directml_available = True
                dml_device_count = torch_directml.device_count()
                if dml_device_count > 0:
                    torch_directml_device = torch_directml.device(0)  # Use first DirectML device
                    self.logger.info(f"Detected {dml_device_count} DirectML device(s) for PyTorch")
            except ImportError:
                if self.is_directml:
                    self.logger.warning("DirectML detected for TensorFlow but torch-directml not installed for PyTorch")
            except Exception as e:
                self.logger.warning(f"Error initializing DirectML for PyTorch: {e}")
                
            # First choice: CUDA if available
            if cuda_available:
                device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                self.logger.info(f"Using PyTorch CUDA device: {device} ({device_name})")
                
                # Get and log available GPU memory
                try:
                    if hasattr(torch.cuda, 'memory_reserved'):
                        reserved = torch.cuda.memory_reserved(0) / 1024**2
                        allocated = torch.cuda.memory_allocated(0) / 1024**2
                        self.logger.info(f"CUDA memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
                except Exception as e:
                    self.logger.debug(f"Could not get CUDA memory stats: {e}")
                    
                return device
                
            # Second choice: DirectML if available
            elif directml_available and torch_directml_device is not None:
                self.logger.info(f"Using PyTorch DirectML device: {torch_directml_device}")
                return torch_directml_device
                
            # Fallback: CPU
            else:
                # No GPU available, check if we need to warn about mismatch
                if self.has_gpu:
                    if self.is_directml:
                        self.logger.warning("DirectML detected for TensorFlow but not available for PyTorch. Models will use CPU.")
                    else:
                        self.logger.warning("TensorFlow detected GPUs but PyTorch can't access them. This may indicate a CUDA/driver version mismatch.")
                
                device = torch.device("cpu")
                self.logger.info(f"Using PyTorch CPU device: {device}")
                return device
                
        except ImportError:
            self.logger.warning("PyTorch not installed. Returning string 'cpu' as fallback.")
            return "cpu"  # Return string as fallback
        except Exception as e:
            self.logger.error(f"Error getting PyTorch device: {e}")
            return "cpu"  # Return string as fallback in case of error
            
    def place_on_device(self, model):
        """
        Place a model on the appropriate device (GPU or CPU).
        Works with both TensorFlow and PyTorch models.
        
        Args:
            model: A TensorFlow or PyTorch model
            
        Returns:
            Tuple of (model on device, device name string)
        """
        try:
            # Check if it's a PyTorch model (has 'to' method)
            if hasattr(model, 'to') and callable(model.to):
                device = self.get_torch_device()
                model = model.to(device)
                return model, str(device)
                
            # If it's not PyTorch, assume it's TensorFlow
            # TensorFlow handles device placement automatically
            device = '/GPU:0' if self.has_gpu else '/CPU:0'
            
            # For TensorFlow, we just need to ensure the device is available
            # The model placement happens automatically during operations
            self.logger.info(f"Using TensorFlow device: {device}")
            
            return model, device
            
        except Exception as e:
            self.logger.warning(f"Error placing model on device: {e}")
            # Return original model with CPU device if anything fails
            return model, "cpu"
