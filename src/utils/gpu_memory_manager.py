import logging
import os
import time
import sys
import numpy as np
import tensorflow as tf

logger = logging.getLogger("GPU_Memory_Manager")

# Import centralized GPU memory management
from src.utils.gpu_memory_management import (
    clean_gpu_memory,
    configure_gpu_memory,
    get_memory_info,
    set_performance_mode,  # Import performance mode setter
    get_gpu_utilization,    # Import utilization checker
    stress_test_gpu,        # Import stress test function
)

# Import adaptive_memory_clean from memory_utils instead of duplicating it
from src.utils.memory_utils import adaptive_memory_clean


class GPUMemoryManager:
    """
    Advanced GPU memory manager for TensorFlow that provides:

    1. Dynamic memory growth and limits per GPU
    2. Automatic batch size adjustment based on available memory
    3. Efficient gradient accumulation for training with large models
    4. Model partitioning across multiple GPUs
    5. Periodic memory cleanup and garbage collection
    6. Memory usage tracking and reporting
    7. Training checkpointing to recover from OOM errors
    8. Performance mode to maximize GPU utilization and get fans spinning
    """

    def __init__(
        self,
        memory_limit_mb=None,
        allow_growth=True,
        visible_devices=None,
        reserve_memory=0.1,
    ):
        """
        Initialize the GPU memory manager.

        Args:
            memory_limit_mb: Memory limit per GPU in MB or None for no limit
            allow_growth: Whether to allow GPU memory growth
            visible_devices: List of GPU indices to use or None for all
            reserve_memory: Fraction of GPU memory to reserve for TensorFlow operations
        """
        self.memory_limit_mb = memory_limit_mb
        self.allow_growth = allow_growth
        self.visible_devices = visible_devices
        self.reserve_memory = reserve_memory
        self.logger = logging.getLogger("GPUMemoryManager")
        self.original_gpus = []
        self.logical_gpus = []
        self.initialized = False
        self.memory_usage_log = []
        self.batch_size_log = []
        self.performance_mode = False  # Default to standard mode
        self.gpu_utilization_history = []  # Track GPU utilization

        # Check for DirectML environment
        self.is_directml = (
            "TENSORFLOW_USE_DIRECTML" in os.environ
            or "DML_VISIBLE_DEVICES" in os.environ
        )
        if self.is_directml:
            self.logger.info("DirectML environment detected in GPUMemoryManager")

    def initialize(self):
        """
        Initialize the GPU memory manager and configure the GPUs using the centralized system.

        Returns:
            List of logical devices available after configuration
        """
        # Use centralized configuration
        config = {
            "allow_growth": self.allow_growth,
            "memory_limit_mb": self.memory_limit_mb,
            "visible_gpus": self.visible_devices,
            "directml_enabled": self.is_directml,  # Pass DirectML flag to configuration
        }

        # New: Check if performance mode is requested
        self.performance_mode = os.environ.get("TF_GPU_PERFORMANCE_MODE", "0") == "1"
        if self.performance_mode:
            set_performance_mode(True)
            self.logger.info("🔥 GPU PERFORMANCE MODE ENABLED - Maximum utilization")
            # Override allow_growth in performance mode
            self.allow_growth = False
            config["allow_growth"] = False
            config["performance_mode"] = True
            
        configure_gpu_memory(config)
        self.initialized = True

        # Get list of GPUs after configuration
        try:
            import tensorflow as tf

            self.original_gpus = tf.config.list_physical_devices("GPU")
            self.logical_gpus = tf.config.list_logical_devices("GPU")
            self.logger.info(f"Available logical GPUs: {len(self.logical_gpus)}")

            # Handle LSTM implementation for DirectML to avoid CudnnRNN errors
            if self.is_directml:
                self._configure_directml_compatibility()
        except Exception as e:
            self.logger.error(f"Error getting GPU list: {e}")

        return self.logical_gpus
    
    def enable_performance_mode(self):
        """Enable maximum performance mode for GPU utilization (will get fans spinning!)"""
        if not self.performance_mode:
            # Import here to avoid circular imports
            self.performance_mode = True
            set_performance_mode(True)
            self.logger.info("🔥 GPU Performance Mode ENABLED - Maximum utilization")
            self.clean_memory(True)
            
            # Apply aggressive optimizations
            try:
                import tensorflow as tf
                
                # Max performance settings
                self.logger.info("Applying aggressive GPU optimization settings...")
                
                # Set options for maximum GPU utilization
                options = {
                    "layout_optimizer": True,
                    "constant_folding": True,
                    "shape_optimization": True,
                    "remapping": True,
                    "arithmetic_optimization": True,
                    "dependency_optimization": True,
                    "loop_optimization": True,
                    "function_optimization": True,
                    "debug_stripper": True,
                    "auto_mixed_precision": True,  # Enable only if using mixed precision
                    "disable_meta_optimizer": False,
                    "scoped_allocator_optimization": True, # Scope allocations for better performance
                }
                tf.config.optimizer.set_experimental_options(options)
                
                # Always use JIT (XLA) for maximum performance
                tf.config.optimizer.set_jit(True)
                
                # Set high-performance thread settings
                if hasattr(tf.config.threading, "get_inter_op_parallelism_threads"):
                    # Use aggressive thread count settings
                    cores = os.cpu_count() or 4
                    tf.config.threading.set_inter_op_parallelism_threads(cores // 2)
                    tf.config.threading.set_intra_op_parallelism_threads(cores)
                
                # Set GPU memory options
                if hasattr(tf, "GPUOptions"):
                    # For older TensorFlow versions
                    gpu_options = tf.GPUOptions(
                        allow_growth=False,
                        per_process_gpu_memory_fraction=0.95, # Take 95% of GPU memory
                        force_gpu_compatible=True
                    )
                    config = tf.ConfigProto(gpu_options=gpu_options)
                    tf.keras.backend.set_session(tf.Session(config=config))
                
                # Environment variables for high performance
                os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
                os.environ["TF_GPU_THREAD_COUNT"] = str(cores)
                os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
                os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
                
                self.logger.info("Aggressive GPU optimizations applied!")
                
            except Exception as e:
                self.logger.error(f"Error applying performance settings: {e}")
                
            return True
        return False
    
    def disable_performance_mode(self):
        """Disable performance mode and return to normal operation"""
        if self.performance_mode:
            self.performance_mode = False
            set_performance_mode(False)
            self.logger.info("GPU Performance Mode DISABLED - Returning to normal operation")
            self.clean_memory(True)
            return True
        return False

    def run_gpu_stress_test(self, duration=10, intensity=0.9):
        """Run a GPU stress test to check performance and get those fans spinning!"""
        self.logger.info("Running GPU stress test to get fans spinning...")
        results = stress_test_gpu(duration_seconds=duration, intensity=intensity)
        
        if "error" in results:
            self.logger.error(f"GPU stress test failed: {results['error']}")
            return False
            
        self.logger.info(f"GPU stress test results:")
        self.logger.info(f"- Max utilization: {results['max_utilization']}%")
        self.logger.info(f"- Average utilization: {results['avg_utilization']:.1f}%")
        
        if "initial_temp" in results and "final_temp" in results:
            self.logger.info(f"- Temperature increased: {results['initial_temp']}°C → {results['final_temp']}°C")
            
        self.logger.info(f"- Fans spinning: {'YES!' if results['fans_spinning'] else 'No'}")
        
        return results

    # New method for rapid warmup to get GPU going
    def warmup_gpu(self, intensity=0.7):
        """
        Quickly warm up the GPU to get it ready for compute-intensive operations.
        This helps "get those fans spinning" by raising GPU temperature to operating levels.
        
        Args:
            intensity: How hard to push the GPU during warmup (0.1 to 1.0)
        
        Returns:
            Peak GPU utilization achieved
        """
        try:
            import tensorflow as tf
            import time
            
            self.logger.info(f"Warming up GPU with {intensity:.0%} intensity...")
            
            # Create a compute-intensive model using a large convolution
            input_shape = (256, 256, 3)
            input_tensor = tf.keras.layers.Input(shape=input_shape)
            x = input_tensor
            
            # Add compute-intensive layers
            for i in range(5):  # Multiple layers increase compute intensity
                x = tf.keras.layers.Conv2D(
                    filters=64 * (i+1), 
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same'
                )(x)
                if i % 2 == 0:
                    x = tf.keras.layers.MaxPooling2D()(x)
            
            # Add a final dense layer
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1000, activation='relu')(x)
            output = tf.keras.layers.Dense(10, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=input_tensor, outputs=output)
            
            # Enable XLA compilation for maximum efficiency
            @tf.function(jit_compile=True)
            def training_step(x):
                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss = tf.reduce_mean(pred)
                gradients = tape.gradient(loss, model.trainable_variables)
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
        return clean_gpu_memory(force_gc)

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

            # Estimate memory usage
            param_memory = total_params * 4  # 4 bytes per float32 parameter
            gradient_memory = trainable_params * 4  # 4 bytes per gradient
            optimizer_memory = trainable_params * 8  # 8 bytes per parameter for Adam

            # Calculate activation memory (rough estimate)
            activation_memory = 0
            if detailed:
                self.logger.info("Layer-by-layer breakdown:")
                self.logger.info(
                    f"{'Layer':<30} {'Output Shape':<20} {'Params':<10} {'Memory (MB)':<15}"
                )
                self.logger.info("-" * 75)

            for layer in model.layers:
                if hasattr(layer, "output_shape") and layer.output_shape:
                    shape = layer.output_shape
                    if isinstance(shape, list):
                        # Multiple outputs
                        layer_act_memory = sum(
                            np.prod(
                                [
                                    dim if dim is not None else sample_batch_size
                                    for dim in output_shape
                                ]
                            )
                            * 4
                            for output_shape in shape
                        )
                    else:
                        # Single output
                        layer_act_memory = (
                            np.prod(
                                [
                                    dim if dim is not None else sample_batch_size
                                    for dim in shape
                                ]
                            )
                            * 4
                        )

                    activation_memory += layer_act_memory

                    if detailed:
                        layer_params = layer.count_params()
                        layer_memory_mb = layer_act_memory / (1024 * 1024)
                        self.logger.info(
                            f"{layer.name:<30} {str(shape):<20} {layer_params:<10} {layer_memory_mb:<15.2f}"
                        )

            # Convert to MB
            param_memory_mb = param_memory / (1024 * 1024)
            gradient_memory_mb = gradient_memory / (1024 * 1024)
            optimizer_memory_mb = optimizer_memory / (1024 * 1024)
            activation_memory_mb = activation_memory / (1024 * 1024)
            total_memory_mb = (
                param_memory_mb
                + gradient_memory_mb
                + optimizer_memory_mb
                + activation_memory_mb
            )

            # Create profile
            profile = {
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
                "total_params": total_params,
                "param_memory_mb": param_memory_mb,
                "gradient_memory_mb": gradient_memory_mb,
                "optimizer_memory_mb": optimizer_memory_mb,
                "activation_memory_mb": activation_memory_mb,
                "total_memory_mb": total_memory_mb,
                "batch_size": sample_batch_size,
            }

            # Log memory stats
            self.logger.info(
                f"Model Memory Profile (batch size = {sample_batch_size}):"
            )
            self.logger.info(
                f"  Parameters: {total_params:,} ({param_memory_mb:.2f} MB)"
            )
            self.logger.info(
                f"  Gradients: {trainable_params:,} ({gradient_memory_mb:.2f} MB)"
            )
            self.logger.info(
                f"  Optimizer state: {trainable_params:,} ({optimizer_memory_mb:.2f} MB)"
            )
            self.logger.info(f"  Activations: ~ {activation_memory_mb:.2f} MB")
            self.logger.info(f"  Total estimated memory: {total_memory_mb:.2f} MB")

            # Clean up
            self.clean_memory()

            # Get GPU utilization monitoring
            try:
                # Get initial GPU utilization
                initial_util = get_gpu_utilization(0)
                
                # Run a forward and backward pass for more realistic utilization
                if hasattr(model, 'train_on_batch'):
                    with tf.GradientTape() as tape:
                        predictions = model(dummy_inputs)
                        loss = tf.reduce_mean(predictions)  # Dummy loss
                        
                        # Calculate gradients
                        if hasattr(model, 'trainable_variables'):
                            gradients = tape.gradient(loss, model.trainable_variables)
                    
                    # Get peak GPU utilization
                    peak_util = get_gpu_utilization(0)
                    
                    # Add to profile
                    profile["initial_gpu_utilization"] = initial_util
                    profile["peak_gpu_utilization"] = peak_util
                    
                    self.logger.info(f"  GPU Utilization: {initial_util}% → {peak_util}%")
                
                return profile
            
            except Exception as e:
                self.logger.error(f"Error in model memory profile: {e}")
                return {}

        except Exception as e:
            self.logger.error(f"Error in model memory profile: {e}")
            return {}

    def setup_checkpointing(self, model, checkpoint_dir, save_freq="epoch"):
        """
        Set up model checkpointing for safe recovery from OOM errors.

        Args:
            model: TensorFlow model to checkpoint
            checkpoint_dir: Directory to save checkpoints
            save_freq: How often to save ('epoch' or number of batches)

        Returns:
            TensorFlow callback for checkpointing
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create a unique model path with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_{timestamp}", "cp-{epoch:04d}.ckpt"
        )

        # Create the checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_freq=save_freq,
            verbose=1,
        )

        self.logger.info(f"Checkpoint callback created. Saving to {checkpoint_dir}")
        return checkpoint_callback

    def create_memory_monitoring_callback(self, log_frequency=100):
        """
        Create a Keras callback for monitoring GPU memory during training.

        Args:
            log_frequency: How often to log memory usage (in batches)

        Returns:
            Keras callback for memory monitoring
        """

        class MemoryMonitorCallback(tf.keras.callbacks.Callback):
            def __init__(self, memory_manager, log_frequency):
                super().__init__()
                self.memory_manager = memory_manager
                self.log_frequency = log_frequency
                self.batch_logs = []

            def on_train_begin(self, logs=None):
                self.memory_manager.log_memory_usage("train_begin")

            def on_train_end(self, logs=None):
                self.memory_manager.log_memory_usage("train_end")

            def on_epoch_begin(self, epoch, logs=None):
                self.memory_manager.log_memory_usage(f"epoch_{epoch}_begin")

            def on_epoch_end(self, epoch, logs=None):
                self.memory_manager.log_memory_usage(f"epoch_{epoch}_end")

            def on_batch_begin(self, batch, logs=None):
                if batch % self.log_frequency == 0:
                    self.memory_manager.log_memory_usage(f"batch_{batch}_begin")

            def on_batch_end(self, batch, logs=None):
                if batch % self.log_frequency == 0:
                    self.memory_manager.log_memory_usage(f"batch_{batch}_end")

                    # Store batch metrics
                    if logs:
                        self.batch_logs.append(
                            {
                                "batch": batch,
                                "memory_mb": self.memory_manager.get_available_memory(),
                                **logs,
                            }
                        )

        return MemoryMonitorCallback(self, log_frequency)

    def get_memory_usage_report(self, save_path=None):
        """
        Generate a report of memory usage during training.

        Args:
            save_path: Path to save the report (or None to return as string)

        Returns:
            Report as a string if save_path is None
        """
        if not self.memory_usage_log:
            return "No memory usage data available."

        # Convert log to DataFrame for analysis
        import pandas as pd

        df = pd.DataFrame(self.memory_usage_log)

        # Generate report
        report = []
        report.append("GPU MEMORY USAGE REPORT")
        report.append("=" * 80)

        # Summary statistics
        report.append("\nSummary Statistics:")
        report.append("-" * 80)

        for device_idx in df["device_idx"].unique():
            device_df = df[df["device_idx"] == device_idx]
            report.append(f"\nGPU {device_idx}:")
            report.append(f"  Min memory: {device_df['memory_mb'].min():.2f} MB")
            report.append(f"  Max memory: {device_df['memory_mb'].max():.2f} MB")
            report.append(f"  Mean memory: {device_df['memory_mb'].mean():.2f} MB")
            report.append(
                f"  Memory range: {device_df['memory_mb'].max() - device_df['memory_mb'].min():.2f} MB"
            )

        # Peak memory usage
        report.append("\nPeak Memory Usage:")
        report.append("-" * 80)

        peak_usage = df.loc[df["memory_mb"].idxmax()]
        report.append(f"Peak memory: {peak_usage['memory_mb']:.2f} MB")
        report.append(f"At: {peak_usage['tag']}")
        report.append(f"On device: GPU {peak_usage['device_idx']}")

        # Memory by operation
        report.append("\nMemory by Operation:")
        report.append("-" * 80)

        for tag in sorted(set([t for t in df["tag"] if t])):
            tag_df = df[df["tag"] == tag]
            report.append(f"{tag:30s}: {tag_df['memory_mb'].mean():.2f} MB (mean)")

        # Batch size information
        if self.batch_size_log:
            report.append("\nBatch Size Information:")
            report.append("-" * 80)

            for entry in self.batch_size_log:
                report.append(f"Device GPU {entry['device_idx']}:")
                report.append(f"  Optimal batch size: {entry['optimal_batch']}")
                report.append(f"  Safe batch size: {entry['safe_batch']}")
                report.append(f"  Input shape: {entry['input_shape']}")

        full_report = "\n".join(report)

        # Save if needed
        if save_path:
            with open(save_path, "w") as f:
                f.write(full_report)

        return full_report


# Helper functions for gradient accumulation
def train_with_gradient_accumulation(
    model,
    train_dataset,
    steps_per_epoch,
    accumulation_steps=1,
    optimizer=None,
    loss_fn=None,
    metrics=None,
    callbacks=None,
    epochs=1,
):
    """
    Train a model with gradient accumulation to simulate larger batch sizes.

    Args:
        model: TensorFlow model to train
        train_dataset: Dataset for training
        steps_per_epoch: Number of steps per epoch
        accumulation_steps: Number of gradient accumulation steps
        optimizer: Optimizer to use (default: model.optimizer)
        loss_fn: Loss function (required if model is not compiled)
        metrics: Metrics to track (default: model.metrics)
        callbacks: List of callbacks
        epochs: Number of epochs to train

    Returns:
        History object
    """
    # Use model's optimizer if not provided
    optimizer = optimizer or model.optimizer
    if optimizer is None:
        raise ValueError("Optimizer must be provided or model must be compiled")

    # Use model's metrics if not provided
    metrics = metrics or model.metrics

    # Check if loss function is available
    if loss_fn is None and not hasattr(model, "loss"):
        raise ValueError("Loss function must be provided or model must be compiled")
    loss_fn = loss_fn or model.loss

    # Initialize variables to track metrics
    history = {"loss": []}
    for metric in metrics:
        history[metric.name] = []

    # Run callbacks if provided
    if callbacks:
        for callback in callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin()

    # Training loop
    for epoch in range(epochs):
        # Reset metrics
        for metric in metrics:
            metric.reset_states()

        # Run epoch callbacks
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "on_epoch_begin"):
                    callback.on_epoch_begin(epoch)

        epoch_loss = 0

        # Create a reference variable to store gradients
        all_gradients = None
        num_accumulated = 0

        # Iterate through dataset
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            # Run batch callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_batch_begin"):
                        callback.on_batch_begin(step)

            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(x_batch, training=True)

                # Calculate loss
                batch_loss = loss_fn(y_batch, predictions)

                # Add regularization losses if any
                if model.losses:
                    batch_loss += sum(model.losses)

                # Scale the loss by accumulation steps (to keep gradients comparable)
                batch_loss = batch_loss / accumulation_steps

            # Calculate gradients
            gradients = tape.gradient(batch_loss, model.trainable_variables)

            # Accumulate gradients
            if all_gradients is None:
                all_gradients = [tf.Variable(tf.zeros_like(g)) for g in gradients]

            for i, grad in enumerate(gradients):
                if grad is not None:
                    all_gradients[i].assign_add(grad)

            num_accumulated += 1

            # Update weights when we've accumulated enough gradients
            if num_accumulated >= accumulation_steps or step == steps_per_epoch - 1:
                # Apply accumulated gradients
                optimizer.apply_gradients(zip(all_gradients, model.trainable_variables))

                # Update metrics
                for metric in metrics:
                    metric.update_state(y_batch, predictions)

                # Reset accumulated gradients
                all_gradients = None
                num_accumulated = 0

            # Update epoch loss
            epoch_loss += batch_loss * accumulation_steps

            # Run batch end callbacks
            if callbacks:
                logs = {"loss": batch_loss.numpy()}
                for metric in metrics:
                    logs[metric.name] = metric.result().numpy()

                for callback in callbacks:
                    if hasattr(callback, "on_batch_end"):
                        callback.on_batch_end(step, logs)

        # Calculate epoch metrics
        epoch_loss /= steps_per_epoch
        history["loss"].append(
            epoch_loss.numpy() if hasattr(epoch_loss, "numpy") else epoch_loss
        )

        for metric in metrics:
            metric_value = metric.result().numpy()
            history[metric.name].append(metric_value)

        # Print epoch results
        metrics_str = " - ".join([f"{k}: {v[-1]:.4f}" for k, v in history.items()])
        print(f"\nEpoch {epoch+1}/{epochs} - {metrics_str}")

        # Run epoch end callbacks
        if callbacks:
            logs = {"loss": history["loss"][-1]}
            for metric in metrics:
                logs[metric.name] = history[metric.name][-1]

            for callback in callbacks:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch, logs)

    # Run train end callbacks
    if callbacks:
        for callback in callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end()

    return history


# Example usage
def example_usage():
    # Initialize the GPU memory manager
    memory_manager = GPUMemoryManager(
        memory_limit_mb=None,  # No specific limit
        allow_growth=True,  # Allow memory growth
        reserve_memory=0.1,  # Reserve 10% of memory
    )

    # Configure GPUs
    memory_manager.initialize()

    # Enable mixed precision for better performance and lower memory usage
    memory_manager.enable_mixed_precision()

    # Create a simple model for demonstration
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Profile the model's memory usage
    memory_profile = memory_manager.model_memory_profile(model, sample_batch_size=32)

    # Estimate optimal batch size
    optimal_batch = memory_manager.estimate_optimal_batch_size(
        model, sample_input_shape=(28, 28, 1), min_batch=1, max_batch=512
    )

    # Set up gradient accumulation
    desired_batch = 256  # What we want
    actual_batch = optimal_batch  # What fits in memory
    accumulation_steps = memory_manager.get_gradient_accumulation_steps(
        desired_batch, actual_batch
    )

    print(
        f"Using actual batch size of {actual_batch} with {accumulation_steps} "
        f"accumulation steps to simulate batch size of {desired_batch}"
    )

    # Set up distribution strategy for multi-GPU training
    memory_manager.create_training_strategy()

    # Create memory monitoring callback
    memory_callback = memory_manager.create_memory_monitoring_callback(log_frequency=10)

    # Set up checkpointing
    checkpoint_callback = memory_manager.setup_checkpointing(
        model, checkpoint_dir="checkpoints"
    )

    # Load some example data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., tf.newaxis].astype("float32") / 255.0

    # Prepare the dataset
    buffer_size = len(x_train)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size).batch(actual_batch)

    # Train with gradient accumulation
    train_with_gradient_accumulation(
        model=model,
        train_dataset=train_dataset,
        steps_per_epoch=len(x_train) // actual_batch,
        accumulation_steps=accumulation_steps,
        callbacks=[memory_callback, checkpoint_callback],
        epochs=3,
    )

    # Generate memory usage report
    report = memory_manager.get_memory_usage_report(save_path="memory_report.txt")
    print("Memory report saved to memory_report.txt")


if __name__ == "__main__":
    example_usage()
