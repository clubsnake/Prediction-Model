"""
Centralized GPU memory management to avoid conflicts between different modules.
This file should be imported BEFORE any TensorFlow operations.

Note: This module only manages GPU memory allocation and does not affect model
hyperparameters like batch sizes, which are determined solely by Optuna during tuning.
"""

import logging
import os
import platform
import sys
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track if memory has already been configured
_MEMORY_CONFIGURED = False

# Add performance mode flag - when True, maximizes GPU utilization
_PERFORMANCE_MODE = False

def set_performance_mode(enabled=True):
    """
    Enable or disable performance mode.
    
    In performance mode, memory growth is disabled and maximum GPU utilization is prioritized
    over multi-model efficiency. This will push your GPU harder and "get those fans spinning".
    
    Args:
        enabled: Whether to enable performance mode
        
    Returns:
        Current performance mode state
    """
    global _PERFORMANCE_MODE
    _PERFORMANCE_MODE = enabled
    logger.info(f"GPU Performance Mode {'ENABLED' if enabled else 'DISABLED'}")
    
    # Apply performance settings if memory already configured
    if _MEMORY_CONFIGURED:
        _apply_performance_optimizations(enabled)
    
    return _PERFORMANCE_MODE

def _apply_performance_optimizations(enabled=True):
    """Apply aggressive performance optimizations to maximize GPU utilization"""
    try:
        import tensorflow as tf
        
        if enabled:
            # For maximum GPU utilization:
            # 1. Enable XLA JIT compilation which can increase performance
            tf.config.optimizer.set_jit(True)
            
            # 2. Set aggressive GPU options
            gpu_options = tf.config.optimizer.get_experimental_options()
            aggressive_options = {
                "layout_optimizer": True,
                "constant_folding": True,
                "shape_optimization": True,
                "remapping": True,
                "arithmetic_optimization": True,
                "dependency_optimization": True,
                "loop_optimization": True,
                "auto_mixed_precision": True,
                "disable_meta_optimizer": False,
                "min_graph_nodes": 1,
            }
            tf.config.optimizer.set_experimental_options(aggressive_options)
            
            # 3. Use autograph aggressively 
            os.environ["TF_AUTOGRAPH_STRICT_CONVERSION"] = "1"
            
            # 4. Allow TensorFlow to use more GPU memory for caching
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
            
            # 5. Set CUDA memory allocation to maximum performance
            os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            
            # 6. Use cudnn autotune for best performance
            os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
            
            # 7. Disable operations on CPU when possible
            os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
            
            logger.info("Applied aggressive GPU performance optimizations")
        else:
            # Reset to default behavior
            tf.config.optimizer.set_jit(False)
            tf.config.optimizer.set_experimental_options({})
            
            # Reset environment variables
            if "TF_AUTOGRAPH_STRICT_CONVERSION" in os.environ:
                del os.environ["TF_AUTOGRAPH_STRICT_CONVERSION"]
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            if "TF_GPU_ALLOCATOR" in os.environ:
                del os.environ["TF_GPU_ALLOCATOR"]
            if "TF_GPU_THREAD_MODE" in os.environ:
                del os.environ["TF_GPU_THREAD_MODE"]
                
            logger.info("Reset GPU optimizations to default")
    except Exception as e:
        logger.error(f"Error applying performance optimizations: {e}")

def detect_amd_gpu() -> bool:
    """Detect if the system has an AMD GPU."""
    try:
        if platform.system() == "Windows":
            import subprocess

            output = subprocess.check_output(
                "wmic path win32_VideoController get name", shell=True
            ).decode()
            return any(gpu in output.lower() for gpu in ["amd", "radeon", "rx"])
        return False
    except Exception as e:
        logger.warning(f"Error detecting GPU type: {e}")
        return False

def configure_gpu_memory(config: dict = None) -> Dict[str, Any]:
    """
    Configure GPU memory settings. This should be called only once.
    
    Note: This only affects hardware resource allocation and never modifies
    model hyperparameters like batch sizes that should be controlled by Optuna.

    Args:
        config: Dictionary with configuration options

    Returns:
        Dictionary with configuration results
    """
    global _MEMORY_CONFIGURED, _PERFORMANCE_MODE

    # Default configuration
    DEFAULT_CONFIG = {
        "allow_growth": True,
        "memory_limit_mb": None,
        "gpu_memory_fraction": "auto",
        "use_xla": True,
        "mixed_precision": False,  # Default to False for safety
        "visible_gpus": None,
        "directml_enabled": True,
        "use_gpu": True,
        "performance_mode": _PERFORMANCE_MODE,  # Use current performance mode
    }

    # Don't reconfigure if already done
    if _MEMORY_CONFIGURED:
        logger.warning(
            "GPU memory already configured. Ignoring new configuration request."
        )
        return {"success": False, "reason": "Already configured"}

    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults for missing values
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value

    # Update performance mode from config
    _PERFORMANCE_MODE = config.get("performance_mode", _PERFORMANCE_MODE)
    
    # If in performance mode, override certain settings
    if _PERFORMANCE_MODE:
        logger.info("PERFORMANCE MODE ENABLED: Configuring for maximum GPU utilization")
        config["allow_growth"] = False  # Disable growth for maximum performance
        config["use_xla"] = True  # Always use XLA
        config["gpu_memory_fraction"] = 0.95  # Use 95% of GPU memory

    # Explicitly check for TF_FORCE_FLOAT32 environment variable
    if "TF_FORCE_FLOAT32" in os.environ and os.environ["TF_FORCE_FLOAT32"] == "1":
        config["mixed_precision"] = False
        logger.info("TF_FORCE_FLOAT32 set to 1, forcing float32 precision")

    results = {"success": True, "gpus_found": 0, "performance_mode": _PERFORMANCE_MODE}

    # Disable GPU if requested
    if not config["use_gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("GPU usage disabled by configuration")
        _MEMORY_CONFIGURED = True
        return results

    try:
        # Check for DirectML environment
        is_directml = False
        if platform.system() == "Windows" and config["directml_enabled"]:
            if (
                "TENSORFLOW_USE_DIRECTML" in os.environ
                or "DML_VISIBLE_DEVICES" in os.environ
            ):
                is_directml = True
                logger.info("DirectML environment detected")
                # Set environment variables for DirectML if not already set
                if "TENSORFLOW_USE_DIRECTML" not in os.environ:
                    os.environ["TENSORFLOW_USE_DIRECTML"] = "1"

                # Configure environment for DirectML
                os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

                # Disable CUDA-specific operations
                os.environ["TF_DIRECTML_KERNEL_SELECTION_ONLY"] = "1"

        # Import TensorFlow
        import tensorflow as tf

        # Handle DirectML configuration specifically
        if is_directml:
            logger.info("Configuring TensorFlow for DirectML")
            # Force TensorFlow to use compatible implementations
            os.environ["TF_DIRECTML_KERNEL_FALLBACK"] = "1"

            # Set implementation selection for LSTM to avoid CudnnRNN errors
            # This forces TensorFlow to use the standard implementation instead of Cudnn
            tf.keras.backend.set_session = lambda session: session
            tf.config.optimizer.set_experimental_options(
                {
                    "auto_mixed_precision": config["mixed_precision"],
                    "disable_meta_optimizer": False,
                }
            )

            results["backend"] = "directml"
            _MEMORY_CONFIGURED = True
            return results

        # Find GPUs
        gpus = tf.config.list_physical_devices("GPU")
        results["gpus_found"] = len(gpus)

        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s)")

            try:
                # Apply memory growth setting - but not in performance mode
                if config["allow_growth"] and not _PERFORMANCE_MODE:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Memory growth enabled for all GPUs")
                elif not config["allow_growth"] or _PERFORMANCE_MODE:
                    logger.info("Memory growth DISABLED for maximum GPU utilization")

                # Enable XLA JIT compilation if requested
                if config["use_xla"]:
                    tf.config.optimizer.set_jit(True)
                    logger.info("XLA JIT compilation enabled")

                # Configure mixed precision - explicitly log the setting
                logger.info(
                    f"Mixed precision setting from config: {config['mixed_precision']}"
                )

                # This is the critical part: ensure we explicitly set the precision policy
                if config["mixed_precision"]:
                    tf.keras.mixed_precision.set_global_policy("mixed_float16")
                    logger.info("Mixed precision ENABLED (float16)")
                else:
                    tf.keras.mixed_precision.set_global_policy("float32")
                    logger.info("Mixed precision DISABLED (float32)")

                # Apply performance optimizations if in performance mode
                if _PERFORMANCE_MODE:
                    _apply_performance_optimizations(True)

                # Verify the policy was applied
                actual_policy = tf.keras.mixed_precision.global_policy()
                logger.info(f"Active precision policy: {actual_policy.name}")

                results["backend"] = "cuda" if len(gpus) > 0 else "cpu"
                _MEMORY_CONFIGURED = True

            except RuntimeError as e:
                logger.error(f"Error configuring GPU memory: {e}")
                results["success"] = False
                results["error"] = str(e)

                # Try a fallback configuration
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Fallback to basic memory growth configuration")
                    _MEMORY_CONFIGURED = True
                    results["success"] = True
                    results["fallback"] = True
                except Exception as e2:
                    logger.error(f"Fallback configuration failed: {e2}")
        else:
            logger.warning("No GPUs detected. Running on CPU only.")
            results["backend"] = "cpu"

    except ImportError:
        logger.error("TensorFlow not installed")
        results["success"] = False
        results["error"] = "TensorFlow not installed"
    except Exception as e:
        logger.error(f"Unexpected error configuring GPU memory: {e}")
        results["success"] = False
        results["error"] = str(e)

    return results

def configure_mixed_precision(use_mixed_precision=None):
    """
    Simplified function to configure mixed precision settings

    Args:
        use_mixed_precision: Boolean to force enable/disable mixed precision,
                           or None to auto-detect from config

    Returns:
        Current precision policy
    """
    try:
        import tensorflow as tf

        # TF_FORCE_FLOAT32 takes highest priority
        if "TF_FORCE_FLOAT32" in os.environ and os.environ["TF_FORCE_FLOAT32"] == "1":
            use_mixed_precision = False
            logger.info("TF_FORCE_FLOAT32 is set - forcing float32 precision")

        # If not specified, try to get from config
        if use_mixed_precision is None:
            try:
                # Avoid circular imports
                sys.path.append(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
                )
                from config.config_loader import get_value

                use_mixed_precision = get_value("hardware.use_mixed_precision", False)
                logger.info(f"Mixed precision from config: {use_mixed_precision}")
            except ImportError:
                use_mixed_precision = False
                logger.warning(
                    "Could not import config_loader, defaulting to float32 precision"
                )

        # Apply the policy and log it clearly
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            logger.info("Setting mixed_float16 precision policy")
        else:
            policy = tf.keras.mixed_precision.Policy("float32")
            logger.info("Setting float32 precision policy")

        # Apply the policy
        tf.keras.mixed_precision.set_global_policy(policy)

        # Verify and log the current policy
        current_policy = tf.keras.mixed_precision.global_policy()
        logger.info(f"Verified active precision policy: {current_policy.name}")

        return current_policy

    except Exception as e:
        logger.error(f"Error configuring mixed precision: {e}")
        return None

def clean_gpu_memory(force_gc=True):
    """
    Clean GPU memory by clearing TensorFlow caches and running garbage collection.
    """
    try:
        import gc

        import tensorflow as tf

        # Clear TensorFlow session
        if hasattr(tf.keras.backend, "clear_session"):
            tf.keras.backend.clear_session()

        # Run garbage collection if requested
        if force_gc:
            gc.collect()

        return True
    except Exception as e:
        logger.error(f"Error cleaning GPU memory: {e}")
        return False

# Add a helper function to get memory info
def get_memory_info(device_idx=0):
    """Get memory info for the specified GPU device"""
    try:

        # Check if DirectML is being used
        is_directml = (
            "TENSORFLOW_USE_DIRECTML" in os.environ
            or "DML_VISIBLE_DEVICES" in os.environ
        )

        if is_directml:
            # DirectML doesn't provide detailed memory info, return estimates
            return {
                "total_mb": 8192,  # Estimate 8GB
                "free_mb": 4096,  # Estimate 4GB free
                "used_mb": 4096,  # Estimate 4GB used
                "current_mb": 4096,  # Estimate current available
            }

        if HAS_GPUTIL:
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if device_idx < len(gpus):
                    gpu = gpus[device_idx]
                    total_mb = gpu.memoryTotal
                    used_mb = gpu.memoryUsed
                    free_mb = total_mb - used_mb
                    return {
                        "total_mb": total_mb,
                        "free_mb": free_mb,
                        "used_mb": used_mb,
                        "current_mb": free_mb,
                    }
            except Exception as e:
                logger.warning(f"Error getting GPU memory info from GPUtil: {e}")

        # Fallback to conservative estimates
        return {
            "total_mb": 4096,  # 4GB estimate
            "free_mb": 2048,  # 2GB estimate
            "used_mb": 2048,  # 2GB estimate
            "current_mb": 2048,  # 2GB estimate
        }
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {"error": str(e)}

def get_gpu_utilization(device_idx=0):
    """Get current GPU utilization percentage (compute usage, not just memory)"""
    try:
        import subprocess
        
        result = subprocess.run(
            ["nvidia-smi", 
             f"--id={device_idx}", 
             "--query-gpu=utilization.gpu", 
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            check=True,
            universal_newlines=True
        )
        
        utilization = int(result.stdout.strip())
        return utilization
    except Exception:
        # If nvidia-smi fails or isn't available, try GPUtil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if device_idx < len(gpus):
                return gpus[device_idx].load * 100  # Convert to percentage
        except ImportError:
            pass
            
        # If all else fails, return an estimate based on memory usage
        mem_info = get_memory_info(device_idx)
        total = mem_info.get("total_mb", 0)
        used = mem_info.get("used_mb", 0)
        if total > 0:
            return (used / total) * 100
        return 0

def stress_test_gpu(duration_seconds=10, intensity=0.9):
    """
    Run a stress test on the GPU to check stability and maximum performance.
    
    Args:
        duration_seconds: How long to run the test
        intensity: How intensive the test should be (0.1-1.0)
        
    Returns:
        Dict with test results
    """
    try:
        import tensorflow as tf
        import time
        import numpy as np
        
        # Create a large tensor to fill GPU memory
        memory_info = get_memory_info(0)
        total_gb = memory_info.get("total_mb", 8192) / 1024  # Convert to GB
        
        # Calculate tensor size to use a percentage of available memory
        tensor_gb = total_gb * intensity * 0.8  # Use 80% of desired intensity
        
        # Create a compute-intensive model
        logger.info(f"Running GPU stress test at {intensity:.1%} intensity for {duration_seconds}s")
        logger.info(f"Creating tensors using approximately {tensor_gb:.2f}GB GPU memory")
        
        # Use half precision if available to allow larger tensors
        dtype = tf.float16 if tensor_gb > 2 else tf.float32
        
        # Calculate tensor dimensions (prefer square tensors)
        elements = int(tensor_gb * (1024**3) / (2 if dtype == tf.float16 else 4))
        side_length = int(np.sqrt(elements // 4))  # 4 dims for matrix multiplication
        
        # Create a stress test model - matrix multiplications are compute-intensive
        @tf.function(jit_compile=True)  # Use XLA for faster execution
        def stress_op(x):
            for _ in range(5):  # Multiple iterations increases compute intensity
                x = tf.matmul(x, x)
            return x
        
        # Create a large tensor
        with tf.device("/gpu:0"):
            # Use random data to prevent optimization shortcuts
            x = tf.random.normal([side_length, side_length], dtype=dtype)
            
            # Record start time and initial utilization
            start_time = time.time()
            initial_util = get_gpu_utilization(0)
            utilization_samples = [initial_util]
            temperature_samples = []
            
            # Try to get initial temperature if available
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, check=True, universal_newlines=True
                )
                initial_temp = int(result.stdout.strip())
                temperature_samples.append(initial_temp)
                has_temp = True
            except:
                has_temp = False
                
            logger.info(f"Starting stress test - initial GPU utilization: {initial_util}%")
            
            # Run the stress test loop
            while time.time() - start_time < duration_seconds:
                # Run compute-intensive operations
                result = stress_op(x)
                
                # Force execution of the op (prevent optimization removing it)
                tf.debugging.check_numerics(result, "Stress test error")
                
                # Sample utilization every second
                if int(time.time() - start_time) != len(utilization_samples) - 1:
                    utilization_samples.append(get_gpu_utilization(0))
                    
                    # Sample temperature if available
                    if has_temp:
                        try:
                            result = subprocess.run(
                                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE, check=True, universal_newlines=True
                            )
                            temperature_samples.append(int(result.stdout.strip()))
                        except:
                            pass
            
            # Calculate results
            end_time = time.time()
            duration = end_time - start_time
            max_util = max(utilization_samples)
            avg_util = sum(utilization_samples) / len(utilization_samples)
            
            # Temperature changes if available
            temp_info = {}
            if has_temp and len(temperature_samples) >= 2:
                temp_info = {
                    "initial_temp": temperature_samples[0],
                    "final_temp": temperature_samples[-1],
                    "temp_increase": temperature_samples[-1] - temperature_samples[0],
                    "max_temp": max(temperature_samples)
                }
                
            logger.info(f"Stress test complete - max GPU utilization: {max_util}%, avg: {avg_util:.1f}%")
            if temp_info:
                logger.info(f"Temperature increased from {temp_info['initial_temp']}°C to {temp_info['final_temp']}°C")
            
            # Clean up and return results
            clean_gpu_memory(True)
            
            # Prepare results
            results = {
                "duration": duration,
                "initial_utilization": initial_util,
                "max_utilization": max_util,
                "avg_utilization": avg_util,
                "utilization_samples": utilization_samples,
                "fans_spinning": avg_util > 80,  # Fan threshold
                **temp_info  # Add temperature info if available
            }
            
            return results
            
    except Exception as e:
        logger.error(f"Error during GPU stress test: {e}")
        return {"error": str(e), "fans_spinning": False}

# Define HAS_GPUTIL for use above
try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    logger.warning("GPUtil not installed. Detailed GPU monitoring will be limited.")
    HAS_GPUTIL = False

# Add an automatic resource management function to replace the static performance mode

def apply_optimal_gpu_settings(workload_intensity=None):
    """
    Dynamically apply optimal GPU settings based on workload intensity.
    
    This replaces the old static performance mode with a more dynamic approach
    that automatically adjusts settings based on the current workload needs.
    
    Args:
        workload_intensity: Optional float from 0.0-1.0 indicating workload intensity
                           (None = auto-detect based on current utilization)
    Returns:
        Dict of applied settings
    """
    import os
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        import tensorflow as tf
        
        # Detect GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.info("No GPUs detected - using CPU only")
            return {"status": "cpu_only"}
            
        # Determine workload intensity if not provided
        if workload_intensity is None:
            # Try to detect current GPU utilization
            try:
                # First attempt: check memory usage via TensorFlow
                gpu_util = 0.0
                for i, gpu in enumerate(gpus):
                    try:
                        mem_info = tf.config.experimental.get_memory_info(f"GPU:{i}")
                        if "current" in mem_info and "total" in mem_info:
                            gpu_util = max(gpu_util, mem_info["current"] / mem_info["total"])
                    except:
                        pass
                        
                # Second attempt: use nvidia-smi
                if gpu_util == 0.0:
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                              stdout=subprocess.PIPE, check=True)
                        util_values = [float(x) for x in result.stdout.decode().strip().split('\n')]
                        if util_values:
                            gpu_util = max(util_values) / 100.0
                    except:
                        pass
                        
                # Set workload intensity based on utilization
                if gpu_util > 0.0:
                    workload_intensity = min(1.0, gpu_util * 1.5)  # Scale up slightly
                else:
                    # Default to medium intensity when detection fails
                    workload_intensity = 0.5
                    
            except Exception as e:
                logger.debug(f"Error detecting GPU utilization: {e}")
                workload_intensity = 0.5  # Default to medium intensity
        
        # Apply settings based on intensity
        settings = {}
        
        # For high intensity workloads (training, complex inference)
        if workload_intensity > 0.7:
            # Enable XLA JIT compilation
            tf.config.optimizer.set_jit(True)
            
            # Set aggressive optimization options
            optimizer_options = {
                "layout_optimizer": True,
                "constant_folding": True,
                "shape_optimization": True,
                "remapping": True,
                "arithmetic_optimization": True,
                "dependency_optimization": True,
                "loop_optimization": True,
                "function_optimization": True,
                "debug_stripper": True,
                "auto_mixed_precision": True,
                "disable_meta_optimizer": False,
                "scoped_allocator_optimization": True,
            }
            tf.config.optimizer.set_experimental_options(optimizer_options)
            
            # Environment variables for high performance
            os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
            os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
            os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            
            settings = {
                "profile": "high_performance",
                "memory_growth": False,
                "xla_enabled": True,
                "mixed_precision": True,
            }
            
            logger.info("Applied high-performance GPU optimization settings")
            
        # For medium intensity workloads
        elif workload_intensity > 0.3:
            # More balanced settings
            tf.config.optimizer.set_jit(True)  # Still use XLA
            
            optimizer_options = {
                "layout_optimizer": True,
                "constant_folding": True, 
                "auto_mixed_precision": True,
            }
            tf.config.optimizer.set_experimental_options(optimizer_options)
            
            # Allow memory growth for more flexibility with multiple models
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.debug(f"Error setting memory growth: {e}")
                    
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
            
            settings = {
                "profile": "balanced",
                "memory_growth": True,
                "xla_enabled": True,
                "mixed_precision": True,
            }
            
            logger.info("Applied balanced GPU optimization settings")
            
        # For low intensity or multiple small workloads
        else:
            # Conservative settings optimized for memory efficiency
            tf.config.optimizer.set_jit(False)  # Disable XLA for more predictable memory usage
            
            # Enable memory growth to conserve memory
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.debug(f"Error setting memory growth: {e}")
            
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
            settings = {
                "profile": "memory_efficient",
                "memory_growth": True,
                "xla_enabled": False,
                "mixed_precision": False,
            }
            
            logger.info("Applied memory-efficient GPU settings")
            
        return settings
    
    except Exception as e:
        logger.warning(f"Error applying GPU settings: {e}")
        return {"status": "error", "message": str(e)}
