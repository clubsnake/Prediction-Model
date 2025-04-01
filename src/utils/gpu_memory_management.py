"""
GPU memory management utilities to optimize GPU utilization for deep learning models.
This module provides functions to:
1. Configure GPU memory allocation 
2. Monitor GPU usage
3. Clean up GPU memory when needed
4. Enable mixed precision training
"""
from typing import Any, Dict, List, Union
import os
import platform
import subprocess
import logging
import gc
import numpy as np
import threading
import time

# Add thread-local storage for GPU contexts
_thread_gpu_contexts = threading.local()

# Add lock for GPU operations to prevent conflicts
_gpu_lock = threading.RLock()

logger = logging.getLogger(__name__)

# Track if memory has already been configured
_MEMORY_CONFIGURED = False

def detect_amd_gpu() -> bool:
    """Detect if the system has an AMD GPU."""
    try:
        if platform.system() == "Windows":
            
            output = subprocess.check_output(
                "wmic path win32_VideoController get name", shell=True
            ).decode()
            return any(gpu in output.lower() for gpu in ["amd", "radeon", "rx"])
        return False
    except Exception as e:
        logger.warning(f"Error detecting GPU type: {e}")
        return False

def check_gpu_availability():
    """
    Check if GPU is available through TensorFlow and return device information.
    
    Returns:
        dict: Information about available devices
    """
    try:
        import tensorflow as tf
                
        gpus = tf.config.list_physical_devices('GPU')
        
        # Get detailed GPU info
        gpu_info = {}
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_info[i] = details
            except Exception as e:
                gpu_info[i] = {"name": str(gpu), "error": str(e)}
                
        result = {
            'gpus_available': len(gpus) > 0,
            'gpu_count': len(gpus),
            'gpu_info': gpu_info,
            'tf_version': tf.__version__
        }
        
        # Check for CUDA availability in more detail
        try:
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            result['cuda_visible_devices'] = cuda_visible
        except Exception:
            pass
            
        # Add XLA availability
        try:
            result['xla_available'] = hasattr(tf, 'function') and callable(tf.function)
        except Exception:
            result['xla_available'] = False
            
        return result
        
    except ImportError:
        logger.warning("TensorFlow not available")
        return {'gpus_available': False, 'error': 'TensorFlow not available'}
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return {'gpus_available': False, 'error': str(e)}

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
    global _MEMORY_CONFIGURED
    logger = logging.getLogger("prediction_model")
    
    # Use lock to prevent multiple threads from configuring GPU simultaneously
    with _gpu_lock:
        logger.info("GPU Memory Configuration Attempt - Already Configured: %s", _MEMORY_CONFIGURED)
        if config:
            logger.info("Config keys: %s", list(config.keys()))
    
        # Default configuration
        DEFAULT_CONFIG = {
            "allow_growth": True,
            "gpu_memory_fraction": 0.95,  # Increased from "auto" to use 95% of GPU memory
            "use_xla": True,
            "mixed_precision": False,  # Default to False for safety
            "visible_gpus": None,
            "directml_enabled": True,
            "use_gpu": True,
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
    
        # Explicitly check for TF_FORCE_FLOAT32 environment variable
        if "TF_FORCE_FLOAT32" in os.environ and os.environ["TF_FORCE_FLOAT32"] == "1":
            config["mixed_precision"] = False
            logger.info("TF_FORCE_FLOAT32 set to 1, forcing float32 precision")
    
        results = {"success": True, "gpus_found": 0}
    
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
    
    
            # Find GPUs
            gpus = tf.config.list_physical_devices("GPU")
            results["gpus_found"] = len(gpus)
    
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s). Configuring memory growth.")
                
                # Always enable memory growth for all GPUs to prevent OOM errors
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for {gpu}")
                    except RuntimeError as e:
                        logger.warning(f"Error setting memory growth for {gpu}: {e}")
                
                # Set visible devices if specified
                if config["visible_gpus"] is not None:
                    visible_gpus = [gpus[i] for i in config["visible_gpus"] if i < len(gpus)]
                    tf.config.set_visible_devices(visible_gpus, 'GPU')
                    logger.info(f"Set visible GPUs to: {config['visible_gpus']}")
                
                # Enable XLA if requested
                if config.get("use_xla", DEFAULT_CONFIG["use_xla"]):
                    tf.config.optimizer.set_jit(True)
                    logger.info("Enabled XLA (Accelerated Linear Algebra)")
                
                # Configure mixed precision if requested
                if config.get("mixed_precision", DEFAULT_CONFIG["mixed_precision"]):
                    policy = tf.keras.mixed_precision.Policy("mixed_float16")
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("Enabled mixed precision (float16)")
            else:
                logger.warning("No GPUs found, using CPU only")
    
            logger.info("GPU configuration complete - Found %d GPUs", len(gpus))
            logger.info("Memory growth enabled: %s", config.get('allow_growth', False))
            
            # Mark as configured
            _MEMORY_CONFIGURED = True

        except ImportError:
            logger.error("TensorFlow not installed")
            results["success"] = False
            results["error"] = "TensorFlow not installed"
        except Exception as e:
            logger.error(f"Unexpected error configuring GPU memory: {e}")
            results["success"] = False
            results["error"] = str(e)
    
        logger.info("Memory configuration results: success=%s", results.get('success', False))
    
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
        
        # Use lock to prevent multiple threads from configuring simultaneously
        with _gpu_lock:
            # TF_FORCE_FLOAT32 takes highest priority
            if "TF_FORCE_FLOAT32" in os.environ and os.environ["TF_FORCE_FLOAT32"] == "1":
                policy = tf.keras.mixed_precision.Policy("float32")
                logger.info("Using float32 precision (forced by TF_FORCE_FLOAT32)")
    
            # If not specified, try to get from config
            if use_mixed_precision is None:
                try:
                    from config.config_loader import get_value
                    use_mixed_precision = get_value("hardware.use_mixed_precision", False)
                except ImportError:
                    use_mixed_precision = False
                    logger.info("Config not available, defaulting to float32 precision")
    
            # Apply the policy and log it clearly
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                logger.info("Using mixed_float16 precision for better performance")
            else:
                policy = tf.keras.mixed_precision.Policy("float32")
                logger.info("Using float32 precision for maximum compatibility")
    
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
        import tensorflow as tf
        
        # Use lock to prevent multiple threads from cleaning simultaneously
        with _gpu_lock:
            # Clear TensorFlow session
            if hasattr(tf.keras.backend, "clear_session"):
                tf.keras.backend.clear_session()
                logger.debug("Cleared TensorFlow session")
    
            # Run garbage collection if requested
            if force_gc:
                gc.collect()
                logger.debug("Ran garbage collection")
    
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

# Add function to make GPU operations thread-safe
def thread_safe_gpu_operation(func):
    """
    Decorator to make GPU operations thread-safe.
    Ensures only one thread accesses GPU resources at a time.
    """
    def wrapper(*args, **kwargs):
        with _gpu_lock:
            return func(*args, **kwargs)
    return wrapper

@thread_safe_gpu_operation
def get_gpu_utilization(device_idx=0):
    """Get current GPU utilization percentage (compute usage, not just memory)"""
    try:
                
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
            if len(gpus) > device_idx:
                return gpus[device_idx].load * 100
            return 0
        except ImportError:
            logger.debug("GPUtil not installed, cannot get detailed GPU utilization")
            
        # If all else fails, return an estimate based on memory usage
        mem_info = get_memory_info(device_idx)
        total = mem_info.get("total_mb", 0)
        used = mem_info.get("used_mb", 0)
        if total > 0:
            return int((used / total) * 100)
        return 0

@thread_safe_gpu_operation
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
        @tf.function(jit_compile=True)
        def stress_function(a, b):
            # Matrix multiplication is compute-intensive
            c = tf.matmul(a, b)
            # Add more operations to increase complexity
            d = tf.nn.relu(c)
            e = tf.matmul(d, a)
            return tf.reduce_sum(e)
        
        # Create a large tensor
        with tf.device("/gpu:0"):
            # Create tensors with appropriate dimensions
            a = tf.random.normal([side_length, side_length], dtype=dtype)
            b = tf.random.normal([side_length, side_length], dtype=dtype)
            
            # Track utilization
            start_time = time.time()
            max_utilization = 0
            total_utilization = 0
            readings = 0
            
            # Get initial temperature if available
            initial_temp = None
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, check=True, universal_newlines=True
                )
                initial_temp = int(result.stdout.strip())
            except:
                pass
            
            # Run computation for specified duration
            while time.time() - start_time < duration_seconds:
                # Run computation
                result = stress_function(a, b)
                
                # Force result evaluation
                _ = result.numpy()
                
                # Check utilization
                try:
                    current_util = get_gpu_utilization(0)
                    max_utilization = max(max_utilization, current_util)
                    total_utilization += current_util
                    readings += 1
                except:
                    pass
                
                # Small sleep to allow other processes
                time.sleep(0.1)
            
            # Get final temperature if available
            final_temp = None
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, check=True, universal_newlines=True
                )
                final_temp = int(result.stdout.strip())
            except:
                pass
            
            # Calculate average utilization
            avg_utilization = total_utilization / max(1, readings)
            
            # Check if fans are likely spinning (utilization > 70% or temperature increased)
            fans_spinning = max_utilization > 70
            if initial_temp is not None and final_temp is not None:
                fans_spinning = fans_spinning or (final_temp - initial_temp) > 5
            
            return {
                "max_utilization": max_utilization,
                "avg_utilization": avg_utilization,
                "initial_temp": initial_temp,
                "final_temp": final_temp,
                "fans_spinning": fans_spinning,
                "duration": duration_seconds
            }
            
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

def apply_optimal_gpu_settings(workload_intensity=None):
    """
    Dynamically apply optimal GPU settings based on workload intensity.
    
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

def get_detailed_gpu_utilization():
    """
    Get current GPU utilization using NVIDIA tools (requires pynvml).
    
    Returns:
        List of dicts with GPU utilization info
    """
    try:
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        utilization = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get device name
            name = pynvml.nvmlDeviceGetName(handle)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            utilization.append({
                'id': i,
                'name': name,
                'gpu_util': util.gpu,  # GPU utilization percentage
                'memory_util': (memory.used / memory.total) * 100,  # Memory utilization percentage
                'memory_total_mb': memory.total / (1024 * 1024),  # Total memory in MB
                'memory_used_mb': memory.used / (1024 * 1024),  # Used memory in MB
                'memory_free_mb': memory.free / (1024 * 1024),  # Free memory in MB
                'temperature': temperature  # Temperature in Celsius
            })
            
        pynvml.nvmlShutdown()
        return utilization
    except ImportError:
        logger.warning("pynvml package not available. Install with: pip install pynvml")
        return []
    except Exception as e:
        logger.error(f"Error getting GPU utilization: {e}")
        return []

def deep_clean_gpu_memory():
    """
    Clean up GPU memory by clearing TensorFlow, PyTorch and CUDA caches.
    
    Returns:
        bool: True if cleanup was successful
    """
    try:
        import gc
        
        # Run Python garbage collector
        gc.collect()
        
        # Clear TensorFlow session
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            logger.info("TensorFlow session cleared")
        except ImportError:
            logger.debug("TensorFlow not available for memory cleanup")
        except Exception as e:
            logger.warning(f"Error clearing TensorFlow session: {e}")
        
        # Clear PyTorch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("PyTorch CUDA cache cleared")
        except ImportError:
            logger.debug("PyTorch not available for memory cleanup")
        except Exception as e:
            logger.warning(f"Error clearing PyTorch cache: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error cleaning GPU memory: {e}")
        return False

def optimize_gpu_performance():
    """
    Apply optimizations to maximize GPU performance including:
    - Setting TensorFlow to use the GPU aggressively
    - Increasing CUDA work queue depth
    - Enabling tensor cores if available
    
    Returns:
        dict: Optimization results
    """
    result = {
        'success': False,
        'optimizations': []
    }
    
    try:
        import tensorflow as tf
        
        # Set CUDA environment variables for performance
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '1'
        os.environ['TF_USE_CUDNN'] = '1'  # Use cuDNN library for deep neural networks
        result['optimizations'].append('environment_variables')
        
        # Enable XLA (Accelerated Linear Algebra)
        try:
            tf.config.optimizer.set_jit(True)
            result['optimizations'].append('xla_jit')
        except:
            pass
            
        # Enable tensor cores for mixed precision if available
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            result['optimizations'].append('mixed_precision')
        except:
            pass
            
        # Configure threading for optimal performance
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # Larger ops like matrix multiply and convolutions benefit from more threads
            tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
            # Setting inter-op to 1 or 2 is often optimal for GPU-heavy workloads
            tf.config.threading.set_inter_op_parallelism_threads(2)
            result['optimizations'].append('threading')
        except:
            pass
            
        result['success'] = True
    except ImportError:
        result['error'] = 'TensorFlow not available'
    except Exception as e:
        result['error'] = str(e)
        
    return result

def get_memory_info():
    """
    Get memory information for the system and GPUs.
    
    Returns:
        dict: Memory information
    """
    memory_info = {
        'system': {},
        'gpus': []
    }
    
    # Get system memory info
    try:
        import psutil
        vm = psutil.virtual_memory()
        memory_info['system'] = {
            'total_gb': vm.total / (1024**3),
            'available_gb': vm.available / (1024**3),
            'used_gb': vm.used / (1024**3),
            'percent_used': vm.percent
        }
    except ImportError:
        memory_info['system']['error'] = 'psutil not available'
    except Exception as e:
        memory_info['system']['error'] = str(e)
    
    # Get GPU memory info
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        # If TensorFlow experimental memory info is available
        for i, gpu in enumerate(gpus):
            try:
                mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                memory_info['gpus'].append({
                    'id': i,
                    'name': gpu.name,
                    'current_bytes': mem_info.get('current', 0),
                    'peak_bytes': mem_info.get('peak', 0),
                    'current_mb': mem_info.get('current', 0) / (1024**2),
                    'peak_mb': mem_info.get('peak', 0) / (1024**2)
                })
            except:
                # Fall back to pynvml if TF experimental API fails
                try:
                    gpu_util = get_gpu_utilization()
                    if gpu_util and i < len(gpu_util):
                        memory_info['gpus'].append(gpu_util[i])
                except:
                    memory_info['gpus'].append({
                        'id': i, 
                        'name': gpu.name,
                        'error': 'Memory info not available'
                    })
    except Exception as e:
        memory_info['gpus_error'] = str(e)
    
    # If no TensorFlow info, try using pynvml directly
    if not memory_info['gpus']:
        try:
            gpu_util = get_gpu_utilization()
            memory_info['gpus'] = gpu_util
        except:
            pass
    
    return memory_info

# Additional function to optimize the model for improved GPU utilization
def optimize_model_for_gpu(model):
    """
    Optimize a TensorFlow model for better GPU utilization.
    
    Args:
        model: A TensorFlow model
        
    Returns:
        The optimized model
    """
    try:
        import tensorflow as tf
        
        # Convert to a function-compiled model which can use XLA and other optimizations
        optimized_model = tf.function(model, jit_compile=True)
        
        # Return the optimized model
        return optimized_model
    except Exception as e:
        logger.warning(f"Could not optimize model for GPU: {e}")
        return model

def ensure_tensor_on_device(tensor, device=None):
    """
    Ensure a tensor is on the specified device.
    Works with both TensorFlow and PyTorch tensors.
    
    Args:
        tensor: Input tensor (TensorFlow or PyTorch)
        device: Target device (if None, uses GPU if available)
        
    Returns:
        Tensor on the specified device
    """
    try:
        # Determine if it's a PyTorch tensor
        is_torch = hasattr(tensor, 'to') and callable(tensor.to)
        
        if is_torch:
            import torch
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return tensor.to(device)
        
        # If it's a TensorFlow tensor, TF handles device placement automatically
        # But we can be explicit:
        import tensorflow as tf
        if device is None:
            device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
            
        with tf.device(device):
            # Create a new tensor on the target device
            return tf.identity(tensor)
        
    except Exception as e:
        logger.warning(f"Error placing tensor on device: {e}")
        return tensor  # Return original tensor if anything fails
