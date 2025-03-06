"""
Centralized GPU memory management to avoid conflicts between different modules.
This file should be imported BEFORE any TensorFlow operations.
"""

import logging
import os
import platform
from typing import Any, Dict


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track if memory has already been configured
_MEMORY_CONFIGURED = False

# Default configuration
DEFAULT_CONFIG = {
    "allow_growth": True,
    "memory_limit_mb": None,
    "gpu_memory_fraction": "auto",
    "use_xla": True,
    "mixed_precision": True,
    "visible_gpus": None,  # None means use all
    "directml_enabled": True,  # For AMD GPUs
    "use_gpu": True,
}


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

    Args:
        config: Dictionary with configuration options

    Returns:
        Dictionary with configuration results
    """
    global _MEMORY_CONFIGURED

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
        # Merge with defaults
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value

    results = {"success": True, "gpus_found": 0}

    # Disable GPU if requested
    if not config["use_gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("GPU usage disabled by configuration")
        _MEMORY_CONFIGURED = True
        return results

    # Set visible GPUs if specified
    if config["visible_gpus"] is not None:
        visible_gpus = [str(gpu_id) for gpu_id in config["visible_gpus"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
        logger.info(f"Using GPUs: {visible_gpus}")

    # Detect AMD GPU and use DirectML if enabled
    is_amd_gpu = detect_amd_gpu()
    results["is_amd_gpu"] = is_amd_gpu

    if is_amd_gpu and config["directml_enabled"]:
        try:
            logger.info("Detected AMD GPU. Attempting to use DirectML...")
            os.environ["TF_DIRECTML_KERNEL_CACHE"] = "1"
            import tensorflow as tf

            if hasattr(tf, "experimental") and hasattr(tf.experimental, "directml"):
                physical_devices = tf.experimental.directml.devices()
                if physical_devices:
                    logger.info(f"Found {len(physical_devices)} DirectML device(s)")
                    # Use first device
                    tf.experimental.directml.enable_device(0)
                    logger.info(f"Using DirectML device: {physical_devices[0]}")
                    _MEMORY_CONFIGURED = True
                    results["gpus_found"] = len(physical_devices)
                    results["backend"] = "directml"
                    return results
        except ImportError:
            logger.warning("TensorFlow DirectML not installed.")
        except Exception as e:
            logger.error(f"DirectML setup failed: {str(e)}")

    # Standard TensorFlow configuration (for NVIDIA GPUs)
    try:
        # Import TensorFlow
        import tensorflow as tf

        # Find GPUs
        gpus = tf.config.list_physical_devices("GPU")
        results["gpus_found"] = len(gpus)

        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s)")

            try:
                # Apply memory growth setting
                if config["allow_growth"]:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Memory growth enabled for all GPUs")

                # Apply memory limit if specified
                if config["gpu_memory_fraction"] == "auto":
                    logger.info("Using TensorFlow's automatic memory management")
                else:
                    try:
                        memory_fraction = float(config["gpu_memory_fraction"])
                        if 0.0 < memory_fraction <= 1.0:
                            memory_limit_mb = int(memory_fraction * get_total_gpu_memory())
                            for gpu in gpus:
                                tf.config.set_logical_device_configuration(
                                    gpu,
                                    [
                                        tf.config.LogicalDeviceConfiguration(
                                            memory_limit=memory_limit_mb
                                        )
                                    ],
                                )
                            logger.info(
                                f"Memory limit set to {memory_limit_mb}MB for all GPUs"
                            )
                        else:
                            logger.warning(
                                "Invalid gpu_memory_fraction. Using TensorFlow's automatic memory management."
                            )
                    except ValueError:
                        logger.warning(
                            "Invalid gpu_memory_fraction format. Using TensorFlow's automatic memory management."
                        )

                # Enable XLA JIT compilation if requested
                if config["use_xla"]:
                    tf.config.optimizer.set_jit(True)
                    logger.info("XLA JIT compilation enabled")

                # Enable mixed precision for faster computation on GPUs that support it
                if config["mixed_precision"]:
                    try:
                        policy = tf.keras.mixed_precision.Policy("mixed_float16")
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger.info(
                            f"Mixed precision enabled with policy: {policy.name}"
                        )
                    except Exception as e:
                        logger.warning(f"Could not enable mixed precision: {e}")

                results["backend"] = "cuda"
                _MEMORY_CONFIGURED = True

            except RuntimeError as e:
                logger.error(f"Error configuring GPU memory: {e}")
                results["success"] = False
                results["error"] = str(e)

                # Try a fallback configuration with just memory growth
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Fallback to basic memory growth configuration.")
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


def get_memory_info(device_idx=0) -> Dict[str, Any]:
    """
    Get memory information for a specific GPU.

    Args:
        device_idx: GPU index

    Returns:
        Dictionary with memory information
    """
    try:
        import tensorflow as tf

        # Check if memory has been configured
        if not _MEMORY_CONFIGURED:
            logger.warning(
                "GPU memory not configured. Call configure_gpu_memory() first."
            )
            configure_gpu_memory()

        memory_info = {}

        # Try to get memory info from TensorFlow
        try:
            mem = tf.config.experimental.get_memory_info(f"/device:GPU:{device_idx}")
            memory_info["current_bytes"] = mem["current"]
            memory_info["peak_bytes"] = mem.get("peak", 0)
            memory_info["current_mb"] = mem["current"] / (1024 * 1024)
            memory_info["peak_mb"] = memory_info.get("peak_bytes", 0) / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not get memory info from TensorFlow: {e}")
            memory_info["error"] = str(e)

        return memory_info

    except ImportError:
        logger.error("TensorFlow not installed")
        return {"error": "TensorFlow not installed"}
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {"error": str(e)}


def clean_gpu_memory(force_gc=True):
    """
    Clean GPU memory by clearing TensorFlow caches and running garbage collection.

    Args:
        force_gc: Whether to force Python garbage collection

    Returns:
        Dictionary with cleanup results
    """
    import gc

    results = {"success": True}

    try:
        import tensorflow as tf

        # Clear TensorFlow session
        if hasattr(tf.keras.backend, "clear_session"):
            tf.keras.backend.clear_session()
            results["tf_session_cleared"] = True

        # Reset memory stats if available
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu)
            results["memory_stats_reset"] = True
        except:
            results["memory_stats_reset"] = False

    except ImportError:
        results["tf_session_cleared"] = False

    # Run garbage collection if requested
    if force_gc:
        collected = gc.collect()
        results["gc_objects_collected"] = collected

    return results


def get_total_gpu_memory() -> int:
    """
    Get the total memory of the first available GPU in MB.
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_info = tf.config.experimental.get_memory_info(gpus[0], memory_type='VRAM')
            total_memory_bytes = gpu_info['total']
            total_memory_mb = total_memory_bytes / (1024 * 1024)
            return int(total_memory_mb)
        else:
            logger.warning("No GPUs detected.")
            return 0
    except Exception as e:
        logger.error(f"Error getting total GPU memory: {e}")
        return 0


# Initialize GPU configuration with default settings when this module is imported
if not _MEMORY_CONFIGURED:
    configure_gpu_memory()
