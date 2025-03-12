"""
Centralized GPU memory management to avoid conflicts between different modules.
This file should be imported BEFORE any TensorFlow operations.
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
                # Apply memory growth setting
                if config["allow_growth"]:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Memory growth enabled for all GPUs")

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


# Define HAS_GPUTIL for use above
try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    logger.warning("GPUtil not installed. Detailed GPU monitoring will be limited.")
    HAS_GPUTIL = False
