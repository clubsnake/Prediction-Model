# utils.py - Enhanced memory management
"""
Utility functions for validation and walk-forward constraints.

This module provides core utility functions for data validation, memory management,
and walk-forward testing configuration. It serves as a foundation for other modules
by providing commonly used functionality like DataFrame validation and memory cleanup.

Key components:
- Walk-forward validation utilities to ensure proper testing windows
- DataFrame validation to ensure consistent column naming and required data
- Memory management tools to prevent memory leaks and OOM errors
- Adaptive memory cleanup with different intensity levels

This module is imported by most other components in the prediction model pipeline,
particularly the data processing, model training, and evaluation modules.
"""

import gc
import logging
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import psutil

# Import constants from config_loader instead of direct from config
try:
    from config.config_loader import WALK_FORWARD_MAX, WALK_FORWARD_MIN
except ImportError:
    # Fallback defaults if config loading fails
    WALK_FORWARD_MIN = 3
    WALK_FORWARD_MAX = 180
    logging.warning("Failed to load walk-forward constants from config, using defaults")

# Configure logging
logger = logging.getLogger("MemoryManager")


class MemoryCleanupLevel:
    """Enumeration of memory cleanup levels."""

    LIGHT = "light"  # Quick cleanup, minimal impact on performance
    MEDIUM = "medium"  # More thorough cleanup, moderate performance impact
    HEAVY = "heavy"  # Most thorough cleanup, significant performance impact
    CRITICAL = "critical"  # Emergency cleanup when OOM is imminent


def validate_walk_forward(window: int) -> int:
    """
    Ensure the walk-forward window is within configured min/max.
    """
    if window < WALK_FORWARD_MIN:
        return WALK_FORWARD_MIN
    elif window > WALK_FORWARD_MAX:
        return WALK_FORWARD_MAX
    return window


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has required columns and consistent naming.

    :param df: Input pandas DataFrame.
    :return: The validated or None if columns are missing.
    """
    if df is None:
        return None

    if df.index.name in ["Date", "Datetime"]:
        df = df.reset_index()

    date_columns = ["Date", "Datetime", "date", "datetime"]
    for col in date_columns:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            break

    required_columns = ["date", "Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_columns):
        logging.error(f"Missing required columns. Available columns: {df.columns}")
        return None

    return df


# Keep the enhanced memory cleanup function
def clean_memory(
    force_gc=False,
    clear_tf_session=True,
    release_gpu_memory=True,
    verbose=False,
    collect_generations=2,
):
    """Enhanced memory cleaning with better control and reporting."""
    import gc
    import time

    start_time = time.time()
    results = {"success": True}

    # Record starting memory
    try:
        import psutil

        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        results["memory_before_mb"] = memory_before
    except:
        if verbose:
            print("Could not measure starting memory (psutil not available)")

    # Clear TensorFlow session
    if clear_tf_session:
        try:
            import tensorflow as tf

            if hasattr(tf.keras.backend, "clear_session"):
                tf.keras.backend.clear_session()
                if verbose:
                    print("TensorFlow session cleared")
                results["tf_session_cleared"] = True
        except Exception as e:
            if verbose:
                print(f"Error clearing TensorFlow session: {e}")
            results["tf_session_cleared"] = False
            results["success"] = False

    # Release GPU memory
    if release_gpu_memory:
        try:
            import tensorflow as tf

            devices = tf.config.list_physical_devices("GPU")
            if devices:
                for device in devices:
                    try:
                        if hasattr(tf.config.experimental, "reset_memory_stats"):
                            tf.config.experimental.reset_memory_stats(device)
                    except Exception as e:
                        if verbose:
                            print(f"Error releasing memory for {device}: {e}")
                if verbose:
                    print(f"Released memory for {len(devices)} GPU(s)")
                results["gpu_memory_released"] = True
        except Exception as e:
            if verbose:
                print(f"Error accessing GPU devices: {e}")
            results["gpu_memory_released"] = False

    # Run garbage collection
    if force_gc:
        try:
            collected = 0
            for gen in range(min(collect_generations + 1, 3)):
                collected += gc.collect(gen)

            if verbose:
                print(f"Garbage collection freed {collected} objects")
            results["gc_objects_collected"] = collected
        except Exception as e:
            if verbose:
                print(f"Error during garbage collection: {e}")
            results["gc_objects_collected"] = 0

    # Measure memory after cleanup
    try:
        if "psutil" in sys.modules and "memory_before" in locals():
            process = psutil.Process()
            memory_after = process.memory_info().rss / (1024 * 1024)
            results["memory_after_mb"] = memory_after
            results["memory_freed_mb"] = memory_before - memory_after
    except:
        pass

    results["time_taken_ms"] = (time.time() - start_time) * 1000
    return results


class MemoryManager:
    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one memory manager exists."""
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the memory manager (only runs once due to singleton pattern)."""
        if self._initialized:
            return

        self.memory_log = []
        self.cleanup_count = 0
        self.last_gpu_memory = None
        self.last_ram_memory = None
        self.enabled = True
        self.last_cleanup_level = None
        self.last_cleanup_time = 0
        self.log_memory_usage_enabled = False
        self._initialized = True

        # Try to detect available GPU backend
        self.gpu_backend = self._detect_gpu_backend()

        # Initialize memory tracking
        self._update_memory_usage()

    def _detect_gpu_backend(self):
        """Detect which GPU backend is available and active."""
        try:
            import tensorflow as tf

            # Check for DirectML
            if hasattr(tf, "experimental") and hasattr(tf.experimental, "directml"):
                try:
                    devices = tf.experimental.directml.devices()
                    if devices:
                        return "directml"
                except:
                    pass

            # Check for CUDA
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return "cuda"

            # No GPU available
            return "cpu"

        except ImportError:
            return "none"

    def _update_memory_usage(self):
        """Update current memory usage statistics."""
        # Get RAM usage
        try:
            import psutil

            process = psutil.Process(os.getpid())
            self.last_ram_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            self.last_ram_memory = 0

        # Get GPU memory usage based on backend
        if self.gpu_backend == "cuda":
            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    # Get memory info for first GPU
                    gpu_idx = 0
                    try:
                        memory_info = tf.config.experimental.get_memory_info(
                            f"/device:GPU:{gpu_idx}"
                        )
                        self.last_gpu_memory = memory_info["current"] / (
                            1024 * 1024
                        )  # MB
                    except:
                        # Fallback if get_memory_info is not available
                        self.last_gpu_memory = -1
                else:
                    self.last_gpu_memory = 0
            except Exception as e:
                logger.warning(f"Error getting CUDA GPU memory: {e}")
                self.last_gpu_memory = -1
        elif self.gpu_backend == "directml":
            # DirectML doesn't provide direct memory usage API
            self.last_gpu_memory = -1
        else:
            self.last_gpu_memory = 0

        # Log if enabled
        if self.log_memory_usage_enabled:
            self.memory_log.append(
                {
                    "timestamp": time.time(),
                    "ram_mb": self.last_ram_memory,
                    "gpu_mb": self.last_gpu_memory,
                }
            )

    def clean_memory(self, level=MemoryCleanupLevel.MEDIUM, force_gc=True):
        """
        Clean up memory with the specified thoroughness level.

        Args:
            level: Cleanup level from MemoryCleanupLevel enum
            force_gc: Whether to force Python garbage collection

        Returns:
            Dict with cleanup results
        """
        if not self.enabled:
            return {"success": False, "reason": "Memory manager disabled"}

        # Record starting memory
        self._update_memory_usage()
        ram_before = self.last_ram_memory
        gpu_before = self.last_gpu_memory

        start_time = time.time()
        results = {"success": True, "level": level}

        # Avoid too frequent heavy cleanups
        current_time = time.time()
        if level in [MemoryCleanupLevel.HEAVY, MemoryCleanupLevel.CRITICAL]:
            # Only allow heavy cleanup every 30 seconds
            if (
                current_time - self.last_cleanup_time < 30
                and self.last_cleanup_level == level
            ):
                return {"success": False, "reason": f"Too frequent {level} cleanup"}

        self.last_cleanup_time = current_time
        self.last_cleanup_level = level

        # Clear TensorFlow session and reset memory stats based on level
        try:
            import tensorflow as tf

            # For all levels, clear session
            if level != MemoryCleanupLevel.LIGHT:
                tf.keras.backend.clear_session()
                results["tf_session_cleared"] = True

            # For heavier cleanup levels, do more thorough cleanup
            if level in [MemoryCleanupLevel.HEAVY, MemoryCleanupLevel.CRITICAL]:
                # Reset memory stats on GPUs if available
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    for gpu in gpus:
                        try:
                            # Reset operation statistics and clear memory
                            tf.config.experimental.reset_memory_stats(gpu)
                        except Exception as e:
                            logger.debug(f"Could not reset memory stats: {e}")

                # Force TensorFlow to release unused memory
                try:
                    if hasattr(tf.keras.backend, "clear_session"):
                        tf.keras.backend.clear_session()
                    if hasattr(tf.compat.v1, "reset_default_graph"):
                        tf.compat.v1.reset_default_graph()
                except Exception as e:
                    logger.debug(f"Error in TF deep cleanup: {e}")
        except ImportError:
            results["tf_session_cleared"] = False

        # Garbage collection based on level
        if force_gc or level != MemoryCleanupLevel.LIGHT:
            collected = 0
            gc_generations = 0

            if level == MemoryCleanupLevel.LIGHT:
                gc_generations = 1  # Only youngest generation
            elif level == MemoryCleanupLevel.MEDIUM:
                gc_generations = 2  # First and second generations
            else:
                gc_generations = 3  # All generations

            # Run garbage collection on specified generations
            for gen in range(gc_generations):
                collected += gc.collect(gen)

            results["gc_objects_collected"] = collected

        # Release large arrays
        if level in [
            MemoryCleanupLevel.MEDIUM,
            MemoryCleanupLevel.HEAVY,
            MemoryCleanupLevel.CRITICAL,
        ]:
            # Find and delete large numpy arrays in globals
            large_arrays_freed = 0
            if level == MemoryCleanupLevel.CRITICAL:
                # In critical mode, look for large arrays in the global namespace
                for var_name in list(globals().keys()):
                    var = globals()[var_name]
                    if (
                        isinstance(var, np.ndarray) and var.size > 1000000
                    ):  # Arrays larger than ~8MB (assuming float64)
                        del globals()[var_name]
                        large_arrays_freed += 1

            results["large_arrays_freed"] = large_arrays_freed

        # Measure memory after cleanup
        self._update_memory_usage()
        ram_after = self.last_ram_memory
        gpu_after = self.last_gpu_memory

        # Record cleanup stats
        self.cleanup_count += 1
        results.update(
            {
                "ram_before_mb": ram_before,
                "ram_after_mb": ram_after,
                "ram_freed_mb": max(0, ram_before - ram_after),
                "gpu_before_mb": gpu_before,
                "gpu_after_mb": gpu_after,
                "time_taken_ms": (time.time() - start_time) * 1000,
                "cleanup_count": self.cleanup_count,
            }
        )

        return results

    def clean_memory_by_level(self, level_str):
        """
        Clean up memory with the specified thoroughness level.

        Args:
            level_str: String level name ("light", "medium", "heavy", "critical")

        Returns:
            Cleanup results dict
        """
        level_map = {
            "light": MemoryCleanupLevel.LIGHT,
            "small": MemoryCleanupLevel.LIGHT,  # Alias
            "medium": MemoryCleanupLevel.MEDIUM,
            "heavy": MemoryCleanupLevel.HEAVY,
            "large": MemoryCleanupLevel.HEAVY,  # Alias
            "critical": MemoryCleanupLevel.CRITICAL,
        }

        level = level_map.get(level_str.lower(), MemoryCleanupLevel.MEDIUM)
        return self.clean_memory(level=level)

    def optimize_batch_size(
        self, model, sample_input_shape, min_batch=1, max_batch=256, safety_factor=0.8
    ):
        """
        Find the optimal batch size that fits in GPU memory.

        Args:
            model: TensorFlow model to test
            sample_input_shape: Shape of a single input sample (without batch dimension)
            min_batch: Minimum batch size to consider
            max_batch: Maximum batch size to consider
            safety_factor: Factor to reduce the final size by (0-1)

        Returns:
            Optimal batch size
        """
        if not self.enabled or self.gpu_backend == "cpu":
            # On CPU, use a reasonable default
            return min(64, max_batch)

        # Clean memory before testing
        self.clean_memory(level=MemoryCleanupLevel.HEAVY)

        try:
            import tensorflow as tf

            # Binary search for largest batch that fits
            low, high = min_batch, max_batch
            optimal_batch = min_batch

            with tf.device("/device:GPU:0"):  # Assuming single GPU
                while low <= high:
                    mid = (low + high) // 2
                    try:
                        # Create test batch
                        batch_shape = (mid,) + tuple(sample_input_shape)
                        test_input = tf.random.normal(batch_shape)

                        # Try a forward and backward pass
                        with tf.GradientTape() as tape:
                            if isinstance(test_input, list):
                                outputs = model(test_input[0])
                            else:
                                outputs = model(test_input)

                            # Force execution
                            if isinstance(outputs, list):
                                _ = [o.numpy() for o in outputs]
                            else:
                                _ = outputs.numpy()

                        # If successful, try a larger batch
                        optimal_batch = mid
                        low = mid + 1

                        # Clean up
                        del test_input, outputs
                        self.clean_memory(level=MemoryCleanupLevel.LIGHT)

                    except (
                        tf.errors.ResourceExhaustedError,
                        tf.errors.InternalError,
                        tf.errors.UnknownError,
                        tf.errors.OOMError,
                    ) as e:
                        # Batch too large, try smaller
                        logger.info(f"Batch size {mid} too large: {e}")
                        high = mid - 1

                        # More thorough cleanup after OOM
                        self.clean_memory(level=MemoryCleanupLevel.CRITICAL)

            # Apply safety factor
            final_batch = max(min_batch, int(optimal_batch * safety_factor))
            logger.info(
                f"Optimal batch size: {optimal_batch}, "
                + f"Final batch size: {final_batch} (safety factor: {safety_factor})"
            )
            return final_batch

        except Exception as e:
            logger.error(f"Error determining batch size: {e}")
            return min(32, max_batch)  # Conservative fallback

    def start_memory_logging(self, interval_sec=5, max_entries=1000):
        """
        Start logging memory usage at regular intervals.

        Args:
            interval_sec: Interval between log entries in seconds
            max_entries: Maximum number of log entries to keep
        """
        self.log_memory_usage_enabled = True

        # Start a background thread for logging
        import threading

        def logging_thread():
            while self.log_memory_usage_enabled:
                self._update_memory_usage()

                # Trim log if too large
                if len(self.memory_log) > max_entries:
                    self.memory_log = self.memory_log[-max_entries:]

                time.sleep(interval_sec)

        thread = threading.Thread(target=logging_thread, daemon=True)
        thread.start()
        logger.info(f"Memory logging started with {interval_sec}s interval")

    def stop_memory_logging(self):
        """Stop memory usage logging."""
        self.log_memory_usage_enabled = False
        logger.info("Memory logging stopped")

    def get_memory_log(self):
        """
        Get the memory usage log.

        Returns:
            List of memory usage entries
        """
        return self.memory_log

    def plot_memory_usage(self, save_path=None):
        """
        Generate a visualization of memory usage over time.

        This method creates a plot showing the memory consumption pattern,
        which can help identify memory leaks or optimization opportunities.

        Args:
            save_path (str, optional): Path to save the generated plot.
                                       If None, the plot is displayed instead.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        if not self.memory_log:
            logger.warning("No memory data to plot")
            return None

        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            # Convert log to DataFrame
            df = pd.DataFrame(self.memory_log)
            df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot RAM usage
            ax.plot(df["elapsed_min"], df["ram_mb"], label="RAM Usage (MB)")
            ax.plot(df["elapsed_min"], df["gpu_mb"], label="GPU Usage (MB)")
            ax.set_xlabel("Elapsed Time (min)")
            ax.set_ylabel("Memory Usage (MB)")
            ax.legend()
            ax.set_title("Memory Usage Over Time")
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        except ImportError as e:
            logger.error(f"Error importing plotting libraries: {e}")
            return None


def adaptive_memory_clean(level="medium"):
    """
    Clean up memory based on specified level.

    This function provides a simplified interface to the memory management system,
    allowing different components to request memory cleanup with varying intensity
    levels based on their needs.

    Args:
        level: Cleanup level ('small', 'medium', 'large')
            - 'small': Quick garbage collection only
            - 'medium': GC plus TensorFlow session clearing
            - 'large': Aggressive cleanup with memory trimming

    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    try:
        # Always collect Python garbage
        gc.collect()

        if level in ["medium", "large"]:
            # Clear TensorFlow session
            try:
                import tensorflow as tf

                tf.keras.backend.clear_session()
            except ImportError:
                logger.warning("TensorFlow not available, skipping session cleanup")

        if level == "large":
            # Force more aggressive cleanup
            process = psutil.Process(os.getpid())
            logger.info(
                f"Memory usage before cleanup: {process.memory_info().rss / 1e6:.1f} MB"
            )

            try:
                # On Windows, try to force malloc trim
                if hasattr(gc, "malloc_trim"):
                    gc.malloc_trim(0)
            except Exception as e:
                logger.warning(f"Memory trim failed: {e}")

            # Call garbage collector multiple times
            for _ in range(3):
                gc.collect()

            logger.info(
                f"Memory usage after cleanup: {process.memory_info().rss / 1e6:.1f} MB"
            )

        return True
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        return False
