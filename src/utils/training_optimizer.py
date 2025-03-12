"""
Optimizer for training resource allocation and visualization.

This module optimizes hardware resource allocation during model training to ensure
efficient utilization of CPU, GPU, and memory. It dynamically distributes resources
based on model complexity and hardware availability, enabling parallel training of
multiple models without resource conflicts.

The module serves as a middleware between the model training pipeline and hardware,
adapting thread counts and memory allocation based on the available resources.
IMPORTANT: This module DOES NOT modify any model hyperparameters (learning rate, batch size, etc.)
and will not affect Optuna's parameter tuning process in any way. It strictly manages
hardware resources like CPU cores and GPU memory allocation.

Key features:
- Auto-detection of available hardware resources
- Model-specific resource profiling and allocation
- Parallel training with optimal resource distribution
- Resource usage monitoring and visualization
- Automatic failover and error handling

This module is primarily used by the training pipeline and is not intended for direct
user interaction. It exposes a singleton instance through the get_training_optimizer() function.
"""

import gc
import json
import logging
import multiprocessing
import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Model type-specific resource requirements (ONLY affecting hardware allocation)
MODEL_RESOURCE_PROFILES = {
    "lstm": {
        "gpu_memory_fraction": 0.3,
        "cpu_weight": 1.0,
        "ram_gb": 2.0,
        "tf_model": True,
    },
    "rnn": {
        "gpu_memory_fraction": 0.25,
        "cpu_weight": 1.0,
        "ram_gb": 1.5,
        "tf_model": True,
    },
    "tft": {
        "gpu_memory_fraction": 0.4,
        "cpu_weight": 1.5,
        "ram_gb": 3.0,
        "tf_model": True,
    },
    "random_forest": {
        "gpu_memory_fraction": 0.0,
        "cpu_weight": 3.0,
        "ram_gb": 3.0,
        "tf_model": False,
    },
    "xgboost": {
        "gpu_memory_fraction": 0.0,
        "cpu_weight": 2.0,
        "ram_gb": 2.0,
        "tf_model": False,
    },
    "tabnet": {
        "gpu_memory_fraction": 0.3,
        "cpu_weight": 1.5,
        "ram_gb": 2.5,
        "tf_model": True,
    },
    "cnn": {
        "gpu_memory_fraction": 0.35,
        "cpu_weight": 1.0,
        "ram_gb": 2.0,
        "tf_model": True,
    },
    "ltc": {
        "gpu_memory_fraction": 0.25,
        "cpu_weight": 1.0,
        "ram_gb": 1.5,
        "tf_model": True,
    },
}


@dataclass
class ModelTask:
    """Represents a model training task with resource requirements."""

    id: str
    model_type: str
    config: Dict[str, Any]
    resources: Dict[str, Any]
    function: Callable
    result: Any = None
    error: str = None
    runtime: float = 0.0
    status: str = "pending"  # pending, running, completed, error


class TrainingOptimizer:
    """
    Optimizes hardware resources for efficient model training without external dependencies.
    Enables all 8 models to run simultaneously with maximum resource utilization.

    IMPORTANT: This optimizer only manages hardware resources (CPU cores, GPU memory,
    RAM allocation) and DOES NOT modify any model parameters that would affect the
    statistical properties or training behavior of the models themselves. All model
    parameters will be controlled entirely by Optuna.
    """

    def __init__(self, config_path=None):
        """Initialize the training optimizer with system information."""
        # System resource detection
        self.cpu_count = multiprocessing.cpu_count()
        self.cpu_affinity_map = self._create_cpu_affinity_map()

        # GPU information
        self.gpus = tf.config.list_physical_devices("GPU")
        self.gpu_count = len(self.gpus)
        self.has_gpu = self.gpu_count > 0

        # Get GPU memory info if available
        self.gpu_memory_gb = self._get_gpu_memory()

        # System memory detection
        self.system_memory_gb = self._get_system_memory()

        # Load config file if provided, otherwise use auto-detection
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = self._auto_configure()

        # Configure TensorFlow based on detected resources
        self._configure_tensorflow()

        # Semaphores for resource management
        self.gpu_semaphore = threading.Semaphore(self.config["max_gpu_models"])
        self.cpu_semaphore = threading.Semaphore(self.config["max_cpu_models"])

        # Resource tracking
        self.used_gpu_memory = 0.0
        self.used_cpu_cores = 0
        self.resource_lock = threading.RLock()

        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}

        # Metrics
        self.performance_metrics = {
            "models_trained": 0,
            "total_training_time": 0.0,
            "average_training_time": 0.0,
            "resource_utilization": 0.0,
            "failures": 0,
        }

        # Initialize cache for models and predictions
        self.model_cache = {}
        self.prediction_cache = {}

        # Add tracking for memory usage
        self.memory_snapshots = []
        self.high_memory_threshold = 0.85  # 85% threshold for high memory warning

        logger.info(
            f"TrainingOptimizer initialized: {self.cpu_count} CPUs, "
            f"{self.gpu_count} GPUs ({self.gpu_memory_gb:.1f} GB), "
            f"{self.system_memory_gb:.1f} GB RAM"
        )

    def _create_cpu_affinity_map(self) -> Dict[int, List[int]]:
        """Create a map of optimal CPU core groupings for affinity settings."""
        affinity_map = {}

        # Simple sequential grouping for now
        # In production, we'd detect cache hierarchy and NUMA nodes
        cores_per_group = max(
            1, self.cpu_count // 8
        )  # Divide available cores among 8 potential models

        for i in range(8):  # Support up to 8 models
            start_core = i * cores_per_group
            end_core = min(start_core + cores_per_group, self.cpu_count)
            affinity_map[i] = list(range(start_core, end_core))

        return affinity_map

    def _get_system_memory(self) -> float:
        """Get total system memory in GB."""
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)  # GB
        except Exception as e:
            logger.warning(f"Could not determine system memory: {e}")
            return 8.0  # Fallback: assume 8GB

    def _get_gpu_memory(self) -> float:
        """Get total GPU memory in GB if available."""
        if not self.has_gpu:
            return 0.0

        try:
            # Try using nvidia-smi through subprocess first
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            memory_values = [int(x) for x in result.stdout.decode().strip().split("\n")]
            return sum(memory_values) / 1024  # Convert MB to GB
        except:
            # Fallback: try TensorFlow experimental API
            try:
                total_memory = 0
                for i, gpu in enumerate(self.gpus):
                    gpu_details = tf.config.experimental.get_memory_info(f"GPU:{i}")
                    if "total" in gpu_details:
                        total_memory += gpu_details["total"]

                return total_memory / (1024**3)  # Bytes to GB
            except:
                # Final fallback: educated guess based on GPU count
                return self.gpu_count * 8.0  # Assume 8GB per GPU

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded training configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using auto-detection.")
            return self._auto_configure()

    def _auto_configure(self) -> Dict:
        """Auto-configure based on available hardware for optimal performance."""
        # Determine max concurrent models for GPU and CPU
        if self.has_gpu:
            # With less than 8GB total GPU memory, be more conservative
            if self.gpu_memory_gb < 8:
                max_gpu_models = 2
            elif self.gpu_memory_gb < 16:
                max_gpu_models = 4
            else:
                max_gpu_models = 8
        else:
            max_gpu_models = 0

        # For CPU models, consider physical cores
        if self.cpu_count <= 4:
            max_cpu_models = 2
        elif self.cpu_count <= 8:
            max_cpu_models = 4
        else:
            max_cpu_models = 8

        # Calculate optimal threads per model
        if self.cpu_count <= 8:
            threads_per_model = 1
            inter_op = 1
            intra_op = 1
        else:
            threads_per_model = max(1, min(4, self.cpu_count // 8))
            inter_op = min(threads_per_model, 2)
            intra_op = max(1, threads_per_model - inter_op)

        return {
            "max_gpu_models": max_gpu_models,
            "max_cpu_models": max_cpu_models,
            "threads_per_model": threads_per_model,
            "inter_op_threads": inter_op,
            "intra_op_threads": intra_op,
            "use_mixed_precision": self.has_gpu,
            "use_xla": self.has_gpu,
            "memory_growth": True,
            "max_parallel_models": min(8, max(1, self.cpu_count // 2)),
            "model_resource_profiles": MODEL_RESOURCE_PROFILES,
            "gpu_memory_headroom_pct": 5,  # Keep 5% GPU memory free
            "cpu_headroom_pct": 10,  # Keep 10% CPU free
        }

    def _configure_tensorflow(self):
        """Apply optimal TensorFlow configuration based on hardware."""
        # Set TensorFlow logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging noise

        # Apply threading configuration
        tf.config.threading.set_inter_op_parallelism_threads(
            self.config["inter_op_threads"]
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            self.config["intra_op_threads"]
        )

        # Configure GPU settings if GPUs are available
        if self.has_gpu:
            # Enable memory growth for all GPUs
            if self.config.get("memory_growth", True):
                for gpu in self.gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for GPU: {gpu}")
                    except Exception as e:
                        logger.warning(f"Error setting memory growth on GPU: {e}")

            # Enable mixed precision if configured
            if self.config.get("use_mixed_precision", True):
                try:
                    policy = tf.keras.mixed_precision.Policy("mixed_float16")
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("Mixed precision enabled for faster training")
                except Exception as e:
                    logger.warning(f"Error enabling mixed precision: {e}")

            # Enable XLA compilation for faster GPU execution
            if self.config.get("use_xla", True):
                try:
                    tf.config.optimizer.set_jit(True)
                    logger.info("XLA compilation enabled for faster training")
                except Exception as e:
                    logger.warning(f"Error enabling XLA compilation: {e}")

    def get_model_settings(
        self, model_type: str, model_complexity: str = "medium"
    ) -> Dict:
        """
        Get hardware resource allocation settings for a specific model type.

        This function STRICTLY configures hardware resources like CPU cores and GPU memory allocation.
        It does NOT set or suggest any model hyperparameters whatsoever.

        Args:
            model_type: Type of model (lstm, cnn, etc.)
            model_complexity: Complexity level (small, medium, large)

        Returns:
            Dictionary with hardware resource settings
        """
        # Get resource profile for this model type
        profile = self.config["model_resource_profiles"].get(
            model_type,
            {
                "gpu_memory_fraction": 0.0,
                "cpu_weight": 1.0,
                "ram_gb": 1.0,
                "tf_model": False,
            },
        )

        # Adjust GPU memory fraction based on complexity
        gpu_fraction = profile["gpu_memory_fraction"]
        if model_complexity == "large":
            gpu_fraction = min(0.8, gpu_fraction * 1.5)
        elif model_complexity == "small":
            gpu_fraction = max(0.05, gpu_fraction * 0.8)

        # Get CPU allocation
        cpu_allocation = self._calculate_cpu_allocation(
            profile["cpu_weight"], model_complexity
        )

        # Create settings dictionary that ONLY affects hardware resources, not model parameters
        settings = {
            "gpu_memory_fraction": gpu_fraction if self.has_gpu else 0.0,
            "cpu_cores": cpu_allocation,
            "ram_gb": profile["ram_gb"],
            "clean_memory": model_complexity == "large",
            "mixed_precision": (
                self.config.get("use_mixed_precision", True) if self.has_gpu else False
            ),
            "threads": {
                "inter_op": self.config["inter_op_threads"],
                "intra_op": self.config["intra_op_threads"],
            },
            "is_tf_model": profile["tf_model"],
        }

        return settings

    def _calculate_cpu_allocation(self, cpu_weight: float, complexity: str) -> int:
        """Calculate optimal CPU core allocation based on weight and complexity."""
        # Adjust weight based on complexity
        if complexity == "large":
            cpu_weight *= 1.5
        elif complexity == "small":
            cpu_weight *= 0.8

        # Calculate raw cores based on weight and available cores
        raw_cores = max(1, int(self.cpu_count * (cpu_weight / 10.0)))

        # Cap at reasonable limits
        max_cores = max(1, self.cpu_count // 2)
        return min(raw_cores, max_cores)

    def create_all_model_tasks(
        self,
        submodel_params_dict: Dict,
        ensemble_weights: Dict,
        training_function: Callable,
    ) -> List[ModelTask]:
        """
        Create training tasks for all model types with non-zero weights.

        This function preserves all model parameters exactly as provided by the caller,
        only adding hardware resource allocation settings that don't affect the model itself.

        Args:
            submodel_params_dict: Dictionary with parameters for each model type
            ensemble_weights: Dictionary with weights for each model type
            training_function: Function to call for model training

        Returns:
            List of ModelTask objects
        """
        tasks = []

        for model_type, weight in ensemble_weights.items():
            if weight <= 0:
                continue  # Skip models with zero weight

            # Get model parameters (preserve exactly as provided)
            model_params = submodel_params_dict.get(model_type, {})

            # Determine complexity level based on parameters
            complexity = self._estimate_model_complexity(model_type, model_params)

            # Get optimized HARDWARE settings for this model
            # These settings won't override any model parameters
            hardware_settings = self.get_model_settings(model_type, complexity)

            # Create task
            task = ModelTask(
                id=f"{model_type}_{int(time.time())}",
                model_type=model_type,
                config={
                    "model_type": model_type,
                    "params": model_params,  # Preserved exactly as provided
                    "hardware_settings": hardware_settings,  # Only affects hardware resources
                },
                resources={
                    "gpu_fraction": hardware_settings["gpu_memory_fraction"],
                    "cpu_cores": hardware_settings["cpu_cores"],
                    "ram_gb": hardware_settings["ram_gb"],
                },
                function=training_function,
            )

            tasks.append(task)

        return tasks

    def _estimate_model_complexity(self, model_type: str, params: Dict) -> str:
        """
        Estimate model complexity based on parameters.

        This is used ONLY for resource allocation and does not modify any model parameters.
        """
        if model_type in ["lstm", "rnn", "tft"]:
            # Neural network complexity based on layer size
            units = params.get("units_per_layer", [64])
            if not isinstance(units, list):
                units = [units]

            total_units = sum(units)

            if total_units > 256:
                return "large"
            elif total_units < 64:
                return "small"
            else:
                return "medium"

        elif model_type in ["random_forest", "xgboost"]:
            # Tree model complexity based on estimators
            n_est = params.get("n_est", 100)

            if n_est > 300:
                return "large"
            elif n_est < 50:
                return "small"
            else:
                return "medium"

        elif model_type == "tabnet":
            # TabNet complexity
            n_d = params.get("n_d", 64)
            n_a = params.get("n_a", 64)

            if n_d + n_a > 200:
                return "large"
            elif n_d + n_a < 100:
                return "small"
            else:
                return "medium"

        elif model_type == "cnn":
            # CNN complexity
            filters = params.get("num_filters", 64)
            layers = params.get("num_conv_layers", 3)

            if filters * layers > 300:
                return "large"
            elif filters * layers < 100:
                return "small"
            else:
                return "medium"

        # Default: medium complexity
        return "medium"

    def optimize_task_groups(self, tasks: List[ModelTask]) -> List[List[ModelTask]]:
        """
        Organize tasks into optimized groups for concurrent execution.
        This ensures all 8 models can run simultaneously if hardware permits.

        Args:
            tasks: List of model tasks to organize

        Returns:
            List of task groups for parallel execution
        """
        # If we have 8 or fewer tasks, try to run them all at once
        if len(tasks) <= self.config["max_parallel_models"]:
            return [tasks]

        # If we have more, need to organize based on resource requirements

        # Separate GPU and CPU-only tasks
        gpu_tasks = [t for t in tasks if t.resources["gpu_fraction"] > 0]
        cpu_tasks = [t for t in tasks if t.resources["gpu_fraction"] == 0]

        # Sort tasks by resource intensity (GPU fraction for GPU tasks, CPU cores for CPU tasks)
        gpu_tasks.sort(key=lambda t: t.resources["gpu_fraction"], reverse=True)
        cpu_tasks.sort(key=lambda t: t.resources["cpu_cores"], reverse=True)

        # Calculate available resources
        available_gpu = 1.0 - (self.config.get("gpu_memory_headroom_pct", 15) / 100.0)
        available_cpu = self.cpu_count * (
            1.0 - (self.config.get("cpu_headroom_pct", 10) / 100.0)
        )

        # Group tasks to maximize resource usage
        groups = []
        current_group = []
        current_gpu = 0.0
        current_cpu = 0.0

        # First attempt to place all GPU tasks
        for task in gpu_tasks:
            # If adding this task would exceed capacity, start a new group
            if (
                current_gpu + task.resources["gpu_fraction"] > available_gpu
                or current_cpu + task.resources["cpu_cores"] > available_cpu
            ):

                # Add any remaining CPU tasks that fit
                for cpu_task in cpu_tasks[:]:
                    if current_cpu + cpu_task.resources["cpu_cores"] <= available_cpu:
                        current_group.append(cpu_task)
                        current_cpu += cpu_task.resources["cpu_cores"]
                        cpu_tasks.remove(cpu_task)

                # Save this group and start a new one
                if current_group:
                    groups.append(current_group)

                current_group = [task]
                current_gpu = task.resources["gpu_fraction"]
                current_cpu = task.resources["cpu_cores"]
            else:
                # Task fits, add to current group
                current_group.append(task)
                current_gpu += task.resources["gpu_fraction"]
                current_cpu += task.resources["cpu_cores"]

        # Add any remaining CPU tasks that fit in the last group
        for cpu_task in cpu_tasks[:]:
            if current_cpu + cpu_task.resources["cpu_cores"] <= available_cpu:
                current_group.append(cpu_task)
                current_cpu += cpu_task.resources["cpu_cores"]
                cpu_tasks.remove(cpu_task)

        # Add the last group if not empty
        if current_group:
            groups.append(current_group)

        # If we have remaining CPU tasks, create CPU-only groups
        if cpu_tasks:
            current_group = []
            current_cpu = 0.0

            for task in cpu_tasks:
                if (
                    current_cpu + task.resources["cpu_cores"] > available_cpu
                    and current_group
                ):
                    groups.append(current_group)
                    current_group = [task]
                    current_cpu = task.resources["cpu_cores"]
                else:
                    current_group.append(task)
                    current_cpu += task.resources["cpu_cores"]

            if current_group:
                groups.append(current_group)

        return groups

    def run_all_models_parallel(
        self,
        model_configs=None,
        training_function=None,
        submodel_params_dict=None,
        ensemble_weights=None,
    ):
        """
        Run all active models in parallel with optimized resource allocation.

        Args:
            model_configs: List of model configurations (new API)
            training_function: Function to call for model training
            submodel_params_dict: Dictionary with parameters for each model type (legacy API)
            ensemble_weights: Dictionary with weights for each model type (legacy API)

        Returns:
            Dictionary with results for each model type
        """
        # Support both new and legacy API
        if model_configs is not None:
            # New API with explicit model configurations
            all_tasks = []
            for config in model_configs:
                model_type = config.get("model_type")
                if not model_type:
                    continue

                task = ModelTask(
                    id=f"{model_type}_{int(time.time())}",
                    model_type=model_type,
                    config=config,
                    resources=config.get("resources", {}),
                    function=training_function,
                )
                all_tasks.append(task)
        else:
            # Legacy API using submodel_params_dict and ensemble_weights
            all_tasks = self.create_all_model_tasks(
                submodel_params_dict, ensemble_weights, training_function
            )

        # Optimize task grouping
        task_groups = self.optimize_task_groups(all_tasks)

        logger.info(
            f"Created {len(task_groups)} training groups for {len(all_tasks)} models"
        )
        for i, group in enumerate(task_groups):
            logger.info(f"Group {i+1}: {[t.model_type for t in group]}")

        # Storage for results
        all_results = {}

        # Report initial status
        self._log_resource_utilization("Initial")

        # Process each group
        for group_idx, group in enumerate(task_groups):
            logger.info(
                f"Running training group {group_idx+1}/{len(task_groups)} with {len(group)} models"
            )

            # Create threads for parallel execution
            threads = []
            results_dict = {}

            for task in group:
                # Create and start thread for this task
                thread = threading.Thread(
                    target=self._run_task_thread, args=(task, results_dict)
                )
                thread.daemon = (
                    True  # Allow program to exit even if threads are running
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete (with timeout)
            timeout_seconds = 3600  # 1 hour timeout
            start_time = time.time()

            for thread in threads:
                remaining_time = max(1, timeout_seconds - (time.time() - start_time))
                thread.join(timeout=remaining_time)

                # Check if we've exceeded timeout
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Timeout reached for group {group_idx+1}")
                    break

            # Check for any threads that didn't complete
            for thread in threads:
                if thread.is_alive():
                    logger.warning(
                        f"Thread didn't complete in time for group {group_idx+1}"
                    )

            # Update all_results with results from this group
            all_results.update(results_dict)

            # Clean up memory after group completes
            self._global_memory_cleanup()

            # Log resource utilization
            self._log_resource_utilization(f"After group {group_idx+1}")

        return all_results

    def _run_task_thread(self, task: ModelTask, results_dict: Dict):
        """Thread worker function to run a single task with resource management."""
        # Acquire appropriate semaphore
        if task.resources["gpu_fraction"] > 0:
            self.gpu_semaphore.acquire()
            using_gpu = True
        else:
            self.cpu_semaphore.acquire()
            using_gpu = False

        # Acquire resource lock for tracking
        with self.resource_lock:
            self.active_tasks[task.id] = task
            task.status = "running"

        # Setup thread environment (CPU/GPU allocation) without modifying model parameters
        self._setup_thread_environment(task)

        # Log start
        logger.info(f"Starting {task.model_type} model training (Task {task.id})")
        start_time = time.time()

        try:
            # Run the task function with original parameters
            result = task.function(task.config)

            # Store results
            task.result = result
            task.status = "completed"
            results_dict[task.model_type] = result

            # Update metrics
            runtime = time.time() - start_time
            task.runtime = runtime

            # Log completion
            logger.info(
                f"Completed {task.model_type} training in {runtime:.2f}s (Task {task.id})"
            )

        except Exception as e:
            # Log error
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Error in {task.model_type} training: {error_msg}")

            # Update task
            task.error = error_msg
            task.status = "error"
            results_dict[task.model_type] = {"error": str(e)}

            # Update metrics
            self.performance_metrics["failures"] += 1

        finally:
            # Clean up individual model resources
            self._cleanup_thread_environment(task)

            # Update task tracking
            with self.resource_lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                self.completed_tasks[task.id] = task

            # Release semaphore
            if using_gpu:
                self.gpu_semaphore.release()
            else:
                self.cpu_semaphore.release()

    def _setup_thread_environment(self, task: ModelTask):
        """
        Setup isolated hardware environment for a task thread.

        This only configures thread and memory allocation without modifying
        any model parameters.
        """
        # Nothing to do for non-TensorFlow models
        if not task.config.get("hardware_settings", {}).get("is_tf_model", False):
            return

        try:
            # For TensorFlow models, set up an isolated thread environment

            # Set thread-specific TensorFlow configuration
            hw_settings = task.config.get("hardware_settings", {})

            if task.resources["gpu_fraction"] > 0 and self.has_gpu:
                # Set up GPU environment
                # We can't truly isolate GPU memory per thread without Ray
                # But we can configure TensorFlow to be more efficient

                # Enable memory growth to avoid allocation all GPU memory
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Set mixed precision if enabled
                if hw_settings.get("mixed_precision", False):
                    policy = tf.keras.mixed_precision.Policy("mixed_float16")
                    tf.keras.mixed_precision.set_global_policy(policy)

            # Set thread-specific CPU config
            if "threads" in hw_settings and all(
                k in hw_settings["threads"] for k in ["inter_op", "intra_op"]
            ):
                tf.config.threading.set_inter_op_parallelism_threads(
                    hw_settings["threads"]["inter_op"]
                )
                tf.config.threading.set_intra_op_parallelism_threads(
                    hw_settings["threads"]["intra_op"]
                )

        except Exception as e:
            logger.warning(f"Error setting up thread environment: {e}")

    def _cleanup_thread_environment(self, task: ModelTask):
        """Clean up resources after a task thread completes."""
        try:
            # For TensorFlow models, clean up session
            if task.config.get("hardware_settings", {}).get("is_tf_model", False):
                tf.keras.backend.clear_session()

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.warning(f"Error cleaning thread environment: {e}")

    def _global_memory_cleanup(self):
        """Perform a more thorough memory cleanup between groups."""
        try:
            # More aggressive TensorFlow cleanup
            tf.keras.backend.clear_session()

            # Force garbage collection
            import gc

            gc.collect()

            # On Linux, try to release memory back to the OS
            if platform.system() == "Linux":
                import ctypes

                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                    logger.debug("Released memory back to the OS")
                except:
                    pass

            logger.debug("Global memory cleanup complete")
        except Exception as e:
            logger.warning(f"Error in global memory cleanup: {e}")

    def _log_resource_utilization(self, label: str):
        """Log current resource utilization."""
        try:
            import psutil

            # Get current process
            process = psutil.Process()

            # Get CPU and memory usage
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            # Get GPU utilization if available
            gpu_info = "N/A"
            if self.has_gpu:
                try:
                    # Try to get GPU info using nvidia-smi
                    import subprocess

                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        stdout=subprocess.PIPE,
                        check=True,
                    )
                    gpu_info = result.stdout.decode().strip()
                except:
                    pass

            logger.info(
                f"Resource utilization ({label}): "
                f"CPU: {cpu_percent:.1f}%, "
                f"Memory: {memory_mb:.1f}MB ({memory_percent:.1f}%), "
                f"GPU: {gpu_info}"
            )

        except Exception as e:
            logger.warning(f"Error logging resource utilization: {e}")

    def get_status_for_dashboard(self) -> Dict:
        """Get current status information for dashboard display."""
        with self.resource_lock:
            return {
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failures": self.performance_metrics["failures"],
                "running_models": [
                    task.model_type for task in self.active_tasks.values()
                ],
                "task_statuses": {
                    model_type: task.status
                    for model_type, task in list(self.active_tasks.items())
                    + list(self.completed_tasks.items())
                },
            }

    def get_model_config(self, model_type):
        """
        Get optimized configuration for a specific model type.

        Args:
            model_type: Type of model (e.g., lstm, cnn, xgboost)

        Returns:
            Dictionary with optimized configuration settings
        """
        # Get resource profile for this model type
        profile = self.config["model_resource_profiles"].get(
            model_type,
            {
                "gpu_memory_fraction": 0.0,
                "cpu_weight": 1.0,
                "ram_gb": 1.0,
                "tf_model": False,
            },
        )

        # Get CPU allocation
        cpu_cores = max(1, int(self.cpu_count * profile["cpu_weight"] / 10))

        # Build configuration
        config = {
            "gpu_memory_fraction": (
                profile["gpu_memory_fraction"] if self.has_gpu else 0.0
            ),
            "cpu_cores": min(
                cpu_cores, self.cpu_count // 2
            ),  # Don't use more than half the cores
            "ram_gb": profile["ram_gb"],
            "mixed_precision": (
                self.config.get("use_mixed_precision", True) if self.has_gpu else False
            ),
            "threads": {
                "inter_op": self.config["inter_op_threads"],
                "intra_op": self.config["intra_op_threads"],
            },
            "is_tf_model": profile["tf_model"],
        }

        return config

    def cleanup_memory(self, level="medium"):
        """
        Unified memory cleanup function that handles TensorFlow sessions,
        garbage collection, and GPU memory clearing.

        Args:
            level: Cleanup intensity level ("light", "medium", "heavy")

        Returns:
            Dict with cleanup results
        """
        import gc
        import time

        start_time = time.time()
        results = {"success": True, "level": level}

        # Log memory before cleanup
        before_memory = self._get_memory_info()
        results["before"] = before_memory

        # Always run garbage collection
        collected = gc.collect()
        results["gc_objects_collected"] = collected

        # Clear TensorFlow session if appropriate
        if level in ["medium", "heavy"]:
            try:
                import tensorflow as tf

                tf.keras.backend.clear_session()
                results["tf_session_cleared"] = True
            except Exception as e:
                logger.warning(f"Error clearing TensorFlow session: {e}")
                results["tf_session_cleared"] = False

        # Additional cleanup for heavy level
        if level == "heavy":
            # Clear caches
            self.clear_model_cache()
            self.clear_prediction_cache()
            results["caches_cleared"] = True

            # Try more aggressive GPU memory cleanup if available
            if self.has_gpu:
                try:
                    self._release_gpu_memory()
                    results["gpu_memory_released"] = True
                except Exception as e:
                    logger.warning(f"Error releasing GPU memory: {e}")
                    results["gpu_memory_released"] = False

        # Log memory after cleanup
        after_memory = self._get_memory_info()
        results["after"] = after_memory

        # Calculate savings
        if "ram_gb" in before_memory and "ram_gb" in after_memory:
            results["ram_saved_gb"] = max(
                0, before_memory["ram_gb"] - after_memory["ram_gb"]
            )

        results["time_taken_ms"] = (time.time() - start_time) * 1000

        logger.debug(
            f"Memory cleanup ({level}): Freed {results.get('ram_saved_gb', 0):.2f} GB RAM, "
            f"collected {collected} objects in {results['time_taken_ms']:.1f}ms"
        )

        return results

    def _get_memory_info(self):
        """Get current memory usage information for both RAM and GPU"""
        memory_info = {}

        # Get RAM usage
        try:
            import psutil

            process = psutil.Process()
            memory_info["ram_gb"] = process.memory_info().rss / 1e9
            memory_info["ram_percent"] = process.memory_percent()

            # System memory
            system = psutil.virtual_memory()
            memory_info["system_total_gb"] = system.total / 1e9
            memory_info["system_available_gb"] = system.available / 1e9
            memory_info["system_percent"] = system.percent
        except Exception as e:
            logger.debug(f"Error getting RAM info: {e}")

        # Get GPU memory info if GPU is available
        if self.has_gpu:
            try:
                import tensorflow as tf

                for i, gpu in enumerate(self.gpus):
                    try:
                        gpu_info = tf.config.experimental.get_memory_info(f"GPU:{i}")
                        memory_info[f"gpu{i}_total_gb"] = gpu_info.get("total", 0) / 1e9
                        memory_info[f"gpu{i}_used_gb"] = (
                            gpu_info.get("current", 0) / 1e9
                        )
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Error getting GPU memory info: {e}")

        return memory_info

    def _release_gpu_memory(self):
        """Attempt to release unused GPU memory back to the system"""
        if not self.has_gpu:
            return False

        try:
            import tensorflow as tf

            # Clear TensorFlow session
            tf.keras.backend.clear_session()

            # Reset memory stats for each GPU
            for i, gpu in enumerate(self.gpus):
                try:
                    tf.config.experimental.reset_memory_stats(f"GPU:{i}")
                except:
                    pass

            return True
        except Exception as e:
            logger.warning(f"Error releasing GPU memory: {e}")
            return False

    def log_memory_usage(self, tag=""):
        """
        Log current memory usage and store in memory snapshots history.

        Args:
            tag: Label to identify this memory snapshot

        Returns:
            boolean: True if memory usage is high (above threshold)
        """
        memory_info = self._get_memory_info()
        memory_info["timestamp"] = time.time()
        memory_info["tag"] = tag

        # Store in memory history
        self.memory_snapshots.append(memory_info)

        # Log current memory usage
        ram_gb = memory_info.get("ram_gb", 0)
        system_percent = memory_info.get("system_percent", 0)

        logger.info(
            f"Memory usage [{tag}]: {ram_gb:.2f} GB RAM, "
            f"System: {system_percent:.1f}%"
        )

        # Check if GPU info is available and log it
        gpu_info = ""
        for i in range(8):  # Check up to 8 GPUs
            if f"gpu{i}_used_gb" in memory_info:
                gpu_info += f"GPU{i}: {memory_info[f'gpu{i}_used_gb']:.2f}GB "

        if gpu_info:
            logger.info(f"GPU Memory [{tag}]: {gpu_info}")

        # Check if memory usage is high
        is_high = system_percent > (self.high_memory_threshold * 100)

        # Trim history if too long
        if len(self.memory_snapshots) > 100:
            self.memory_snapshots = self.memory_snapshots[-100:]

        return is_high

    def clear_model_cache(self):
        """Clear the model cache to free memory"""
        count = len(self.model_cache)
        self.model_cache.clear()
        logger.debug(f"Cleared {count} models from cache")
        return count

    def clear_prediction_cache(self):
        """Clear the prediction cache to free memory"""
        count = len(self.prediction_cache)
        self.prediction_cache.clear()
        logger.debug(f"Cleared {count} predictions from cache")
        return count


# Fixed version that correctly belongs to TrainingOptimizer class as a method
def adjust_resources_for_imbalance(self, model_runtimes):
    """
    Dynamically adjust HARDWARE resources when significant training time imbalance is detected.

    This function only adjusts hardware resource allocation (CPU cores, GPU memory)
    and does NOT modify any model parameters that would affect Optuna's tuning.

    Args:
        model_runtimes: Dictionary mapping model types to their training runtime in seconds
    """
    if not model_runtimes or len(model_runtimes) <= 1:
        return False

    # Find slowest and fastest models
    slowest_model = max(model_runtimes.items(), key=lambda x: x[1])
    fastest_model = min(model_runtimes.items(), key=lambda x: x[1])

    # Check if imbalance is significant (3x or more)
    imbalance_ratio = slowest_model[1] / max(fastest_model[1], 0.1)

    if imbalance_ratio >= 3.0:
        logger.info(
            f"Significant training imbalance detected: {slowest_model[0]} is {imbalance_ratio:.1f}x slower"
        )

        # Get resource profiles
        if slowest_model[0] not in self.config["model_resource_profiles"]:
            return False

        profiles = self.config["model_resource_profiles"]

        # Calculate adjustment factor (capped to avoid overallocation)
        # Using square root to moderate the adjustment (e.g., 9x imbalance → 3x resources)
        adjustment_factor = min(3.0, max(1.2, np.sqrt(imbalance_ratio / 3.0)))

        # Apply adjustments to the slowest model's HARDWARE resources only
        slow_profile = profiles[slowest_model[0]]

        # Record original settings for logging
        original_settings = {
            "gpu": slow_profile.get("gpu_memory_fraction", 0),
            "cpu": slow_profile.get("cpu_weight", 1.0),
        }

        # Adjust GPU allocation if it's a GPU model
        if slow_profile.get("gpu_memory_fraction", 0) > 0:
            # Increase GPU allocation, capped at 80%
            new_gpu = min(0.8, slow_profile["gpu_memory_fraction"] * adjustment_factor)
            slow_profile["gpu_memory_fraction"] = new_gpu
        else:
            # For CPU-only models, increase CPU allocation
            new_cpu = slow_profile["cpu_weight"] * adjustment_factor
            slow_profile["cpu_weight"] = new_cpu

        # Log the adjustments (hardware resources only)
        logger.info(
            f"Adjusted hardware resources for {slowest_model[0]}: "
            f"GPU: {original_settings['gpu']:.2f} → {slow_profile.get('gpu_memory_fraction', 0):.2f}, "
            f"CPU: {original_settings['cpu']:.1f} → {slow_profile.get('cpu_weight', 1.0):.1f}"
        )

        return True

    return False


# Add as a method to the TrainingOptimizer class
TrainingOptimizer.adjust_resources_for_imbalance = adjust_resources_for_imbalance


# Streamlit Dashboard Components
def render_optimizer_dashboard(optimizer):
    """
    Render dashboard components for monitoring the training optimizer.

    This function creates a Streamlit UI showing hardware resource allocation and
    utilization statistics. It displays CPU cores, system memory, GPU resources,
    and execution statistics of completed training tasks.

    Args:
        optimizer: TrainingOptimizer instance containing resource information
    """
    try:
        import streamlit as st

        st.subheader("Hardware Resource Allocation")

        # Hardware specs
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Cores", f"{optimizer.cpu_count}")
            st.metric("System Memory", f"{optimizer.system_memory_gb:.1f} GB")

        with col2:
            st.metric("GPUs", f"{optimizer.gpu_count}")
            if optimizer.has_gpu:
                st.metric("GPU Memory", f"{optimizer.gpu_memory_gb:.1f} GB")

        # Execution stats
        if hasattr(optimizer, "completed_tasks") and optimizer.completed_tasks:
            st.subheader("Execution Statistics")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Models", len(optimizer.completed_tasks))
            with col2:
                avg_time = sum(
                    task.runtime for task in optimizer.completed_tasks.values()
                ) / len(optimizer.completed_tasks)
                st.metric("Avg Training Time", f"{avg_time:.2f}s")
            with col3:
                failures = sum(
                    1
                    for task in optimizer.completed_tasks.values()
                    if task.status == "error"
                )
                st.metric("Failures", failures)

        # Current model status
        status = optimizer.get_status_for_dashboard()
        if status["active_tasks"] > 0:
            st.subheader("Active Models")
            for model_type in status["running_models"]:
                st.info(f"Training {model_type}...")

        # Resource utilization
        try:
            import psutil

            st.subheader("Current Resource Utilization")

            # Get current process
            process = psutil.Process()

            # CPU and Memory
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()

            col1, col2 = st.columns(2)
            with col1:
                st.progress(cpu_percent / 100, text=f"CPU: {cpu_percent:.1f}%")
            with col2:
                st.progress(memory_percent / 100, text=f"Memory: {memory_percent:.1f}%")

            # GPU if available
            if optimizer.has_gpu:
                try:
                    import subprocess

                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        stdout=subprocess.PIPE,
                        check=True,
                    )
                    lines = result.stdout.decode().strip().split("\n")

                    st.subheader("GPU Utilization")
                    for i, line in enumerate(lines):
                        values = line.split(", ")
                        if len(values) >= 3:
                            util_pct = int(values[0])
                            used_mb = int(values[1])
                            total_mb = int(values[2])
                            mem_pct = used_mb / max(1, total_mb) * 100

                            st.write(f"GPU {i}:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.progress(
                                    util_pct / 100, text=f"Compute: {util_pct}%"
                                )
                            with col2:
                                st.progress(
                                    mem_pct / 100,
                                    text=f"Memory: {used_mb}MB/{total_mb}MB",
                                )
                except:
                    st.write("GPU utilization data unavailable")
        except:
            pass
    except ImportError:
        logger.warning("Streamlit not installed. Dashboard rendering disabled.")


# Singleton instance management
_optimizer_instance = None


def get_training_optimizer(config_path=None):
    """Create and return a TrainingOptimizer instance with optimal hardware settings."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = TrainingOptimizer(config_path)
    return _optimizer_instance
