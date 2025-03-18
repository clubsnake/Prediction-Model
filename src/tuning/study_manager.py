# Study Manager for hyperparameter optimization
# Provides centralized management of Optuna studies for different model types

# Imports
import os
import sys
import logging
import traceback
import multiprocessing
import threading
import time
from datetime import datetime

# Configure logger before using it
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Import cross-platform locking from your existing threadsafe module
try:
    from src.utils.threadsafe import FileLock
except ImportError:
    logger.warning("Could not import FileLock from threadsafe, using fallback implementation")
    # Simple fallback implementation if import fails
    class FileLock:
        def __init__(self, filepath, timeout=10.0, retry_delay=0.1):
            self.filepath = filepath
        
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

# Fix import path
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import traceback
import optuna
import numpy as np

# Import GPU resource management tools
try:
    from src.utils.gpu_memory_management import set_performance_mode, get_memory_info, get_gpu_utilization
    from src.utils.gpu_memory_manager import GPUMemoryManager
    from src.utils.training_optimizer import get_training_optimizer
    HAS_GPU_MANAGEMENT = True
except ImportError:
    logger.warning("GPU memory management modules not available")
    HAS_GPU_MANAGEMENT = False

from src.tuning.progress_helper import TESTED_MODELS_FILE, update_trial_info_in_yaml

"""
Hyperparameter optimization study management module.

This module provides functionality to create and manage Optuna studies for
hyperparameter optimization across all model types. It includes:

1. Functions to create objective functions for different model types
2. Study creation and optimization management
3. Result evaluation and processing
4. Parallel execution and resource allocation

The StudyManager class centralizes Optuna study management to ensure consistent
hyperparameter optimization across all model types while maximizing hardware utilization.
"""

# Initialize GPU memory manager if available
gpu_memory_manager = None
if HAS_GPU_MANAGEMENT:
    try:
        gpu_memory_manager = GPUMemoryManager()
        gpu_memory_manager.initialize()
        logger.info("GPU memory manager initialized for study management")
    except Exception as e:
        logger.warning(f"Failed to initialize GPU memory manager: {e}")

def create_model_objective(
    model_type, ticker, timeframe, range_cat, metric_weights=None
):
    """
    Create an Optuna objective function for optimizing a specific model type.

    This function generates an objective function that can be passed to Optuna
    for hyperparameter optimization. The objective function:
    1. Creates model instances with trial-suggested parameters
    2. Trains and evaluates the model using walk-forward validation
    3. Returns a metric value to be minimized (typically RMSE or weighted metrics)

    Args:
        model_type: Type of model to optimize (e.g., 'lstm', 'xgboost')
        ticker: Stock/crypto symbol to use for optimization
        timeframe: Time interval for data (e.g., '1d', '1h')
        range_cat: Range category for prediction horizon
        metric_weights: Optional dictionary of weights for different metrics

    Returns:
        callable: An objective function that can be passed to Optuna's optimize method
    """
    # Default weights if none provided
    if metric_weights is None:
        metric_weights = {
            "rmse": 1.0,  # Root Mean Squared Error
            "mape": 0.5,  # Mean Absolute Percentage Error
            "da": 0.3,  # Directional Accuracy (higher is better, will be inverted)
        }

    # Normalize weights
    total_weight = sum(metric_weights.values())
    normalized_weights = {k: v / total_weight for k, v in metric_weights.items()}

    def objective(trial):
        # Force this model type
        trial.suggest_categorical("model_type", [model_type])

        # Use your existing evaluation function
        results = evaluate_model_with_walkforward(
            trial, ticker, timeframe, range_cat, model_type
        )

        # Extract individual metrics
        rmse = results.get("rmse", float("inf"))
        mape = results.get("mape", float("inf"))

        # Directional accuracy (higher is better, invert for Optuna's minimization)
        da = results.get("directional_accuracy", 0)
        inverted_da = 100 - da  # Assuming DA is in percentage (0-100)

        # Calculate weighted score
        weighted_score = (
            normalized_weights.get("rmse", 0) * rmse
            + normalized_weights.get("mape", 0) * mape
            + normalized_weights.get("da", 0) * inverted_da
        )

        # Store metrics in trial
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("mape", mape)
        trial.set_user_attr("directional_accuracy", da)
        trial.set_user_attr("combined_score", weighted_score)
        trial.set_user_attr("model_type", model_type)

        # Log progress
        logger.info(
            f"Trial {trial.number} ({model_type}): RMSE={rmse:.4f}, MAPE={mape:.2f}%, DA={da:.2f}%"
        )
        logger.info(f"Combined score: {weighted_score:.4f}")

        # Save trial info
        trial_info = {
            "number": trial.number,
            "params": trial.params,
            "model_type": model_type,
            "value": float(weighted_score),
            "metrics": {
                "rmse": float(rmse),
                "mape": float(mape),
                "directional_accuracy": float(da),
                "combined_score": float(weighted_score),
            },
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "timeframe": timeframe,
        }

        update_trial_info_in_yaml(TESTED_MODELS_FILE, trial_info)

        return weighted_score

    return objective


def evaluate_model_with_walkforward(
    trial, ticker, timeframe, range_cat, model_type=None
):
    """
    Evaluate a single model using walk-forward validation.

    Args:
        trial: Optuna trial object
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category
        model_type: Type of model to evaluate

    Returns:
        Dict with evaluation metrics
    """
    # Get active model types from config
    from config.config_loader import ACTIVE_MODEL_TYPES

    # If model_type not specified, get it from trial params
    if model_type is None:
        model_type = trial.params.get("model_type")
        if model_type is None:
            # Default to whatever model type is being suggested
            model_type = trial.suggest_categorical("model_type", ACTIVE_MODEL_TYPES)

    # Store the model type in trial user attributes
    trial.set_user_attr("model_type", model_type)

    # Get model-specific hyperparameters
    params = suggest_model_hyperparameters(trial, model_type)

    try:
        # Get data for this ticker and timeframe
        from src.data.data import fetch_data

        df = fetch_data(ticker, timeframe)

        # Get feature columns
        from config.config_loader import get_active_feature_names

        feature_cols = get_active_feature_names()

        # Import walk-forward validation function
        from src.training.walk_forward import unified_walk_forward

        # Create submodel params dict with just this model
        submodel_params_dict = {model_type: params}

        # Create ensemble weights with just this model having weight 1.0
        ensemble_weights = {
            mt: 1.0 if mt == model_type else 0.0 for mt in ACTIVE_MODEL_TYPES
        }

        # Perform walk-forward validation
        _, metrics = unified_walk_forward(
            df=df,
            feature_cols=feature_cols,
            submodel_params_dict=submodel_params_dict,
            ensemble_weights=ensemble_weights,
            trial=trial,
            update_dashboard=False,
        )

        return metrics
    except Exception as e:
        logger.error(f"Error in walk-forward validation: {e}")
        logger.error(traceback.format_exc())
        return {"rmse": float("inf"), "mape": float("inf")}


def suggest_model_hyperparameters(trial, model_type):
    """
    Suggest hyperparameters for a specific model type.

    Args:
        trial: Optuna trial object
        model_type: Model type to suggest hyperparameters for

    Returns:
        Dict with suggested hyperparameters
    """
    if model_type in ["lstm", "rnn"]:
        return {
            "lr": trial.suggest_float(f"{model_type}_lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float(f"{model_type}_dropout", 0.0, 0.5),
            "lookback": trial.suggest_int(f"{model_type}_lookback", 7, 90),
            "units_per_layer": [
                trial.suggest_categorical(
                    f"{model_type}_units", [32, 64, 128, 256, 512]
                )
            ],
            "loss_function": trial.suggest_categorical(
                f"{model_type}_loss",
                ["mean_squared_error", "mean_absolute_error", "huber_loss"],
            ),
            "epochs": trial.suggest_int(f"{model_type}_epochs", 5, 50),
            "batch_size": trial.suggest_categorical(
                f"{model_type}_batch_size", [16, 32, 64, 128]
            ),
        }
    elif model_type == "random_forest":
        return {
            "n_est": trial.suggest_int("rf_n_est", 50, 500),
            "mdepth": trial.suggest_int("rf_mdepth", 3, 25),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        }
    elif model_type == "xgboost":
        return {
            "n_est": trial.suggest_int("xgb_n_est", 50, 500),
            "lr": trial.suggest_float("xgb_lr", 1e-4, 0.5, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
        }
    elif model_type == "ltc":
        return {
            "lr": trial.suggest_float("ltc_lr", 1e-5, 1e-2, log=True),
            "units": trial.suggest_int("ltc_units", 32, 512),
            "lookback": trial.suggest_int("ltc_lookback", 7, 90),
            "loss_function": trial.suggest_categorical(
                "ltc_loss",
                ["mean_squared_error", "mean_absolute_error", "huber_loss"],
            ),
            "epochs": trial.suggest_int("ltc_epochs", 5, 50),
            "batch_size": trial.suggest_categorical(
                "ltc_batch_size", [16, 32, 64, 128]
            ),
        }
    elif model_type == "tabnet":
        return {
            "n_d": trial.suggest_int("n_d", 8, 256, log=True),
            "n_a": trial.suggest_int("n_a", 8, 256, log=True),
            "n_steps": trial.suggest_int("n_steps", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.5, 3.0),
            "lambda_sparse": trial.suggest_float(
                "lambda_sparse", 1e-7, 1e-1, log=True
            ),
            "optimizer_lr": trial.suggest_float(
                "optimizer_lr", 1e-5, 5e-1, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [128, 256, 512, 1024, 2048, 4096]
            ),
            "virtual_batch_size": trial.suggest_int(
                "virtual_batch_size", 16, 1024, log=True
            ),
            "momentum": trial.suggest_float("momentum", 0.005, 0.5),
            "max_epochs": trial.suggest_int("max_epochs", 50, 500),
            "patience": trial.suggest_int("patience", 5, 50),
            "optimizer_params": {
                "lr": trial.suggest_float("optimizer_lr", 1e-5, 5e-1, log=True)
            },
        }
    elif model_type == "nbeats":
        return {
            "lookback": trial.suggest_int("nbeats_lookback", 7, 180),
            "lr": trial.suggest_float("nbeats_lr", 5e-6, 2e-2, log=True),
            "layer_width": trial.suggest_int("nbeats_layer_width", 32, 1024),
            "num_blocks": trial.suggest_int("nbeats_num_blocks", 1, 10),
            "num_layers": trial.suggest_int("nbeats_num_layers", 1, 12),
            "thetas_dim": trial.suggest_int("nbeats_thetas_dim", 3, 40),
            "include_price_specific_stack": trial.suggest_categorical("nbeats_price_specific", [True, False]),
            "dropout_rate": trial.suggest_float("nbeats_dropout", 0.0, 0.5),
            "use_batch_norm": trial.suggest_categorical("nbeats_batch_norm", [True, False]),
        }
    elif model_type == "cnn":
        return {
            "num_conv_layers": trial.suggest_int("cnn_num_conv_layers", 1, 5),
            "num_filters": trial.suggest_int("cnn_num_filters", 16, 256, log=True),
            "kernel_size": trial.suggest_int("cnn_kernel_size", 2, 7),
            "stride": trial.suggest_int("cnn_stride", 1, 2),
            "dropout_rate": trial.suggest_float("cnn_dropout_rate", 0.0, 0.5),
            "activation": trial.suggest_categorical("cnn_activation", ["relu", "leaky_relu", "elu"]),
            "use_adaptive_pooling": trial.suggest_categorical("cnn_use_adaptive_pooling", [True, False]),
            "fc_layers": [
                trial.suggest_int("cnn_fc_layer_1", 32, 256, log=True),
                trial.suggest_int("cnn_fc_layer_2", 16, 128, log=True),
            ],
            "lookback": trial.suggest_int("cnn_lookback", 7, 90),
            "lr": trial.suggest_float("cnn_lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("cnn_batch_size", [16, 32, 64, 128]),
            "epochs": trial.suggest_int("cnn_epochs", 1, 20),
            "early_stopping_patience": trial.suggest_int("cnn_early_stopping_patience", 3, 10),
        }
    elif model_type == "tft":
        return {
            "lr": trial.suggest_float("tft_lr", 1e-5, 1e-2, log=True),
            "hidden_size": trial.suggest_int("tft_hidden_size", 16, 256),
            "dropout": trial.suggest_float("tft_dropout", 0.0, 0.5),
            "lookback": trial.suggest_int("tft_lookback", 7, 90),
            "num_heads": trial.suggest_int("tft_num_heads", 1, 8),
            "use_batch_norm": trial.suggest_categorical("tft_batch_norm", [True, False]),
            "epochs": trial.suggest_int("tft_epochs", 5, 50),
            "batch_size": trial.suggest_categorical("tft_batch_size", [16, 32, 64, 128]),
        }
    else:
        # Default empty params
        return {}


class StudyManager:
    """
    Centralized manager for Optuna studies.
    Ensures consistent study creation and prevents duplicate studies.
    """

    def __init__(self):
        """Initialize the study manager."""
        self.studies = {}
        self.study_configs = {}
        self.db_dir = os.path.join(project_root, "Data", "DB")
        os.makedirs(self.db_dir, exist_ok=True)
        
        # New diagnostic attributes
        self.active_model_types = set()  # Track which model types are active
        self.study_status = {}  # Track study status information
        
        # Resource tracking
        self.finished_models = set()  # Models that have completed all trials
        self.running_models = set()  # Models still running
        self.cycle_barrier = threading.Barrier(1)  # Will be resized based on active models
        self.current_cycle = 1  # Track current cycle
        self.cycle_lock = threading.RLock()  # Lock for cycle synchronization
        self.resource_lock = threading.RLock()  # Lock for resource allocation
        
        # Status flags
        self.cycle_complete = threading.Event()
        self.reallocation_needed = threading.Event()
        
        # Dynamic trial count management
        self.trials_per_model = 1000  # Default starting value
        self.min_trials_per_cycle = 5  # Minimum trials to run per cycle
        self.max_trials_per_cycle = 5000  # Maximum trials to run per cycle
        self.trial_history = []  # Store trial counts and times for optimization
        self.trial_count_lock = threading.RLock()  # Lock for updating trial counts
        
        # GPU resource tracking
        self.gpu_utilization_history = []
        self.memory_usage_history = []
        self.last_gpu_check = 0
        self.gpu_check_interval = 5  # seconds
        
        # Performance mode management
        self.performance_mode_enabled = False
        
        # Get training optimizer
        try:
            self.training_optimizer = get_training_optimizer()
        except Exception as e:
            logger.warning(f"Failed to initialize training optimizer: {e}")
            self.training_optimizer = None

    def determine_optimal_trial_count(self):
        """
        Dynamically determine the optimal number of trials per model per cycle.
        Allows scaling from 5 to 5000 trials per day based on convergence stage.
        """
        with self.trial_count_lock:
            # Start with default value if no history
            if not self.trial_history:
                # For first cycle, use a high trial count to explore parameter space
                logger.info(f"First cycle: using exploration trial count: {self.max_trials_per_cycle}")
                return self.max_trials_per_cycle
            
            # Get current cycle number
            current_cycle = len(self.trial_history)
            
            # Extract improvements from history
            improvements = []
            best_values = []
            
            for i in range(1, len(self.trial_history)):
                prev_entry = self.trial_history[i-1]
                curr_entry = self.trial_history[i]
                
                prev_best = prev_entry.get('best_value', float('inf'))
                curr_best = curr_entry.get('best_value', float('inf'))
                
                if curr_best != float('inf'):
                    best_values.append(curr_best)
                
                if prev_best != float('inf') and curr_best != float('inf') and prev_best > 0:
                    rel_improvement = (prev_best - curr_best) / prev_best
                    improvements.append(rel_improvement)
                    logger.debug(f"Cycle {i}: improvement {rel_improvement:.4f}")
            
            # Calculate improvement metrics
            if not improvements:
                # If no improvement data yet, keep exploring
                logger.info(f"No improvement data yet, continuing exploration: {self.trials_per_model} trials")
                return self.trials_per_model
                
            # Calculate average and recent improvement rate
            avg_improvement = sum(improvements) / len(improvements)
            recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements
            recent_improvement = sum(recent_improvements) / len(recent_improvements)
            
            logger.info(f"Average improvement: {avg_improvement:.4f}, Recent: {recent_improvement:.4f}")
            
            # Calculate optimal trials based on convergence stage
            if recent_improvement > 0.1:
                # Major improvements (>10%): Aggressive exploration
                optimal_trials = int(self.max_trials_per_cycle * 0.8)
                reason = "major improvements (>10%)"
            elif recent_improvement > 0.05:
                # Significant improvements (5-10%): Strong exploration
                optimal_trials = int(self.max_trials_per_cycle * 0.6) 
                reason = "significant improvements (5-10%)"
            elif recent_improvement > 0.02:
                # Good improvements (2-5%): Moderate exploration
                optimal_trials = int(self.max_trials_per_cycle * 0.4)
                reason = "good improvements (2-5%)"
            elif recent_improvement > 0.01:
                # Minor improvements (1-2%): Focused exploration
                optimal_trials = int(self.max_trials_per_cycle * 0.2)
                reason = "minor improvements (1-2%)"
            elif recent_improvement > 0.005:
                # Minimal improvements (0.5-1%): Fine tuning
                optimal_trials = int(self.max_trials_per_cycle * 0.1)
                reason = "minimal improvements (0.5-1%)"
            else:
                # Near convergence (<0.5%): Precision tuning
                optimal_trials = max(self.min_trials_per_cycle, int(self.max_trials_per_cycle * 0.05))
                reason = "near convergence (<0.5%)"
            
            # Check for recent fluctuations to detect instability
            if len(best_values) >= 4:
                recent_values = best_values[-4:]
                has_fluctuation = any(recent_values[i] < recent_values[i-1] and 
                                      recent_values[i] > recent_values[i+1] 
                                      for i in range(1, len(recent_values)-1))
                
                if has_fluctuation:
                    # Increase trials to help navigate unstable landscape
                    optimal_trials = int(optimal_trials * 1.5)
                    logger.info("Detected fluctuations in best values, increasing trials to stabilize")
            
            # Apply min/max constraints
            optimal_trials = max(self.min_trials_per_cycle, min(self.max_trials_per_cycle, optimal_trials))
            
            logger.info(f"Cycle {current_cycle+1}: {optimal_trials} trials due to {reason}")
            return optimal_trials

    def _monitor_resources(self, tag=""):
        """Monitor GPU and memory resources."""
        if not HAS_GPU_MANAGEMENT:
            return
            
        current_time = time.time()
        # Only check periodically to avoid overhead
        if current_time - self.last_gpu_check < self.gpu_check_interval:
            return
            
        self.last_gpu_check = current_time
        
        try:
            # Get GPU utilization
            gpu_util = get_gpu_utilization(0)
            
            # Get memory info
            mem_info = get_memory_info(0)
            
            # Store in history
            self.gpu_utilization_history.append({
                "timestamp": current_time,
                "utilization": gpu_util,
                "tag": tag
            })
            
            self.memory_usage_history.append({
                "timestamp": current_time,
                "free_mb": mem_info.get("free_mb", 0),
                "total_mb": mem_info.get("total_mb", 0),
                "tag": tag
            })
            
            # Log if high utilization
            if gpu_util > 90:
                logger.info(f"High GPU utilization detected: {gpu_util}% [{tag}]")
                
            # Log if low memory
            free_pct = mem_info.get("free_mb", 1) / max(1, mem_info.get("total_mb", 1)) * 100
            if free_pct < 10:  # less than 10% free
                logger.warning(f"Low GPU memory: {free_pct:.1f}% free [{tag}]")
                
        except Exception as e:
            logger.debug(f"Error monitoring resources: {e}")

    def enable_performance_mode(self):
        """Enable GPU performance mode for intensive operations."""
        if not HAS_GPU_MANAGEMENT or self.performance_mode_enabled:
            return
            
        try:
            set_performance_mode(True)
            if gpu_memory_manager:
                gpu_memory_manager.enable_performance_mode()
            self.performance_mode_enabled = True
            logger.info("GPU performance mode enabled for intensive operations")
        except Exception as e:
            logger.warning(f"Failed to enable performance mode: {e}")
    
    def disable_performance_mode(self):
        """Disable GPU performance mode to save energy."""
        if not HAS_GPU_MANAGEMENT or not self.performance_mode_enabled:
            return
            
        try:
            set_performance_mode(False)
            if gpu_memory_manager:
                gpu_memory_manager.disable_performance_mode()
            self.performance_mode_enabled = False
            logger.info("GPU performance mode disabled")
        except Exception as e:
            logger.warning(f"Failed to disable performance mode: {e}")
    
    def clean_gpu_memory(self, force_gc=True):
        """Clean GPU memory to free up resources."""
        if not HAS_GPU_MANAGEMENT:
            return
            
        try:
            if gpu_memory_manager:
                gpu_memory_manager.clean_memory(force_gc)
                logger.debug("GPU memory cleaned")
            
            # Also get memory info before/after if in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                if self.training_optimizer:
                    self.training_optimizer.log_memory_usage("After GPU memory cleanup")
        except Exception as e:
            logger.warning(f"Failed to clean GPU memory: {e}")

    def create_study(
        self,
        model_type,
        ticker,
        timeframe,
        range_cat,
        cycle=1,
        n_trials=None,
        metric_weights=None,
    ):
        """
        Create (or get existing) study for a specific model.

        Args:
            model_type: Type of model ('lstm', 'rnn', etc.)
            ticker: Ticker symbol
            timeframe: Timeframe
            range_cat: Range category
            cycle: Tuning cycle number
            n_trials: Number of trials
            metric_weights: Optional dict of metric weights

        Returns:
            Optuna study
        """
        # Get all model types from config to validate
        try:
            from config.config_loader import ACTIVE_MODEL_TYPES as CONFIG_MODEL_TYPES
            expected_model_types = set(CONFIG_MODEL_TYPES)
        except ImportError:
            logger.warning("Could not import ACTIVE_MODEL_TYPES from config_loader, using fallback")
            expected_model_types = {"lstm", "rnn", "cnn", "xgboost", "random_forest", "ltc", "nbeats", "tabnet", "tft"}
        
        # Create a unique study name
        study_name = f"{ticker}_{timeframe}_{range_cat}_{model_type}_cycle{cycle}"
        
        # Log study creation attempt
        logger.info(f"Creating/loading study for model type: {model_type} with name: {study_name}")
        
        # Validate model type is expected
        if model_type not in expected_model_types:
            logger.warning(f"Model type '{model_type}' not in configured model types: {expected_model_types}")

        # Check if study already exists
        if study_name in self.studies:
            logger.info(f"Using existing study: {study_name}")
            return self.studies[study_name]

        # Set up storage
        storage_name = os.path.join(self.db_dir, f"{study_name}.db")
        storage_url = f"sqlite:///{storage_name}"
        
        # Track in active model types
        self.active_model_types.add(model_type)
        
        # Add to running models set
        with self.resource_lock:
            self.running_models.add(model_type)

        # Create study
        try:
            # Load existing study if it exists in the database
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            logger.info(f"Loaded existing study from database: {study_name}")
            
            # Log information about existing trials
            num_trials = len(study.trials)
            best_value = getattr(study, "best_value", None)
            logger.info(f"Study '{study_name}' has {num_trials} existing trials. Best value: {best_value}")
            
            # Update study status tracking
            self.study_status[study_name] = {
                "model_type": model_type,
                "status": "loaded",
                "trials": num_trials,
                "best_value": best_value,
                "timestamp": datetime.now().isoformat()
            }
        except (KeyError, Exception) as e:
            # Create a new study if it doesn't exist
            logger.info(f"Study not found ({str(e)}), creating new study: {study_name}")
            try: 
                # Use the FileLock from your threadsafe module
                with FileLock(storage_name):
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=storage_url,
                        direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=42),
                        load_if_exists=True,
                    )
                    logger.info(f"Created new study: {study_name}")
                    
                    # Update study status tracking
                    self.study_status[study_name] = {
                        "model_type": model_type,
                        "status": "created",
                        "trials": 0,
                        "best_value": None,
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                logger.error(f"Error creating study: {str(e)}")
                # Fallback: in-memory storage
                study = optuna.create_study(direction="minimize")
                logger.warning(f"Falling back to in-memory study due to error: {e}")
                
                # Update study status tracking
                self.study_status[study_name] = {
                    "model_type": model_type,
                    "status": "in-memory",
                    "trials": 0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # Store study in dict
        self.studies[study_name] = study

        # Store config
        self.study_configs[study_name] = {
            "model_type": model_type,
            "ticker": ticker,
            "timeframe": timeframe,
            "range_cat": range_cat,
            "cycle": cycle,
            "n_trials": n_trials,
            "metric_weights": metric_weights,
        }

        # Store cycle number in study user attributes
        study.user_attrs["cycle"] = cycle
        study.user_attrs["n_trials"] = n_trials
        
        return study
    
    def log_study_status(self):
        """Log the status of all studies managed by this instance."""
        active_models = list(self.active_model_types)
        total_studies = len(self.studies)
        
        logger.info(f"StudyManager status: {total_studies} studies for {len(active_models)} model types")
        logger.info(f"Active model types: {active_models}")
        
        # Log status of each study
        for study_name, status in self.study_status.items():
            model_type = status.get("model_type", "unknown")
            study_status = status.get("status", "unknown")
            trials = status.get("trials", 0)
            best_value = status.get("best_value", "N/A")
            
            logger.info(f"Study '{study_name}' - Model: {model_type}, Status: {study_status}, Trials: {trials}, Best: {best_value}")
    
    def get_active_model_types(self):
        """Return a set of model types that have active studies."""
        return self.active_model_types

    def run_all_studies(self, studies_config):
        """
        Run all studies completely in parallel, maximizing hardware utilization.
        When faster models finish, their resources are reallocated to slower ones.
        
        Args:
            studies_config: List of study configurations for different model types
            
        Returns:
            Dict with results for each model type
        """
        import concurrent.futures
        from collections import defaultdict
        results = {}
        
        # Enable performance mode for intensive operations
        self.enable_performance_mode()
        
        # Warm up GPU if available
        if HAS_GPU_MANAGEMENT and gpu_memory_manager:
            try:
                gpu_memory_manager.warmup_gpu(intensity=0.7)
                logger.info("GPU warmed up for study execution")
            except Exception as e:
                logger.warning(f"Failed to warm up GPU: {e}")
        
        # Monitor initial resource state
        self._monitor_resources("initial")
        
        # Determine optimal number of trials for this cycle
        if self.training_optimizer:
            try:
                trials_per_model = self.training_optimizer.optimize_trial_count(
                    min_trials=self.min_trials_per_cycle, 
                    max_trials=self.max_trials_per_cycle
                )
                logger.info(f"Using optimizer-suggested trial count: {trials_per_model}")
            except Exception as e:
                logger.warning(f"Error optimizing trial count: {e}")
                trials_per_model = self.min_trials_per_cycle
        else:
            trials_per_model = self.min_trials_per_cycle
        
        # Create a thread pool executor for running studies
        max_workers = min(len(studies_config), multiprocessing.cpu_count())
        
        # Adjust for GPU vs CPU models to avoid overloading GPU
        if HAS_GPU_MANAGEMENT:
            gpu_models = sum(1 for cfg in studies_config if "lstm" in cfg["component_type"] 
                          or "cnn" in cfg["component_type"] or "tft" in cfg["component_type"])
            if gpu_models > 0:
                max_workers = min(max_workers, 3)  # Limit concurrent GPU workers
                
        logger.info(f"Running {len(studies_config)} studies with {max_workers} workers")
        
        # Set up cycle barrier for coordination if needed
        if len(studies_config) > 1:
            self.setup_cycle_barrier(len(studies_config))
        
        # Start all studies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {}
            
            # Submit all study configs to thread pool
            for config in studies_config:
                model_type = config["component_type"]
                n_trials = config.get("n_trials", trials_per_model)
                objective_fn = config["objective_func"]
                callbacks = config.get("callbacks", [])
                
                # Create a future for this model
                future = executor.submit(
                    self._run_model_optimization,
                    model_type,
                    self.create_study(model_type, "", "", "", 1, n_trials),
                    objective_fn,
                    callbacks,
                    n_trials,
                    studies=self.studies
                )
                
                future_to_model[future] = model_type
                self.running_models.add(model_type)
            
            # Record start time for performance tracking
            start_time = time.time()
            completed_models = {}
            model_runtimes = {}
            
            # Process completed studies as they finish
            for i, future in enumerate(concurrent.futures.as_completed(future_to_model)):
                model_type = future_to_model[future]
                
                try:
                    model_result = future.result()
                    results[model_type] = model_result
                    
                    # Record completion
                    completed_models[model_type] = model_result
                    model_runtime = time.time() - start_time
                    model_runtimes[model_type] = model_runtime
                    
                    logger.info(f"Model {model_type} completed in {model_runtime:.2f}s")
                    
                    # Update finished models
                    with self.resource_lock:
                        self.finished_models.add(model_type)
                        self.running_models.remove(model_type)
                    
                    # If we still have running models, try to reallocate resources
                    if self.running_models:
                        self._reallocate_resources(self.finished_models, self.running_models)
                        
                except Exception as e:
                    logger.error(f"Error in study for {model_type}: {e}")
                    results[model_type] = {"error": str(e), "traceback": traceback.format_exc()}
                
                # Monitor resources after each model completes
                self._monitor_resources(f"after_{model_type}")
                
                # Clean memory periodically
                if i % 2 == 0:
                    self.clean_gpu_memory()
        
        # Try to adjust resources if we detected significant runtime imbalance
        if len(model_runtimes) > 1 and self.training_optimizer:
            try:
                self.training_optimizer.adjust_resources_for_imbalance(model_runtimes)
            except Exception as e:
                logger.warning(f"Error adjusting resources: {e}")
        
        # Disable performance mode when done
        self.disable_performance_mode()
        
        # Clean memory after all studies
        self.clean_gpu_memory(True)
        
        return results

    def _run_model_optimization(self, model_type, study, objective, callbacks, n_trials, studies):
        """Run optimization for a single model."""
        logger.info(f"Starting optimization for {model_type} with {n_trials} trials")
        
        # Apply model-specific resource settings
        self._apply_resource_settings(model_type)
        
        try:
            # Set start time for tracking
            study.user_attrs["start_time"] = time.time()
            
            # Run the study
            study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
            
            # Check if all trials completed
            completed_trials = len(study.trials)
            remaining = n_trials - completed_trials
            
            # When a model completes all trials
            if remaining <= 0:
                logger.info(f"Model {model_type} completed all {n_trials} trials")
                
                # Mark this model as finished
                with self.resource_lock:
                    if model_type in self.running_models:
                        self.running_models.remove(model_type)
                        self.finished_models.add(model_type)
                        
                        # This is the key part - trigger resource reallocation
                        from src.utils.training_optimizer import get_training_optimizer
                        optimizer = get_training_optimizer()
                        optimizer.prioritize_slower_models(
                            finished_models={model_type}, 
                            running_models=self.running_models
                        )
            
            # Get best trial
            best_trial = study.best_trial if study.trials else None
            
            # Collect metrics
            metrics = self._update_model_metrics(model_type, study)
            
            # Wait at cycle barrier if we're using synchronized cycles
            try:
                if hasattr(self, 'cycle_barrier') and self.cycle_barrier.parties > 1:
                    logger.info(f"Model {model_type} waiting at cycle barrier")
                    self.cycle_barrier.wait()
                    logger.info(f"All models reached barrier, continuing to next cycle")
            except threading.BrokenBarrierError:
                logger.warning(f"Barrier was broken, some models may have failed")
            except Exception as e:
                logger.error(f"Error at cycle barrier: {e}")
            
            # Return results
            result = {
                "study": study,
                "best_trial": best_trial,
                "n_completed_trials": len(study.trials),
                "metrics": metrics,
                "runtime": time.time() - study.user_attrs.get("start_time", time.time())
            }
            
            # Store updated study in shared dict
            for study_name, s in studies.items():
                if s == study:
                    result["study_name"] = study_name
                    break
                    
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing {model_type}: {e}")
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _reallocate_resources(self, finished_models, running_models):
        """
        Reallocate resources from finished models to still-running models.
        
        Args:
            finished_models: Set of model types that finished
            running_models: Set of model types still running
        """
        if not finished_models or not running_models:
            return
            
        logger.info(f"Reallocating resources from {finished_models} to {running_models}")
        
        # Use training optimizer to handle resource reallocation if available
        if self.training_optimizer:
            try:
                self.training_optimizer.prioritize_slower_models(finished_models, running_models)
            except Exception as e:
                logger.warning(f"Error in resource reallocation: {e}")
        
        # Reset resources to let the system know resources are available
        self.clean_gpu_memory(True)
    
    def _apply_resource_settings(self, model_type):
        """Apply model-specific resource settings."""
        # If we have a training optimizer, get optimal settings
        if self.training_optimizer:
            try:
                # Get optimal settings for this model
                settings = self.training_optimizer.get_model_settings(model_type, "medium")
                
                # Apply GPU settings if applicable
                if settings.get("gpu_memory_fraction", 0) > 0 and HAS_GPU_MANAGEMENT:
                    # If this is a GPU-intensive model, ensure we have performance mode
                    if settings["gpu_memory_fraction"] > 0.2:
                        self.enable_performance_mode()
                    
            except Exception as e:
                logger.warning(f"Error applying resource settings for {model_type}: {e}")

    def get_best_model(self):
        """
        Get the best model across all studies.

        Returns:
            Tuple of (model_type, best_params, best_value)
        """
        best_value = float("inf")
        best_model_type = None
        best_params = None

        # Log all studies being considered
        logger.info(f"Finding best model from {len(self.studies)} studies")
        model_types_considered = set()
        
        for study_name, study in self.studies.items():
            model_type = self.study_configs[study_name]["model_type"]
            model_types_considered.add(model_type)
            
            if study.trials:
                if study.best_value < best_value:
                    best_value = study.best_value
                    best_model_type = model_type
                    best_params = study.best_params
                
                logger.info(f"Study {study_name} for {model_type}: best value = {study.best_value}")
            else:
                logger.warning(f"Study {study_name} for {model_type} has no completed trials")
        
        logger.info(f"Model types considered: {model_types_considered}")
        
        if best_model_type:
            logger.info(f"Best model: {best_model_type} with value: {best_value}")
        else:
            logger.warning("No best model found across studies")
            
        return best_model_type, best_params, best_value

    def get_best_params_for_model(self, model_type):
        """
        Get the best parameters for a specific model.

        Args:
            model_type: Model type to get best parameters for

        Returns:
            Dict with best parameters or None if not found
        """
        model_studies = []
        
        for study_name, study in self.studies.items():
            if (
                self.study_configs[study_name]["model_type"] == model_type
                and study.trials
            ):
                model_studies.append((study_name, study))
                
        if not model_studies:
            logger.warning(f"No studies found for model type {model_type}")
            return None
            
        # Find the study with the best value
        best_study = min(model_studies, key=lambda x: x[1].best_value, default=None)
        
        if best_study:
            study_name, study = best_study
            logger.info(f"Found best parameters for {model_type} from study {study_name}")
            return study.best_params
        
        return None

    def _update_model_metrics(self, model_type, study):
        """
        Update cycle metrics for an individual model once it completes.
        
        Args:
            model_type: Model type
            study: Completed Optuna study
        """
        try:
            from src.tuning.progress_helper import update_cycle_metrics
            
            # Get current cycle if available
            cycle = getattr(study, 'user_attrs', {}).get('cycle', 1)
            
            # Extract metrics from best trial if it exists
            best_metrics = {}
            try:
                if hasattr(study, 'best_trial') and study.best_trial:
                    best_trial = study.best_trial
                    best_metrics = {
                        "rmse": best_trial.user_attrs.get("rmse", float('inf')),
                        "mape": best_trial.user_attrs.get("mape", float('inf')),
                        "directional_accuracy": best_trial.user_attrs.get("directional_accuracy", 0.0),
                        "best_value": best_trial.value
                    }
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not extract best trial metrics for {model_type}: {e}")
            
            # Create model-specific metrics dictionary
            model_metrics = {
                "model_type": model_type,
                "completed_trials": len(study.trials),
                "total_trials": study.user_attrs.get("n_trials", 0),
                "timestamp": datetime.now().isoformat(),
                **best_metrics  # Include best metrics if available
            }
            
            # Update cycle metrics with this model's data
            update_cycle_metrics(model_metrics, cycle_num=cycle)
            
        except Exception as e:
            logger.warning(f"Could not update model metrics for {model_type}: {e}")

    def _prepare_for_next_cycle(self):
        """Prepare for the next tuning cycle"""
        with self.cycle_lock:
            # Increment cycle
            self.current_cycle += 1
            
            # Reset tracking sets
            self.finished_models = set()
            self.running_models = self.active_model_types.copy()
            
            # Reset flags
            self.cycle_complete.clear()
            self.reallocation_needed.clear()
            
            # Clean up resources
            from src.utils.memory_utils import adaptive_memory_clean
            adaptive_memory_clean("heavy")
            
            # Update progress tracking
            try:
                from src.tuning.progress_helper import update_progress_in_yaml
                
                # Update cycle in progress tracking
                progress_data = {
                    "cycle": self.current_cycle,
                    "status": "started",
                    "timestamp": time.time()
                }
                update_progress_in_yaml(progress_data)
                
                logger.info(f"Prepared for cycle {self.current_cycle}")
            except Exception as e:
                logger.error(f"Error updating cycle in progress: {e}")
    
    def setup_cycle_barrier(self, count=None):
        """Set up a synchronization barrier for coordinating models"""
        if count is None:
            # Use number of active model types
            count = len(self.active_model_types)
        
        # Minimum of 1 to avoid errors
        count = max(1, count)
        
        try:
            # Create a new barrier with the specified count
            self.cycle_barrier = threading.Barrier(count)
            logger.info(f"Created cycle barrier with {count} participants")
            return True
        except Exception as e:
            logger.error(f"Error creating cycle barrier: {e}")
            # Make sure we have a valid barrier even on error
            self.cycle_barrier = threading.Barrier(1)
            return False


# Create a singleton instance
study_manager = StudyManager()