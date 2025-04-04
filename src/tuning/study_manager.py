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

# Imports
from datetime import datetime
import os
import sys
import logging
import traceback
import multiprocessing
import threading
import time
import queue
import concurrent.futures
from datetime import datetime


# Fix import path
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import traceback
import optuna
import numpy as np


# Configure logger before using it
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)



# Import GPU resource management tools
try:
    from src.utils.gpu_memory_management import get_memory_info, get_gpu_utilization
    from src.utils.gpu_memory_manager import GPUMemoryManager
    from src.utils.training_optimizer import get_training_optimizer
    HAS_GPU_MANAGEMENT = True
except ImportError:
    logger.warning("GPU memory management modules not available")
    HAS_GPU_MANAGEMENT = False

from src.tuning.progress_helper import TESTED_MODELS_FILE, update_trial_info_in_yaml




# Import cross-platform locking from your existing threadsafe module
try:
    from src.utils.threadsafe import FileLock
except ImportError:
    logger.warning("FileLock not available, will use simple file locking")
    
    # Define a simple fallback FileLock class
    class FileLock:
        def __init__(self, path):
            self.path = path + ".lock"
            self._locked = False
            
        def __enter__(self):
            try:
                with open(self.path, 'x') as f:  # Create lock file exclusively
                    f.write(str(os.getpid()))
                self._locked = True
            except FileExistsError:
                logger.warning(f"Lock file exists: {self.path}")
                raise RuntimeError(f"Resource locked: {self.path}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._locked and os.path.exists(self.path):
                os.remove(self.path)
                self._locked = False




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
        """Model-specific objective function for Optuna."""
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

        # Also update model-specific progress file - Note: 'study' is accessed from trial
        # We need to get the study from the trial to access n_trials
        try:
            from src.tuning.progress_helper import update_model_progress
            # Access study from trial
            study = trial.study
            
            update_model_progress(model_type, {
                "current_trial": trial.number,
                "total_trials": study.user_attrs.get("n_trials", 1000),
                "current_rmse": float(rmse),
                "current_mape": float(mape),
                "directional_accuracy": float(da),
                "completion_percentage": (trial.number / max(1, study.user_attrs.get("n_trials", 1000))) * 100,
                "timestamp": datetime.now().isoformat(),
                "cycle": study.user_attrs.get("cycle", 1)
            })
        except Exception as e:
            logger.warning(f"Error updating model progress: {e}")

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
        
        # Update cycle barrier to match n_startup_trials from config
        from config.config_loader import N_STARTUP_TRIALS
        self.cycle_barrier = threading.Barrier(1)  # Will be resized based on active models
        self.n_startup_trials = N_STARTUP_TRIALS  # Store the n_startup_trials value
        self.current_cycle = 1  # Track current cycle
        self.cycle_lock = threading.RLock()  # Lock for cycle synchronization
        self.resource_lock = threading.RLock()  # Lock for resource allocation
        
        # Status flags
        self.cycle_complete = threading.Event()
        self.reallocation_needed = threading.Event()
        
        # Dynamic trial count management
        self.trials_per_model = 1000  # Default starting value
        self.min_trials_per_cycle = 5  # Minimum trials to run per cycle
        self.max_trials_per_cycle = 10000  # Maximum trials to run per cycle
        self.trial_history = []  # Store trial counts and times for optimization
        self.trial_count_lock = threading.RLock()  # Lock for updating trial counts
        
        # GPU resource tracking
        self.gpu_utilization_history = []
        self.memory_usage_history = []
        self.last_gpu_check = 0
        self.gpu_check_interval = 5  # seconds
        
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
        logger = logging.getLogger("prediction_model")
        logger.info("Creating study for %s - cycle %d", model_type, cycle)
        
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
            # Check if we should force a new study based on cycle
            force_new_study = False
            
            # Get existing study info to check cycle, if available
            try:
                existing_study = optuna.load_study(study_name=study_name, storage=storage_url)
                # Only force new study if ticker/timeframe changed
                # Get existing ticker/timeframe from study attributes
                existing_ticker = existing_study.user_attrs.get("ticker")
                existing_timeframe = existing_study.user_attrs.get("timeframe")
                
                # If ticker or timeframe changed, force new study
                if existing_ticker and existing_timeframe and (
                    existing_ticker != ticker or existing_timeframe != timeframe):
                    logger.info(f"Study found for {existing_ticker}/{existing_timeframe} but requested {ticker}/{timeframe} - forcing new study")
                    force_new_study = True
                else:
                    # Keep using same study even if cycle changed
                    logger.info(f"Study exists for same ticker/timeframe - continuing with existing study")
                    
                    # Rename old study with cycle suffix to preserve it
                    try:
                        # Get cycle from study attributes
                        existing_cycle = existing_study.user_attrs.get("cycle", 0)
                        old_study_name = f"{study_name}_old_cycle{existing_cycle}"
                        existing_study.set_study_attr("renamed_from", study_name)
                        logger.info(f"Renamed old study {study_name} to {old_study_name}")
                    except Exception as rename_error:
                        logger.warning(f"Could not rename old study: {rename_error}")
                
                # Check if we need a new study anyway based on trials
                if n_trials is not None and len(existing_study.trials) >= n_trials:
                    # If study exists and has already reached required trials, still force new
                    logger.info(f"Study already has {len(existing_study.trials)} trials (target: {n_trials}) - forcing new study")
                    force_new_study = True
            except Exception as e:
                # Study doesn't exist yet, which is fine
                logger.debug(f"No existing study found: {e}")
                force_new_study = True
                
            if force_new_study:
                # Create a new study with unique name
                timestamp = int(time.time())
                new_study_name = f"{study_name}_{timestamp}"
                new_storage_url = f"sqlite:///{os.path.join(self.db_dir, f'{new_study_name}.db')}"
                
                logger.info(f"Creating new study with name: {new_study_name}")
                with FileLock(new_storage_url + '.lock'):
                    study = optuna.create_study(
                        study_name=new_study_name,
                        storage=new_storage_url,
                        direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=42)
                    )
                    
                # Update tracking with new name
                study_name = new_study_name
                
                # Update study status tracking
                self.study_status[study_name] = {
                    "model_type": model_type,
                    "status": "created_new",
                    "trials": 0,
                    "best_value": None,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Load existing study if not forcing new
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
        
        # After study creation or loading
        if study and hasattr(study, 'trials'):
            logger.info("Study %s loaded with %d existing trials", study_name, len(study.trials))
        else:
            logger.info("Created new study: %s", study_name)
        
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
        Run multiple Optuna studies concurrently for multiple model types.
        
        Args:
            studies_config: List or Dict of study configurations
            
        Returns:
            Dict with results for each model type
        """
        # Create a thread-safe queue for results
        self.results_queue = queue.Queue()
        
        # Convert list to dict if necessary for uniform processing
        if isinstance(studies_config, list):
            # Convert list to dict using model_type as key
            studies_dict = {}
            for config in studies_config:
                model_type = config.get("model_type")
                if model_type:
                    studies_dict[model_type] = config
            studies_config = studies_dict
        
        # Initialize status tracking for all models
        self.study_status = {
            model_type: {"status": "pending", "error": None} 
            for model_type in studies_config.keys()
        }
        
        # Clear thread local storage for device contexts
        thread_local = threading.local()
        if hasattr(thread_local, 'device_contexts'):
            thread_local.device_contexts = {}
        
        # Determine optimal concurrency based on available resources
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # Check if we have GPU resource management
        if HAS_GPU_MANAGEMENT:
            try:
                gpu_count = len(gpu_memory_manager.logical_gpus) if gpu_memory_manager.logical_gpus else 0
            except:
                gpu_count = 0
        else:
            gpu_count = 0
            
        # Calculate max workers based on CPU cores and model count
        # Ensure at least 1 worker even if studies_config is empty
        max_workers = max(1, min(
            len(studies_config),  # Don't use more threads than models
            max(2, min(cpu_count - 1, 8))  # Use at most 8 threads and leave 1 core free
        ))
        
        logger.info(f"Running {len(studies_config)} model studies with {max_workers} worker threads")
        
        # Create executor with proper naming
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # Launch studies for all model types
            for model_type, config in studies_config.items():
                # Create the worker function with proper thread naming
                def worker_fn(model_type=model_type, config=config):
                    # Set thread name for better debugging
                    threading.current_thread().name = f"ModelWorker-{model_type}"
                    
                    # Configure this thread for GPU access
                    from src.utils.training_optimizer import configure_thread_for_gpu
                    device = configure_thread_for_gpu(model_type)
                    
                    try:
                        # Update status
                        self.study_status[model_type]["status"] = "running"
                        
                        # Create study
                        study = self.create_study(
                            model_type=model_type,
                            ticker=config["ticker"],
                            timeframe=config["timeframe"],
                            range_cat=config["range_cat"],
                            cycle=config.get("cycle", 1),
                            n_trials=config.get("n_trials"),
                            metric_weights=config.get("metric_weights"),
                        )
                        
                        # Create objective function
                        objective = create_model_objective(
                            model_type=model_type,
                            ticker=config["ticker"],
                            timeframe=config["timeframe"],
                            range_cat=config["range_cat"],
                            metric_weights=config.get("metric_weights"),
                        )
                        
                        # Define callbacks
                        callbacks = []
                        
                        # Run optimization
                        self._run_model_optimization(
                            model_type=model_type,
                            study=study,
                            objective=objective,
                            callbacks=callbacks,
                            n_trials=config.get("n_trials"),
                            studies=studies_config
                        )
                        
                        # Update metrics
                        self._update_model_metrics(model_type, study)
                        
                        # Update status
                        self.study_status[model_type]["status"] = "completed"
                        
                        # Get best parameters
                        best_params = self.get_best_params_for_model(model_type)
                        
                        # Create result
                        result = {
                            "best_params": best_params,
                            "best_value": study.best_value if hasattr(study, "best_value") else None,
                            "status": "completed"
                        }
                        
                        # Add result to queue
                        self.results_queue.put((model_type, result))
                        
                        return result
                        
                    except Exception as e:
                        logger.error(f"Error in {model_type} study: {e}", exc_info=True)
                        # Handle error and update status
                        return self.handle_thread_error(model_type, e)
                
                # Submit worker to executor
                futures[model_type] = executor.submit(worker_fn)
            
            # Collect results
            results = {}
            running_models = set(studies_config.keys())
            
            while running_models:
                # Check for completed futures
                completed_models = set()
                
                for model_type in list(running_models):
                    if futures[model_type].done():
                        try:
                            # Get result from future
                            result = futures[model_type].result()
                            results[model_type] = result
                            completed_models.add(model_type)
                            
                            # Update status
                            self.study_status[model_type]["status"] = "completed"
                            
                        except Exception as e:
                            # Handle unexpected errors
                            logger.error(f"Error getting result for {model_type}: {e}")
                            results[model_type] = {"error": str(e), "status": "error"}
                            completed_models.add(model_type)
                            self.study_status[model_type]["status"] = "error"
                            self.study_status[model_type]["error"] = str(e)
                
                # If models completed, update running set and reallocate resources
                if completed_models:
                    running_models -= completed_models
                    # Reallocate resources from finished models to running ones
                    self._reallocate_resources(completed_models, running_models)
                
                # Process any results collected via queue
                while not self.results_queue.empty():
                    try:
                        model_type, result = self.results_queue.get(block=False)
                        results[model_type] = result
                    except queue.Empty:
                        break
                    
                # Small delay to avoid busy waiting
                if running_models:
                    import time
                    time.sleep(0.5)
        
        # Final pass to process any remaining queue items
        while not self.results_queue.empty():
            try:
                model_type, result = self.results_queue.get(block=False)
                results[model_type] = result
            except queue.Empty:
                break
                
        # Clean up GPU memory after all studies
        if HAS_GPU_MANAGEMENT:
            try:
                self.clean_gpu_memory(force_gc=True)
            except Exception as e:
                logger.warning(f"Error cleaning GPU memory: {e}")
                
        # Prepare for next cycle
        self._prepare_for_next_cycle()
        
        return results

    def _run_model_optimization(self, model_type, study, objective, callbacks, n_trials, studies):
        """Run optimization for a single model type."""
        try:
            # Update status tracking
            if model_type in self.study_status:
                self.study_status[model_type]["status"] = "optimizing"
            
            # If we have GPU management, apply resource settings
            resource_settings = self._apply_resource_settings(model_type)
            if resource_settings:
                logger.info(f"Applied resource settings for {model_type}: {resource_settings}")
            
            # Monitor resources before optimization
            self._monitor_resources(f"before_{model_type}")
            
            # Run optimization
            study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
            
            # Monitor resources after optimization
            self._monitor_resources(f"after_{model_type}")
            
            # Update status
            if model_type in self.study_status:
                self.study_status[model_type]["status"] = "optimized"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in {model_type} optimization: {e}", exc_info=True)
            # Update status
            if model_type in self.study_status:
                self.study_status[model_type]["status"] = "error"
                self.study_status[model_type]["error"] = str(e)
            
            # Raise to caller for handling
            raise

    def handle_thread_error(self, model_type, error):
        """Handle errors from worker threads in a centralized way."""
        logger.error(f"Error in thread for {model_type}: {error}")
        # Update status tracking
        if model_type in self.study_status:
            self.study_status[model_type]['status'] = 'error'
            self.study_status[model_type]['error'] = str(error)
        return {"error": str(error), "status": "error"}

    def _reallocate_resources(self, finished_models, running_models):
        """
        Reallocate resources from finished models to still-running models.
        
        Args:
            finished_models: Set of model types that finished
            running_models: Set of model types still running
        """
        if not finished_models or not running_models:
            return False
            
        logger.info(f"Reallocating resources from {finished_models} to {running_models}")
        
        # If we have GPU management, use training optimizer's prioritization
        if HAS_GPU_MANAGEMENT:
            try:
                optimizer = get_training_optimizer()
                if hasattr(optimizer, 'prioritize_slower_models'):
                    optimizer.prioritize_slower_models(finished_models, running_models)
                    return True
            except Exception as e:
                logger.warning(f"Error using training optimizer for reallocation: {e}")
        
        # If we couldn't use the optimizer, use our own simple reallocation
        try:
            # Calculate how many more resources each running model can get
            if running_models:
                boost_factor = 1 + (len(finished_models) / len(running_models) * 0.5)
                logger.info(f"Simple resource boost factor: {boost_factor:.2f}x")
                
                # Future: Implement resource boosting logic
                return True
        except Exception as e:
            logger.warning(f"Error in simple resource reallocation: {e}")
        
        return False
    
    def _apply_resource_settings(self, model_type):
        """Apply resource settings for a specific model type."""
        if not HAS_GPU_MANAGEMENT:
            return None
            
        try:
            # Get the training optimizer
            optimizer = get_training_optimizer()
            
            # Get resource config for this model type
            config = optimizer.get_model_config(model_type)
            
            # Apply memory settings if needed
            if config.get("clean_memory", False):
                self.clean_gpu_memory()
                
            return config
        except Exception as e:
            logger.warning(f"Error applying resource settings for {model_type}: {e}")
            return None

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
            metrics = {}
            
            # Calculate per-model metrics
            if study.trials:
                try:
                    best_trial = study.best_trial
                    metrics = {
                        "best_trial": best_trial.number,
                        "best_value": float(best_trial.value),
                        "rmse": float(best_trial.user_attrs.get("rmse", float("inf"))),
                        "mape": float(best_trial.user_attrs.get("mape", float("inf"))),
                        "directional_accuracy": float(best_trial.user_attrs.get("directional_accuracy", 0)),
                    }
                except (ValueError, AttributeError, RuntimeError):
                    logger.warning(f"No best trial available for {model_type}")
            
            # Count completed trials
            metrics["completed_trials"] = len(study.trials)
            
            # Update individual model progress file
            from src.tuning.progress_helper import update_model_progress, _update_aggregated_progress
            progress_data = {
                "current_trial": len(study.trials),
                "total_trials": study.user_attrs.get("n_trials", 1000),
                "current_rmse": metrics.get("rmse", None),
                "current_mape": metrics.get("mape", None),
                "completion_percentage": (len(study.trials) / max(1, study.user_attrs.get("n_trials", 1000))) * 100,
                "cycle": self.current_cycle,
                "timestamp": datetime.now().isoformat()
            }
            update_model_progress(model_type, progress_data)
            
            # Update the aggregated progress after updating individual model progress
            _update_aggregated_progress()
            
            # Update cycle metrics
            from src.tuning.progress_helper import get_cycle_metrics, update_cycle_metrics
            cycle_metrics = get_cycle_metrics(self.current_cycle) or {}
            
            # Make sure model_metrics dict exists
            if "model_metrics" not in cycle_metrics:
                cycle_metrics["model_metrics"] = {}
                
            # Add metrics for this model
            cycle_metrics["model_metrics"][model_type] = metrics
            
            # Update cycle metrics file
            update_cycle_metrics(cycle_metrics, self.current_cycle)
            
            return metrics
        except Exception as e:
            logger.error(f"Error updating metrics for {model_type}: {e}")
            return {}

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
