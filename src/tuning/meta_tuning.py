"""
Implements an Optuna-based hyperparameter search with ensemble approach and walk-forward.
Logs detailed trial information live to session state and YAML files:
- progress.yaml for current progress,
- best_params.yaml when thresholds are met,
- tested_models.yaml with details for each trial,
- cycle_metrics.yaml with cycle-level summaries.
"""
from datetime import datetime, timedelta
import os
import sys

# Add project root to sys.path for absolute imports
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DATA_DIR, MODELS_DIR

# Define log file paths
LOG_DIR = os.path.join(DATA_DIR, "Models", "Tuning_Logs")
os.makedirs(LOG_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(LOG_DIR, "progress.yaml")
BEST_PARAMS_FILE = os.path.join(DATA_DIR, "Models", "Hyperparameters", "best_params.yaml")
TESTED_MODELS_FILE = os.path.join(LOG_DIR, "tested_models.yaml")
CYCLE_METRICS_FILE = os.path.join(LOG_DIR, "cycle_metrics.yaml")

import logging
import platform
import random
import signal
import threading
import time

# Set optimized environment variables early (before any other imports)
from src.utils.env_setup import setup_tf_environment

# Get mixed precision setting from config first
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from config.config_loader import get_value
    use_mixed_precision = get_value("hardware.use_mixed_precision", False)
except ImportError:
    use_mixed_precision = False

env_vars = setup_tf_environment(memory_growth=True, mixed_precision=use_mixed_precision)

import numpy as np
import optuna
import streamlit as st
import yaml
from src.models.cnn_model import CNNPricePredictor, EnsembleModel 
from src.training.walk_forward import run_walk_forward as walk_forward_ensemble_eval
from src.tuning.progress_helper import update_progress_in_yaml 
from src.utils.threadsafe import (AtomicFileWriter, convert_to_native_types,
                                safe_read_json, safe_read_yaml,
                                safe_write_json, safe_write_yaml)
from config.config_loader import (LOSS_FUNCTIONS, MAPE_THRESHOLD, MODEL_TYPES,
                    N_STARTUP_TRIALS, RMSE_THRESHOLD, START_DATE, TFT_GRID,
                    TICKER, TIMEFRAMES, TUNING_LOOP,
                    TUNING_TRIALS_PER_CYCLE_max, TUNING_TRIALS_PER_CYCLE_min,
                    INTERVAL, get_active_feature_names, get_horizon_for_category,
                    ACTIVE_MODEL_TYPES)

# Import pruning settings from config
from config.config_loader import (PRUNING_ABSOLUTE_MAPE_FACTOR, PRUNING_ABSOLUTE_RMSE_FACTOR,
                    PRUNING_ENABLED, PRUNING_MEDIAN_FACTOR, PRUNING_MIN_TRIALS)

# Configure logger properly
logger = logging.getLogger(__name__)

# Session state initialization
if "trial_logs" not in st.session_state:
    st.session_state["trial_logs"] = []

# Global stop control via event
stop_event = threading.Event()

def set_stop_requested(val: bool):
    """Set the stop event flag."""
    if val:
        stop_event.set()
        print("Stop requested - flag set")
    else:
        stop_event.clear()

def is_stop_requested():
    """Check if stop has been requested."""
    return stop_event.is_set()

# Signal handler for non-Windows platforms
if sys.platform != "win32":
    def signal_handler(sig, frame):
        print("\nManual stop requested. Exiting tuning loop.")
        set_stop_requested(True)
    signal.signal(signal.SIGINT, signal_handler)

# Data directories and file paths
DB_DIR = os.path.join(DATA_DIR, "DB")

os.makedirs(DB_DIR, exist_ok=True)

def convert_to_builtin_type(obj):
    """Convert numpy types to Python built-in types for serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    else:
        return obj

class AdaptiveEnsembleWeighting:
    """Class for managing ensemble weights based on model performance."""
    def __init__(self, model_types, initial_weights, memory_factor, min_weight, exploration_factor):
        self.weights = initial_weights.copy()
        self.memory_factor = memory_factor
        self.min_weight = min_weight
        self.exploration_factor = exploration_factor
        self.performance_history = {mt: [] for mt in model_types}

    def get_weights(self):
        """Return current weights."""
        return self.weights

    def update_performance(self, mtype, mse):
        """Update performance history for a specific model type."""
        if mtype not in self.weights:
            return  # Ignore unknown model types
        
        # Add new performance
        self.performance_history[mtype].append(mse)
        
        # Limit history length
        max_history = 10  # Keep last 10 entries
        if len(self.performance_history[mtype]) > max_history:
            self.performance_history[mtype] = self.performance_history[mtype][-max_history:]

    def update_weights(self):
        """Recompute weights based on recent performance."""
        # Calculate average performance for each model
        avg_perf = {}
        for mtype, history in self.performance_history.items():
            if history:
                # Lower MSE is better, so use inverse for weighting
                avg_perf[mtype] = 1.0 / (np.mean(history) + 1e-10)  # Avoid division by zero
            else:
                avg_perf[mtype] = 0.0
                
        # Apply memory factor to blend with previous weights
        total_weight = sum(avg_perf.values())
        
        if (total_weight > 0):
            for mtype in self.weights:
                # New weight is a blend of old weight and new performance
                new_weight = ((1.0 - self.memory_factor) * (avg_perf[mtype] / total_weight) + 
                            self.memory_factor * self.weights[mtype])
                
                # Apply minimum weight constraint
                self.weights[mtype] = max(self.min_weight, new_weight)
                
            # Add exploration factor for diversity
            if self.exploration_factor > 0:
                for mtype in self.weights:
                    self.weights[mtype] += np.random.normal(0, self.exploration_factor)
                    self.weights[mtype] = max(0, self.weights[mtype])  # Ensure non-negative
            
            # Normalize weights to sum to 1
            weight_sum = sum(self.weights.values())
            if weight_sum > 0:
                for mtype in self.weights:
                    self.weights[mtype] /= weight_sum

    def get_weighted_prediction(self, predictions):
        """Combine predictions using the current weights."""
        valid = [(self.weights[mtype], np.array(preds)) for mtype, preds in predictions.items() if preds is not None]
        if valid:
            total = sum(weight for weight, _ in valid)
            if total == 0:
                return None
            weighted = sum(weight * preds for weight, preds in valid) / total
            return weighted
        return None

def reset_progress():
    """Reset progress tracking file."""
    import yaml
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        yaml.safe_dump({}, f)

def prune_old_cycles(filename=CYCLE_METRICS_FILE, max_cycles=50):
    """
    Prune old cycle metrics from the YAML file if more than max_cycles exist.
    Keeps only the most recent max_cycles cycles.
    """
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                cycles = yaml.safe_load(f) or []
            if len(cycles) > max_cycles:
                cycles = cycles[-max_cycles:]
                with open(filename, "w") as f:
                    yaml.safe_dump(convert_to_builtin_type(cycles), f)
                print(f"Pruned old cycles. Now storing {len(cycles)} cycles.")
    except Exception as e:
        print(f"Error pruning old cycles: {e}")

class LazyImportManager:
    """
    Handles lazy imports to avoid circular dependencies.
    All functions below are invoked on demand and import only when necessary.
    """
    @staticmethod
    def get_model_builder():
        # Returns the model-building function.
        from src.models.model import build_model_by_type
        return build_model_by_type

    @staticmethod
    def get_data_fetcher():
        from src.data.data import fetch_data
        return fetch_data

    @staticmethod
    def get_feature_engineering():
        from src.features.features import feature_engineering_with_params
        return feature_engineering_with_params

    @staticmethod
    def get_scaling_function():
        from src.data.preprocessing import scale_data
        return scale_data

    @staticmethod
    def get_sequence_function():
        from src.data.preprocessing import create_sequences
        return create_sequences

    @staticmethod
    def get_optuna_sampler():
        import optuna
        return optuna.samplers.TPESampler

    @staticmethod
    def get_evaluation_function():
        from src.models.model import evaluate_predictions
        return evaluate_predictions

    @staticmethod
    def get_cnn_model():
        from src.models.cnn_model import CNNPricePredictor
        return CNNPricePredictor

# ...existing imports...
from src.utils.training_optimizer import get_training_optimizer

# Initialize training optimizer
training_optimizer = get_training_optimizer()

def get_model_prediction(mtype, submodel_params, X_train, y_train, X_test, horizon, unified_lookback, feature_cols):
    """
    Get predictions from a model without circular imports.
    Uses lazy imports to avoid circular dependencies.
    """
    # Get optimal configuration for this model type
    model_config = training_optimizer.get_model_config(mtype)
    
    # Use optimized batch size if not specified
    if mtype in ["lstm", "rnn", "tft", "cnn", "ltc"]:
        if "batch_size" not in submodel_params[mtype]:
            submodel_params[mtype]["batch_size"] = model_config["batch_size"]
    
    if (mtype in ["lstm", "rnn"]):
        arch_params = {
            "units_per_layer": submodel_params[mtype].get("units_per_layer", [64, 32]),
            "use_batch_norm": submodel_params[mtype].get("use_batch_norm", False),
            "l2_reg": submodel_params[mtype].get("l2_reg", 0.0),
            "use_attention": submodel_params[mtype].get("use_attention", True),
            "attention_type": submodel_params[mtype].get("attention_type", "dot"),
            "attention_size": submodel_params[mtype].get("attention_size", 64),
            "attention_heads": submodel_params[mtype].get("attention_heads", 1),
            "attention_dropout": submodel_params[mtype].get("attention_dropout", 0.0)
        }
        
        # Lazy import the model builder
        build_model_by_type = LazyImportManager.get_model_builder()
        
        model = build_model_by_type(
            model_type=mtype,
            num_features=len(feature_cols),
            horizon=horizon,
            learning_rate=submodel_params[mtype]["lr"],
            dropout_rate=submodel_params[mtype]["dropout"],
            loss_function=submodel_params[mtype]["loss_function"],
            lookback=unified_lookback,
            architecture_params=arch_params
        )
        epochs = submodel_params[mtype].get("epochs", 1)
        batch_size = submodel_params[mtype].get("batch_size", 32)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        preds = model.predict(X_test)
    elif mtype == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=submodel_params[mtype]["n_est"],
            max_depth=submodel_params[mtype]["mdepth"],
            min_samples_split=submodel_params[mtype].get("min_samples_split", 2),
            min_samples_leaf=submodel_params[mtype].get("min_samples_leaf", 1),
            random_state=42
        )
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        y_tr_flat = y_train[:, 0]
        rf.fit(X_tr_flat, y_tr_flat)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)
        preds_1d = rf.predict(X_te_flat)
        preds = np.tile(preds_1d.reshape(-1,1), (1, horizon))
    elif mtype == "xgboost":
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=submodel_params[mtype]["n_est"],
            learning_rate=submodel_params[mtype]["lr"],
            max_depth=submodel_params[mtype].get("max_depth", 6),
            subsample=submodel_params[mtype].get("subsample", 1.0),
            colsample_bytree=submodel_params[mtype].get("colsample_bytree", 1.0),
            random_state=42
        )
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        y_tr_flat = y_train[:, 0]
        xgb_model.fit(X_tr_flat, y_tr_flat)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)
        preds_1d = xgb_model.predict(X_te_flat)
        preds = np.tile(preds_1d.reshape(-1,1), (1, horizon))
    elif mtype == "ltc":
        # Import ltc model builder
        from src.models.ltc_model import build_ltc_model
        
        model = build_ltc_model(
            num_features=len(feature_cols),
            horizon=horizon,
            learning_rate=submodel_params[mtype]["lr"],
            loss_function=submodel_params[mtype]["loss_function"],
            lookback=unified_lookback,
            units=submodel_params[mtype]["units"]
        )
        
        epochs = submodel_params[mtype].get("epochs", 1)
        batch_size = submodel_params[mtype].get("batch_size", 32)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        preds = model.predict(X_test)
    elif mtype == "cnn":
        # Get CNN parameters
        cnn_params = submodel_params[mtype]
        
        # Create model
        cnn_model = CNNPricePredictor(
            input_dim=len(feature_cols),
            output_dim=horizon,
            num_conv_layers=cnn_params.get("num_conv_layers", 3),
            num_filters=cnn_params.get("num_filters", 64),
            kernel_size=cnn_params.get("kernel_size", 3),
            stride=cnn_params.get("stride", 1),
            dropout_rate=cnn_params.get("dropout_rate", 0.2),
            activation=cnn_params.get("activation", "relu"),
            use_adaptive_pooling=cnn_params.get("use_adaptive_pooling", True),
            fc_layers=cnn_params.get("fc_layers", [128, 64]),
            lookback=cnn_params.get("lookback", unified_lookback),
            learning_rate=cnn_params.get("lr", 0.001),
            batch_size=cnn_params.get("batch_size", 32),
            epochs=cnn_params.get("epochs", 10),
            early_stopping_patience=cnn_params.get("early_stopping_patience", 5),
            verbose=0
        )
        
        # Fit and predict
        cnn_model.fit(X_train, y_train)
        preds = cnn_model.predict(X_test)
        return preds
    
    return preds

class OptunaSuggester:
    """Simple wrapper for Optuna parameter suggestions."""
    def __init__(self, trial):
        self.trial = trial

    def suggest(self, param_name):
        from config.config_loader import HYPERPARAMETER_REGISTRY
        if param_name not in HYPERPARAMETER_REGISTRY:
            raise ValueError(f"Parameter {param_name} not registered")
        param_config = HYPERPARAMETER_REGISTRY[param_name]
        param_type = param_config["type"]
        param_range = param_config["range"]
        param_default = param_config["default"]
        
        # Add support for additional configuration options
        use_log = param_config.get("log_scale", False)
        step = param_config.get("step", None)
        
        if param_type == "int":
            if param_range:
                return self.trial.suggest_int(param_name, param_range[0], param_range[1], 
                                             step=step if step else 1)
            else:
                return self.trial.suggest_int(param_name, max(1, param_default // 2), 
                                             param_default * 2)
        elif param_type == "float":
            if param_range:
                return self.trial.suggest_float(param_name, param_range[0], param_range[1],
                                              log=use_log)
            else:
                # Auto-determine log scale for small values
                auto_log = use_log or param_default < 0.01
                return self.trial.suggest_float(param_name, 
                                              max(1e-6, param_default / 10 if auto_log else param_default / 2), 
                                              param_default * 10 if auto_log else param_default * 2,
                                              log=auto_log)
        elif param_type == "categorical":
            if param_range:
                return self.trial.suggest_categorical(param_name, param_range)
            else:
                return param_default
        elif param_type == "bool":
            return self.trial.suggest_categorical(param_name, [True, False])
        
        # Handle uniform and other distributions if specified
        if "distribution" in param_config:
            dist_type = param_config["distribution"]["type"]
            if dist_type == "uniform":
                low = param_config["distribution"]["low"]
                high = param_config["distribution"]["high"]
                return self.trial.suggest_float(param_name, low, high)
                
        return param_default

    def suggest_model_params(self, model_type):
        """Suggest hyperparameters for a specific model type."""
        if model_type in ["lstm", "rnn"]:
            return {
                "lr": self.trial.suggest_float(f"{model_type}_lr", 1e-5, 1e-2, log=True),
                "dropout": self.trial.suggest_float(f"{model_type}_dropout", 0.0, 0.5),
                "lookback": self.trial.suggest_int(f"{model_type}_lookback", 7, 90),
                "units_per_layer": [self.trial.suggest_categorical(f"{model_type}_units", [32, 64, 128, 256, 512])],
                "loss_function": self.trial.suggest_categorical(f"{model_type}_loss", ["mean_squared_error", "mean_absolute_error", "huber_loss"]),
                "epochs": self.trial.suggest_int(f"{model_type}_epochs", 5, 50),
                "batch_size": self.trial.suggest_categorical(f"{model_type}_batch_size", [16, 32, 64, 128])
            }
        elif model_type == "random_forest":
            return {
                "n_est": self.trial.suggest_int("rf_n_est", 50, 500),
                "mdepth": self.trial.suggest_int("rf_mdepth", 3, 25),
                "min_samples_split": self.trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": self.trial.suggest_int("rf_min_samples_leaf", 1, 10)
            }
        elif model_type == "xgboost":
            return {
                "n_est": self.trial.suggest_int("xgb_n_est", 50, 500),
                "lr": self.trial.suggest_float("xgb_lr", 1e-4, 0.5, log=True),
                "max_depth": self.trial.suggest_int("xgb_max_depth", 3, 12),
                "subsample": self.trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": self.trial.suggest_float("xgb_colsample", 0.5, 1.0)
            }
        elif model_type == "ltc":
            return {
                "lr": self.trial.suggest_float("ltc_lr", 1e-5, 1e-2, log=True),
                "units": self.trial.suggest_int("ltc_units", 32, 512),
                "lookback": self.trial.suggest_int("ltc_lookback", 7, 90),
                "loss_function": self.trial.suggest_categorical("ltc_loss", ["mean_squared_error", "mean_absolute_error", "huber_loss"]),
                "epochs": self.trial.suggest_int("ltc_epochs", 5, 50),
                "batch_size": self.trial.suggest_categorical("ltc_batch_size", [16, 32, 64, 128])
            }
        elif model_type == "tabnet":
            # TabNet architecture parameters with wider ranges
            return {
                "n_d": self.trial.suggest_int("n_d", 8, 256, log=True),
                "n_a": self.trial.suggest_int("n_a", 8, 256, log=True),
                "n_steps": self.trial.suggest_int("n_steps", 1, 15),
                "gamma": self.trial.suggest_float("gamma", 0.5, 3.0),
                "lambda_sparse": self.trial.suggest_float("lambda_sparse", 1e-7, 1e-1, log=True),
                # Optimizer parameters
                "optimizer_lr": self.trial.suggest_float("optimizer_lr", 1e-5, 5e-1, log=True),
                # Training parameters
                "batch_size": self.trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096]),
                "virtual_batch_size": self.trial.suggest_int("virtual_batch_size", 16, 1024, log=True),
                "momentum": self.trial.suggest_float("momentum", 0.005, 0.5),
                "max_epochs": self.trial.suggest_int("max_epochs", 50, 500),
                "patience": self.trial.suggest_int("patience", 5, 50),
                # Convert optimizer_lr to optimizer_params dict
                "optimizer_params": {"lr": self.trial.suggest_float("optimizer_lr", 1e-5, 5e-1, log=True)}
            }
        elif model_type == "cnn":
            # Calculate epochs based on multiplier
            base_epochs = self.trial.suggest_int("cnn_base_epochs", 1, 10)
            actual_epochs = max(1, int(base_epochs * st.session_state.get("epochs_multiplier", 1.0)))
            
            # Get complexity multiplier
            complexity_multiplier = st.session_state.get("tuning_multipliers", {}).get("complexity_multiplier", 1.0)
            
            # Adjust ranges based on complexity multiplier
            max_filters = int(256 * complexity_multiplier)
            
            return {
                "num_conv_layers": self.trial.suggest_int("cnn_num_conv_layers", 1, 5),
                "num_filters": self.trial.suggest_int("cnn_num_filters", 16, max_filters, log=True),
                "kernel_size": self.trial.suggest_int("cnn_kernel_size", 2, 7),
                "stride": self.trial.suggest_int("cnn_stride", 1, 2),
                "dropout_rate": self.trial.suggest_float("cnn_dropout_rate", 0.0, 0.5),
                "activation": self.trial.suggest_categorical("cnn_activation", ["relu", "leaky_relu", "elu"]),
                "use_adaptive_pooling": self.trial.suggest_categorical("cnn_use_adaptive_pooling", [True, False]),
                "fc_layers": [self.trial.suggest_int("cnn_fc_layer_1", 32, 256, log=True),
                              self.trial.suggest_int("cnn_fc_layer_2", 16, 128, log=True)],
                "lookback": self.trial.suggest_int("cnn_lookback", 7, 90),
                "lr": self.trial.suggest_float("cnn_lr", 1e-5, 1e-2, log=True),
                "batch_size": self.trial.suggest_categorical("cnn_batch_size", [16, 32, 64, 128]),
                "epochs": actual_epochs,
                "early_stopping_patience": self.trial.suggest_int("cnn_early_stopping_patience", 3, 10)
            }
        return {}

def ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat):
    """Enhanced objective function for Optuna that exposes all possible parameters for tuning"""
    suggester = OptunaSuggester(trial)
    
    # Set random state for reproducibility
    random_state = trial.suggest_int("random_state", 10, 1000)
    
    def set_all_random_states(seed):
        import random

        import numpy as np
        import tensorflow as tf
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        return seed
    set_all_random_states(random_state)
    
    # Get multipliers from session state
    epochs_multiplier = st.session_state.get("epochs_multiplier", 1.0)
    
    # Get comprehensive multipliers
    tuning_multipliers = st.session_state.get("tuning_multipliers", {
        "trials_multiplier": 1.0, 
        "epochs_multiplier": 1.0, 
        "timeout_multiplier": 1.0,
        "complexity_multiplier": 1.0  # Default to 1.0 if not set
    })
    
    # Extract the complexity multiplier (affects model size/complexity)
    complexity_multiplier = tuning_multipliers.get("complexity_multiplier", 1.0)
    
    # --- Feature Engineering Hyperparameters ---
    # Make ALL feature engineering parameters tunable
    rsi_period = trial.suggest_int("rsi_period", 5, 30)
    macd_fast = trial.suggest_int("macd_fast", 5, 20)
    macd_slow = trial.suggest_int("macd_slow", 15, 40)
    macd_signal = trial.suggest_int("macd_signal", 5, 15)
    boll_window = trial.suggest_int("boll_window", 5, 30)
    boll_nstd = trial.suggest_float("boll_nstd", 1.0, 3.0)
    atr_period = trial.suggest_int("atr_period", 5, 30)
    
    # WERPI parameters
    werpi_wavelet = trial.suggest_categorical("werpi_wavelet", ["db1", "db2", "db4", "sym2", "sym4", "haar"])
    werpi_level = trial.suggest_int("werpi_level", 1, 5)
    werpi_n_states = trial.suggest_int("werpi_n_states", 2, 5)
    werpi_scale = trial.suggest_float("werpi_scale", 0.5, 2.0)
    
    # VMLI parameters
    vmli_window_mom = trial.suggest_int("vmli_window_mom", 5, 30)
    vmli_window_vol = trial.suggest_int("vmli_window_vol", 5, 30)
    vmli_smooth_period = trial.suggest_int("vmli_smooth_period", 1, 10)
    vmli_winsorize_pct = trial.suggest_float("vmli_winsorize_pct", 0.001, 0.05)
    vmli_use_ema = trial.suggest_categorical("vmli_use_ema", [True, False])
    
    # Additional indicators to use
    use_keltner = trial.suggest_categorical("use_keltner", [True, False])
    use_ichimoku = trial.suggest_categorical("use_ichimoku", [True, False])
    use_fibonacci = trial.suggest_categorical("use_fibonacci", [True, False])
    use_volatility = trial.suggest_categorical("use_volatility", [True, False])
    use_momentum = trial.suggest_categorical("use_momentum", [True, False])
    use_breakout = trial.suggest_categorical("use_breakout", [True, False])
    use_deep_analytics = trial.suggest_categorical("use_deep_analytics", [True, False])
    
    # --- Model Type Selection ---
    # Use active model types from session state if available, otherwise use config
    active_model_types = st.session_state.get("active_model_types", ACTIVE_MODEL_TYPES)
    
    # Ensure we have at least one model type
    if not active_model_types:
        active_model_types = MODEL_TYPES  # Fallback to all model types if list is empty
    
    # Let Optuna choose the model type from active models only
    model_type = trial.suggest_categorical("model_type", active_model_types)
    
    # --- Walk-forward parameters ---
    wf_size = trial.suggest_int("walk_forward_window", 3, 60)
    horizon = get_horizon_for_category(range_cat)
    
    # --- Dynamic model-specific parameter tuning ---
    submodel_params = {}
    
    # Let Optuna configure the chosen model type with ALL nuanced parameters
    if model_type == "lstm":
        # Calculate epochs based on multiplier
        base_epochs = trial.suggest_int("lstm_base_epochs", 1, 10)
        actual_epochs = max(1, int(base_epochs * epochs_multiplier))
        
        # Adjust unit ranges based on complexity multiplier
        if complexity_multiplier < 1.0:
            # Simpler model range for quick tuning
            max_units_1 = int(256 * complexity_multiplier)
            max_units_2 = int(128 * complexity_multiplier)
            max_units_3 = int(64 * complexity_multiplier)
        else:
            # Normal or expanded range
            max_units_1 = int(512 * complexity_multiplier)
            max_units_2 = int(256 * complexity_multiplier)
            max_units_3 = int(128 * complexity_multiplier)
        
        submodel_params["lstm"] = {
            "lr": trial.suggest_float("lstm_lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
            
            # Dynamic layer structure with adjusted complexity
            "num_layers": trial.suggest_int("lstm_num_layers", 1, 3),
            "units_per_layer": [
                trial.suggest_int("lstm_units_1", 16, max_units_1, log=True),
                trial.suggest_int("lstm_units_2", 16, max_units_2, log=True) if trial.suggest_categorical("use_layer_2", [True, False]) else 0,
                trial.suggest_int("lstm_units_3", 16, max_units_3, log=True) if trial.suggest_categorical("use_layer_3", [True, False]) else 0
            ],
            
            # Loss function
            "loss_function": trial.suggest_categorical("lstm_loss", LOSS_FUNCTIONS),
            
            # Training parameters
            "epochs": actual_epochs,  # Apply multiplier
            "batch_size": trial.suggest_categorical("lstm_batch_size", [16, 32, 64, 128, 256]),
            
            # Regularization
            "use_batch_norm": trial.suggest_categorical("lstm_batch_norm", [True, False]),
            "l2_reg": trial.suggest_float("lstm_l2_reg", 0.0, 0.01),
            
            # Attention mechanism
            "use_attention": trial.suggest_categorical("lstm_use_attention", [True, False]),
            "attention_type": trial.suggest_categorical("lstm_attention_type", ["dot", "multiplicative", "additive"]),
            "attention_size": trial.suggest_int("lstm_attention_size", 32, 128),
            "attention_heads": trial.suggest_int("lstm_attention_heads", 1, 4),
            "attention_dropout": trial.suggest_float("lstm_attention_dropout", 0.0, 0.5),
            
            # Additional options
            "use_bidirectional": trial.suggest_categorical("lstm_bidirectional", [True, False]),
            "recurrent_dropout": trial.suggest_float("lstm_recurrent_dropout", 0.0, 0.3),
            "activation": trial.suggest_categorical("lstm_activation", ["tanh", "relu"]),
            "recurrent_activation": trial.suggest_categorical("lstm_recurrent_activation", ["sigmoid", "hard_sigmoid"]),
            "stateful": trial.suggest_categorical("lstm_stateful", [True, False])
        }
    elif model_type == "rnn":
        # Calculate epochs based on multiplier
        base_epochs = trial.suggest_int("rnn_base_epochs", 1, 10)
        actual_epochs = max(1, int(base_epochs * epochs_multiplier))
        
        # Adjust unit ranges based on complexity multiplier
        max_units_1 = int(256 * complexity_multiplier)
        max_units_2 = int(128 * complexity_multiplier)
        
        submodel_params["rnn"] = {
            "lr": trial.suggest_float("rnn_lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("rnn_dropout", 0.0, 0.5),
            
            # Dynamic layer structure with adjusted complexity
            "num_layers": trial.suggest_int("rnn_num_layers", 1, 2),
            "units_per_layer": [
                trial.suggest_int("rnn_units_1", 16, max_units_1, log=True),
                trial.suggest_int("rnn_units_2", 16, max_units_2, log=True) if trial.suggest_categorical("rnn_use_layer_2", [True, False]) else 0
            ],
            
            # Loss function
            "loss_function": trial.suggest_categorical("rnn_loss", LOSS_FUNCTIONS),
            
            # Training parameters
            "epochs": actual_epochs,  # Apply multiplier
            "batch_size": trial.suggest_categorical("rnn_batch_size", [16, 32, 64, 128]),
            
            # Regularization
            "use_batch_norm": trial.suggest_categorical("rnn_batch_norm", [True, False]),
            "l2_reg": trial.suggest_float("rnn_l2_reg", 0.0, 0.01)
        }
    elif model_type == "random_forest":
        # Adjust trees count based on complexity multiplier
        base_n_est_max = 1000
        actual_n_est_max = max(100, int(base_n_est_max * complexity_multiplier))
        
        submodel_params["random_forest"] = {
            "n_est": trial.suggest_int("rf_n_est", 50, actual_n_est_max, log=True),
            "mdepth": trial.suggest_int("rf_mdepth", 5, 50),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
            "criterion": trial.suggest_categorical("rf_criterion", ["squared_error", "absolute_error", "poisson"])
        }
    elif model_type == "xgboost":
        # Adjust trees count based on complexity multiplier
        base_n_est_max = 1000
        actual_n_est_max = max(100, int(base_n_est_max * complexity_multiplier))
        
        submodel_params["xgboost"] = {
            "n_est": trial.suggest_int("xgb_n_est", 50, actual_n_est_max, log=True),
            "lr": trial.suggest_float("xgb_lr", 0.001, 0.5, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 15),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "gamma": trial.suggest_float("xgb_gamma", 0, 5),
            "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
            "objective": trial.suggest_categorical("xgb_objective", 
                ["reg:squarederror", "reg:absoluteerror", "reg:squaredlogerror"])
        }
    elif model_type == "tft":
        # Calculate epochs based on multiplier
        base_epochs = trial.suggest_int("tft_base_epochs", 1, 10)
        actual_epochs = max(1, int(base_epochs * epochs_multiplier))
        
        # Adjust size parameters based on complexity multiplier
        max_hidden_size = int(512 * complexity_multiplier)
        max_lstm_units = int(512 * complexity_multiplier)
        
        submodel_params["tft"] = {
            "lr": trial.suggest_float("tft_lr", 1e-5, 1e-2, log=True),
            "hidden_size": trial.suggest_int("tft_hidden_size", 32, max_hidden_size, log=True),
            "lstm_units": trial.suggest_int("tft_lstm_units", 32, max_lstm_units, log=True),
            "num_heads": trial.suggest_int("tft_num_heads", 1, 8),
            "dropout": trial.suggest_float("tft_dropout", 0.0, 0.5),
            "loss_function": trial.suggest_categorical("tft_loss", LOSS_FUNCTIONS),
            "epochs": actual_epochs,  # Apply multiplier
            "batch_size": trial.suggest_categorical("tft_batch_size", [16, 32, 64, 128])
        }
    elif model_type == "ltc":
        # Calculate epochs based on multiplier
        base_epochs = trial.suggest_int("ltc_base_epochs", 1, 10)
        actual_epochs = max(1, int(base_epochs * epochs_multiplier))
        
        # Adjust units based on complexity multiplier
        max_units = int(512 * complexity_multiplier)
        
        submodel_params["ltc"] = {
            "lr": trial.suggest_float("ltc_lr", 1e-5, 1e-2, log=True),
            "units": trial.suggest_int("ltc_units", 32, max_units, log=True),
            "loss_function": trial.suggest_categorical("ltc_loss", LOSS_FUNCTIONS),
            "epochs": actual_epochs,  # Apply multiplier
            "batch_size": trial.suggest_categorical("ltc_batch_size", [16, 32, 64, 128])
        }
    elif model_type == "tabnet":
        # Calculate epochs based on multiplier
        # TabNet already scales epochs based on complexity
        base_max_epochs = trial.suggest_int("tabnet_base_max_epochs", 50, 200)
        actual_max_epochs = max(10, int(base_max_epochs * epochs_multiplier))
        
        # Adjust dimension sizes based on complexity multiplier
        max_n_d = int(256 * complexity_multiplier)
        max_n_a = int(256 * complexity_multiplier)
        
        # TabNet architecture parameters with adjusted ranges
        submodel_params["tabnet"] = {
            "n_d": trial.suggest_int("n_d", 8, max_n_d, log=True),
            "n_a": trial.suggest_int("n_a", 8, max_n_a, log=True),
            "n_steps": trial.suggest_int("n_steps", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.5, 3.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-7, 1e-1, log=True),
            
            # Optimizer parameters
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-5, 5e-1, log=True),
            
            # Training parameters
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096]),
            "virtual_batch_size": trial.suggest_int("virtual_batch_size", 16, 1024, log=True),
            "momentum": trial.suggest_float("momentum", 0.005, 0.5),
            "max_epochs": actual_max_epochs,  # Apply multiplier
            "patience": trial.suggest_int("patience", 5, 50),
            
            # Convert optimizer_lr to optimizer_params dict
            "optimizer_params": {"lr": trial.suggest_float("optimizer_lr", 1e-5, 5e-1, log=True)}
        }
    elif model_type == "cnn":
        # Calculate epochs based on multiplier
        base_epochs = trial.suggest_int("cnn_base_epochs", 1, 10)
        actual_epochs = max(1, int(base_epochs * epochs_multiplier))
        
        # Get complexity multiplier
        complexity_multiplier = tuning_multipliers.get("complexity_multiplier", 1.0)
        
        # Adjust ranges based on complexity multiplier
        max_filters = int(256 * complexity_multiplier)
        
        submodel_params["cnn"] = {
            "num_conv_layers": suggester.trial.suggest_int("cnn_num_conv_layers", 1, 5),
            "num_filters": suggester.trial.suggest_int("cnn_num_filters", 16, max_filters, log=True),
            "kernel_size": suggester.trial.suggest_int("cnn_kernel_size", 2, 7),
            "stride": suggester.trial.suggest_int("cnn_stride", 1, 2),
            "dropout_rate": suggester.trial.suggest_float("cnn_dropout_rate", 0.0, 0.5),
            "activation": suggester.trial.suggest_categorical("cnn_activation", ["relu", "leaky_relu", "elu"]),
            "use_adaptive_pooling": suggester.trial.suggest_categorical("cnn_use_adaptive_pooling", [True, False]),
            "fc_layers": [suggester.trial.suggest_int("cnn_fc_layer_1", 32, 256, log=True),
                          suggester.trial.suggest_int("cnn_fc_layer_2", 16, 128, log=True)],
            "lookback": suggester.trial.suggest_int("cnn_lookback", 7, 90),
            "lr": suggester.trial.suggest_float("cnn_lr", 1e-5, 1e-2, log=True),
            "batch_size": suggester.trial.suggest_categorical("cnn_batch_size", [16, 32, 64, 128]),
            "epochs": actual_epochs,
            "early_stopping_patience": suggester.trial.suggest_int("cnn_early_stopping_patience", 3, 10)
        }
    
    # Set ensemble weights - use only active model types
    ensemble_weights = {mtype: 0.0 for mtype in MODEL_TYPES}
    ensemble_weights[model_type] = 1.0
    
    # Use LazyImportManager for fetching data and feature engineering
    fetch_data = LazyImportManager.get_data_fetcher()
    feature_engineering_with_params = LazyImportManager.get_feature_engineering()
    
    df_raw = fetch_data(ticker, start=START_DATE, interval=timeframe)
    if df_raw is None or len(df_raw) < 50:
        return 1e9

    df_raw = feature_engineering_with_params(
        df_raw,
        ticker=ticker,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        boll_window=boll_window,
        boll_nstd=boll_nstd,
        atr_period=atr_period,
        werpi_wavelet=werpi_wavelet,
        werpi_level=werpi_level,
        werpi_n_states=werpi_n_states,
        werpi_scale=werpi_scale,
        use_keltner=use_keltner,
        use_ichimoku=use_ichimoku,
        use_fibonacci=use_fibonacci,
        use_volatility=use_volatility,
        use_momentum=use_momentum,
        use_breakout=use_breakout,
        use_deep_analytics=use_deep_analytics
    )
    
    # Assume get_active_feature_names is imported from config
    from config.config_loader import get_active_feature_names
    feature_cols = get_active_feature_names()
    
    mse_val, mape_val = walk_forward_ensemble_eval(
        df=df_raw,
        feature_cols=feature_cols,
        horizon=horizon,
        wf_size=wf_size,
        submodel_params_dict=submodel_params,
        ensemble_weights=ensemble_weights,
        trial=trial
    )
    
    rmse = np.sqrt(mse_val)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("mape", mape_val)
    
    trial.report(rmse, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    print(f"Trial {repr(trial.number)}: rmse = {rmse:.4f}, mape = {mape_val:.2f}%")
    
    # Dynamic pruning based on study state
    if PRUNING_ENABLED and len(trial.study.trials) > PRUNING_MIN_TRIALS:  
        # Get median performance of completed trials
        completed_trials = [t for t in trial.study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            median_rmse = np.median([t.value for t in completed_trials if t.value is not None])
            
            # Prune if significantly worse than median
            if rmse > median_rmse * PRUNING_MEDIAN_FACTOR:
                logger.info(f"Pruning trial {trial.number}: RMSE {rmse:.4f} vs median {median_rmse:.4f}")
                raise optuna.exceptions.TrialPruned()
    
    # Always prune if absolutely terrible
    if PRUNING_ENABLED and (rmse > RMSE_THRESHOLD * PRUNING_ABSOLUTE_RMSE_FACTOR or 
                         mape_val > MAPE_THRESHOLD * PRUNING_ABSOLUTE_MAPE_FACTOR):
        logger.info(f"Pruning trial {trial.number} due to poor absolute performance: "
                  f"RMSE={rmse:.4f}, MAPE={mape_val:.2f}%")
        raise optuna.exceptions.TrialPruned()
    
    if rmse < RMSE_THRESHOLD and mape_val < MAPE_THRESHOLD:
        print(f"🌟 Found solution meeting thresholds: RMSE={rmse:.4f}, MAPE={mape_val:.2f}%")
        best_params = {
            "model_type": model_type,
            "walk_forward_size": wf_size,
            "feature_engineering": {
                "rsi_period": rsi_period,
                "macd_fast": macd_fast,
                "macd_slow": macd_slow,
                "macd_signal": macd_signal,
                "boll_window": boll_window,
                "boll_nstd": boll_nstd,
                "atr_period": atr_period,
                "werpi_wavelet": werpi_wavelet,
                "werpi_level": werpi_level,
                "werpi_n_states": werpi_n_states,
                "werpi_scale": werpi_scale,
                "use_keltner": use_keltner,
                "use_ichimoku": use_ichimoku,
                "use_fibonacci": use_fibonacci,
                "use_volatility": use_volatility,
                "use_momentum": use_momentum,
                "use_breakout": use_breakout,
                "use_deep_analytics": use_deep_analytics
            },
            "model_params": submodel_params,
            "metrics": {"rmse": float(rmse), "mape": float(mape_val)},
            "timestamps": {"found_at": datetime.now().isoformat()}
        }
        safe_write_yaml(BEST_PARAMS_FILE, best_params)
    return rmse

# [Rest of the file with function definitions unchanged]

def main():
    """Main entry point for tuning."""
    from src.utils.utils import adaptive_memory_clean

    # Clean memory at start
    adaptive_memory_clean("large")
    
    # ...existing code...
    
    # Loop through tuning cycles
    for cycle in range(1, TUNING_LOOP + 1):
        # ...cycle code...
        
        # Clean memory between cycles
        adaptive_memory_clean("large")
        
    # Final memory cleanup
    adaptive_memory_clean("large")

# Cross-platform file locking utility
def safe_read_yaml(filepath, max_attempts=5, retry_delay=0.1):
    """Read YAML file with cross-platform locking support"""
    if not os.path.exists(filepath):
        return {}
        
    for attempt in range(max_attempts):
        try:
            if platform.system() == 'Windows':
                # Windows approach using file existence checking
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    return data
            else:
                # Unix-like systems can use fcntl
                import fcntl
                with open(filepath, "r", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
                    data = yaml.safe_load(f) or {}
                    fcntl.flock(f, fcntl.LOCK_UN)  # Release lock
                    return data
        except (IOError, OSError) as e:
            # File might be locked by another process
            if attempt < max_attempts - 1:
                time.sleep(retry_delay * (1 + random.random()))  # Randomized exponential backoff
                continue
            print(f"Error reading {filepath} after {max_attempts} attempts: {e}")
            return {}
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return {}

def safe_write_yaml(filepath, data, max_attempts=5, retry_delay=0.1):
    """Write to YAML file with cross-platform locking support"""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Ensure data is serializable
    processed_data = convert_to_builtin_type(data)
    
    for attempt in range(max_attempts):
        try:
            if platform.system() == 'Windows':
                # Windows approach - just try to write
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.safe_dump(processed_data, f, default_flow_style=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                    return True
            else:
                # Unix-like systems can use fcntl
                import fcntl
                with open(filepath, "w", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
                    yaml.safe_dump(processed_data, f, default_flow_style=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                    fcntl.flock(f, fcntl.LOCK_UN)  # Release lock
                    return True
        except (IOError, OSError) as e:
            # File might be locked by another process
            if attempt < max_attempts - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            print(f"Error writing to {filepath} after {max_attempts} attempts: {e}")
            return False
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")
            return False

# Define a stop callback to halt tuning when stop is requested
class StopStudyCallback:
    def __call__(self, study, trial):
        if is_stop_requested():
            study.stop()

# Add to meta_tuning.py after other imports but before main code
def get_model_registry():
    """Lazily load the model registry to avoid circular imports"""
    # Use function attribute for caching
    if not hasattr(get_model_registry, "_registry"):
        try:
            # Import only when needed
            from src.training.incremental_learning import ModelRegistry  # Fixed typo: ModelRegistryry → ModelRegistry

            # Create registry directory
            registry_dir = os.path.join(DATA_DIR, "model_registry")
            os.makedirs(registry_dir, exist_ok=True)
            
            # Create or get registry
            registry = ModelRegistry(registry_dir=registry_dir, create_if_missing=True)
            print(f"Model registry initialized at {registry_dir}")
            get_model_registry._registry = registry
        except (ImportError, Exception) as e:
            print(f"Warning: Could not initialize model registry: {e}")
            print("Model registration will be disabled")
            get_model_registry._registry = None
    
    return get_model_registry._registry

def register_best_model(study, trial, ticker, timeframe):
    """Register the best model from a tuning run to the model registry"""
    # Get model registry (or return if not available)
    registry = get_model_registry()
    if not registry:
        print("Model registry not available - skipping registration")
        return None
    
    # Get model parameters
    try:
        best_params = trial.params.copy()
        
        # Get feature columns and horizon
        feature_cols = get_active_feature_names()
        horizon = get_horizon_for_category(best_params.get("range_category", "all"))
        
        # Extract model type
        model_type = best_params.get("model_type", "ensemble")
        if "model_type" in best_params:
            del best_params["model_type"]  # Remove to avoid duplication in hyperparams
        
        # Get model builder using lazy import manager
        build_model_by_type = LazyImportManager.get_model_builder()
        
        # Prepare architecture parameters
        architecture_params = {
            "units_per_layer": best_params.get("units", [64, 32]),
            "hidden_size": best_params.get("hidden_size", 64),
            "lstm_units": best_params.get("lstm_units", 128),
            "num_heads": best_params.get("num_heads", 4),
            "use_batch_norm": best_params.get("use_batch_norm", False)
        }
        
        # Only try to build and register neural network models
        if model_type in ["lstm", "rnn", "tft"]:
            # Build the model
            build_model_by_type = LazyImportManager.get_model_builder()
            model = build_model_by_type(
                model_type=model_type,
                num_features=len(feature_cols),
                horizon=horizon,
                learning_rate=best_params.get("lr", 0.001),
                dropout_rate=best_params.get("dropout", 0.2),
                loss_function=best_params.get("loss_function", "mean_squared_error"),
                lookback=best_params.get("lookback", 30),
                architecture_params=architecture_params
            )
            
            # Performance metrics
            metrics = {
                "rmse": trial.user_attrs.get("rmse", 0),
                "mape": trial.user_attrs.get("mape", 0),
                "objective_value": trial.value,
                "trial_number": trial.number,
                "timestamp": datetime.now().isoformat()
            }
            
            # Register the model
            model_id = registry.register_model(
                model=model,
                model_type=model_type,
                ticker=ticker,
                timeframe=timeframe,
                metrics=metrics,
                hyperparams=best_params,
                tags=["optuna", "best_trial", f"cycle_{study.user_attrs.get('cycle', 0)}"]
            )
            
            print(f"✅ Registered best {model_type} model with ID: {model_id}")
            return model_id
            
        else:
            print(f"⚠️ Skipping registration for non-neural model type: {model_type}")
            return None
    
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return None
    
# In Scripts/meta_tuning.py
def tune_for_combo(ticker, timeframe, range_cat="all", n_trials=None, cycle=1):
    """Run hyperparameter optimization for a specific ticker-timeframe combination."""
    from src.utils.utils import adaptive_memory_clean

    # Clean memory before starting
    adaptive_memory_clean("large")
    
    # Get tuning multipliers from session state
    tuning_multipliers = st.session_state.get("tuning_multipliers", {
        "trials_multiplier": 1.0,
        "epochs_multiplier": 1.0,
        "timeout_multiplier": 1.0,
        "complexity_multiplier": 1.0
    })
    
    # Determine trials dynamically between min and max using multiplier
    if n_trials is None:
        # Use random value between min and max for automatic exploration
        from config.config_loader import TUNING_TRIALS_PER_CYCLE_min, TUNING_TRIALS_PER_CYCLE_max
        import random
        
        # Adjust trials based on multiplier
        trials_multiplier = tuning_multipliers.get("trials_multiplier", 1.0)
        adjusted_min = max(5, int(TUNING_TRIALS_PER_CYCLE_min * trials_multiplier))
        adjusted_max = max(adjusted_min + 5, int(TUNING_TRIALS_PER_CYCLE_max * trials_multiplier))
        
        n_trials = random.randint(adjusted_min, adjusted_max)
        logger.info(f"Auto-selecting {n_trials} trials for this cycle (between {adjusted_min} and {adjusted_max})")
    
    study_name = f"{ticker}_{timeframe}_{range_cat}_cycle{cycle}"
    storage_name = f"sqlite:///{os.path.join(DB_DIR, f'{study_name}.db')}"
    
    # Apply complexity multiplier to startup trials
    if tuning_multipliers.get("complexity_multiplier", 1.0) < 1.0:
        # Reduce startup trials for quick tuning
        adjusted_startup_trials = max(2, int(N_STARTUP_TRIALS * tuning_multipliers.get("complexity_multiplier", 1.0)))
    else:
        adjusted_startup_trials = N_STARTUP_TRIALS
        
    # Set cycle in study attributes for tracking
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=adjusted_startup_trials)
    )
    study.set_user_attr("cycle", cycle)
    study.set_user_attr("tuning_multipliers", tuning_multipliers)  # Store for reference
    
    # Fix: Safely check for best trial using a try-except block
    try:
        if study.best_trial:
            logger.info(f"Current best: {study.best_value:.6f} (Trial {study.best_trial.number})")
    except (ValueError, AttributeError, RuntimeError):
        logger.info(f"No trials completed yet for study {study_name}")
    
    # Define callbacks
    stop_callback = StopStudyCallback()
    progress_callback = create_progress_callback(cycle=cycle)
    
    # Set timeout based on multiplier
    timeout_seconds = None
    if tuning_multipliers.get("timeout_multiplier", 1.0) > 0:
        base_timeout_seconds = 3600  # 1 hour baseline
        timeout_seconds = int(base_timeout_seconds * tuning_multipliers.get("timeout_multiplier", 1.0))
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat),
            n_trials=n_trials,
            callbacks=[progress_callback, stop_callback],
            timeout=timeout_seconds
        )
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
    finally:
        # Clean memory after optimization
        adaptive_memory_clean("large")
    
    # Register the best model if study wasn't interrupted
    try:
        if not is_stop_requested() and study.best_trial:
            model_id = register_best_model(study, study.best_trial, ticker, timeframe)
            
            # If successful, save model ID to best_params.yaml
            if model_id:
                best_params_data = safe_read_yaml(BEST_PARAMS_FILE) or {}
                if "best_models" not in best_params_data:
                    best_params_data["best_models"] = {}
                
                # Store by ticker and timeframe
                if ticker not in best_params_data["best_models"]:
                    best_params_data["best_models"][ticker] = {}
                
                best_params_data["best_models"][ticker][timeframe] = model_id
                safe_write_yaml(BEST_PARAMS_FILE, best_params_data)
    except (ValueError, AttributeError, RuntimeError) as e:
        logger.warning(f"Could not register best model: {e}")
    
    return study

def create_progress_callback(cycle=1):
    """Create a callback that updates progress file."""
    from src.utils.utils import adaptive_memory_clean
    
    def progress_callback(study, trial):
        # Update progress
        # ...existing code...
        
        # Perform adaptive memory cleaning based on trial number
        if trial.number % 5 == 0:  # Every 5 trials
            adaptive_memory_clean("small")
            
        if trial.number % 20 == 0:  # Every 20 trials
            adaptive_memory_clean("medium")
            
    return progress_callback

# Add these functions after your other function definitions

def get_model_registry():
    """Lazily load the model registry to avoid circular imports"""
    # Use function attribute for caching
    if not hasattr(get_model_registry, "_registry"):
        try:
            # Import only when needed
            from src.training.incremental_learning import ModelRegistry  # Fixed typo: ModelRegistryry → ModelRegistry

            # Create registry directory
            registry_dir = os.path.join(DATA_DIR, "model_registry")
            os.makedirs(registry_dir, exist_ok=True)
            
            # Create or get registry
            registry = ModelRegistry(registry_dir=registry_dir, create_if_missing=True)
            print(f"Model registry initialized at {registry_dir}")
            get_model_registry._registry = registry
        except (ImportError, Exception) as e:
            print(f"Warning: Could not initialize model registry: {e}")
            print("Model registration will be disabled")
            get_model_registry._registry = None
    
    return get_model_registry._registry

def register_best_model(study, trial, ticker, timeframe):
    """Register the best model from a tuning run to the model registry"""
    # Get model registry (or return if not available)
    registry = get_model_registry()
    if not registry:
        print("Model registry not available - skipping registration")
        return None
    
    # Get model parameters
    try:
        best_params = trial.params.copy()
        
        # Get feature columns and horizon
        feature_cols = get_active_feature_names()
        horizon = get_horizon_for_category(best_params.get("range_category", "all"))
        
        # Extract model type
        model_type = best_params.get("model_type", "ensemble")
        if "model_type" in best_params:
            del best_params["model_type"]  # Remove to avoid duplication in hyperparams
        
        # Get model builder using lazy import manager
        build_model_by_type = LazyImportManager.get_model_builder()
        
        # Prepare architecture parameters
        architecture_params = {
            "units_per_layer": best_params.get("units", [64, 32]),
            "hidden_size": best_params.get("hidden_size", 64),
            "lstm_units": best_params.get("lstm_units", 128),
            "num_heads": best_params.get("num_heads", 4),
            "use_batch_norm": best_params.get("use_batch_norm", False)
        }
        
        # Only try to build and register neural network models
        if model_type in ["lstm", "rnn", "tft"]:
            # Build the model
            build_model_by_type = LazyImportManager.get_model_builder()
            model = build_model_by_type(
                model_type=model_type,
                num_features=len(feature_cols),
                horizon=horizon,
                learning_rate=best_params.get("lr", 0.001),
                dropout_rate=best_params.get("dropout", 0.2),
                loss_function=best_params.get("loss_function", "mean_squared_error"),
                lookback=best_params.get("lookback", 30),
                architecture_params=architecture_params
            )
            
            # Performance metrics
            metrics = {
                "rmse": trial.user_attrs.get("rmse", 0),
                "mape": trial.user_attrs.get("mape", 0),
                "objective_value": trial.value,
                "trial_number": trial.number,
                "timestamp": datetime.now().isoformat()
            }
            
            # Register the model
            model_id = registry.register_model(
                model=model,
                model_type=model_type,
                ticker=ticker,
                timeframe=timeframe,
                metrics=metrics,
                hyperparams=best_params,
                tags=["optuna", "best_trial", f"cycle_{study.user_attrs.get('cycle', 0)}"]
            )
            
            print(f"✅ Registered best {model_type} model with ID: {model_id}")
            return model_id
            
        else:
            print(f"⚠️ Skipping registration for non-neural model type: {model_type}")
            return None
    
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return None

# ...existing code...

# Add import to get default INTERVAL from config
from config.config_loader import INTERVAL

# ...existing code...

# ...existing code...

# Add this variable to store the training dataframe
df = None  # Will be set by dashboard.py when starting tuning

def main():
    """Main entry point for meta-tuning process"""
    global df
    if df is None and 'df_train' in st.session_state:
        df = st.session_state['df_train']
        logger.info(f"Using training data from session state: {len(df)} rows")
    
    # Determine interval: from session state if set, otherwise default to config.INTERVAL
    interval = st.session_state.get("selected_timeframe", INTERVAL)
    
    # If we don't have training data yet, attempt to fetch it
    if df is None:
        try:
            # Get training dates from session state or use defaults
            training_start_date = st.session_state.get("training_start_date", 
                                                     (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
            
            if isinstance(training_start_date, datetime):
                training_start_str = training_start_date.strftime('%Y-%m-%d')
            else:
                training_start_str = training_start_date
                
            end_date = datetime.now().strftime('%Y-%m-%d')
            ticker = st.session_state.get("selected_ticker", TICKER)
            
            # Use the determined interval here
            logger.info(f"Fetching training data from {training_start_str} to {end_date} for {ticker} with interval {interval}")
            from src.data.data import fetch_data
            df = fetch_data(ticker=ticker, start=training_start_str, end=end_date, interval=interval)
            
            if df is not None:
                logger.info(f"Successfully fetched training data: {len(df)} rows")
            else:
                logger.error("Failed to fetch training data")
                return False
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return False

# ...existing code...

# Import configuration properly
from config.config_loader import get_config

# Use this before importing walk_forward
config = get_config()
LOOKBACK = config.get('LOOKBACK', 30)
PREDICTION_HORIZON = config['PREDICTION_HORIZON']

# Now import from training
from src.training.walk_forward import run_walk_forward as walk_forward_ensemble_eval

# ...existing code...
