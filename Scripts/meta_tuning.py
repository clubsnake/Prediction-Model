"""
Implements an Optuna-based hyperparameter search with ensemble approach and walk-forward.
Logs detailed trial information live to session state and YAML files:
- progress.yaml for current progress,
- best_params.yaml when thresholds are met,
- tested_models.yaml with details for each trial,
- cycle_metrics.yaml with cycle-level summaries.
"""

# Keep only essential imports at the module level
import numpy as np
import yaml
import sys
import os
import signal
import optuna
import logging
from datetime import datetime
import threading
import streamlit as st
import platform
import time
import random
from Scripts.walk_forward import run_walk_forward as walk_forward_ensemble_eval

# Import direct configuration that doesn't create cycles
from config import (
    N_STARTUP_TRIALS,
    TUNING_TRIALS_PER_CYCLE_max,
    TUNING_TRIALS_PER_CYCLE_min,
    TUNING_LOOP,
    RMSE_THRESHOLD,
    MAPE_THRESHOLD,
    MODEL_TYPES,
    LOSS_FUNCTIONS,
    START_DATE,
    get_active_feature_names,
    get_horizon_for_category,
    get_hyperparameter_ranges,
    TICKER,
    TIMEFRAMES, TFT_GRID
)

# Import local helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from progress_helper import update_progress_in_yaml
from Scripts.threadsafe import (
    safe_read_yaml, 
    safe_write_yaml,
    safe_read_json,
    safe_write_json,
    AtomicFileWriter,
    convert_to_native_types
)

# Add the following imports after your existing imports
import logging

# Configure logger properly
logger = logging.getLogger(__name__)

# Import pruning settings from config
from config import (
    PRUNING_ENABLED,
    PRUNING_MIN_TRIALS, 
    PRUNING_MEDIAN_FACTOR,
    PRUNING_ABSOLUTE_RMSE_FACTOR,
    PRUNING_ABSOLUTE_MAPE_FACTOR,
)

# Session state initialization
if "trial_logs" not in st.session_state:
    st.session_state["trial_logs"] = []

# Global stop control via event
stop_event = threading.Event()

def set_stop_requested(val: bool):
    if val:
        stop_event.set()
        print("Stop requested - flag set")
    else:
        stop_event.clear()

def is_stop_requested():
    return stop_event.is_set()

if sys.platform != "win32":
    def signal_handler(sig, frame):
        print("\nManual stop requested. Exiting tuning loop.")
        set_stop_requested(True)
    signal.signal(signal.SIGINT, signal_handler)

# Data directories and file paths
DATA_DIR = "Data"
DB_DIR = os.path.join(DATA_DIR, "DB")
LOGS_DIR = os.path.join(DATA_DIR, "Logs")
MODELS_DIR = os.path.join(DATA_DIR, "Models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")
CYCLE_METRICS_FILE = os.path.join(DATA_DIR, "cycle_metrics.yaml")
BEST_PARAMS_FILE = os.path.join(DATA_DIR, "best_params.yaml")

def convert_to_builtin_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    else:
        return obj

class AdaptiveEnsembleWeighting:
    def __init__(self, model_types, initial_weights, memory_factor, min_weight, exploration_factor):
        self.weights = initial_weights.copy()
        self.memory_factor = memory_factor
        self.min_weight = min_weight
        self.exploration_factor = exploration_factor

    def get_weights(self):
        return self.weights

def update_performance(self, mtype, mse):
    """Update performance history for a specific model type."""
    if mtype not in self.weights:
        return  # Ignore unknown model types
    
    # Initialize performance history if not exists
    if not hasattr(self, 'performance_history'):
        self.performance_history = {mt: [] for mt in self.weights.keys()}
    
    # Add new performance
    self.performance_history[mtype].append(mse)
    
    # Limit history length
    max_history = 10  # Keep last 10 entries
    if len(self.performance_history[mtype]) > max_history:
        self.performance_history[mtype] = self.performance_history[mtype][-max_history:]

def update_weights(self):
    """Recompute weights based on recent performance."""
    if not hasattr(self, 'performance_history'):
        return  # No history, weights remain unchanged
        
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
    
    if total_weight > 0:
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
        valid = [(self.weights[mtype], np.array(preds)) for mtype, preds in predictions.items() if preds is not None]
        if valid:
            total = sum(weight for weight, _ in valid)
            if total == 0:
                return None
            weighted = sum(weight * preds for weight, preds in valid) / total
            return weighted
        return None

def reset_progress():
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
    Handles lazy imports to avoid circular import issues.
    Imports only happen when functions are called (on demand), not at module load time.
    """
    @staticmethod
    def get_model_builder():
        from model import build_model_by_type
        return build_model_by_type

    @staticmethod
    def get_data_fetcher():
        from data import fetch_data
        return fetch_data

    @staticmethod
    def get_feature_engineering():
        from features import feature_engineering_with_params
        return feature_engineering_with_params

    @staticmethod
    def get_scaling_function():
        from preprocessing import scale_data
        return scale_data

    @staticmethod
    def get_sequence_function():
        from preprocessing import create_sequences
        return create_sequences

    @staticmethod
    def get_optuna_sampler():
        import optuna
        return optuna.samplers.TPESampler

    @staticmethod
    def get_evaluation_function():
        from model import evaluate_predictions
        return evaluate_predictions

def get_model_prediction(mtype, submodel_params, X_train, y_train, X_test, horizon, unified_lookback, feature_cols):
    """
    Get predictions from a model without circular imports.
    Uses lazy imports to avoid circular dependencies.
    """
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
        from ltc_model import build_ltc_model
        
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
        
    return preds

class OptunaSuggester:
    """Simple wrapper for Optuna parameter suggestions."""
    def __init__(self, trial):
        self.trial = trial

    def suggest(self, param_name):
        from config import HYPERPARAMETER_REGISTRY
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
        return {}

def ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat):
    from config import META_TUNING_PARAMS  # Ensure proper config import
    suggester = OptunaSuggester(trial)
    
    # Set random state for reproducibility (if not suggested, use a default)
    try:
        random_state = suggester.suggest("random_state")
    except Exception:
        random_state = 42  # Fallback
    
    def set_all_random_states(seed):
        import random
        import numpy as np
        import tensorflow as tf
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        return seed
    set_all_random_states(random_state)
    
    # --- Feature Engineering Hyperparameters ---
    rsi_period = suggester.suggest("rsi_period")
    macd_fast = suggester.suggest("macd_fast")
    macd_slow = suggester.suggest("macd_slow")
    macd_signal = suggester.suggest("macd_signal")
    boll_window = suggester.suggest("boll_window")
    boll_nstd = suggester.suggest("boll_nstd")
    atr_period = suggester.suggest("atr_period")
    werpi_wavelet = suggester.suggest("werpi_wavelet")
    werpi_level = suggester.suggest("werpi_level")
    werpi_n_states = suggester.suggest("werpi_n_states")
    werpi_scale = suggester.suggest("werpi_scale")
    
    use_keltner = trial.suggest_categorical("use_keltner", [True, False])
    use_ichimoku = trial.suggest_categorical("use_ichimoku", [True, False])
    use_fibonacci = trial.suggest_categorical("use_fibonacci", [True, False])
    use_volatility = trial.suggest_categorical("use_volatility", [True, False])
    use_momentum = trial.suggest_categorical("use_momentum", [True, False])
    use_breakout = trial.suggest_categorical("use_breakout", [True, False])
    use_deep_analytics = trial.suggest_categorical("use_deep_analytics", [True, False])
    
    from config import MODEL_TYPES  # Ensure consistent import
    model_type = trial.suggest_categorical("model_type", MODEL_TYPES)
    
    wf_size = suggester.suggest("walk_forward_default")
    horizon = get_horizon_for_category(range_cat)
    
    submodel_params = {}
    submodel_params[model_type] = suggester.suggest_model_params(model_type)
    
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
    from config import get_active_feature_names
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
    from utils import adaptive_memory_clean
    
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
            from Scripts.incremental_learning import ModelRegistry
            
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
    from utils import adaptive_memory_clean  # Import the memory cleaning function
    
    # Clean memory before starting
    adaptive_memory_clean("large")
    
    study_name = f"{ticker}_{timeframe}_{range_cat}_cycle{cycle}"
    storage_name = f"sqlite:///{os.path.join(DB_DIR, f'{study_name}.db')}"
    
    # Set cycle in study attributes for tracking
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    )
    study.set_user_attr("cycle", cycle)
    
    if study.best_trial:
        print(f"Current best: {study.best_value:.6f} (Trial {study.best_trial.number})")
    
    # Define callbacks
    stop_callback = StopStudyCallback()
    progress_callback = create_progress_callback(cycle=cycle)
    
    # Run optimization
    study.optimize(
        lambda trial: ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat),
        n_trials=n_trials,
        callbacks=[progress_callback, stop_callback]
    )
    
    # Clean memory after optimization
    adaptive_memory_clean("large")
    
    # Register the best model if study wasn't interrupted
    if study.best_trial and not is_stop_requested():
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
    
    return study

def create_progress_callback(cycle=1):
    """Create a callback that updates progress file."""
    from utils import adaptive_memory_clean
    
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
            from Scripts.incremental_learning import ModelRegistry
            
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
