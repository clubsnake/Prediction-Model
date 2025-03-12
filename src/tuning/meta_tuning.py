"""
Implements an Optuna-based hyperparameter search with ensemble approach and walk-forward.
Logs detailed trial information live to session state and YAML files:
- progress.yaml for current progress,
- best_params.yaml when thresholds are met,
- tested_models.yaml with details for each trial,
- cycle_metrics.yaml with cycle-level summaries.
"""

import logging
import os
import sys
import traceback
from datetime import datetime

# Add project root to sys.path for absolute imports
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from config
from config import DATA_DIR, MODELS_DIR

# Import StopStudyCallback from callbacks
from src.training.callbacks import StopStudyCallback
from src.training.walk_forward import perform_walkforward_validation

# Define log file paths - FIXED LOCATIONS IN ROOT DATA DIR
# This should match the paths in progress_helper.py
LOG_DIR = os.path.join(DATA_DIR, "Models", "Tuning_Logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Use the same paths as in progress_helper.py for consistency
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")
TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
CYCLE_METRICS_FILE = os.path.join(DATA_DIR, "cycle_metrics.yaml")
BEST_PARAMS_FILE = os.path.join(
    DATA_DIR, "Models", "Hyperparameters", "best_params.yaml"
)
os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)  # Ensure directory exists

# Import necessary modules
import platform
import random
import signal
import time

# Import study_manager for unified model tuning
from src.tuning.study_manager import (
    create_model_objective,
    create_resilient_study,
    evaluate_model_with_walkforward,
    study_manager,
)

# Set optimized environment variables early
from src.utils.env_setup import setup_tf_environment

# Get mixed precision setting from config first
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

# Import pruning settings from config
# Import config settings
from config.config_loader import (
    ACTIVE_MODEL_TYPES,
    N_STARTUP_TRIALS,
    TUNING_LOOP,
    TUNING_TRIALS_PER_CYCLE_max,
    TUNING_TRIALS_PER_CYCLE_min,
    get_active_feature_names,
    get_horizon_for_category,
)
from src.models.cnn_model import CNNPricePredictor
from src.tuning.progress_helper import update_progress_in_yaml
from src.utils.threadsafe import (
    safe_read_yaml,
    safe_write_yaml,
)

# Configure logger properly
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Session state initialization
if "trial_logs" not in st.session_state:
    st.session_state["trial_logs"] = []

# Use the stop event from progress_helper for consistency
from src.tuning.progress_helper import is_stop_requested, set_stop_requested

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

    def __init__(
        self,
        model_types,
        initial_weights,
        memory_factor,
        min_weight,
        exploration_factor,
    ):
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
            self.performance_history[mtype] = self.performance_history[mtype][
                -max_history:
            ]

    def update_weights(self):
        """Recompute weights based on recent performance."""
        # Calculate average performance for each model
        avg_perf = {}
        for mtype, history in self.performance_history.items():
            if history:
                # Lower MSE is better, so use inverse for weighting
                avg_perf[mtype] = 1.0 / (
                    np.mean(history) + 1e-10
                )  # Avoid division by zero
            else:
                avg_perf[mtype] = 0.0

        # Apply memory factor to blend with previous weights
        total_weight = sum(avg_perf.values())

        if total_weight > 0:
            for mtype in self.weights:
                # New weight is a blend of old weight and new performance
                new_weight = (1.0 - self.memory_factor) * (
                    avg_perf[mtype] / total_weight
                ) + self.memory_factor * self.weights[mtype]

                # Apply minimum weight constraint
                self.weights[mtype] = max(self.min_weight, new_weight)

            # Add exploration factor for diversity
            if self.exploration_factor > 0:
                for mtype in self.weights:
                    self.weights[mtype] += self.exploration_factor

            # Normalize weights to sum to 1
            weight_sum = sum(self.weights.values())
            if weight_sum > 0:
                for mtype in self.weights:
                    self.weights[mtype] /= weight_sum

    def get_weighted_prediction(self, predictions):
        """Combine predictions using the current weights."""
        valid = [
            (self.weights[mtype], np.array(preds))
            for mtype, preds in predictions.items()
            if preds is not None
        ]
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
                cycles = yaml.safe_load(f)
            if len(cycles) > max_cycles:
                cycles = cycles[-max_cycles:]
                with open(filename, "w") as f:
                    yaml.safe_dump(cycles, f)
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


# Add import for ensemble_utils

# ...existing imports...
from src.utils.training_optimizer import get_training_optimizer

# Initialize training optimizer
training_optimizer = get_training_optimizer()


def get_model_prediction(
    mtype,
    submodel_params,
    X_train,
    y_train,
    X_test,
    horizon,
    unified_lookback,
    feature_cols,
):
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

    if mtype in ["lstm", "rnn"]:
        arch_params = {
            "units_per_layer": submodel_params[mtype].get("units_per_layer", [64, 32]),
            "use_batch_norm": submodel_params[mtype].get("use_batch_norm", False),
            "l2_reg": submodel_params[mtype].get("l2_reg", 0.0),
            "use_attention": submodel_params[mtype].get("use_attention", True),
            "attention_type": submodel_params[mtype].get("attention_type", "dot"),
            "attention_size": submodel_params[mtype].get("attention_size", 64),
            "attention_heads": submodel_params[mtype].get("attention_heads", 1),
            "attention_dropout": submodel_params[mtype].get("attention_dropout", 0.0),
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
            architecture_params=arch_params,
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
            random_state=42,
        )
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        y_tr_flat = y_train[:, 0]
        rf.fit(X_tr_flat, y_tr_flat)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)
        preds_1d = rf.predict(X_te_flat)
        preds = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
    elif mtype == "xgboost":
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=submodel_params[mtype]["n_est"],
            learning_rate=submodel_params[mtype]["lr"],
            max_depth=submodel_params[mtype].get("max_depth", 6),
        )
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        y_tr_flat = y_train[:, 0]
        xgb_model.fit(X_tr_flat, y_tr_flat)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)
        preds_1d = xgb_model.predict(X_te_flat)
        preds = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
    elif mtype == "ltc":
        # Import ltc model builder
        from src.models.ltc_model import build_ltc_model

        model = build_ltc_model(
            num_features=len(feature_cols),
            horizon=horizon,
            learning_rate=submodel_params[mtype]["lr"],
            loss_function=submodel_params[mtype]["loss_function"],
            lookback=unified_lookback,
            units=submodel_params[mtype]["units"],
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
            verbose=0,
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
                return self.trial.suggest_int(
                    param_name, param_range[0], param_range[1], step=step if step else 1
                )
            else:
                return self.trial.suggest_int(
                    param_name, max(1, param_default // 2), param_default * 2
                )
        elif param_type == "float":
            if param_range:
                return self.trial.suggest_float(
                    param_name, param_range[0], param_range[1], log=use_log
                )
            else:
                # Auto-determine log scale for small values
                auto_log = use_log or param_default < 0.01
                return self.trial.suggest_float(
                    param_name,
                    max(1e-6, param_default / 10 if auto_log else param_default / 2),
                    param_default * 10 if auto_log else param_default * 2,
                    log=auto_log,
                )
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
                "lr": self.trial.suggest_float(
                    f"{model_type}_lr", 1e-5, 1e-2, log=True
                ),
                "dropout": self.trial.suggest_float(f"{model_type}_dropout", 0.0, 0.5),
                "lookback": self.trial.suggest_int(f"{model_type}_lookback", 7, 90),
                "units_per_layer": [
                    self.trial.suggest_categorical(
                        f"{model_type}_units", [32, 64, 128, 256, 512]
                    )
                ],
                "loss_function": self.trial.suggest_categorical(
                    f"{model_type}_loss",
                    ["mean_squared_error", "mean_absolute_error", "huber_loss"],
                ),
                "epochs": self.trial.suggest_int(f"{model_type}_epochs", 5, 50),
                "batch_size": self.trial.suggest_categorical(
                    f"{model_type}_batch_size", [16, 32, 64, 128]
                ),
            }
        elif model_type == "random_forest":
            return {
                "n_est": self.trial.suggest_int("rf_n_est", 50, 500),
                "mdepth": self.trial.suggest_int("rf_mdepth", 3, 25),
                "min_samples_split": self.trial.suggest_int(
                    "rf_min_samples_split", 2, 20
                ),
                "min_samples_leaf": self.trial.suggest_int(
                    "rf_min_samples_leaf", 1, 10
                ),
            }
        elif model_type == "xgboost":
            return {
                "n_est": self.trial.suggest_int("xgb_n_est", 50, 500),
                "lr": self.trial.suggest_float("xgb_lr", 1e-4, 0.5, log=True),
                "max_depth": self.trial.suggest_int("xgb_max_depth", 3, 12),
                "subsample": self.trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": self.trial.suggest_float("xgb_colsample", 0.5, 1.0),
            }
        elif model_type == "ltc":
            return {
                "lr": self.trial.suggest_float("ltc_lr", 1e-5, 1e-2, log=True),
                "units": self.trial.suggest_int("ltc_units", 32, 512),
                "lookback": self.trial.suggest_int("ltc_lookback", 7, 90),
                "loss_function": self.trial.suggest_categorical(
                    "ltc_loss",
                    ["mean_squared_error", "mean_absolute_error", "huber_loss"],
                ),
                "epochs": self.trial.suggest_int("ltc_epochs", 5, 50),
                "batch_size": self.trial.suggest_categorical(
                    "ltc_batch_size", [16, 32, 64, 128]
                ),
            }
        elif model_type == "tabnet":
            # TabNet architecture parameters with wider ranges
            return {
                "n_d": self.trial.suggest_int("n_d", 8, 256, log=True),
                "n_a": self.trial.suggest_int("n_a", 8, 256, log=True),
                "n_steps": self.trial.suggest_int("n_steps", 1, 15),
                "gamma": self.trial.suggest_float("gamma", 0.5, 3.0),
                "lambda_sparse": self.trial.suggest_float(
                    "lambda_sparse", 1e-7, 1e-1, log=True
                ),
                # Optimizer parameters
                "optimizer_lr": self.trial.suggest_float(
                    "optimizer_lr", 1e-5, 5e-1, log=True
                ),
                # Training parameters
                "batch_size": self.trial.suggest_categorical(
                    "batch_size", [128, 256, 512, 1024, 2048, 4096]
                ),
                "virtual_batch_size": self.trial.suggest_int(
                    "virtual_batch_size", 16, 1024, log=True
                ),
                "momentum": self.trial.suggest_float("momentum", 0.005, 0.5),
                "max_epochs": self.trial.suggest_int("max_epochs", 50, 500),
                "patience": self.trial.suggest_int("patience", 5, 50),
                # Convert optimizer_lr to optimizer_params dict
                "optimizer_params": {
                    "lr": self.trial.suggest_float("optimizer_lr", 1e-5, 5e-1, log=True)
                },
            }
        elif model_type == "cnn":
            # Calculate epochs based on multiplier
            base_epochs = self.trial.suggest_int("cnn_base_epochs", 1, 10)
            actual_epochs = max(
                1, int(base_epochs * st.session_state.get("epochs_multiplier", 1.0))
            )

            # Get complexity multiplier
            complexity_multiplier = st.session_state.get("tuning_multipliers", {}).get(
                "complexity_multiplier", 1.0
            )

            # Adjust ranges based on complexity multiplier
            max_filters = int(256 * complexity_multiplier)

            return {
                "num_conv_layers": self.trial.suggest_int("cnn_num_conv_layers", 1, 5),
                "num_filters": self.trial.suggest_int(
                    "cnn_num_filters", 16, max_filters, log=True
                ),
                "kernel_size": self.trial.suggest_int("cnn_kernel_size", 2, 7),
                "stride": self.trial.suggest_int("cnn_stride", 1, 2),
                "dropout_rate": self.trial.suggest_float("cnn_dropout_rate", 0.0, 0.5),
                "activation": self.trial.suggest_categorical(
                    "cnn_activation", ["relu", "leaky_relu", "elu"]
                ),
                "use_adaptive_pooling": self.trial.suggest_categorical(
                    "cnn_use_adaptive_pooling", [True, False]
                ),
                "fc_layers": [
                    self.trial.suggest_int("cnn_fc_layer_1", 32, 256, log=True),
                    self.trial.suggest_int("cnn_fc_layer_2", 16, 128, log=True),
                ],
                "lookback": self.trial.suggest_int("cnn_lookback", 7, 90),
                "lr": self.trial.suggest_float("cnn_lr", 1e-5, 1e-2, log=True),
                "batch_size": self.trial.suggest_categorical(
                    "cnn_batch_size", [16, 32, 64, 128]
                ),
                "epochs": actual_epochs,
                "early_stopping_patience": self.trial.suggest_int(
                    "cnn_early_stopping_patience", 3, 10
                ),
            }
        return {}


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
            if platform.system() == "Windows":
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
                time.sleep(
                    retry_delay * (1 + random.random())
                )  # Randomized exponential backoff
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
            if platform.system() == "Windows":
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
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff
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
            from src.training.incremental_learning import (
                ModelRegistry,
            )  # Fixed typo: ModelRegistryry → ModelRegistry

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
            "use_batch_norm": best_params.get("use_batch_norm", False),
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
                architecture_params=architecture_params,
            )

            # Performance metrics
            metrics = {
                "rmse": trial.user_attrs.get("rmse", 0),
                "mape": trial.user_attrs.get("mape", 0),
                "objective_value": trial.value,
                "trial_number": trial.number,
                "timestamp": datetime.now().isoformat(),
            }

            # Register the model
            model_id = registry.register_model(
                model=model,
                model_type=model_type,
                ticker=ticker,
                timeframe=timeframe,
                metrics=metrics,
                hyperparams=best_params,
                tags=[
                    "optuna",
                    "best_trial",
                    f"cycle_{study.user_attrs.get('cycle', 0)}",
                ],
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
def tune_for_combo(
    ticker, timeframe, range_cat="all", n_trials=None, cycle=1, tuning_multipliers=None
):
    """Run hyperparameter optimization for a specific ticker-timeframe combination."""
    from config.config_loader import (
        DB_DIR,
        N_STARTUP_TRIALS,
        TUNING_TRIALS_PER_CYCLE_max,
    )
    from src.tuning.progress_helper import update_cycle_metrics, update_progress_in_yaml
    from src.utils.utils import adaptive_memory_clean

    # Run diagnostics first to help troubleshoot DB issues
    diagnose_db_folder_issues()

    # Clean memory before starting
    adaptive_memory_clean("large")

    # Get tuning multipliers from session state if not provided
    if tuning_multipliers is None:
        tuning_multipliers = st.session_state.get(
            "tuning_multipliers",
            {
                "trials_multiplier": 1.0,
                "epochs_multiplier": 1.0,
                "timeout_multiplier": 1.0,
                "complexity_multiplier": 1.0,
            },
        )

    # Determine trials dynamically between min and max using multiplier
    if n_trials is None:
        # Use random value between min and max for automatic exploration
        import random

        # Adjust trials based on multiplier
        trials_multiplier = tuning_multipliers.get("trials_multiplier", 1.0)
        adjusted_min = max(5, int(TUNING_TRIALS_PER_CYCLE_min * trials_multiplier))
        adjusted_max = max(
            adjusted_min + 5, int(TUNING_TRIALS_PER_CYCLE_max * trials_multiplier)
        )

        n_trials = random.randint(adjusted_min, adjusted_max)
        logger.info(
            f"Auto-selecting {n_trials} trials for this cycle (between {adjusted_min} and {adjusted_max})"
        )

    # Initialize progress to show tuning has started
    initial_progress = {
        "current_trial": 0,
        "total_trials": n_trials,
        "current_rmse": None,
        "current_mape": None,
        "cycle": cycle,
        "trial_progress": 0.0,
        "timestamp": time.time(),
        "ticker": ticker,
        "timeframe": timeframe,
    }
    update_progress_in_yaml(initial_progress)

    # Create a unique study name with timestamp to avoid conflict with other processes
    timestamp = int(time.time())
    study_name = f"{ticker}_{timeframe}_{range_cat}_cycle{cycle}_{timestamp}"

    # Ensure DB directory exists
    os.makedirs(DB_DIR, exist_ok=True)

    # Create proper SQLite URL with absolute path
    db_path = os.path.join(DB_DIR, f"metatune_{study_name}.db")
    storage_name = f"sqlite:///{os.path.abspath(db_path)}"

    # Apply complexity multiplier to startup trials
    if tuning_multipliers.get("complexity_multiplier", 1.0) < 1.0:
        # Reduce startup trials for quick tuning
        adjusted_startup_trials = max(
            2,
            int(
                N_STARTUP_TRIALS * tuning_multipliers.get("complexity_multiplier", 1.0)
            ),
        )
    else:
        adjusted_startup_trials = N_STARTUP_TRIALS

    try:
        study = create_study(
            study_name=study_name,
            storage_name=storage_name,
            direction="minimize",
            n_startup_trials=adjusted_startup_trials,
        )
    except Exception as e:
        logger.error(f"Failed to create study: {e}")
        # Fall back to in-memory study as last resort
        logger.info("Creating in-memory study as fallback")
        import optuna

        study = optuna.create_study(
            study_name=f"{study_name}_fallback",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=adjusted_startup_trials
            ),
        )

    study.set_user_attr("cycle", cycle)
    study.set_user_attr("n_trials", n_trials)
    study.set_user_attr("tuning_multipliers", tuning_multipliers)  # Store for reference
    study.set_user_attr("ticker", ticker)
    study.set_user_attr("timeframe", timeframe)

    # Safely check for best trial using a try-except block
    try:
        if study.best_trial:
            logger.info(
                f"Current best: {study.best_value:.6f} (Trial {study.best_trial.number})"
            )
    except (ValueError, AttributeError, RuntimeError):
        logger.info(f"No trials completed yet for study {study_name}")

    # Define callbacks
    stop_callback = StopStudyCallback()
    progress_callback = create_progress_callback(cycle=cycle)

    # Set timeout based on multiplier
    timeout_seconds = None
    if tuning_multipliers.get("timeout_multiplier", 1.0) > 0:
        base_timeout_seconds = 3600  # 1 hour baseline
        timeout_seconds = int(
            base_timeout_seconds * tuning_multipliers.get("timeout_multiplier", 1.0)
        )

    # Run optimization
    try:
        study.optimize(
            robust_objective_wrapper(
                lambda trial: ensemble_with_walkforward_objective(
                    trial, ticker, timeframe, range_cat
                )
            ),
            n_trials=n_trials,
            callbacks=[progress_callback, stop_callback],
            timeout=timeout_seconds,
        )
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
    finally:
        # Clean memory after optimization
        adaptive_memory_clean("large")

        # Update cycle metrics with summary information
        try:
            # Get best trial info
            best_trial_info = None
            try:
                if study.best_trial:
                    best_trial = study.best_trial
                    best_trial_info = {
                        "trial_number": best_trial.number,
                        "rmse": best_trial.user_attrs.get("rmse", None),
                        "mape": best_trial.user_attrs.get("mape", None),
                        "model_type": best_trial.params.get("model_type", "unknown"),
                        "value": best_trial.value,
                    }
            except (ValueError, AttributeError, RuntimeError):
                logger.warning("No best trial available for study")

            # Count completed and pruned trials
            completed_trials = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            pruned_trials = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            )

            # Prepare cycle summary
            cycle_summary = {
                "cycle": cycle,
                "ticker": ticker,
                "timeframe": timeframe,
                "total_trials": len(study.trials),
                "completed_trials": completed_trials,
                "pruned_trials": pruned_trials,
                "best_trial": best_trial_info,
                "timestamp": datetime.now().isoformat(),
            }

            # Save cycle metrics
            update_cycle_metrics(cycle_summary, cycle_num=cycle)

            # Update final progress to indicate completion
            final_progress = {
                "current_trial": len(study.trials),
                "total_trials": n_trials,
                "current_rmse": best_trial_info["rmse"] if best_trial_info else None,
                "current_mape": best_trial_info["mape"] if best_trial_info else None,
                "cycle": cycle,
                "trial_progress": 1.0,  # Set to 100% complete
                "timestamp": time.time(),
                "ticker": ticker,
                "timeframe": timeframe,
                "status": "completed",
            }
            update_progress_in_yaml(final_progress)

        except Exception as e:
            logger.error(f"Error updating cycle metrics: {e}")

    # Register the best model if study wasn't interrupted


def create_progress_callback(cycle=1):
    """Create a callback that updates progress file with thread-safety."""
    from src.tuning.progress_helper import (
        update_trial_info_in_yaml,
    )
    from src.utils.utils import adaptive_memory_clean

    def progress_callback(study, trial):
        """Update progress when a trial completes."""
        # Only process when trial completes
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Get metrics from trial
        rmse = trial.user_attrs.get("rmse", float("inf"))
        mape = trial.user_attrs.get("mape", float("inf"))

        # Determine ticker and timeframe from study
        ticker = study.user_attrs.get("ticker", "unknown")
        timeframe = study.user_attrs.get("timeframe", "unknown")

        # Get best metrics if available
        try:
            if study.best_trial:
                best_rmse = study.best_trial.user_attrs.get("rmse", float("inf"))
                best_mape = study.best_trial.user_attrs.get("mape", float("inf"))
                best_model = study.best_trial.params.get("model_type", "unknown")
            else:
                best_rmse = rmse
                best_mape = mape
                best_model = trial.params.get("model_type", "unknown")
        except (ValueError, AttributeError, RuntimeError):
            best_rmse = rmse
            best_mape = mape
            best_model = trial.params.get("model_type", "unknown")

        # Prepare progress data
        progress_data = {
            "ticker": ticker,
            "timeframe": timeframe,
            "current_trial": trial.number,
            "total_trials": study.user_attrs.get(
                "n_trials", TUNING_TRIALS_PER_CYCLE_max
            ),
            "current_rmse": float(rmse),
            "current_mape": float(mape),
            "best_rmse": float(best_rmse),
            "best_mape": float(best_mape),
            "best_model": best_model,
            "cycle": cycle,
            "trial_progress": trial.number
            / study.user_attrs.get("n_trials", TUNING_TRIALS_PER_CYCLE_max),
            "timestamp": time.time(),
        }

        # Also record the detailed trial information separately
        trial_info = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
            "model_type": trial.params.get("model_type", "unknown"),
            "rmse": rmse,
            "mape": mape,
            "timestamp": datetime.now().isoformat(),
        }

        # Update progress and trial info using thread-safe operations
        update_progress_in_yaml(progress_data)
        update_trial_info_in_yaml(TESTED_MODELS_FILE, trial_info)

        # Save to session state if available
        if "streamlit" in sys.modules and hasattr(st, "session_state"):
            # Store summary in session state for UI
            if "trial_logs" not in st.session_state:
                st.session_state["trial_logs"] = []

            st.session_state["trial_logs"].append(trial_info)

            # Keep only the most recent 100 trials in memory
            if len(st.session_state["trial_logs"]) > 100:
                st.session_state["trial_logs"] = st.session_state["trial_logs"][-100:]

        # Clean memory periodically
        if trial.number % 5 == 0:
            adaptive_memory_clean("small")
        if trial.number % 20 == 0:
            adaptive_memory_clean("medium")

        print(f"TUNING-DEBUG: Completed trial {trial.number}")

    return progress_callback


def start_tuning_process(ticker, timeframe, multipliers=None):
    """
    Start the tuning process by calling tune_for_combo.
    Sets session state to indicate tuning is in progress.

    Args:
        ticker (str): The ticker symbol
        timeframe (str): The timeframe (e.g. "1d")
        multipliers (dict, optional): Dictionary of tuning multipliers
    """
    import time
    from datetime import datetime

    print(f"TUNING-DEBUG: Starting tuning process for {ticker} {timeframe}")

    # Import needed modules here to avoid circular imports
    from src.tuning.progress_helper import (
        read_tuning_status,
        write_tuning_status,
    )

    # Run diagnostics to check DB folder - explicit call
    diagnose_db_folder_issues()

    # Check if tuning is already running
    current_status = read_tuning_status()
    if current_status.get("is_running", False):
        logger.warning(
            f"Tuning already in progress for {current_status.get('ticker')} ({current_status.get('timeframe')})"
        )
        # Update session state to reflect this
        st.session_state["tuning_in_progress"] = True
        return None

    # Reset the stop event to make sure we're not in a stopped state
    set_stop_requested(False)

    # Set session state to indicate tuning is in progress
    st.session_state["tuning_in_progress"] = True
    st.session_state["tuning_start_time"] = time.time()

    # Store multipliers in session state if provided
    if multipliers is not None:
        st.session_state["tuning_multipliers"] = multipliers
        # Calculate correct trial count
        trials_multiplier = multipliers.get("trials_multiplier", 1.0)
        # Get startup trials from config
        try:
            adjusted_trials = max(100, int(N_STARTUP_TRIALS * trials_multiplier))
        except:
            adjusted_trials = 5000  # Default if can't access N_STARTUP_TRIALS
    else:
        try:
            adjusted_trials = N_STARTUP_TRIALS
        except:
            adjusted_trials = 5000  # Default if can't access N_STARTUP_TRIALS

    # Update tuning status file first
    status_written = write_tuning_status(
        {
            "ticker": ticker,
            "timeframe": timeframe,
            "is_running": True,
            "start_time": time.time(),
            "timestamp": datetime.now().isoformat(),
        }
    )

    if not status_written:
        logger.error("Failed to write tuning status - may cause coordination issues")

    # Ensure progress file is updated with correct total_trials
    progress_written = update_progress_in_yaml(
        {
            "ticker": ticker,
            "timeframe": timeframe,
            "total_trials": adjusted_trials,
            "timestamp": time.time(),
            "current_trial": 0,
        }
    )

    if not progress_written:
        logger.error(
            "Failed to update progress file - UI may not reflect tuning progress"
        )

    try:
        # Start the actual tuning process with the correct trial count
        # Use the study_manager version of tune_for_combo
        logger.info(
            f"Starting tuning for {ticker} ({timeframe}) with {adjusted_trials} trials"
        )
        print("TUNING-DEBUG: About to start tuning with study_manager")

        # Define default metric weights
        metric_weights = {"rmse": 1.0, "mape": 0.5, "da": 0.3}

        # Call the tune_for_combo that uses study_manager
        result = tune_for_combo(
            ticker,
            timeframe,
            range_cat="all",
            n_trials=adjusted_trials,
            cycle=1,
            tuning_multipliers=multipliers,
            metric_weights=metric_weights,
        )
        return result
    except Exception as e:
        # Reset tuning status on error
        st.session_state["tuning_in_progress"] = False
        write_tuning_status(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "is_running": False,
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.error(f"Error in tuning process: {e}")
        traceback.print_exc()
        return None
    finally:
        # Ensure status is updated when tuning completes (successfully or with error)
        if not is_stop_requested():  # Only if it wasn't explicitly stopped
            st.session_state["tuning_in_progress"] = False
            write_tuning_status(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "is_running": False,
                    "end_time": time.time(),
                    "timestamp": datetime.now().isoformat(),
                }
            )


def stop_tuning_process():
    """
    Stop the tuning process by signaling the stop event.
    Updates session state to indicate tuning has stopped.
    """
    # Signal the stop event
    set_stop_requested(True)

    # Update session state
    st.session_state["tuning_in_progress"] = False

    # Update tuning status using progress_helper
    from src.tuning.progress_helper import write_tuning_status

    write_tuning_status(
        {"is_running": False, "stopped_manually": True, "stop_time": time.time()}
    )

    # Clear stored multipliers to use defaults next time
    if "tuning_multipliers" in st.session_state:
        del st.session_state["tuning_multipliers"]

    logger.info("Tuning process stop requested")
    return True


__all__ = [
    "start_tuning_process",
    "stop_tuning_process",
    "tune_for_combo",
    "register_best_model",
    "get_model_registry",
]

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def robust_objective_wrapper(objective_fn):
    """
    Wrap the Optuna objective function to provide better error handling.
    """

    def wrapped_objective(trial):
        try:
            print(f"Starting trial {trial.number}")
            result = objective_fn(trial)
            print(f"Trial {trial.number} completed with result: {result}")
            return result
        except optuna.exceptions.TrialPruned:
            print(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            import traceback

            traceback.print_exc()
            # If this is a critical error, we might want to stop the study
            if isinstance(e, (MemoryError, KeyboardInterrupt, SystemExit)):
                print(f"Critical error in trial {trial.number} - will stop study")
                from src.tuning.progress_helper import set_stop_requested

                set_stop_requested(True)
                return float("inf")
            return 1.0e6

    return wrapped_objective


# Improve the create_resilient_study function to better handle paths


# Add this after imports but before using SQLite
def check_sqlite_availability():
    """Verify SQLite is available and working properly"""
    try:
        import sqlite3

        # Test create an in-memory database
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()
        print("SQLite is available and working")
        return True
    except ImportError:
        print(
            "ERROR: SQLite (sqlite3) module not available! Install with 'pip install pysqlite3'"
        )
        return False
    except Exception as e:
        print(f"ERROR: SQLite test failed: {e}")
        return False


# Add this after directory definitions
def verify_directories():
    """Verify all required directories exist and are writable"""
    required_dirs = {
        "DATA_DIR": DATA_DIR,
        "DB_DIR": DB_DIR,
        "MODELS_DIR": MODELS_DIR,
        "HYPERPARAMS_DIR": os.path.join(MODELS_DIR, "Hyperparameters"),
    }

    for name, directory in required_dirs.items():
        # Check directory exists
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"ERROR: Failed to create {name} directory ({directory}): {e}")
                continue

        # Check directory is writable
        try:
            test_file = os.path.join(directory, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"✓ {name} directory is writable: {directory}")
        except Exception as e:
            print(f"ERROR: {name} directory is not writable ({directory}): {e}")


# Call these functions early in your main function or script execution
def initialize_tuning_environment():
    """Initialize the environment for tuning with proper checks"""
    check_sqlite_availability()
    verify_directories()


# Call the initialize function when imported
if __name__ != "__main__":
    # Only run initialization when imported as a module (not when run directly)
    initialize_tuning_environment()


# Add this function at the top level of the file
def diagnose_db_folder_issues():
    """Diagnose issues with the DB folder and SQLite connectivity"""
    import os

    print("\n--- DIAGNOSING DB FOLDER ISSUES ---")

    # Get the DB_DIR path safely
    try:
        from config.config_loader import DB_DIR

        db_dir = DB_DIR
    except ImportError:
        # Fallback to construct DB_DIR if import fails
        try:
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(current_file))
            project_root = os.path.dirname(src_dir)
            db_dir = os.path.join(project_root, "data", "DB")
        except:
            db_dir = "data/DB"  # Ultimate fallback

    # 1. Check DB_DIR path and existence
    print(f"DB_DIR path: {db_dir}")
    if os.path.exists(db_dir):
        print("✓ DB_DIR exists")

        # Check if it's writable
        try:
            test_file = os.path.join(db_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print("✓ DB_DIR is writable")
        except Exception as e:
            print(f"❌ ERROR: DB_DIR is not writable: {e}")
            print(
                "   Try running the application with administrator privileges or check folder permissions"
            )
    else:
        print("❌ ERROR: DB_DIR does not exist")
        print("   Attempting to create DB_DIR...")
        try:
            os.makedirs(db_dir, exist_ok=True)
            print(f"   Successfully created DB_DIR: {db_dir}")
        except Exception as e:
            print(f"   Failed to create DB_DIR: {e}")

    # 2. Check SQLite functionality
    print("\nChecking SQLite functionality:")
    try:
        import sqlite3

        print(f"✓ SQLite version: {sqlite3.sqlite_version}")

        # Try creating a test database in the DB_DIR
        test_db_path = os.path.join(db_dir, "test_connection.db")
        print(f"Attempting to create test database at: {test_db_path}")

        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, test_value TEXT)"
        )
        cursor.execute(
            "INSERT INTO test_table (test_value) VALUES (?)", ("Test successful",)
        )
        conn.commit()

        # Verify data was written
        cursor.execute("SELECT test_value FROM test_table LIMIT 1")
        result = cursor.fetchone()
        print(f"✓ Successfully wrote and read from test database: {result[0]}")

        # Close and clean up
        conn.close()

        try:
            os.remove(test_db_path)
            print("✓ Cleaned up test database")
        except:
            print("  (Note: Could not remove test database, but it's not critical)")
    except ImportError:
        print("❌ ERROR: SQLite3 module not available")
        print("   Install with: pip install pysqlite3")
    except Exception as e:
        print(f"❌ ERROR with SQLite: {e}")
        print("   Check SQLite installation and permissions")

    # 3. Check actual database files
    print("\nChecking existing database files:")
    try:
        db_files = [f for f in os.listdir(db_dir) if f.endswith(".db")]
        if db_files:
            print(f"Found {len(db_files)} database files:")
            for db_file in db_files:
                full_path = os.path.join(db_dir, db_file)
                size = os.path.getsize(full_path)
                print(f"  - {db_file} ({size} bytes)")

                # Try to open the database and check for trials table
                try:
                    conn = sqlite3.connect(full_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='trials'"
                    )
                    if cursor.fetchone():
                        cursor.execute("SELECT COUNT(*) FROM trials")
                        count = cursor.fetchone()[0]
                        print(f"    Contains {count} trials")
                    else:
                        print(
                            "    No trials table found (might not be an Optuna database)"
                        )
                    conn.close()
                except Exception as e:
                    print(f"    Could not inspect database: {e}")
        else:
            print("No database files found in DB_DIR")
    except Exception as e:
        print(f"Error checking database files: {e}")

    print("\n--- END OF DIAGNOSTICS ---\n")


# Modify create_resilient_study function to return detailed errors


# Modify tune_for_combo to add diagnostics
def tune_for_combo(
    ticker, timeframe, range_cat="all", n_trials=None, cycle=1, tuning_multipliers=None
):
    """Run hyperparameter optimization for a specific ticker-timeframe combination."""
    from src.utils.utils import adaptive_memory_clean

    # Run diagnostics first to help troubleshoot DB issues
    diagnose_db_folder_issues()

    # Clean memory before starting
    adaptive_memory_clean("large")

    # Rest of the function remains the same
    # ...existing code...


# Modify start_tuning_process to call diagnose_db_folder_issues
def start_tuning_process(ticker, timeframe, multipliers=None):
    """Start the tuning process by calling tune_for_combo."""

    print(f"TUNING-DEBUG: Starting tuning process for {ticker} {timeframe}")

    # Run diagnostics to check DB folder
    diagnose_db_folder_issues()


if __name__ != "__main__":
    # Run diagnostics on import
    try:
        print("Running DB diagnostics on module import")
        diagnose_db_folder_issues()
    except Exception as e:
        print(f"Error running diagnostics: {e}")

# ...existing code...


def tune_for_combo(
    ticker,
    timeframe,
    range_cat="all",
    n_trials=None,
    cycle=1,
    tuning_multipliers=None,
    metric_weights=None,
):
    """
    Run hyperparameter optimization for all models in parallel with proper resource sharing.

    Args:
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category ("all" or specific range)
        n_trials: Number of trials per model (or None for auto)
        cycle: Tuning cycle number
        tuning_multipliers: Dict with tuning multipliers
        metric_weights: Dict with metric weights

    Returns:
        Dict with tuning results for all models
    """
    from config.config_loader import (
        ACTIVE_MODEL_TYPES,
        TUNING_TRIALS_PER_CYCLE_max,
    )
    from src.tuning.progress_helper import (
        set_stop_requested,
        update_cycle_metrics,
        update_progress_in_yaml,
    )
    from src.utils.memory_utils import adaptive_memory_clean

    # Reset stop request flag
    set_stop_requested(False)

    # Clean memory before starting
    adaptive_memory_clean("large")

    # Get tuning multipliers
    if tuning_multipliers is None:
        tuning_multipliers = st.session_state.get(
            "tuning_multipliers",
            {
                "trials_multiplier": 1.0,
                "epochs_multiplier": 1.0,
                "timeout_multiplier": 1.0,
                "complexity_multiplier": 1.0,
            },
        )

    # Determine trials dynamically
    if n_trials is None:
        trials_multiplier = tuning_multipliers.get("trials_multiplier", 1.0)
        adjusted_min = max(5, int(TUNING_TRIALS_PER_CYCLE_min * trials_multiplier))
        adjusted_max = max(
            adjusted_min + 5, int(TUNING_TRIALS_PER_CYCLE_max * trials_multiplier)
        )

        n_trials = random.randint(adjusted_min, adjusted_max)
        logger.info(
            f"Auto-selecting {n_trials} trials per model (between {adjusted_min} and {adjusted_max})"
        )

    # Initial progress information
    initial_progress = {
        "ticker": ticker,
        "timeframe": timeframe,
        "cycle": cycle,
        "total_trials": n_trials * len(ACTIVE_MODEL_TYPES),
        "completed_trials": 0,
        "current_progress": 0.0,
        "timestamp": time.time(),
    }
    update_progress_in_yaml(initial_progress)

    # Create callback for progress reporting
    progress_callback = create_progress_callback(
        cycle=cycle, ticker=ticker, timeframe=timeframe, range_cat=range_cat
    )
    callbacks = [progress_callback]

    # Add stop callback
    stop_callback = StopStudyCallback()
    callbacks.append(stop_callback)

    # Prepare study configurations
    studies_config = []

    for model_type in ACTIVE_MODEL_TYPES:
        # Create study for this model type
        study = study_manager.create_study(
            model_type, ticker, timeframe, range_cat, cycle, n_trials, metric_weights
        )

        # Create objective function
        objective = create_model_objective(
            model_type, ticker, timeframe, range_cat, metric_weights
        )

        # Add to configuration
        studies_config.append(
            {
                "component_type": model_type,
                "n_trials": n_trials,
                "objective_func": objective,
                "callbacks": callbacks.copy(),  # Use copy to avoid shared state
            }
        )

    # Run all studies with proper resource allocation
    results = study_manager.run_all_studies(studies_config)

    # Update cycle metrics with summary
    try:
        # Find best model across all studies
        best_model_type, best_params, best_value = study_manager.get_best_model()

        if best_model_type:
            # Get best trial info
            best_result = results.get(best_model_type, {})
            best_trial = best_result.get("best_trial")

            if best_trial:
                best_trial_info = {
                    "trial_number": best_trial.number,
                    "model_type": best_model_type,
                    "rmse": best_trial.user_attrs.get("rmse"),
                    "mape": best_trial.user_attrs.get("mape"),
                    "directional_accuracy": best_trial.user_attrs.get(
                        "directional_accuracy"
                    ),
                    "combined_score": best_value,
                    "params": best_params,
                }

                # Prepare cycle summary
                cycle_summary = {
                    "cycle": cycle,
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "total_trials": n_trials * len(ACTIVE_MODEL_TYPES),
                    "completed_trials": sum(
                        len(results[mt].get("study", {}).trials)
                        for mt in results
                        if "study" in results[mt]
                    ),
                    "best_trial": best_trial_info,
                    "best_model_type": best_model_type,
                    "timestamp": datetime.now().isoformat(),
                }

                # Save cycle metrics
                update_cycle_metrics(cycle_summary, cycle_num=cycle)

        else:
            logger.warning("No best model found across studies")

    except Exception as e:
        logger.error(f"Error updating cycle metrics: {e}")
        traceback.print_exc()

    # Clean memory after optimization
    adaptive_memory_clean("large")

    return results


def evaluate_model_with_walkforward(
    trial, ticker, timeframe, range_cat, model_type=None
):
    """
    Evaluate a single model using walk-forward validation.
    Modified version of evaluate_ensemble_with_walkforward that focuses on one model type.

    Args:
        trial: Optuna trial object
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category
        model_type: Type of model to evaluate

    Returns:
        Dict with evaluation metrics
    """
    # If model_type not specified, get it from trial params
    if model_type is None:
        model_type = trial.params.get("model_type")
        if model_type is None:
            # Default to whatever model type is being suggested
            model_type = trial.suggest_categorical("model_type", ACTIVE_MODEL_TYPES)

    # Store the model type in trial user attributes
    trial.set_user_attr("model_type", model_type)

    # Get model-specific hyperparameters based on model type
    params = suggest_model_hyperparameters(trial, model_type)

    # Perform walk-forward validation for this model

    metrics = perform_walkforward_validation(
        ticker, timeframe, range_cat, model_type, params
    )

    # Store metrics in trial
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    return metrics


# Function to replace create_study and create_resilient_study calls
def create_study(
    study_name,
    storage_name=None,
    direction="minimize",
    n_startup_trials=10,
    load_if_exists=True,
):
    """
    Create a new Optuna study or load an existing one.
    This is a simplified wrapper around optuna.create_study that handles common errors.

    Args:
        study_name: Name of the study
        storage_name: Database URL for storage (e.g., "sqlite:///db.sqlite3")
        direction: Direction of optimization ("minimize" or "maximize")
        n_startup_trials: Number of random trials before TPE kicks in
        load_if_exists: Whether to load existing study if it exists

    Returns:
        optuna.Study: Created or loaded study
    """
    import optuna

    # Default to TPE sampler with seed for reproducibility
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42)

    try:
        # Try to create study with specified storage
        if storage_name:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                load_if_exists=load_if_exists,
                sampler=sampler,
                direction=direction,
            )
        else:
            # In-memory study if no storage specified
            study = optuna.create_study(sampler=sampler, direction=direction)

        return study
    except (ValueError, KeyError, ImportError) as e:
        logger.warning(f"Error creating study with storage {storage_name}: {e}")
        logger.warning("Creating in-memory study as fallback")

        # Create in-memory study as fallback
        study = optuna.create_study(sampler=sampler, direction=direction)
        return study
    except Exception as e:
        logger.error(f"Unexpected error creating study: {e}")
        raise


# Create backward compatibility alias
create_resilient_study = create_study


# Define ensemble_with_walkforward_objective function
def ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat="all"):
    """
    Objective function for ensemble model evaluation using walk-forward validation.

    Args:
        trial: Optuna trial object
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category

    Returns:
        float: Combined objective score (lower is better)
    """
    # Choose a model type
    from config.config_loader import ACTIVE_MODEL_TYPES

    model_type = trial.suggest_categorical("model_type", ACTIVE_MODEL_TYPES)

    # Import study_manager modules directly
    from src.tuning.study_manager import (
        evaluate_model_with_walkforward,
        suggest_model_hyperparameters,
    )

    # Get hyperparameters for the selected model type
    params = suggest_model_hyperparameters(trial, model_type)

    # Evaluate model using walk-forward validation
    metrics = evaluate_model_with_walkforward(
        trial, ticker, timeframe, range_cat, model_type
    )

    # Calculate composite score (rmse + 0.5*mape)
    score = metrics.get("rmse", float("inf")) + 0.5 * metrics.get("mape", float("inf"))

    return score


# Import needed function from study_manager
from src.tuning.study_manager import suggest_model_hyperparameters
