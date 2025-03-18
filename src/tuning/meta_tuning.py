"""
Implements an Optuna-based hyperparameter search with ensemble approach and walk-forward.
Logs detailed trial information live to session state and YAML files:
- progress.yaml for current progress,
- best_params.yaml when thresholds are met,
- tested_models.yaml with details for each trial,
- cycle_metrics.yaml with cycle-level summaries.
"""

# Standard library imports - Make sure these are at the very top
import os
import sys
import traceback
import logging
import platform
import random
import signal
import time
from datetime import datetime
import warnings

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add project root to sys.path for absolute imports
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from config
from config import DATA_DIR, MODELS_DIR

# Define log file paths - FIXED LOCATIONS IN ROOT DATA DIR
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

# Data directories and file paths
DB_DIR = os.path.join(DATA_DIR, "DB")
os.makedirs(DB_DIR, exist_ok=True)

# Third-party imports
import numpy as np
import optuna
import streamlit as st
import yaml
import torch

# Set optimized environment variables early
from src.utils.env_setup import setup_tf_environment

# Import GPU memory management tools
from src.utils.gpu_memory_manager import GPUMemoryManager
from src.utils.gpu_memory_management import set_performance_mode, get_memory_info, get_gpu_utilization

# Get mixed precision setting from config first
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from config.config_loader import get_value
    use_mixed_precision = get_value("hardware.use_mixed_precision", False)
except ImportError:
    use_mixed_precision = False

env_vars = setup_tf_environment(memory_growth=True, mixed_precision=use_mixed_precision)

# Import study_manager for unified model tuning
from src.tuning.study_manager import (
    StudyManager,
    create_model_objective,
    evaluate_model_with_walkforward,
    study_manager,
    suggest_model_hyperparameters,
)

# Import StopStudyCallback from callbacks
from src.training.callbacks import StopStudyCallback
from src.training.walk_forward import perform_walkforward_validation

# Import pruning settings from config
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
from src.tuning.progress_helper import (
    update_progress_in_yaml,
    is_stop_requested,
    set_stop_requested,
    update_cycle_metrics,
    update_trial_info_in_yaml,
    read_tuning_status,
    write_tuning_status,
)
from src.utils.threadsafe import safe_read_yaml, safe_write_yaml
from src.utils.training_optimizer import get_training_optimizer
from src.utils.utils import adaptive_memory_clean

# Initialize training optimizer and GPU memory manager
training_optimizer = get_training_optimizer()
gpu_memory_manager = GPUMemoryManager()

# Ignore future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Session state initialization
if "trial_logs" not in st.session_state:
    st.session_state["trial_logs"] = []

# Signal handler for non-Windows platforms
if sys.platform != "win32":
    def signal_handler(sig, frame):
        print("\nManual stop requested. Exiting tuning loop.")
        set_stop_requested(True)
    signal.signal(signal.SIGINT, signal_handler)

# Initialize GPU for optimal performance
def initialize_gpu_for_tuning():
    """Initialize GPU for optimal performance during tuning."""
    try:
        logger.info("Initializing GPU for hyperparameter tuning...")
        
        # Initialize GPU memory manager
        gpu_memory_manager.initialize()
        
        # Warm up the GPU to improve initial performance
        peak_util = gpu_memory_manager.warmup_gpu(intensity=0.7)
        logger.info(f"GPU warmed up to {peak_util}% utilization")
        
        # Log initial memory state
        mem_info = get_memory_info()
        logger.info(f"Initial GPU memory: {mem_info.get('free_mb', 0)} MB free")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing GPU: {e}")
        return False

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

        if (total_weight > 0):
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
            if cycles and len(cycles) > max_cycles:
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
        """Returns the model-building function."""
        from src.models.model import build_model_by_type
        return build_model_by_type

    @staticmethod
    def get_data_fetcher():
        """Returns the data fetching function."""
        from src.data.data import fetch_data
        return fetch_data

    @staticmethod
    def get_feature_engineering():
        """Returns the feature engineering function."""
        from src.features.features import feature_engineering_with_params
        return feature_engineering_with_params

    @staticmethod
    def get_scaling_function():
        """Returns the data scaling function."""
        from src.data.preprocessing import scale_data
        return scale_data

    @staticmethod
    def get_sequence_function():
        """Returns the sequence creation function."""
        from src.data.preprocessing import create_sequences
        return create_sequences

    @staticmethod
    def get_optuna_sampler():
        """Returns the Optuna sampler."""
        import optuna
        return optuna.samplers.TPESampler

    @staticmethod
    def get_evaluation_function():
        """Returns the prediction evaluation function."""
        from src.models.model import evaluate_predictions
        return evaluate_predictions

    @staticmethod
    def get_cnn_model():
        """Returns the CNN price predictor class."""
        from src.models.cnn_model import CNNPricePredictor
        return CNNPricePredictor


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
    elif mtype == "nbeats":
        # Import nbeats model builder
        from src.models.nbeats_model import build_nbeats_model
        
        # Parse NBEATS parameters
        nbeats_params = submodel_params[mtype]
        
        # Apply optimizer settings
        if "batch_size" not in nbeats_params:
            nbeats_params["batch_size"] = model_config.get("batch_size", 32)
            
        # Create model with optimized parameters
        model = build_nbeats_model(
            lookback=nbeats_params.get("lookback", unified_lookback),
            horizon=horizon,
            num_features=len(feature_cols),
            learning_rate=nbeats_params.get("lr", 0.001),
            layer_width=nbeats_params.get("layer_width", 256),
            num_blocks=nbeats_params.get("num_blocks", [3, 3]),
            num_layers=nbeats_params.get("num_layers", [4, 4]),
            thetas_dim=nbeats_params.get("thetas_dim", 10),
            include_price_specific_stack=nbeats_params.get("include_price_specific_stack", True),
            dropout_rate=nbeats_params.get("dropout_rate", 0.1),
            use_batch_norm=nbeats_params.get("use_batch_norm", True),
        )
        
        # Train the model
        batch_size = nbeats_params.get("batch_size", 32)
        epochs = nbeats_params.get("epochs", 10)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Make predictions
        preds = model.predict(X_test)
        return preds
    elif mtype == "tabnet":
        # Import TabNet module - using lazy import pattern
        try:
            # Try to import PyTorch TabNet implementation first
            from src.models.tabnet_model import TabNetRegressor
            
            # Get TabNet parameters
            tabnet_params = submodel_params[mtype]
            
            # Create TabNet model
            tabnet_model = TabNetRegressor(
                n_d=tabnet_params.get("n_d", 64),
                n_a=tabnet_params.get("n_a", 64),
                n_steps=tabnet_params.get("n_steps", 5),
                gamma=tabnet_params.get("gamma", 1.5),
                lambda_sparse=tabnet_params.get("lambda_sparse", 1e-3),
                optimizer_params={"lr": tabnet_params.get("optimizer_lr", 0.02)},
                optimizer_fn=torch.optim.Adam if "torch" in sys.modules else None,
                mask_type='entmax',  # 'entmax' or 'sparsemax'
                scheduler_params={'step_size': 10, 'gamma': 0.9},
                device_name="auto" if "torch" in sys.modules else "cpu",
                verbose=0,
            )
            
            # Reshape data for TabNet
            X_tr_flat = X_train.reshape(X_train.shape[0], -1)
            y_tr_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
            
            # Train model
            tabnet_model.fit(
                X_tr_flat, y_tr_flat,
                batch_size=tabnet_params.get("batch_size", 1024),
                virtual_batch_size=tabnet_params.get("virtual_batch_size", 128),
                max_epochs=tabnet_params.get("max_epochs", 100),
                patience=tabnet_params.get("patience", 10)
            )
            
            # Predict
            X_te_flat = X_test.reshape(X_test.shape[0], -1)
            preds_1d = tabnet_model.predict(X_te_flat)
            preds = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
            
            return preds
        except ImportError:
            logger.warning("TabNet implementation not found. Returning zeros as predictions.")
            preds = np.zeros((X_test.shape[0], horizon))
    elif mtype == "tft":
        # Implement TFT model prediction
        try:
            from src.models.temporal_fusion_transformer import build_tft_model
            
            # Get TFT parameters
            tft_params = submodel_params[mtype]
            
            # Build TFT model
            model = build_tft_model(
                num_features=len(feature_cols),
                horizon=horizon,
                learning_rate=tft_params.get("lr", 0.001),
                hidden_size=tft_params.get("hidden_size", 64),
                dropout_rate=tft_params.get("dropout", 0.1),
                lookback=tft_params.get("lookback", unified_lookback),
                num_heads=tft_params.get("num_heads", 4),
                use_batch_norm=tft_params.get("use_batch_norm", True),
            )
            
            # Train the model
            epochs = tft_params.get("epochs", 10)
            batch_size = tft_params.get("batch_size", 32)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Make predictions
            preds = model.predict(X_test)
            return preds
        except ImportError:
            logger.warning("TFT model implementation not found. Returning zeros as predictions.")
            preds = np.zeros((X_test.shape[0], horizon))

    return preds


class OptunaSuggester:
    """Simple wrapper for Optuna parameter suggestions."""

    def __init__(self, trial):
        self.trial = trial

    def suggest(self, param_name):
        """
        Suggest a parameter value based on its configuration.
        
        Args:
            param_name: Name of the parameter to suggest
            
        Returns:
            Parameter value
        """
        from config.config_loader import HYPERPARAMETER_REGISTRY

        if (param_name not in HYPERPARAMETER_REGISTRY):
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
        """
        Suggest hyperparameters for a specific model type.
        
        Args:
            model_type: Type of model (e.g., 'lstm', 'rnn', 'cnn')
            
        Returns:
            Dict with suggested hyperparameters
        """
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
        elif model_type == "nbeats":
            return {
                "lookback": self.suggest("nbeats_lookback"),
                "lr": self.suggest("nbeats_lr"),
                "layer_width": self.suggest("nbeats_layer_width"),
                "num_blocks": self.suggest("nbeats_num_blocks"),
                "num_layers": self.suggest("nbeats_num_layers"),
                "thetas_dim": self.suggest("nbeats_thetas_dim"),
                "include_price_specific_stack": self.suggest("nbeats_price_specific"),
                "dropout_rate": self.suggest("nbeats_dropout"),
                "use_batch_norm": self.suggest("nbeats_batch_norm"),
            }
        elif model_type == "cnn":
            base_epochs = self.trial.suggest_int("cnn_base_epochs", 1, 10)
            actual_epochs = max(1, int(base_epochs * st.session_state.get("epochs_multiplier", 1.0)))
            complexity_multiplier = st.session_state.get("tuning_multipliers", {}).get("complexity_multiplier", 1.0)
            max_filters = int(256 * complexity_multiplier)
            return {
                "num_conv_layers": self.trial.suggest_int("cnn_num_conv_layers", 1, 5),
                "num_filters": self.trial.suggest_int("cnn_num_filters", 16, max_filters, log=True),
                "kernel_size": self.trial.suggest_int("cnn_kernel_size", 2, 7),
                "stride": self.trial.suggest_int("cnn_stride", 1, 2),
                "dropout_rate": self.trial.suggest_float("cnn_dropout_rate", 0.0, 0.5),
                "activation": self.trial.suggest_categorical("cnn_activation", ["relu", "leaky_relu", "elu"]),
                "use_adaptive_pooling": self.trial.suggest_categorical("cnn_use_adaptive_pooling", [True, False]),
                "fc_layers": [
                    self.trial.suggest_int("cnn_fc_layer_1", 32, 256, log=True),
                    self.trial.suggest_int("cnn_fc_layer_2", 16, 128, log=True),
                ],
                "lookback": self.trial.suggest_int("cnn_lookback", 7, 90),
                "lr": self.trial.suggest_float("cnn_lr", 1e-5, 1e-2, log=True),
                "batch_size": self.trial.suggest_categorical("cnn_batch_size", [16, 32, 64, 128]),
                "epochs": actual_epochs,
                "early_stopping_patience": self.trial.suggest_int("cnn_early_stopping_patience", 3, 10),
            }
        elif model_type == "tft":
            return {
                "lr": self.trial.suggest_float("tft_lr", 1e-5, 1e-2, log=True),
                "hidden_size": self.trial.suggest_int("tft_hidden_size", 16, 256),
                "dropout": self.trial.suggest_float("tft_dropout", 0.0, 0.5),
                "lookback": self.trial.suggest_int("tft_lookback", 7, 90),
                "num_heads": self.trial.suggest_int("tft_num_heads", 1, 8),
                "use_batch_norm": self.trial.suggest_categorical("tft_batch_norm", [True, False]),
                "epochs": self.trial.suggest_int("tft_epochs", 5, 50),
                "batch_size": self.trial.suggest_categorical("tft_batch_size", [16, 32, 64, 128]),
            }
        return {}


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


def initialize_tuning_environment():
    """Initialize the environment for tuning with proper checks"""
    check_sqlite_availability()
    verify_directories()


# Define a stop callback to halt tuning when stop is requested
class StopStudyCallback:
    """Callback that stops an Optuna study when stop is requested."""
    def __call__(self, study, trial):
        if is_stop_requested():
            study.stop()


def get_model_registry():
    """Lazily load the model registry to avoid circular imports"""
    # Use function attribute for caching
    if not hasattr(get_model_registry, "_registry"):
        try:
            # Import only when needed
            from src.training.incremental_learning import ModelRegistry

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
    """
    Register the best model from a tuning run to the model registry
    
    Args:
        study: Optuna study object
        trial: Best trial from the study
        ticker: Ticker symbol
        timeframe: Timeframe
        
    Returns:
        str: Model ID if registered, None otherwise
    """
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
        if model_type in ["lstm", "rnn", "tft", "cnn", "ltc", "nbeats"]:
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


def robust_objective_wrapper(objective_fn):
    """
    Wrap the Optuna objective function to provide better error handling.
    
    Args:
        objective_fn: Original objective function
        
    Returns:
        function: Wrapped objective function with error handling
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
                set_stop_requested(True)
                return float("inf")
            return 1.0e6

    return wrapped_objective


def create_progress_callback(cycle=1, ticker="unknown", timeframe="unknown", range_cat="all"):
    """
    Create a callback that updates progress file with thread-safety.
    
    Args:
        cycle: Current tuning cycle
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category
        
    Returns:
        function: Callback function for Optuna
    """
    def progress_callback(study, trial):
        """Update progress when a trial completes."""
        # Only process when trial completes
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Get metrics from trial
        rmse = trial.user_attrs.get("rmse", float("inf"))
        mape = trial.user_attrs.get("mape", float("inf"))

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

        # Save to session state if available - using more robust error handling
        try:
            import streamlit as st
            if "st" in globals() and hasattr(st, "session_state"):
                # First check if trial_logs exists, create if not
                if "trial_logs" not in st.session_state:
                    st.session_state["trial_logs"] = []
                    
                # Now it's safe to append
                st.session_state["trial_logs"].append(trial_info)
                
                # Keep only the most recent 100 trials in memory
                if len(st.session_state["trial_logs"]) > 100:
                    st.session_state["trial_logs"] = st.session_state["trial_logs"][-100:]
        except Exception as e:
            # Just log the error but continue - don't let UI issues break tuning
            logger.warning(f"Error updating session state: {e}")

        # Clean memory periodically
        if trial.number % 5 == 0:
            # Use training optimizer for more efficient cleanup
            training_optimizer.cleanup_memory(level="light") 
            
        if trial.number % 20 == 0:
            # More thorough cleanup every 20 trials
            gpu_memory_manager.clean_memory(True)
            # Check GPU utilization
            gpu_util = get_gpu_utilization(0)
            logger.info(f"GPU utilization at trial {trial.number}: {gpu_util}%")

        print(f"TUNING-DEBUG: Completed trial {trial.number}")

    return progress_callback


def ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat="all"):
    """
    Objective function for ensemble model evaluation using walk-forward validation.

    Args:
        trial: Optuna trial object
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category ("all" or specific range)

    Returns:
        float: Combined objective score (lower is better)
    """
    # Choose a model type
    model_type = trial.suggest_categorical("model_type", ACTIVE_MODEL_TYPES)

    # Get hyperparameters for the selected model type
    params = suggest_model_hyperparameters(trial, model_type)

    # Evaluate model using walk-forward validation
    metrics = evaluate_model_with_walkforward(
        trial, ticker, timeframe, range_cat, model_type
    )

    # Calculate composite score (rmse + 0.5*mape)
    score = metrics.get("rmse", float("inf")) + 0.5 * metrics.get("mape", float("inf"))

    return score


def calculate_combined_metrics(results):
    """
    Calculate combined metrics from all model results.
    Also includes ensemble weights in the metrics.
    
    Args:
        results: Dictionary of results from different models
        
    Returns:
        Dictionary with combined metrics
    """
    # Initialize with worst possible values
    combined_metrics = {
        "rmse": float('inf'),
        "mape": float('inf'),
        "directional_accuracy": 0.0,
        "combined_score": float('inf'),
        "total_trials": 0,
        "completed_trials": 0,
        "model_metrics": {},
        "timestamp": datetime.now().isoformat(),
        "ensemble_weights": {}  # Track ensemble weights
    }
    
    # Extract and combine metrics from each model
    for model_type, result in results.items():
        if "error" in result:
            logger.warning(f"Skipping {model_type} due to error: {result['error']}")
            continue
            
        study = result.get("study")
        best_trial = result.get("best_trial")
        
        if not study or not study.trials:
            logger.warning(f"No trials for {model_type}, skipping")
            continue
            
        # Count trials
        combined_metrics["total_trials"] += study.user_attrs.get("n_trials", 0)
        combined_metrics["completed_trials"] += len(study.trials)
        
        # Extract best metrics from this model's best trial
        try:
            if best_trial:
                model_rmse = best_trial.user_attrs.get("rmse", float('inf'))
                model_mape = best_trial.user_attrs.get("mape", float('inf'))
                model_da = best_trial.user_attrs.get("directional_accuracy", 0.0)
                model_score = best_trial.value if hasattr(best_trial, 'value') else float('inf')
                
                # Track the best model's weight (from trial params or default to 1.0)
                weight = best_trial.params.get("ensemble_weight", 1.0)
                combined_metrics["ensemble_weights"][model_type] = weight
                
                # Update best overall metrics
                if model_rmse < combined_metrics["rmse"]:
                    combined_metrics["rmse"] = model_rmse
                    combined_metrics["best_rmse"] = model_rmse
                    combined_metrics["best_model_rmse"] = model_type
                    
                if model_mape < combined_metrics["mape"]:
                    combined_metrics["mape"] = model_mape
                    combined_metrics["best_mape"] = model_mape
                    combined_metrics["best_model_mape"] = model_type
                    
                if model_da > combined_metrics["directional_accuracy"]:
                    combined_metrics["directional_accuracy"] = model_da
                    combined_metrics["best_directional_accuracy"] = model_da
                    combined_metrics["best_model_da"] = model_type
                    
                if model_score < combined_metrics["combined_score"]:
                    combined_metrics["combined_score"] = model_score
                    combined_metrics["best_model"] = model_type
                
                # Store individual model metrics
                combined_metrics["model_metrics"][model_type] = {
                    "rmse": model_rmse,
                    "mape": model_mape,
                    "directional_accuracy": model_da,
                    "combined_score": model_score,
                    "completed_trials": len(study.trials),
                    "total_trials": study.user_attrs.get("n_trials", 0),
                    "weight": weight  # Store the model's weight for ensemble calculation
                }
        except Exception as e:
            logger.error(f"Error processing metrics for {model_type}: {e}")
    
    # Calculate weighted ensemble metrics if we have weights
    if combined_metrics["ensemble_weights"]:
        try:
            # Normalize weights
            total_weight = sum(combined_metrics["ensemble_weights"].values())
            if total_weight > 0:
                normalized_weights = {
                    model: weight/total_weight 
                    for model, weight in combined_metrics["ensemble_weights"].items()
                }
                
                # Calculate weighted metrics
                weighted_rmse = sum(
                    normalized_weights.get(model, 0) * metrics.get("rmse", float('inf'))
                    for model, metrics in combined_metrics["model_metrics"].items()
                )
                
                weighted_mape = sum(
                    normalized_weights.get(model, 0) * metrics.get("mape", float('inf'))
                    for model, metrics in combined_metrics["model_metrics"].items()
                )
                
                weighted_da = sum(
                    normalized_weights.get(model, 0) * metrics.get("directional_accuracy", 0)
                    for model, metrics in combined_metrics["model_metrics"].items()
                )
                
                # Store weighted ensemble metrics
                combined_metrics["ensemble_rmse"] = weighted_rmse
                combined_metrics["ensemble_mape"] = weighted_mape
                combined_metrics["ensemble_directional_accuracy"] = weighted_da
                combined_metrics["normalized_weights"] = normalized_weights
        except Exception as e:
            logger.error(f"Error calculating ensemble metrics: {e}")
    
    return combined_metrics


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
    # Reset stop request flag
    set_stop_requested(False)

    # Clean memory before starting
    adaptive_memory_clean("large")
    
    # Initialize GPU for optimal performance
    initialize_gpu_for_tuning()
    
    # Enable performance mode for intensive operations
    logger.info("Enabling GPU performance mode for tuning...")
    set_performance_mode(True)

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
        # Calculate combined metrics from all model results
        combined_metrics = calculate_combined_metrics(results)
        
        # Add cycle and identification info
        combined_metrics.update({
            "cycle": cycle,
            "ticker": ticker,
            "timeframe": timeframe
        })
        
        # Find best model across all studies (keep original logic)
        best_model_type, best_params, best_value = study_manager.get_best_model()
        
        # Add best model info
        if best_model_type:
            combined_metrics["best_model_type"] = best_model_type
            combined_metrics["best_model_value"] = best_value
            combined_metrics["best_model_params"] = best_params
        
        # Save cycle metrics
        update_cycle_metrics(combined_metrics, cycle_num=cycle)
        
        # Update final progress to indicate completion
        final_progress = {
            "current_trial": combined_metrics["completed_trials"],
            "total_trials": combined_metrics["total_trials"],
            "current_rmse": combined_metrics.get("rmse"),
            "current_mape": combined_metrics.get("mape"),
            "best_rmse": combined_metrics.get("best_rmse"),
            "best_mape": combined_metrics.get("best_mape"),
            "best_model": combined_metrics.get("best_model"),
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
        traceback.print_exc()

    # Disable performance mode to save energy
    set_performance_mode(False)
    
    # Clean memory after optimization
    adaptive_memory_clean("large")

    return results


def start_tuning_process(ticker, timeframe, multipliers=None, force_start=False):
    """
    Start the tuning process by calling tune_for_combo.
    Sets session state to indicate tuning is in progress.

    Args:
        ticker (str): The ticker symbol
        timeframe (str): The timeframe (e.g. "1d")
        multipliers (dict, optional): Dictionary of tuning multipliers
        force_start (bool, optional): Whether to force start even if already running
        
    Returns:
        dict: Results from tuning, or None if error
    """
    print(f"TUNING-DEBUG: Starting tuning process for {ticker} {timeframe}")
    
    # Check if tuning is already running
    current_status = read_tuning_status()
    if current_status.get("is_running", False) and not force_start:
        logger.warning("Tuning process is already running")
        return None

    # Reset the stop event to make sure we're not in a stopped state
    set_stop_requested(False)
    
    #Start all studies
    from src.tuning.study_manager import run_all_studies
    run_all_studies(ticker, timeframe)

    # Set session state to indicate tuning is in progress
    st.session_state["tuning_in_progress"] = True
    st.session_state["tuning_start_time"] = time.time()

    # Store multipliers in session state if provided
    if multipliers is not None:
        st.session_state["tuning_multipliers"] = multipliers
    else:
        multipliers = st.session_state.get("tuning_multipliers", None)

    # Calculate adjusted trials based on mode multipliers
    adjusted_trials = TUNING_TRIALS_PER_CYCLE_min
    if multipliers:
        # Get mode from multipliers or use default
        trials_multiplier = multipliers.get("trials", 1.0)
        adjusted_trials = max(
            TUNING_TRIALS_PER_CYCLE_min,
            min(TUNING_TRIALS_PER_CYCLE_max, int(TUNING_TRIALS_PER_CYCLE_min * trials_multiplier)),
        )

    # Update tuning status
    write_tuning_status({
        "is_running": True,
        "ticker": ticker,
        "timeframe": timeframe,
        "total_trials": adjusted_trials * len(ACTIVE_MODEL_TYPES),
        "start_time": time.time(),
        "cycle": 1
    })

    # Initialize GPU for optimal performance
    initialize_gpu_for_tuning()

    try:
        # Run tuning for combination
        results = tune_for_combo(
            ticker,
            timeframe,
            range_cat="all",
            n_trials=adjusted_trials,
            cycle=1,
            tuning_multipliers=multipliers,
        )
        
        # Update tuning status on completion
        write_tuning_status({
            "is_running": False,
            "status": "completed",
            "end_time": time.time()
        })
        
        # Update session state
        st.session_state["tuning_in_progress"] = False
        
        return results
    except Exception as e:
        logger.error(f"Error in tuning process: {e}")
        # Always disable performance mode on error
        set_performance_mode(False)
        
        # Update status file
        write_tuning_status({
            "is_running": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        })
        
        return None


def stop_tuning_process():
    """
    Stop the tuning process by signaling the stop event.
    Updates session state to indicate tuning has stopped.
    
    Returns:
        bool: True if stop was signaled
    """
    # Signal the stop event
    set_stop_requested(True)

    # Update session state
    st.session_state["tuning_in_progress"] = False

    # Update tuning status
    write_tuning_status(
        {"is_running": False, "stopped_manually": True, "stop_time": time.time()}
    )

    # Clear stored multipliers to use defaults next time
    if "tuning_multipliers" in st.session_state:
        del st.session_state["tuning_multipliers"]

    logger.info("Tuning process stop requested")
    return True


def main():
    """Main entry point for tuning."""
    # Clean memory at start
    adaptive_memory_clean("large")
    
    # Initialize GPU for optimal performance
    initialize_gpu_for_tuning()

    # Loop through tuning cycles
    for cycle in range(1, TUNING_LOOP + 1):
        # Clean memory between cycles
        adaptive_memory_clean("large")

    # Final memory cleanup
    adaptive_memory_clean("large")
    
    # Disable performance mode when done to save energy
    set_performance_mode(False)


# Call the initialize function when imported
if __name__ != "__main__":
    # Only run initialization when imported as a module (not when run directly)
    initialize_tuning_environment()

# Exports
__all__ = [
    "start_tuning_process",
    "stop_tuning_process",
    "tune_for_combo",
    "register_best_model",
    "get_model_registry",
]
