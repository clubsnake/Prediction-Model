# walk_forward.py
"""
Implements a unified walk-forward validation scheme that combines:
1. Realistic incremental learning with no data leakage
2. Ensemble model approach with weighted predictions
3. Memory efficient implementations and caching
4. Integration with dashboard for forecasting
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os

# Import the config properly
import sys

# Set environment variables BEFORE importing TensorFlow
from src.utils.env_setup import setup_tf_environment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from config.config_loader import (
        WALK_FORWARD_DEFAULT,
        WALK_FORWARD_MAX,
        WALK_FORWARD_MIN,
        get_value,
    )

    use_mixed_precision = get_value("hardware.use_mixed_precision", False)
except ImportError:
    use_mixed_precision = False
    WALK_FORWARD_MIN = 3
    WALK_FORWARD_MAX = 180
    WALK_FORWARD_DEFAULT = 30

tf_env = setup_tf_environment(mixed_precision=use_mixed_precision)
import json
import logging
import traceback
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import streamlit as st

# Now import TensorFlow after environment variables are set
import tensorflow as tf

from src.utils.memory_utils import WeakRefCache, cleanup_tf_session, log_memory_usage

# Improved imports with better error handling
try:
    from config.config_loader import (
        ACTIVE_MODEL_TYPES,
        MODEL_TYPES,
        PREDICTION_HORIZON,
        START_DATE,
        WALK_FORWARD_DEFAULT,
    )
except ImportError:
    # Define fallbacks
    ACTIVE_MODEL_TYPES = [
        "rnn",
        "tft",
        "cnn",
        "ltc",
        "lstm",
        "random_forest",
        "xgboost",
        "tabnet",
    ]
    MODEL_TYPES = ACTIVE_MODEL_TYPES + [
        "rnn",
        "tft",
        "cnn",
        "ltc",
        "lstm",
        "random_forest",
        "xgboost",
        "tabnet",
    ]
    PREDICTION_HORIZON = 30
    START_DATE = "2020-01-01"
    WALK_FORWARD_DEFAULT = 30
    logger.warning("Failed to import config values, using defaults")

try:
    from src.data.preprocessing import create_sequences
except ImportError:
    logger.error("Failed to import create_sequences")

    # Define a minimal fallback implementation
    def create_sequences(df, feature_cols, target_col, lookback, horizon):
        logger.warning("Using fallback create_sequences implementation")
        return None, None


# Handle model imports safely
try:
    from src.models.model import build_model_by_type, record_weight_snapshot
    from src.models.model_factory import BaseModel
except ImportError:
    logger.error("Failed to import model classes")
    # Will cause errors if these functions are used

# Safely import TabNet components
try:
    from src.models.tabnet_model import TabNetPricePredictor

    HAS_TABNET = True
    logger.info("TabNet successfully imported")
except ImportError as e:
    HAS_TABNET = False
    logger.warning(f"TabNet import failed: {e}")

# Safely import trainer
try:
    from src.training.trainer import ModelTrainer
except ImportError:
    logger.error("Failed to import ModelTrainer")

# Import utilities safely
try:
    from src.utils.memory_utils import (
        WeakRefCache,
        cleanup_tf_session,
        log_memory_usage,
    )
    from src.utils.training_optimizer import TrainingOptimizer, get_training_optimizer
    from src.utils.vectorized_ops import numba_mse, vectorized_sequence_creation
except ImportError as e:
    logger.error(f"Failed to import utility functions: {e}")

    # Define minimal fallbacks for critical functions
    def numba_mse(x, y):
        return ((x - y) ** 2).mean()

    class WeakRefCache:
        def __init__(self):
            self.cache = {}

    def log_memory_usage(tag):
        logger.info(f"Memory logging not available: {tag}")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("walkforward.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Log the environment settings
logger.info(f"TensorFlow environment settings: {json.dumps(tf_env, indent=2)}")

# Create a cache for model predictions to reduce redundant computation
prediction_cache = WeakRefCache()
model_cache = WeakRefCache()


# Initialize the training optimizer globally
training_optimizer = get_training_optimizer()
logger.info(
    f"Training optimizer initialized with {training_optimizer.cpu_count} CPUs, {training_optimizer.gpu_count} GPUs"
)

# Import configuration properly with safety checks
from config.config_loader import get_config

# Get configuration values safely with defaults
try:
    config = get_config()
    LOOKBACK = config.get("LOOKBACK", 30)  # Use get() with default value
    PREDICTION_HORIZON = config.get("PREDICTION_HORIZON", 30)
except Exception as e:
    logger.warning(f"Error loading config values: {e}")
    # Set default values if config loading fails
    LOOKBACK = 30
    PREDICTION_HORIZON = 30


def calculate_mse(predictions, actuals):
    """
    Calculate Mean Squared Error between predictions and actuals.

    Args:
        predictions: List of prediction arrays
        actuals: List of actual value arrays

    Returns:
        float: Mean Squared Error
    """
    try:
        # Convert lists to arrays if needed
        if isinstance(predictions, list) and isinstance(actuals, list):
            pred_array = np.vstack(predictions).flatten()
            actuals_array = np.vstack(actuals).flatten()
        else:
            pred_array = np.array(predictions).flatten()
            actuals_array = np.array(actuals).flatten()

        # Calculate MSE
        return np.mean((pred_array - actuals_array) ** 2)
    except Exception as e:
        logger.error(f"Error calculating MSE: {e}")
        return float("inf")


def calculate_mape(predictions, actuals, epsilon=1e-7):
    """
    Calculate Mean Absolute Percentage Error between predictions and actuals.

    Args:
        predictions: List of prediction arrays
        actuals: List of actual value arrays
        epsilon: Small value to avoid division by zero

    Returns:
        float: Mean Absolute Percentage Error (as percentage)
    """
    try:
        # Convert lists to arrays if needed
        if isinstance(predictions, list) and isinstance(actuals, list):
            pred_array = np.vstack(predictions).flatten()
            actuals_array = np.vstack(actuals).flatten()
        else:
            pred_array = np.array(predictions).flatten()
            actuals_array = np.array(actuals).flatten()

        # Calculate MAPE with epsilon to avoid division by zero
        absolute_percentage_errors = (
            np.abs((actuals_array - pred_array) / (actuals_array + epsilon)) * 100
        )
        return float(np.mean(absolute_percentage_errors))
    except Exception as e:
        logger.error(f"Error calculating MAPE: {e}")
        return float("inf")


def enable_xla_compilation():
    """Enable XLA compilation for TensorFlow models"""
    try:
        # Enable XLA
        tf.config.optimizer.set_jit(True)

        # Use mixed precision only if configured in settings
        try:
            from config.config_loader import get_value

            use_mixed_precision = get_value("hardware.use_mixed_precision", False)
        except ImportError:
            use_mixed_precision = False

        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("XLA compilation and mixed precision enabled")
        else:
            logger.info("XLA compilation enabled (mixed precision disabled per config)")

        return True
    except Exception as e:
        logger.error(f"Error enabling XLA compilation: {e}")
        # Continue execution
        logger.warning(f"Failed to enable XLA compilation: {e}")
        return False


def generate_future_forecast(model, df, feature_cols, lookback=None, horizon=None):
    """Generate predictions for the next 'horizon' days into the future"""
    try:
        if model is None or df is None or df.empty:
            logger.warning("Cannot generate forecast: model or data is missing")
            return []

        # Use values from session state if not provided
        lookback = lookback or st.session_state.get("lookback", 30)
        horizon = horizon or st.session_state.get("forecast_window", 30)

        logger.info(
            f"Generating future forecast: lookback={lookback}, horizon={horizon}"
        )

        # Get the last 'lookback' days of data for input
        last_data = df.iloc[-lookback:].copy()

        # Create a scaler for feature normalization
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(last_data[feature_cols])

        # Initialize array to store predictions
        future_prices = []
        current_data = last_data.copy()

        try:
            # Import create_sequences
            from src.data.preprocessing import create_sequences

            # First attempt - use create_sequences to generate input sequence
            X_input, _ = create_sequences(
                current_data, feature_cols, "Close", lookback, 1
            )

            # Make prediction for each day in the horizon
            for i in range(horizon):
                # Use model to predict
                preds = model.predict(X_input, verbose=0)

                # Get the predicted price (first value if multiple outputs)
                if hasattr(preds, "shape") and len(preds.shape) > 1:
                    next_price = float(preds[0][0])
                else:
                    next_price = float(preds[0])

                future_prices.append(next_price)

                # Update input data with the prediction
                next_row = current_data.iloc[-1:].copy()
                if isinstance(next_row.index[0], pd.Timestamp):
                    next_row.index = [next_row.index[0] + pd.Timedelta(days=1)]
                else:
                    next_row.index = [next_row.index[0] + 1]

                next_row["Close"] = next_price
                current_data = pd.concat([current_data.iloc[1:], next_row])

                # Scale features for consistent input
                current_scaled = current_data.copy()
                current_scaled[feature_cols] = scaler.transform(
                    current_data[feature_cols]
                )

                # Create new sequence for next prediction
                X_input, _ = create_sequences(
                    current_scaled, feature_cols, "Close", lookback, 1
                )

            logger.info(f"Generated {len(future_prices)} day forecast")
            return future_prices

        except ImportError:
            logger.warning(
                "Using simplified forecast approach without create_sequences"
            )

            # Fallback simple approach
            # ...existing code...
    except Exception as e:
        logger.error(f"Error generating future forecast: {e}", exc_info=True)
        return []


# Use a dynamic import approach to avoid circular imports
def get_dashboard_update_function():
    """Dynamically import and return the dashboard forecast update function"""
    try:
        import importlib

        module = importlib.import_module("src.dashboard.prediction_service")
        if hasattr(module, "update_dashboard_forecast"):
            return module.update_dashboard_forecast
        else:
            # Fallback to local implementation
            return update_forecast_in_session_state
    except ImportError:
        logger.info("Could not import prediction_service, using local implementation")
        return update_forecast_in_session_state


# Define a unified function
def update_forecast_in_session_state(
    ensemble_model, df, feature_cols, ensemble_weights=None
):
    """Update forecast in session state for dashboard display"""
    try:
        # Generate and save forecast data for the dashboard
        lookback = st.session_state.get("lookback", 30)
        forecast_window = st.session_state.get("forecast_window", 30)

        # Generate forecast with ensemble model
        future_forecast = generate_future_forecast(
            ensemble_model, df, feature_cols, lookback, forecast_window
        )

        # Update session state
        st.session_state["future_forecast"] = future_forecast
        st.session_state["last_forecast_update"] = datetime.now()

        # Also store the ensemble weights if provided
        if ensemble_weights:
            st.session_state["ensemble_weights"] = ensemble_weights

        logger.info(f"Updated dashboard forecast with {len(future_forecast)} days")
        return future_forecast
    except Exception as e:
        logger.error(f"Error updating forecast in dashboard: {e}", exc_info=True)
        return None


# Use dynamic function retrieval at the module level to assign this function once
update_dashboard_forecast_function = get_dashboard_update_function()
update_forecast_in_dashboard = (
    update_forecast_in_session_state  # Maintain backward compatibility
)


def get_ensemble_model(model_types, models_dict, ensemble_weights):
    """Create an ensemble model from individual models with given weights"""

    class EnsembleModel:
        def __init__(self, models_dict, weights, model_types):
            self.models = models_dict
            self.weights = weights
            self.model_types = model_types
            logger.info(
                f"Created ensemble model with {len(models_dict)} models and types: {list(models_dict.keys())}"
            )

        def predict(self, X, verbose=0):
            """Generate weighted ensemble prediction"""
            ensemble_pred = None
            total_weight = 0.0

            for mtype, weight in self.weights.items():
                if weight <= 0 or mtype not in self.models:
                    continue

                model = self.models[mtype]
                if model is None:
                    continue

                try:
                    # Get prediction from this model
                    pred = model.predict(X, verbose=0)

                    # Weighted contribution to ensemble
                    if pred is not None:
                        w = max(0, weight)  # Ensure non-negative
                        total_weight += w

                        if ensemble_pred is None:
                            ensemble_pred = w * pred
                        else:
                            ensemble_pred += w * pred
                except Exception as e:
                    logger.error(f"Error in ensemble prediction for {mtype}: {e}")
                    continue

            # Normalize by total weight
            if ensemble_pred is not None and total_weight > 0:
                ensemble_pred /= total_weight
                return ensemble_pred

            # Fallback: use first available model if ensemble fails
            for mtype, model in self.models.items():
                if model is not None:
                    logger.warning(f"Using fallback model {mtype} for prediction")
                    return model.predict(X, verbose=0)

            # Last resort: return zeros
            logger.error("Ensemble prediction failed, returning zeros")
            if isinstance(X, np.ndarray):
                return np.zeros((X.shape[0], 1))  # Default shape
            return None

    # Create and return the ensemble model
    return EnsembleModel(models_dict, ensemble_weights, model_types)


def _get_default_submodel_params(mtype, optimized=False):
    if mtype == "lstm":
        if optimized:
            return {
                "loss_function": "mean_squared_error",
                "epochs": 1,
                "batch_size": 32,
            }
        else:
            return {"batch_size": 32}
    elif mtype == "random_forest":
        return {"n_est": 100, "mdepth": 10}
    elif mtype == "xgboost":
        return {"n_est": 100, "lr": 0.1}
    elif mtype == "tabnet":
        params = {
            "n_d": 64,
            "n_a": 64,
            "n_steps": 5,
            "gamma": 1.5,
            "lambda_sparse": 0.001,
            "optimizer_params": {"lr": 0.02},
            "max_epochs": 200,
        }
        if not optimized:
            params["patience"] = 15
            params["batch_size"] = 1024
        return params
    else:
        return {}


def unified_walk_forward(
    df,
    feature_cols,
    submodel_params_dict=None,
    ensemble_weights=None,
    training_start_date=None,
    window_size=None,
    update_dashboard=True,
    trial=None,
    enable_parallel=True,
    update_frequency=5,
):
    """
    Unified walk-forward implementation that combines realistic incremental learning
    with ensemble modeling, parallel training, and proper forecast generation.

    Args:
        df: DataFrame with features & target columns
        feature_cols: List of feature column names
        submodel_params_dict: Dictionary of model parameters for each type
        ensemble_weights: Dictionary of weights for ensemble (default: equal weights)
        training_start_date: Start date for training (default: config.START_DATE)
        window_size: Size of each walk-forward step (default: config.WALK_FORWARD_DEFAULT)
        update_dashboard: Whether to update the dashboard forecast (default: True)
        trial: Optuna trial object for hyperparameter tuning (default: None)
        enable_parallel: Whether to use parallel training (default: True)
        update_frequency: How often to report progress and update dashboard (default: 5)

    Returns:
        (ensemble_model, metrics_dict): The trained ensemble model and performance metrics
    """
    # Default values and input validation
    target_col = "Close"  # Default target column
    log_memory_usage("Starting unified_walk_forward")

    # Initialize training optimizer for parallel training
    training_optimizer = get_training_optimizer()
    logger.info(
        f"Using training optimizer with {training_optimizer.cpu_count} CPUs, {training_optimizer.gpu_count} GPUs"
    )

    # Log initial memory usage
    training_optimizer.log_memory_usage("start_unified_walk_forward")

    # Set defaults if not provided
    if ensemble_weights is None:
        # Default to equal weights for active model types
        ensemble_weights = {
            mtype: 1.0 / len(ACTIVE_MODEL_TYPES) for mtype in ACTIVE_MODEL_TYPES
        }

    if submodel_params_dict is None:
        submodel_params_dict = {}
        for mtype in ACTIVE_MODEL_TYPES:
            submodel_params_dict[mtype] = _get_default_submodel_params(
                mtype, optimized=True
            )

    # Validate window size for walk-forward
    def validate_walk_forward(size):
        if size is None:
            return WALK_FORWARD_DEFAULT
        try:
            size = int(size)
            return max(WALK_FORWARD_MIN, min(WALK_FORWARD_MAX, size))
        except:
            return WALK_FORWARD_DEFAULT

    wf_size = validate_walk_forward(window_size)
    logger.info(f"Walk-forward window size: {wf_size}")

    # Validate inputs
    if df is None or df.empty:
        logger.error("Empty DataFrame provided")
        return None, {"mse": float("inf"), "mape": float("inf")}

    # Check that all feature columns exist in DataFrame
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns: {missing_cols}")
        return None, {"mse": float("inf"), "mape": float("inf")}

    # Check target column exists
    if target_col not in df.columns:
        logger.error(f"Missing target column: {target_col}")
        return None, {"mse": float("inf"), "mape": float("inf")}

    # Determine lookback window - use maximum from all model configurations
    lookback_values = []
    for mtype, params in submodel_params_dict.items():
        if ensemble_weights.get(mtype, 0) > 0:
            lookback_values.append(params.get("lookback", 30))

    # If no specific lookback values, use default
    lookback = max(lookback_values) if lookback_values else 30
    logger.info(f"Using lookback window: {lookback}")

    # Determine prediction horizon
    horizon = PREDICTION_HORIZON  # Use configured horizon
    logger.info(f"Using prediction horizon: {horizon}")

    # Determine training start date
    start_date = training_start_date or START_DATE
    logger.info(f"Training start date: {start_date}")

    # Find index for training start
    start_idx = 0
    if "date" in df.columns:
        valid_indices = df.index[df["date"] >= start_date]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]

    # Initialize models dict and storage
    models_dict = {}
    all_predictions = []
    all_actuals = []

    # Initial training data - validate we have enough
    if start_idx + lookback >= len(df):
        logger.error("Not enough data for training with specified lookback")
        return None, {"mse": float("inf"), "mape": float("inf")}

    # Set training end cutoff (we don't use future data)
    train_end_idx = start_idx + lookback

    # Calculate number of cycles for progress reporting
    total_samples = len(df)
    total_cycles = (total_samples - train_end_idx - horizon) // wf_size

    # Enable XLA compilation if TensorFlow models are used
    if any(
        mt in ["lstm", "rnn", "tft", "ltc"]
        for mt, w in ensemble_weights.items()
        if w > 0
    ):
        enable_xla_compilation()

    # Main walk-forward loop
    current_idx = train_end_idx
    cycle = 0

    try:
        while current_idx + horizon < len(df):
            try:
                # Report progress periodically
                if cycle % update_frequency == 0 or cycle == total_cycles - 1:
                    logger.info(
                        f"Walk-forward cycle {cycle+1}/{total_cycles} (idx={current_idx}/{len(df)})"
                    )
                    # Report to Optuna if provided
                    if (
                        trial is not None
                        and cycle > 0
                        and all_predictions
                        and all_actuals
                    ):
                        try:
                            from src.utils.vectorized_ops import numba_mse

                            interim_mse = numba_mse(
                                np.array(all_predictions).flatten(),
                                np.array(all_actuals).flatten(),
                            )
                        except Exception:
                            interim_mse = calculate_mse(all_predictions, all_actuals)

                        trial.report(np.sqrt(interim_mse), step=cycle)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                # 1. Define current train/test windows
                train_data = df.iloc[:current_idx].copy()
                test_data = df.iloc[current_idx : current_idx + horizon].copy()

                # 2. Scale features properly
                scaler = StandardScaler()
                train_scaled = train_data.copy()
                test_scaled = test_data.copy()

                # Fit scaler on training data only
                train_scaled[feature_cols] = scaler.fit_transform(
                    train_data[feature_cols]
                )

                # Transform test data with the same scaler
                test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])

                # 3. Create sequences for model training and testing
                X_train, y_train = vectorized_sequence_creation(
                    train_scaled, feature_cols, target_col, lookback, horizon
                )

                X_test, y_test = vectorized_sequence_creation(
                    test_scaled, feature_cols, target_col, lookback, horizon
                )

                # Skip if sequences couldn't be created
                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"Empty sequences at cycle {cycle}, skipping")
                    current_idx += wf_size
                    cycle += 1
                    continue

                # Define unified model training function used by both sequential and parallel approaches
                def train_model_function(config):
                    model_type = config["model_type"]
                    params = config["params"]
                    settings = config["settings"]

                    try:
                        # 1. Neural network models: LSTM, RNN, TFT
                        if model_type in ["lstm", "rnn", "tft"]:
                            from src.models.model import build_model_by_type

                            # Extract architecture parameters
                            arch_params = {
                                "units_per_layer": params.get(
                                    "units_per_layer", [64, 32]
                                ),
                                "use_batch_norm": params.get("use_batch_norm", False),
                                "l2_reg": params.get("l2_reg", 0.0),
                                "use_attention": params.get("use_attention", True),
                                "attention_type": params.get("attention_type", "dot"),
                                "attention_size": params.get("attention_size", 64),
                                "attention_heads": params.get("attention_heads", 1),
                                "attention_dropout": params.get(
                                    "attention_dropout", 0.0
                                ),
                            }

                            # Get or build model
                            if (
                                model_type in models_dict
                                and models_dict[model_type] is not None
                            ):
                                model = models_dict[model_type]
                            else:
                                model = build_model_by_type(
                                    model_type=model_type,
                                    num_features=len(feature_cols),
                                    horizon=horizon,
                                    learning_rate=params.get("lr", 0.001),
                                    dropout_rate=params.get("dropout", 0.2),
                                    loss_function=params.get(
                                        "loss_function", "mean_squared_error"
                                    ),
                                    lookback=lookback,
                                    architecture_params=arch_params,
                                )

                            # Train model
                            model.fit(
                                X_train,
                                y_train,
                                epochs=params.get("epochs", 1),
                                batch_size=params.get(
                                    "batch_size", settings.get("batch_size", 32)
                                ),
                                verbose=0,
                            )

                            # Generate predictions
                            pred = model.predict(X_test, verbose=0)

                            return {"predictions": pred, "model": model}

                        # 2. Tree-based models: Random Forest
                        elif model_type == "random_forest":
                            from sklearn.ensemble import RandomForestRegressor

                            # Flatten inputs for tree models
                            X_tr_flat = X_train.reshape(X_train.shape[0], -1)
                            y_tr_flat = y_train[:, 0]

                            # Configure model with optimal core count
                            core_count = settings.get("cpu_cores", -1)
                            model = RandomForestRegressor(
                                n_estimators=params.get("n_est", 100),
                                max_depth=params.get("mdepth", 10),
                                random_state=42,
                                n_jobs=core_count,
                            )

                            # Train model
                            model.fit(X_tr_flat, y_tr_flat)

                            # Generate predictions
                            X_te_flat = X_test.reshape(X_test.shape[0], -1)
                            preds_1d = model.predict(X_te_flat)

                            # Match shape of neural net predictions
                            pred = np.tile(preds_1d.reshape(-1, 1), (1, horizon))

                            return {"predictions": pred, "model": model}

                        # 3. Tree-based models: XGBoost
                        elif model_type == "xgboost":
                            import xgboost as xgb

                            # Flatten inputs for tree models
                            X_tr_flat = X_train.reshape(X_train.shape[0], -1)
                            y_tr_flat = y_train[:, 0]

                            # Configure model with optimal core count
                            core_count = settings.get("cpu_cores", -1)
                            model = xgb.XGBRegressor(
                                n_estimators=params.get("n_est", 100),
                                learning_rate=params.get("lr", 0.1),
                                max_depth=params.get("max_depth", 6),
                                subsample=params.get("subsample", 1.0),
                                colsample_bytree=params.get("colsample_bytree", 1.0),
                                random_state=42,
                                n_jobs=core_count,
                            )

                            # Train model
                            model.fit(X_tr_flat, y_tr_flat)

                            # Generate predictions
                            X_te_flat = X_test.reshape(X_test.shape[0], -1)
                            preds_1d = model.predict(X_te_flat)

                            # Match shape of neural net predictions
                            pred = np.tile(preds_1d.reshape(-1, 1), (1, horizon))

                            return {"predictions": pred, "model": model}

                        # 4. TabNet models
                        elif model_type == "tabnet":
                            if not HAS_TABNET:
                                return {"error": "TabNet is not available"}

                            from src.models.tabnet_model import TabNetPricePredictor

                            tabnet_params = params

                            # Apply optimizer settings
                            if "batch_size" in settings:
                                tabnet_params["batch_size"] = settings["batch_size"]

                            if (
                                "learning_rate" in settings
                                and "optimizer_params" in tabnet_params
                            ):
                                tabnet_params["optimizer_params"]["lr"] = settings[
                                    "learning_rate"
                                ]

                            # Create model with optimized parameters
                            model = TabNetPricePredictor(
                                n_d=tabnet_params.get("n_d", 64),
                                n_a=tabnet_params.get("n_a", 64),
                                n_steps=tabnet_params.get("n_steps", 5),
                                gamma=tabnet_params.get("gamma", 1.5),
                                lambda_sparse=tabnet_params.get("lambda_sparse", 0.001),
                                optimizer_params=tabnet_params.get(
                                    "optimizer_params", {"lr": 0.02}
                                ),
                                feature_names=feature_cols,
                                max_epochs=tabnet_params.get("max_epochs", 200),
                                patience=tabnet_params.get("patience", 15),
                                batch_size=tabnet_params.get("batch_size", 1024),
                                virtual_batch_size=tabnet_params.get(
                                    "virtual_batch_size", 128
                                ),
                                momentum=tabnet_params.get("momentum", 0.02),
                            )

                            # Train model
                            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

                            # Generate predictions
                            pred = model.predict(X_test)

                            return {"predictions": pred, "model": model}

                        # 5. CNN models
                        elif model_type == "cnn":
                            try:
                                from src.models.cnn_model import CNNPricePredictor

                                # Apply optimizer settings
                                if "batch_size" in settings:
                                    params["batch_size"] = settings["batch_size"]
                                if "learning_rate" in settings:
                                    params["lr"] = settings["learning_rate"]

                                # Create model
                                model = CNNPricePredictor(
                                    input_dim=len(feature_cols),
                                    output_dim=horizon,
                                    num_conv_layers=params.get("num_conv_layers", 3),
                                    num_filters=params.get("num_filters", 64),
                                    kernel_size=params.get("kernel_size", 3),
                                    stride=params.get("stride", 1),
                                    dropout_rate=params.get("dropout_rate", 0.2),
                                    activation=params.get("activation", "relu"),
                                    use_adaptive_pooling=params.get(
                                        "use_adaptive_pooling", True
                                    ),
                                    fc_layers=params.get("fc_layers", [128, 64]),
                                    lookback=params.get("lookback", lookback),
                                    learning_rate=params.get("lr", 0.001),
                                    batch_size=params.get("batch_size", 32),
                                    epochs=params.get("epochs", 10),
                                    early_stopping_patience=params.get(
                                        "early_stopping_patience", 5
                                    ),
                                    verbose=0,
                                )

                                # Train and predict
                                model.fit(X_train, y_train)
                                pred = model.predict(X_test)

                                return {"predictions": pred, "model": model}

                            except ImportError:
                                return {"error": "CNN model not available"}

                        # 6. Liquid Time-Constant (LTC) models
                        elif model_type == "ltc":
                            try:
                                from src.models.ltc_model import build_ltc_model

                                # Apply optimizer settings
                                if "batch_size" in settings:
                                    params["batch_size"] = settings["batch_size"]
                                if "learning_rate" in settings:
                                    params["lr"] = settings["learning_rate"]

                                # Build model
                                model = build_ltc_model(
                                    num_features=len(feature_cols),
                                    horizon=horizon,
                                    learning_rate=params.get("lr", 0.001),
                                    loss_function=params.get(
                                        "loss_function", "mean_squared_error"
                                    ),
                                    lookback=lookback,
                                    units=params.get("units", 64),
                                )

                                # Train and predict
                                model.fit(
                                    X_train,
                                    y_train,
                                    epochs=params.get("epochs", 10),
                                    batch_size=params.get("batch_size", 32),
                                    verbose=0,
                                )
                                pred = model.predict(X_test, verbose=0)

                                return {"predictions": pred, "model": model}

                            except ImportError:
                                return {"error": "LTC model not available"}

                        # Unknown model type
                        else:
                            logger.warning(f"Unknown model type: {model_type}")
                            return {"error": f"Unknown model type: {model_type}"}

                    except Exception as e:
                        logger.error(f"Error training {model_type} model: {e}")
                        import traceback

                        logger.error(traceback.format_exc())
                        return {"error": str(e)}

                # 4. Use training optimizer for parallel model training if enabled
                if enable_parallel:
                    try:
                        # Create model configs
                        model_configs = []
                        for mtype, weight in ensemble_weights.items():
                            if weight <= 0 or mtype not in submodel_params_dict:
                                continue  # Skip models with zero weight

                            # Add configuration for this model type
                            model_configs.append(
                                {
                                    "model_type": mtype,
                                    "params": submodel_params_dict[mtype],
                                    "settings": training_optimizer.get_model_config(
                                        mtype, "medium"
                                    ),
                                }
                            )

                        # Run models in parallel
                        model_results = training_optimizer.run_all_models_parallel(
                            model_configs=model_configs,
                            training_function=train_model_function,
                        )

                        # Update models dict with newly trained models
                        for mtype, result in model_results.items():
                            if "error" not in result and "model" in result:
                                models_dict[mtype] = result["model"]

                    except Exception as e:
                        logger.error(f"Error in parallel training: {e}")
                        logger.info("Falling back to sequential training")
                        enable_parallel = False

                # Sequential fallback if parallel is disabled or fails
                if not enable_parallel:
                    model_results = {}

                    for mtype, weight in ensemble_weights.items():
                        if weight <= 0 or mtype not in submodel_params_dict:
                            continue  # Skip models with zero weight

                        # Create configuration for this model
                        config = {
                            "model_type": mtype,
                            "params": submodel_params_dict[mtype],
                            "settings": training_optimizer.get_model_config(
                                mtype, "small"
                            ),
                        }

                        # Train model
                        result = train_model_function(config)
                        model_results[mtype] = result

                        if "error" not in result and "model" in result:
                            models_dict[mtype] = result["model"]

                # 5. Create ensemble prediction
                ensemble_pred = None
                total_weight = 0.0

                for mtype, result in model_results.items():
                    # Skip models with errors or missing predictions
                    if "error" in result or "predictions" not in result:
                        continue

                    # Get predictions
                    pred = result["predictions"]

                    # Get weight for model type
                    weight = ensemble_weights.get(mtype, 0.0)
                    if weight <= 0:
                        continue

                    # Add to ensemble prediction
                    total_weight += weight

                    if ensemble_pred is None:
                        ensemble_pred = weight * pred
                    else:
                        ensemble_pred += weight * pred

                # 6. Normalize and store predictions
                if ensemble_pred is not None and total_weight > 0:
                    # Normalize by total weight
                    ensemble_pred /= total_weight

                    # Store predictions and actuals
                    all_predictions.append(ensemble_pred)
                    all_actuals.append(y_test)

                    # 7. Create ensemble model
                    ensemble_model = get_ensemble_model(
                        ACTIVE_MODEL_TYPES, models_dict, ensemble_weights
                    )

                    # 8. Update dashboard forecast periodically
                    if update_dashboard and cycle % update_frequency == 0:
                        try:
                            update_dashboard_forecast_function(
                                ensemble_model, df, feature_cols, ensemble_weights
                            )
                        except Exception as e:
                            logger.warning(f"Could not update dashboard forecast: {e}")

                    # Record model weights for visualization if using neural networks
                    if "weight_history" not in st.session_state:
                        st.session_state["weight_history"] = []

                    if cycle % 10 == 0:
                        for mtype, model in models_dict.items():
                            if mtype in ["lstm", "rnn", "tft"] and model is not None:
                                try:
                                    record_weight_snapshot(
                                        model, len(st.session_state["weight_history"])
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error recording weight snapshot: {e}"
                                    )
                else:
                    logger.warning(
                        f"Could not create ensemble prediction for cycle {cycle}"
                    )

                # Track training times
                model_runtimes = {}
                for mtype, result in model_results.items():
                    if isinstance(result, dict) and "runtime" in result:
                        model_runtimes[mtype] = result["runtime"]

                # Adjust resources if significant imbalance detected
                if model_runtimes and hasattr(
                    training_optimizer, "adjust_resources_for_imbalance"
                ):
                    resources_adjusted = (
                        training_optimizer.adjust_resources_for_imbalance(
                            model_runtimes
                        )
                    )
                    if resources_adjusted:
                        logger.info("Resources adjusted for imbalanced models")

                # Clean up memory periodically
                if cycle % 10 == 0:
                    high_usage = training_optimizer.log_memory_usage(
                        f"Walk-forward cycle {cycle}"
                    )
                    if high_usage:
                        training_optimizer.cleanup_memory(level="heavy")
                    else:
                        training_optimizer.cleanup_memory(level="light")

                # Move forward to next window
                current_idx += wf_size
                cycle += 1

            except optuna.exceptions.TrialPruned:
                # Re-raise for Optuna to handle
                logger.info(f"Trial pruned at cycle {cycle}")
                raise
            except Exception as e:
                logger.error(f"Error in cycle {cycle}: {e}")
                logger.error(traceback.format_exc())

                # Try to continue with the next cycle
                current_idx += wf_size
                cycle += 1

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping walk-forward validation")
        # Just continue to return results

    # Calculate final metrics
    metrics = {}
    if all_predictions and all_actuals:
        try:
            # Calculate metrics
            try:
                from src.utils.vectorized_ops import numba_mse

                mse = numba_mse(
                    np.vstack(all_predictions).flatten(),
                    np.vstack(all_actuals).flatten(),
                )
            except:
                # Fallback to regular MSE calculation
                mse = calculate_mse(all_predictions, all_actuals)

            metrics["mse"] = float(mse)
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["mape"] = calculate_mape(all_predictions, all_actuals)

            logger.info(
                f"Walk-forward results: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%"
            )
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics = {"mse": float("inf"), "mape": float("inf"), "rmse": float("inf")}
    else:
        logger.error("No predictions were generated")
        metrics = {"mse": float("inf"), "mape": float("inf"), "rmse": float("inf")}

    # Final update of dashboard forecast
    if update_dashboard:
        try:
            if "ensemble_model" in locals():
                # Use generate_future_forecast to generate forecasts
                future_forecast = generate_future_forecast(
                    ensemble_model=ensemble_model,
                    df=df,
                    feature_cols=feature_cols,
                    lookback=lookback,
                    horizon=PREDICTION_HORIZON,
                )

                # Update dashboard with forecast
                if future_forecast:
                    st.session_state["future_forecast"] = future_forecast
                    st.session_state["last_forecast_update"] = datetime.now()
                    st.session_state["ensemble_weights"] = ensemble_weights
                    logger.info(
                        f"Updated dashboard forecast with {len(future_forecast)} days"
                    )
            else:
                logger.warning("No ensemble model available for dashboard update")
        except Exception as e:
            logger.warning(f"Could not update dashboard forecast: {e}")

    # Create ensemble model if needed
    if "ensemble_model" not in locals() or ensemble_model is None:
        ensemble_model = get_ensemble_model(
            ACTIVE_MODEL_TYPES, models_dict, ensemble_weights
        )

    # Store the trained ensemble model in session state for later use
    st.session_state["current_model"] = ensemble_model

    # Clean up before returning
    training_optimizer.log_memory_usage("end_unified_walk_forward")
    training_optimizer.cleanup_memory(level="medium")

    return ensemble_model, metrics


# Remove the duplicate optimized implementation and make it an alias
unified_walk_forward_optimized = unified_walk_forward

# Keep compatibility with other aliases
run_walk_forward_ensemble_eval = unified_walk_forward


# Update the backward compatibility wrapper function
def run_walk_forward(
    df,
    feature_cols,
    horizon,
    wf_size,
    submodel_params_dict=None,
    ensemble_weights=None,
    trial=None,
    update_frequency=5,
):
    """
    Walk-forward training wrapper for compatibility with existing code.
    Delegates to the unified implementation.
    """
    _, metrics = unified_walk_forward(
        df=df,
        feature_cols=feature_cols,
        submodel_params_dict=submodel_params_dict,
        ensemble_weights=ensemble_weights,
        window_size=wf_size,
        update_dashboard=False,  # Don't update dashboard during tuning
        trial=trial,
        update_frequency=update_frequency,
    )

    return metrics.get("mse", float("inf")), metrics.get("mape", float("inf"))


def train_tabnet_model(X_train, y_train, X_val, y_val, params, feature_names=None):
    """Train a TabNet model with the given parameters"""
    if not HAS_TABNET:
        logger.error("TabNet is not available")
        return None

    try:
        # Apply epochs multiplier from session state if available
        epochs_multiplier = st.session_state.get("epochs_multiplier", 1.0)
        max_epochs = params.get("max_epochs", 200)
        adjusted_max_epochs = max(1, int(max_epochs * epochs_multiplier))

        # Create TabNet model with all tunable parameters
        model = TabNetPricePredictor(
            n_d=params.get("n_d", 64),
            n_a=params.get("n_a", 64),
            n_steps=params.get("n_steps", 5),
            gamma=params.get("gamma", 1.5),
            lambda_sparse=params.get("lambda_sparse", 0.001),
            optimizer_params=params.get("optimizer_params", {"lr": 0.02}),
            feature_names=feature_names,
            max_epochs=adjusted_max_epochs,
            patience=params.get("patience", 15),
            batch_size=params.get("batch_size", 1024),
            virtual_batch_size=params.get("virtual_batch_size", 128),
            momentum=params.get("momentum", 0.02),
        )

        # Train the model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        return model
    except Exception as e:
        logger.error(f"Error training TabNet model: {e}")
        return None
