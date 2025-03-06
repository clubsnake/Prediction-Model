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

# Set environment variables BEFORE importing TensorFlow
from src.utils.env_setup import setup_tf_environment

tf_env = setup_tf_environment()

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

# Import our utilities
from src.utils.vectorized_ops import numba_mse, vectorized_sequence_creation
from sklearn.preprocessing import StandardScaler

from config.config_loader import (
    ACTIVE_MODEL_TYPES,
    MODEL_TYPES,
    PREDICTION_HORIZON,
    START_DATE,
    WALK_FORWARD_DEFAULT,
)
from src.data.preprocessing import create_sequences
from src.models.model import build_model_by_type, record_weight_snapshot
from src.utils import validate_walk_forward
from src.models.model_factory import BaseModel
from src.utils import Visualization
from src.training.trainer import ModelTrainer

# Import TabNet components
try:
    from src.models.tabnet_model import TabNetPricePredictor
    HAS_TABNET = True
    logger.info("TabNet successfully imported")
except ImportError as e:
    HAS_TABNET = False
    logger.warning(f"TabNet import failed: {e}")

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


def enable_xla_compilation():
    """Enable XLA compilation for TensorFlow models"""
    try:
        # Enable XLA
        tf.config.optimizer.set_jit(True)
        # Use mixed precision for suitable GPUs
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("XLA compilation and mixed precision enabled")
        return True
    except Exception as e:
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
        scaler = StandardScaler()
        scaler.fit(last_data[feature_cols])

        # Initialize array to store predictions
        future_prices = []
        current_data = last_data.copy()

        # Create sequences for prediction
        try:
            X_input, _ = create_sequences(
                current_data, feature_cols, "Close", lookback, 1
            )

            # Make prediction for full horizon at once if model supports it
            if hasattr(model, "predict") and callable(model.predict):
                preds = model.predict(X_input, verbose=0)
                if isinstance(preds, np.ndarray) and preds.shape[1] >= horizon:
                    # If model can predict full horizon at once
                    logger.info(f"Generated {horizon}-day forecast at once")
                    return preds[0, :horizon].tolist()

                # Otherwise, predict one day at a time
                next_price = float(preds[0][0])
                future_prices.append(next_price)

                # Continue with iterative prediction for remaining days
                for i in range(1, horizon):
                    # Update data with previous prediction
                    next_row = current_data.iloc[-1:].copy()
                    if isinstance(next_row.index[0], pd.Timestamp):
                        next_row.index = [next_row.index[0] + pd.Timedelta(days=1)]
                    else:
                        next_row.index = [next_row.index[0] + 1]

                    next_row["Close"] = next_price
                    current_data = pd.concat([current_data.iloc[1:], next_row])

                    # Rescale features
                    current_scaled = current_data.copy()
                    current_scaled[feature_cols] = scaler.transform(
                        current_data[feature_cols]
                    )

                    # Create new sequence and predict
                    X_input, _ = create_sequences(
                        current_scaled, feature_cols, "Close", lookback, 1
                    )
                    preds = model.predict(X_input, verbose=0)
                    next_price = float(preds[0][0])
                    future_prices.append(next_price)

                logger.info(f"Generated {len(future_prices)}-day forecast iteratively")

            return future_prices
        except Exception as e:
            logger.error(f"Error in sequence prediction: {e}", exc_info=True)
            return []

    except Exception as e:
        logger.error(f"Error generating future forecast: {e}", exc_info=True)
        return []


def update_forecast_in_dashboard(
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


def unified_walk_forward(
    df,
    feature_cols,
    submodel_params_dict=None,
    ensemble_weights=None,
    training_start_date=None,
    window_size=None,
    update_dashboard=True,
    trial=None,
):
    """
    Unified walk-forward implementation that combines realistic incremental learning
    with ensemble modeling and proper forecast generation.

    Args:
        df: DataFrame with features & target columns
        feature_cols: List of feature column names
        submodel_params_dict: Dictionary of model parameters for each type
        ensemble_weights: Dictionary of weights for ensemble (default: equal weights)
        training_start_date: Start date for training (default: config.START_DATE)
        window_size: Size of each walk-forward step (default: config.WALK_FORWARD_DEFAULT)
        update_dashboard: Whether to update the dashboard forecast (default: True)
        trial: Optuna trial object for hyperparameter tuning (default: None)

    Returns:
        (ensemble_model, metrics_dict): The trained ensemble model and performance metrics
    """
    # Default values and input validation
    target_col = "Close"  # Default target column
    log_memory_usage("Starting unified_walk_forward")

    # Set defaults if not provided
    if ensemble_weights is None:
        # Default to equal weights for active model types
        ensemble_weights = {
            mtype: 1.0 / len(ACTIVE_MODEL_TYPES) for mtype in ACTIVE_MODEL_TYPES
        }

    if submodel_params_dict is None:
        # Create default parameters for each model type
        submodel_params_dict = {}
        for mtype in ACTIVE_MODEL_TYPES:
            if mtype in ["lstm", "rnn", "tft"]:
                submodel_params_dict[mtype] = {
                    "lr": 0.001,
                    "dropout": 0.2,
                    "units_per_layer": [64, 32],
                    "loss_function": "mean_squared_error",
                    "epochs": 1,
                    "batch_size": 32,
                }
            elif mtype == "random_forest":
                submodel_params_dict[mtype] = {"n_est": 100, "mdepth": 10}
            elif mtype == "xgboost":
                submodel_params_dict[mtype] = {"n_est": 100, "lr": 0.1}
            elif mtype == "tabnet":
                submodel_params_dict[mtype] = {
                    "n_d": 64,
                    "n_a": 64,
                    "n_steps": 5,
                    "gamma": 1.5,
                    "lambda_sparse": 0.001,
                    "optimizer_params": {"lr": 0.02},
                    "max_epochs": 200,
                    "patience": 15,
                    "batch_size": 1024,
                    "virtual_batch_size": 128,
                    "momentum": 0.02
                }

    # Determine window size for walk-forward
    wf_size = validate_walk_forward(window_size or WALK_FORWARD_DEFAULT)
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
        if mtype in ["lstm", "rnn", "tft"]:
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

    # Initialize models for each type
    models_dict = {}
    for mtype, weight in ensemble_weights.items():
        if weight <= 0 or mtype not in submodel_params_dict:
            continue  # Skip models with zero weight

        try:
            if mtype in ["lstm", "rnn"]:
                # Initialize neural network models
                arch_params = {
                    "units_per_layer": submodel_params_dict[mtype].get(
                        "units_per_layer", [64, 32]
                    )
                }
                model = build_model_by_type(
                    model_type=mtype,
                    num_features=len(feature_cols),
                    horizon=horizon,
                    learning_rate=submodel_params_dict[mtype].get("lr", 0.001),
                    dropout_rate=submodel_params_dict[mtype].get("dropout", 0.2),
                    loss_function=submodel_params_dict[mtype].get(
                        "loss_function", "mean_squared_error"
                    ),
                    lookback=lookback,
                    architecture_params=arch_params,
                )
                models_dict[mtype] = model
            elif mtype == "tft":
                # Initialize TFT model
                from src.models.temporal_fusion_transformer import build_tft_model

                tft_params = submodel_params_dict[mtype]
                model = build_tft_model(
                    num_features=len(feature_cols),
                    horizon=horizon,
                    learning_rate=tft_params.get("lr", 0.001),
                    hidden_size=tft_params.get("hidden_size", 256),
                    lstm_units=tft_params.get("lstm_units", 256),
                    num_heads=tft_params.get("num_heads", 8),
                    dropout_rate=tft_params.get("dropout", 0.15),
                    loss_function=tft_params.get("loss_function", "mse"),
                )
                models_dict[mtype] = model
            elif mtype == "random_forest":
                from sklearn.ensemble import RandomForestRegressor

                models_dict[mtype] = RandomForestRegressor(
                    n_estimators=submodel_params_dict[mtype].get("n_est", 100),
                    max_depth=submodel_params_dict[mtype].get("mdepth", 10),
                    random_state=42,
                )
            elif mtype == "xgboost":
                import xgboost as xgb

                models_dict[mtype] = xgb.XGBRegressor(
                    n_estimators=submodel_params_dict[mtype].get("n_est", 100),
                    learning_rate=submodel_params_dict[mtype].get("lr", 0.1),
                    random_state=42,
                )
            elif mtype == "tabnet":
                from src.models.tabnet_model import TabNetPricePredictor

                tabnet_params = submodel_params_dict[mtype]
                model = TabNetPricePredictor(
                    n_d=tabnet_params.get('n_d', 64),
                    n_a=tabnet_params.get('n_a', 64),
                    n_steps=tabnet_params.get('n_steps', 5),
                    gamma=tabnet_params.get('gamma', 1.5),
                    lambda_sparse=tabnet_params.get('lambda_sparse', 0.001),
                    optimizer_params=tabnet_params.get('optimizer_params', {"lr": 0.02}),
                    feature_names=feature_cols,
                    max_epochs=tabnet_params.get('max_epochs', 200),
                    patience=tabnet_params.get('patience', 15),
                    batch_size=tabnet_params.get('batch_size', 1024),
                    virtual_batch_size=tabnet_params.get('virtual_batch_size', 128),
                    momentum=tabnet_params.get('momentum', 0.02)
                )
                models_dict[mtype] = model
        except Exception as e:
            logger.error(f"Error initializing model {mtype}: {e}")
            models_dict[mtype] = None

    # Create an initial ensemble model
    ensemble_model = get_ensemble_model(
        ACTIVE_MODEL_TYPES, models_dict, ensemble_weights
    )

    # Initialize storage for predictions and actuals
    all_predictions = []
    all_actuals = []

    # Initial training data - from start_idx to lookback window
    if start_idx + lookback >= len(df):
        logger.error("Not enough data for training with specified lookback")
        return ensemble_model, {"mse": float("inf"), "mape": float("inf")}

    # Set training end cutoff (we don't use future data)
    train_end_idx = start_idx + lookback

    # Calculate number of cycles for progress reporting
    total_samples = len(df)
    total_cycles = (total_samples - train_end_idx - horizon) // wf_size

    # Main walk-forward loop
    current_idx = train_end_idx
    cycle = 0

    while current_idx + horizon < len(df):
        try:
            # Report progress periodically
            if cycle % 5 == 0 or cycle == total_cycles - 1:
                logger.info(
                    f"Walk-forward cycle {cycle+1}/{total_cycles} (idx={current_idx}/{len(df)})"
                )
                # Report to Optuna if provided
                if trial is not None and cycle > 0 and all_predictions and all_actuals:
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
            train_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])

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

            # 4. Train/update each model and collect predictions
            model_predictions = {}
            updated_models = {}

            for mtype, model in models_dict.items():
                weight = ensemble_weights.get(mtype, 0.0)
                if weight <= 0 or model is None:
                    continue

                try:
                    # Train neural networks
                    if mtype in ["lstm", "rnn", "tft"]:
                        # Update with incremental learning (warm start)
                        epochs = submodel_params_dict[mtype].get("epochs", 1)
                        batch_size = submodel_params_dict[mtype].get("batch_size", 32)

                        model.fit(
                            X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0,
                        )
                        pred = model.predict(X_test, verbose=0)
                        updated_models[mtype] = model

                    # Train tree-based models (need to retrain from scratch each time)
                    elif mtype in ["random_forest", "xgboost"]:
                        # Flatten inputs for tree models
                        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
                        y_tr_flat = y_train[:, 0]

                        # Train model
                        model.fit(X_tr_flat, y_tr_flat)

                        # Generate predictions
                        X_te_flat = X_test.reshape(X_test.shape[0], -1)
                        preds_1d = model.predict(X_te_flat)

                        # Match shape of neural net predictions
                        pred = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
                        updated_models[mtype] = model

                    # Train TabNet model
                    elif mtype == "tabnet":
                        # Update with incremental learning (warm start)
                        epochs = submodel_params_dict[mtype].get("max_epochs", 200)
                        batch_size = submodel_params_dict[mtype].get("batch_size", 1024)

                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_test, y_test)],
                            max_epochs=epochs,
                            patience=submodel_params_dict[mtype].get("patience", 15),
                            batch_size=batch_size,
                            virtual_batch_size=submodel_params_dict[mtype].get("virtual_batch_size", 128),
                        )
                        pred = model.predict(X_test)
                        updated_models[mtype] = model

                    # Store predictions
                    model_predictions[mtype] = pred

                except Exception as e:
                    logger.error(
                        f"Error training/predicting with {mtype} in cycle {cycle}: {e}"
                    )
                    # Keep the previous model version
                    updated_models[mtype] = models_dict[mtype]

            # 5. Update models dictionary with newly trained models
            models_dict = updated_models

            # 6. Create ensemble predictions
            ensemble_pred = None
            total_weight = 0.0

            for mtype, pred in model_predictions.items():
                if pred is None:
                    continue

                weight = ensemble_weights.get(mtype, 0.0)
                total_weight += weight

                if ensemble_pred is None:
                    ensemble_pred = weight * pred
                else:
                    ensemble_pred += weight * pred

            # Normalize by total weight
            if ensemble_pred is not None and total_weight > 0:
                ensemble_pred /= total_weight

                # 7. Store predictions and actuals
                all_predictions.append(ensemble_pred)
                all_actuals.append(y_test)

                # 8. Update ensemble model object
                ensemble_model = get_ensemble_model(
                    ACTIVE_MODEL_TYPES, models_dict, ensemble_weights
                )

                # 9. Periodically update dashboard forecast
                if update_dashboard and cycle % 5 == 0:
                    update_forecast_in_dashboard(
                        ensemble_model, df, feature_cols, ensemble_weights
                    )

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
                                logger.error(f"Error recording weight snapshot: {e}")
            else:
                logger.warning(
                    f"Could not create ensemble prediction for cycle {cycle}"
                )

            # Clean up memory periodically
            if cycle % 10 == 0:
                high_usage = log_memory_usage(f"cycle {cycle}")
                if high_usage:
                    cleanup_tf_session()
                    # Recreate any TensorFlow models that were cleared
                    for mtype in ["lstm", "rnn", "tft"]:
                        if mtype in models_dict and models_dict[mtype] is None:
                            # Rebuild the model
                            logger.info(
                                f"Rebuilding {mtype} model after memory cleanup"
                            )
                            models_dict[mtype] = build_model_by_type(
                                model_type=mtype,
                                num_features=len(feature_cols),
                                horizon=horizon,
                                learning_rate=submodel_params_dict[mtype].get(
                                    "lr", 0.001
                                ),
                                dropout_rate=submodel_params_dict[mtype].get(
                                    "dropout", 0.2
                                ),
                                loss_function=submodel_params_dict[mtype].get(
                                    "loss_function", "mean_squared_error"
                                ),
                                lookback=lookback,
                                architecture_params={
                                    "units_per_layer": submodel_params_dict[mtype].get(
                                        "units_per_layer", [64, 32]
                                    )
                                },
                            )

            # Move forward to next window
            current_idx += wf_size
            cycle += 1

        except optuna.exceptions.TrialPruned:
            # Re-raise for Optuna to handle
            logger.info(f"Trial pruned at cycle {cycle}")
            raise
        except Exception as e:
            logger.error(f"Error in cycle {cycle}: {e}\n{traceback.format_exc()}")
            # Try to continue with the next cycle
            current_idx += wf_size
            cycle += 1

    # Calculate final metrics
    metrics = {}
    if all_predictions and all_actuals:
        try:
            metrics["mse"] = calculate_mse(all_predictions, all_actuals)
            metrics["rmse"] = np.sqrt(metrics["mse"])
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
        update_forecast_in_dashboard(ensemble_model, df, feature_cols, ensemble_weights)

    # Store the trained ensemble model in session state for later use
    st.session_state["current_model"] = ensemble_model

    # Clean up before returning
    log_memory_usage("end of unified_walk_forward")

    return ensemble_model, metrics


def calculate_mse(predictions, actuals):
    """Calculate Mean Squared Error between predictions and actuals."""
    if isinstance(predictions, list) and isinstance(actuals, list):
        # If they're lists of arrays, we need to concatenate
        try:
            predictions = np.vstack(predictions)
            actuals = np.vstack(actuals)
        except ValueError:
            logger.error(
                "Error stacking arrays for MSE calculation - shapes may be inconsistent"
            )
            return float("inf")  # Return worst possible score

    # Use numba-accelerated MSE if available
    try:
        return numba_mse(predictions.flatten(), actuals.flatten())
    except:
        # Fallback to numpy
        return np.mean((predictions.flatten() - actuals.flatten()) ** 2)


def calculate_mape(predictions, actuals):
    """Calculate Mean Absolute Percentage Error between predictions and actuals."""
    if isinstance(predictions, list) and isinstance(actuals, list):
        try:
            predictions = np.vstack(predictions)
            actuals = np.vstack(actuals)
        except ValueError:
            logger.error("Error stacking arrays for MAPE calculation")
            return float("inf")

    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    return 100.0 * np.mean(
        np.abs(
            (actuals.flatten() - predictions.flatten()) / (actuals.flatten() + epsilon)
        )
    )


# Alias for backward compatibility
run_walk_forward_ensemble_eval = unified_walk_forward


# This is kept only for backward compatibility
def run_walk_forward(
    model, df, feature_cols, target_col, lookback, horizon, window_size=None
):
    """Legacy function maintained for compatibility - redirects to unified implementation"""
    logger.warning(
        "Using legacy run_walk_forward - consider using unified_walk_forward instead"
    )

    # Create single-model ensemble weights
    if hasattr(model, "__class__") and hasattr(model.__class__, "__name__"):
        model_type = model.__class__.__name__.lower()
        if "lstm" in model_type:
            mtype = "lstm"
        elif "rnn" in model_type:
            mtype = "rnn"
        elif "forest" in model_type:
            mtype = "random_forest"
        elif "xgb" in model_type:
            mtype = "xgboost"
        elif "tft" in model_type:
            mtype = "tft"
        else:
            mtype = "lstm"  # default

        # Create ensemble with just this model
        ensemble_weights = {mt: 0.0 for mt in MODEL_TYPES}
        ensemble_weights[mtype] = 1.0

        # Create params dict
        submodel_params_dict = {
            mtype: {
                "lr": 0.001,
                "dropout": 0.2,
                "units_per_layer": [64],
                "loss_function": "mean_squared_error",
                "lookback": lookback,
                "epochs": 1,
                "batch_size": 32,
            }
        }

        # Store the model in a dict
        models_dict = {mtype: model}

        # Create ensemble model
        get_ensemble_model(MODEL_TYPES, models_dict, ensemble_weights)

        # Run unified implementation
        _, metrics = unified_walk_forward(
            df=df,
            feature_cols=feature_cols,
            submodel_params_dict=submodel_params_dict,
            ensemble_weights=ensemble_weights,
            window_size=window_size,
            update_dashboard=False,
        )

        # Return in expected format
        preds = []
        actuals = []

        return model, preds, actuals
    else:
        # If can't determine model type, return empty results
        logger.error("Could not determine model type in legacy function")
        return model, [], []


# This is kept only for backward compatibility
def run_walk_forward_realistic(
    df, feature_cols, target_col, lookback=30, horizon=7, window_size=7
):
    """Legacy function maintained for compatibility - redirects to unified implementation"""
    logger.warning(
        "Using legacy run_walk_forward_realistic - consider using unified_walk_forward instead"
    )

    # Set up a single LSTM model in ensemble
    ensemble_weights = {mt: 0.0 for mt in MODEL_TYPES}
    ensemble_weights["lstm"] = 1.0

    # Create params dict
    submodel_params_dict = {
        "lstm": {
            "lr": 0.001,
            "dropout": 0.2,
            "units_per_layer": [64],
            "loss_function": "mean_squared_error",
            "lookback": lookback,
            "epochs": 5,  # Initial higher epoch count to match original function
            "batch_size": 32,
        }
    }

    # Run unified implementation
    ensemble_model, _ = unified_walk_forward(
        df=df,
        feature_cols=feature_cols,
        submodel_params_dict=submodel_params_dict,
        ensemble_weights=ensemble_weights,
        window_size=window_size,
        update_dashboard=True,
    )

    # Extract the lstm model from the ensemble
    if hasattr(ensemble_model, "models") and "lstm" in ensemble_model.models:
        model = ensemble_model.models["lstm"]
    else:
        # Create new model if needed
        model = build_model_by_type(
            "lstm", num_features=len(feature_cols), horizon=horizon
        )

    # Return in expected format (empty predictions since they're already in dashboard)
    return (
        model,
        [],
        [],
    )  # For quantile loss, also tune the quantile values        if "quantile_loss" in loss_functions:            quantile_bounds = param_bounds.get("quantile_values", {})            allowed_values = quantile_bounds.get("allowed_values", [0.1, 0.25, 0.5, 0.75, 0.9])                        # Decide how many quantile values to use (between min_count and max_count)            min_count = quantile_bounds.get("min_count", 1)            max_count = min(quantile_bounds.get("max_count", 3), len(allowed_values))            num_quantiles = trial.suggest_int("num_quantiles", min_count, max_count)                        # For each potential quantile slot, decide whether to use it


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
            n_d=params.get('n_d', 64),
            n_a=params.get('n_a', 64),
            n_steps=params.get('n_steps', 5),
            gamma=params.get('gamma', 1.5),
            lambda_sparse=params.get('lambda_sparse', 0.001),
            optimizer_params=params.get('optimizer_params', {"lr": 0.02}),
            feature_names=feature_names,
            max_epochs=adjusted_max_epochs,
            patience=params.get('patience', 15),
            batch_size=params.get('batch_size', 1024),
            virtual_batch_size=params.get('virtual_batch_size', 128),
            momentum=params.get('momentum', 0.02)
        )
        
        # Train the model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)]
        )
        
        return model
    except Exception as e:
        logger.error(f"Error training TabNet model: {e}")
        return None
