"""
dashboard_model.py

Functions for model creation, training, tuning, loading, and prediction.
"""

import os
import sys
import threading
import time
import traceback
import yaml
from datetime import datetime

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.dashboard.dashboard_error import robust_error_boundary, write_tuning_status
from config.config_loader import (
    DATA_DIR, 
    N_STARTUP_TRIALS,
    TUNING_TRIALS_PER_CYCLE_min,
    TUNING_TRIALS_PER_CYCLE_max
)
from config.logger_config import logger
from src.utils.training_optimizer import get_training_optimizer

# Initialize training optimizer
try:
    training_optimizer = get_training_optimizer()
    logger.info(f"Training optimizer initialized with {training_optimizer.cpu_count} CPUs, {training_optimizer.gpu_count} GPUs")
except Exception as e:
    logger.warning(f"Could not initialize training optimizer: {e}")
    training_optimizer = None


@robust_error_boundary
def generate_future_forecast(model, df, feature_cols, lookback=30, horizon=30):
    """Generate predictions for the next 'horizon' days into the future"""
    try:
        if model is None or df is None or df.empty:
            return []

        # Use values from session state if not provided
        lookback = lookback or st.session_state.get("lookback", 30)
        horizon = horizon or st.session_state.get("forecast_window", 30)

        # Get the last 'lookback' days of data for input
        last_data = df.iloc[-lookback:].copy()

        # Create a scaler for feature normalization
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(last_data[feature_cols])

        # Initialize array to store predictions
        future_prices = []
        current_data = last_data.copy()

        # Create sequences for prediction
        try:
            from src.data.preprocessing import create_sequences

            X_input, _ = create_sequences(
                current_data, feature_cols, "Close", lookback, 1
            )

            # Make prediction for full horizon at once if model supports it
            if hasattr(model, "predict") and callable(model.predict):
                preds = model.predict(X_input, verbose=0)
                if isinstance(preds, np.ndarray) and preds.shape[1] >= horizon:
                    # If model can predict full horizon at once
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

            return future_prices
        except Exception as e:
            logger.error(f"Error in sequence prediction: {e}", exc_info=True)
            return []

    except Exception as e:
        logger.error(f"Error generating future forecast: {e}", exc_info=True)
        return []


@robust_error_boundary
def start_tuning(ticker, timeframe, multipliers=None):
    """
    Start hyperparameter tuning for the specified ticker and timeframe.
    
    Args:
        ticker (str): The ticker symbol (e.g. "BTC-USD")
        timeframe (str): The timeframe (e.g. "1d")
        multipliers (dict, optional): Dictionary of multipliers for tuning parameters
    """
    # Update tuning status to prevent multiple tuning processes
    st.session_state["tuning_in_progress"] = True
    st.session_state["tuning_start_time"] = time.time()
    
    # Write status to file so other processes know tuning is in progress
    write_tuning_status(ticker, timeframe, is_running=True)
    
    try:
        # Import tuning function only when needed
        from src.tuning.meta_tuning import start_tuning_process
        
        # Start tuning in a separate process
        start_tuning_process(ticker, timeframe, multipliers)
        
        st.success(f"Started hyperparameter tuning for {ticker} ({timeframe})")
    except Exception as e:
        st.error(f"Failed to start tuning: {e}")
        # Reset status on error
        st.session_state["tuning_in_progress"] = False
        write_tuning_status(ticker, timeframe, is_running=False)


@robust_error_boundary
def stop_tuning():
    """Stop the currently running hyperparameter tuning process."""
    try:
        # Import tuning function only when needed
        from src.tuning.meta_tuning import stop_tuning_process
        
        # Stop the tuning process
        stop_tuning_process()
        
        # Reset tuning status
        st.session_state["tuning_in_progress"] = False
        write_tuning_status(None, None, is_running=False)
        
        st.success("Stopped hyperparameter tuning")
    except Exception as e:
        st.error(f"Failed to stop tuning: {e}")


@robust_error_boundary
def load_model(model_path, model_type=None):
    """
    Load a model from disk with proper handling for different model types.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    try:
        # Try different loading strategies based on model_type or file structure
        if model_type == 'nbeats' or (os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'nbeats_config.json'))):
            # Load NBEATS model
            try:
                from src.models.nbeats_model import NBEATSModel
                return NBEATSModel.load(model_path)
            except ImportError:
                raise ImportError("NBEATS model module not available")
                # Default TensorFlow model loading
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except Exception as tf_error:
            # If TensorFlow loading fails, try other formats
            try:
                import joblib
                return joblib.load(model_path)
            except Exception as joblib_error:
                # Try pickle as a last resort
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as pickle_error:
                    # Combine all errors for better debugging
                    raise ValueError(f"Failed to load model: TensorFlow error: {tf_error}, "
                                    f"Joblib error: {joblib_error}, "
                                    f"Pickle error: {pickle_error}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        raise


@robust_error_boundary
def model_save_load_controls():
    """Controls for saving and loading models"""
    st.subheader("Model Save/Load Options")

    # Use session state with default to prevent KeyError
    if "saved_model_dir" not in st.session_state:
        st.session_state["saved_model_dir"] = "saved_models"

    st.session_state["saved_model_dir"] = st.text_input(
        "Models Directory", value=st.session_state["saved_model_dir"]
    )

    if "continue_from_old_weights" not in st.session_state:
        st.session_state["continue_from_old_weights"] = False

    st.session_state["continue_from_old_weights"] = st.checkbox(
        "Continue from old weights if available?",
        value=st.session_state["continue_from_old_weights"],
    )

    # Add model type selection for loading
    model_types = ["auto", "lstm", "nbeats", "random_forest", "xgboost", "tabnet"]
    selected_model_type = st.selectbox("Model Type", model_types, index=0)

    if st.button("Save Current Model"):
        model = st.session_state.get("current_model")
        if (model is None):
            st.warning("No model in session to save.")
        else:
            save_dir = st.session_state["saved_model_dir"]
            try:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(save_dir, f"model_{timestamp}")

                # Try to save based on model type
                try:
                    # First try using the model's save method
                    model.save(model_path)
                    st.success(f"Model saved at: {model_path}")
                except AttributeError:
                    # If the model doesn't have a save method, try importing tensorflow
                    import tensorflow as tf
                    tf.keras.models.save_model(model, model_path)
                    st.success(f"Model saved at: {model_path}")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
                logger.error(f"Error saving model: {str(e)}", exc_info=True)

    st.write("### Load a Previously Saved Model")
    load_path = st.text_input("Path to model folder/file", key="load_model_path")
    if st.button("Load Model"):
        if not load_path:
            st.warning("Please enter a valid model path to load.")
            return

        if not os.path.exists(load_path):
            st.error(f"Path does not exist: {load_path}")
            return

        try:
            # Use the enhanced loading function
            model_type = None if selected_model_type == "auto" else selected_model_type
            loaded_model = load_model(load_path, model_type=model_type)
            st.session_state["current_model"] = loaded_model
            st.session_state["model_loaded"] = True
            st.success(f"Model loaded from: {load_path}")
        except Exception as e:
            st.error(f"Error loading model from {load_path}: {str(e)}")
            logger.error(f"Error loading model: {str(e)}", exc_info=True)


@robust_error_boundary
def display_tested_models():
    """Display models from the tested_models.yaml file"""
    tested_models_file = os.path.join(DATA_DIR, "tested_models.yaml")
    if not os.path.exists(tested_models_file):
        st.info("No tested models available yet.")
        return

    try:
        # Try to use the threadsafe module if available
        try:
            from src.utils.threadsafe import safe_read_yaml
            tested_models = safe_read_yaml(tested_models_file, default=[])
        except ImportError:
            # Fallback if threadsafe module is not available
            with open(tested_models_file, "r") as f:
                tested_models = yaml.safe_load(f) or []
    except Exception as e:
        st.error("Error loading tested models: %s" % str(e))
        tested_models = []

    if tested_models:
        st.subheader("Tested Models")
        try:
            df_tested = pd.DataFrame(tested_models)
            if "trial_number" in df_tested.columns:
                df_tested.sort_values("trial_number", inplace=True)
            st.dataframe(df_tested)
        except Exception as e:
            st.error(f"Error displaying tested models: {e}")
            # Fallback display for tested models
            st.json(tested_models)
    else:
        st.info("No tested models available yet.")


@robust_error_boundary
def train_model_with_params(model, X_train, y_train, X_val, y_val, params):
    """
    Train a model with optimized parameters
    """
    if training_optimizer is not None and params.get("batch_size") is None:
        model_type = params.get("model_type", "generic")
        model_config = training_optimizer.get_model_config(model_type, "medium")
        params["batch_size"] = model_config["batch_size"]


@robust_error_boundary
def write_tuning_status(ticker, timeframe, is_running=False):
    """
    Write tuning status to file for coordination between processes.
    
    Args:
        ticker: The ticker being tuned
        timeframe: The timeframe being tuned
        is_running: Boolean indicating if tuning is active
    """
    status_info = {
        "ticker": ticker,
        "timeframe": timeframe,
        "is_running": is_running,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        from config.config_loader import TUNING_STATUS_FILE
        # Ensure directory exists
        os.makedirs(os.path.dirname(TUNING_STATUS_FILE), exist_ok=True)
        
        with open(TUNING_STATUS_FILE, "w") as f:
            yaml.dump(status_info, f)
    except Exception as e:
        # Handle missing config or write errors
        logger.error(f"Error writing tuning status: {e}")
        try:
            # Try to write to a default location as fallback
            with open(os.path.join(DATA_DIR, "tuning_status.yaml"), "w") as f:
                yaml.dump(status_info, f)
        except Exception as nested_e:
            logger.error(f"Failed to write tuning status file: {nested_e}")

