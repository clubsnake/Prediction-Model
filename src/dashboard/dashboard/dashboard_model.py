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

from dashboard_error import robust_error_boundary, write_tuning_status
from config.config_loader import DATA_DIR
from config.logger_config import logger


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
    """Start hyperparameter tuning with specified multipliers"""
    # Default multipliers if none provided
    if multipliers is None:
        from config.config_loader import get_value
        default_mode = get_value("hyperparameter.default_mode", "normal")
        multipliers = get_value(f"hyperparameter.tuning_modes.{default_mode}", 
                               {"trials_multiplier": 1.0, "epochs_multiplier": 1.0, "timeout_multiplier": 1.0})
    
    # Apply multipliers to tuning parameters
    from config.config_loader import (TUNING_TRIALS_PER_CYCLE_min, 
                                    TUNING_TRIALS_PER_CYCLE_max)
    
    # Calculate adjusted trial counts
    min_trials = max(1, int(TUNING_TRIALS_PER_CYCLE_min * multipliers["trials_multiplier"]))
    max_trials = max(min_trials, int(TUNING_TRIALS_PER_CYCLE_max * multipliers["trials_multiplier"]))
    
    # Set adjusted parameters in session state
    st.session_state["tuning_multipliers"] = multipliers
    st.session_state["tuning_trials_min"] = min_trials
    st.session_state["tuning_trials_max"] = max_trials
    
    # Update epochs multiplier in session state for models to use
    st.session_state["epochs_multiplier"] = multipliers["epochs_multiplier"]
    
    # Start tuning thread
    from src.tuning.meta_tuning import tune_for_combo, set_stop_requested
    import threading
    
    # Reset stop flag
    set_stop_requested(False)
    
    # Use random value between min and max for trials
    import random
    n_trials = random.randint(min_trials, max_trials)
    
    # Start tuning in a thread
    tuning_thread = threading.Thread(
        target=tune_for_combo,
        args=(ticker, timeframe, "all", n_trials, 1),
        daemon=True
    )
    tuning_thread.start()
    
    # Update session state
    st.session_state["tuning_in_progress"] = True
    st.session_state["tuning_thread"] = tuning_thread
    st.session_state["tuning_start_time"] = time.time()
    
    # Set timeout based on multiplier
    timeout_hours = 1.0 * multipliers["timeout_multiplier"]  # 1 hour baseline
    st.session_state["tuning_timeout"] = timeout_hours * 3600  # Convert to seconds
    
    st.success(f"Started tuning with {n_trials} trials (Mode: {multipliers['trials_multiplier']}x)")


@robust_error_boundary
def stop_tuning():
    """Thread-safe way to stop tuning process"""
    try:
        # Import here to avoid circular imports
        from src.tuning.meta_tuning import set_stop_requested
        # Use a single consistent method to request stop
        set_stop_requested(True)

        # Update UI state
        st.warning(
            "⚠️ Stop requested! Waiting for current trial to complete... This may take several minutes."
        )
        st.session_state["stop_request_time"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["tuning_in_progress"] = False  # Mark as stopped in UI immediately

        # Update file-based flag
        write_tuning_status(False)
    except ImportError:
        st.error("Meta tuning module not available. Cannot stop tuning.")
    except Exception as e:
        st.error(f"Error stopping tuning: {e}")
        logger.error(f"Error stopping tuning: {e}", exc_info=True)


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

    if st.button("Save Current Model"):
        model = st.session_state.get("current_model")
        if model is None:
            st.warning("No model in session to save.")
        else:
            save_dir = st.session_state["saved_model_dir"]
            try:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(save_dir, f"model_{timestamp}")

                # Try to import tensorflow directly if needed
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
            # Try to import tensorflow directly
            try:
                import tensorflow as tf

                loaded_model = tf.keras.models.load_model(load_path)
                st.session_state["current_model"] = loaded_model
                st.session_state["model_loaded"] = True
                st.success(f"Model loaded from: {load_path}")
            except ImportError:
                st.error("TensorFlow is not available. Cannot load model.")
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
        st.error(f"Error loading tested models: {e}")
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
