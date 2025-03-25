"""
dashboard_model.py

This module provides the core model management functionality for the dashboard interface.
It handles all model-related operations including:

- Loading and saving trained models
- Generating predictions and forecasts
- Managing hyperparameter tuning processes
- Interfacing with the Optuna optimization framework
- Processing model training results

The module serves as a bridge between the dashboard UI and the underlying
prediction model infrastructure, allowing users to interact with models
through the Streamlit interface.

Dependencies:
- Requires access to the config module for settings
- Uses the prediction_service module for forecasting
- Integrates with the tuning and training modules
- Depends on Optuna for hyperparameter optimization
"""

import os
import json
import sys
import time
from datetime import datetime

import optuna
import yaml

# Add project root to Python path to enable relative imports
# This ensures all project modules are accessible regardless of where
# the dashboard is launched from
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

# Import configuration settings - these control global behavior
# of the model training and prediction processes
from config.config_loader import (
    DATA_DIR,
    N_STARTUP_TRIALS,
    HYPERPARAMS_DIR,  # Import correct path from config_loader
    BEST_PARAMS_FILE, # Import correct path from config_loader
)
from config.logger_config import logger

# Import dashboard error handling utilities for more robust operation
from src.dashboard.dashboard.dashboard_error import robust_error_boundary

# Ensure directories exist
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

# Initialize training optimizer to efficiently allocate computational resources
# based on the available hardware (CPUs/GPUs)
try:
    from src.utils.training_optimizer import get_training_optimizer

    training_optimizer = get_training_optimizer()
    logger.info(
        f"Training optimizer initialized with {training_optimizer.cpu_count} CPUs, {training_optimizer.gpu_count} GPUs"
    )
except Exception as e:
    logger.warning(f"Could not initialize training optimizer: {e}")
    training_optimizer = None

# Import utilities for hyperparameter tuning and progress tracking
try:
    from src.tuning.meta_tuning import create_progress_callback
    from src.training.callbacks import StopStudyCallback
    from src.tuning.progress_helper import (
        is_stop_requested,
        read_progress_from_yaml,
        read_tuning_status,
        set_stop_requested,
        update_progress_in_yaml,
        write_tuning_status,
        get_progress_file_path,  # Import the new function
    )
    from src.utils.gpu_memory_manager import adaptive_memory_clean
except ImportError as e:
    logger.warning(f"Could not import tuning helpers: {e}")

    # Define fallbacks for critical functions to maintain minimum functionality
    # even when full tuning capabilities aren't available
    def update_progress_in_yaml(progress_data):
        logger.warning("Using fallback update_progress_in_yaml")

    def is_stop_requested():
        return False
    
    def get_progress_file_path(file_type, model_type=None):
        return os.path.join(DATA_DIR, f"{file_type}.yaml")

# Set the logger level to DEBUG to capture more detailed messages
import logging
logger.setLevel(logging.DEBUG)

@robust_error_boundary
def generate_future_forecast(model, df, feature_cols, lookback=30, horizon=30):
    """
    Generate predictions for the next 'horizon' days into the future.
    Wrapper around prediction_service.generate_forecast for dashboard compatibility.
    
    Args:
        model: The trained model
        df: DataFrame with historical data
        feature_cols: Feature columns to use
        lookback: Number of past days to use for input
        horizon: Number of days to forecast
        
    Returns:
        List of forecasted values
    """
    try:
        if model is None or df is None or df.empty:
            return []

        # Use values from session state if not provided
        lookback = lookback or st.session_state.get("lookback", 30)
        horizon = horizon or st.session_state.get("forecast_window", 30)

        # Use PredictionService for consistency
        from src.dashboard.prediction_service import PredictionService
        service = PredictionService(model_instance=model)
        
        return service.generate_forecast(df, feature_cols, lookback, horizon)
    except Exception as e:
        logger.error(f"Error generating future forecast: {e}", exc_info=True)
        return []


@robust_error_boundary
def start_tuning(ticker, timeframe, multipliers=None):
    """Start the tuning process"""
    logger.info(f"start_tuning called with ticker={ticker} timeframe={timeframe}")
    print(f"Starting tuning for {ticker} {timeframe}")
    
    # Validate inputs
    if not ticker or not timeframe:
        logger.error("Ticker or timeframe not provided")
        st.error("Ticker and timeframe must be provided")
        return False
    
    logger.info("Reading existing tuning status")
    tuning_status = read_tuning_status()
    logger.debug(f"Current tuning status: {tuning_status}")
    
    if tuning_status.get("is_running", False):
        logger.info("Tuning already in progress, checking if status is stale")
        try:
            start_time = float(tuning_status.get("start_time", 0))
            if time.time() - start_time > 1800:  # stale if >30 minutes
                logger.warning("Found stale tuning status; resetting status")
                write_tuning_status({
                    "is_running": False,
                    "error": "Reset due to stale status",
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                current_ticker = tuning_status.get("ticker")
                current_timeframe = tuning_status.get("timeframe")
                
                # If the ticker or timeframe is different and we're running
                if current_ticker and current_timeframe and (current_ticker != ticker or current_timeframe != timeframe):
                    logger.warning(f"Tuning in progress for {current_ticker}/{current_timeframe}; requested {ticker}/{timeframe}")
                    
                    # Show a warning with options to stop or reset
                    st.warning(f"Tuning is already in progress for {current_ticker}/{current_timeframe}. Do you want to stop it and start tuning for {ticker}/{timeframe}?")
                    
                    # Create three columns for actions
                    stop_col, reset_col, _ = st.columns(3)
                    
                    with stop_col:
                        if st.button("Stop current tuning"):
                            if stop_tuning():
                                st.success("Tuning stopped. Please click 'Start Tuning' again.")
                            else:
                                st.error("Failed to stop tuning. Try using Reset.")
                            return False
                    
                    with reset_col:
                        if st.button("Force Reset Status"):
                            if reset_tuning_status():
                                st.success("Tuning status has been reset. You can now start a new tuning process.")
                                time.sleep(1)  # Short delay before rerun
                                st.experimental_rerun()
                            else:
                                st.error("Failed to reset tuning status.")
                            return False
                    
                    return False
                else:
                    logger.info("Tuning already in progress for the same ticker/timeframe")
                    
                    # Show an error message with a reset option
                    st.error(f"Tuning is already in progress for {ticker}/{timeframe}. Please wait for it to complete or stop it first.")
                    
                    # Add a reset button for stuck processes
                    if st.button("Force Reset Tuning Status"):
                        if reset_tuning_status():
                            st.success("Tuning status has been reset. You can now start a new tuning process.")
                            time.sleep(1)  # Short delay before rerun
                            st.experimental_rerun()
                        else:
                            st.error("Failed to reset tuning status.")
                        
                    return False
        except Exception as e:
            logger.error(f"Exception while checking tuning status: {e}", exc_info=True)
            print(f"Error checking tuning status: {e}")
    
    if not multipliers:
        logger.info("No multipliers provided, setting default multipliers")
        from config.config_loader import N_STARTUP_TRIALS
        multipliers = {
            "n_startup_trials": N_STARTUP_TRIALS,
            "trials_multiplier": 1.0,
            "epochs_multiplier": 1.0,
            "patience_multiplier": 1.0,
            "complexity_multiplier": 1.0,
        }
        print(f"Using default n_startup_trials: {N_STARTUP_TRIALS}")
    
    # FIXED: Set up a flag to avoid recursive reruns
    if st.session_state.get("_tuning_starting", False):
        logger.warning("Tuning startup is already in progress - preventing loop")
        return True
        
    # Mark that we are starting tuning
    st.session_state["_tuning_starting"] = True
    st.session_state["tuning_ticker"] = ticker
    st.session_state["tuning_timeframe"] = timeframe
    
    start_timestamp = datetime.now()
    status_update = {
        "is_running": True,
        "start_time": time.time(),
        "ticker": ticker,
        "timeframe": timeframe,
        "timestamp": start_timestamp.isoformat(),
        "status": "initializing",
        "reset": True
    }
    logger.info(f"Writing initial tuning status: {status_update}")
    print(f"Writing initial status with ticker={ticker}, timeframe={timeframe}")
    
    # Ensure directory exists before writing status
    status_dir = os.path.dirname(os.path.join(DATA_DIR, "tuning_status.txt"))
    os.makedirs(status_dir, exist_ok=True)
    
    write_tuning_status(status_update)
    
    st.session_state["tuning_in_progress"] = True
    st.session_state["last_progress_update"] = time.time()
    
    logger.info(f"Starting tuning for {ticker}/{timeframe} after status update")
    
    try:
        logger.info("Importing start_tuning_process from meta_tuning")
        from src.tuning.meta_tuning import start_tuning_process
        logger.debug("Successfully imported start_tuning_process")
        
        # Create a progress placeholder
        progress_ph = st.empty()
        with progress_ph.container():
            st.info(f"Starting tuning process for {ticker}/{timeframe}...")
            progress_bar = st.progress(0)
        
        # Update progress while waiting for completion
        update_progress_in_yaml({
            "ticker": ticker,
            "timeframe": timeframe,
            "status": "initializing",
            "timestamp": time.time(),
            "is_running": True,
            "cycle": 1
        })
        
        # CRITICAL FIX: Use threading to prevent blocking the Streamlit UI
        import threading
        
        def run_tuning_process():
            try:
                # FIXED: Clear the starting flag at the beginning of the thread
                if "_tuning_starting" in st.session_state:
                    del st.session_state["_tuning_starting"]
                    
                logger.info("Starting tuning process via start_tuning_process (threaded)")
                result = start_tuning_process(ticker=ticker, timeframe=timeframe, multipliers=multipliers, force_start=True)
                logger.info(f"Tuning process completed with result: {result}")
                
                # Update status when complete
                write_tuning_status({
                    "is_running": False,
                    "status": "completed",
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat(),
                    "completion_time": time.time()
                })
                
            except Exception as e:
                logger.error(f"Error in tuning thread: {e}", exc_info=True)
                # Reset status on error
                write_tuning_status({
                    "is_running": False,
                    "error": str(e),
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat()
                })
                
                # FIXED: Also clear the starting flag on exception
                if "_tuning_starting" in st.session_state:
                    del st.session_state["_tuning_starting"]
        
        # Start the tuning in a separate thread
        tuning_thread = threading.Thread(target=run_tuning_process)
        tuning_thread.daemon = False  # FIXED: Keep as False for proper cleanup
        tuning_thread.start()
        
        # FIXED: Sleep briefly to allow the thread to initialize
        # This prevents immediate reruns while the thread is still setting up
        time.sleep(0.5)
        
        # Show success message
        logger.info(f"Tuning thread started for {ticker}/{timeframe}")
        st.success(f"Tuning process for {ticker}/{timeframe} has been launched in the background. Check the 'Model Progress' tab for updates.")
        
        # Force flush all logger handlers to ensure logs are written
        for handler in logger.handlers:
            handler.flush()
        
        # FIXED: Clear the starting flag before returning
        if "_tuning_starting" in st.session_state:
            del st.session_state["_tuning_starting"]
            
        return True
    except Exception as e:
        logger.error(f"Exception in start_tuning: {e}", exc_info=True)
        print(f"DEBUG-ERROR: Failed to run tuning process: {e}")
        st.session_state["tuning_in_progress"] = False
        
        # FIXED: Clear the starting flag on exception
        if "_tuning_starting" in st.session_state:
            del st.session_state["_tuning_starting"]
            
        write_tuning_status({
            "is_running": False,
            "error": str(e),
            "stopped_at": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "timeframe": timeframe
        })
        st.error(f"Failed to start tuning: {e}")
        return False


@robust_error_boundary
def stop_tuning():
    """Stop the currently running hyperparameter tuning process."""
    try:
        # Set stop flag in session state immediately
        st.session_state["tuning_in_progress"] = False
        st.session_state["stopping_tuning"] = True  # Add a stopping flag to prevent reruns
        
        # Try using the progress_helper to set stop flags
        from src.tuning.progress_helper import set_stop_requested, write_tuning_status
        from src.tuning.meta_tuning import stop_tuning_process
        
        # Use centralized tuning stop function if available
        try:
            stop_tuning_process()
            logger.info("Called stop_tuning_process() from meta_tuning")
        except Exception as e:
            logger.warning(f"Could not call stop_tuning_process: {e}")
            # Set the stop flag as fallback
            set_stop_requested(True)
            logger.info("Set stop_requested flag as fallback")
        
        # Update the tuning status file using the global datetime import
        write_tuning_status({
            "is_running": False,
            "stopped_manually": True,
            "stop_time": time.time(),
            "timestamp": datetime.now().isoformat()
        })
        
        # Log the stop
        logger.info("Tuning process manually stopped")
        
        # Add a slight delay to allow messages to propagate
        time.sleep(0.5)
        
        # Show success message but without causing a rerun
        st.success("Tuning process stopped. You can start a new tuning process now.")
        
        return True
    except Exception as e:
        logger.error(f"Error stopping tuning: {e}")
        # Try direct approach as fallback
        try:
            # Write directly to tuning status file
            import os
            import json
            
            status_file = os.path.join("Data", "tuning_status.txt")
            with open(status_file, "w") as f:
                f.write("is_running: False\n")
                f.write("status: stopped_manually\n")
                f.write(f"timestamp: {datetime.now().isoformat()}\n")
                f.write("force_stop: True\n")
            
            # Add a slight delay to allow messages to propagate
            time.sleep(0.5)
            
            return True
        except Exception as e2:
            logger.error(f"Error with fallback stopping: {e2}")
            return False


@robust_error_boundary
def reset_tuning_status():
    """
    Reset tuning status for when it gets stuck in a "running" state.
    This is a forceful reset that should be used when the dashboard 
    incorrectly shows tuning is running when it's not.
    """
    try:
        # Reset session state
        st.session_state["tuning_in_progress"] = False
        if "_tuning_starting" in st.session_state:
            del st.session_state["_tuning_starting"]
        
        # Reset the tuning status file
        from src.tuning.progress_helper import write_tuning_status
        
        # Write a clean status
        write_tuning_status({
            "is_running": False,
            "status": "reset_manually",
            "reset_time": time.time(),
            "timestamp": datetime.now().isoformat()
        })
        
        # Also try to reset stop_requested flag
        try:
            from src.tuning.progress_helper import set_stop_requested
            set_stop_requested(False)
        except Exception as e:
            logger.warning(f"Could not reset stop_requested flag: {e}")
        
        # Clean any lock files
        try:
            status_lock = os.path.join(DATA_DIR, "tuning_status.txt.lock")
            if os.path.exists(status_lock):
                os.remove(status_lock)
                logger.info("Removed tuning_status.txt.lock file")
        except Exception as e:
            logger.warning(f"Could not remove lock file: {e}")
        
        # Log the reset
        logger.info("Tuning status manually reset")
        
        return True
    except Exception as e:
        logger.error(f"Error resetting tuning status: {e}")
        return False


@robust_error_boundary
def load_model(model_path, model_type=None):
    """
    Load a model from disk with proper handling for different model types.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    try:
        # Try different loading strategies based on model_type or file structure
        if model_type == "nbeats" or (
            os.path.isdir(model_path)
            and os.path.exists(os.path.join(model_path, "nbeats_config.json"))
        ):
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

                    with open(model_path, "rb") as f:
                        return pickle.load(f)
                except Exception as pickle_error:
                    # Combine all errors for better debugging
                    raise ValueError(
                        f"Failed to load model: TensorFlow error: {tf_error}, "
                        f"Joblib error: {joblib_error}, "
                        f"Pickle error: {pickle_error}"
                    )
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
        if (model is None) or (not hasattr(model, "save")):
            st.warning("No model in session to save or model does not have a save method.")
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
            import yaml
            
            # Define local function instead of redefining imported one
            def read_yaml_file(file_path, default=None):
                try:
                    with open(file_path, "r") as f:
                        return yaml.safe_load(f) or default
                except Exception:
                    return default
                    
            tested_models = read_yaml_file(tested_models_file, default=[])
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


# First, let's modify the create_resilient_study to call the one from meta_tuning
def create_resilient_study(
    study_name, storage_name, direction="minimize", n_startup_trials=10000
):
    """
    Create an Optuna study with better error handling and fallback to in-memory if needed.
    Delegates to the implementation in meta_tuning.py to avoid code duplication.
    """
    # Import the function from meta_tuning to avoid duplication
    from src.tuning.meta_tuning import create_resilient_study as _create_resilient_study

    return _create_resilient_study(
        study_name, storage_name, direction, n_startup_trials
    )


# Similarly for the objective wrapper
def robust_objective_wrapper(objective_fn):
    """
    Wrap the Optuna objective function to provide better error handling.
    Delegates to the implementation in meta_tuning.py to avoid code duplication.
    """
    # Import the function from meta_tuning to avoid duplication
    from src.tuning.meta_tuning import (
        robust_objective_wrapper as _robust_objective_wrapper,
    )

    return _robust_objective_wrapper(objective_fn)


def ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat):
    """Enhanced objective function for Optuna that exposes all possible parameters for tuning"""
    # ...existing code...

    # Clean up memory
    import gc

    gc.collect()

    if "tf" in sys.modules:
        import tensorflow as tf

        tf.keras.backend.clear_session()

    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate rmse (Root Mean Squared Error)
    import numpy as np
    from sklearn.metrics import mean_squared_error

    y_true = ...  # Replace with actual true values
    y_pred = ...  # Replace with actual predicted values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return rmse


def tune_for_combo(
    ticker, timeframe, range_cat="all", n_trials=None, cycle=1, tuning_multipliers=None
):
    """Run hyperparameter optimization for a specific ticker-timeframe combination."""

    # Define storage_name based on your requirements
    storage_name = "sqlite:///example.db"  # Replace with actual storage path

    # Create a unique study name based on ticker and timeframe
    study_name = f"study_{ticker}_{timeframe}_{range_cat}_cycle{cycle}"

    # Determine number of startup trials
    adjusted_startup_trials = (
        N_STARTUP_TRIALS if "N_STARTUP_TRIALS" in globals() else 5000
    )

    # Initialize tuning_multipliers if None
    if tuning_multipliers is None:
        tuning_multipliers = {}

    study = create_resilient_study(
        study_name=study_name,
        storage_name=storage_name,
        direction="minimize",
        n_startup_trials=adjusted_startup_trials,
    )
    study.set_user_attr("cycle", cycle)
    study.set_user_attr("n_trials", n_trials)
    study.set_user_attr("tuning_multipliers", tuning_multipliers)  # Store for reference
    study.set_user_attr("ticker", ticker)
    study.set_user_attr(
        "using_multipliers", tuning_multipliers is not None and bool(tuning_multipliers)
    )
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
    try:
        if not is_stop_requested() and study.best_trial:
            model_id = register_best_model(study, study.best_trial, ticker, timeframe)

            # If successful, save model ID to best_params.yaml
            if model_id:
                from src.utils.threadsafe import safe_read_yaml, safe_write_yaml

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


@robust_error_boundary
def update_cycle_metrics(cycle_summary, cycle_num=1):
    """
    Update metrics for the current tuning cycle.

    Args:
        cycle_summary: Dictionary with cycle metrics
        cycle_num: Current cycle number
    """
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Create cycle metrics file
        cycle_file = os.path.join(metrics_dir, f"cycle_{cycle_num}_metrics.json")

        with open(cycle_file, "w") as f:
            json.dump(cycle_summary, f, indent=2)

        logger.info(f"Updated cycle metrics for cycle {cycle_num}")
        
        # Also update YAML-based metrics for consistency
        try:
            from src.tuning.progress_helper import update_cycle_metrics as update_yaml_cycle_metrics
            update_yaml_cycle_metrics(cycle_summary, cycle_num)
        except ImportError:
            logger.warning("Could not update YAML cycle metrics - import failed")
            
    except Exception as e:
        logger.error(f"Error updating cycle metrics: {e}")


@robust_error_boundary
def register_best_model(study, trial, ticker, timeframe):
    """
    Register the best model from a completed study.

    Args:
        study: The completed Optuna study
        trial: The best trial from the study
        ticker: Ticker symbol
        timeframe: Timeframe

    Returns:
        str: Model ID if registration succeeded, None otherwise
    """
    try:
        from src.training.incremental_learning import get_model_registry

        registry = get_model_registry()
        if registry is None:
            logger.warning("Model registry not available for registration")
            return None

        # Get model parameters
        params = trial.params
        model_type = params.get("model_type", "unknown")

        # Get metrics
        metrics = {
            "rmse": trial.user_attrs.get("rmse"),
            "mape": trial.user_attrs.get("mape"),
            "direction_accuracy": trial.user_attrs.get("direction_accuracy", 0.0),
            "value": trial.value,
        }

        # Register metadata only since we don't have the actual model object
        model_id = registry.register_model_metadata(
            model_type=model_type,
            ticker=ticker,
            timeframe=timeframe,
            metrics=metrics,
            hyperparams=params,
            tags=[f"study_{study.study_name}", f"trial_{trial.number}", "optuna"],
            description=f"Best model from study {study.study_name}, trial {trial.number}",
        )

        logger.info(f"Registered best model with ID {model_id}")
        return model_id
    except Exception as e:
        logger.error(f"Error registering best model: {e}")
        return None


def get_model_insights(ticker=None, timeframe=None):
    """
    Get insights about model performance and feature usage.
    
    Args:
        ticker: Ticker symbol to filter insights
        timeframe: Timeframe to filter insights
        
    Returns:
        dict: Dictionary of model insights
    """
    insights = {
        "feature_usage": {},
        "performance_metrics": {},
        "feature_importance": {},
        "model_weights": {}
    }
    
    # Get active features
    if "active_features" in st.session_state:
        insights["feature_usage"]["active_features"] = st.session_state["active_features"]
    
    # Get feature importance if available
    if "feature_importance" in st.session_state:
        insights["feature_importance"] = st.session_state["feature_importance"]
    
    # Get ensemble weights if available
    if "ensemble_weights" in st.session_state:
        insights["model_weights"] = st.session_state["ensemble_weights"]
    
    # Get performance metrics if available
    if "best_metrics" in st.session_state:
        insights["performance_metrics"]["best"] = st.session_state["best_metrics"]
    
    # Add drift metrics if available
    if "drift_visualization" in st.session_state:
        insights["drift"] = st.session_state["drift_visualization"]
    
    return insights


@robust_error_boundary
def display_model_metrics(filter_by_cycle=None):
    """Display both combined and individual model metrics in the dashboard."""
    try:
        # Import functions to get metrics
        from src.tuning.progress_helper import get_cycle_metrics, get_individual_model_progress, read_progress_from_yaml
        
        # Get both cycle metrics and aggregated progress data
        cycle_metrics = get_cycle_metrics(filter_by_cycle)
        progress_data = read_progress_from_yaml()
        
        if not cycle_metrics and not progress_data:
            st.info("No cycle metrics or progress data available yet.")
            return
            
        # Display combined metrics first
        st.subheader("Combined Model Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Use ensemble RMSE if available, otherwise best individual RMSE
            rmse = cycle_metrics.get("ensemble_rmse", 
                   cycle_metrics.get("best_rmse", 
                   progress_data.get("best_rmse", "N/A")))
            if isinstance(rmse, (int, float)):
                st.metric("Best RMSE", f"{rmse:.4f}")
            else:
                st.metric("Best RMSE", "N/A")
                
        with col2:
            # Use ensemble MAPE if available, otherwise best individual MAPE
            mape = cycle_metrics.get("ensemble_mape", 
                   cycle_metrics.get("best_mape", 
                   progress_data.get("best_mape", "N/A")))
            if isinstance(mape, (int, float)):
                st.metric("Best MAPE", f"{mape:.2f}%")
            else:
                st.metric("Best MAPE", "N/A")
                
        with col3:
            # Use ensemble DA if available, otherwise best individual DA
            da = cycle_metrics.get("ensemble_directional_accuracy", 
                 cycle_metrics.get("directional_accuracy", "N/A"))
            if isinstance(da, (int, float)):
                st.metric("Direction Acc.", f"{da:.2f}%")
            else:
                st.metric("Direction Acc.", "N/A")
                
        with col4:
            # Get cycle number from either source
            cycle_num = cycle_metrics.get("cycle", progress_data.get("cycle", 1))
            st.metric("Cycle", f"{cycle_num}")
            
            # Display best model if available
            best_model = progress_data.get("best_model", cycle_metrics.get("best_model", None))
            if best_model:
                st.caption(f"Best: {best_model}")
        
        # Show ensemble weights if available
        if "ensemble_weights" in cycle_metrics or "normalized_weights" in cycle_metrics:
            st.subheader("Ensemble Weights")
            weights = cycle_metrics.get("normalized_weights", cycle_metrics.get("ensemble_weights", {}))
            
            # Convert to DataFrame for better display
            weights_data = [{"Model": model, "Weight": f"{weight:.2%}"} 
                           for model, weight in weights.items()]
            
            if weights_data:
                st.dataframe(pd.DataFrame(weights_data))
            
        # Show progress bar for overall completion using aggregated counts
        current = progress_data.get("aggregated_current_trial", 
                 progress_data.get("current_trial", 
                 cycle_metrics.get("current_trial", 0)))
                 
        total = progress_data.get("aggregated_total_trials", 
               progress_data.get("total_trials", 
               cycle_metrics.get("total_trials", 1)))
               
        progress_pct = current / max(1, total)
        st.progress(progress_pct)
        st.caption(f"Overall Progress: {current}/{total} trials ({progress_pct*100:.1f}%)")
            
        # Now display individual model metrics from both sources
        st.subheader("Individual Model Metrics")
        
        # First try to get model metrics from cycle_metrics
        model_data = []
        if "model_metrics" in cycle_metrics and cycle_metrics["model_metrics"]:
            for model_type, metrics in cycle_metrics["model_metrics"].items():
                model_data.append({
                    "Model Type": model_type,
                    "RMSE": f"{metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else "N/A",
                    "MAPE": f"{metrics.get('mape', 'N/A'):.2f}%" if isinstance(metrics.get('mape'), (int, float)) else "N/A",
                    "Dir. Acc.": f"{metrics.get('directional_accuracy', 'N/A'):.2f}%" 
                               if isinstance(metrics.get('directional_accuracy'), (int, float)) else "N/A",
                    "Weight": f"{metrics.get('weight', 0):.2%}" if isinstance(metrics.get('weight'), (int, float)) else "N/A",
                    "Trials": f"{metrics.get('current_trial', metrics.get('completed_trials', 0))}/{metrics.get('total_trials', 'N/A')}"
                })
        
        # If no model metrics in cycle_metrics or empty, get from model_trials in progress_data
        if not model_data and "model_trials" in progress_data:
            model_progress = get_individual_model_progress()
            for model_type, progress in model_progress.items():
                # Get best metrics for this model if available
                is_best_model = model_type == progress_data.get("best_model", "")
                model_data.append({
                    "Model Type": f"{model_type} {'★' if is_best_model else ''}",
                    "Trials": f"{progress.get('current_trial', 0)}/{progress.get('total_trials', 'N/A')}",
                    "Completion": f"{progress.get('completion_percentage', 0):.1f}%",
                    "Status": "Running" if progress.get('current_trial', 0) < progress.get('total_trials', 1) else "Complete"
                })
        
        # Display the model data
        if model_data:
            df = pd.DataFrame(model_data)
            st.dataframe(df)
        else:
            st.info("No individual model metrics available.")
    except Exception as e:
        st.error(f"Error displaying model metrics: {e}")
        logger.error(f"Error in display_model_metrics: {e}", exc_info=True)

