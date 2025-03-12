"""
dashboard_model.py

Functions for model creation, training, tuning, loading, and prediction.
"""

import json
import os
import sys
import time
from datetime import datetime

import optuna
import yaml

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

# Import config
from config.config_loader import (
    DATA_DIR,
    N_STARTUP_TRIALS,
)
from config.logger_config import logger

# Import required dashboard modules
from src.dashboard.dashboard.dashboard_error import robust_error_boundary

# Initialize required paths
HYPERPARAMS_DIR = os.path.join(DATA_DIR, "hyperparams")
BEST_PARAMS_FILE = os.path.join(HYPERPARAMS_DIR, "best_params.yaml")
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

# Initialize training optimizer
try:
    from src.utils.training_optimizer import get_training_optimizer

    training_optimizer = get_training_optimizer()
    logger.info(
        f"Training optimizer initialized with {training_optimizer.cpu_count} CPUs, {training_optimizer.gpu_count} GPUs"
    )
except Exception as e:
    logger.warning(f"Could not initialize training optimizer: {e}")
    training_optimizer = None

# Import tuning-related functions
try:
    from src.training.callbacks import StopStudyCallback, create_progress_callback
    from src.tuning.progress_helper import (
        is_stop_requested,
        read_progress_from_yaml,
        read_tuning_status,
        set_stop_requested,
        update_progress_in_yaml,
        write_tuning_status,
    )
    from src.utils.gpu_memory_manager import adaptive_memory_clean
except ImportError as e:
    logger.warning(f"Could not import tuning helpers: {e}")

    # Define fallbacks for critical functions
    def update_progress_in_yaml(progress_data):
        logger.warning("Using fallback update_progress_in_yaml")

    def is_stop_requested():
        return False


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
    """
    # Force clean any stale lock files first
    try:
        from src.utils.threadsafe import cleanup_stale_locks

        cleanup_stale_locks(force=True)
        print("Cleaned up stale lock files before starting tuning")
    except ImportError:
        print("Could not import cleanup_stale_locks - continuing anyway")

    # Check if tuning is already in progress - if yes, just return
    if st.session_state.get("tuning_in_progress", False):
        st.warning(
            "Tuning is already in progress. Use the Stop button if you want to restart."
        )
        return

    # Reset session state to ensure clean start
    st.session_state["tuning_in_progress"] = False

    # Check and reset tuning status from file
    try:
        # Import directly from progress_helper for consistency
        from src.tuning.progress_helper import read_tuning_status, write_tuning_status

        tuning_status = read_tuning_status()

        # Force reset if stale
        if tuning_status.get("is_running", False):
            # Check if the status is stale (older than 30 minutes)
            try:
                start_time = float(tuning_status.get("start_time", 0))
                if time.time() - start_time > 1800:  # 30 minutes
                    print("Found stale tuning status - resetting")
                    write_tuning_status(
                        {
                            "is_running": False,
                            "error": "Reset due to stale status",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    tuning_status["is_running"] = False
                else:
                    st.warning(
                        "Tuning appears to be running in another process. Please stop that process first."
                    )
                    st.session_state["tuning_in_progress"] = True
                    return
            except (ValueError, TypeError):
                # If we can't parse the start_time, assume it's stale
                print("Could not parse start_time from tuning status - resetting")
                write_tuning_status(
                    {
                        "is_running": False,
                        "error": "Reset due to unparseable status",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                tuning_status["is_running"] = False
    except ImportError:
        # Fall back to local version if import fails
        from src.dashboard.dashboard.dashboard_error import read_tuning_status

        tuning_status = read_tuning_status()

    if tuning_status.get("is_running", False):
        st.warning(
            "Tuning is already running in another process. Please stop that process first."
        )
        # Update session state for consistency
        st.session_state["tuning_in_progress"] = True
        return

    # Update session state
    st.session_state["tuning_in_progress"] = True
    st.session_state["tuning_start_time"] = time.time()

    # Write status file to indicate tuning is in progress
    try:
        # Import from progress_helper for consistency
        from src.tuning.progress_helper import (
            read_progress_from_yaml,
            write_tuning_status,
        )

        # Get startup trials from config, defaulting to 5000 if not available
        try:
            from config.config_loader import N_STARTUP_TRIALS

            startup_trials = int(N_STARTUP_TRIALS)
        except (ImportError, ValueError):
            startup_trials = 5000

        # If we have multipliers, adjust the number of trials
        if multipliers and "trials_multiplier" in multipliers:
            total_trials = max(
                100, int(startup_trials * multipliers["trials_multiplier"])
            )
        else:
            total_trials = startup_trials

        tuning_status_data = {
            "ticker": ticker,
            "timeframe": timeframe,
            "is_running": True,
            "start_time": time.time(),
            "timestamp": datetime.now().isoformat(),
        }

        write_tuning_status(tuning_status_data)

        # Read existing progress to preserve cycle value if available
        current_progress = read_progress_from_yaml()

        # Initialize progress file while preserving cycle if it exists for this ticker/timeframe
        current_cycle = 1  # Default to 1

        # Only preserve cycle if ticker and timeframe match
        if (
            current_progress
            and current_progress.get("ticker") == ticker
            and current_progress.get("timeframe") == timeframe
        ):
            current_cycle = current_progress.get("cycle", 1)
            # Log that we're preserving the cycle
            logger.info(f"Preserving cycle {current_cycle} for {ticker}/{timeframe}")

        from src.tuning.progress_helper import update_progress_in_yaml

        initial_progress = {
            "ticker": ticker,
            "timeframe": timeframe,
            "current_trial": 0,
            "total_trials": total_trials,
            "cycle": current_cycle,  # Use preserved cycle value
            "timestamp": time.time(),
        }
        update_progress_in_yaml(initial_progress)

        # Start tuning process
        from src.tuning.meta_tuning import start_tuning_process

        start_tuning_process(ticker, timeframe, multipliers)

        st.success(f"Started hyperparameter tuning for {ticker} ({timeframe})")
        st.experimental_rerun()  # Force rerun to update UI
    except Exception as e:
        st.error(f"Failed to start tuning: {e}")
        # Reset status on error
        st.session_state["tuning_in_progress"] = False

        # Update status file
        try:
            from src.tuning.progress_helper import write_tuning_status

            write_tuning_status(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "is_running": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            pass  # Just ignore if we can't update the file


@robust_error_boundary
def stop_tuning():
    """Stop the currently running hyperparameter tuning process."""
    try:
        # Import tuning function
        from src.tuning.meta_tuning import stop_tuning_process

        # Stop tuning process
        result = stop_tuning_process()

        # Update session state
        st.session_state["tuning_in_progress"] = False

        # Update status file directly
        from src.tuning.progress_helper import write_tuning_status

        write_tuning_status(
            {
                "is_running": False,
                "stopped_manually": True,
                "stop_time": time.time(),
                "timestamp": datetime.now().isoformat(),
            }
        )

        st.success("Stopped hyperparameter tuning")
        st.experimental_rerun()  # Force rerun to update UI
        return result
    except Exception as e:
        st.error(f"Failed to stop tuning: {e}")
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
        if model is None:
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
    study_name, storage_name, direction="minimize", n_startup_trials=100
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
