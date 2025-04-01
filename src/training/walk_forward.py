"""
Implements a unified walk-forward validation scheme...
"""

# Standard library imports
import os
import sys
import logging
import json
import numpy as np
import traceback
from datetime import datetime
import time


# Configure paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("walkforward.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import the registry from incremental_learning
try:
    from src.training.incremental_learning import ModelRegistry
except ImportError as e:
    logger.error(f"Failed to import registry: {e}")
    # Create a placeholder if import fails
    registry = None

# Set environment variables BEFORE importing TensorFlow
from src.utils.env_setup import setup_tf_environment


# Import configuration settings with fallbacks
try:
    from config.config_loader import (
        ACTIVE_MODEL_TYPES,
        MODEL_TYPES,
        PREDICTION_HORIZON,
        START_DATE,
        WALK_FORWARD_DEFAULT,
        WALK_FORWARD_MAX,
        WALK_FORWARD_MIN,
        UPDATE_DURING_WALK_FORWARD,  # Import the new setting
        UPDATE_DURING_WALK_FORWARD_INTERVAL,  # Import the new interval setting
        get_value,
        get_config,
    )
    use_mixed_precision = get_value("hardware.use_mixed_precision", False)
except ImportError as e:
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
        "nbeats",
    ]
    MODEL_TYPES = ACTIVE_MODEL_TYPES.copy()
    PREDICTION_HORIZON = 30
    START_DATE = "2020-01-01"
    WALK_FORWARD_DEFAULT = 30
    WALK_FORWARD_MIN = 3
    WALK_FORWARD_MAX = 180
    UPDATE_DURING_WALK_FORWARD = True  # Default to True if not found in config
    UPDATE_DURING_WALK_FORWARD_INTERVAL = 5  # Default to updating every 5 cycles
    use_mixed_precision = False
    
    # Define a fallback for get_value and get_config functions
    def get_value(path, default=None):
        logger.warning(f"Using fallback value for {path}: {default}")
        return default
    
    def get_config():
        logger.warning("Using fallback empty config")
        return {
            "LOOKBACK": 30,
            "PREDICTION_HORIZON": 30,
            "WALK_FORWARD_DEFAULT": WALK_FORWARD_DEFAULT
        }
    
    logger.warning(f"Config import failed, using fallbacks: {e}")

# Set up TensorFlow environment
tf_env = setup_tf_environment(mixed_precision=use_mixed_precision)

# Third-party imports
import optuna
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Project imports - handle safely with fallbacks
try:
    from src.data.preprocessing import create_sequences
    from src.models.model import build_model_by_type, record_weight_snapshot
    from src.models.model_factory import BaseModel
    from src.models.tabnet_model import TabNetPricePredictor
    HAS_TABNET = True
except ImportError as e:
    logger.warning(f"Some imports failed: {e}")
    HAS_TABNET = False
    
    # Define fallback for create_sequences if import fails
    def create_sequences(df, feature_cols, target_col, lookback, horizon):
        logger.warning("Using fallback create_sequences implementation")
        return None, None

try:
    from src.training.trainer import ModelTrainer
    from src.utils.memory_utils import WeakRefCache, cleanup_tf_session, log_memory_usage
    from src.utils.training_optimizer import TrainingOptimizer, get_training_optimizer
    from src.utils.vectorized_ops import numba_mse, vectorized_sequence_creation
except ImportError as e:
    logger.error(f"Failed to import utility functions: {e}")
    
    # Define minimal fallbacks for critical functions
    def numba_mse(x, y):
        return ((x - y) ** 2).mean()
    
    def vectorized_sequence_creation(df, feature_cols, target_col, lookback, horizon):
        try:
            return create_sequences(df, feature_cols, target_col, lookback, horizon)
        except:
            return None, None
    
    class WeakRefCache:
        def __init__(self):
            self.cache = {}
    
    def log_memory_usage(tag):
        logger.info(f"Memory logging not available: {tag}")
        
    def cleanup_tf_session():
        logger.info("TF session cleanup not available")
        
    def get_training_optimizer():
        class DummyOptimizer:
            def __init__(self):
                self.cpu_count = 1
                self.gpu_count = 0
                
            def get_model_config(self, model_type, size="medium"):
                return {"batch_size": 32, "cpu_cores": 1}
                
            def log_memory_usage(self, tag):
                return False
                
            def cleanup_memory(self, level="light"):
                pass
                
            def run_all_models_parallel(self, model_configs, training_function):
                results = {}
                for config in model_configs:
                    model_type = config["model_type"]
                    results[model_type] = training_function(config)
                return results
                
        return DummyOptimizer()

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

# Get configuration values safely with defaults
try:
    config = get_config()
    LOOKBACK = config.get("LOOKBACK", 30)
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
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("XLA compilation and mixed precision enabled")
        else:
            logger.info("XLA compilation enabled (mixed precision disabled per config)")

        return True
    except Exception as e:
        logger.warning(f"Failed to enable XLA compilation: {e}")
        return False


def generate_future_forecast(model, df, feature_cols, lookback=None, horizon=None):
    """
    Generate predictions for the next 'horizon' days into the future.
    Used by walk-forward validation for forecasting.
    
    Args:
        model: The model to use for prediction
        df: DataFrame with historical data
        feature_cols: Features to use
        lookback: Lookback window size (defaults to model input shape or 30)
        horizon: Prediction horizon (defaults to 30)
        
    Returns:
        List of forecasted values
    """
    try:
        if model is None or df is None or df.empty:
            logger.warning("Cannot generate forecast: model or data is missing")
            return []

        # Set defaults
        if lookback is None:
            # Try to infer from model input shape
            if hasattr(model, 'input_shape') and len(model.input_shape) > 1:
                lookback = model.input_shape[1]
            else:
                lookback = 30
        
        horizon = horizon or 30

        logger.info(f"Generating future forecast: lookback={lookback}, horizon={horizon}")

        # Use consolidated prediction approach
        from src.dashboard.prediction_service import PredictionService
        service = PredictionService(model_instance=model)
        
        # Use market regime if this is an ensemble model
        use_market_regime = hasattr(model, 'models') and hasattr(model, 'weights')
        
        return service.generate_forecast(
            df=df, 
            feature_cols=feature_cols,
            lookback=lookback,
            horizon=horizon,
            market_regime=use_market_regime
        )
        
    except Exception as e:
        logger.error(f"Error generating future forecast: {e}", exc_info=True)
        return []


def update_forecast_in_session_state(
    ensemble_model, df, feature_cols, ensemble_weights=None
):
    """
    Update forecast in session state for dashboard display with confidence and regime information.
    
    Args:
        ensemble_model: The trained ensemble model
        df: DataFrame with historical data
        feature_cols: Feature columns to use
        ensemble_weights: Dictionary of ensemble weights
        
    Returns:
        List of forecast values or None on error
    """
    try:
        # Generate and save forecast data for the dashboard
        lookback = st.session_state.get("lookback", 30)
        forecast_window = st.session_state.get("forecast_window", 30)

        # Use the consolidated forecast generation with confidence and regime analysis
        from src.dashboard.prediction_service import PredictionService
        service = PredictionService(model_instance=ensemble_model)
        
        # Generate forecast with all enhancements
        future_forecast, confidence_scores, confidence_components = service.generate_forecast(
            df=df,
            feature_cols=feature_cols,
            lookback=lookback,
            horizon=forecast_window,
            with_confidence=True,
            market_regime=True
        )

        # Update session state with all the information
        st.session_state["future_forecast"] = future_forecast
        st.session_state["forecast_confidence"] = confidence_scores
        st.session_state["confidence_components"] = confidence_components
        st.session_state["last_forecast_update"] = datetime.now()

        # Store metadata for dashboard display
        if not hasattr(service, 'metadata'):
            service.metadata = {}
            
        service.metadata['current_regime'] = confidence_components.get('market_regime', 'unknown')
        service.metadata['context_similarity'] = confidence_components.get('context_similarity', 0.5)
        
        if hasattr(service, 'market_regime_system'):
            service.metadata['regime_stats'] = service.market_regime_system.get_regime_stats()
            # Also store regime history if available
            if hasattr(service.market_regime_system, 'regime_history'):
                st.session_state['regime_history'] = service.market_regime_system.regime_history
            
        st.session_state['metadata'] = service.metadata

        # Store ensemble weights
        if ensemble_weights:
            st.session_state["ensemble_weights"] = ensemble_weights

        logger.info(f"Updated dashboard forecast with {len(future_forecast)} days and confidence scores")
        return future_forecast
    except Exception as e:
        logger.error(f"Error updating forecast in dashboard: {e}", exc_info=True)
        return None


# Use a dynamic import approach to avoid circular imports
def get_dashboard_update_function():
    """
    Dynamically import and return the dashboard forecast update function.
    
    Returns:
        function: Dashboard update function
    """
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


# Use dynamic function retrieval at the module level to assign this function once
update_dashboard_forecast_function = get_dashboard_update_function()
update_forecast_in_dashboard = update_forecast_in_session_state  # Maintain backward compatibility


def get_ensemble_model(model_types, models_dict, ensemble_weights):
    """
    Create an ensemble model from individual models with given weights.
    
    Args:
        model_types: List of model types
        models_dict: Dictionary of trained models
        ensemble_weights: Dictionary of weights for each model type
        
    Returns:
        EnsembleModel: Model that combines individual models with weights
    """
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
            
        def predict_with_confidence(self, X, verbose=0):
            """Generate weighted ensemble prediction with confidence scores"""
            ensemble_pred = None
            total_weight = 0.0
            individual_preds = {}
            model_uncertainties = {}
            
            for mtype, weight in self.weights.items():
                if weight <= 0 or mtype not in self.models:
                    continue

                model = self.models[mtype]
                if model is None:
                    continue

                try:
                    # Get prediction from this model
                    pred = model.predict(X, verbose=0)
                    
                    # Store individual model prediction
                    individual_preds[mtype] = pred

                    # Collect model-specific uncertainty if available
                    if hasattr(model, 'predict_uncertainty'):
                        # Some models can provide prediction uncertainty
                        try:
                            uncertainty = model.predict_uncertainty(X)
                            model_uncertainties[mtype] = uncertainty
                        except:
                            pass
                    elif hasattr(model, 'predict_proba') and mtype in ['random_forest', 'xgboost']:
                        # For tree-based models, prediction variance can be used
                        try:
                            if mtype == 'random_forest':
                                # For RandomForest, compute variance across trees
                                tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
                                uncertainty = np.std(tree_preds, axis=0) / (np.mean(tree_preds, axis=0) + 1e-8)
                                model_uncertainties[mtype] = uncertainty
                            elif mtype == 'xgboost':
                                # For XGBoost, use prediction variance as uncertainty
                                # This is a simplification; ideally use quantile regression
                                uncertainty = np.ones_like(pred) * 0.2  # default uncertainty
                                model_uncertainties[mtype] = uncertainty
                        except:
                            pass
                            
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
                
                # Calculate confidence scores
                try:
                    # Check for disagreement between models - more disagreement = lower confidence
                    model_disagreement = 0.0
                    if len(individual_preds) > 1:
                        # Calculate standard deviation across model predictions
                        all_preds = np.array([pred for pred in individual_preds.values()])
                        model_std = np.std(all_preds, axis=0)
                        model_mean = np.mean(all_preds, axis=0)
                        model_disagreement = model_std / (model_mean + 1e-6)  # Coefficient of variation
                    
                    # Base confidence on model disagreement (higher disagreement = lower confidence)
                    base_confidence = 80.0 * np.exp(-5.0 * model_disagreement)
                    
                    # Adjust confidence based on forecast horizon (further = less confident)
                    if hasattr(ensemble_pred, 'shape') and len(ensemble_pred.shape) > 1:
                        horizon_decay = np.exp(-0.05 * np.arange(ensemble_pred.shape[1]))
                        confidence_scores = base_confidence * horizon_decay
                    else:
                        horizon_decay = np.exp(-0.05 * np.arange(len(ensemble_pred)))
                        confidence_scores = base_confidence * horizon_decay
                    
                    # Ensure confidence is bounded between 10 and 95
                    confidence_scores = np.clip(confidence_scores, 10, 95)
                    
                    # Create confidence components dictionary
                    confidence_components = {
                        'ensemble': base_confidence,
                        'historical': np.ones_like(confidence_scores) * 70,  # Placeholder
                        'volatility': np.ones_like(confidence_scores) * 60,  # Placeholder
                        'model': np.ones_like(confidence_scores) * 75,      # Placeholder
                        'final': confidence_scores
                    }
                    
                    return ensemble_pred, confidence_scores, confidence_components
                except Exception as e:
                    logger.error(f"Error calculating confidence: {e}")
                    # Return prediction with default confidence
                    default_confidence = np.ones_like(ensemble_pred) * 50
                    return ensemble_pred, default_confidence, {}
            
            # Fallback: use first available model if ensemble fails
            for mtype, model in self.models.items():
                if model is not None:
                    logger.warning(f"Using fallback model {mtype} for prediction")
                    try:
                        pred = model.predict(X, verbose=0)
                        # Return with low confidence
                        return pred, np.ones_like(pred) * 30, {}
                    except:
                        continue

            # Last resort: return zeros
            logger.error("Ensemble prediction failed, returning zeros")
            if isinstance(X, np.ndarray):
                zeros = np.zeros((X.shape[0], 1))
                return zeros, np.zeros_like(zeros), {}
            
            return None, None, {}

    # Create and return the ensemble model
    return EnsembleModel(models_dict, ensemble_weights, model_types)


def _get_default_submodel_params(mtype, optimized=False):
    """
    Get default parameters for a specific model type.
    
    Args:
        mtype: Model type
        optimized: Whether to use optimized parameters
        
    Returns:
        dict: Default parameters
    """
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
    model_type=None,  # Ensure model_type is included in parameters
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
        model_type: Type of the model being trained (default: None)

    Returns:
        tuple: (ensemble_model, metrics_dict) - The trained ensemble model and performance metrics
    """
    # Default values and input validation
    target_col = "Close"  # Default target column
    log_memory_usage("Starting unified_walk_forward")

    # Load model progress and sync with global cycle if model_type is provided
    if model_type is not None:
        try:
            from src.tuning.progress_helper import get_model_progress, get_current_cycle
            import time
            
            # Get model progress
            model_progress = get_model_progress(model_type)
            
            # Wait here until this model's cycle matches the global cycle
            model_cycle = model_progress.get("cycle", 0)
            global_cycle = get_current_cycle()
            
            # Add timeout protection (10 minutes max wait)
            max_wait_seconds = 600
            wait_start_time = time.time()
            
            while model_cycle < global_cycle:
                logger.info(f"[{model_type}] Waiting for cycle sync: model at {model_cycle}, global at {global_cycle}")
                
                # Check timeout
                if time.time() - wait_start_time > max_wait_seconds:
                    logger.warning(f"[{model_type}] Timeout waiting for cycle sync after {max_wait_seconds} seconds")
                    break
                    
                # Wait before checking again
                time.sleep(5)
                global_cycle = get_current_cycle()
        except ImportError as e:
            logger.warning(f"Could not import progress helper modules: {e}")
        except Exception as e:
            logger.warning(f"Error during cycle synchronization: {e}")

    # Check for dashboard update interval in session state
    if hasattr(st, "session_state") and "update_during_walk_forward_interval" in st.session_state:
        # Use the interval from session state if available
        update_frequency = st.session_state["update_during_walk_forward_interval"]
        logger.info(f"Using dashboard update interval from session state: {update_frequency}")
    # Otherwise use the value from config (if imported successfully)
    elif 'UPDATE_DURING_WALK_FORWARD_INTERVAL' in globals():
        update_frequency = UPDATE_DURING_WALK_FORWARD_INTERVAL
        logger.info(f"Using dashboard update interval from config: {update_frequency}")
    # Otherwise use the provided parameter (default: 5)

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

    # Add defensive checks before accessing feature_cols
    if feature_cols is None:
        logger.warning(f"Feature columns are None in unified_walk_forward. Using empty list.")
        feature_cols = []
        
    if df is None:
        error_msg = f"DataFrame is None in unified_walk_forward"
        logger.error(error_msg)
        return None, {"rmse": float('inf'), "mape": float('inf'), "direction_accuracy": 0.0}
    
    # Now it's safe to check for missing columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        
    # Log which features are being used - makes it visible which indicators are active
    logger.info(f"Using {len(feature_cols)} features for model training: {feature_cols}")
    
    # Update session state with features for Model Insights tab
    if hasattr(st, 'session_state'):
        st.session_state["active_features"] = feature_cols
    
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

                    # Log which features this model is using
                    logger.info(f"Training {model_type} model with {len(feature_cols)} features")
                    
                    try:
                        # 1. Neural network models: LSTM, RNN, TFT
                        if model_type in ["lstm", "rnn", "tft"]:
                                     
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

                    # 8. Update dashboard forecast periodically - now respects both config setting and interval
                    if update_dashboard and UPDATE_DURING_WALK_FORWARD and cycle % update_frequency == 0:
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

    # Initialize drift tuning scheduler if not in session state
    if "drift_tuning_scheduler" not in st.session_state:
        try:
            from src.training.drift_scheduler import DriftTuningScheduler
            st.session_state["drift_tuning_scheduler"] = DriftTuningScheduler(
                registry=registry if 'registry' in locals() else None
            )
            logger.info("Initialized drift tuning scheduler")
        except ImportError:
            logger.warning("Could not initialize drift tuning scheduler")

    # After calculating metrics, update scheduler
    if "drift_tuning_scheduler" in st.session_state:
        drift_scheduler = st.session_state["drift_tuning_scheduler"]
        
        # Get ticker and timeframe from session state if available
        ticker = st.session_state.get("ticker", "UNKNOWN")
        timeframe = st.session_state.get("timeframe", "1d")
        
        # Record performance metrics
        if "rmse" in metrics:
            drift_scheduler.record_performance(ticker, timeframe, "rmse", metrics["rmse"])
        if "mape" in metrics:
            drift_scheduler.record_performance(ticker, timeframe, "mape", metrics["mape"])
        
        # Record confidence if available
        if all_predictions and len(all_predictions) > 0:
            # Calculate average confidence
            avg_confidence = np.mean([np.max(pred) for pred in all_predictions[-10:]])
            drift_scheduler.record_confidence(ticker, timeframe, avg_confidence)
        
        # Check if we should tune drift hyperparameters
        should_tune, reason = drift_scheduler.should_tune_now(ticker, timeframe, df)
        
        if should_tune:
            logger.warning(f"Drift hyperparameter tuning triggered: {reason}")
            
            # Get optimization configuration
            opt_config = drift_scheduler.get_optimization_config(ticker, timeframe)
            
            try:
                # Import meta optimizer
                from src.tuning.drift_optimizer import optimize_drift_hyperparameters
                
                # Run optimization
                logger.info(f"Starting drift hyperparameter optimization with {opt_config['n_trials']} trials")
                
                # Define training function with current data
                def train_with_drift_params(drift_hyperparams):
                    from src.training.concept_drift import MultiDetectorDriftSystem, DriftHyperparameters
                    # Create hyperparameters object
                    hp_obj = DriftHyperparameters(**drift_hyperparams)
                    
                    # Create drift detector
                    drift_detector = MultiDetectorDriftSystem(hyperparams=hp_obj)
                    
                    # Run a mini walk-forward validation
                    _, mini_metrics = unified_walk_forward(
                        df=df.iloc[-min(1000, len(df)):].copy(),  # Use subset for speed
                        feature_cols=feature_cols,
                        submodel_params_dict=submodel_params_dict,
                        ensemble_weights=ensemble_weights,
                        window_size=wf_size,
                        update_dashboard=False,
                        drift_detector=drift_detector
                    )
                    
                    return mini_metrics
                
                # Define evaluation function
                def evaluate_params(metrics_dict):
                    return metrics_dict
                
                # Run optimization with resource limits
                best_hyperparams = optimize_drift_hyperparameters(
                    train_function=train_with_drift_params,
                    eval_function=evaluate_params,
                    n_trials=opt_config.get("n_trials", 10),
                    timeout=opt_config.get("timeout", 1800),
                    search_space=opt_config.get("search_space", None),
                    prior_params=opt_config.get("prior_params", None)
                )
                
                # Record optimization result
                result_metrics = train_with_drift_params(best_hyperparams)
                drift_scheduler.record_optimization(ticker, timeframe, best_hyperparams, result_metrics)
                
                # Create new drift detector with optimized hyperparameters
                from src.training.concept_drift import MultiDetectorDriftSystem, DriftHyperparameters
                hp_obj = DriftHyperparameters(**best_hyperparams)
                st.session_state["drift_detector"] = MultiDetectorDriftSystem(
                    hyperparams=hp_obj,
                    base_window_size=wf_size
                )
                
                logger.info(f"Updated drift detector with optimized hyperparameters")
                
            except Exception as e:
                logger.error(f"Error during drift hyperparameter optimization: {e}")

    # Initialize drift detection system if not in st.session_state:
    if "drift_detector" not in st.session_state:
        try:
            from src.training.concept_drift import MultiDetectorDriftSystem, DriftHyperparameters
            import yaml
            
            # Try to load optimized hyperparameters
            hyperparams = None
            try:
                from config.config_loader import get_data_dir
                
                # Get ticker and timeframe if available in session state
                ticker = st.session_state.get("ticker", "UNKNOWN")
                timeframe = st.session_state.get("timeframe", "1d")
                
                hyperparams_dir = os.path.join(get_data_dir(), "Hyperparameters")
                drift_params_file = os.path.join(hyperparams_dir, f"drift_params_{ticker}_{timeframe}.yaml")
                
                if os.path.exists(drift_params_file):
                    with open(drift_params_file, "r") as f:
                        hyperparams_dict = yaml.safe_load(f)
                        hyperparams = DriftHyperparameters(**hyperparams_dict)
                        logger.info(f"Loaded optimized drift hyperparameters from {drift_params_file}")
            except Exception as e:
                logger.warning(f"Could not load drift hyperparameters: {e}")
            
            # Create detector with hyperparameters if available
            st.session_state["drift_detector"] = MultiDetectorDriftSystem(
                hyperparams=hyperparams,
                base_window_size=wf_size,
                statistical_window=wf_size,
                performance_window=wf_size*2
            )
            logger.info("Initialized concept drift detection system")
        except ImportError:
            logger.warning("Could not initialize drift detection system")

    # Check for drift using predictions and actuals
    if "drift_detector" in st.session_state and all_predictions and all_actuals:
        drift_detector = st.session_state["drift_detector"]
        
        # Sample recent predictions for drift detection
        sample_size = min(20, len(all_predictions))
        recent_preds = all_predictions[-sample_size:]
        recent_actuals = all_actuals[-sample_size:]
        
        # Update drift detector with recent results
        drift_detected = False
        for i in range(sample_size):
            pred = recent_preds[i][0] if len(recent_preds[i].shape) > 0 else recent_preds[i]
            actual = recent_actuals[i][0] if len(recent_actuals[i].shape) > 0 else recent_actuals[i]
            
            result, drift_type, drift_score, adaptation = drift_detector.update_error(
                timestamp=time.time(),
                actual=float(actual),
                predicted=float(pred)
            )
            
            if result:
                drift_detected = True
                logger.warning(f"Drift detected during walk-forward validation: {drift_type}, score={drift_score:.4f}")
                
                # Apply adaptation to ensemble weights
                if "ensemble_weight_strategy" in adaptation:
                    # Get adjusted weights
                    adjusted_weights = drift_detector.get_ensemble_weight_adjustments(ensemble_weights)
                    ensemble_weights = adjusted_weights
                    logger.info(f"Adjusted ensemble weights: {ensemble_weights}")
                
                # Apply adaptation to window size
                if "window_size_factor" in adaptation:
                    old_wf_size = wf_size
                    wf_size = drift_detector.get_adaptive_window_size(wf_size)
                    logger.info(f"Adjusted window size: {old_wf_size} -> {wf_size}")
                
                # Store adaptation in session state
                st.session_state["last_drift_adaptation"] = adaptation
                
                # Update dashboard with drift info if enabled
                if update_dashboard:
                    if "drift_events" not in st.session_state:
                        st.session_state["drift_events"] = []
                    
                    st.session_state["drift_events"].append({
                        "timestamp": datetime.now(),
                        "type": drift_type,
                        "score": drift_score,
                        "adaptation": adaptation
                    })
        
        # Store visualization data in session state
        viz_data = drift_detector.get_visualization_data()
        if viz_data:
            st.session_state["drift_visualization"] = viz_data
    
    # Final update of dashboard forecast - still respect the config setting
    if update_dashboard and UPDATE_DURING_WALK_FORWARD:
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


# Compatibility aliases
unified_walk_forward_optimized = unified_walk_forward
run_walk_forward_ensemble_eval = unified_walk_forward


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
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        horizon: Prediction horizon
        wf_size: Walk-forward window size
        submodel_params_dict: Dictionary of model parameters for each type
        ensemble_weights: Dictionary of weights for ensemble
        trial: Optuna trial object for hyperparameter tuning
        update_frequency: How often to report progress
        
    Returns:
        tuple: (mse, mape) - Mean squared error and mean absolute percentage error
    """
    _, metrics = unified_walk_forward(
        df=df,
        feature_cols=feature_cols,
        submodel_params_dict=submodel_params_dict,
        ensemble_weights=ensemble_weights,
        window_size=wf_size,
        update_dashboard=True,  # Update dashboard during tuning
        trial=trial,
        update_frequency=update_frequency,
    )

    return metrics.get("mse", float("inf")), metrics.get("mape", float("inf"))


def train_tabnet_model(X_train, y_train, X_val, y_val, params, feature_names=None):
    """
    Train a TabNet model with the given parameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: Dictionary of model parameters
        feature_names: List of feature names
        
    Returns:
        TabNetPricePredictor: Trained model or None if error
    """
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


def perform_walkforward_validation(ticker, timeframe, range_cat, model_type, params):
    """
    Perform walk-forward validation for a single model type.
    
    Args:
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category
        model_type: Model type
        params: Dictionary of model parameters
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        # Import necessary modules
        from src.data.data import fetch_data
        from src.features.features import feature_engineering_with_params
        
        # Fetch data
        df = fetch_data(ticker, timeframe, range_cat)
        if df is None or df.empty:
            logger.error(f"No data for {ticker} {timeframe} {range_cat}")
            return {"rmse": float("inf"), "mape": float("inf")}
            
        # Feature engineering
        df, feature_cols = feature_engineering_with_params(df, model_type)
        
        # Create model parameters dictionary
        submodel_params_dict = {model_type: params}
        
        # Set weights to use only this model type
        ensemble_weights = {mt: 0.0 for mt in ACTIVE_MODEL_TYPES}
        ensemble_weights[model_type] = 1.0
        
        # Run walk-forward validation
        mse, mape = run_walk_forward(
            df=df,
            feature_cols=feature_cols,
            horizon=PREDICTION_HORIZON,
            wf_size=WALK_FORWARD_DEFAULT,
            submodel_params_dict=submodel_params_dict,
            ensemble_weights=ensemble_weights,
            update_frequency=10,
        )
        
        # Calculate RMSE
        rmse = np.sqrt(mse) if mse != float("inf") else float("inf")
        
        # Return metrics dictionary
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
        }
    except Exception as e:
        logger.error(f"Error in perform_walkforward_validation: {e}")
        return {"rmse": float("inf"), "mape": float("inf")}