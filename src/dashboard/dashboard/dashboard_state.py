"""
dashboard_state.py

Functions for managing Streamlit session state and dashboard state persistence.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
import logging

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import streamlit, but handle case where it's not available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Warning: Streamlit not available, using fallback state management")

# Setup logger - defined before imports to catch any errors
logger = logging.getLogger("dashboard_state")

# Dictionary to store global state when not using streamlit
_global_state = {}

# Try to import robust_error_boundary with proper error handling
try:
    # Use absolute import to avoid circular imports
    from src.dashboard.dashboard.dashboard_error import robust_error_boundary
except ImportError:
    # Define a simple fallback if dashboard_error is not available
    def robust_error_boundary(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return None
        return wrapper


def get_state(key: str, default=None) -> Any:
    """
    Get state value by key name, works with or without streamlit.

    Args:
        key: State key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The state value or default
    """
    if HAS_STREAMLIT:
        # Use streamlit session state if available
        return st.session_state.get(key, default)
    else:
        # Fall back to global state dictionary
        return _global_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """
    Set state value by key name, works with or without streamlit.

    Args:
        key: State key to set
        value: Value to store
    """
    if HAS_STREAMLIT:
        # Use streamlit session state if available
        st.session_state[key] = value
    else:
        # Fall back to global state dictionary
        _global_state[key] = value


# Convenience functions for accessing common state with error handling
def get_current_ticker() -> str:
    """Get the currently selected ticker symbol with fallback to default"""
    ticker = get_state("selected_ticker")
    if ticker is None:
        # Try to get from config, but handle potential import errors
        try:
            from config.config_loader import TICKER
            ticker = TICKER
        except ImportError:
            ticker = "AAPL"  # Default fallback if config not available
        # Store for future use
        set_state("selected_ticker", ticker)
    return ticker


def get_current_timeframe() -> str:
    """Get the currently selected timeframe with fallback to default"""
    timeframe = get_state("selected_timeframe")
    if timeframe is None:
        # Try to get from config, but handle potential import errors
        try:
            from config.config_loader import TIMEFRAMES
            timeframe = TIMEFRAMES[0] if TIMEFRAMES else "1d"
        except ImportError:
            timeframe = "1d"  # Default fallback if config not available
        # Store for future use
        set_state("selected_timeframe", timeframe)
    return timeframe


def get_lookback_period() -> int:
    """Get the lookback period for model training"""
    lookback = get_state("lookback")
    if lookback is None:
        # Check for historical_window alias first
        lookback = get_state("historical_window", 30)
        # Store it properly
        set_state("lookback", lookback)
    return lookback


def get_forecast_window() -> int:
    """Get the forecast window for predictions"""
    return get_state("forecast_window", 30)


def get_current_model():
    """Get the current model from state"""
    return get_state("current_model")


# ----- PREDICTION MONITOR MANAGEMENT -----
def get_prediction_monitor():
    """Get or create the prediction monitor instance"""
    monitor = get_state("prediction_monitor")
    if monitor is None:
        try:
            # Lazy import to avoid circular dependencies
            from src.dashboard.monitoring import PredictionMonitor

            # Create a new monitor
            logs_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
            )
            # Ensure logs directory exists
            os.makedirs(logs_path, exist_ok=True)
            
            monitor = PredictionMonitor(logs_path=logs_path)
            set_state("prediction_monitor", monitor)
            logger.info("Created new prediction monitor")
        except Exception as e:
            logger.warning(f"Could not create PredictionMonitor: {e}")
            monitor = None

    return monitor


# ----- PREDICTION SERVICE MANAGEMENT -----
def get_prediction_service():
    """Get or create the prediction service instance"""
    service = get_state("prediction_service")
    if service is None:
        try:
            # Lazy import to avoid circular dependencies
            from src.dashboard.prediction_service import PredictionService

            # Get the current model with a defensive check
            model = get_current_model()
            if model is None:
                logger.warning("No model available for prediction service")
                return None
                
            # Create a new service with current context
            service = PredictionService(
                model_instance=model,
                ticker=get_current_ticker(),
                timeframe=get_current_timeframe(),
                monitor=get_prediction_monitor(),
            )
            set_state("prediction_service", service)
            logger.info("Created new prediction service")
        except ImportError:
            logger.warning("PredictionService not available")
        except Exception as e:
            logger.error(f"Error creating prediction service: {e}")
            service = None

    return service


# ----- ENSEMBLE WEIGHTS MANAGEMENT -----
def get_ensemble_weights() -> Dict[str, float]:
    """Get the current ensemble model weights"""
    weights = get_state("ensemble_weights")
    if weights is None:
        # Create default equal weights using model types from config
        try:
            from config.config_loader import ACTIVE_MODEL_TYPES

            weights = {
                model_type: 1.0 / len(ACTIVE_MODEL_TYPES)
                for model_type in ACTIVE_MODEL_TYPES
            }
        except ImportError:
            # Fallback default weights
            weights = {
                "lstm": 0.2,
                "rnn": 0.2,
                "random_forest": 0.2,
                "xgboost": 0.2,
                "tabnet": 0.2,
            }
        set_state("ensemble_weights", weights)
    return weights


def update_ensemble_weights(weights: Dict[str, float]) -> None:
    """
    Update ensemble weights, ensuring they sum to 1.0

    Args:
        weights: Dictionary of model type to weight
    """
    # Ensure weights are valid
    if not weights:
        logger.warning("Empty weights dictionary provided, not updating")
        return

    # Ensure all weights are numeric
    for key, value in list(weights.items()):
        if not isinstance(value, (int, float)):
            logger.warning(f"Non-numeric weight for {key}: {value}, setting to 0")
            weights[key] = 0.0

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total <= 0:
        logger.warning("Invalid weights - sum is zero or negative, using equal weights")
        weights = {k: 1.0/len(weights) for k in weights}
    else:
        weights = {k: v / total for k, v in weights.items()}

    set_state("ensemble_weights", weights)
    logger.info(f"Updated ensemble weights: {weights}")


# ----- INITIALIZATION -----
@robust_error_boundary
def initialize_state():
    """Initialize all required state with default values"""
    # Skip if already initialized
    if HAS_STREAMLIT and get_state("state_initialized", False):
        return

    # Common initial state values (used by all components)
    default_values = {
        "state_initialized": True,
        "last_refresh": time.time(),
        "tuning_in_progress": False,
        "best_metrics": {},
        "model_history": [],
        "prediction_history": [],
        "accuracy_metrics_history": {},
        "neural_network_state": {},
        "current_model": None,
        "model_loaded": False,
        "error_log": [],
        "training_history": None,
        "ensemble_predictions_log": [],
        "df_raw": None,
        "historical_window": 60,
        "forecast_window": 30,
        "lookback": 60,  # Alias for historical_window
        "trials_per_cycle": 1000,  # Default value
        "initial_trials": 10000,  # Default value
        "saved_model_dir": os.path.join(project_root, "saved_models"),
        
        
        # Auto-refresh intervals (in seconds)
        "progress_refresh_sec": 1,
        "full_refresh_sec": 30,
        "last_progress_refresh": time.time(),
        "last_full_refresh": time.time(),
        "update_heavy_components": True,
        
        # UI Control state
        "start_date": datetime.now() - timedelta(days=30),
        "end_date": datetime.now() + timedelta(days=30),
        "training_start_date": datetime.now() - timedelta(days=365),
        "auto_refresh": True,
        "refresh_interval": 30,
        
        # Model data structures
        "model_comparison_data": [],
        "feature_importance_data": {},
        "model_weights_history": [],

        # Containers for progressive display
        "progress_container": None,
        "progress_bar": None,
        "metrics_container": None,
        
        # Status tracking
        "live_progress": {},
        "trial_logs": [],

        # Empty defaults for containers that should always exist
        "past_predictions": {},
    }

    # Try to initialize ticker and timeframe from config
    try:
        from config.config_loader import TICKER, TIMEFRAMES
        default_values["selected_ticker"] = TICKER
        default_values["selected_timeframe"] = TIMEFRAMES[0] if TIMEFRAMES else "1d"
    except ImportError:
        # Use hardcoded defaults if config not available
        default_values["selected_ticker"] = "ETH-USD"
        default_values["selected_timeframe"] = "1d"
        logger.warning("Could not import config for ticker/timeframe, using defaults")

    # Try to read tuning status from file
    try:
        from src.dashboard.dashboard.dashboard_error import read_tuning_status
        status_data = read_tuning_status()
        default_values["tuning_in_progress"] = status_data.get("is_running", False)
        if "ticker" in status_data and status_data["ticker"]:
            default_values["tuning_ticker"] = status_data["ticker"]
        if "timeframe" in status_data and status_data["timeframe"]:
            default_values["tuning_timeframe"] = status_data["timeframe"]
    except Exception as e:
        logger.warning(f"Could not read tuning status: {e}")

    # Set defaults only for keys not already in session_state
    for key, value in default_values.items():
        if get_state(key) is None:
            set_state(key, value)

    # Handle directories that need to exist
    model_dir = get_state("saved_model_dir")
    if model_dir and not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")
        except Exception as e:
            logger.error(f"Could not create model directory: {e}")
            
    # Make sure ensemble weights are initialized
    get_ensemble_weights()
    
    logger.info("Dashboard state initialized")


@robust_error_boundary
def init_session_state():
    """Initialize session state variables with defensive checks"""
    # Create initial refresh timestamp if it doesn't exist
    if get_state("last_refresh") is None:
        set_state("last_refresh", time.time())
        
    # Make sure state is completely initialized
    initialize_state()
    
    # Set initialized flag
    set_state("initialized", True)
    
    if "trial_logs" not in st.session_state:
        st.session_state["trial_logs"] = []
    if "tuning_in_progress" not in st.session_state:
        st.session_state["tuning_in_progress"] = False
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = "ETH-USD"
    if "selected_timeframe" not in st.session_state:
        st.session_state["selected_timeframe"] = "1d"
    
    # Return success indicator
    return True


# Initialize state when this module is imported
try:
    initialize_state()
except Exception as e:
    logger.error(f"Error initializing dashboard state: {e}")
