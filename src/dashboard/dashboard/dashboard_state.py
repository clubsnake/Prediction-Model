"""
dashboard_state.py

Functions for managing Streamlit session state and dashboard state persistence.
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

from dashboard_error import robust_error_boundary, read_tuning_status
from config.config_loader import (
    DATA_DIR,
    DB_DIR,
    HYPERPARAMS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TICKER,
    TIMEFRAMES,
    N_STARTUP_TRIALS,
)

"""
Central state management module to avoid circular imports between modules.
This module provides access to shared state without directly importing entire modules.
"""
import logging
import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Try to use streamlit if available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    
logger = logging.getLogger(__name__)

# Dictionary to store global state when not using streamlit
_global_state = {}

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

# Convenience functions for accessing common state
def get_current_ticker() -> str:
    """Get the currently selected ticker symbol"""
    return get_state("selected_ticker", "AAPL")

def get_current_timeframe() -> str:
    """Get the currently selected timeframe"""
    return get_state("selected_timeframe", "1d")

def get_lookback_period() -> int:
    """Get the lookback period for model training"""
    return get_state("lookback", 30)

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
            monitor = PredictionMonitor(logs_path=logs_path)
            set_state("prediction_monitor", monitor)
            logger.info("Created new prediction monitor")
        except ImportError:
            logger.warning("PredictionMonitor not available")
            
    return monitor

# ----- PREDICTION SERVICE MANAGEMENT -----

def get_prediction_service():
    """Get or create the prediction service instance"""
    service = get_state("prediction_service")
    if service is None:
        try:
            # Lazy import to avoid circular dependencies
            from src.dashboard.prediction_service import PredictionService
            
            # Create a new service with current context
            service = PredictionService(
                model_instance=get_current_model(),
                ticker=get_current_ticker(),
                timeframe=get_current_timeframe(),
                monitor=get_prediction_monitor()
            )
            set_state("prediction_service", service)
            logger.info("Created new prediction service")
        except ImportError:
            logger.warning("PredictionService not available")
            
    return service

# ----- ENSEMBLE WEIGHTS MANAGEMENT -----

def get_ensemble_weights() -> Dict[str, float]:
    """Get the current ensemble model weights"""
    weights = get_state("ensemble_weights")
    if weights is None:
        # Create default equal weights using model types from config
        try:
            from config.config_loader import ACTIVE_MODEL_TYPES
            weights = {model_type: 1.0/len(ACTIVE_MODEL_TYPES) for model_type in ACTIVE_MODEL_TYPES}
        except ImportError:
            # Fallback default weights
            weights = {
                "lstm": 0.2, 
                "rnn": 0.2, 
                "random_forest": 0.2, 
                "xgboost": 0.2,
                "tabnet": 0.2
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
        return
        
    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total <= 0:
        logger.warning("Invalid weights - sum is zero or negative")
        return
        
    normalized = {k: v/total for k, v in weights.items()}
    set_state("ensemble_weights", normalized)
    logger.info(f"Updated ensemble weights: {normalized}")

# ----- INITIALIZATION -----

def initialize_state():
    """Initialize all required state with default values"""
    if HAS_STREAMLIT and get_state("state_initialized", False):
        return  # Already initialized
        
    # Core state
    set_state("state_initialized", True)
    set_state("last_refresh", time.time())
    
    # Model state
    if get_state("ensemble_weights") is None:
        # Will initialize with defaults
        get_ensemble_weights()
        
    # Dashboard UI state
    if get_state("selected_ticker") is None:
        from config.config_loader import TICKER
        set_state("selected_ticker", TICKER)
        
    if get_state("selected_timeframe") is None:
        from config.config_loader import TIMEFRAMES
        set_state("selected_timeframe", TIMEFRAMES[0] if TIMEFRAMES else "1d")
    
    # Training defaults
    if get_state("lookback") is None:
        set_state("lookback", 30)
        
    if get_state("forecast_window") is None:
        set_state("forecast_window", 30)
        
    # Initialize monitor and service
    get_prediction_monitor()
    get_prediction_service()
    
    logger.info("Dashboard state initialized")

# Initialize state when this module is imported
if HAS_STREAMLIT:
    try:
        initialize_state()
    except Exception as e:
        logger.error(f"Error initializing dashboard state: {e}")

@robust_error_boundary
def init_session_state():
    """Initialize session state variables with defensive checks"""
    # Check if already initialized to avoid resetting values
    if "initialized" in st.session_state:
        return

    # Create a new timestamp if it doesn't exist
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()

    # Set default values
    default_values = {
        "initialized": True,
        "tuning_in_progress": read_tuning_status(),  # Read from file for persistence
        "best_metrics": {},
        "model_history": [],
        "last_refresh": time.time(),
        "prediction_history": [],
        "accuracy_metrics_history": {},
        "neural_network_state": {},
        "current_model": None,
        "model_loaded": False,
        "error_log": [],
        "training_history": None,
        "ensemble_predictions_log": [],
        "df_raw": None,
        "historical_window": 30,
        "forecast_window": 30,
        "trials_per_cycle": N_STARTUP_TRIALS,
        "initial_trials": N_STARTUP_TRIALS,
        "saved_model_dir": "saved_models",
        # Auto-refresh intervals (in seconds)
        "progress_refresh_sec": 2,
        "full_refresh_sec": 30,
        "last_progress_refresh": time.time(),
        "last_full_refresh": time.time(),
        "update_heavy_components": True,
        # UI Control state
        "selected_ticker": TICKER,
        "selected_timeframe": TIMEFRAMES[0] if TIMEFRAMES else "1d",
        "start_date": datetime.now() - timedelta(days=30),
        "end_date": datetime.now(),
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
    }

    # Set defaults only for keys not already in session_state
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Check if the load path exists before attempting to load the model
    if not os.path.exists(st.session_state["saved_model_dir"]):
        st.session_state["model_loaded"] = False


@robust_error_boundary
def handle_auto_refresh():
    """More efficient auto-refresh with memory management"""
    if "auto_refresh" not in st.session_state:
        st.session_state["auto_refresh"] = False

    if not st.session_state.get("auto_refresh", False):
        return False

    now = time.time()

    # Get refresh intervals with defaults
    progress_refresh_sec = st.session_state.get("progress_refresh_sec", 5)
    full_refresh_sec = st.session_state.get("full_refresh_sec", 30)

    # Get last refresh times with defaults
    last_progress_refresh = st.session_state.get("last_progress_refresh", 0)
    last_full_refresh = st.session_state.get("last_full_refresh", 0)

    # Check if progress refresh is needed
    time_since_progress = now - last_progress_refresh
    if time_since_progress >= progress_refresh_sec:
        st.session_state["last_progress_refresh"] = now

        # Clean up memory before updating display
        from dashboard_utils import clean_memory
        clean_memory(force_gc=False)

        # Update only the progress components
        from dashboard_data import update_progress_display
        update_progress_display()

    # Check if full refresh is needed
    time_since_full = now - last_full_refresh
    if time_since_full >= full_refresh_sec:
        st.session_state["last_full_refresh"] = now

        # Clean up memory before full refresh
        from dashboard_utils import clean_memory
        clean_memory(force_gc=True)

        return True  # Signal for full refresh

    return False  # No full refresh needed
