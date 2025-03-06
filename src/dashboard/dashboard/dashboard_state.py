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


@robust_error_boundary
def init_session_state():
    """Initialize session state variables with defensive checks"""
    # Check if already initialized to avoid resetting values
    if "initialized" in st.session_state:
        return

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
