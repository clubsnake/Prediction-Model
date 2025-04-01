"""
Enhanced UI components for the Streamlit dashboard.

This module provides UI components for the main dashboard, including:
- Header and branding elements
- Control panel with user inputs
- Tuning monitoring panels
- Status displays and metrics cards
- Shutdown functionality

The components are designed to be imported and used by dashboard_core.py.
"""

# Add thread-local storage and context preservation
import threading
import streamlit as st

# Create a thread-local storage for each thread
thread_local = threading.local()

# Store the original run function if it exists and hasn't been overridden yet
if not hasattr(st, '_original_run') and hasattr(st, 'run'):
    st._original_run = st.run
    
    # Override the run function to ensure each thread has proper context
    def run_with_context_wrapper(*args, **kwargs):
        # Check if we already attached context to this thread
        if not hasattr(thread_local, 'has_context'):
            # Set the context flag
            thread_local.has_context = True
        
        # Call the original function
        return st._original_run(*args, **kwargs)

    # Replace the original function with our wrapper
    st.run = run_with_context_wrapper

# Fix for ThreadPoolExecutor to maintain context
import concurrent.futures

# Save the original ThreadPoolExecutor if not already saved
if not hasattr(concurrent.futures, '_original_ThreadPoolExecutor'):
    concurrent.futures._original_ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor
    
    # Create a wrapper that preserves context
    def ThreadPoolExecutorWithContext(*args, **kwargs):
        """ThreadPoolExecutor that preserves Streamlit context."""
        
        # Create the executor
        executor = concurrent.futures._original_ThreadPoolExecutor(*args, **kwargs)
        
        # Save the original submit method
        _original_submit = executor.submit
        
        # Override submit to preserve context
        def submit_with_context(fn, *args, **kwargs):
            # Wrap the function to ensure it has context
            def wrapped_fn(*args, **kwargs):
                # Set thread context if needed
                if not hasattr(thread_local, 'has_context'):
                    thread_local.has_context = True
                
                # Run the original function
                return fn(*args, **kwargs)
            
            # Submit the wrapped function
            return _original_submit(wrapped_fn, *args, **kwargs)
        
        # Replace the submit method
        executor.submit = submit_with_context
        
        return executor

    # Replace the original ThreadPoolExecutor
    concurrent.futures.ThreadPoolExecutor = ThreadPoolExecutorWithContext

import time

def diagnose_imports():
    """Test all critical imports and report errors"""
    import sys
    print(f"Python path: {sys.path}")
    
    import_tests = [
        ("src.dashboard.dashboard.dashboard_model", ["start_tuning", "stop_tuning"]),
        ("src.tuning.meta_tuning", ["start_tuning_process", "stop_tuning_process"]),
        ("src.tuning.progress_helper", ["read_tuning_status", "write_tuning_status"]),
        ("src.utils.threadsafe", ["cleanup_stale_locks"]),
    ]
    
    for module_path, functions in import_tests:
        try:
            print(f"Trying to import {module_path}...")
            module = __import__(module_path, fromlist=functions)
            
            for func in functions:
                if hasattr(module, func):
                    print(f"  ✅ {func} found in {module_path}")
                else:
                    print(f"  ❌ {func} NOT found in {module_path}")
                    
            print(f"✅ Successfully imported {module_path}")
        except Exception as e:
            print(f"❌ Failed to import {module_path}: {e}")

# Call at the beginning of your app
diagnose_imports()

# Solution for fixing path issues
def fix_import_paths():
    """Make sure all necessary paths are in sys.path"""
    import os
    import sys
    
    # Get current file location
    current_file = os.path.abspath(__file__)
    dashboard_dir = os.path.dirname(current_file)
    dashboard_parent = os.path.dirname(dashboard_dir)
    src_dir = os.path.dirname(dashboard_parent)
    project_root = os.path.dirname(src_dir)
    
    # Critical paths to add
    paths_to_add = [
        project_root,
        src_dir,
        dashboard_parent,
        os.path.join(src_dir, "tuning")
    ]
    
    # Add paths if not already there
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added path to sys.path: {path}")

fix_import_paths()

def cleanup_stale_locks_on_startup():
    """Clean up any stale lock files at application startup."""
    try:
        from src.utils.threadsafe import cleanup_stale_locks
        from src.utils.threadsafe import cleanup_stale_temp_files
        
        # Use a more aggressive timeout for startup cleanup
        lock_count = cleanup_stale_locks(max_age=60, force=True)  
        temp_count = cleanup_stale_temp_files(max_age=60)
        
        print(f"Startup cleanup: Removed {lock_count} stale lock files and {temp_count} temporary files")
    except Exception as e:
        print(f"Error during startup cleanup: {e}")

cleanup_stale_locks_on_startup()

# Add this function to dashboard_ui.py to always run at startup
def reset_tuning_status_at_startup():
    """Reset tuning status file at application startup and ensure all required directories exist"""
    try:
        # First, create all required directories
        try:
            # Import needed path constants
            from src.tuning.progress_helper import (
                DATA_DIR, MODEL_PROGRESS_DIR, TESTED_MODELS_DIR,
                PROGRESS_FILE, TUNING_STATUS_FILE
            )
            import os
            
            # Create directories
            os.makedirs(DATA_DIR, exist_ok=True)
            os.makedirs(MODEL_PROGRESS_DIR, exist_ok=True)
            os.makedirs(TESTED_MODELS_DIR, exist_ok=True)
            
            # Ensure parent directories of these files exist
            os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
            os.makedirs(os.path.dirname(TUNING_STATUS_FILE), exist_ok=True)
            
            print(f"Created directory structure: {DATA_DIR}")
            
            # Create initial empty files if they don't exist
            if not os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, 'w') as f:
                    f.write('{}')  # Empty JSON object
                print(f"Created initial progress file: {PROGRESS_FILE}")
                
            if not os.path.exists(TUNING_STATUS_FILE):
                with open(TUNING_STATUS_FILE, 'w') as f:
                    f.write('is_running: False\nstatus: initialized\n')
                print(f"Created initial tuning status file: {TUNING_STATUS_FILE}")
                
        except ImportError:
            # Fallback if import fails
            print("Warning: Could not import paths from progress_helper. Using defaults.")
            import os
            
            # Use default paths
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            data_dir = os.path.join(project_root, "data")
            
            # Create default directories
            dirs_to_create = [
                data_dir,
                os.path.join(data_dir, "model_progress"),
                os.path.join(data_dir, "tested_models"),
                os.path.join(data_dir, "models", "hyperparameters")
            ]
            
            for directory in dirs_to_create:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            
            # Create initial files
            status_file = os.path.join(data_dir, "tuning_status.txt")
            progress_file = os.path.join(data_dir, "progress.yaml")
            
            if not os.path.exists(progress_file):
                with open(progress_file, 'w') as f:
                    f.write('{}')
                    
            if not os.path.exists(status_file):
                with open(status_file, 'w') as f:
                    f.write('is_running: False\nstatus: initialized\n')
                    
        # Now reset the tuning status
        from src.tuning.progress_helper import write_tuning_status
        import time
        from datetime import datetime
        
        # Force reset tuning status - always set to not running at startup
        write_tuning_status({
            "is_running": False,
            "status": "reset_at_startup",
            "timestamp": datetime.now().isoformat(),
            "reset": True  # Force total reset of the file
        })
        
        print("Reset tuning status at startup")
        
        # Clear any stale lock files
        try:
            from src.utils.threadsafe import cleanup_stale_locks
            cleaned = cleanup_stale_locks()
            if cleaned > 0:
                print(f"Cleaned {cleaned} stale lock files")
        except Exception as e:
            print(f"Could not clean lock files: {e}")
            
    except Exception as e:
        print(f"Error in startup initialization: {e}")
        import traceback
        traceback.print_exc()

# Call this function when your app starts up
reset_tuning_status_at_startup()

import os
import sys
from datetime import datetime as dt_module
from datetime import timedelta, date as date_type
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# Set up imports with proper error handling
try:
    current_file = os.path.abspath(__file__)
    dashboard_dir = os.path.dirname(current_file)
    dashboard_parent = os.path.dirname(dashboard_dir)
    src_dir = os.path.dirname(dashboard_parent)
    project_root = os.path.dirname(src_dir)
    DATA_DIR = os.path.join(project_root, "data")
    
    # Add project root to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    print(f"Error setting up paths: {e}")
    # Fallback to a reasonable default
    DATA_DIR = "data"

# Import configuration with fallbacks for each import
try:
    from config.config_loader import N_STARTUP_TRIALS, TICKER, TICKERS, TIMEFRAMES
except ImportError:
    # Default values if config import fails
    N_STARTUP_TRIALS = 10000
    TICKER = "ETH-USD"
    TICKERS = ["ETH-USD", "BTC-USD"]
    TIMEFRAMES = ["1d", "1h"]

try:
    from config.logger_config import logger
except ImportError:
    # Create basic logger if logger_config not available
    import logging
    logger = logging.getLogger("dashboard_ui")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

# Import error handling utilities
try:
    from src.dashboard.dashboard.dashboard_error import robust_error_boundary, load_latest_progress, read_tuning_status
except ImportError:
    # Define minimal versions if imports fail
    def robust_error_boundary(func):
        """Simple error boundary decorator"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Error in {func.__name__}: {e}")
                return None
        return wrapper
    
    def load_latest_progress(ticker=None, timeframe=None):
        """Fallback for loading progress"""
        return {
            "current_trial": 0,
            "total_trials": 1,
            "current_rmse": None,
            "current_mape": None,
            "cycle": 1,
        }
    
    def read_tuning_status():
        """Fallback for reading tuning status"""
        return {"status": "unknown", "is_running": False}

# Import tuning functions with fallbacks
try:
    from src.dashboard.dashboard.dashboard_model import start_tuning, stop_tuning
except ImportError:
    # Define fallback functions
    def start_tuning(ticker, timeframe, multipliers=None):
        st.error("Tuning module not available - could not start tuning")
        return False
    
    def stop_tuning():
        st.error("Tuning module not available - could not stop tuning")
        return False

# Initialize default tuning multipliers
tuning_multipliers = {
    "n_startup_trials": N_STARTUP_TRIALS,
}

# Import progress helper modules with proper error handling
try:
    from src.tuning.progress_helper import (
        read_progress_from_yaml, 
        read_tuning_status,
        get_individual_model_progress
    )
except ImportError:
    # Define minimal versions if imports fail
    def read_progress_from_yaml():
        """Fallback for reading progress"""
        return {}
    
    def read_tuning_status():
        """Fallback for reading tuning status"""
        return {"status": "unknown", "is_running": False}
        
    def get_model_specific_trials(model_type=None, limit=50):
        """Fallback for getting model-specific trials"""
        return []


@robust_error_boundary
def create_header():
    """Create a visually appealing header section with app branding."""
    # Use columns for better layout
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown(
            """
        <div style="display: flex; align-items: center;">
            <h1 style="margin: 0; padding: 0;">
                <span style="color: #1E88E5;">📈</span> AI Price Prediction Dashboard
            </h1>
        </div>
        <p style="font-size: 1.1em; color: #455a64;">
            Advanced machine learning for financial market prediction and analysis
        </p>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        # Status indicator with dynamic styling
        if st.session_state.get("tuning_in_progress", False):
            st.markdown(
                """
            <div style="text-align: center; color: #FF9800; font-size: 1.1em;">
                <strong>Tuning in Progress</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style="text-align: center; color: #4CAF50; font-size: 1.1em;">
                <strong>Ready</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        # Last updated timestamp with refresh button
        st.markdown(
            f"""
        <div style="text-align: right; color: #78909c; font-size: 0.9em;">
            Last updated:<br/>{dt_module.now().strftime('%H:%M:%S')}
        </div>
        """,
            unsafe_allow_html=True,
        )
        
        # Add refresh button
        if st.button("🔄 Refresh", key="btn_refresh_dashboard"):
            st.experimental_rerun()

    # Add a horizontal line for visual separation
    st.markdown(
        "<hr style='margin: 0.5rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True
    )


@robust_error_boundary
def create_control_panel() -> Dict:
    """
    Create an enhanced control panel for user inputs with better organization.
    
    Returns:
        Dictionary with all user-selected parameters
    """
    print("DEBUG: create_control_panel function called")
    st.sidebar.markdown(
        """
    <div style="text-align: center;">
        <h2 style="color: #1E88E5;">Control Panel</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Add tuning controls at the top
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if not st.session_state.get("tuning_in_progress", False):
            if st.sidebar.button("🚀 Start Tuning", key="sidebar_btn_start_tuning", use_container_width=True):
                # Get parameters
                ticker = st.session_state.get("selected_ticker", "ETH-USD")
                timeframe = st.session_state.get("selected_timeframe", "1d")
                multipliers = st.session_state.get("tuning_multipliers", {})
                
                try:

                   # Import the start_tuning function  
                    from src.dashboard.dashboard.dashboard_model import start_tuning

                    # Start tuning process
                    ticker = st.session_state.get("selected_ticker", "ETH-USD")
                    timeframe = st.session_state.get("selected_timeframe", "1d")
                    multipliers = st.session_state.get("tuning_multipliers", {})
                    start_tuning(ticker, timeframe, multipliers)

                    # First set tuning status to true, BEFORE starting the actual tuning
                    st.session_state["tuning_in_progress"] = True
                    
                    # Update tuning status file for immediate UI feedback
                    from src.tuning.progress_helper import write_tuning_status
                    from datetime import datetime  # Import datetime class from datetime module
                    write_tuning_status({
                        "is_running": True,
                        "status": "starting",
                        "ticker": ticker,
                        "timeframe": timeframe,
                        "timestamp": datetime.now().isoformat(),
                        "start_time": time.time()
                    })

                    # Force a rerun to update UI BEFORE starting tuning
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Failed to initialize tuning: {str(e)}")
                    logger.error(f"Error initializing tuning UI state: {e}", exc_info=True)
                    st.session_state["tuning_in_progress"] = False
                
    with col2:
        # If tuning is ongoing, give option to stop
        if st.session_state.get("tuning_in_progress", False):
            if st.sidebar.button(
                "⏹️ Stop Tuning", key="sidebar_btn_stop_tuning", use_container_width=True
            ):
                # Try to stop tuning immediately instead of using a flag
                try:
                    # First update the session state to show stopping
                    st.session_state["stopping_tuning"] = True
                    st.session_state["tuning_in_progress"] = False
                    
                    # Update the tuning status file first
                    from src.tuning.progress_helper import write_tuning_status
                    write_tuning_status({
                        "is_running": False,
                        "status": "stopping",
                        "stopped_manually": True,
                        "timestamp": dt_module.now().isoformat(),
                        "stop_time": time.time()
                    })
                    
                    # Then call stop_tuning without rerunning immediately after
                    from src.dashboard.dashboard.dashboard_model import stop_tuning
                    stop_result = stop_tuning()
                    
                    if stop_result:
                        st.sidebar.success("Tuning process has been stopped")
                    else:
                        st.sidebar.warning("Stop request sent, but process may still be running")
                    
                    # Clear any remaining flags - don't rerun here
                    st.session_state.pop("stopping_tuning", None)
                    
                except Exception as e:
                    st.sidebar.error(f"Error stopping tuning: {str(e)}")
                    logger.error(f"Error stopping tuning: {e}", exc_info=True)

    # Process button clicks outside of column contexts
    print(f"DEBUG: Checking for start_tuning_clicked: {st.session_state.get('start_tuning_clicked', False)}")
    if st.session_state.pop("start_tuning_clicked", False):
        # This section can be kept for backward compatibility but won't be reached 
        # with the new direct approach above
        try:
            # Properly validate and convert date parameters
            start_date = st.session_state.get("start_date_user")
            if not isinstance(start_date, date_type):
                logger.warning(f"Invalid start date type: {type(start_date)}, value: {start_date}")
                # Ensure we have a proper date object
                if isinstance(start_date, str):
                    try:
                        start_date = dt_module.strptime(start_date, "%Y-%m-%d").date()
                    except ValueError:
                        # If the string looks like a timeframe ('1d'), use current date
                        logger.warning(f"Using current date - {start_date} is not a valid date")
                        start_date = dt_module.now().date() - timedelta(days=60)
                else:
                    # Default to 60 days ago
                    start_date = dt_module.now().date() - timedelta(days=60)
                st.session_state["start_date_user"] = start_date
            
            # Do the same for training start date
            training_start_date = st.session_state.get("training_start_date_user")
            if not isinstance(training_start_date, date_type):
                logger.warning(f"Invalid training start date: {type(training_start_date)}, value: {training_start_date}")
                # Ensure we have a proper date object
                if isinstance(training_start_date, str):
                    if training_start_date == 'auto':
                        # 'auto' means 5 years ago
                        training_start_date = dt_module.now().date() - timedelta(days=365*5)
                    else:
                        try:
                            training_start_date = dt_module.strptime(training_start_date, "%Y-%m-%d").date()
                        except ValueError:
                            # If the string looks like a timeframe ('1d'), use 5 years ago
                            logger.warning(f"Using 5 years ago - {training_start_date} is not a valid date")
                            training_start_date = dt_module.now().date() - timedelta(days=365*5)
                else:
                    # Default to 5 years ago
                    training_start_date = dt_module.now().date() - timedelta(days=365*5)
                st.session_state["training_start_date_user"] = training_start_date
            
            # Set tuning status to true immediately for UI responsiveness
            st.session_state["tuning_in_progress"] = True
            
            # Pre-cache data to ensure it's available for all trials
            try:
                # Import the caching function
                from src.dashboard.dashboard.dashboard_core import load_data_with_caching

                # Get the parameters
                ticker = st.session_state.get("selected_ticker", "ETH-USD")
                timeframe = st.session_state.get("selected_timeframe", "1d")

                # Get training start date
                training_start_date = st.session_state.get("training_start_date_user")

                # Format the training_start_date if needed
                if isinstance(training_start_date, dt_module.date):
                    training_start_date = training_start_date.strftime("%Y-%m-%d")

                # Pre-cache data with long timeout (12 hours)
                # Move info message outside column
                st.sidebar.info(f"Pre-caching data for {ticker}...")
                df = load_data_with_caching(
                    ticker=ticker,
                    start_date=training_start_date,
                    interval=timeframe,
                    force_refresh=False,
                    cache_timeout_minutes=720  # 12 hours for tuning
                )
                if df is not None:
                    # Move success message outside column
                    st.sidebar.success(f"Successfully cached {len(df)} data points for tuning")
            except Exception as e:
                st.sidebar.warning(f"Data caching error: {e}")
            
            # Clear any stale lock files first
            status_file = os.path.join(DATA_DIR, "tuning_status.txt")
            lock_file = f"{status_file}.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                    logger.info(f"Removed stale lock file: {lock_file}")
                except Exception as lock_err:
                    logger.warning(f"Could not remove lock file: {lock_err}")
            
            # Get parameters
            ticker = st.session_state.get("selected_ticker", "ETH-USD")
            timeframe = st.session_state.get("selected_timeframe", "1d")
            multipliers = st.session_state.get("tuning_multipliers", {})
            
            # Store validated date parameters in multipliers to pass to tuning function
            multipliers['start_date'] = start_date.strftime("%Y-%m-%d")
            multipliers['training_start_date'] = training_start_date.strftime("%Y-%m-%d")
            
            logger.info(f"Starting tuning for {ticker} ({timeframe}) with validated date parameters")
            logger.info(f"Using start_date={multipliers['start_date']}, training_start_date={multipliers['training_start_date']}")
            
            # Start tuning with the current parameters
            print(f"DEBUG: About to call start_tuning with ticker={ticker}, timeframe={timeframe}")
            print(f"DEBUG: multipliers={multipliers}")
            try:
                # Try importing before calling
                from src.dashboard.dashboard.dashboard_model import start_tuning
                print("DEBUG: Successfully imported start_tuning")
            except Exception as e:
                print(f"DEBUG-ERROR: Failed to import start_tuning: {e}")
                
            start_tuning(ticker, timeframe, multipliers)
        except Exception as e:
            st.sidebar.error(f"Failed to start tuning: {str(e)}")
            logger.error(f"Error starting tuning: {e}", exc_info=True)
            # Reset tuning state if there was an error
            st.session_state["tuning_in_progress"] = False

    # Process stop tuning clicks outside of column contexts
    if st.session_state.pop("stop_tuning_clicked", False):
        try:
            from src.dashboard.dashboard.dashboard_model import stop_tuning
            stop_tuning()
            # Set tuning status to false immediately for UI responsiveness
            st.session_state["tuning_in_progress"] = False
        except Exception as e:
            st.sidebar.error(f"Failed to stop tuning: {str(e)}")
            logger.error(f"Error stopping tuning: {e}", exc_info=True)
            
            # Provide emergency stop button
            if st.sidebar.button("Force Stop", key="force_stop"):
                st.session_state["tuning_in_progress"] = False
                st.experimental_rerun()

    # Add space between sections
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Data selection section
    st.sidebar.markdown("### 📊 Data Selection")

    # Add custom ticker input option
    use_custom_ticker = st.sidebar.checkbox(
        "Use custom ticker",
        key="sidebar_cb_use_custom_ticker",
        value=st.session_state.get("use_custom_ticker", False),
    )

    if use_custom_ticker:
        # Text input for custom ticker
        ticker = st.sidebar.text_input(
            "Enter ticker symbol:",
            key="sidebar_input_custom_ticker",
            value=st.session_state.get("custom_ticker", ""),
            help="Example: AAPL, MSFT, BTC-USD, ETH-USD",
        )
        if not ticker:  # If empty, use default
            ticker = TICKER
    else:
        # Standard dropdown selection
        ticker = st.sidebar.selectbox(
            "Select ticker:",
            key="sidebar_select_ticker",
            options=TICKERS,
            index=(
                TICKERS.index(st.session_state.get("selected_ticker", TICKER))
                if st.session_state.get("selected_ticker", TICKER) in TICKERS
                else 0
            ),
        )

    # Store the custom ticker in session state
    if use_custom_ticker:
        st.session_state["custom_ticker"] = ticker
    st.session_state["use_custom_ticker"] = use_custom_ticker
    st.session_state["selected_ticker"] = ticker

    # Select timeframe with default selection handling
    selected_timeframe_index = 0
    if (
        "selected_timeframe" in st.session_state
        and st.session_state["selected_timeframe"] in TIMEFRAMES
    ):
        selected_timeframe_index = TIMEFRAMES.index(
            st.session_state["selected_timeframe"]
        )

    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        TIMEFRAMES,
        key="sidebar_select_timeframe",
        index=selected_timeframe_index,
        help="Choose data frequency/interval",
    )
    st.session_state["selected_timeframe"] = timeframe

    # Date range section with improved handling
    st.sidebar.markdown("### 📅 Date Range")

    # VISUALIZATION DATE RANGE - only for chart display
    st.sidebar.markdown("#### Chart Visualization Range")
   
    default_start = dt_module.now().date() - timedelta(days=60)
    start_date = st.sidebar.date_input(
        "Chart Start Date",
        key="sidebar_input_start_date",
        value=st.session_state.get("start_date_user", default_start),
        help="Starting date for visualization only (defaults to 60 days ago)",
    )
    # Only store in session state if user made a change
    if st.session_state.get("start_date_user") != start_date:
        st.session_state["start_date_user"] = start_date
        st.session_state["start_date"] = start_date

    # Make end_date completely independent
    default_end = dt_module.now().date() + timedelta(days=30)
    end_date = st.sidebar.date_input(
        "Forecast End Date",
        key="sidebar_input_end_date",
        value=st.session_state.get("end_date_user", default_end),
        help="End date for forecast visualization (future date for predictions)",
    )
    # Only store in session state if user made a change
    if st.session_state.get("end_date_user") != end_date:
        st.session_state["end_date_user"] = end_date
        st.session_state["end_date"] = end_date

    # MODEL TRAINING DATA RANGE - separate section
    st.sidebar.markdown("#### Model Training Data Range")

    default_training_start = dt_module.now().date() - timedelta(days=365 * 5)  # Use renamed module
    training_start_date = st.sidebar.date_input(
        "Training Start Date",
        key="sidebar_input_training_start_date",
        value=st.session_state.get("training_start_date_user", default_training_start),
        help="Starting date for training data (earlier means more data for training)",
    )
    # Only store in session state if user made a change
    if st.session_state.get("training_start_date_user") != training_start_date:
        st.session_state["training_start_date_user"] = training_start_date
        st.session_state["training_start_date"] = training_start_date
        # Do NOT update start_date here to keep them independent

    # Advanced settings in an expander
    with st.sidebar.expander("Advanced Settings", expanded=False):
        # Calculate windows from dates (but don't expose as separate controls)
        current_date = dt_module.now().date()
        
        # These are calculated values for information only, not controls
        historical_window = (current_date - start_date).days  # Chart visualization window
        forecast_window = (end_date - current_date).days  # Future forecast window
        training_window = (current_date - training_start_date).days  # Total training data window
        
        # Store calculated values in session state (not as controls)
        st.session_state["historical_window"] = historical_window
        st.session_state["forecast_window"] = forecast_window
        st.session_state["lookback"] = historical_window  # Add lookback as an alias
        
        # Display the calculated values for information only
        st.info(f"""
        **Calculated Windows:**
        - Visualization window: {historical_window} days (from chart start to today)
        - Forecast window: {forecast_window} days (from today to forecast end)
        - Training data window: {training_window} days (from training start to today)
        """)
        
        # Add walk-forward update control (keep this part)
        st.write("**Walk-Forward Settings:**")
        update_during_wf = st.checkbox(
            "Update Dashboard During Walk-Forward",
            key="sidebar_cb_update_during_wf",
            value=st.session_state.get("update_during_walk_forward", True),
            help="Enable/disable dashboard updates during walk-forward validation"
        )
        st.session_state["update_during_walk_forward"] = update_during_wf
        
        if update_during_wf:
            update_interval = st.slider(
                "Update Interval (cycles)",
                min_value=1,
                max_value=20, 
                value=st.session_state.get("update_during_walk_forward_interval", 5),
                help="Number of cycles between dashboard updates"
            )
            st.session_state["update_during_walk_forward_interval"] = update_interval

    # Chart settings section
    st.sidebar.markdown("### 📊 Chart Settings")

    # Custom indicators first
    st.sidebar.write("**Custom Indicators:**")
    show_werpi = st.sidebar.checkbox(
        "WERPI",
        key="sidebar_cb_show_werpi",
        value=st.session_state.get("show_werpi", False),
        help="Wavelet-based Encoded Relative Price Indicator",
    )
    show_vmli = st.sidebar.checkbox(
        "VMLI",
        key="sidebar_cb_show_vmli",
        value=st.session_state.get("show_vmli", False),
        help="Volatility-Momentum-Liquidity Indicator",
    )

    # Standard indicators - all off by default
    st.sidebar.write("**Technical Indicators:**")
    show_ma = st.sidebar.checkbox(
        "Moving Averages",
        key="sidebar_cb_show_ma",
        value=st.session_state.get("show_ma", False),
    )
    show_bb = st.sidebar.checkbox(
        "Bollinger Bands",
        key="sidebar_cb_show_bb",
        value=st.session_state.get("show_bb", False),
    )
    show_rsi = st.sidebar.checkbox(
        "RSI", 
        key="sidebar_cb_show_rsi", 
        value=st.session_state.get("show_rsi", False)
    )
    show_macd = st.sidebar.checkbox(
        "MACD",
        key="sidebar_cb_show_macd",
        value=st.session_state.get("show_macd", False),
    )

    # Forecast options
    st.sidebar.write("**Forecast Options:**")
    show_forecast = st.sidebar.checkbox(
        "Show Forecast",
        key="sidebar_cb_show_forecast",
        value=st.session_state.get("show_forecast", True),
    )
    
    show_confidence = st.sidebar.checkbox(
        "Show Confidence Bands",
        key="sidebar_cb_show_confidence",
        value=st.session_state.get("show_confidence", True),
    )

    # Auto-refresh settings
    st.sidebar.write("**Auto-Refresh Settings:**")
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        key="sidebar_cb_auto_refresh",
        value=st.session_state.get("auto_refresh", True),
        help="Automatically refresh the dashboard",
    )
    st.session_state["auto_refresh"] = auto_refresh

    refresh_interval = 30  # Default
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=600,
            value=st.session_state.get("refresh_interval", 30),
            step=5,
        )
        st.session_state["refresh_interval"] = refresh_interval

        # Show a countdown timer if auto-refresh is enabled
        if "last_refresh" in st.session_state:
            time_since_refresh = int(
                dt_module.now().timestamp() - st.session_state["last_refresh"]
            )
            time_to_next_refresh = max(0, refresh_interval - time_since_refresh)

            # Show progress bar for refresh timer
            refresh_progress = 1 - (time_to_next_refresh / refresh_interval)
            st.sidebar.progress(min(1.0, max(0.0, refresh_progress)))
            st.sidebar.text(f"Next refresh in {time_to_next_refresh} seconds")

    # Store indicator preferences in session state
    indicators = {
        "show_ma": show_ma,
        "show_bb": show_bb,
        "show_rsi": show_rsi,
        "show_macd": show_macd,
        "show_werpi": show_werpi,
        "show_vmli": show_vmli,
        "show_forecast": show_forecast,
        "show_confidence": show_confidence,
    }

    # Update individual indicator flags in session state
    for key, value in indicators.items():
        st.session_state[key] = value

    st.session_state["indicators"] = indicators

    # Add helpful information
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <div style="font-size: 0.85em; color: #78909c;">
        <strong>Tips:</strong>
        <ul>
            <li>Choose a longer historical window for more context</li>
            <li>Auto-refresh keeps predictions and metrics updated</li>
            <li>Training with more data gives better results but takes longer</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Add shutdown button at the bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <div style="text-align: center;">
        <h4 style="color: #e0e0e0;">Application Control</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button(
        "🛑 Shutdown Application",
        key="shutdown_button",
        help="Safely shutdown the application",
    ):
        try:
            # Stop tuning if it's running
            if st.session_state.get("tuning_in_progress", False):
                stop_tuning()

            st.sidebar.warning("Shutting down application... Please wait.")

            # Try to initiate shutdown tasks
            try:
                from src.dashboard.dashboard.dashboard_shutdown import initiate_shutdown
                initiate_shutdown(source="User requested shutdown")
            except ImportError:
                # If shutdown module not available, do our best
                st.session_state["shutdown_requested"] = True

            # Give feedback that shutdown is happening
            st.info("Application is shutting down. You can close this window.")
            
            # Try a clean exit
            try:
                sys.exit(0)
            except:
                # Try a less clean exit if sys.exit doesn't work
                try:
                    os._exit(0)
                except:
                        # Last resort - just tell the user to close the tab
                    st.error("Could not automatically shut down. Please close this browser tab.")
        except Exception as e:
            st.sidebar.error(f"Error shutting down: {e}")

    # Log the current tuning state for debugging
    st.sidebar.markdown(f"<p>Tuning status: {st.session_state.get('tuning_in_progress', False)}</p>", unsafe_allow_html=True)

    # Return selected parameters with calculated values and indicators
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "start_date": start_date,  # Chart visualization start date
        "end_date": end_date,
        "training_start_date": training_start_date,  # Model training start date (independent)
        "historical_window": historical_window,
        "forecast_window": forecast_window,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "indicators": indicators,
    }


@robust_error_boundary
def handle_deferred_tuning_start():
    """
    Handle deferred tuning start requests.
    This function checks if a tuning start was requested but not yet executed,
    and initiates the tuning process if needed.
    """
    if st.session_state.get("start_tuning_clicked", False):
        try:
            # Get parameters
            ticker = st.session_state.get("selected_ticker", "ETH-USD")
            timeframe = st.session_state.get("selected_timeframe", "1d")
            multipliers = st.session_state.get("tuning_multipliers", {})
            
            # Validate essential parameters
            if not ticker or not timeframe:
                logger.error(f"Invalid tuning parameters: ticker={ticker}, timeframe={timeframe}")
                st.error("Invalid tuning parameters. Please select a ticker and timeframe.")
                st.session_state.pop("start_tuning_clicked", None)
                return
                
            logger.info(f"Deferred tuning start for {ticker}/{timeframe}")
            
            # Properly validate and convert date parameters
            start_date = st.session_state.get("start_date_user")
            if not isinstance(start_date, date_type):
                logger.warning(f"Invalid start date type: {type(start_date)}, value: {start_date}")
                # Ensure we have a proper date object
                if isinstance(start_date, str):
                    try:
                        start_date = dt_module.strptime(start_date, "%Y-%m-%d").date()
                    except ValueError:
                        # If the string looks like a timeframe ('1d'), use 5 years ago
                        logger.warning(f"Using 5 years ago - {start_date} is not a valid date")
                        start_date = dt_module.now().date() - timedelta(days=365*5)
                else:
                    # Default to 5 years ago
                    start_date = dt_module.now().date() - timedelta(days=365*5)
                st.session_state["start_date_user"] = start_date
            
            # No need to explicitly update status here, let start_tuning handle it
            # This avoids double updates and potential race conditions
            
            # Start tuning with the current parameters
            try:
                from src.dashboard.dashboard.dashboard_model import start_tuning
                
                # Call start_tuning with validated parameters
                success = start_tuning(ticker, timeframe, multipliers)
                
                if not success:
                    logger.error(f"Failed to start tuning for {ticker}/{timeframe}")
                    st.error("Failed to start tuning process. Check the logs for details.")
                
                # Clear the flag after processing whether successful or not
                st.session_state.pop("start_tuning_clicked", None)
                
            except ImportError as e:
                logger.error(f"Could not import start_tuning: {e}")
                st.error(f"Could not start tuning process: {e}")
                st.session_state.pop("start_tuning_clicked", None)
                
        except Exception as e:
            logger.error(f"Error in handle_deferred_tuning_start: {e}", exc_info=True)
            st.error(f"Failed to start tuning: {str(e)}")
            # Reset tuning state and flags
            st.session_state["tuning_in_progress"] = False
            st.session_state.pop("start_tuning_clicked", None)


# Initialize session state for trial_logs if it doesn't exist
if "trial_logs" not in st.session_state:
    st.session_state["trial_logs"] = []

# Add an auto-refresh mechanism for tuning status
def create_auto_refresh():
    """Create an automatic refresh mechanism for tuning status pages."""
    # Ensure last_refresh_time and refresh_interval are set
    if "last_refresh_time" not in st.session_state:
        st.session_state["last_refresh_time"] = 0
    if "refresh_interval" not in st.session_state:
        st.session_state["refresh_interval"] = 5  # seconds
    
    current_time = time.time()
    # Only trigger a refresh if elapsed time exceeds interval
    if current_time - st.session_state["last_refresh_time"] > st.session_state["refresh_interval"]:
        st.session_state["last_refresh_time"] = current_time
        st_autorefresh(interval=st.session_state["refresh_interval"] * 1000, key="auto_refresh")
    else:
        # Delay a bit if called too early
        time.sleep(0.5)

@robust_error_boundary
def create_tuning_panel():
    """
    Unified tuning panel function that consolidates the duplicated functionality.
    Displays model tuning information and controls.
    Fixed to show individual model progress and avoid nesting issues.
    """
    st.header("Model Tuning Monitor")
    
    # Add auto-refresh for dynamic updates
    create_auto_refresh()

    # Initialize the watchdog if not already in session state
    if "watchdog" not in st.session_state:
        try:
            from src.utils.watchdog import TuningWatchdog
            watchdog = TuningWatchdog(
                ticker=st.session_state.get("selected_ticker", "ETH-USD"),
                timeframe=st.session_state.get("selected_timeframe", "1d"),
                auto_restart=True,
                check_interval=300,  # 5 minutes
                monitor_resources=True,
            )
            st.session_state["watchdog"] = watchdog
            print("Initialized watchdog for tuning panel")
        except Exception as e:
            print(f"Error initializing watchdog: {e}")
            st.session_state["watchdog"] = None

    # Start the watchdog if it exists
    if st.session_state.get("watchdog"):
        try:
            st.session_state["watchdog"].start()
        except Exception as e:
            print(f"Error starting watchdog: {e}")

    # Create radio buttons for tab selection instead of tabs to avoid nesting issues
    tab_options = ["Status", "Trials", "Technical", "Resources", "Watchdog"]
    selected_tab = st.radio("Select View", tab_options, horizontal=True)

    # Display selected tab content
    if selected_tab == "Status":
        display_tuning_status()
    
    elif selected_tab == "Trials":
        st.subheader("Tuning Trials")
        display_trials_table()
    
    elif selected_tab == "Technical":
        st.subheader("Technical Indicators")
        display_technical_indicators()
    
    elif selected_tab == "Resources":
        st.subheader("Resource Usage")
        display_resource_usage()
    
    elif selected_tab == "Watchdog":
        st.subheader("Watchdog")
        display_watchdog()


@st.cache_data(ttl=2)  # Only cache for 2 seconds
def get_fresh_progress():
    """Get fresh progress data from files, bypassing cache."""
    from src.tuning.progress_helper import read_progress_from_yaml, get_individual_model_progress
    return {
        "main": read_progress_from_yaml(),
        "models": get_individual_model_progress() or {}
    }

def create_live_progress_updater():
    """
    Create a component that automatically updates the progress display.
    Uses st_autorefresh to periodically refresh the page when tuning is active.
    """
    # Only add the autorefresh component if tuning is in progress
    if st.session_state.get("tuning_in_progress", False):
        # Create autorefresh with shorter interval during tuning (2 seconds)
        refresh_interval = 2000  # milliseconds
        st_autorefresh(interval=refresh_interval, key="tuning_progress_autorefresh")
        
        # Show a small indicator that auto-refresh is active
        st.caption("🔄 Auto-refreshing progress...")

@robust_error_boundary
def display_tuning_status():
    """Display current tuning status with progress bars."""
    # Get fresh data
    progress_data = get_fresh_progress()
    main_progress = progress_data["main"]
    model_progress = progress_data["models"]
    
    # Show current status
    st.subheader("💨 Tuning Status")
    
    # Get latest progress and status
    progress = load_latest_progress()
    status = read_tuning_status()
    is_running = status.get("is_running", False)

    # Create columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        if is_running:
            st.markdown(
                "<h2 style='color:#4CAF50'>⚡ TUNING ACTIVE</h2>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<h2 style='color:#888888'>⏹️ TUNING IDLE</h2>", unsafe_allow_html=True
            )

        # Add current model information below tuning status
        ticker = status.get("ticker", st.session_state.get("selected_ticker", ""))
        timeframe = status.get(
            "timeframe", st.session_state.get("selected_timeframe", "")
        )

        if ticker and timeframe:
            model_type = progress.get("best_model", "Unknown")
            st.markdown(
                f"""
            <div style='margin-top: 5px; font-size: 1.1em;'>
                Current model for <span style='color:#1E88E5; font-weight: bold;'>{ticker}/{timeframe}</span>: 
                <span style='color:#4CAF50;'>{model_type}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Show larger, more prominent cycle and trial information
    if progress and "current_trial" in progress and "total_trials" in progress:
        # Create columns for major metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            # Larger, more prominent cycle display
            cycle = progress.get("cycle", "N/A")
            st.markdown(
                f"<h2 style='text-align: center; color:#1E88E5'>Cycle {cycle}</h2>",
                unsafe_allow_html=True,
            )

        with metric_col2:
            # Larger, more prominent trial information (aggregated)
            current = progress.get("current_trial", 0)
            total = progress.get("total_trials", 0)
            
            # Explicitly prefer aggregated values if available
            if "aggregated_current_trial" in progress:
                current = progress.get("aggregated_current_trial", current)
            
            if "aggregated_total_trials" in progress:
                total = progress.get("aggregated_total_trials", total)
                
            st.markdown(
                f"<h2 style='text-align: center; color:#1E88E5'>Trial {current}/{total}</h2>",
                unsafe_allow_html=True,
            )

            # Add progress bar for aggregated trials
            progress_val = current / max(1, total)
            st.progress(min(1.0, max(0.0, float(progress_val))))

        with metric_col3:
            # Display current best metrics
            best_rmse = progress.get("best_rmse", progress.get("current_rmse", "N/A"))
            best_mape = progress.get("best_mape", progress.get("current_mape", "N/A"))

            if isinstance(best_rmse, (int, float)):
                best_rmse = f"{best_rmse:.4f}"
            if isinstance(best_mape, (int, float)):
                best_mape = f"{best_mape:.2f}%"

            st.markdown(
                f"<div style='text-align: center;'><b>Best RMSE:</b> {best_rmse}<br><b>Best MAPE:</b> {best_mape}</div>",
                unsafe_allow_html=True,
            )
    
    # NEW: Display individual model progress
    st.subheader("Individual Model Progress")
    
    # Get individual model progress information with error handling
    try:
        from src.tuning.progress_helper import get_individual_model_progress
        model_progress = get_individual_model_progress()
        
        # Check if model_progress is a valid dictionary and not empty
        if isinstance(model_progress, dict) and model_progress:
            # Create a progress bar for each model
            for model_type, info in model_progress.items():
                # Extract trial information
                current = info.get("current_trial", 0)
                total = info.get("total_trials", 1)
                pct = current / max(1, total) * 100
                
                # Display model progress with color coding
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{model_type}**:")
                with col2:
                    st.progress(pct/100)
                    st.caption(f"{current}/{total} trials ({pct:.1f}%) - Cycle: {info.get('cycle', 'N/A')}")
                    
                    # Display metrics if available
                    metrics_text = []
                    if "current_rmse" in info and info["current_rmse"] is not None:
                        metrics_text.append(f"RMSE: {info['current_rmse']:.4f}")
                    if "current_mape" in info and info["current_mape"] is not None:
                        metrics_text.append(f"MAPE: {info['current_mape']:.2f}%")
                    
                    if metrics_text:
                        st.caption(" | ".join(metrics_text))
        else:
            st.info("No individual model progress information available")
    except Exception as e:
        st.error(f"Error displaying model progress: {str(e)}")
        st.info("Could not load model progress information")

    # Additional details in expandable section
    with st.expander("Additional Details", expanded=False):
        # Format and display all progress information
        if progress:
            details = []
            for k, v in progress.items():
                if k == "timestamp":
                    try:
                        dt_obj = dt_module.fromtimestamp(v)
                        v = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                details.append(f"**{k}:** {v}")

            st.markdown("<br>".join(details), unsafe_allow_html=True)
        else:
            st.info("No tuning progress information available.")
    
    # Add live progress updater at the end
    create_live_progress_updater()


@robust_error_boundary
def display_trials_table(max_trials=20):
    """
    Display the most recent tuning trials in a table format.
    Enhanced to show trial data from all models.
    
    Args:
        max_trials: Maximum number of trials to display
    """
    # Import required definitions
    try:
        from config.config_loader import ACTIVE_MODEL_TYPES
    except ImportError:
        # Fallback if import fails
        ACTIVE_MODEL_TYPES = ["lstm", "rnn", "cnn", "xgboost", "random_forest", "ltc", "nbeats", "tabnet", "tft"]
    
    try:
        from src.tuning.progress_helper import TESTED_MODELS_DIR, TESTED_MODELS_FILE
    except ImportError:
        # Fallback if import fails
        from config.config_loader import DATA_DIR
        TESTED_MODELS_DIR = os.path.join(DATA_DIR, "tested_models")
        TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
    
    # First try to load all trials from individual model files
    try:
        # Create a selection for model type filter
        model_filter = st.selectbox(
            "Filter by model type:", 
            ["All Models"] + ACTIVE_MODEL_TYPES,
            index=0
        )
        
        # Get paths to individual model trial files
        all_trials = []
        model_types_to_load = ACTIVE_MODEL_TYPES if model_filter == "All Models" else [model_filter]
        
        for model_type in model_types_to_load:
            model_file = os.path.join(TESTED_MODELS_DIR, f"{model_type}_tested_models.yaml")
            
            # Skip if file doesn't exist
            if not os.path.exists(model_file):
                continue
                
            try:
                # Use safe_read_yaml to read the file
                from src.utils.threadsafe import safe_read_yaml
                model_trials = safe_read_yaml(model_file, default=[])
                
                if model_trials and isinstance(model_trials, list):
                    # Add trials from this model
                    all_trials.extend(model_trials)
            except Exception as e:
                st.warning(f"Error reading trials for {model_type}: {str(e)}")
        
        # If no individual model files found, try the main tested_models file
        if not all_trials:
            if os.path.exists(TESTED_MODELS_FILE):
                from src.utils.threadsafe import safe_read_yaml
                all_trials = safe_read_yaml(TESTED_MODELS_FILE, default=[])
        
        # If still nothing, check session state
        if not all_trials and "trial_logs" in st.session_state and st.session_state["trial_logs"]:
            all_trials = st.session_state["trial_logs"]
        
        # If we have trials, display them
        if all_trials:
            # Sort by timestamp if available, otherwise by trial number
            all_trials.sort(key=lambda x: (x.get("timestamp", ""), x.get("number", 0)))
            
            # Take the most recent trials
            recent_trials = all_trials[-max_trials:]
            
            # Convert to DataFrame for display
            df_data = []
            for trial in recent_trials:
                # Extract metrics - handle different formats
                rmse = trial.get("metrics", {}).get("rmse", trial.get("rmse", "N/A"))
                mape = trial.get("metrics", {}).get("mape", trial.get("mape", "N/A"))
                da = trial.get("metrics", {}).get("directional_accuracy", 
                               trial.get("directional_accuracy", "N/A"))
                
                # Create entry for display
                entry = {
                    "Trial": trial.get("number", "N/A"),
                    "Model": trial.get("model_type", "Unknown"),
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Dir. Acc.": da,
                    "Time": trial.get("timestamp", "N/A")
                }
                df_data.append(entry)
            
            if df_data:
                df = pd.DataFrame(df_data)
                
                # Format numeric columns
                if "RMSE" in df.columns:
                    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce").round(4)
                
                if "MAPE" in df.columns:
                    df["MAPE"] = pd.to_numeric(df["MAPE"], errors="coerce").round(2)
                    df["MAPE"] = df["MAPE"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    
                if "Dir. Acc." in df.columns:
                    df["Dir. Acc."] = pd.to_numeric(df["Dir. Acc."], errors="coerce").round(2)
                    df["Dir. Acc."] = df["Dir. Acc."].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                
                st.dataframe(df, use_container_width=True)
                
                # Show total number of trials
                st.caption(f"Showing {len(df_data)} of {len(all_trials)} total trials")
                
                return
        
        # If we get here, no trials were found
        st.info("No trial logs available. Try running some tuning cycles.")
            
    except Exception as e:
        st.error(f"Error displaying model trials: {e}")
        st.info("No trial logs available in session state. Try running some tuning cycles.")


def get_price_column(df, ticker=None, column_type="Close"):
    """
    Intelligently find price columns in a DataFrame with flexible naming.
    
    Args:
        df: DataFrame to search within
        ticker: Optional ticker symbol to help narrow down column name
        column_type: Type of price data to find (Close, Open, High, Low, etc.)
    
    Returns:
        String name of the price column or None if not found
    """
    if df is None or df.empty:
        return None
    
    # Check if columns exist in DataFrame
    all_columns = df.columns.tolist()
    
    # Possible naming patterns for closing price columns
    possible_names = []
    
    # 1. Standard column name
    possible_names.append(column_type)
    possible_names.append(column_type.lower())
    
    # 2. With ticker suffix
    if ticker:
        possible_names.append(f"{column_type}_{ticker}")
        possible_names.append(f"{column_type.lower()}_{ticker}")
        
        # Handle ticker without prefix like 'USD'
        base_ticker = ticker.split('-')[0] if '-' in ticker else ticker
        possible_names.append(f"{column_type}_{base_ticker}")
        possible_names.append(f"{column_type.lower()}_{base_ticker}")
    
    # 3. Prefixed with ticker
    if ticker:
        possible_names.append(f"{ticker}_{column_type}")
        possible_names.append(f"{ticker}_{column_type.lower()}")
    
    # 4. Common variations
    possible_names.append(f"price_{column_type.lower()}")
    possible_names.append(f"{column_type.lower()}_price")
    
    # 5. Partial matches with column type prefix
    partial_matches = [col for col in all_columns if col.lower().startswith(column_type.lower())]
    possible_names.extend(partial_matches)
    
    # Try to find an exact match first
    for col_name in possible_names:
        if col_name in all_columns:
            logger.info(f"Found price column: {col_name}")
            return col_name
    
    # Try partial match if no exact match was found
    if not ticker:
        # Look for any column containing "close" if no specific ticker
        for col in all_columns:
            if column_type.lower() in col.lower():
                logger.info(f"Using partial match for price column: {col}")
                return col
    
    # If we get here, no suitable column was found
    available_cols = ', '.join(all_columns[:10]) + ('...' if len(all_columns) > 10 else '')
    logger.warning(f"Could not find '{column_type}' column. Available columns: {available_cols}")
    
    # Return None if no match found
    return None


@robust_error_boundary
def display_technical_indicators():
    """
    Display a preview of computed technical indicators.
    Uses sample data if real data isn't available.
    """
    if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
        df = st.session_state["df_raw"]
        
        # Find closing price column using our flexible function
        ticker = st.session_state.get("selected_ticker", "")
        close_column = get_price_column(df, ticker, "Close") 
        
        if (close_column):
            # Add price column to potential indicators
            indicator_cols = [close_column]
        else:
            indicator_cols = []
            
        # Check if technical indicators are already computed
        indicator_cols.extend([col for col in df.columns if col in [
            "RSI", "MACD", "BB_upper", "BB_lower", "BB_middle", 
            "WERPI", "VMLI", "MA20", "MA50", "MA200"
        ]])
        
        if indicator_cols:
            st.write("### Technical Indicators")
            
            # Select which indicators to show
            selected_indicators = st.multiselect(
                "Select indicators to display",
                options=indicator_cols,
                default=indicator_cols[:min(5, len(indicator_cols))]
            )
            
            if selected_indicators:
                # Show the selected indicators
                st.line_chart(df[selected_indicators])
                
                # Show a sample of the data in tabular form
                st.write("### Indicator Data Sample")
                display_cols = ["date"] if "date" in df.columns else []
                display_cols.extend(selected_indicators)
                st.dataframe(df[display_cols].tail(10))
            else:
                st.info("Please select at least one indicator to display")
        else:
            st.info("No technical indicators or price data found. They may need to be computed first.")
    else:
        st.info("No data available. Load data first to view technical indicators.")


@robust_error_boundary
def display_resource_usage():
    """
    Display system resource usage by leveraging the advanced training resource optimizer dashboard.
    """
    try:
        # Import the advanced resource monitoring dashboard
        from src.dashboard.training_resource_optimizer_dashboard import render_hardware_resources_section, get_training_optimizer
        
        # Get the training optimizer
        if "training_optimizer" not in st.session_state:
            st.session_state["training_optimizer"] = get_training_optimizer()
        
        optimizer = st.session_state["training_optimizer"]
        
        # Render the advanced hardware resources section
        render_hardware_resources_section(optimizer)
        
        # Optionally add an expander to show more sections
        with st.expander("Advanced Training Optimization", expanded=False):
            st.info("Access the full Training Resource Optimization dashboard for more advanced features:")
            if st.button("Open Full Training Optimizer Dashboard"):
                # Set session state to show the full optimizer dashboard
                st.session_state["show_training_optimizer"] = True
                st.experimental_rerun()
                
    except ImportError as e:
        module = str(e).split("'")[-2] if "'" in str(e) else "required module"
        st.warning(f"Advanced resource monitoring requires the {module} package.")
        
        # Fallback to a basic monitoring display if import fails
        _display_basic_resource_usage()
    except Exception as e:
        st.error(f"Error monitoring system resources: {str(e)}")


def _display_basic_resource_usage():
    """Fallback basic resource usage display in case the advanced dashboard isn't available"""
    
    # First check if psutil is available
    try:
        import psutil
        have_psutil = True
    except ImportError:
        have_psutil = False
        st.error("Required module 'psutil' is not installed")
        st.info("Please install psutil to enable resource monitoring")
        return
        
    # Only proceed if we have psutil
    if have_psutil:
        try:
            # CPU and memory metrics
            cpu_usage = psutil.cpu_percent(interval=0.5)
            memory_info = psutil.virtual_memory()
            mem_percent = memory_info.percent
            mem_used_gb = memory_info.used / (1024 ** 3)
            mem_total_gb = memory_info.total / (1024 ** 3)
            
            # Display basic metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CPU Usage", f"{cpu_usage:.1f}%")
                st.metric("Thread Count", f"{psutil.cpu_count()}")
            with col2:
                st.metric("Memory Usage", f"{mem_percent:.1f}%")
                st.metric("Memory", f"{mem_used_gb:.1f} GB / {mem_total_gb:.1f} GB")
                
            # Basic chart - store historical data in session state
            if "basic_resource_history" not in st.session_state:
                st.session_state["basic_resource_history"] = []
                
            # Add current snapshot
            st.session_state["basic_resource_history"].append({
                "timestamp": dt_module.now().strftime("%H:%M:%S"),
                "CPU": cpu_usage,
                "Memory": mem_percent
            })
            
            # Keep only the last 30 samples
            if len(st.session_state["basic_resource_history"]) > 30:
                st.session_state["basic_resource_history"] = st.session_state["basic_resource_history"][-30:]
                
            # Create simple chart
            chart_data = pd.DataFrame(st.session_state["basic_resource_history"])
            if len(chart_data) > 1:
                st.line_chart(chart_data.set_index("timestamp"))
        except Exception as e:
            st.error(f"Basic resource monitoring failed: {e}")


@robust_error_boundary
def display_watchdog():
    """Display watchdog information and controls."""
    if "watchdog" in st.session_state and st.session_state["watchdog"] is not None:
        watchdog = st.session_state["watchdog"]
        
        try:
            # Try to use the built-in streamlit component if available
            watchdog.create_streamlit_component()
        except AttributeError:
            # Fallback if the method doesn't exist
            st.info("Watchdog is running but detailed visualization is not available.")
            
            # Display basic watchdog controls
            st.write("### Watchdog Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Watchdog", use_container_width=True):
                    try:
                        watchdog.start()
                        st.success("Watchdog started")
                    except Exception as e:
                        st.error(f"Error starting watchdog: {e}")
            
            with col2:
                if st.button("Stop Watchdog", use_container_width=True):
                    try:
                        watchdog.stop()
                        st.success("Watchdog stopped")
                    except Exception as e:
                        st.error(f"Error stopping watchdog: {e}")
            
            # Display watchdog status if available
            try:
                status = watchdog.get_status()
                st.json(status)
            except:
                st.write("Watchdog status not available")
        
    else:
        st.warning(
            "Watchdog is not available. It needs to be initialized in the session state."
        )
        
        # Allow initializing the watchdog
        if st.button("Initialize Watchdog"):
            try:
                from src.utils.watchdog import TuningWatchdog
                st.session_state["watchdog"] = TuningWatchdog()
                st.success("Watchdog initialized!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error initializing watchdog: {e}")


@robust_error_boundary
def create_metrics_cards():
    """
    Create a row of key metrics cards with improved styling and visual indicators.
    Displays current status of model training and prediction accuracy.
    """
    # Get the ticker and timeframe from session state
    current_ticker = st.session_state.get("selected_ticker", "Unknown")
    current_timeframe = st.session_state.get("selected_timeframe", "Unknown")

    # Get the latest progress from YAML file
    try:
        progress = load_latest_progress(ticker=current_ticker, timeframe=current_timeframe)
    except Exception as e:
        logger.error(f"Error reading progress data: {e}")
        progress = {}

    # Extract values from progress data
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get("total_trials", 1)  # Use 1 as default to avoid division by zero
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)
    cycle = progress.get("cycle", 1)

    # If specific ticker/timeframe don't match progress data, add a note
    if progress.get("ticker") != current_ticker or progress.get("timeframe") != current_timeframe:
        if progress.get("ticker") is not None:
            current_ticker = f"{current_ticker} (showing data for {progress.get('ticker', 'Unknown')})"

    # Display ticker and timeframe info
    st.markdown(
        f"""
    <div style="text-align: center; margin-bottom: 10px;">
        <span style="background-color: rgba(33, 150, 243, 0.1); padding: 5px 10px; border-radius: 4px; color: #2196F3; font-weight: bold;">
            {current_ticker} / {current_timeframe}
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Retrieve best metrics from session state
    best_metrics = st.session_state.get("best_metrics", {})
    best_rmse = best_metrics.get("rmse")
    best_mape = best_metrics.get("mape")
    
    # Ensure metrics are numeric when comparing
    if current_rmse is not None and best_rmse is not None:
        try:
            rmse_delta = float(best_rmse) - float(current_rmse)
        except (ValueError, TypeError):
            rmse_delta = None
    else:
        rmse_delta = None
        
    if current_mape is not None and best_mape is not None:
        try:
            mape_delta = float(best_mape) - float(current_mape)
        except (ValueError, TypeError):
            mape_delta = None
    else:
        mape_delta = None

    # Create a 5-column layout for metrics
    cols = st.columns(5)

    # Col 1: Cycle indicator
    with cols[0]:
        st.metric(
            "CURRENT CYCLE",
            cycle,
            delta=None,
            delta_color="normal"
        )
        # Add a simple progress bar
        st.progress(min(1.0, cycle/10))

    # Col 2: Trial Progress
    with cols[1]:
        trial_progress = f"{current_trial}/{total_trials}"
        st.metric(
            "TRIAL PROGRESS",
            trial_progress,
            delta=None,
            delta_color="normal"
        )
        # Add a progress bar
        progress_pct = current_trial / max(1, total_trials)
        st.progress(min(1.0, progress_pct))

    # Col 3: Current RMSE
    with cols[2]:
        rmse_display = f"{current_rmse:.4f}" if isinstance(current_rmse, (int, float)) else "N/A"
        rmse_delta_display = f"{rmse_delta:.4f}" if rmse_delta is not None else None
        delta_color = "normal"
        if rmse_delta is not None:
            delta_color = "good" if rmse_delta > 0 else "inverse"
        
        st.metric(
            "CURRENT RMSE",
            rmse_display,
            delta=rmse_delta_display,
            delta_color=delta_color
        )

    # Col 4: Current MAPE
    with cols[3]:
        mape_display = f"{current_mape:.2f}%" if isinstance(current_mape, (int, float)) else "N/A"
        mape_delta_display = f"{mape_delta:.2f}%" if mape_delta is not None else None
        delta_color = "normal"
        if mape_delta is not None:
            delta_color = "good" if mape_delta > 0 else "inverse"
        
        st.metric(
            "CURRENT MAPE",
            mape_display,
            delta=mape_delta_display,
            delta_color=delta_color
        )

    # Col 5: Direction Accuracy
    with cols[4]:
        # Calculate direction accuracy if we have prediction data
        if "ensemble_predictions_log" in st.session_state and st.session_state["ensemble_predictions_log"]:
            predictions = st.session_state["ensemble_predictions_log"]
            success_rate = 0

            if len(predictions) > 1:
                try:
                    # Calculate direction accuracy
                    correct_direction = 0
                    for i in range(1, len(predictions)):
                        if predictions[i].get("actual") is None or predictions[i-1].get("actual") is None:
                            continue
                            
                        actual_direction = predictions[i]["actual"] > predictions[i-1]["actual"]
                        pred_direction = predictions[i]["predicted"] > predictions[i-1]["predicted"]
                        if actual_direction == pred_direction:
                            correct_direction += 1

                    if (len(predictions) - 1) > 0:
                        success_rate = (correct_direction / (len(predictions) - 1)) * 100
                except Exception as e:
                    logger.error(f"Error calculating direction accuracy: {e}")
                    success_rate = 0

            st.metric(
                "DIRECTION ACCURACY",
                f"{success_rate:.1f}%",
                delta=None
            )
        else:
            st.metric(
                "DIRECTION ACCURACY",
                "N/A",
                delta=None
            )


@robust_error_boundary
def initiate_shutdown(source="unknown"):
    """
    Initiate a graceful shutdown of the dashboard.
    
    Args:
        source: Information about what triggered the shutdown
    """
    logger.info(f"Shutdown initiated from {source}")

    # Stop the watchdog if it's running
    if "watchdog" in st.session_state:
        try:
            st.session_state["watchdog"].stop()
            logger.info("Stopped watchdog during shutdown")
        except Exception as e:
            logger.error(f"Error stopping watchdog: {e}")

    # Stop any tuning process
    if st.session_state.get("tuning_in_progress", False):
        try:
            stop_tuning()
            logger.info("Stopped tuning during shutdown")
        except Exception as e:
            logger.error(f"Error stopping tuning during shutdown: {e}")

    # Save session state for shutdown
    st.session_state["shutdown_requested"] = True
    st.session_state["shutdown_time"] = dt_module.now().isoformat()
    st.session_state["shutdown_source"] = source

    # Display shutdown message
    st.warning("Dashboard is shutting down. Please close this browser tab.")
    
    # Try to write shutdown flag file for other components to detect
    try:
        shutdown_flag_file = os.path.join(DATA_DIR, "shutdown_requested.txt")
        with open(shutdown_flag_file, "w") as f:
            f.write(f"Shutdown requested at {dt_module.now().isoformat()} by {source}")
    except Exception as e:
        logger.error(f"Error writing shutdown flag file: {e}")