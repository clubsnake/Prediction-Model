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

# Call at the beginning of your app
fix_import_paths()

# Add this function to dashboard_ui.py to always run at startup
def reset_tuning_status_at_startup():
    """Reset tuning status file at application startup"""
    try:
        from src.tuning.progress_helper import write_tuning_status
        import time
        from datetime import datetime
        
        # Force reset tuning status
        write_tuning_status({
            "is_running": False,
            "status": "reset_at_startup",
            "timestamp": datetime.now().isoformat(),
        })
        
        print("Reset tuning status at startup")
    except Exception as e:
        print(f"Error resetting tuning status at startup: {e}")

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
            if st.sidebar.button(
                "🚀 Start Tuning",
                key="sidebar_btn_start_tuning",
                use_container_width=True,
            ):
                try:
                    # First make sure status is set to NOT RUNNING
                    from src.tuning.progress_helper import write_tuning_status
                    
                    # Force reset tuning status before starting
                    write_tuning_status({
                        "is_running": False,
                        "status": "reset",
                        "timestamp": dt_module.now().isoformat(),
                    })
                    
                    # Add a small delay to ensure file is updated
                    import time
                    time.sleep(0.2)
                    
                    # Get parameters
                    ticker = st.session_state.get("selected_ticker", "ETH-USD")
                    timeframe = st.session_state.get("selected_timeframe", "1d")
                    multipliers = st.session_state.get("tuning_multipliers", {})
                    
                    # Add date parameters
                    multipliers['start_date'] = st.session_state.get("start_date_user").strftime("%Y-%m-%d")
                    multipliers['training_start_date'] = st.session_state.get("training_start_date_user").strftime("%Y-%m-%d")
                    
                    # Set UI state for better responsiveness
                    st.session_state["tuning_in_progress"] = True
                    
                    # Import and call start_tuning
                    from src.dashboard.dashboard.dashboard_model import start_tuning
                    start_tuning(ticker, timeframe, multipliers)
                    
                except Exception as e:
                    st.sidebar.error(f"Failed to start tuning: {str(e)}")
                    print(f"ERROR starting tuning: {e}")
                    st.session_state["tuning_in_progress"] = False

    with col2:
        # If tuning is ongoing, give option to stop
        if st.session_state.get("tuning_in_progress", False):
            if st.sidebar.button(
                "⏹️ Stop Tuning", key="sidebar_btn_stop_tuning", use_container_width=True
            ):
                # Set flag for processing after column context
                st.session_state["stop_tuning_clicked"] = True

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
def create_tuning_panel():
    """
    Unified tuning panel function that consolidates the duplicated functionality.
    Displays model tuning information and controls.
    Fixed to show individual model progress and avoid nesting issues.
    """
    st.header("Model Tuning Monitor")

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


@robust_error_boundary
def display_tuning_status():
    """
    Display the current tuning status with emphasis on cycle and trial information.
    Enhanced to show individual model progress.
    """
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
            
            # Use aggregated values if available
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
    
    # Get individual model progress information
    model_progress = get_individual_model_progress()
    
    if model_progress:
        # Create a progress bar for each model
        for model_type, info in model_progress.items():
            current = info.get("current_trial", 0)
            total = info.get("total_trials", 1)
            percentage = info.get("completion_percentage", 0)
            
            # Display model type and progress
            st.write(f"**{model_type}**: Trial {current}/{total}")
            st.progress(min(1.0, percentage / 100))
    else:
        st.info("No individual model progress information available")

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


@robust_error_boundary
def display_trials_table(max_trials=20):
    """
    Display the most recent tuning trials in a table format.
    Enhanced to show trial data from all models.
    
    Args:
        max_trials: Maximum number of trials to display
    """
    if "trial_logs" in st.session_state and st.session_state["trial_logs"]:
        logs = st.session_state["trial_logs"][-max_trials:]

        # Convert to DataFrame for better display
        df_data = []
        for log in logs:
            entry = {
                "Trial": log.get("trial", log.get("number", "N/A")),
                "Model": log.get("params", {}).get("model_type", "N/A"),
                "RMSE": log.get("rmse", 0),
                "MAPE": log.get("mape", 0),
                "State": log.get("state", "N/A"),
                "Time": log.get("timestamp", "N/A"),
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
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Trial logs contain no valid data.")
    else:
        # Try to load trial logs directly from file
        try:
            from config.config_loader import TESTED_MODELS_FILE
            from src.utils.threadsafe import safe_read_yaml
            
            if os.path.exists(TESTED_MODELS_FILE):
                tested_models = safe_read_yaml(TESTED_MODELS_FILE, default=[])
                
                if tested_models and isinstance(tested_models, list):
                    # Use most recent trials
                    logs = tested_models[-max_trials:]
                    
                    # Convert to DataFrame
                    df_data = []
                    for log in logs:
                        entry = {
                            "Trial": log.get("number", "N/A"),
                            "Model": log.get("model_type", "Unknown"),
                            "RMSE": log.get("rmse", 0),
                            "MAPE": log.get("mape", 0),
                            "State": log.get("state", "N/A"),
                            "Time": log.get("timestamp", "N/A"),
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
                        
                        st.dataframe(df, use_container_width=True)
                        st.info("Loaded trial logs directly from file")
                        return
        except Exception as e:
            st.error(f"Error loading trial logs from file: {e}")
        
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
        
        if close_column:
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