"""
dashboard_core.py

Core functionality for the unified dashboard. This file serves as the main entry point
for the dashboard application and orchestrates all the dashboard components.
"""

import base64
import os
import sys
import traceback
from datetime import datetime, timedelta

# Add project root to Python path for reliable imports
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import streamlit as st


# First, define a simple error boundary function for fallback
def simple_error_boundary(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            st.code(traceback.format_exc())
            return None

    return wrapper


# Use the fallback function by default
robust_error_boundary = simple_error_boundary

# Define fallback functions for when imports fail
def read_tuning_status_fallback():
    return {"status": "unknown", "is_running": False}

def load_latest_progress_fallback(ticker=None, timeframe=None):
    return {
        "current_trial": 0,
        "total_trials": 1,
        "current_rmse": None,
        "current_mape": None,
        "cycle": 1,
    }


# Import dashboard modules with error handling
def import_dashboard_modules():
    """Import dashboard modules with error handling to avoid circular imports"""
    modules = {}

    try:
        # Don't import UI components at module level - will be imported in functions where needed

        # Try to import hyperparameter dashboard as a fallback
        try:
            from src.dashboard.hyperparameter_dashboard import (
                main as hyperparameter_dashboard,
            )

            modules["create_hyperparameter_tuning_panel"] = hyperparameter_dashboard
        except ImportError:
            print("Error importing hyperparameter_dashboard")

        try:
            from src.dashboard.dashboard.dashboard_data import (
                calculate_indicators,
                ensure_date_column,
                generate_dashboard_forecast,
                load_data,
            )

            modules.update(
                {
                    "load_data": load_data,
                    "calculate_indicators": calculate_indicators,
                    "generate_dashboard_forecast": generate_dashboard_forecast,
                    "ensure_date_column": ensure_date_column,
                }
            )
        except ImportError as e:
            print(f"Error importing dashboard_data: {e}")

            # Define fallback functions for data operations
            def fallback_load_data(
                ticker, start_date, end_date=None, interval="1d", training_mode=False
            ):
                print(f"Using fallback load_data for {ticker}")
                return None

            def fallback_calculate_indicators(df):
                print("Using fallback calculate_indicators")
                return df

            def fallback_generate_forecast(model, df, feature_cols):
                print("Using fallback generate_forecast")
                return []

            def fallback_ensure_date_column(df, default_name="date"):
                print("Using fallback ensure_date_column")
                if df is None or df.empty:
                    return df, default_name

                # Try to find a date column
                date_col = None
                if "date" in df.columns:
                    date_col = "date"
                elif "Date" in df.columns:
                    date_col = "Date"

                # If no date column found, create one
                if date_col is None:
                    from datetime import datetime, timedelta

                    import pandas as pd

                    df = df.copy()
                    df[default_name] = pd.date_range(
                        start=datetime.now() - timedelta(days=len(df)), periods=len(df)
                    )
                    date_col = default_name

                return df, date_col

            modules.update(
                {
                    "load_data": fallback_load_data,
                    "calculate_indicators": fallback_calculate_indicators,
                    "generate_dashboard_forecast": fallback_generate_forecast,
                    "ensure_date_column": fallback_ensure_date_column,
                }
            )

        # Import visualization functions - moved inside functions to avoid circular imports
        # Do NOT import from dashboard_visualization here - will do it inside functions

        # Add standardize_column_names utility function
        def standardize_column_names(df, ticker=None):
            """
            Standardize column names by removing ticker-specific parts for OHLCV.
            """
            if df is None or df.empty:
                return df
            df_copy = df.copy()

            # First check if standard columns already exist
            has_standard = all(
                col in df_copy.columns for col in ["Open", "High", "Low", "Close"]
            )
            if has_standard:
                return df_copy

            # Handle ticker-specific column names
            if ticker:
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    # Check for pattern like 'Close_ETH-USD'
                    if f"{col}_{ticker}" in df_copy.columns:
                        df_copy[col] = df_copy[f"{col}_{ticker}"]
                    # Check for pattern like 'ETH-USD_Close'
                    elif f"{ticker}_{col}" in df_copy.columns:
                        df_copy[col] = df_copy[f"{ticker}_{col}"]
                    # Check for pattern with underscores removed
                    ticker_nohyphen = ticker.replace("-", "")
                    if f"{col}_{ticker_nohyphen}" in df_copy.columns:
                        df_copy[col] = df_copy[f"{col}_{ticker_nohyphen}"]

            # Also handle lowercase variants
            for old_col, std_col in [
                ("open", "Open"),
                ("high", "High"),
                ("low", "Low"),
                ("close", "Close"),
                ("volume", "Volume"),
            ]:
                if old_col in df_copy.columns and std_col not in df_copy.columns:
                    df_copy[std_col] = df_copy[old_col]

            # Log missing columns that we couldn't standardize
            missing = [
                c for c in ["Open", "High", "Low", "Close"] if c not in df_copy.columns
            ]
            if missing:
                print(
                    f"Warning: Missing required columns after standardization: {missing}"
                )
                print(f"Available columns: {df_copy.columns.tolist()}")

            return df_copy

        modules["standardize_column_names"] = standardize_column_names

    except Exception as e:
        print(f"Error during module imports: {e}")
        import traceback

        traceback.print_exc()

    return modules


# Import modules using the function
dashboard_modules = import_dashboard_modules()

# Try to import config after other imports to avoid circular dependencies
try:
    from config.config_loader import get_config

    config = get_config()
    DATA_DIR = config.get("DATA_DIR", os.path.join(project_root, "data"))
    TICKER = config.get("TICKER", "ETH-USD")
    TICKERS = config.get("TICKERS", ["ETH-USD", "BTC-USD"])
    TIMEFRAMES = config.get("TIMEFRAMES", ["1d", "1h"])
    # Basic logger
    from config.logger_config import logger

    print("Successfully imported config")
except ImportError as e:
    print(f"Error importing config: {e}")
    # Fallback values
    DATA_DIR = os.path.join(project_root, "data")
    TICKER = "ETH-USD"
    TICKERS = ["ETH-USD", "BTC-USD"]
    TIMEFRAMES = ["1d", "1h"]
    # Basic logger
    import logging

    logger = logging.getLogger("dashboard")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)


@robust_error_boundary
def set_page_config():
    """Configure the Streamlit page settings with modern styling"""
    try:
        st.set_page_config(
            page_title="AI Price Prediction Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        # Custom CSS for dark mode look
        st.markdown(
            """
        <style>
        .main {
            padding-top: 1rem;
            background-color: #121212;
            color: #e0e0e0;
        }
        h1, h2, h3 {
            font-family: 'Roboto', sans-serif;
            color: #2196F3;
        }
        .sidebar .sidebar-content {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .metric-container {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            border-left: 4px solid #2196F3;
            color: #e0e0e0;
        }
        .stButton>button {
            background-color: #2196F3;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
        }
        .chart-container {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
            border-top: 4px solid #2196F3;
            color: #e0e0e0;
        }
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #4CAF50, #8BC34A);
        }
        /* Dark theme tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e1e1e;
            border-radius: 8px 8px 0px 0px;
            gap: 1px;
            padding-top: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #2a2a2a;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 500;
            color: #e0e0e0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2196F3 !important;
            color: white !important;
        }
        /* Header bar styling */
        header[data-testid="stHeader"] {
            background-color: #121212;
            color: white;
        }
        /* Improve contrast for text */
        p, span, div {
            color: #e0e0e0;
        }
        .metric-value {
            color: #2196F3;
            font-size: 1.5em;
            font-weight: bold;
        }
        /* Make inputs and selects readable in dark mode */
        .stSelectbox>div>div, .stDateInput>div>div {
            background-color: #2a2a2a;
            color: #e0e0e0;
        }
        .stDataFrame {
            background-color: #1e1e1e;
        }
        /* Make tab sizing better */
        .stTabs [data-baseweb="tab"] {
            min-width: 120px;
            padding: 12px 24px;
            font-size: 1.05em;
            text-align: center;
        }
        /* Increase clickable area for tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            padding: 0px 0px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
    except Exception:
        # This might fail if called twice
        pass


@robust_error_boundary
def initialize_session_state():
    """Initialize session state variables with defensive checks"""
    if "error_log" not in st.session_state:
        st.session_state["error_log"] = []
    if "metrics_container" not in st.session_state:
        st.session_state["metrics_container"] = None
    if "tuning_in_progress" not in st.session_state:
        st.session_state["tuning_in_progress"] = False
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = TICKER
    if "selected_timeframe" not in st.session_state:
        st.session_state["selected_timeframe"] = TIMEFRAMES[0]

    # Check if watchdog should be initialized early
    if "watchdog" not in st.session_state:
        try:
            # Import inside the function to avoid circular imports
            from src.utils.watchdog import TuningWatchdog

            watchdog = TuningWatchdog(
                ticker=st.session_state.get("selected_ticker", TICKER),
                timeframe=st.session_state.get("selected_timeframe", TIMEFRAMES[0]),
                auto_restart=True,
                check_interval=300,  # 5 minutes
                monitor_resources=True,
            )
            # Don't start it yet - will be started in the tuning panel
            st.session_state["watchdog"] = watchdog
            logger.info("Watchdog initialized during session state setup")
        except Exception as e:
            logger.warning(f"Could not initialize watchdog during startup: {e}")
            # Don't block the dashboard if watchdog initialization fails
            st.session_state["watchdog"] = None


@robust_error_boundary
def check_tuning_status():
    """Check tuning status from file and update session state accordingly."""
    try:
        # First try to import from progress_helper for more reliable status
        try:
            from src.tuning.progress_helper import read_tuning_status as read_from_helper
            status = read_from_helper()
        except ImportError:
            # Fall back to dashboard_error version - import inside function to avoid circular imports
            try:
                from src.dashboard.dashboard.dashboard_error import read_tuning_status
                status = read_tuning_status()
            except ImportError:
                # Use fallback function if neither import works
                status = read_tuning_status_fallback()

        print(f"Checking tuning status: {status}")

        # Make sure is_running is a boolean
        is_running = status.get("is_running", False)
        if isinstance(is_running, str):
            # Convert string "True"/"False" to boolean
            is_running = is_running.lower() == "true"

        # Log any mismatch
        current_status = st.session_state.get("tuning_in_progress", False)
        if is_running != current_status:
            print(
                f"Tuning status mismatch - File: {is_running}, Session: {current_status}"
            )
            logger.info(
                f"Tuning status mismatch - File: {is_running}, Session: {current_status}"
            )
            # Update session state to match file
            st.session_state["tuning_in_progress"] = is_running

        # Store additional info from status
        if "ticker" in status:
            st.session_state["tuning_ticker"] = status["ticker"]
        if "timeframe" in status:
            st.session_state["tuning_timeframe"] = status["timeframe"]
        if "start_time" in status:
            st.session_state["tuning_start_time"] = status["start_time"]

        return status
    except Exception as e:
        logger.error(f"Error checking tuning status: {e}")
        return {"is_running": False, "error": str(e)}


@robust_error_boundary
def clean_stale_locks():
    """Remove any stale lock files that might be preventing proper operation."""
    try:
        # Use threadsafe's built-in cleanup function first
        from src.utils.threadsafe import cleanup_stale_locks

        print("Running stale lock cleanup...")
        cleanup_stale_locks()

        # If tuning status shows running but UI doesn't think so, reset it
        try:
            # Check tuning status - import inside function to avoid circular imports
            from src.dashboard.dashboard.dashboard_error import read_tuning_status
            status = read_tuning_status()

            if status.get("is_running", False) and not st.session_state.get(
                "tuning_in_progress", False
            ):
                print("Tuning status mismatch - resetting tuning status file")
                try:
                    # Get current values
                    ticker = status.get(
                        "ticker", st.session_state.get("selected_ticker", "unknown")
                    )
                    timeframe = status.get(
                        "timeframe", st.session_state.get("selected_timeframe", "1d")
                    )

                    # Write with is_running: False - import inside function
                    from src.dashboard.dashboard.dashboard_error import write_tuning_status
                    write_tuning_status(ticker, timeframe, is_running=False)
                    print("Reset stale tuning status to not running")
                except Exception as e:
                    print(f"Error resetting tuning status: {e}")
        except Exception as e:
            print(f"Error checking tuning status: {e}")

    except Exception as e:
        print(f"Error in clean_stale_locks: {e}")


@robust_error_boundary
def main_dashboard():
    """Main dashboard entry point with robust error handling"""
    try:
        # Clean up stale locks first thing
        clean_stale_locks()

        # Initialize session state at the beginning
        initialize_session_state()

        # Setup page and session state
        set_page_config()

        # Check if shutdown was requested
        try:
            from src.dashboard.dashboard.dashboard_shutdown import is_shutting_down

            if is_shutting_down():
                from src.dashboard.dashboard.dashboard_shutdown import show_shutdown_message
                show_shutdown_message()
                st.stop()
        except ImportError as e:
            print(f"Could not import dashboard_shutdown: {e}")
            # Continue without shutdown check

        # Check tuning status and update session state - more prominent log
        print("Checking tuning status...")
        tuning_status = check_tuning_status()
        print(f"Current tuning status: {tuning_status}")

        # Build UI components - import inside function to avoid circular imports
        try:
            from src.dashboard.dashboard.dashboard_ui import create_header
        except ImportError:
            def create_header():
                st.title("AI Price Prediction Dashboard")
        
        create_header()

        # Create sidebar with controls (now includes shutdown button) - import inside function
        try:
            from src.dashboard.dashboard.dashboard_ui import create_control_panel
        except ImportError:
            def create_control_panel():
                ticker = st.sidebar.selectbox("Ticker", TICKERS)
                timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES)
                return {"ticker": ticker, "timeframe": timeframe}
        
        params = create_control_panel()

        # Check if params is None (indicating an error in create_control_panel)
        if params is None:
            st.error("Error in control panel. Using default parameters instead.")
            # Provide default values as fallback
            params = {
                "ticker": st.session_state.get("selected_ticker", TICKER),
                "timeframe": st.session_state.get("selected_timeframe", TIMEFRAMES[0]),
                "start_date": datetime.now() - timedelta(days=90),
                "end_date": datetime.now() + timedelta(days=30),
                "training_start_date": datetime.now() - timedelta(days=365 * 5),
                "historical_window": st.session_state.get("historical_window", 90),
                "forecast_window": st.session_state.get("forecast_window", 30),
                "auto_refresh": True,
                "refresh_interval": 30,
            }

        # Get parameters from the control panel state for other components
        ticker = params["ticker"]
        timeframe = params["timeframe"]
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        training_start = params.get("training_start_date")
        # Get values from params, which are now retrieved from session state if available
        historical_window = params.get("historical_window")
        forecast_window = params.get("forecast_window")

        # Make sure these values are in session state for other components
        # This ensures persistence between dashboard refreshes
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["training_start_date"] = training_start
        st.session_state["historical_window"] = historical_window
        st.session_state["forecast_window"] = forecast_window
        st.session_state["lookback"] = historical_window  # Add lookback as an alias for historical_window

        st.sidebar.info(
            """
        **Dashboard Settings:**
        - Training data from: {}
        - Display range: {} to {}
        """.format(
                params.get("training_start_date", datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d"),
                params.get("start_date", datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                params.get("end_date", datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            )
        )

        # Ensure session state dates are updated
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["training_start_date"] = training_start

        # Show a loading spinner while fetching data
        with st.spinner("Loading market data..."):
            # Modify this part to ensure we only fetch historical data up to today
            today = datetime.now().strftime("%Y-%m-%d")
            # Fetch historical market data (cached) for the selected ticker/timeframe
            load_data_func = dashboard_modules.get("load_data", lambda *args, **kwargs: None)
            df_vis = load_data_func(
                ticker,
                params.get("start_date", datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                today,  # Use current date instead of future end_date
                timeframe,
            )
            if df_vis is not None and not df_vis.empty:
                # Standardize column names
                standardize_func = dashboard_modules.get("standardize_column_names", lambda df, ticker: df)
                df_vis = standardize_func(df_vis, ticker)
                # Fix: Use the centralized date handling function
                ensure_date_func = dashboard_modules.get("ensure_date_column", lambda df, name="date": (df, name))
                df_vis, date_col = ensure_date_func(df_vis)
                st.session_state["df_raw"] = df_vis  # cache data in session
            else:
                # If no new data, use cached data if available
                if st.session_state.get("df_raw") is not None:
                    df_vis = st.session_state["df_raw"]
                    st.warning("Using previously loaded data as new data could not be fetched.")
                else:
                    st.error(f"No data available for {ticker} from {start_date} to {end_date}.")
                    return  # abort if we have absolutely no data

        # UNIFIED CHART AT THE TOP 
        # Create a container for the chart with styling
        chart_container = st.container()
        with chart_container:
            # Initialize future_forecast as None before attempting to generate it
            future_forecast = None
            # Get indicator preferences from params
            indicators = params.get("indicators", {})
            model = st.session_state.get("model")
            if model and indicators.get("show_forecast", True):
                with st.spinner("Generating forecast..."):
                    # Determine feature columns (exclude date and target 'Close')
                    feature_cols = [
                        col for col in df_vis.columns if col not in ["date", "Date", "Close"]
                    ]
                    # Use the consolidated function from dashboard_data
                    generate_forecast_func = dashboard_modules.get("generate_dashboard_forecast", lambda model, df, features: None)
                    future_forecast = generate_forecast_func(model, df_vis, feature_cols)
                    # Save the prediction for historical comparison
                    if future_forecast and len(future_forecast) > 0:
                        try:
                            # Import here to avoid circular imports
                            from src.dashboard.dashboard.dashboard_visualization import save_best_prediction
                            save_best_prediction(df_vis, future_forecast)
                        except ImportError:
                            # Just continue if save function isn't available
                            pass

            # Calculate indicators and plot the chart with increased height
            calculate_indicators_func = dashboard_modules.get("calculate_indicators", lambda df: df)
            df_vis_indicators = calculate_indicators_func(df_vis)

            # Apply custom indicators if available
            apply_custom_indicators_func = dashboard_modules.get("apply_custom_indicators", lambda df, timeframe, indicators: df)
            if "indicators" in st.session_state:
                df_vis_indicators = apply_custom_indicators_func(
                    df_vis_indicators, timeframe, st.session_state["indicators"]
                )

            # Pass indicator options to visualization function - IMPORTANT FIX:
            # Import inside function to avoid circular imports
            try:
                from src.dashboard.dashboard.dashboard_visualization import create_interactive_price_chart
                create_interactive_price_chart(
                    df_vis_indicators,
                    params,
                    future_forecast=future_forecast,
                    indicators=indicators,
                    height=850,
                )
            except ImportError:
                # Fallback to dashboard_modules if direct import fails
                create_chart_func = dashboard_modules.get(
                    "create_interactive_price_chart",
                    lambda df, params, **kwargs: st.line_chart(df[["Close"]])
                )
                create_chart_func(
                    df_vis_indicators,
                    params,
                    future_forecast=future_forecast,
                    indicators=indicators,
                    height=850,
                )

        # Get model metrics and progress information
        best_metrics = st.session_state.get("best_metrics", {})
        best_rmse = best_metrics.get("rmse")
        best_mape = best_metrics.get("mape")
        direction_accuracy = best_metrics.get("direction_accuracy")

        # Get progress data from YAML - import inside function to avoid circular imports
        try:
            from src.dashboard.dashboard.dashboard_error import load_latest_progress
            progress = load_latest_progress(ticker=ticker, timeframe=timeframe)
        except ImportError:
            progress = load_latest_progress_fallback(ticker=ticker, timeframe=timeframe)
            
        cycle = progress.get("cycle", 1)
        current_trial = progress.get("current_trial", 0)
        total_trials = progress.get("total_trials", 1)

        # Show metrics below chart in a row
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = (
            st.columns(5)
        )
        # Display metrics in the row
        with metrics_col1:
            if best_rmse is not None:
                st.metric("Model RMSE", f"{best_rmse:.2f}")
            else:
                st.metric("Model RMSE", "N/A")
        with metrics_col2:
            if best_mape is not None:
                st.metric("Model MAPE", f"{best_mape:.2f}%")
            else:
                st.metric("Model MAPE", "N/A")
        with metrics_col3:
            if direction_accuracy is not None:
                st.metric("Direction Accuracy", f"{direction_accuracy:.1f}%")
            else:
                st.metric("Direction Accuracy", "N/A")
        with metrics_col4:
            st.markdown(
                """
                <div style="font-size: 1.15em;">
                    <p style="margin-bottom: 5px;"><strong>Cycle Status:</strong></p>
                    <p>Cycle: <strong>{}</strong> | Trial: <strong>{}/{}</strong></p>
                </div>
                """.format(
                    cycle, current_trial, total_trials
                ),
                unsafe_allow_html=True,
            )
        # Progress bar in the last column
        with metrics_col5:
            st.markdown(
                "<p style='font-size: 1.15em; margin-bottom: 5px;'><strong>Cycle Progress:</strong></p>",
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, current_trial / max(1, total_trials)))

        # Create main content tabs with enhanced UI
        main_tabs = st.tabs(
            ["🧠 Model Insights", "⚙️ Model Tuning", "📈 Technical Indicators", "📊 Price Data"]
        )

        # Tab 1: Model Insights
        with main_tabs[0]:
            st.subheader("Model Performance & Insights")
            insight_tabs = st.tabs(
                ["Performance Metrics", "Feature Importance", "Prediction Analysis"]
            )
            with insight_tabs[0]:
                if (
                    "best_metrics" in st.session_state
                    and st.session_state["best_metrics"]
                ):
                    metrics = st.session_state["best_metrics"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best RMSE", metrics.get("rmse", "N/A"))
                    with col2:
                        st.metric("Best MAPE", f"{metrics.get('mape', 'N/A'):.2f}%")
                    with col3:
                        st.metric(
                            "Direction Accuracy",
                            f"{metrics.get('direction_accuracy', 'N/A'):.1f}%",
                        )
                    with col4:
                        st.metric("Model Type", metrics.get("model_type", "N/A"))
                    # Show performance history if available
                    if (
                        "model_history" in st.session_state
                        and st.session_state["model_history"]
                    ):
                        st.subheader("Training History")
                        history_df = pd.DataFrame(st.session_state["model_history"])
                        st.line_chart(history_df)
                else:
                    st.info(
                        "No model metrics available yet. Train a model to see performance insights."
                    )
            with insight_tabs[1]:
                if (
                    "feature_importance" in st.session_state
                    and st.session_state["feature_importance"]
                ):
                    st.subheader("Feature Importance")
                    feature_df = pd.DataFrame(
                        {
                            "Feature": st.session_state["feature_importance"].keys(),
                            "Importance": st.session_state[
                                "feature_importance"
                            ].values(),
                        }
                    ).sort_values(by="Importance", ascending=False)
                    st.dataframe(
                        dashboard_modules["prepare_dataframe_for_display"](feature_df)
                    )
                else:
                    st.info("No feature importance data available yet.")
            with insight_tabs[2]:
                if (
                    "prediction_history" in st.session_state
                    and st.session_state["prediction_history"]
                ):
                    st.subheader("Prediction History")
                    pred_df = pd.DataFrame(st.session_state["prediction_history"])
                    st.dataframe(
                        dashboard_modules["prepare_dataframe_for_display"](pred_df)
                    )
                    # Calculate direction accuracy
                    pred_df["correct_direction"] = (
                        pred_df["actual_direction"] == pred_df["predicted_direction"]
                    )
                    correct_dir = (pred_df["correct_direction"]).mean() * 100
                    st.metric("Direction Accuracy", "{:.1f}%".format(correct_dir))
                else:
                    st.info("No prediction history available yet.")

        # Tab 2: Model Tuning
        with main_tabs[1]:
            st.subheader("Hyperparameter Tuning")
            tuning_tabs = st.tabs(["Tuning Status", "Tuning Panel"])
            with tuning_tabs[0]:
                st.subheader("Current Tuning Status")
                if (
                    "tuning_in_progress" in st.session_state
                    and st.session_state["tuning_in_progress"]
                ):
                    st.success("Tuning is currently in progress.")
                else:
                    st.info("No tuning in progress.")

                if (
                    "best_params" in st.session_state
                    and st.session_state["best_params"]
                ):
                    st.subheader("Best Parameters")
                    best_params = st.session_state["best_params"]
                    for param, value in best_params.items():
                        st.write(f"**{param}:** {value}")

                with st.expander("View All Tested Models", expanded=False):
                    display_tested_models = dashboard_modules.get(
                        "display_tested_models",
                        lambda: st.info("Tested models display not available"),
                    )
                    display_tested_models()

            with tuning_tabs[1]:
                # Hyperparameter tuning panel directly embedded here - import inside function
                with st.spinner("Loading hyperparameter tuning panel..."):
                    try:
                        from src.dashboard.dashboard.dashboard_ui import create_hyperparameter_tuning_panel
                        create_hyperparameter_tuning_panel()
                    except ImportError:
                        # Fallback to function from dashboard_modules
                        tuning_panel_func = dashboard_modules.get(
                            "create_hyperparameter_tuning_panel", 
                            lambda: st.info("Hyperparameter tuning panel not available")
                        )
                        tuning_panel_func()

        # Tab 3: Technical Analysis
        with main_tabs[2]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Technical Indicators")

            with st.spinner("Loading technical indicators..."):
                # Only render if we have data
                if (
                    "df_raw" in st.session_state
                    and st.session_state["df_raw"] is not None
                ):
                    # Calculate indicators on the FULL dataset
                    df_indicators = calculate_indicators_func(
                        st.session_state["df_raw"]
                    )

                    # Apply custom indicators if needed
                    if "indicators" in st.session_state:
                        df_indicators = apply_custom_indicators_func(
                            df_indicators, timeframe, st.session_state["indicators"]
                        )

                    # Enhanced technical indicators
                    try:
                        # Import inside function to avoid circular imports
                        from src.dashboard.dashboard.dashboard_visualization import create_technical_indicators_chart
                        create_technical_indicators_chart(df_indicators, params)
                    except ImportError:
                        # Fallback to dashboard_modules if direct import fails
                        create_tech_chart_func = dashboard_modules.get(
                            "create_technical_indicators_chart",
                            lambda df, params: st.line_chart(df[["Close"]]),
                        )
                        create_tech_chart_func(df_indicators, params)
                else:
                    st.warning("No data available for technical indicators")
            st.markdown("</div>", unsafe_allow_html=True)

            # Advanced analysis dashboard
            if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
                with st.spinner("Loading advanced analysis..."):
                    try:
                        show_advanced_dashboard_func = dashboard_modules.get(
                            "show_advanced_dashboard_tabs", lambda df: None
                        )
                        show_advanced_dashboard_func(st.session_state["df_raw"])
                    except Exception as e:
                        st.error(f"Error creating advanced analysis: {e}")
            else:
                st.warning("No data available for advanced analysis")

        # Tab 4: Price Data
        with main_tabs[3]:
            with st.spinner("Loading price data..."):
                if (
                    "df_raw" in st.session_state
                    and st.session_state["df_raw"] is not None
                ):
                    df_vis = st.session_state["df_raw"]

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.subheader("Data Summary")
                        st.write(f"**Ticker:** {ticker}")
                        st.write(f"**Timeframe:** {timeframe}")
                        st.write(f"**Period:** {start_date} to {end_date}")
                        st.write(f"**Data points:** {len(df_vis)}")

                        # Show key statistics
                        if len(df_vis) > 0:
                            try:
                                st.metric(
                                    "Current Price", f"${df_vis['Close'].iloc[-1]:.2f}"
                                )
                                if len(df_vis) > 1:
                                    change = float(df_vis["Close"].iloc[-1]) - float(
                                        df_vis["Close"].iloc[-2]
                                    )
                                    pct_change = (
                                        change / float(df_vis["Close"].iloc[-2])
                                    ) * 100
                                    st.metric(
                                        "Last Change",
                                        f"${change:.2f}",
                                        f"{pct_change:.2f}%",
                                    )
                            except Exception as e:
                                st.error(f"Error displaying metrics: {e}")

                        # Download button for CSV
                        if not df_vis.empty:
                            try:
                                csv_data = df_vis.to_csv(index=False)
                                b64 = base64.b64encode(csv_data.encode()).decode()
                                download_link = '<a href="data:file/csv;base64,{}" download="{}_{}_data.csv" class="download-button">📥 Download Data</a>'.format(
                                    b64, ticker, timeframe
                                )
                                st.markdown(download_link, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error creating download link: {e}")

                    with col2:
                        st.subheader("Market Data")

                        # Prepare dataframe for display
                        try:
                            display_df = dashboard_modules[
                                "prepare_dataframe_for_display"
                            ](df_vis)

                            # FIX: Check if display_df is not None and not empty before styling
                            if display_df is not None and not display_df.empty:
                                try:
                                    # Since styling can fail, wrap it in try/except
                                    # Show only a preview (first 100 rows) for performance
                                    preview_df = display_df.head(100)
                                    styled_df = preview_df.style.format(
                                        {
                                            "Open": "${:,.2f}",
                                            "High": "${:,.2f}",
                                            "Low": "${:,.2f}",
                                            "Close": "${:,.2f}",
                                            "Volume": "{:,.0f}",
                                        }
                                    )
                                    st.dataframe(styled_df, height=400)

                                    # Show total number of rows
                                    if len(display_df) > 100:
                                        st.info(
                                            f"Showing 100 of {len(display_df)} rows. Download the CSV for full data."
                                        )
                                except Exception as e:
                                    # Fallback if styling fails
                                    logger.error(f"Error styling dataframe: {e}")
                                    st.dataframe(display_df.head(100), height=400)
                            else:
                                st.warning("No data available to display.")
                        except Exception as e:
                            st.error(f"Error preparing dataframe: {e}")
                            # Absolute fallback - just show raw data
                            st.dataframe(df_vis.head(100))

                        # Summary Statistics
                        with st.expander("Summary Statistics"):
                            if df_vis is not None and not df_vis.empty:
                                try:
                                    numeric_cols = df_vis.select_dtypes(
                                        include=["float64", "int64"]
                                    ).columns
                                    stats_df = (
                                        df_vis[numeric_cols].describe().transpose()
                                    )
                                    st.dataframe(stats_df)
                                except Exception as e:
                                    st.error(
                                        f"Error displaying summary statistics: {e}"
                                    )
                            else:
                                st.info("No data available for statistics.")
                else:
                    st.warning("No data available to display.")

        # Auto-refresh logic at the end of the function
        if params.get("auto_refresh", False):
            current_time = datetime.now().timestamp()
            if "last_refresh" in st.session_state:
                time_since_last_refresh = (
                    current_time - st.session_state["last_refresh"]
                )
                refresh_interval = params.get("refresh_interval", 30)

                if time_since_last_refresh >= refresh_interval:
                    # Update last refresh time
                    st.session_state["last_refresh"] = current_time
                    st.experimental_rerun()
            else:
                # Initialize last_refresh if not set
                st.session_state["last_refresh"] = current_time

    except Exception as e:
        st.error(f"Critical error in main dashboard: {e}")
        logger.error(f"Critical error in main dashboard: {e}", exc_info=True)
        st.code(traceback.format_exc())


# Only run the app if this script is executed directly
if __name__ == "__main__":
    try:
        main_dashboard()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        st.code(traceback.format_exc())
