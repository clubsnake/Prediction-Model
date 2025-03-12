"""
Enhanced UI components for the Streamlit dashboard.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yaml

current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)
DATA_DIR = os.path.join(project_root, "data")  #
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import N_STARTUP_TRIALS, TICKER, TICKERS, TIMEFRAMES
from config.logger_config import logger

# Import our data module with custom indicator support
from src.dashboard.dashboard.dashboard_error import (
    load_latest_progress,
    robust_error_boundary,
)

# Import dashboard components
from src.utils.watchdog import TuningWatchdog

# Initialize tuning multipliers from session state
tuning_multipliers = st.session_state.get("tuning_multipliers", {
        "n_startup_trials": N_STARTUP_TRIALS,
})


# Define start_tuning and stop_tuning functions
def start_tuning():
    """Start the hyperparameter tuning process"""
    st.session_state["tuning_in_progress"] = True


def stop_tuning():
    """Stop the hyperparameter tuning process"""
    st.session_state["tuning_in_progress"] = False


@robust_error_boundary
def create_header():
    """Create a visually appealing header section with app branding"""
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
        # Status indicator with dynamic styling - adjust vertical position
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
            Last updated:<br/>{datetime.now().strftime('%H:%M:%S')}
            <br><br>
            <button onclick="Streamlit.experimental_rerun()" style="background-color: #4CAF50; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer;">
                🔄 Refresh
            </button>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Add a horizontal line for visual separation
    st.markdown(
        "<hr style='margin: 0.5rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True
    )


@robust_error_boundary
def create_control_panel():
    """Create an enhanced control panel for user inputs with better organization"""
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
                    # Clear any stale lock files first
                    status_file = os.path.join(DATA_DIR, "tuning_status.txt")
                    lock_file = f"{status_file}.lock"
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                            print(f"Removed stale lock file: {lock_file}")
                        except:
                            pass

                    # Import from dashboard_model to avoid circular imports
                    from src.dashboard.dashboard.dashboard_model import start_tuning

                    # Start tuning with the current parameters
                    start_tuning(
                        st.session_state.get("selected_ticker", "BTC-USD"),
                        st.session_state.get("selected_timeframe", "1d"),
                        st.session_state.get("tuning_multipliers"),
                    )
                except Exception as e:
                    st.error(f"Failed to start tuning: {str(e)}")
                    import traceback

                    print(traceback.format_exc())

    with col2:
        # If tuning is ongoing, give option to stop
        if st.session_state.get("tuning_in_progress", False):
            if st.sidebar.button(
                "⏹️ Stop Tuning", key="sidebar_btn_stop_tuning", use_container_width=True
            ):
                try:
                    from src.dashboard.dashboard.dashboard_model import stop_tuning

                    stop_tuning()
                except Exception as e:
                    st.error(f"Failed to stop tuning: {str(e)}")
                    import traceback

                    print(traceback.format_exc())

    # Add some space
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Create sections in sidebar for better organization
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

    # Date range section with better organization
    st.sidebar.markdown("### 📅 Date Range")

    # Ask for start_date and end_date without automatic recalculation
    default_start = datetime.now().date() - timedelta(days=60)
    start_date = st.sidebar.date_input(
        "Start Date",
        key="sidebar_input_start_date",
        value=st.session_state.get("start_date_user", default_start),
        help="Starting date for visualization",
    )
    # Only store in session state if user made a change
    if st.session_state.get("start_date_user") != start_date:
        st.session_state["start_date_user"] = start_date

    # Make end_date completely independent
    default_end = datetime.now().date() + timedelta(days=30)
    end_date = st.sidebar.date_input(
        "Forecast End Date",
        key="sidebar_input_end_date",
        value=st.session_state.get("end_date_user", default_end),
        help="End date for forecast visualization (future date for predictions)",
    )
    # Only store in session state if user made a change
    if st.session_state.get("end_date_user") != end_date:
        st.session_state["end_date_user"] = end_date

    # Training settings section
    st.sidebar.markdown("### 🧠 Model Training Settings")

    default_training_start = datetime.now().date() - timedelta(days=365 * 5)
    training_start_date = st.sidebar.date_input(
        "Training Start Date",
        key="sidebar_input_training_start_date",
        value=st.session_state.get("training_start_date_user", default_training_start),
        help="Starting date for training data (earlier means more data)",
    )
    # Only store in session state if user made a change
    if st.session_state.get("training_start_date_user") != training_start_date:
        st.session_state["training_start_date_user"] = training_start_date

    # Advanced settings in an expander
    with st.sidebar.expander("Advanced Settings", expanded=False):
        # Calculate windows but make them independent of each other
        current_date = datetime.now().date()

        historical_window = (current_date - start_date).days
        forecast_window = (end_date - current_date).days

    # Add indicator selection at the top of chart settings
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
        "RSI", key="sidebar_cb_show_rsi", value=st.session_state.get("show_rsi", False)
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

    # Auto-refresh settings moved here
    st.sidebar.write("**Auto-Refresh Settings:**")
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        key="sidebar_cb_auto_refresh",
        value=st.session_state.get("auto_refresh", True),
        help="Automatically refresh the dashboard",
    )
    st.session_state["auto_refresh"] = auto_refresh

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
                datetime.now().timestamp() - st.session_state["last_refresh"]
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
    }

    # Update individual indicator flags in session state
    for key, value in indicators.items():
        st.session_state[key] = value

    st.session_state["indicators"] = indicators

    # Add some helpful information
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

    # Add shutdown button at the bottom of sidebar instead of the hyperparameter tuning link
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
            # First stop tuning if it's running
            if st.session_state.get("tuning_in_progress", False):
                from src.dashboard.dashboard.dashboard_model import stop_tuning

                stop_tuning()

            # Display shutdown message and initiate shutdown
            st.sidebar.warning("Shutting down application... Please wait.")
            from src.dashboard.dashboard.dashboard_shutdown import (
                initiate_shutdown,
                show_shutdown_message,
            )

            show_shutdown_message()
            initiate_shutdown(source="User requested shutdown")

            # Use streamlit stop to terminate the app
            st.stop()
        except Exception as e:
            st.sidebar.error(f"Error shutting down: {e}")

    # Return selected parameters with auto-calculated values and indicators
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "training_start_date": training_start_date,
        "historical_window": historical_window,
        "forecast_window": forecast_window,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "indicators": indicators,
    }


@robust_error_boundary
def create_hyperparameter_tuning_panel():
    """Create a panel for hyperparameter tuning controls and information."""
    st.header("Hyperparameter Tuning Monitor")

    # Initialize the watchdog if not already in session state
    if "watchdog" not in st.session_state:
        try:
            watchdog = TuningWatchdog(
                ticker=st.session_state.get("selected_ticker", "BTC-USD"),
                timeframe=st.session_state.get("selected_timeframe", "1d"),
                auto_restart=True,
                check_interval=300,  # 5 minutes
                monitor_resources=True,
            )
            watchdog.start()
            st.session_state["watchdog"] = watchdog
            logger.info("Initialized and started TuningWatchdog")
        except Exception as e:
            logger.error(f"Failed to initialize TuningWatchdog: {e}", exc_info=True)
            st.error(f"Failed to initialize watchdog: {str(e)}")
    else:
        watchdog = st.session_state["watchdog"]

    # Create tabs for different monitoring aspects
    tuning_tabs = st.tabs(["Status", "Trials", "Resources", "Watchdog"])

    with tuning_tabs[0]:
        display_tuning_status()

    with tuning_tabs[1]:
        display_trials_table()

    with tuning_tabs[2]:
        st.subheader("System Resources")
        # Add resource monitoring visualization here

    with tuning_tabs[3]:
        # Display watchdog component
        if "watchdog" in st.session_state and st.session_state["watchdog"] is not None:
            try:
                st.session_state["watchdog"].create_streamlit_component()
            except Exception as e:
                st.error(f"Error displaying watchdog interface: {str(e)}")
                st.info("Watchdog is still running in the background")
        else:
            st.warning("Watchdog not initialized")


@robust_error_boundary
def create_metrics_cards():
    """Create a row of key metrics cards with improved styling and visual indicators"""
    # Get the ticker and timeframe from session state
    current_ticker = st.session_state.get("selected_ticker", "Unknown")
    current_timeframe = st.session_state.get("selected_timeframe", "Unknown")

    # Get the latest progress from progress.yaml file
    try:
        # Use read_progress_from_yaml from progress_helper if available
        try:
            from src.tuning.progress_helper import read_progress_from_yaml

            progress = read_progress_from_yaml()
        except ImportError:
            # Fallback to direct file reading
            progress_file = os.path.join(DATA_DIR, "progress.yaml")
            if os.path.exists(progress_file):
                with open(progress_file, "r") as f:
                    progress = yaml.safe_load(f) or {}
            else:
                progress = {}

        # Log what we read for debugging
        print(f"Read progress data: {progress}")
    except Exception as e:
        print(f"Error reading progress: {e}")
        progress = {}

    # Extract values from progress file
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get(
        "total_trials", 1
    )  # Use 1 as default to avoid division by zero
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)
    cycle = progress.get("cycle", 1)

    # If specific ticker/timeframe don't match progress file, adjust display
    if (
        progress.get("ticker") != current_ticker
        or progress.get("timeframe") != current_timeframe
    ):
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

    # Retrieve best metrics from session (or infinities if not set)
    best_rmse = st.session_state.get("best_metrics", {}).get("rmse", float("inf"))
    best_mape = st.session_state.get("best_metrics", {}).get("mape", float("inf"))

    # Ensure best metrics are numeric
    import numpy as np

    if not isinstance(best_rmse, (int, float)) or np.isnan(best_rmse):
        best_rmse = float("inf")
    if not isinstance(best_mape, (int, float)) or np.isnan(best_mape):
        best_mape = float("inf")

    # Compute improvement deltas if current metrics exist
    rmse_delta = None if current_rmse is None else best_rmse - current_rmse
    mape_delta = None if current_mape is None else best_mape - current_mape

    # Handle any NaN or inf in deltas
    if rmse_delta is not None and (np.isnan(rmse_delta) or np.isinf(rmse_delta)):
        rmse_delta = None
    if mape_delta is not None and (np.isnan(mape_delta) or np.isinf(mape_delta)):
        mape_delta = None

    # Create a 5-column layout for metrics with improved styling
    st.markdown(
        '<div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">',
        unsafe_allow_html=True,
    )

    # Column 1: Cycle indicator - Fix string formatting
    cycle_html = f"""
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">CURRENT CYCLE</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{cycle}</h2>
        <div style="height: 4px; background: #f0f0f0; width: 100%; border-radius: 2px;">
            <div style="height: 100%; width: {min(100, cycle*10)}%; background: #4CAF50; border-radius: 2px;"></div>
        </div>
    </div>
    """
    st.markdown(cycle_html, unsafe_allow_html=True)

    # Column 2: Trial Progress - Fix string formatting
    progress_pct = int((current_trial / total_trials) * 100) if total_trials else 0
    progress_html = f"""
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">TRIAL PROGRESS</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{current_trial}/{total_trials}</h2>
        <div style="height: 4px; background: #f0f0f0; width: 100%; border-radius: 2px;">
            <div style="height: 100%; width: {progress_pct}%; background: #2196F3; border-radius: 2px;"></div>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

    # Column 3: Current RMSE - Fix string formatting
    rmse_color = "#4CAF50" if rmse_delta and rmse_delta > 0 else "#F44336"
    rmse_arrow = "↓" if rmse_delta and rmse_delta > 0 else "↑"
    rmse_delta_display = f"{rmse_arrow} {abs(rmse_delta):.2f}" if rmse_delta else ""

    # Fix null formatting issue with conditional display
    rmse_display = f"{current_rmse:.2f}" if current_rmse is not None else "N/A"

    rmse_html = f"""
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">CURRENT RMSE</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{rmse_display}</h2>
        <p style="font-size: 0.9em; margin: 0; color: {rmse_color};">{rmse_delta_display}</p>
    </div>
    """
    st.markdown(rmse_html, unsafe_allow_html=True)

    # Column 4: Current MAPE - Fix string formatting
    mape_color = "#4CAF50" if mape_delta and mape_delta > 0 else "#F44336"
    mape_arrow = "↓" if mape_delta and mape_delta > 0 else "↑"
    mape_delta_display = f"{mape_arrow} {abs(mape_delta):.2f}%" if mape_delta else ""

    # Fix null formatting issue with conditional display
    mape_display = f"{current_mape:.2f}%" if current_mape is not None else "N/A"

    mape_html = f"""
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">CURRENT MAPE</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{mape_display}</h2>
        <p style="font-size: 0.9em; margin: 0; color: {mape_color};">{mape_delta_display}</p>
    </div>
    """
    st.markdown(mape_html, unsafe_allow_html=True)

    # Column 5: Direction Accuracy - Fix string formatting
    if (
        "ensemble_predictions_log" in st.session_state
        and st.session_state["ensemble_predictions_log"]
    ):
        predictions = st.session_state["ensemble_predictions_log"]
        success_rate = 0

        if len(predictions) > 1:
            try:
                # Calculate direction accuracy
                correct_direction = 0
                for i in range(1, len(predictions)):
                    actual_direction = (
                        predictions[i]["actual"] > predictions[i - 1]["actual"]
                    )
                    pred_direction = (
                        predictions[i]["predicted"] > predictions[i - 1]["predicted"]
                    )
                    if actual_direction == pred_direction:
                        correct_direction += 1

                success_rate = (correct_direction / (len(predictions) - 1)) * 100
            except (KeyError, TypeError) as e:
                logger.error(f"Error calculating direction accuracy: {e}")
                success_rate = 0

        # Color based on accuracy
        accuracy_color = (
            "#4CAF50"
            if success_rate >= 60
            else "#FFC107" if success_rate >= 50 else "#F44336"
        )

        accuracy_html = f"""
        <div class="metric-container" style="flex: 1; min-width: 150px;">
            <p style="font-size: 0.8em; margin: 0; color: #78909c;">DIRECTION ACCURACY</p>
            <h2 style="font-size: 1.8em; margin: 5px 0; color: {accuracy_color};">{success_rate:.1f}%</h2>
            <p style="font-size: 0.9em; margin: 0; color: #78909c;">{len(predictions)} predictions</p>
        </div>
        """
        st.markdown(accuracy_html, unsafe_allow_html=True)
    else:
        st.markdown(
            """
        <div class="metric-container" style="flex: 1; min-width: 150px;">
            <p style="font-size: 0.8em; margin: 0; color: #78909c;">DIRECTION ACCURACY</p>
            <h2 style="font-size: 1.8em; margin: 5px 0;">N/A</h2>
            <p style="font-size: 0.9em; margin: 0; color: #78909c;">No predictions yet</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


@robust_error_boundary
def initiate_shutdown(source="unknown"):
    """Initiate a graceful shutdown of the dashboard."""
    logger.info(f"Shutdown initiated from {source}")

    # Stop the watchdog if it's running
    if "watchdog" in st.session_state:
        try:
            st.session_state["watchdog"].stop()
            logger.info("Stopped watchdog during shutdown")
        except Exception as e:
            logger.error(f"Error stopping watchdog: {e}")

    # Perform any other shutdown tasks here
    try:
        # Save session state if needed
        st.session_state["shutdown_requested"] = True
        st.session_state["shutdown_time"] = datetime.now().isoformat()
        st.session_state["shutdown_source"] = source

        # Display shutdown message
        st.warning("Dashboard is shutting down. Please close this browser tab.")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


"""
UI components for the dashboard.
"""
import os


from src.dashboard.dashboard.dashboard_error import (
    read_tuning_status,
)

# For existing code...


def display_tuning_status():
    """Display the current tuning status with emphasis on cycle and trial information."""
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
            # Larger, more prominent trial information
            current = progress.get("current_trial", 0)
            total = progress.get("total_trials", 0)
            st.markdown(
                f"<h2 style='text-align: center; color:#1E88E5'>Trial {current}/{total}</h2>",
                unsafe_allow_html=True,
            )

            # Add progress bar
            progress_val = progress.get("trial_progress", 0)
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

    # Additional details in expandable section
    with st.expander("Additional Details", expanded=False):
        # Format and display all progress information
        if progress:
            details = []
            for k, v in progress.items():
                if k == "timestamp":
                    try:
                        dt_obj = datetime.fromtimestamp(v)
                        v = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                details.append(f"**{k}:** {v}")

            st.markdown("<br>".join(details), unsafe_allow_html=True)
        else:
            st.info("No tuning progress information available.")


# For existing code...


def display_trials_table(max_trials=20):
    """Display the most recent tuning trials in a table format."""
    if "trial_logs" in st.session_state and st.session_state["trial_logs"]:
        logs = st.session_state["trial_logs"][-max_trials:]

        # Convert to DataFrame for better display
        df_data = []
        for log in logs:
            entry = {
                "Trial": log.get("trial", log.get("number", "N/A")),
                "Model": log.get("params", {}).get("model_type", "N/A"),
                "RMSE": f"{log.get('rmse', 0):.4f}",
                "MAPE": f"{log.get('mape', 0):.2f}%",
                "State": log.get("state", "N/A"),
                "Time": log.get("timestamp", "N/A"),
            }
            df_data.append(entry)

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No trial logs available.")


# For existing code...
