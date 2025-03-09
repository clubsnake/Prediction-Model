"""
dashboard_core.py

Core functionality for the unified dashboard. This file serves as the main entry point
for the dashboard application and orchestrates all the dashboard components.
"""

import os
import sys
import traceback
from datetime import datetime, timedelta
import base64

# Add project root to Python path for reliable imports
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

# Then try to import the real one (if available)
try:
    # Import dashboard error handling separately to avoid circular imports
    from src.dashboard.dashboard.dashboard_error import robust_error_boundary, read_tuning_status, load_latest_progress
    print("Successfully imported dashboard_error")
except ImportError as e:
    print(f"Error importing dashboard_error: {e}")
    # Use the fallback function we defined
    robust_error_boundary = simple_error_boundary
    
    # Define fallback functions
    def read_tuning_status():
        return {"status": "unknown", "is_running": False}
        
    def load_latest_progress(ticker=None, timeframe=None):
        return {"current_trial": 0, "total_trials": 1, "current_rmse": None, "current_mape": None, "cycle": 1}

# Import dashboard modules with error handling
def import_dashboard_modules():
    """Import dashboard modules with error handling to avoid circular imports"""
    modules = {}
    
    try:
        # Import UI components
        try:
            from src.dashboard.dashboard.dashboard_ui import create_header, create_control_panel, create_metrics_cards, create_hyperparameter_tuning_panel, create_interactive_price_chart
            from src.dashboard.dashboard.dashboard_ui import start_tuning, stop_tuning
            modules.update({
                "create_header": create_header,
                "create_control_panel": create_control_panel,
                "create_metrics_cards": create_metrics_cards,
                "create_hyperparameter_tuning_panel": create_hyperparameter_tuning_panel,
                "create_interactive_price_chart": create_interactive_price_chart,
                "start_tuning": start_tuning,
                "stop_tuning": stop_tuning
            })
        except ImportError as e:
            print(f"Error importing dashboard_ui: {e}")
            # Define fallback functions
            def fallback_header():
                st.title("AI Price Prediction Dashboard")
            
            def fallback_control_panel():
                ticker = st.sidebar.selectbox("Ticker", TICKERS)
                timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES)
                return {"ticker": ticker, "timeframe": timeframe}
            
            def fallback_metrics():
                st.info("Metrics not available")
                
            def fallback_tuning_panel():
                st.warning("Hyperparameter tuning panel is not available")
                
            def fallback_start_tuning(ticker, timeframe, multipliers=None):
                st.warning("Tuning functionality not available")
                
            def fallback_stop_tuning():
                st.warning("Tuning functionality not available")
                
            modules.update({
                "create_header": fallback_header,
                "create_control_panel": fallback_control_panel,
                "create_metrics_cards": fallback_metrics,
                "create_hyperparameter_tuning_panel": fallback_tuning_panel,
                "start_tuning": fallback_start_tuning,
                "stop_tuning": fallback_stop_tuning
            })
        
        # Try to import hyperparameter dashboard as a fallback
        try:
            from src.dashboard.hyperparameter_dashboard import main as hyperparameter_dashboard
            if "create_hyperparameter_tuning_panel" not in modules:
                modules["create_hyperparameter_tuning_panel"] = hyperparameter_dashboard
        except ImportError:
            print("Error importing hyperparameter_dashboard")
        
        try:
            from src.dashboard.dashboard.dashboard_data import load_data, calculate_indicators, generate_dashboard_forecast, ensure_date_column
            modules.update({
                "load_data": load_data,
                "calculate_indicators": calculate_indicators,
                "generate_dashboard_forecast": generate_dashboard_forecast,
                "ensure_date_column": ensure_date_column
            })
        except ImportError as e:
            print(f"Error importing dashboard_data: {e}")
            # Define fallback functions for data operations
            def fallback_load_data(ticker, start_date, end_date=None, interval="1d", training_mode=False):
                print(f"Using fallback load_data for {ticker}")
                return None
                
            def fallback_calculate_indicators(df):
                print("Using fallback calculate_indicators")
                return df
                
            def fallback_generate_forecast(model, df, feature_cols):
                print("Using fallback generate_forecast")
                return []
            
            def fallback_ensure_date_column(df, default_name='date'):
                print("Using fallback ensure_date_column")
                if df is None or df.empty:
                    return df, default_name
                
                # Try to find a date column
                date_col = None
                if 'date' in df.columns:
                    date_col = 'date'
                elif 'Date' in df.columns:
                    date_col = 'Date'
                    
                # If no date column found, create one
                if date_col is None:
                    import pandas as pd
                    from datetime import datetime, timedelta
                    df = df.copy()
                    df[default_name] = pd.date_range(start=datetime.now() - timedelta(days=len(df)), periods=len(df))
                    date_col = default_name
                
                return df, date_col
                
            modules.update({
                "load_data": fallback_load_data,
                "calculate_indicators": fallback_calculate_indicators,
                "generate_dashboard_forecast": fallback_generate_forecast,
                "ensure_date_column": fallback_ensure_date_column
            })
        
        # Import visualization functions
        try:
            from src.dashboard.dashboard.dashboard_visualization import create_interactive_price_chart, create_technical_indicators_chart, prepare_dataframe_for_display
            modules.update({
                "create_interactive_price_chart": create_interactive_price_chart,
                "create_technical_indicators_chart": create_technical_indicators_chart,
                "prepare_dataframe_for_display": prepare_dataframe_for_display
            })
        except ImportError as e:
            print(f"Error importing dashboard_visualization: {e}")
        
        # Add standardize_column_names utility function
        def standardize_column_names(df, ticker=None):
            """
            Standardize column names by removing ticker-specific parts for OHLCV.
            """
            if df is None or df.empty:
                return df
            df_copy = df.copy()
            
            # First check if standard columns already exist
            has_standard = all(col in df_copy.columns for col in ['Open', 'High', 'Low', 'Close'])
            if has_standard:
                return df_copy
                
            # Handle ticker-specific column names
            if ticker:
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
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
            for old_col, std_col in [('open', 'Open'), ('high', 'High'), ('low', 'Low'), 
                                      ('close', 'Close'), ('volume', 'Volume')]:
                if old_col in df_copy.columns and std_col not in df_copy.columns:
                    df_copy[std_col] = df_copy[old_col]
            
            # Log missing columns that we couldn't standardize
            missing = [c for c in ['Open', 'High', 'Low', 'Close'] if c not in df_copy.columns]
            if missing:
                print(f"Warning: Missing required columns after standardization: {missing}")
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
        st.markdown("""
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
        """, unsafe_allow_html=True)
    except Exception as e:
        # This might fail if called twice
        pass

@robust_error_boundary
def initialize_session_state():
    if "error_log" not in st.session_state:
        st.session_state["error_log"] = []
    if "metrics_container" not in st.session_state:
        st.session_state["metrics_container"] = None
    # Add other necessary state initializations here
        
@robust_error_boundary
def main_dashboard():
    """Main dashboard entry point with robust error handling"""
    try:
        # Initialize session state at the beginning
        initialize_session_state()
        
        # Setup page and session state
        set_page_config()
        
        # Build UI components - header with status and app info
        create_header = dashboard_modules.get("create_header", lambda: st.title("Dashboard"))
        create_header()
        
        # Create sidebar with controls
        create_control_panel = dashboard_modules.get("create_control_panel", lambda: {"ticker": TICKER, "timeframe": "1d"})
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
                "training_start_date": datetime.now() - timedelta(days=365*2),
                "historical_window": 90,
                "forecast_window": 30, 
                "auto_refresh": True,
                "refresh_interval": 30
            }
        
        # Get parameters from the control panel state for other components
        ticker = params["ticker"]
        timeframe = params["timeframe"]
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        training_start = params.get("training_start_date")
        
        # Calculate derived values automatically
        historical_window = params.get("historical_window")
        forecast_window = params.get("forecast_window")
        
        # Make sure these values are in session state for other components
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["training_start_date"] = training_start
        st.session_state["historical_window"] = historical_window 
        st.session_state["forecast_window"] = forecast_window
        
        # Instead show a cleaner summary
        st.sidebar.info("""
        **Dashboard Settings:**
        - Training data from: {}
        - Display range: {} to {}
        """.format(params.get("training_start_date", datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'), 
                 params.get("start_date", datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                 params.get("end_date", datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')))
        
        # Ensure session state dates are updated
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["training_start_date"] = training_start
        
        # Show a loading spinner while fetching data
        with st.spinner("Loading market data..."):
            # Modify this part to ensure we only fetch historical data up to today
            today = datetime.now().strftime("%Y-%m-%d")
            # Fetch historical market data (cached) for the selected ticker/timeframe
            df_vis = dashboard_modules["load_data"](
                ticker, 
                params.get("start_date", datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), 
                today,  # Use current date instead of future end_date
                timeframe
            )
            if df_vis is not None and not df_vis.empty:
                # Standardize column names
                df_vis = dashboard_modules["standardize_column_names"](df_vis, ticker)
                # Fix: Use the centralized date handling function
                df_vis, date_col = dashboard_modules["ensure_date_column"](df_vis)
                st.session_state["df_raw"] = df_vis  # cache data in session
            else:
                # If no new data, use cached data if available
                if st.session_state.get("df_raw") is not None:
                    df_vis = st.session_state["df_raw"]
                    st.warning("Using previously loaded data as new data could not be fetched.")
                else:
                    st.error(f"No data available for {ticker} from {start_date} to {end_date}.")
                    return  # abort if we have absolutely no data

        # UNIFIED CHART AT THE TOP (replacing metrics cards)
        # Create a container for the chart with styling
        chart_container = st.container()
        with chart_container:
            # Add ticker/timeframe and model status as a header row
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                # Show significant metrics
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="background-color: rgba(33, 150, 243, 0.1); padding: 5px 10px; 
                    border-radius: 4px; color: #2196F3; font-weight: bold; margin-right: 15px;">
                        {ticker} / {timeframe}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with header_col2:
                # Model status badge
                model = st.session_state.get('model')
                if model:
                    st.markdown("""
                    <span style="background-color: rgba(76, 175, 80, 0.1); padding: 5px 10px; 
                    border-radius: 4px; color: #4CAF50; font-weight: bold;">
                        ✓ Model Loaded
                    </span>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <span style="background-color: rgba(255, 152, 0, 0.1); padding: 5px 10px; 
                    border-radius: 4px; color: #FF9800; font-weight: bold;">
                        ⚠ No Model Loaded
                    </span>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Full-width chart
            # Initialize future_forecast as None before attempting to generate it
            future_forecast = None
            # Get indicator preferences from params
            indicators = params.get("indicators", {})
            model = st.session_state.get('model')
            if model and indicators.get("show_forecast", True):
                with st.spinner("Generating forecast..."):
                    # Determine feature columns (exclude date and target 'Close')
                    feature_cols = [col for col in df_vis.columns if col not in ["date", "Date", "Close"]]
                    # Use the consolidated function from dashboard_data
                    from src.dashboard.dashboard.dashboard_data import generate_dashboard_forecast
                    future_forecast = generate_dashboard_forecast(model, df_vis, feature_cols)
                    # Save the prediction for historical comparison
                    if future_forecast and len(future_forecast) > 0:
                        from src.dashboard.dashboard.dashboard_visualization import save_best_prediction
                        save_best_prediction(df_vis, future_forecast)
            
            # Calculate indicators and plot the chart
            df_vis_indicators = dashboard_modules["calculate_indicators"](df_vis)
            # Pass indicator options to visualization function
            dashboard_modules["create_interactive_price_chart"](df_vis_indicators, params, 
                                         future_forecast=future_forecast, 
                                         indicators=indicators,
                                         height=700) 
            
            # Get model metrics and progress information
            best_metrics = st.session_state.get("best_metrics", {})
            best_rmse = best_metrics.get("rmse")
            best_mape = best_metrics.get("mape")
            direction_accuracy = best_metrics.get("direction_accuracy")
            
            # Get progress data from YAML
            progress = load_latest_progress(ticker=ticker, timeframe=timeframe)
            cycle = progress.get("cycle", 1)
            current_trial = progress.get("current_trial", 0)
            total_trials = progress.get("total_trials", 1)
            
            # Show metrics below chart in a row
            metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
            
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
                st.markdown("""
                <div style="font-size: 1.15em;">
                    <p style="margin-bottom: 5px;"><strong>Cycle Status:</strong></p>
                    <p>Cycle: <strong>{}</strong> | Trial: <strong>{}/{}</strong></p>
                </div>
                """.format(cycle, current_trial, total_trials), unsafe_allow_html=True)
            # Progress bar in the last column
            with metrics_col5:
                st.markdown("<p style='font-size: 1.15em; margin-bottom: 5px;'><strong>Cycle Progress:</strong></p>", unsafe_allow_html=True)
                st.progress(min(1.0, current_trial/max(1, total_trials)))
        
        # Create main content tabs with enhanced UI
        main_tabs = st.tabs([
            "🧠 Model Insights",
            "⚙️ Model Tuning",
            "📈 Technical Indicators", 
            "📊 Price Data"
        ])
        
        # Tab 1: Model Insights
        with main_tabs[0]:
            st.subheader("Model Performance & Insights")
            insight_tabs = st.tabs(["Performance Metrics", "Feature Importance", "Prediction Analysis"])
            
            with insight_tabs[0]:
                if "best_metrics" in st.session_state and st.session_state["best_metrics"]:
                    metrics = st.session_state["best_metrics"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best RMSE", metrics.get('rmse', 'N/A'))
                    with col2:
                        st.metric("Best MAPE", f"{metrics.get('mape', 'N/A'):.2f}%")
                    with col3:
                        st.metric("Direction Accuracy", f"{metrics.get('direction_accuracy', 'N/A'):.1f}%")
                    with col4:
                        st.metric("Model Type", metrics.get('model_type', 'N/A'))
                    # Show performance history if available
                    if "model_history" in st.session_state and st.session_state["model_history"]:
                        st.subheader("Training History")
                        history_df = pd.DataFrame(st.session_state["model_history"])
                        st.line_chart(history_df)
                else:
                    st.info("No model metrics available yet. Train a model to see performance insights.")
            
            with insight_tabs[1]:
                if "feature_importance" in st.session_state and st.session_state["feature_importance"]:
                    st.subheader("Feature Importance")
                    feature_df = pd.DataFrame({
                        "Feature": st.session_state["feature_importance"].keys(),
                        "Importance": st.session_state["feature_importance"].values()
                    }).sort_values("Importance", ascending=False)
                    
                    st.bar_chart(feature_df.set_index("Feature"))
                    
                    # Show as table too
                    st.dataframe(dashboard_modules["prepare_dataframe_for_display"](feature_df))
                    
                    # Add TabNet-specific feature importance if available
                    if "tabnet_feature_importance" in st.session_state:
                        st.subheader("TabNet-Specific Feature Importance")
                        st.markdown("""
                        TabNet uses a sparse feature selection mechanism that provides different insights
                        than traditional feature importance metrics.
                        """)
                        
                        tabnet_imp = st.session_state["tabnet_feature_importance"]
                        tabnet_df = pd.DataFrame({
                            "Feature": tabnet_imp.keys(),
                            "Importance": tabnet_imp.values()
                        }).sort_values("Importance", ascending=False)
                        
                        # Create a bar chart with a different color 
                        st.bar_chart(tabnet_df.set_index("Feature"))
                        
                        # Show explanatory text
                        st.markdown("""
                        **Interpretation:** TabNet's feature importance values indicate which features were most
                        influential in the model's decision making through its sparse attention mechanism.
                        Higher values indicate features that received more attention during the prediction process.
                        """)
                        
                        # Show as table with prepared dataframe
                        st.dataframe(dashboard_modules["prepare_dataframe_for_display"](tabnet_df))
                else:
                    st.info("No feature importance data available yet.")
            
            with insight_tabs[2]:
                if "prediction_history" in st.session_state and st.session_state["prediction_history"]:
                    st.subheader("Prediction History")
                    pred_df = pd.DataFrame(st.session_state["prediction_history"])
                    
                    # Show accuracy metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Prediction Error", "{:.2f}%".format(pred_df['error'].mean()))
                    with col2:
                        # Calculate direction accuracy
                        correct_dir = (pred_df["actual_direction"] == pred_df["predicted_direction"]).mean() * 100
                        st.metric("Direction Accuracy", "{:.1f}%".format(correct_dir))
                    
                    # Show prediction vs actual
                    st.line_chart(pred_df[["actual", "predicted"]])
                else:
                    st.info("No prediction history available yet.")

        # Tab 2: Model Tuning
        with main_tabs[1]:
            st.subheader("Model Tuning & Optimization")
            
            # Move hyperparameter tuning into this tab
            tuning_tabs = st.tabs(["Parameters", "Hyperparameter Tuning"])
            
            with tuning_tabs[0]:
                tuning_col1, tuning_col2 = st.columns([1, 1])
                
                with tuning_col2:
                    # Show current best parameters if available
                    st.markdown("### Current Best Parameters")
                    if "best_params" in st.session_state and st.session_state["best_params"]:
                        best_params = st.session_state["best_params"]
                        for param, value in best_params.items():
                            st.write(f"**{param}:** {value}")
                    else:
                        st.info("No best parameters available yet.")
                
                # Show tested models in an expandable section
                with st.expander("View All Tested Models", expanded=False):
                    display_tested_models = dashboard_modules.get("display_tested_models", lambda: st.info("Tested models display not available"))
                    display_tested_models()
            
            with tuning_tabs[1]:
                # Hyperparameter tuning panel directly embedded here
                with st.spinner("Loading hyperparameter tuning panel..."):
                    dashboard_modules["create_hyperparameter_tuning_panel"]()

        # Tab 3: Technical Analysis
        with main_tabs[2]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Technical Indicators")
            
            with st.spinner("Loading technical indicators..."):
                # Only render if we have data
                if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
                    # Calculate indicators on the FULL dataset
                    df_indicators = dashboard_modules["calculate_indicators"](st.session_state['df_raw'])
                    
                    # Enhanced technical indicators
                    try:
                        dashboard_modules["create_technical_indicators_chart"](df_indicators, params)
                    except Exception as e:
                        st.error(f"Error creating technical indicators chart: {e}")
                else:
                    st.warning("No data available for technical indicators")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Advanced analysis dashboard
            if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
                with st.spinner("Loading advanced analysis..."):
                    try:
                        dashboard_modules["show_advanced_dashboard_tabs"](st.session_state['df_raw'])
                    except Exception as e:
                        st.error(f"Error creating advanced analysis: {e}")
            else:
                st.warning("No data available for advanced analysis")

        # Tab 4: Price Data
        with main_tabs[3]:
            with st.spinner("Loading price data..."):
                if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
                    df_vis = st.session_state['df_raw']
                    
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
                                st.metric("Current Price", f"${df_vis['Close'].iloc[-1]:.2f}")
                                if len(df_vis) > 1:
                                    change = float(df_vis['Close'].iloc[-1]) - float(df_vis['Close'].iloc[-2])
                                    pct_change = (change / float(df_vis['Close'].iloc[-2])) * 100
                                    st.metric("Last Change", f"${change:.2f}", f"{pct_change:.2f}%")
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
                            display_df = dashboard_modules["prepare_dataframe_for_display"](df_vis)
                            
                            # FIX: Check if display_df is not None and not empty before styling
                            if display_df is not None and not display_df.empty:
                                try:
                                    # Since styling can fail, wrap it in try/except
                                    # Show only a preview (first 100 rows) for performance
                                    preview_df = display_df.head(100)
                                    styled_df = preview_df.style.format({
                                        'Open': '${:,.2f}',
                                        'High': '${:,.2f}',
                                        'Low': '${:,.2f}',
                                        'Close': '${:,.2f}',
                                        'Volume': '{:,.0f}'
                                    })
                                    st.dataframe(styled_df, height=400)
                                    
                                    # Show total number of rows
                                    if len(display_df) > 100:
                                        st.info(f"Showing 100 of {len(display_df)} rows. Download the CSV for full data.")
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
                                    numeric_cols = df_vis.select_dtypes(include=['float64', 'int64']).columns
                                    stats_df = df_vis[numeric_cols].describe().transpose()
                                    st.dataframe(stats_df)
                                except Exception as e:
                                    st.error(f"Error displaying summary statistics: {e}")
                            else:
                                st.info("No data available for statistics.")
                else:
                    st.warning("No data available to display.")

        # Auto-refresh logic
        if params.get("auto_refresh", False):
            current_time = datetime.now().timestamp()
            if "last_refresh" in st.session_state and current_time - st.session_state["last_refresh"] >= params.get("refresh_interval", 30):
                # Update last refresh time
                st.session_state["last_refresh"] = current_time
                # Rerun the app
                st.experimental_rerun()

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