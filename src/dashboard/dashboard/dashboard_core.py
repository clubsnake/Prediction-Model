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

# Use robust error handling
try:
    from dashboard_error import robust_error_boundary, read_tuning_status, load_latest_progress
    print("Successfully imported dashboard_error")
except ImportError as e:
    print(f"Error importing dashboard_error: {e}")
    # Define a simple error boundary if it fails
    def robust_error_boundary(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Error in {func.__name__}: {str(e)}")
                st.code(traceback.format_exc())
                return None
        return wrapper
    
    # Provide a fallback for load_latest_progress
    def load_latest_progress(ticker=None, timeframe=None):
        return {"current_trial": 0, "total_trials": 1, "current_rmse": None, "current_mape": None, "cycle": 1}

# Try to import our dashboard modules - with fallbacks
try:
    from dashboard_ui import create_header, create_control_panel, create_metrics_cards, create_hyperparameter_tuning_panel
    from dashboard_data import load_data as load_market_data, calculate_indicators
    from dashboard_model import start_tuning, stop_tuning, display_tested_models, generate_future_forecast
    from dashboard_visualization import (
        create_interactive_price_chart, 
        create_technical_indicators_chart, 
        show_advanced_dashboard_tabs,
        prepare_dataframe_for_display
    )
    from dashboard_state import init_session_state
except ImportError as e:
    st.error(f"Error importing dashboard modules: {e}")
    st.stop()

# Try to import config
try:
    from config import DATA_DIR, TICKER, TICKERS, TIMEFRAMES
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
        }
        .stApp {
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
        init_session_state()

        # Build UI components - header with status and app info
        create_header()
        
        # Create sidebar with controls
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

        # Get parameters from the control panel
        ticker = params["ticker"]
        timeframe = params["timeframe"]
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        training_start = params.get("training_start_date")
        
        # Calculate derived values automatically
        historical_window = params["historical_window"]
        forecast_window = params["forecast_window"]
        
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
        """.format(params["training_start_date"].strftime('%Y-%m-%d'), 
                 params["start_date"].strftime('%Y-%m-%d'), 
                 params["end_date"].strftime('%Y-%m-%d')))

        # Don't update session state here to avoid overwriting user selections
        # Just use the values from params for this run of the app
        
        # Ensure session state dates are updated
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        st.session_state["training_start_date"] = training_start

        # Show a loading spinner while fetching data
        with st.spinner("Loading market data..."):
            # Fetch historical market data (cached) for the selected ticker/timeframe
            df_vis = load_market_data(
                ticker, 
                params["start_date"].strftime("%Y-%m-%d"), 
                params["end_date"].strftime("%Y-%m-%d"), 
                interval=timeframe
            )
            
        if df_vis is not None and not df_vis.empty:
            # Ensure date column is properly formatted
            try:
                date_col = 'date' if 'date' in df_vis.columns else 'Date'
                if date_col in df_vis.columns:
                    df_vis[date_col] = pd.to_datetime(df_vis[date_col])
            except Exception as e:
                logger.warning(f"Error converting date column: {e}")
                # Create a backup date column
                df_vis['date'] = pd.date_range(start=params["start_date"], periods=len(df_vis))
                
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
                """, unsafe_allow_html=True)
                
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
            
            # Only attempt forecast if model exists and user wants to see it
            model = st.session_state.get('model')
            if model and indicators.get("show_forecast", True):
                with st.spinner("Generating forecast..."):
                    # Determine feature columns (exclude date and target 'Close')
                    feature_cols = [col for col in df_vis.columns if col not in ["date", "Date", "Close"]]
                    future_forecast = generate_future_forecast(model, df_vis, feature_cols)
            
            # Calculate indicators and plot the chart
            df_vis_indicators = calculate_indicators(df_vis)
            
            # Pass indicator options to visualization function
            create_interactive_price_chart(df_vis_indicators, params, 
                                         future_forecast=future_forecast, 
                                         indicators=indicators,
                                         height=700)  # Increase height for better visualization
            
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
            
            # Put cycle/trial info between DA and Progress with larger font
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

        # Store the selected tab index in session state if not already there
        if "selected_main_tab" not in st.session_state:
            st.session_state.selected_main_tab = 0

        # Use callbacks to track selected tab
        def set_main_tab(tab_idx):
            st.session_state.selected_main_tab = tab_idx

        # Tab 1: Model Insights (now the first tab)
        with main_tabs[0]:
            # Invisible button to track tab selection
            st.button("Tab 1", on_click=set_main_tab, args=(0,), key="main_tab_1_btn", help=None, 
                     type="hidden")
            
            # Only render content if this tab is selected
            if st.session_state.selected_main_tab == 0:
                st.subheader("Model Performance & Insights")
                
                insight_tabs = st.tabs(["Performance Metrics", "Feature Importance", "Prediction Analysis"])
                
                # Store nested tab selection
                if "selected_insight_tab" not in st.session_state:
                    st.session_state.selected_insight_tab = 0
                
                def set_insight_tab(tab_idx):
                    st.session_state.selected_insight_tab = tab_idx
                
                with insight_tabs[0]:
                    # Invisible button to track nested tab selection
                    st.button("Insight Tab 1", on_click=set_insight_tab, args=(0,), key="insight_tab_1_btn", 
                             help=None, type="hidden")
                    
                    if st.session_state.selected_insight_tab == 0:
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
                    # Invisible button to track nested tab selection
                    st.button("Insight Tab 2", on_click=set_insight_tab, args=(1,), key="insight_tab_2_btn", 
                             help=None, type="hidden")
                    
                    if st.session_state.selected_insight_tab == 1:
                        if "feature_importance" in st.session_state and st.session_state["feature_importance"]:
                            st.subheader("Feature Importance")
                            # Convert to dataframe for better visualization
                            importance = st.session_state["feature_importance"]
                            feature_df = pd.DataFrame({
                                "Feature": importance.keys(),
                                "Importance": importance.values()
                            }).sort_values("Importance", ascending=False)
                            
                            # Create a bar chart
                            st.bar_chart(feature_df.set_index("Feature"))
                            
                            # Show as table too
                            st.dataframe(prepare_dataframe_for_display(feature_df))
                            
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
                                st.dataframe(prepare_dataframe_for_display(tabnet_df))
                        else:
                            st.info("No feature importance data available yet.")
                
                with insight_tabs[2]:
                    # Invisible button to track nested tab selection
                    st.button("Insight Tab 3", on_click=set_insight_tab, args=(2,), key="insight_tab_3_btn", 
                             help=None, type="hidden")
                    
                    if st.session_state.selected_insight_tab == 2:
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

        # Tab 2: Model Tuning (was Tab 1)
        with main_tabs[1]:
            # Invisible button to track tab selection
            st.button("Tab 2", on_click=set_main_tab, args=(1,), key="main_tab_2_btn", help=None, 
                     type="hidden")
            
            # Only render content if this tab is selected
            if st.session_state.selected_main_tab == 1:
                st.subheader("Model Tuning & Optimization")
                
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
                    display_tested_models()

        # Tab 3: Technical Analysis (was Tab 2)
        with main_tabs[2]:
            # Invisible button to track tab selection
            st.button("Tab 3", on_click=set_main_tab, args=(2,), key="main_tab_3_btn", help=None, 
                     type="hidden")
            
            # Only render content if this tab is selected
            if st.session_state.selected_main_tab == 2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.subheader("Technical Indicators")
                
                with st.spinner("Loading technical indicators..."):
                    # Enhanced technical indicators
                    create_technical_indicators_chart(df_vis_indicators, params)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Advanced analysis dashboard
                with st.spinner("Loading advanced analysis..."):
                    show_advanced_dashboard_tabs(df_vis_indicators)

        # Tab 4: Price Data (was Tab 1)
        with main_tabs[3]:
            # Invisible button to track tab selection
            st.button("Tab 4", on_click=set_main_tab, args=(3,), key="main_tab_4_btn", help=None, 
                     type="hidden")
            
            # Only render content if this tab is selected
            if st.session_state.selected_main_tab == 3:
                with st.spinner("Loading price data..."):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.subheader("Data Summary")
                        st.write(f"**Ticker:** {ticker}")
                        st.write(f"**Timeframe:** {timeframe}")
                        st.write(f"**Period:** {start_date} to {end_date}")
                        st.write(f"**Data points:** {len(df_vis)}")
                        
                        # Show key statistics - Directly use pandas values without formatting
                        if len(df_vis) > 0:
                            # Fix: don't format pandas values with f-strings
                            st.metric("Current Price", df_vis['Close'].iloc[-1])
                            if len(df_vis) > 1:
                                change = df_vis['Close'].iloc[-1] - df_vis['Close'].iloc[-2]
                                pct_change = (change / df_vis['Close'].iloc[-2]) * 100
                                
                                # Convert to float and handle None
                                change = float(change) if change is not None else 0.0
                                pct_change = float(pct_change) if pct_change is not None else 0.0
                                
                                st.metric("Last Change", change, "{:.2f}%".format(pct_change))
                        
                        # Download button for CSV
                        csv_data = df_vis.to_csv(index=False)
                        b64 = base64.b64encode(csv_data.encode()).decode()
                        download_link = '<a href="data:file/csv;base64,{}" download="{}_{}_data.csv" class="download-button">📥 Download Data</a>'.format(
                            b64, ticker, timeframe
                        )
                        st.markdown(download_link, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Market Data")
                        
                        # Prepare dataframe for display
                        display_df = prepare_dataframe_for_display(df_vis)
                        
                        # Now safely display it
                        st.dataframe(
                            display_df.style.background_gradient(
                                cmap='Blues', 
                                subset=['Volume']
                            ).format({
                                'Open': '${:,.2f}',
                                'High': '${:,.2f}',
                                'Low': '${:,.2f}',
                                'Close': '${:,.2f}',
                                'Volume': '{:,.0f}'
                            }),
                            height=400
                        )
                        
                        with st.expander("Summary Statistics"):
                            stats_df = prepare_dataframe_for_display(df_vis.describe().transpose())
                            st.dataframe(stats_df)

        # Add hyperparameter tuning section - this section always renders
        st.markdown("<a name='hyperparameter-tuning'></a>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Check if we should show the tuning panel
        if "show_tuning_panel" not in st.session_state:
            st.session_state.show_tuning_panel = False
            
        if st.button("Toggle Hyperparameter Tuning Panel", key="toggle_tuning_btn"):
            st.session_state.show_tuning_panel = not st.session_state.show_tuning_panel
            
        if st.session_state.show_tuning_panel:
            with st.spinner("Loading hyperparameter tuning panel..."):
                create_hyperparameter_tuning_panel()

        # Auto-refresh logic
        if params["auto_refresh"]:
            current_time = datetime.now().timestamp()
            if current_time - st.session_state["last_refresh"] >= params["refresh_interval"]:
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