"""
Revised dashboard.py

This dashboard integrates:
- Tuning progress and logs,
- A unified price chart (historical and predicted),
- Model visualization,
- Ensemble prediction logs,
- Advanced Dashboard tabs (WERPI, technical indicators, sentiment, learning visualization, ensemble learning),
- Model Save/Load functionality.

Key improvements:
• Uses a dedicated progress container updated via update_progress_display().
• Uses a full page refresh (st.experimental_rerun) only at a full refresh interval.
• Uses meta_tuning’s thread-safe stop mechanism.
"""

import sys
import os
import time
import threading
from datetime import datetime, timedelta
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf

# Append paths for modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import resource_config  # Configure GPU first


# Import modules from your project
import meta_tuning  # Import the whole module to avoid circular imports
from meta_tuning import set_stop_requested  # Import the set_stop_requested function
from config import (
    TICKERS, TIMEFRAMES, START_DATE, RMSE_THRESHOLD, MAPE_THRESHOLD,
    N_STARTUP_TRIALS, TUNING_TRIALS_PER_CYCLE_max, TICKER
)
from data import fetch_data, validate_data
from features import feature_engineering
from model_visualization import (
    visualize_neural_network,
    plot_training_history,
    plot_feature_importance,
    plot_prediction_errors
)
from visualization import (
    visualize_weight_histograms
)
from preprocessing import create_sequences, scale_data
from utils import validate_dataframe

# Advanced dashboard functions
from advanced_dashboard import (
    load_data, calculate_indicators, WERPIIndicator, create_werpi_chart,
    create_secondary_charts, get_sentiment_data, visualize_sentiment,
    visualize_learning_progress, create_learning_animation, animate_ensemble_learning
)

# Directories and file paths
DATA_DIR = "Data"
DB_DIR = os.path.join(DATA_DIR, "DB")
LOGS_DIR = os.path.join(DATA_DIR, "Logs")
MODELS_DIR = os.path.join(DATA_DIR, "Models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")
CYCLE_METRICS_FILE = os.path.join(DATA_DIR, "cycle_metrics.yaml")
BEST_PARAMS_FILE = os.path.join(DATA_DIR, "best_params.yaml")

# -------------------------------
# Tuning status file helpers
# -------------------------------
def write_tuning_status(status: bool):
    """Write tuning status to file."""
    try:
        with open(TUNING_STATUS_FILE, "w") as f:
            f.write(str(status))
    except Exception as e:
        print(f"Error writing tuning status: {e}")

def read_tuning_status() -> bool:
    """Read tuning status from file."""
    try:
        if os.path.exists(TUNING_STATUS_FILE):
            with open(TUNING_STATUS_FILE, "r") as f:
                status = f.read().strip()
                return status.lower() == "true"
        return False
    except Exception as e:
        print(f"Error reading tuning status: {e}")
        return False

# -------------------------------
# Progress file and display helpers
# -------------------------------
def load_latest_progress():
    """Get most up-to-date progress from either session state or file"""
    # First check session state (fastest)
    if 'live_progress' in st.session_state:
        return st.session_state['live_progress']
        
    # Fall back to file if no session state data or dashboard just loaded
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r") as f:
                progress = yaml.safe_load(f) or {}
            return progress
        return {}
    except Exception as e:
        print(f"Error loading progress: {e}")
        return {}

def update_progress_display():
    """Update just the progress display without a full dashboard rerun."""
    # Create dedicated containers if not already set
    if 'progress_container' not in st.session_state:
        st.session_state['progress_container'] = st.empty()
        st.session_state['progress_bar'] = st.empty()
        st.session_state['metrics_container'] = st.empty()
    progress = load_latest_progress()
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get("total_trials", 0)
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)
    
    # Update progress bar and container
    if total_trials > 0:
        percent = int((current_trial / total_trials) * 100)
        st.session_state['progress_bar'].progress(current_trial / total_trials)
        st.session_state['progress_container'].markdown(
            f"### Trial Progress: {current_trial}/{total_trials} ({percent}%)"
        )
    with st.session_state['metrics_container'].container():
        cols = st.columns(3)
        with cols[0]:
            st.metric("Current Trial", current_trial)
        with cols[1]:
            st.metric("Current RMSE", f"{current_rmse:.4f}" if current_rmse is not None else "N/A")
        with cols[2]:
            st.metric("Current MAPE", f"{current_mape:.2f}%" if current_mape is not None else "N/A")

# -------------------------------
# Session state initialization
# -------------------------------
def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        st.session_state.setdefault("tuning_in_progress", False)
        st.session_state.setdefault("best_metrics", {})
        st.session_state.setdefault("model_history", [])
        st.session_state.setdefault("last_refresh", time.time())
        st.session_state.setdefault("prediction_history", [])
        st.session_state.setdefault("accuracy_metrics_history", {})
        st.session_state.setdefault("neural_network_state", {})
        st.session_state.setdefault("current_model", None)
        st.session_state.setdefault("model_loaded", False)
        st.session_state.setdefault("error_log", [])
        st.session_state.setdefault("training_history", None)
        st.session_state.setdefault("ensemble_predictions_log", [])
        st.session_state.setdefault("df_raw", None)
        st.session_state.setdefault("historical_window", 30)
        st.session_state.setdefault("forecast_window", 30)
        st.session_state.setdefault("trials_per_cycle", N_STARTUP_TRIALS)
        st.session_state.setdefault("initial_trials", N_STARTUP_TRIALS)
        st.session_state.setdefault("saved_model_dir", "saved_models")
        # Auto-refresh intervals (in seconds)
        st.session_state.setdefault("progress_refresh_sec", 2)
        st.session_state.setdefault("full_refresh_sec", 30)
        st.session_state.setdefault("last_progress_refresh", time.time())
        st.session_state.setdefault("last_full_refresh", time.time())
        st.session_state.setdefault("update_heavy_components", True)
    # Initialize ensemble predictions log
    if "ensemble_predictions_log" not in st.session_state:
        st.session_state["ensemble_predictions_log"] = []

# -------------------------------
# Error boundary decorator
# -------------------------------
def error_boundary(func):
    def wrapper(*args, **kwargs):
        st.session_state.setdefault('error_log', [])
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(f"Error in {func.__name__}: {str(e)}")
            st.session_state.error_log.append({
                'timestamp': datetime.now(),
                'function': func.__name__,
                'error': str(e)
            })
            return None
    return wrapper

# -------------------------------
# Ensemble predictions logging
# -------------------------------
def record_ensemble_prediction(date, actual, predicted, rmse=None, mape=None):
    if 'ensemble_predictions_log' not in st.session_state:
        st.session_state['ensemble_predictions_log'] = []
    error = abs(predicted - actual)
    st.session_state['ensemble_predictions_log'].append({
        'timestamp': datetime.now(),
        'date': date,
        'actual': float(actual),
        'predicted': float(predicted),
        'error': float(error),
        'rmse': rmse,
        'mape': mape
    })
    log_file = "ensemble_predictions.csv"
    df_log = pd.DataFrame(st.session_state['ensemble_predictions_log'])
    df_log.to_csv(log_file, index=False)

def display_past_ensemble_predictions():
    if not st.session_state.get('ensemble_predictions_log'):
        st.info("No past ensemble predictions logged yet.")
        return
    df_log = pd.DataFrame(st.session_state['ensemble_predictions_log'])
    df_log.sort_values('date', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_log['date'], y=df_log['actual'],
                             mode='lines+markers', name='Actual Price'))
    fig.add_trace(go.Scatter(x=df_log['date'], y=df_log['predicted'],
                             mode='lines+markers', name='Predicted Price'))
    fig.update_layout(title="Past Ensemble Predictions vs. Actual",
                      xaxis_title="Date", yaxis_title="Price (Close)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Detailed Ensemble Prediction Log")
    st.dataframe(df_log)

# -------------------------------
# Chart and visualization functions
# -------------------------------
@error_boundary
def show_unified_chart(df, historical_window, forecast_window):
    if df.empty:
        st.warning("DataFrame is empty; cannot display chart.")
        return
    current_idx = len(df) - 1
    hist_start = max(0, current_idx - historical_window)
    hist_data = df.iloc[hist_start:current_idx+1]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data['date'],
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Historical'
    ))
    if st.session_state.get('ensemble_predictions_log'):
        df_log = pd.DataFrame(st.session_state['ensemble_predictions_log'])
        last_date = hist_data['date'].iloc[-1]
        df_past = df_log[df_log['date'] <= last_date].copy()
        if not df_past.empty:
            fig.add_trace(go.Scatter(
                x=df_past['date'], y=df_past['predicted'],
                mode='markers', name='Past Predictions',
                marker=dict(color='blue', symbol='circle')
            ))
    if 'future_forecast' in st.session_state:
        future_array = st.session_state['future_forecast']
        if isinstance(future_array, np.ndarray) and future_array.size == forecast_window:
            future_dates = pd.date_range(
                start=hist_data['date'].iloc[-1] + timedelta(days=1),
                periods=forecast_window, freq='D'
            )
            fig.add_trace(go.Scatter(
                x=future_dates, y=future_array,
                mode='lines+markers', name='Future Predictions',
                line=dict(dash='dot', color='red')
            ))
    progress = load_latest_progress()
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)
    rmse_text = f"RMSE: {current_rmse:.2f}" if current_rmse is not None else "RMSE: N/A"
    mape_text = f"MAPE: {current_mape:.2f}%" if current_mape is not None else "MAPE: N/A"
    fig.update_layout(
        title=f"Price Chart (Past & Future Predictions) | {rmse_text} | {mape_text}",
        xaxis_title="Date", yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

@error_boundary
def show_model_visualization():
    if not st.session_state.get('update_heavy_components', True) and st.session_state.get('tuning_in_progress', False):
        st.info("Chart will update on next full refresh...")
        return
    st.subheader("Model Architecture & Training History")
    model = st.session_state.get('current_model')
    if model is None:
        st.info("No model found. Please train or load a model first.")
        return
    feature_names = []
    try:
        from config import get_active_feature_names
        feature_names = get_active_feature_names()
    except:
        pass
    arch_fig = visualize_neural_network(model, feature_names)
    if arch_fig:
        st.pyplot(arch_fig)
    history_data = st.session_state.get('training_history', None)
    if isinstance(history_data, dict):
        st.subheader("Training History (Plotly)")
        fig_hist = plot_training_history(history_data)
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True)
    if st.checkbox("Show Weight Histograms"):
        st.write("Generating weight histograms.")
        visualize_weight_histograms(model)

@error_boundary
def show_predictions_vs_actuals(df, historical_window, forecast_window):
    if not st.session_state.get('update_heavy_components', True) and st.session_state.get('tuning_in_progress', False):
        st.info("Chart will update on next full refresh...")
        return
    st.subheader("Prediction Errors & Analysis")
    model = st.session_state.get('current_model')
    if not model:
        st.info("No model loaded. Please load or train one.")
        return
    feature_cols = []
    try:
        from config import get_active_feature_names
        feature_cols = get_active_feature_names()
    except:
        pass
    scaled_df, _, _ = scale_data(df, feature_cols, "Close")
    X, y = create_sequences(scaled_df, feature_cols, "Close", lookback=30, horizon=10)
    if len(X) == 0:
        st.warning("Not enough data to create sequences.")
    else:
        preds = model.predict(X)
        if preds.shape == y.shape:
            st.subheader("Error Plot")
            fig_err = plot_prediction_errors(y.flatten(), preds.flatten())
            if fig_err:
                st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.info("Prediction shape doesn't match; cannot display error plot.")
    st.subheader("Data Preview")
    current_idx = len(df) - 1
    hist_start = max(0, current_idx - historical_window)
    hist_data = df.iloc[hist_start:current_idx+1]
    st.write(f"Showing last {historical_window} days of data plus {forecast_window} forecast days.")
    st.dataframe(hist_data)

def display_tuning_logs():
    if 'trial_logs' in st.session_state and st.session_state['trial_logs']:
        st.write("### Detailed Tuning Logs")
        df_logs = pd.DataFrame(st.session_state['trial_logs'])
        st.dataframe(df_logs)
    else:
        st.info("No tuning logs available yet.")

def display_feature_correlations():
    if "df_raw" not in st.session_state or st.session_state["df_raw"] is None:
        st.info("No data available for correlation analysis.")
        return
    technical_cols = ["RSI", "MACD", "WERPI", "OBV", "volume",
                      "Bollinger_Upper", "Bollinger_Lower", "ATR",
                      "Open", "High", "Low", "Close", "Volume"]
    df_raw = st.session_state["df_raw"]
    available_cols = [col for col in technical_cols if col in df_raw.columns]
    if not available_cols:
        st.info("None of the specified columns are available in the data.")
        return
    corr_matrix = df_raw[available_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix of Technical Indicators, Volume & Price")
    st.plotly_chart(fig, use_container_width=True)

def display_cycle_metrics():
    if os.path.exists(CYCLE_METRICS_FILE):
        with open(CYCLE_METRICS_FILE, "r") as f:
            cycle_metrics = yaml.safe_load(f) or []
        if cycle_metrics:
            st.subheader("Cycle Metrics")
            df_cycle = pd.DataFrame(cycle_metrics)
            st.dataframe(df_cycle)
        else:
            st.info("No cycle metrics available yet.")
    else:
        st.info("Cycle metrics file not found.")

# -------------------------------
# Advanced Dashboard Tabs
# -------------------------------
def show_advanced_dashboard_tabs():
    tabs = st.tabs(["WERPI", "Indicators", "Sentiment", "Learning Visualization", "Ensemble Learning"])
    with tabs[0]:
        st.write("## WERPI Indicator")
        ticker_w = st.selectbox("Select Ticker for WERPI", TICKERS, key="werpi_ticker")
        interval_w = st.selectbox("Select Interval", TIMEFRAMES, key="werpi_interval")
        historical_window_werpi = st.number_input("Historical Window (days) for WERPI", min_value=1, value=30)
        today = pd.Timestamp('today')
        default_start = today - pd.Timedelta(days=historical_window_werpi)
        date_range = st.date_input("Select Training Date Range", [default_start, today])
        if len(date_range) != 2:
            st.error("Please select a valid start and end date.")
        else:
            start_date = date_range[0].strftime("%Y-%m-%d")
            end_date = date_range[1].strftime("%Y-%m-%d")
            data_w = load_data(ticker_w, interval=interval_w, start_date=start_date)
            data_w = calculate_indicators(data_w)
        data_w = data_w[(data_w['Date'] >= pd.to_datetime(start_date)) &
                        (data_w['Date'] <= pd.to_datetime(end_date))]
        st.write(f"Training data: {len(data_w)} rows from {start_date} to {end_date}")
        werpi = WERPIIndicator(ticker_w, interval_w)
        if not werpi.load_or_create():
            st.error("Failed to initialize WERPI model.")
        else:
            if "best_werpi_model" in st.session_state:
                werpi.model = st.session_state["best_werpi_model"]
            try:
                werpi_values = werpi.predict(data_w)
            except Exception as e:
                st.warning("WERPI model not fitted. Auto-training now...")
                if werpi.train(data_w, optimize=False):
                    st.success("Auto-trained WERPI model using the selected date range.")
                else:
                    st.error("Auto-training failed.")
                try:
                    werpi_values = werpi.predict(data_w)
                except Exception as e2:
                    st.error(f"Prediction still failed: {str(e2)}")
                    werpi_values = None
            if werpi_values is not None:
                werpi_fig = create_werpi_chart(data_w, werpi_values)
                if werpi_fig:
                    st.plotly_chart(werpi_fig, use_container_width=True)
            try:
                if hasattr(werpi.model, "estimators_"):
                    feature_names = ['return_1d', 'return_5d', 'return_20d',
                                     'volatility_5d', 'volatility_20d',
                                     'volume_change', 'volume_ma_ratio',
                                     'RSI', 'MACD', 'MACD_Hist']
                    fi_fig = plot_feature_importance(werpi.model.feature_importances_, feature_names, model_type="WERPI")
                    if fi_fig:
                        st.plotly_chart(fi_fig, use_container_width=True)
                else:
                    st.info("WERPI model is not sufficiently trained to display feature importances.")
            except Exception as e:
                st.error(f"Error displaying feature importances: {str(e)}")
    with tabs[1]:
        st.write("## RSI / MACD / Volume / Secondary Charts")
        ticker_i = st.selectbox("Ticker for Indicators", TICKERS, key="ind_ticker")
        interval_i = st.selectbox("Interval", TIMEFRAMES, key="ind_interval")
        data_i = load_data(ticker_i, "1y", interval_i)
        data_i = calculate_indicators(data_i)
        sec_fig = create_secondary_charts(data_i)
        if sec_fig:
            st.plotly_chart(sec_fig, use_container_width=True)
    with tabs[2]:
        st.write("## Sentiment")
        ticker_s = st.selectbox("Ticker for Sentiment", TICKERS, key="sent_ticker")
        sentiment_df = get_sentiment_data(ticker_s)
        sent_fig = visualize_sentiment(sentiment_df)
        if sent_fig:
            st.plotly_chart(sent_fig, use_container_width=True)
    with tabs[3]:
        st.write("## Learning Visualization")
        if st.session_state.get('model_weights_history'):
            lv_fig = visualize_learning_progress()
            if lv_fig:
                st.plotly_chart(lv_fig, use_container_width=True)
            anim_fig = create_learning_animation(st.session_state.model_weights_history)
            if anim_fig:
                st.plotly_chart(anim_fig, use_container_width=True)
        else:
            st.write("No training weight history available.")
    with tabs[4]:
        st.write("## Ensemble Learning Visualization")
        ensemble_anim_fig = animate_ensemble_learning()
        if ensemble_anim_fig:
            st.plotly_chart(ensemble_anim_fig, use_container_width=True)
        else:
            st.write("No ensemble prediction data available for animation.")

# -------------------------------
# Model Save/Load Controls
# -------------------------------
def save_current_model():
    model = st.session_state.get('current_model')
    if model is None:
        st.warning("No model in session to save.")
        return
    save_dir = st.session_state['saved_model_dir']
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"model_{timestamp}")
    try:
        model.save(model_path)
        st.success(f"Model saved at: {model_path}")
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

def load_existing_model(model_path: str):
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        st.session_state['current_model'] = loaded_model
        st.session_state['model_loaded'] = True
        st.success(f"Model loaded from: {model_path}")
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {str(e)}")

def model_save_load_controls():
    st.subheader("Model Save/Load Options")
    st.session_state['saved_model_dir'] = st.text_input(
        "Models Directory",
        value=st.session_state.get('saved_model_dir', "saved_models")
    )
    st.session_state['continue_from_old_weights'] = st.checkbox(
        "Continue from old weights if available?",
        value=st.session_state.get('continue_from_old_weights', False)
    )
    if st.button("Save Current Model"):
        save_current_model()
    st.write("### Load a Previously Saved Model")
    load_path = st.text_input("Path to model folder/file", key="load_model_path")
    if st.button("Load Model"):
        if load_path:
            load_existing_model(load_path)
        else:
            st.warning("Please enter a valid model path to load.")

def display_tested_models():
    if os.path.exists(TESTED_MODELS_FILE):
        try:
            with open(TESTED_MODELS_FILE, "r") as f:
                tested_models = yaml.safe_load(f) or []
        except Exception as e:
            st.error(f"Error loading tested models: {e}")
            tested_models = []
    else:
        st.info("Tested models file not found.")
        tested_models = []
    if tested_models:
        st.subheader("Tested Models")
        df_tested = pd.DataFrame(tested_models)
        if "trial_number" in df_tested.columns:
            df_tested.sort_values("trial_number", inplace=True)
        st.dataframe(df_tested)
    else:
        st.info("No tested models available yet.")

# -------------------------------
# Tuning Controls: Start and Stop
# -------------------------------
def start_tuning(ticker, timeframe):
    st.session_state["tuning_in_progress"] = True
    write_tuning_status(True)
    meta_tuning.set_stop_requested(False)  # Reset the stop flag in meta_tuning
    def tuning_thread_inner():
        try:
            meta_tuning.main()  # Contains the tuning cycle loop
        except Exception as e:
            st.session_state.get("error_log", []).append({
                'timestamp': datetime.now(),
                'function': 'tuning_thread',
                'error': str(e)
            })
            logging.error(f"Error in tuning thread: {str(e)}")
        finally:
            st.session_state['tuning_in_progress'] = False
            write_tuning_status(False)
    thread = threading.Thread(target=tuning_thread_inner)
    thread.daemon = True
    st.session_state["tuning_thread"] = thread
    thread.start()
    st.success(f"Tuning started for {ticker} on {timeframe}. Check logs for progress.")

def stop_tuning():
    """Thread-safe way to stop tuning process"""
    # Use central stop flag mechanism from progress_helper
    print("Stop tuning requested - setting stop flag")
    set_stop_requested(True)  # This sets the Event flag
    
    # Also call meta_tuning's version to ensure synchronization
    meta_tuning.set_stop_requested(True)
    
    # Show warning and set timestamp
    st.warning("⚠️ Stop requested! Waiting for current trial to complete... This may take several minutes.")
    st.session_state['stop_request_time'] = datetime.now().strftime("%H:%M:%S")
    
    # Update file-based flag as well
    write_tuning_status(False)

# -------------------------------
# Main Dashboard Entry Point
# -------------------------------
@error_boundary
def main_dashboard():
    if "df_raw" not in st.session_state or st.session_state["df_raw"] is None:
        ticker = TICKER
        timeframe = TIMEFRAMES[0]
        try:
            df_raw = fetch_data(ticker, start=START_DATE, interval=timeframe)
            if df_raw is None:
                st.error(f"No data returned for {ticker} ({timeframe}). Check your network connection.")
                df_raw = pd.DataFrame()
            else:
                df_raw = feature_engineering(df_raw)
            st.session_state["df_raw"] = df_raw
        except Exception as e:
            st.error(f"Error loading data: {e}")
            df_raw = pd.DataFrame()
    else:
        df_raw = st.session_state["df_raw"]
    init_session_state()
    st.title("🤖 AI Price Prediction Model Dashboard")
    
    # Consolidated progress display (only when tuning is active)
    if st.session_state.get('tuning_in_progress', False):
        st.subheader("Tuning Status")
        progress = load_latest_progress()
        current_trial = progress.get("current_trial", 0)
        total_trials = progress.get("total_trials", 0)
        current_rmse = progress.get("current_rmse", None)
        current_mape = progress.get("current_mape", None)
        cycle = progress.get("cycle", 1)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cycle", cycle)
            st.metric("Trial", f"{current_trial}/{total_trials}")
        with col2:
            st.metric("Current RMSE", f"{current_rmse:.4f}" if current_rmse is not None else "N/A")
        with col3:
            best_rmse = st.session_state.get('best_metrics', {}).get('rmse', 0)
            st.metric("Best RMSE", f"{best_rmse:.4f}" if best_rmse else "N/A")
        with col4:
            st.metric("Current MAPE", f"{current_mape:.2f}%" if current_mape is not None else "N/A")
            
        # Add tuning control buttons here (between metrics and progress bar)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Start Tuning", key="start_tuning_progress"):
                set_stop_requested(False)  # Ensure stop flag is reset
                if "tuning_thread" not in st.session_state or not st.session_state["tuning_thread"].is_alive():
                    start_tuning(ticker, timeframe)
                else:
                    st.info("Tuning is already running.")
        with btn_col2:
            if st.button("Stop Tuning", key="stop_tuning_progress"):
                stop_tuning()
            if 'stop_request_time' in st.session_state:
                st.info(f"Stop requested at {st.session_state['stop_request_time']}. Tuning will end after current trial.")
        
        # Progress bar
        if total_trials > 0:
            percent = int((current_trial / total_trials) * 100)
            st.progress(current_trial / total_trials)
            st.markdown(f"### Trial Progress: {current_trial}/{total_trials} ({percent}%)")
    else:
        # When tuning is not active, still show the buttons but in a cleaner format
        st.subheader("Tuning Controls")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Start Tuning", key="start_tuning_main"):
                set_stop_requested(False)  # Ensure stop flag is reset
                if "tuning_thread" not in st.session_state or not st.session_state["tuning_thread"].is_alive():
                    start_tuning(ticker, timeframe)
                else:
                    st.info("Tuning is already running.")
        with btn_col2:
            if st.button("Stop Tuning", key="stop_tuning_main"):
                stop_tuning()
    
    # Synchronize tuning state from file (only once at start)
    file_tuning_status = read_tuning_status()
    if file_tuning_status != st.session_state.get('tuning_in_progress', False):
        st.session_state['tuning_in_progress'] = file_tuning_status

    # Sidebar: Data & Model Settings and Refresh Controls
    ticker_index = 0
    timeframe_index = 0
    if "global_ticker" in st.session_state and st.session_state["global_ticker"] in TICKERS:
        ticker_index = TICKERS.index(st.session_state["global_ticker"])
    if "global_timeframe" in st.session_state and st.session_state["global_timeframe"] in TIMEFRAMES:
        timeframe_index = TIMEFRAMES.index(st.session_state["global_timeframe"])
    ticker = st.sidebar.selectbox("Select Ticker", TICKERS, index=ticker_index, key="global_ticker")
    timeframe = st.sidebar.selectbox("Select Timeframe", TIMEFRAMES, index=timeframe_index, key="global_timeframe")
    df_raw = fetch_data(ticker, start=START_DATE, interval=timeframe)
    validate_data(df_raw)
    st.sidebar.header("Data & Model Settings")
    historical_window = st.sidebar.number_input("Historical Window (days)", min_value=1, value=30)
    forecast_window = st.sidebar.number_input("Forecast Window (days)", min_value=1, value=30)
    st.sidebar.header("Refresh Settings")
    st.session_state['auto_refresh_enabled'] = st.sidebar.checkbox("Auto Refresh", value=True, key="auto_refresh_checkbox")
    progress_refresh_sec = st.sidebar.slider("Progress Refresh (seconds)", min_value=1, max_value=60, value=2, step=1, key="progress_refresh_slider")
    full_refresh_sec = st.sidebar.slider("Full Dashboard Refresh (seconds)", min_value=10, max_value=3600, value=30, step=10, key="full_refresh_slider")
    st.session_state['progress_refresh_sec'] = progress_refresh_sec
    st.session_state['full_refresh_sec'] = full_refresh_sec
    st.sidebar.header("Other Options")
    st.session_state['realtime_update'] = st.sidebar.checkbox("Enable Realtime Update", value=True, key="realtime_update_checkbox")
    st.session_state['cleanup_old_logs'] = st.sidebar.checkbox("Cleanup Old Logs", value=False, key="cleanup_logs_checkbox")
    if st.sidebar.button("Stop Program"):
        st.session_state['stop_program'] = True
        st.warning("Stop program requested.")
    
    # Display unified price chart
    show_unified_chart(df_raw, historical_window, forecast_window)
    show_model_visualization()
    show_predictions_vs_actuals(df_raw, historical_window, forecast_window)
    display_past_ensemble_predictions()
    st.divider()
    st.subheader("Detailed Tuning Logs")
    display_tuning_logs()
    display_cycle_metrics()
    
    # Addon tabs
    st.header("Addons")
    addons_tabs = st.tabs(["Advanced Dashboard", "Model Save/Load", "Tested Models", "Data Preview", "Correlation Heatmap"])
    with addons_tabs[0]:
        show_advanced_dashboard_tabs()
    with addons_tabs[1]:
        model_save_load_controls()
    with addons_tabs[2]:
        display_tested_models()
    with addons_tabs[3]:
        st.subheader("Data Preview")
        if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
            df = st.session_state["df_raw"]
            current_idx = len(df) - 1
            hist_start = max(0, current_idx - historical_window)
            hist_data = df.iloc[hist_start: current_idx+1]
            st.write(f"Showing last {historical_window} days of data.")
            st.dataframe(hist_data)
        else:
            st.info("No data available for preview.")
    with addons_tabs[4]:
        st.subheader("Correlation Heatmap")
        display_feature_correlations()
    
    # Auto-refresh logic: update progress metrics separately from full dashboard refresh
    if st.session_state['auto_refresh_enabled']:
        now = time.time()
        time_since_progress = now - st.session_state.get('last_progress_refresh', 0)
        time_since_full = now - st.session_state.get('last_full_refresh', 0)
        if time_since_progress >= st.session_state['progress_refresh_sec']:
            st.session_state['last_progress_refresh'] = now
            update_progress_display()
        if time_since_full >= st.session_state['full_refresh_sec']:
            st.session_state['last_full_refresh'] = now
            st.experimental_rerun()

if __name__ == "__main__":
    main_dashboard()
