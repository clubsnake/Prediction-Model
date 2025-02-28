# dashboard.py
"""
Streamlit dashboard that integrates:
- Hyperparameter tuning progress & logs,
- Unified price chart showing historical data + past & future predictions,
- Data fetch and basic plots,
- Model visualization (architecture, training history, etc.),
- Past ensemble logs,
- Tuning logs, and more.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yaml
import time
import threading
from datetime import datetime, timedelta
import logging

from config import (
    TICKERS, TIMEFRAMES, START_DATE, RMSE_THRESHOLD, MAPE_THRESHOLD,
    AUTOREFRESH_INTERVAL_SEC, MODEL_TYPES, get_active_feature_names,
    LOOKBACK, BATCH_SIZE, VALIDATION_SPLIT, SHUFFLE,
    WALK_FORWARD_MIN, WALK_FORWARD_MAX, PREDICTION_HORIZON,
    ENABLE_EMAIL_ALERTS
)
from meta_tuning import tune_for_combo, set_stop_requested
from data import fetch_data, validate_data
from features import feature_engineering
from model_visualization import (
    visualize_neural_network,
    plot_training_history,
    plot_feature_importance,
    plot_prediction_errors,
    plot_ensemble_contribution
)
from visualization import (
    visualize_training_history, 
    visualize_weight_histograms,
    visualize_predictions
)
from preprocessing import create_sequences, scale_data
from utils import validate_dataframe
from model import (
    build_model_by_type,
    evaluate_predictions
)

# If you have a progress_helper, import it, e.g.:
# from progress_helper import update_progress_in_yaml

def init_session_state():
    """
    Initialize necessary session state variables without
    removing anything that might already be set.
    """
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True

        st.session_state.setdefault('tuning_in_progress', False)
        st.session_state.setdefault('best_metrics', {})
        st.session_state.setdefault('model_history', [])
        st.session_state.setdefault('last_refresh', time.time())
        st.session_state.setdefault('prediction_history', [])
        st.session_state.setdefault('accuracy_metrics_history', {})
        st.session_state.setdefault('neural_network_state', {})
        st.session_state.setdefault('current_model', None)
        st.session_state.setdefault('model_loaded', False)
        st.session_state.setdefault('error_log', [])
        st.session_state.setdefault('training_history', None)
        # For storing day-by-day or incremental ensemble logs:
        st.session_state.setdefault('ensemble_predictions_log', [])

def error_boundary(func):
    """
    Decorator that logs and displays errors gracefully in Streamlit.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(f"Error in {func.__name__}: {str(e)}")
            if 'error_log' in st.session_state:
                st.session_state.error_log.append({
                    'timestamp': datetime.now(),
                    'function': func.__name__,
                    'error': str(e)
                })
            return None
    return wrapper

def safe_load_progress() -> dict:
    """
    Safely load 'progress.yaml' to display in UI. 
    This file is updated by meta_tuning after each trial if you call update_progress_in_yaml().
    """
    try:
        with open("progress.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except:
        return {}

def display_metrics_and_progress():
    """
    Display best metrics and progress bar from progress.yaml,
    as well as any 'current_rmse' or 'current_mape' that meta_tuning might store.
    """
    progress = safe_load_progress()
    current_trial = progress.get("current_trial", 0)
    total_trials  = progress.get("total_trials", 0)
    trial_progress = progress.get("trial_progress", 0.0)
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)

    best_rmse = st.session_state['best_metrics'].get('rmse', 0)
    best_mape = st.session_state['best_metrics'].get('mape', 0)
    cycle = progress.get("cycle", None)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if cycle is not None:
            st.metric("Cycle", cycle)
        st.metric("Trial", f"{current_trial}/{total_trials}")
    with col2:
        if current_rmse is not None:
            st.metric("Current RMSE", f"{current_rmse:.4f}")
        else:
            st.metric("Current RMSE", "N/A")
    with col3:
        st.metric("Best RMSE", f"{best_rmse:.4f}")
    with col4:
        if current_mape is not None:
            st.metric("Current MAPE", f"{current_mape:.2f}%")
        else:
            st.metric("Current MAPE", "N/A")

    if trial_progress > 0:
        st.progress(trial_progress)

def display_tuning_logs():
    """
    A placeholder: show more detailed logs of each trial, e.g. trial #, 
    hyperparameters, best score so far, etc.
    You could read from a file or session_state if your objective logs them.
    """
    st.info("Detailed Tuning Logs are not yet implemented in this snippet. "
            "If you store each trial's hyperparams, metrics, etc. in a file "
            "or session_state, you can display them here in a table or chart.")

def record_ensemble_prediction(date, actual, predicted, rmse=None, mape=None):
    """
    Append a single ensemble prediction record. 
    'date' can be a datetime, 'actual'/'predicted' are floats, etc.
    """
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

def display_past_ensemble_predictions():
    """
    Show a plot of all past ensemble predictions vs. actual from
    st.session_state['ensemble_predictions_log'].
    """
    if not st.session_state['ensemble_predictions_log']:
        st.info("No past ensemble predictions logged yet.")
        return
    
    df_log = pd.DataFrame(st.session_state['ensemble_predictions_log'])
    df_log.sort_values('date', inplace=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_log['date'],
        y=df_log['actual'],
        mode='lines+markers',
        name='Actual Price'
    ))
    fig.add_trace(go.Scatter(
        x=df_log['date'],
        y=df_log['predicted'],
        mode='lines+markers',
        name='Predicted Price'
    ))
    fig.update_layout(
        title="Past Ensemble Predictions vs. Actual",
        xaxis_title="Date",
        yaxis_title="Price (Close)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Ensemble Prediction Log")
    st.dataframe(df_log)

def show_unified_chart(df, historical_window, forecast_window):
    """
    Displays:
      - A candlestick for the last 'historical_window' days,
      - Past predictions from st.session_state['ensemble_predictions_log'] 
        if they fall within or before that date range,
      - Future predictions (the next 'forecast_window' days) if we have them
        in st.session_state (e.g. st.session_state['future_forecast']) or similar.

    Also includes a place to do "data fetch and basic plots" underneath.
    """
    if df.empty:
        st.warning("DataFrame is empty; cannot display chart.")
        return

    # 1) Subset data for the last 'historical_window' days
    current_idx = len(df) - 1
    hist_start = max(0, current_idx - historical_window)
    hist_data = df.iloc[hist_start:current_idx+1]

    fig = go.Figure()

    # 2) Candlestick for the historical portion
    fig.add_trace(go.Candlestick(
        x=hist_data['date'],
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Historical'
    ))

    # 3) Past predictions from ensemble_predictions_log
    if st.session_state['ensemble_predictions_log']:
        # We'll show any predictions whose 'date' <= the last date in hist_data
        # so they appear in the same timeframe. If you want older predictions,
        # you can just include them as well.
        df_log = pd.DataFrame(st.session_state['ensemble_predictions_log'])
        # Filter for those up to the last date of hist_data
        last_date_in_hist = hist_data['date'].iloc[-1]
        # We'll show all that are <= last_date_in_hist to see how they compare
        df_past = df_log[df_log['date'] <= last_date_in_hist].copy()

        if not df_past.empty:
            fig.add_trace(go.Scatter(
                x=df_past['date'],
                y=df_past['predicted'],
                mode='markers',
                name='Past Predictions',
                marker=dict(color='blue', symbol='circle')
            ))

    # 4) Future predictions for the next 'forecast_window' days
    # If you have something like st.session_state['future_forecast'] or we do a quick mock
    # For demonstration, let's see if we have an array of predicted future prices
    # st.session_state['future_forecast'] might be e.g. an np.array of length forecast_window
    if 'future_forecast' in st.session_state:
        future_array = st.session_state['future_forecast']
        if isinstance(future_array, np.ndarray) and future_array.size == forecast_window:
            # We'll create new dates for the future: daily freq from last_date_in_hist+1
            future_dates = pd.date_range(start=hist_data['date'].iloc[-1] + timedelta(days=1),
                                         periods=forecast_window, freq='D')
            # Plot these
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_array,
                mode='lines+markers',
                name='Future Predictions',
                line=dict(dash='dot', color='red')
            ))

    fig.update_layout(
        title="Price Chart (Past & Future Predictions)",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5) Data fetch and basic plots "underneath that"
    st.subheader("Basic Plots / Data Preview")
    st.write(f"Showing last {historical_window} days of historical data plus {forecast_window} days of potential future predictions (if any).")
    st.dataframe(hist_data)
    # Possibly more basic plots below if you want.

def show_model_visualization():
    """
    Display your existing model in a user-friendly manner:
    architecture, training history, weight histograms, etc.
    """
    st.subheader("Model Architecture & Training History")
    model = st.session_state.get('current_model')
    if model is None:
        st.info("No model found in session_state['current_model']. Please train or load a model first.")
        return

    # 1) Architecture
    feature_names = get_active_feature_names()
    arch_fig = visualize_neural_network(model, feature_names)
    st.pyplot(arch_fig)

    # 2) If we have training history in session_state
    history_data = st.session_state.get('training_history', None)
    if isinstance(history_data, dict):
        st.subheader("Training History (Plotly)")
        fig_hist = plot_training_history(history_data)
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True)

    # 3) Weight histograms (optional)
    if st.checkbox("Show Weight Histograms"):
        st.write("Generating weight histograms from 'visualization.py' or similar code.")
        visualize_weight_histograms(model)

def show_predictions_vs_actuals(df):
    """
    Example function to display predictions vs actual using 
    code from model_visualization or visualization modules.
    """
    st.subheader("Prediction Errors & Analysis")
    model = st.session_state.get('current_model')
    if not model:
        st.info("No model loaded. Please load or train one.")
        return

    feature_cols = get_active_feature_names()
    scaled_df, _, _ = scale_data(df, feature_cols, "Close")
    X, y = create_sequences(scaled_df, feature_cols, "Close", lookback=LOOKBACK, horizon=7)
    if len(X) == 0:
        st.warning("Not enough data to create sequences.")
        return
    
    preds = model.predict(X)
    if preds.shape == y.shape:
        st.subheader("Error Plot (Plotly)")
        fig_err = plot_prediction_errors(y.flatten(), preds.flatten())
        if fig_err:
            st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.info("Prediction shape doesn't match y shape; can't do a direct error plot.")

@error_boundary
def main_dashboard():
    st.set_page_config(page_title="AI Price Prediction Dashboard", layout="wide")
    init_session_state()
    
    st.title("🤖 AI Price Prediction Model Dashboard")

    # Sidebar controls
    st.sidebar.header("Data & Model Settings")
    ticker = st.sidebar.selectbox("Select Ticker", TICKERS)
    timeframe = st.sidebar.selectbox("Select Timeframe", TIMEFRAMES)

    historical_window = st.sidebar.slider(
        "Historical Window (days)", 
        min_value=WALK_FORWARD_MIN, 
        max_value=WALK_FORWARD_MAX, 
        value=30
    )
    forecast_window = st.sidebar.slider(
        "Forecast Window (days)",
        min_value=1,
        max_value=PREDICTION_HORIZON,
        value=7
    )

    st.sidebar.header("Hyperparameter Tuning")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start Tuning"):
            if not st.session_state.tuning_in_progress:
                st.session_state.tuning_in_progress = True
                def _tune():
                    tune_for_combo(ticker, timeframe, "all")  # e.g. horizon="all"
                    st.session_state.tuning_in_progress = False
                threading.Thread(target=_tune, daemon=True).start()
    with col2:
        if st.button("Stop Tuning"):
            set_stop_requested(True)
            st.session_state.tuning_in_progress = False

    # Manual refresh
    if st.sidebar.button("Manual Refresh"):
        st.session_state.last_refresh = time.time()

   # Auto-refresh interval
    update_interval = st.sidebar.number_input(
        "Update Interval (seconds)",
        min_value=1,
        value=AUTOREFRESH_INTERVAL_SEC
    )

    # Tabs
    tab_prices, tab_tuning, tab_model_viz, tab_past_ensemble = st.tabs(
        ["Price & Predictions", "Tuning Status", "Model Visualization", "Past Ensemble"]
    )

    # -----------------------------------------------------------------------
    # 1) PRICE & PREDICTIONS TAB
    # -----------------------------------------------------------------------
    with tab_prices:
        st.header("Price Chart: Historical + Past & Future Predictions")

        # Fetch data & validate
        df = fetch_data(ticker, START_DATE, interval=timeframe)
        if df is None or not validate_data(df):
            st.warning("Data not available or insufficient. Check logs or yfinance settings.")
            return
        df = feature_engineering(df, ticker=ticker)

        # Possibly do a daily ensemble update every update_interval
        current_time = datetime.now()
        if (current_time - datetime.fromtimestamp(st.session_state.last_refresh)).total_seconds() >= update_interval:
            # Example: store a random "future forecast" array in session_state for demonstration
            future_arr = np.linspace(df["Close"].iloc[-1], df["Close"].iloc[-1]*1.05, forecast_window)
            st.session_state['future_forecast'] = future_arr

            # If you want to record a day-by-day ensemble prediction:
            last_close = df["Close"].iloc[-1]
            predicted_price = float(last_close * (1 + np.random.normal(0, 0.01)))
            record_ensemble_prediction(
                date=df["date"].iloc[-1],
                actual=last_close,
                predicted=predicted_price,
                rmse=np.random.uniform(0.01, 0.1),
                mape=np.random.uniform(1,5)
            )
            st.session_state.last_refresh = time.time()

        # Show a single chart that merges historical data, past predictions, future predictions
        show_unified_chart(df, historical_window, forecast_window)

        st.success("Above is the unified chart. Below, we can do additional analysis...")

        # Basic model-based predictions vs actual if you want
        show_predictions_vs_actuals(df)

    # -----------------------------------------------------------------------
    # 2) TUNING STATUS TAB
    # -----------------------------------------------------------------------
    with tab_tuning:
        st.header("Hyperparameter Tuning Progress")
        display_metrics_and_progress()
        st.subheader("Detailed Tuning Logs")
        display_tuning_logs()

    # -----------------------------------------------------------------------
    # 3) MODEL VISUALIZATION TAB
    # -----------------------------------------------------------------------
    with tab_model_viz:
        st.header("Model Visualization, Architecture, & Training History")
        show_model_visualization()

    # -----------------------------------------------------------------------
    # 4) PAST ENSEMBLE TAB
    # -----------------------------------------------------------------------
    with tab_past_ensemble:
        st.header("Past Ensemble Predictions (Daily Logs)")
        display_past_ensemble_predictions()

    # Possibly auto-refresh
    if st.sidebar.checkbox("Enable Auto-refresh", value=True):
        if time.time() - st.session_state.last_refresh > AUTOREFRESH_INTERVAL_SEC:
            st.rerun()

def create_mini_monitor():
    """
    Example minimal monitor for a separate page if desired.
    """
    st.title("Mini Monitor Page")
    st.write("This is a placeholder for a minimal real-time monitor.")
    if "ensemble_predictions_log" in st.session_state:
        df_log = pd.DataFrame(st.session_state["ensemble_predictions_log"])
        st.write(df_log)

if __name__ == "__main__":
    main_dashboard()
