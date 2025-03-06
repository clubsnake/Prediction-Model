"""
dashboard_data.py

Functions for loading and processing data for the dashboard.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import os
import sys

# Add project root to sys.path if not already there
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import with error handling
try:
    from dashboard_error import robust_error_boundary
    from config.logger_config import logger
except ImportError as e:
    # Set up basic logger if import fails
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dashboard")
    logger.warning(f"Error importing modules: {e}")
    
    # Define error boundary if not available
    def robust_error_boundary(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Error in {func.__name__}: {str(e)}")
                return None
        return wrapper


# Check if newer Streamlit caching functions are available, otherwise use legacy cache
if hasattr(st, "cache_data"):
    streamlit_cache = st.cache_data(ttl=600)  # Cache for 10 minutes
else:
    # Fall back to legacy cache for older Streamlit versions
    streamlit_cache = st.cache(ttl=600, allow_output_mutation=True)


@robust_error_boundary
@streamlit_cache  # Using our compatible cache decorator
def load_data(ticker, start_date, end_date=None, interval="1d", training_mode=False):
    """Load market data with caching and robust error handling"""
    try:
        # Attempt to load data from the API
        df = fetch_data_from_api(
            ticker, start_date, end_date, interval
        )

        if df is None or df.empty:
            logging.warning(
                f"No data received from API for {ticker} between {start_date} and {end_date}"
            )
            return None
            
        # Ensure the date column is properly handled
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'Date' in df.columns and 'date' not in df.columns:
                df = df.rename(columns={'Date': 'date'})
            elif 'index' in df.columns and 'date' not in df.columns:
                df = df.rename(columns={'index': 'date'})
                
        # Ensure date column is datetime type
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df

    except Exception as e:
        logging.error(f"Error loading data from API for {ticker}: {e}", exc_info=True)
        # Implement fallback mechanism (e.g., load from local backup)
        df = load_data_from_backup(
            ticker, start_date, end_date, interval
        )

        if df is None or df.empty:
            logging.critical(f"Failed to load data from API and backup for {ticker}!")
            st.error(f"Failed to load data for {ticker}. Please check the logs.")
            return None

        logging.warning(f"Successfully loaded data from backup for {ticker}.")
        return df


def fetch_data_from_api(ticker, start_date, end_date, interval):
    """Fetch data from yfinance API with robust error handling"""
    try:
        import yfinance as yf

        # Show a download message in the UI
        with st.spinner(f"Downloading data for {ticker}..."):
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data is not None and not data.empty:
            # Consistently add date column
            data = data.reset_index()
            # Normalize column name to lowercase 'date'
            if 'Date' in data.columns and 'date' not in data.columns:
                data = data.rename(columns={'Date': 'date'})
            return data
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching data from yfinance: {e}", exc_info=True)
        return None


def load_data_from_backup(ticker, start_date, end_date, interval):
    """Load data from backup source with progress indication"""
    try:
        # Try multiple backup locations
        backup_paths = [
            f"data/{ticker}_{start_date}_{end_date}_{interval}.csv",
            f"Data/{ticker}_{interval}.csv",
            f"Data/Backup/{ticker}.csv"
        ]
        
        for filepath in backup_paths:
            if os.path.exists(filepath):
                with st.spinner(f"Loading backup data from {filepath}..."):
                    df = pd.read_csv(filepath)
                    
                    # Ensure date column is properly formatted
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    elif 'Date' in df.columns:
                        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df = df.drop(columns=['Date'])
                        
                    return df
        
        # If we've checked all paths and found nothing
        st.warning(f"No backup data found for {ticker}")
        return None
    except Exception as e:
        logging.error(f"Error loading data from backup file: {e}", exc_info=True)
        return None


@robust_error_boundary
def calculate_indicators(data):
    """Calculate technical indicators for the data"""
    if data is None or data.empty:
        return data

    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure there's a Date column with consistent name
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "date" not in df.columns:
            # If neither exists, create from index if possible
            if isinstance(df.index, pd.DatetimeIndex):
                df["date"] = df.index
            else:
                # Last resort: create a dummy date column
                df["date"] = pd.date_range(start="2020-01-01", periods=len(df))

        # Calculate moving averages
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["MA_50"] = df["Close"].rolling(window=50).mean()
        df["MA_200"] = df["Close"].rolling(window=200).mean()

        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Std"] = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
        df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Volume indicators - fixed to avoid DataFrame to Series assignment
        if "Volume" in df.columns:
            df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
            
            # Fix for the Volume_Ratio calculation
            # Calculate as a Series operation, not DataFrame
            vol_ratio = df["Volume"] / df["Volume_MA_20"]
            df["Volume_Ratio"] = vol_ratio

        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        # Return the original data if calculation fails
        return data


@robust_error_boundary
def update_progress_display():
    """Update just the progress display without a full dashboard rerun"""
    from dashboard_error import load_latest_progress
    
    # Create dedicated containers if not already set
    if "progress_container" not in st.session_state:
        st.session_state["progress_container"] = st.empty()
        st.session_state["progress_bar"] = st.empty()
        st.session_state["metrics_container"] = st.empty()

    progress = load_latest_progress()
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get("total_trials", 0)
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)

    # Update progress bar and container
    if total_trials > 0:
        percent = int((current_trial / total_trials) * 100)
        st.session_state["progress_bar"].progress(current_trial / total_trials)
        st.session_state["progress_container"].markdown(
            f"### Trial Progress: {current_trial}/{total_trials} ({percent}%)"
        )

    with st.session_state["metrics_container"].container():
        cols = st.columns(3)
        with cols[0]:
            st.metric("Current Trial", current_trial)
        with cols[1]:
            st.metric(
                "Current RMSE",
                f"{current_rmse:.4f}" if current_rmse is not None else "N/A",
            )
        with cols[2]:
            st.metric(
                "Current MAPE",
                f"{current_mape:.2f}%" if current_mape is not None else "N/A",
            )
            
            
def get_stock_sentiment(ticker, days_back=7):
    """Get stock sentiment from news articles (placeholder implementation)"""
    try:
        # In a real implementation, this would call a sentiment analysis API
        # For now, we'll return random values as a placeholder
        import random
        sentiment_values = [random.uniform(-1, 1) for _ in range(days_back)]
        dates = pd.date_range(end=datetime.now(), periods=days_back)
        
        sentiment_df = pd.DataFrame({
            'date': dates,
            'sentiment': sentiment_values
        })
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiment_values)
        sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
        
        return {
            'data': sentiment_df,
            'average': avg_sentiment,
            'label': sentiment_label
        }
    except Exception as e:
        logger.error(f"Error getting sentiment data: {e}")
        return None