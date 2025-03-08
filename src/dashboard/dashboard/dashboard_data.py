"""
dashboard_data.py

Functions for loading and processing data for the dashboard.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import os
import sys
end_date = datetime.now()

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
    from src.dashboard.dashboard.dashboard_error import robust_error_boundary
    from config.logger_config import logger
    
    # Try to import sequence_utils if available
    try:
        from src.data.sequence_utils import numba_mse, numba_mape, vectorized_sequence_creation
    except ImportError:
        logger.warning("Could not import sequence_utils. Some optimizations will be unavailable.")
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


def ensure_date_column(df, default_name='date'):
    """
    Ensure the dataframe has a proper datetime date column.
    
    Args:
        df: DataFrame to process
        default_name: Default column name to use for the date column
        
    Returns:
        tuple: (DataFrame with guaranteed date column, name of date column)
    """
    # Handle invalid dataframe cases
    if df is None:
        return None, default_name
    
    # Check if we received a tuple instead of a DataFrame
    if isinstance(df, tuple):
        logger.error(f"Received tuple instead of DataFrame: {df}")
        return None, default_name
        
    if hasattr(df, 'empty') and df.empty:
        return df, default_name
    
    df = df.copy()  # Don't modify the original
    
    # Safely process column names
    date_col = None
    date_cols = []
    for col in df.columns:
        if isinstance(col, str) and 'date' in col.lower():
            date_cols.append(col)
        elif not isinstance(col, str):
            logger.warning(f"Non-string column name encountered: {type(col)}")
    
    # Try to find an existing date column
    if 'date' in df.columns:
        date_col = 'date'
    elif 'Date' in df.columns:
        date_col = 'Date'
    elif date_cols:
        date_col = date_cols[0]
    
    # If no date column but index is datetime, convert index to column
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        df[default_name] = df.index
        date_col = default_name
    
    # If still no date column, create a synthetic one
    if date_col is None:
        df[default_name] = pd.date_range(start=datetime.now() - timedelta(days=len(df)), periods=len(df))
        date_col = default_name
    
    # Ensure date column is datetime type
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.warning(f"Error converting {date_col} to datetime: {e}")
        # If conversion fails, create a new date column
        df[default_name] = pd.date_range(start=datetime.now() - timedelta(days=len(df)), periods=len(df))
        date_col = default_name
    
    return df, date_col


@robust_error_boundary
@streamlit_cache  # Using our compatible cache decorator
def load_data(ticker, start_date, end_date=None, interval="1d", training_mode=False):
    """Load market data with caching and robust error handling"""
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str) and end_date is not None:
            end_date = pd.to_datetime(end_date).date()

        # Handle future dates - use current date as the end date for API call
        current_date = datetime.now().date()
        if end_date and end_date > current_date:
            logger.info(f"End date {end_date} is in the future. Using current date {current_date} for API call.")
            api_end_date = current_date
        else:
            api_end_date = end_date

        # Attempt to load data from the API using adjusted end date
        df = fetch_data_from_api(
            ticker, start_date, api_end_date, interval
        )

        if df is None or df.empty:
            logging.warning(
                f"No data received from API for {ticker} between {start_date} and {api_end_date}"
            )
            return None
            
        # Ensure the date column is properly handled
        df, _ = ensure_date_column(df)

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

        # Ensure start_date and end_date are strings (yfinance expects string dates)
        if not isinstance(start_date, str):
            start_date = start_date.strftime('%Y-%m-%d')
        
        if end_date is not None and not isinstance(end_date, str):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # If end_date is None, use current date
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Show a download message in the UI
        with st.spinner(f"Downloading data for {ticker}..."):
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data is not None and not data.empty:
            # Consistently add date column
            data, _ = ensure_date_column(data)
            return data
        else:
            logger.warning(f"No data received from API for {ticker} between {start_date} and {end_date}")
            return None
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}", exc_info=True)
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
                    df, _ = ensure_date_column(df)
                        
                    return df
        
        # If we've checked all paths and found nothing
        st.warning(f"No backup data found for {ticker}")
        return None
    except Exception as e:
        logging.error(f"Error loading data from backup file: {e}")
        return None


@robust_error_boundary
def calculate_indicators(df):
    """
    Calculate technical indicators for the given dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    if df is None or df.empty:
        return df
        
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns for indicators. Have: {df.columns.tolist()}")
        return df
        
    try:
        # Simple Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # Volume indicators - only if Volume column exists
        if 'Volume' in df.columns:
            try:
                # Volume Moving Average
                df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
                
                # Fix the Volume_Ratio calculation:
                # 1. Extract Volume as Series to ensure it's not a DataFrame
                volume_series = df['Volume'].astype(float)
                # 2. Extract Volume_MA as Series
                volume_ma_series = df['Volume_MA'].replace(0, 0.001).astype(float)
                # 3. Perform element-wise division between Series
                df['Volume_Ratio'] = volume_series.div(volume_ma_series)
                
            except Exception as e:
                logger.error(f"Error calculating volume indicators: {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        
    return df


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


# Add this function to handle forecast generation in the dashboard
@robust_error_boundary
def generate_dashboard_forecast(model, df, feature_cols):
    """
    Generate forecast for the dashboard using available model.
    Abstracts the forecast generation to handle different sources.
    
    Args:
        model: Model to use for forecasting
        df: DataFrame with historical data
        feature_cols: List of feature column names
        
    Returns:
        List of forecast values
    """
    try:
        # Try multiple approaches to generate forecast
        
        # First try walk_forward method (preferred)
        try:
            from src.training.walk_forward import generate_future_forecast
            
            # Get settings from session state
            import streamlit as st
            lookback = st.session_state.get("lookback", 30)
            forecast_window = st.session_state.get("forecast_window", 30)
            
            # Generate and return forecast
            forecast = generate_future_forecast(model, df, feature_cols, lookback, forecast_window)
            return forecast
        except ImportError:
            logger.info("Could not import generate_future_forecast from walk_forward")
        
        # Try prediction service approach
        try:
            from src.dashboard.prediction_service import PredictionService
            
            service = PredictionService(model_instance=model)
            import streamlit as st
            lookback = st.session_state.get("lookback", 30)
            forecast_window = st.session_state.get("forecast_window", 30)
            
            forecast = service.generate_forecast(df, feature_cols, lookback, forecast_window)
            return forecast
        except ImportError:
            logger.info("Could not import PredictionService")
        
        # Last resort - basic forecast with direct model calls
        import streamlit as st
        from sklearn.preprocessing import StandardScaler
        
        lookback = st.session_state.get("lookback", 30)
        forecast_window = st.session_state.get("forecast_window", 30)
        
        # Get the last 'lookback' days of data for input
        last_data = df.iloc[-lookback:].copy()
        
        # Create a scaler for feature normalization
        scaler = StandardScaler()
        scaler.fit(last_data[feature_cols])
        
        # Initialize array to store predictions
        future_prices = []
        current_data = last_data.copy()
        
        # Try to get create_sequences from preprocessing
        try:
            from src.data.preprocessing import create_sequences
        except ImportError:
            # Define a minimal version if not available
            def create_sequences(data, features, target, window, horizon):
                import numpy as np
                X = []
                for i in range(len(data) - window):
                    X.append(data[features].values[i:i+window])
                return np.array(X), None
        
        # Create initial input sequence
        X_input, _ = create_sequences(current_data, feature_cols, "Close", lookback, 1)
        
        # Generate forecasts one step at a time
        for i in range(forecast_window):
            try:
                preds = model.predict(X_input, verbose=0)
                next_price = float(preds[0][0])
                future_prices.append(next_price)
                
                # Update input data with prediction
                next_row = current_data.iloc[-1:].copy()
                if isinstance(next_row.index[0], pd.Timestamp):
                    next_row.index = [next_row.index[0] + pd.Timedelta(days=1)]
                else:
                    next_row.index = [next_row.index[0] + 1]
                
                next_row["Close"] = next_price
                current_data = pd.concat([current_data.iloc[1:], next_row])
                
                # Create new sequence for next prediction
                X_input, _ = create_sequences(current_data, feature_cols, "Close", lookback, 1)
            except Exception as e:
                logger.error(f"Error in forecast iteration {i}: {e}")
                break
        
        return future_prices
    
    except Exception as e:
        logger.error(f"Error generating dashboard forecast: {e}")
        return None