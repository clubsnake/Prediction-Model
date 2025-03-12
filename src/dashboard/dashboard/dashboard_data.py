"""
dashboard_data.py

Functions for loading and processing data for the dashboard.
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

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
    from config.logger_config import logger
    from src.dashboard.dashboard.dashboard_error import robust_error_boundary

    # Try to import sequence_utils if available
    try:
        from src.data.sequence_utils import (
            numba_mape,
            numba_mse,
            vectorized_sequence_creation,
        )
    except ImportError:
        logger.warning(
            "Could not import sequence_utils. Some optimizations will be unavailable."
        )

    # Import custom indicators - add explicit imports for feature modules
    try:
        # Import VMLI indicator
        # Import feature selection utilities if needed
        from src.features.feature_selection import (
            FeatureSelector,
            select_optimal_features,
        )

        # Import WERPI and other feature functions
        from src.features.features import (
            add_werpi_indicator,
            feature_engineering_with_params,
        )
        from src.features.vmli_indicator import VMILIndicator

        logger.info("Successfully imported custom indicators and feature modules")
    except ImportError as e:
        logger.warning(f"Could not import custom indicator modules: {e}")

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


def ensure_date_column(df, default_name="date"):
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

    if hasattr(df, "empty") and df.empty:
        return df, default_name

    df = df.copy()  # Don't modify the original

    # Handle MultiIndex columns by flattening them
    if hasattr(df, "columns") and isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in df.columns
        ]
        logger.info("Flattened MultiIndex columns")
    elif hasattr(df, "columns"):
        # Handle individual tuple columns
        df.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in df.columns
        ]

    # Safely process column names
    date_col = None
    date_cols = []
    for col in df.columns:
        if isinstance(col, str) and "date" in col.lower():
            date_cols.append(col)
        elif not isinstance(col, str):
            logger.warning(f"Non-string column name encountered: {type(col)}")

    # Try to find an existing date column
    if "date" in df.columns:
        date_col = "date"
    elif "Date" in df.columns:
        date_col = "Date"
    elif date_cols:
        date_col = date_cols[0]

    # If no date column but index is datetime, convert index to column
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        df[default_name] = df.index
        date_col = default_name

    # If still no date column, create a synthetic one
    if date_col is None:
        df[default_name] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)), periods=len(df)
        )
        date_col = default_name

    # Ensure date column is datetime type
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.warning(f"Error converting {date_col} to datetime: {e}")
        # If conversion fails, create a new date column
        df[default_name] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)), periods=len(df)
        )
        date_col = default_name

    return df, date_col


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
    required_cols = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required_cols):
        logger.warning(
            f"Missing required columns for indicators. Have: {df.columns.tolist()}"
        )
        return df

    try:
        # Simple Moving Averages
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()

        # Exponential Moving Averages
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_middle"] = df["Close"].rolling(window=20).mean()
        df["BB_std"] = df["Close"].rolling(window=20).std()
        df["BB_upper"] = df["BB_middle"] + 2 * df["BB_std"]
        df["BB_lower"] = df["BB_middle"] - df["BB_std"]

        # Volume indicators - only if Volume column exists
        if "Volume" in df.columns:
            try:
                # Volume Moving Average
                df["Volume_MA"] = df["Volume"].rolling(window=20).mean()

                # Create Volume_Ratio more safely
                df["Volume_Ratio"] = np.nan  # Initialize with NaN

                # Create boolean mask for valid values (as Series)
                mask = (
                    ~df["Volume"].isna()
                    & ~df["Volume_MA"].isna()
                    & (df["Volume_MA"] > 0)
                )

                # Only calculate for valid values using loc
                if mask.any():
                    df.loc[mask, "Volume_Ratio"] = (
                        df.loc[mask, "Volume"] / df.loc[mask, "Volume_MA"]
                    )

                # Handle any infinity values
                df["Volume_Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

            except Exception as e:
                logger.error(f"Error calculating volume indicators: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")

    return df


@robust_error_boundary
@streamlit_cache  # Using our compatible cache decorator
def load_data(ticker, start_date, end_date=None, interval="1d", training_mode=False):
    """Load market data with caching and robust error handling from multiple sources"""
    try:
        # Determine the appropriate data source based on ticker and settings
        data_source = st.session_state.get("data_source", "auto")

        # Auto-detect source based on ticker
        if data_source == "auto":
            if ticker.endswith("-USD") or ticker.endswith("USDT"):
                data_source = "yfinance"  # Use Yahoo Finance for crypto
            else:
                data_source = "yfinance"  # Default to Yahoo Finance for stocks

        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

        if isinstance(end_date, str) and end_date is not None:
            end_date = pd.to_datetime(end_date)

        # Handle future dates - use current date as the end date for API call
        current_date = datetime.now().date()
        if end_date and pd.to_datetime(end_date).date() > current_date:
            api_end_date = current_date
        else:
            api_end_date = end_date

        # Log data fetch attempt
        logger.info(
            f"Fetching {ticker} data from {start_date} to {api_end_date} using {data_source}"
        )

        # Attempt to load data from the selected API
        df = None
        if data_source == "yfinance":
            # Use Yahoo Finance as the primary data source

            df = fetch_data_from_yfinance(ticker, start_date, api_end_date, interval)
        elif data_source == "coingecko":
            # Use CoinGecko for cryptocurrency data
            df = fetch_data_from_coingecko(ticker, start_date, api_end_date, interval)
        elif data_source == "alphavantage":
            # Use Alpha Vantage for stock data
            df = fetch_data_from_alphavantage(
                ticker, start_date, api_end_date, interval
            )
        else:
            # Default to Yahoo Finance

            df = fetch_data_from_yfinance(ticker, start_date, api_end_date, interval)

        if df is None or df.empty:
            logger.warning(
                f"No data received for {ticker}. Trying backup data sources..."
            )
            df = load_data_from_backup(ticker, start_date, end_date, interval)

        # Ensure the date column is properly handled
        if df is not None and not df.empty:
            df, date_col = ensure_date_column(df)

            # Apply custom indicators if in session state
            if "indicators" in st.session_state:
                df = apply_custom_indicators(
                    df, timeframe=interval, indicators=st.session_state["indicators"]
                )

            # Store in session state for other components to use
            st.session_state[f"{ticker}_{interval}_data"] = df

        return df

    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {e}", exc_info=True)
        # Try to load from backup
        df = load_data_from_backup(ticker, start_date, end_date, interval)

        if df is None or df.empty:
            logger.error(f"Failed to load data for {ticker} from all sources")
            return None

        logger.info(f"Successfully loaded data from backup for {ticker}")
        return df


def fetch_data_from_yfinance(ticker, start_date, end_date, interval):
    """Fetch data from yfinance API with robust error handling"""
    try:
        import yfinance as yf

        # Format dates for yfinance
        if not isinstance(start_date, str):
            start_date = start_date.strftime("%Y-%m-%d")

        if end_date is not None and not isinstance(end_date, str):
            end_date = end_date.strftime("%Y-%m-%d")

        # If end_date is None, use current date
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Show a download message in the UI
        with st.spinner(f"Downloading data from Yahoo Finance for {ticker}..."):
            data = yf.download(
                ticker, start=start_date, end=end_date, interval=interval
            )

        if data is not None and not data.empty:
            # Consistently add date column
            data, _ = ensure_date_column(data)
            return data
        else:
            logger.warning(
                f"No data received from Yahoo Finance for {ticker} between {start_date} and {end_date}"
            )
            return None
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}", exc_info=True)
        return None


def fetch_data_from_coingecko(ticker, start_date, end_date, interval):
    """Fetch cryptocurrency data from CoinGecko API"""
    try:
        from pycoingecko import CoinGeckoAPI

        # Format ticker for CoinGecko (remove -USD suffix)
        coin_id = ticker.replace("-USD", "").replace("USDT", "").lower()

        # Convert start_date and end_date to unix timestamps
        if not isinstance(start_date, str):
            start_timestamp = int(
                datetime.combine(start_date, datetime.min.time()).timestamp()
            )
        else:
            start_timestamp = int(pd.to_datetime(start_date).timestamp())

        if end_date is None:
            end_timestamp = int(datetime.now().timestamp())
        elif not isinstance(end_date, str):
            end_timestamp = int(
                datetime.combine(end_date, datetime.min.time()).timestamp()
            )
        else:
            end_timestamp = int(pd.to_datetime(end_date).timestamp())

        # Initialize CoinGecko API
        cg = CoinGeckoAPI()

        # Convert interval to days
        if interval.endswith("d"):
            days = int(interval[:-1])
        elif interval.endswith("h"):
            days = 1  # Default to daily for any hourly interval
        else:
            days = 1  # Default to daily

        # Show download message
        with st.spinner(f"Downloading data from CoinGecko for {ticker}..."):
            # Get market chart data
            chart_data = cg.get_coin_market_chart_range_by_id(
                id=coin_id,
                vs_currency="usd",
                from_timestamp=start_timestamp,
                to_timestamp=end_timestamp,
            )

            # Convert to DataFrame
            prices = chart_data.get("prices", [])
            volumes = chart_data.get("total_volumes", [])

            # Create DataFrame
            if prices:
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

                # Add volume data if available
                if volumes:
                    volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
                    df = df.merge(volume_df, on="timestamp", how="left")

                # Format OHLCV structure
                df["Open"] = df["price"]
                df["High"] = df["price"]
                df["Low"] = df["price"]
                df["Close"] = df["price"]
                if "volume" in df.columns:
                    df["Volume"] = df["volume"]

                # Remove temporary columns
                df = df.drop(["timestamp", "price"], axis=1)
                if "volume" in df.columns:
                    df = df.drop(["volume"], axis=1)

                return df
            else:
                logger.warning(f"No price data received from CoinGecko for {coin_id}")
                return None
    except Exception as e:
        logger.error(f"Error fetching data from CoinGecko: {e}", exc_info=True)
        return None


def fetch_data_from_alphavantage(ticker, start_date, end_date, interval):
    """Fetch stock data from Alpha Vantage API"""
    try:
        from alpha_vantage.timeseries import TimeSeries

        # Get API key from session state or config
        api_key = st.session_state.get("alphavantage_api_key")
        if not api_key:
            try:
                from config.config_loader import get_value

                api_key = get_value("api_keys.alphavantage", "demo")
            except ImportError:
                api_key = "demo"  # Use demo key as fallback

        # Initialize TimeSeries
        ts = TimeSeries(key=api_key, output_format="pandas")

        # Map interval to Alpha Vantage function
        if interval == "1d":
            data_function = ts.get_daily_adjusted
            time_param = "full"
        elif interval == "1h":
            data_function = ts.get_intraday
            time_param = "60min"
        elif interval == "5min":
            data_function = ts.get_intraday
            time_param = "5min"
        else:
            data_function = ts.get_daily_adjusted
            time_param = "full"

        # Show download message
        with st.spinner(f"Downloading data from Alpha Vantage for {ticker}..."):
            # Get data
            if data_function == ts.get_intraday:
                data, meta_data = data_function(
                    symbol=ticker, interval=time_param, outputsize="full"
                )
            else:
                data, meta_data = data_function(symbol=ticker, outputsize=time_param)

            # Rename columns to match OHLCV format
            data = data.rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume",
                }
            )

            # Add date column and filter by date range
            data["date"] = data.index

            if start_date:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                else:
                    start_date = pd.to_datetime(start_date)
                data = data[data["date"] >= start_date]

            if end_date:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                else:
                    end_date = pd.to_datetime(end_date)
                data = data[data["date"] <= end_date]

            return data
    except Exception as e:
        logger.error(f"Error fetching data from Alpha Vantage: {e}", exc_info=True)
        return None


def fetch_data_from_finnhub(ticker, start_date, end_date, interval):
    """Fetch stock data from Finnhub API"""
    try:
        import finnhub

        # Get API key from session state or config
        api_key = st.session_state.get("finnhub_api_key")
        if not api_key:
            try:
                from config.config_loader import get_value

                api_key = get_value("api_keys.finnhub", "demo")
            except ImportError:
                api_key = "demo"  # Use demo key as fallback

        # Initialize client
        finnhub_client = finnhub.Client(api_key=api_key)

        # Convert dates to Unix timestamps
        if not isinstance(start_date, str):
            start_timestamp = int(
                datetime.combine(start_date, datetime.min.time()).timestamp()
            )
        else:
            start_timestamp = int(pd.to_datetime(start_date).timestamp())

        if end_date is None:
            end_timestamp = int(datetime.now().timestamp())
        elif not isinstance(end_date, str):
            end_timestamp = int(
                datetime.combine(end_date, datetime.min.time()).timestamp()
            )
        else:
            end_timestamp = int(pd.to_datetime(end_date).timestamp())

        # Map interval to Finnhub resolution
        if interval == "1d":
            resolution = "D"
        elif interval == "1h":
            resolution = "60"
        elif interval == "5min":
            resolution = "5"
        else:
            resolution = "D"

        # Show download message
        with st.spinner(f"Downloading data from Finnhub for {ticker}..."):
            # Get data
            stock_data = finnhub_client.stock_candles(
                ticker, resolution, start_timestamp, end_timestamp
            )

            # Check if data is valid
            if stock_data["s"] == "ok":
                # Create DataFrame
                df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(stock_data["t"], unit="s"),
                        "Open": stock_data["o"],
                        "High": stock_data["h"],
                        "Low": stock_data["l"],
                        "Close": stock_data["c"],
                        "Volume": stock_data["v"],
                    }
                )

                return df
            else:
                logger.warning(f"No data received from Finnhub for {ticker}")
                return None
    except Exception as e:
        logger.error(f"Error fetching data from Finnhub: {e}", exc_info=True)
        return None


def load_data_from_backup(ticker, start_date, end_date, interval):
    """Load data from backup source with progress indication"""
    try:
        # Try multiple backup locations
        backup_paths = [
            f"data/{ticker}_{start_date}_{end_date}_{interval}.csv",
            f"Data/{ticker}_{interval}.csv",
            f"Data/Backup/{ticker}.csv",
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
    required_cols = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required_cols):
        logger.warning(
            f"Missing required columns for indicators. Have: {df.columns.tolist()}"
        )
        return df

    try:
        # Simple Moving Averages
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()

        # Exponential Moving Averages
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_middle"] = df["Close"].rolling(window=20).mean()
        df["BB_std"] = df["Close"].rolling(window=20).std()
        df["BB_upper"] = df["BB_middle"] + 2 * df["BB_std"]
        df["BB_lower"] = df["BB_middle"] - df["BB_std"]

        # Volume indicators - only if Volume column exists
        if "Volume" in df.columns:
            try:
                # Volume Moving Average
                df["Volume_MA"] = df["Volume"].rolling(window=20).mean()

                # Fix the Volume_Ratio calculation:
                # 1. Extract Volume as Series to ensure it's not a DataFrame
                volume_series = df["Volume"].astype(float)
                # 2. Extract Volume_MA as Series and replace zeros to avoid division by zero
                volume_ma_series = df["Volume_MA"].replace(0, 0.001).astype(float)

                # 3. Create a valid index mask to filter out NaN values
                valid_idx = ~volume_series.isna() & ~volume_ma_series.isna()

                # 4. Initialize Volume_Ratio with NaN values
                df["Volume_Ratio"] = np.nan

                # 5. Only calculate for valid indices
                if len(valid_idx[valid_idx == True]) > 0:
                    # For div operations that may return a DataFrame instead of Series,
                    # select first column to maintain Series format
                    division_result = volume_series[valid_idx].div(
                        volume_ma_series[valid_idx]
                    )
                    if isinstance(division_result, pd.DataFrame):
                        df.loc[valid_idx, "Volume_Ratio"] = division_result.iloc[:, 0]
                    else:
                        df.loc[valid_idx, "Volume_Ratio"] = division_result

                    # 6. Handle any infinity or NaN values that might result
                    df["Volume_Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

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

        sentiment_df = pd.DataFrame({"date": dates, "sentiment": sentiment_values})

        # Calculate average sentiment
        avg_sentiment = np.mean(sentiment_values)
        sentiment_label = (
            "Positive"
            if avg_sentiment > 0.2
            else "Negative" if avg_sentiment < -0.2 else "Neutral"
        )

        return {
            "data": sentiment_df,
            "average": avg_sentiment,
            "label": sentiment_label,
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
            # Get settings from session state
            import streamlit as st

            from src.training.walk_forward import generate_future_forecast

            lookback = st.session_state.get("lookback", 30)
            forecast_window = st.session_state.get("forecast_window", 30)

            # Generate and return forecast
            forecast = generate_future_forecast(
                model, df, feature_cols, lookback, forecast_window
            )
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

            forecast = service.generate_forecast(
                df, feature_cols, lookback, forecast_window
            )
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
                    X.append(data[features].values[i : i + window])
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
                X_input, _ = create_sequences(
                    current_data, feature_cols, "Close", lookback, 1
                )
            except Exception as e:
                logger.error(f"Error in forecast iteration {i}: {e}")
                break

        return future_prices

    except Exception as e:
        logger.error(f"Error generating dashboard forecast: {e}")
        return None


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
    missing = [c for c in ["Open", "High", "Low", "Close"] if c not in df_copy.columns]
    if missing:
        print(f"Warning: Missing required columns after standardization: {missing}")
        print(f"Available columns: {df_copy.columns.tolist()}")

    return df_copy


@robust_error_boundary
def apply_custom_indicators(df, timeframe="1d", indicators=None):
    """
    Apply custom indicators (VMLI and WERPI) to the dataframe based on user selections.

    Args:
        df: DataFrame with price data
        timeframe: Data timeframe (e.g., "1d", "1h")
        indicators: Dictionary of indicator flags

    Returns:
        DataFrame with added indicators
    """
    if df is None or df.empty:
        return df

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Default indicators dictionary if none provided
    if indicators is None:
        indicators = {"show_werpi": False, "show_vmli": False}

    # Apply WERPI if requested
    if indicators.get("show_werpi", False):
        try:
            logger.info("Adding WERPI indicator")
            # Try to import from features module
            try:
                from src.features.features import add_werpi_indicator

                df = add_werpi_indicator(
                    df, wavelet_name="db4", level=3, n_states=2, scale_factor=1.0
                )
            except ImportError:
                # Fallback case - log warning
                logger.warning("Could not import WERPI function, skipping indicator")
                df["WERPI"] = np.nan
        except Exception as e:
            logger.error(f"Error calculating WERPI indicator: {e}")
            # Create empty WERPI column if calculation fails
            df["WERPI"] = np.nan

    # Apply VMLI if requested
    if indicators.get("show_vmli", False):
        try:
            logger.info("Adding VMLI indicator")
            # Try to import VMILIndicator
            try:
                from src.features.vmli_indicator import VMILIndicator

                vmli = VMILIndicator(
                    window_mom=14,
                    window_vol=14,
                    smooth_period=3,
                    winsorize_pct=0.01,
                    use_ema=True,
                )

                # Calculate VMLI with components, passing timeframe
                components = vmli.compute(
                    data=df,
                    price_col="Close",
                    volume_col="Volume" if "Volume" in df.columns else None,
                    include_components=True,
                    timeframe=timeframe,
                )

                # Add VMLI components to the dataframe
                df["VMLI"] = components["vmli"]
                df["VMLI_Momentum"] = components["momentum"]
                df["VMLI_Volatility"] = components["volatility"]
                df["VMLI_AdjMomentum"] = components["adj_momentum"]
                df["VMLI_Liquidity"] = components["liquidity"]
                df["VMLI_Raw"] = components["vmli_raw"]
            except (ImportError, KeyError) as e:
                # Fallback case - log warning
                logger.warning(f"Could not import or use VMLI: {e}")
                df["VMLI"] = np.nan

        except Exception as e:
            logger.error(f"Error calculating VMLI indicator: {e}")
            # Create empty VMLI column if calculation fails
            df["VMLI"] = np.nan

    return df


def ensure_indicators_applied(df, ticker, interval="1d"):
    """
    Ensure that all selected indicators are applied to the dataframe.

    Args:
        df: DataFrame with price data
        ticker: Ticker symbol
        interval: Data interval/timeframe

    Returns:
        DataFrame with indicators applied
    """
    if df is None or df.empty:
        return df

    # First apply basic technical indicators
    df = calculate_indicators(df)

    # Then apply custom indicators if requested
    if "indicators" in st.session_state:
        df = apply_custom_indicators(
            df, timeframe=interval, indicators=st.session_state["indicators"]
        )

    # Standardize column names
    df = standardize_column_names(df, ticker)

    return df


# Modified function to update dashboard data with indicators
@robust_error_boundary
def get_dashboard_data(ticker, start_date, end_date=None, interval="1d"):
    """
    Get data for dashboard with all indicators applied based on user selections.
    """
    # First load the raw data
    df = load_data(ticker, start_date, end_date, interval)

    if df is not None and not df.empty:
        # Ensure all selected indicators are applied
        df = ensure_indicators_applied(df, ticker, interval)

    return df
