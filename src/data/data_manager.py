"""
Data manager module for centralized data handling and caching.
Provides a unified interface for data fetching, preprocessing and storage.
"""

import json
import logging
import os
import queue
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)

# Setup logging
logger = logging.getLogger(__name__)

from config.config_loader import (
    DATA_DIR,
    DB_DIR,
    HYPERPARAMS_DIR,
    INTERVAL,
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
    TICKER,
    USE_CACHING,
)

# Import utility modules
from data.data import (
    _download_data_alphavantage,
    _download_data_coingecko,
    _download_data_finnhub,
    fetch_data,
    fetch_data_parallel,
)

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


class DataManagerException(Exception):
    """Base exception class for data manager errors"""


class DataFetchError(DataManagerException):
    """Raised when data fetching fails"""


class DataProcessingError(DataManagerException):
    """Raised when data processing fails"""


class APIRateLimitError(DataManagerException):
    """Raised when API rate limits are exceeded"""


# Create necessary directories
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
RAW_DATA_DIR = os.path.join(DATA_DIR, "Raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "Processed")
PRICE_CACHE_DIR = os.path.join(DATA_DIR, "Prices")

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PRICE_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global processing queue
data_queue = queue.Queue()
processing_active = threading.Event()
processing_active.set()

# Track API rate limits: changed to 60 calls per minute for both APIs
api_calls = {
    "finnhub": {"count": 0, "reset_time": datetime.now(), "limit_per_minute": 60},
    "yfinance": {"count": 0, "reset_time": datetime.now(), "limit_per_minute": 60},
}

# Thread lock for API rate limiting
_api_lock = threading.RLock()


def check_rate_limit(api_name: str) -> bool:
    """
    Thread-safe check if we're within rate limits for the specified API.

    Args:
        api_name: Name of API to check ('finnhub' or 'yfinance')

    Returns:
        True if we can make a call, False if we should wait
    """
    global api_calls

    with _api_lock:
        now = datetime.now()
        api_info = api_calls.get(api_name)

        if not api_info:
            return True

        # Reset counter if 60 seconds have elapsed
        if (now - api_info["reset_time"]).total_seconds() >= 60:
            api_info["count"] = 0
            api_info["reset_time"] = now

        # Check if we're under the limit (60 per minute)
        if api_info["count"] >= api_info["limit_per_minute"]:
            seconds_to_wait = 60 - (now - api_info["reset_time"]).total_seconds()
            logger.warning(
                f"{api_name} rate limit reached. Need to wait {seconds_to_wait:.0f} seconds."
            )
            return False

        # Increment the counter since we'll be making a call
        api_calls[api_name]["count"] += 1
        return True


# Try to import finnhub, handle gracefully if not available
try:
    import finnhub

    FINNHUB_AVAILABLE = True
except ImportError:
    logger.warning("Finnhub module not found. Will use yfinance exclusively.")
    FINNHUB_AVAILABLE = False

# Get Finnhub API key from environment or config
try:
    from config.config_loader import FINNHUB_API_KEY
except ImportError:
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    if not FINNHUB_API_KEY:
        logger.warning("No Finnhub API key found. Finnhub will be disabled.")
        FINNHUB_AVAILABLE = False


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""


def check_rate_limit(api_name: str) -> bool:
    """
    Check if we're within rate limits for the specified API.

    Args:
        api_name: Name of API to check ('finnhub' or 'yfinance')

    Returns:
        True if we can make a call, False if we should wait
    """
    global api_calls
    now = datetime.now()
    api_info = api_calls.get(api_name)

    if not api_info:
        return True

    # Reset counter if 60 seconds have elapsed
    if (now - api_info["reset_time"]).total_seconds() >= 60:
        api_info["count"] = 0
        api_info["reset_time"] = now

    # Check if we're under the limit (60 per minute)
    if api_info["count"] >= api_info["limit_per_minute"]:
        seconds_to_wait = 60 - (now - api_info["reset_time"]).total_seconds()
        logger.warning(
            f"{api_name} rate limit reached. Need to wait {seconds_to_wait:.0f} seconds."
        )
        return False

    return True


def fetch_ticker_data_finnhub(
    ticker: str, start_date: str, end_date: Optional[str] = None, interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Fetch data from Finnhub API with rate limiting.

    Args:
        ticker: Stock symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Data interval ('1d', '1h', etc.)

    Returns:
        DataFrame with data or None if failed
    """
    if not FINNHUB_AVAILABLE or not FINNHUB_API_KEY:
        return None

    if not check_rate_limit("finnhub"):
        return None

    try:
        # Increment the API call counter
        api_calls["finnhub"]["count"] += 1

        # Convert interval to Finnhub format
        resolution_map = {
            "1d": "D",
            "1h": "60",
            "30m": "30",
            "15m": "15",
            "5m": "5",
            "1m": "1",
        }
        resolution = resolution_map.get(interval, "D")

        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date or datetime.now()).timestamp())

        # Make API call
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        logger.info(f"Fetching data from Finnhub for {ticker} with interval {interval}")
        resp = client.stock_candles(ticker, resolution, start_ts, end_ts)

        # Check response status
        if resp.get("s") != "ok":
            logger.warning(f"Finnhub returned status {resp.get('s')} for {ticker}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(resp)
        if df.empty:
            logger.warning(f"Finnhub returned empty data for {ticker}")
            return None

        # Process date column
        if "t" in df.columns:
            df["Date"] = pd.to_datetime(df["t"], unit="s")
            df.drop(columns=["t"], inplace=True)

        # Rename columns to standard format
        column_map = {
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "s": "Status",
        }
        df.rename(columns=column_map, inplace=True, errors="ignore")

        # Drop any unnecessary columns
        if "Status" in df.columns:
            df.drop(columns=["Status"], inplace=True, errors="ignore")

        logger.info(f"Successfully fetched {len(df)} records from Finnhub for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data from Finnhub for {ticker}: {str(e)}")
        return None


def fetch_ticker_data_yfinance(
    ticker: str, start_date: str, end_date: Optional[str] = None, interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance with rate limiting.

    Args:
        ticker: Stock symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Data interval ('1d', '1h', etc.)

    Returns:
        DataFrame with data
    """
    if not check_rate_limit("yfinance"):
        # Wait a bit and then try again
        time.sleep(10)

    # Increment the counter
    api_calls["yfinance"]["count"] += 1

    try:
        logger.info(
            f"Fetching data from YFinance for {ticker} with interval {interval}"
        )
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Download data
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            group_by="column",
        )

        if df.empty:
            raise DataFetchError(f"YFinance returned empty data for {ticker}")

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        # Flatten columns if they're MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(col).rstrip("_") if isinstance(col, tuple) else col
                for col in df.columns
            ]

        # Ensure date column is properly formatted
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "Datetime" in df.columns:
            df["Date"] = pd.to_datetime(df["Datetime"])
            df.drop("Datetime", axis=1, inplace=True)

        # Rename columns to standard format if needed
        for pattern, replacement in [
            # For stock tickers
            ("Adj Close", "Adj_Close"),
            ("Stock Splits", "Stock_Splits"),
            # For crypto tickers
            (f"Open_{ticker}", "Open"),
            (f"High_{ticker}", "High"),
            (f"Low_{ticker}", "Low"),
            (f"Close_{ticker}", "Close"),
            (f"Adj Close_{ticker}", "Adj_Close"),
            (f"Volume_{ticker}", "Volume"),
        ]:
            if pattern in df.columns:
                df.rename(columns={pattern: replacement}, inplace=True)

        logger.info(
            f"Successfully fetched {len(df)} records from YFinance for {ticker}"
        )
        return df

    except Exception as e:
        logger.error(
            f"Error fetching data from YFinance for {ticker}: {str(e)}\n{traceback.format_exc()}"
        )
        raise DataFetchError(
            f"Failed to fetch data from YFinance for {ticker}: {str(e)}"
        )


def fetch_ticker_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch historical stock data with preferred source and fallback.
    Tries Finnhub first, then falls back to YFinance.

    Args:
        ticker: Stock symbol
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD', defaults to today
        interval: Data interval ('1d', '1h', etc.)
        max_retries: Maximum number of retries

    Returns:
        DataFrame with historical data
    """
    retry_count = 0
    backoff = 1

    while retry_count <= max_retries:
        try:
            # Try Finnhub first if available
            df = None
            if FINNHUB_AVAILABLE and FINNHUB_API_KEY:
                df = fetch_ticker_data_finnhub(ticker, start_date, end_date, interval)

            # Fall back to yfinance if finnhub fails or is not available
            if df is None or df.empty:
                logger.info(f"Falling back to YFinance for {ticker}")
                df = fetch_ticker_data_yfinance(ticker, start_date, end_date, interval)

            if df is None or df.empty:
                raise DataFetchError(
                    f"Both data sources returned empty data for {ticker}"
                )

            # Validate the DataFrame
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in required_cols if col not in df.columns]

            if missing:
                logger.warning(f"Missing columns in data: {missing}")
                if "Date" in missing:
                    raise DataFetchError("Date column missing from data")

                # For any other missing columns, add them with zeros
                for col in missing:
                    df[col] = 0.0

            return df

        except Exception as e:
            retry_count += 1
            logger.warning(f"Fetch attempt {retry_count} failed for {ticker}: {str(e)}")
            if retry_count <= max_retries:
                sleep_time = backoff * 2**retry_count  # Exponential backoff
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"All {max_retries+1} attempts failed for {ticker}")
                raise DataFetchError(
                    f"Failed to fetch data after {max_retries+1} attempts"
                )


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing required columns: {missing_cols}")
        return df

    try:
        # Simple Moving Averages
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # Exponential Moving Averages
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        std_dev = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (std_dev * 2)
        df["BB_Lower"] = df["BB_Middle"] - (std_dev * 2)

        # Average True Range (ATR)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()

        # On-Balance Volume (OBV)
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

        # Money Flow Index (MFI)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        raw_money_flow = typical_price * df["Volume"]

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()

        # Avoid division by zero
        mfi_ratio = positive_mf / negative_mf.replace(0, np.finfo(float).eps)
        df["MFI"] = 100 - (100 / (1 + mfi_ratio))

        return df

    except Exception as e:
        logger.error(
            f"Error calculating technical indicators: {str(e)}\n{traceback.format_exc()}"
        )
        return df


def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom features for machine learning.

    Args:
        df: DataFrame with price data and technical indicators

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    try:
        # Price momentum features
        for window in [1, 3, 5, 10, 20]:
            df[f"return_{window}d"] = df["Close"].pct_change(window)

        # Volatility features
        for window in [5, 10, 20]:
            df[f"volatility_{window}d"] = df["Close"].pct_change().rolling(window).std()

        # Volume-based features
        df["volume_change"] = df["Volume"].pct_change()
        df["volume_ma_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        # Price distance from moving averages
        df["price_sma20_ratio"] = df["Close"] / df["SMA_20"]
        df["price_sma50_ratio"] = df["Close"] / df["SMA_50"]

        # Crossover features
        df["sma_5_20_cross"] = np.where(df["SMA_5"] > df["SMA_20"], 1, -1)

        # Trend strength indicators
        df["adx"] = df["ATR"] / df["Close"] * 100  # Simple ADX approximation

        # Add gap features (overnight or weekend price jumps)
        df["gap"] = df["Open"] / df["Close"].shift(1) - 1

        # Add day of week (could be useful for some stocks that have weekly patterns)
        if "Date" in df.columns:
            df["day_of_week"] = df["Date"].dt.dayofweek

        return df

    except Exception as e:
        logger.error(
            f"Error adding custom features: {str(e)}\n{traceback.format_exc()}"
        )
        return df


def preprocess_and_save(ticker: str, df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Preprocess data and save to processed directory.

    Args:
        ticker: Stock symbol
        df: DataFrame with raw data

    Returns:
        (success, filepath) tuple where success is a boolean and filepath is the saved file path
    """
    try:
        if df.empty:
            logger.warning(f"Empty DataFrame for {ticker}, skipping preprocessing")
            return False, ""

        # Add technical indicators
        df = calculate_technical_indicators(df)

        # Add custom features
        df = add_custom_features(df)

        # Handle missing values
        df.fillna(method="ffill", inplace=True)  # Forward fill
        df.fillna(method="bfill", inplace=True)  # Backward fill for any remaining NaNs

        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d")
        outfile = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_{timestamp}.csv")

        # Save to processed directory
        df.to_csv(outfile, index=False)
        logger.info(f"Processed data saved to {outfile}")

        # Save a metadata file with processing details
        metadata = {
            "ticker": ticker,
            "processed_time": datetime.now().isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": [
                df["Date"].min().strftime("%Y-%m-%d"),
                df["Date"].max().strftime("%Y-%m-%d"),
            ],
        }

        meta_file = os.path.splitext(outfile)[0] + "_meta.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return True, outfile

    except Exception as e:
        logger.error(
            f"Error preprocessing data for {ticker}: {str(e)}\n{traceback.format_exc()}"
        )
        return False, ""


def process_data_job(
    tickers: List[str] = None,
    interval: str = "1d",
    days_back: int = 365,
    refresh_rate: int = 3600,
):
    """
    Main data processing job that runs continuously to update data.
    Now with improved rate limiting and source prioritization.

    Args:
        tickers: List of stock symbols to process
        interval: Data interval ('1d', '1h', etc.)
        days_back: Number of days of historical data to fetch
        refresh_rate: Seconds between data refresh cycles
    """
    # Use default tickers if none provided
    if tickers is None:
        try:
            from config.config_loader import TICKERS

            tickers = TICKERS
        except ImportError:
            tickers = ["ETH-USD", "BTC-USD", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    logger.info(f"Starting data processor for tickers: {tickers}")

    while processing_active.is_set():
        try:
            logger.info(f"Beginning data refresh cycle for {len(tickers)} tickers")
            start_time = time.time()

            # Calculate start date based on days_back
            start_date = (datetime.now() - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
            end_date = datetime.now().strftime("%Y-%m-%d")

            # Process each ticker with proper spacing between API calls
            for i, ticker in enumerate(tickers):
                try:
                    logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})")

                    # 1. Fetch data (with Finnhub prioritized for live data)
                    df = fetch_ticker_data(ticker, start_date, end_date, interval)

                    # 2. Save raw data
                    raw_file = os.path.join(
                        RAW_DATA_DIR,
                        f"{ticker}_{datetime.now().strftime('%Y%m%d')}_raw.csv",
                    )
                    df.to_csv(raw_file, index=False)

                    # 3. Process and save processed data
                    success, processed_file = preprocess_and_save(ticker, df)

                    # 4. Push to queue for other components
                    if success:
                        data_queue.put(
                            {
                                "ticker": ticker,
                                "timestamp": datetime.now().isoformat(),
                                "file": processed_file,
                                "rows": len(df),
                            }
                        )

                    # Space out API calls to avoid hitting rate limits
                    if i < len(tickers) - 1:
                        # Use different delays based on which API was used
                        delay = 60 if FINNHUB_AVAILABLE else 10  # seconds
                        logger.debug(
                            f"Waiting {delay}s before next ticker to respect API limits"
                        )
                        # Check for shutdown during wait
                        for _ in range(delay):
                            if not processing_active.is_set():
                                break
                            time.sleep(1)

                except Exception as e:
                    logger.error(f"Error processing ticker {ticker}: {str(e)}")
                    continue

            # Log completion of cycle
            cycle_duration = time.time() - start_time
            logger.info(f"Data refresh cycle completed in {cycle_duration:.2f} seconds")

            # Sleep until next refresh
            sleep_time = max(1, refresh_rate - cycle_duration)
            logger.info(f"Sleeping for {sleep_time:.2f} seconds until next refresh")

            # Use a loop with small sleeps to check for shutdown more frequently
            for _ in range(int(sleep_time / 5) + 1):
                if not processing_active.is_set():
                    break
                time.sleep(min(5, sleep_time))

        except Exception as e:
            logger.error(
                f"Error in data processor main loop: {str(e)}\n{traceback.format_exc()}"
            )
            # If there's a problem, wait a bit and try again
            for _ in range(12):  # 1 minute
                if not processing_active.is_set():
                    break
                time.sleep(5)


def stop_processing():
    """Stop the data processing job."""
    processing_active.clear()
    logger.info("Data processing job stop requested")


def get_latest_processed_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Get the latest processed data for a ticker.

    Args:
        ticker: Stock symbol

    Returns:
        DataFrame with latest processed data or None if not found
    """
    try:
        # Find all files for this ticker
        files = [
            f
            for f in os.listdir(PROCESSED_DATA_DIR)
            if f.startswith(ticker)
            and f.endswith(".csv")
            and not f.endswith("_meta.csv")
        ]

        if not files:
            logger.warning(f"No processed files found for {ticker}")
            return None

        # Sort by modification time (newest first)
        file_paths = [os.path.join(PROCESSED_DATA_DIR, f) for f in files]
        latest_file = max(file_paths, key=os.path.getmtime)

        # Load the data
        df = pd.read_csv(latest_file)
        logger.info(
            f"Loaded latest data for {ticker} from {os.path.basename(latest_file)}"
        )

        # Ensure date column is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        return df

    except Exception as e:
        logger.error(f"Error getting latest data for {ticker}: {str(e)}")
        return None


def fetch_latest_data_for_prediction(
    ticker: str, days: int = 60
) -> Optional[pd.DataFrame]:
    """
    Fetch and process the latest data for a ticker for prediction purposes.

    Args:
        ticker: Stock symbol
        days: Number of days of data to fetch

    Returns:
        Processed DataFrame ready for prediction
    """
    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = fetch_ticker_data(ticker, start_date)

        if df.empty:
            return None

        # Calculate indicators and features
        df = calculate_technical_indicators(df)
        df = add_custom_features(df)

        # Handle missing values
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        logger.info(f"Prepared latest prediction data for {ticker} ({len(df)} rows)")
        return df

    except Exception as e:
        logger.error(f"Error fetching latest prediction data for {ticker}: {str(e)}")
        return None


def fetch_data_for_training(
    ticker=TICKER,
    training_start_date=None,
    end_date=None,
    interval=INTERVAL,
    max_retries=3,
):
    """
    Fetch a larger dataset specifically for training purposes.

    Args:
        ticker: Symbol of the asset
        training_start_date: Earlier start date for adequate training data
        end_date: End date for data collection
        interval: Data interval
        max_retries: Maximum number of retry attempts

    Returns:
        DataFrame with extended historical data for training
    """
    # Default to 1 year ago if no training_start_date provided
    if training_start_date is None:
        training_start_date = (datetime.now() - timedelta(days=365)).strftime(
            "%Y-%m-%d"
        )

    # If end_date is None, use current date
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(
        f"Fetching training data for {ticker} from {training_start_date} to {end_date}"
    )

    # Use the existing fetch_data function with our training date range
    return fetch_data(
        ticker=ticker,
        start=training_start_date,
        end=end_date,
        interval=interval,
        max_retries=max_retries,
    )


# Define price cache directory
# (Moved to the top with other directory definitions)


class DataManager:
    """
    Centralized class for managing data operations.
    Provides methods for fetching, caching and preprocessing financial data.
    """

    def __init__(self, cache_dir=PRICE_CACHE_DIR, use_cache=USE_CACHING):
        """
        Initialize the data manager.

        Args:
            cache_dir: Directory for cached data files
            use_cache: Whether to use data caching
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        os.makedirs(cache_dir, exist_ok=True)
        self.data_sources = {
            "coingecko": _download_data_coingecko,
            "finnhub": _download_data_finnhub,
            "alphavantage": _download_data_alphavantage,
        }

        # Keep a runtime cache of loaded DataFrames
        self._dataframe_cache = {}

    def get_cache_path(self, ticker, interval, start_date, end_date=None):
        """Generate a cache file path for the given parameters"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Create a cache key
        cache_key = f"{ticker}_{interval}_{start_date}_{end_date}"
        return os.path.join(self.cache_dir, f"{cache_key.replace('-', '_')}.parquet")

    def load_from_cache(self, ticker, interval, start_date, end_date=None):
        """Load data from cache if available"""
        if not self.use_cache:
            return None

        cache_path = self.get_cache_path(ticker, interval, start_date, end_date)

        # First check runtime cache
        cache_key = f"{ticker}_{interval}_{start_date}_{end_date}"
        if cache_key in self._dataframe_cache:
            logger.info(f"Loading {ticker} from runtime cache")
            return self._dataframe_cache[cache_key]

        # Then check file cache
        if os.path.exists(cache_path):
            try:
                # Check if file was created recently (within 1 day for daily data)
                file_age = time.time() - os.path.getmtime(cache_path)
                max_age = 3600 * 24  # 1 day in seconds

                if interval == "1d" and file_age > max_age:
                    logger.info(f"Cache for {ticker} is older than 1 day, refreshing")
                    return None

                # Load from parquet file
                df = pd.read_parquet(cache_path)
                if df is not None and not df.empty:
                    logger.info(f"Loaded {ticker} data from cache: {len(df)} rows")

                    # Update runtime cache
                    self._dataframe_cache[cache_key] = df
                    return df
            except Exception as e:
                logger.error(f"Error loading from cache: {e}")

        return None

    def save_to_cache(self, df, ticker, interval, start_date, end_date=None):
        """Save data to cache"""
        if not self.use_cache or df is None or df.empty:
            return

        try:
            cache_path = self.get_cache_path(ticker, interval, start_date, end_date)

            # Save to parquet file (more efficient than CSV)
            df.to_parquet(cache_path, index=False)
            logger.info(f"Saved {ticker} data to cache: {len(df)} rows")

            # Update runtime cache
            cache_key = f"{ticker}_{interval}_{start_date}_{end_date}"
            self._dataframe_cache[cache_key] = df

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    def get_data(
        self,
        ticker=TICKER,
        start_date=START_DATE,
        end_date=None,
        interval=INTERVAL,
        force_refresh=False,
    ):
        """
        Get data for the specified ticker and parameters.

        Args:
            ticker: Symbol of the asset
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            interval: Data interval (1d, 1h, etc.)
            force_refresh: Whether to bypass cache and fetch fresh data

        Returns:
            pandas DataFrame with the requested data
        """
        # Try cache first unless force refresh is specified
        if not force_refresh:
            cached_df = self.load_from_cache(ticker, interval, start_date, end_date)
            if cached_df is not None:
                return cached_df

        # Fetch fresh data
        df = fetch_data(ticker, start_date, end_date, interval)

        # Validate and cache if successful
        if df is not None and not df.empty:
            self.save_to_cache(df, ticker, interval, start_date, end_date)
            return df

        return None

    def get_data_from_specific_source(
        self, source, ticker, start_date, end_date=None, interval=INTERVAL
    ):
        """
        Get data from a specific source without the fallback mechanism.

        Args:
            source: Name of source ('finnhub', 'alphavantage', 'coingecko')
            ticker: Symbol of the asset
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            interval: Data interval

        Returns:
            pandas DataFrame with the requested data or None if failed
        """
        if source not in self.data_sources:
            logger.error(f"Unknown data source: {source}")
            return None

        try:
            df = self.data_sources[source](ticker, start_date, end_date, interval)

            if df is not None and not df.empty:
                # Process date column if needed
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])

                return df

            return None
        except Exception as e:
            logger.error(f"Error fetching from {source}: {e}")
            return None

    def get_multiple_tickers(
        self,
        tickers,
        start_date=START_DATE,
        end_date=None,
        interval=INTERVAL,
        force_refresh=False,
    ):
        """
        Get data for multiple tickers in parallel.

        Args:
            tickers: List of ticker symbols
            Other parameters same as get_data()

        Returns:
            Dictionary mapping tickers to their respective DataFrames
        """
        results = {}

        # First check cache for each ticker
        if not force_refresh:
            for ticker in tickers:
                cached_df = self.load_from_cache(ticker, interval, start_date, end_date)
                if cached_df is not None:
                    results[ticker] = cached_df

        # Get the remaining tickers in parallel
        remaining_tickers = [t for t in tickers if t not in results]
        if remaining_tickers:
            fresh_results = fetch_data_parallel(
                remaining_tickers, start_date, end_date, interval
            )

            # Cache the new results
            for ticker, df in fresh_results.items():
                if df is not None and not df.empty:
                    self.save_to_cache(df, ticker, interval, start_date, end_date)

            # Merge with cache results
            results.update(fresh_results)

        return results

    def clear_cache(self, older_than_days=None):
        """
        Clear the data cache.

        Args:
            older_than_days: If specified, only clear files older than this many days
        """
        # Clear runtime cache
        self._dataframe_cache = {}

        # Clear file cache
        try:
            now = time.time()
            files_removed = 0

            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".parquet"):
                    filepath = os.path.join(self.cache_dir, filename)

                    # Check file age if specified
                    if older_than_days is not None:
                        file_age_days = (now - os.path.getmtime(filepath)) / (3600 * 24)
                        if file_age_days <= older_than_days:
                            continue

                    os.remove(filepath)
                    files_removed += 1

            logger.info(f"Cleared cache: {files_removed} files removed")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_training_data(
        self,
        ticker=TICKER,
        training_start_date=None,
        end_date=None,
        interval=INTERVAL,
        force_refresh=False,
    ):
        """
        Get extended historical data for model training.

        Args:
            ticker: Symbol of the asset
            training_start_date: Earlier start date for training (defaults to 1 year ago)
            end_date: End date (defaults to today)
            interval: Data interval
            force_refresh: Whether to bypass cache

        Returns:
            DataFrame with extended historical data
        """
        # Default to 1 year ago if no training_start_date provided
        if training_start_date is None:
            training_start_date = (datetime.now() - timedelta(days=365)).strftime(
                "%Y-%m-%d"
            )

        # Use cache key that reflects this is training data
        cache_key = (
            f"train_{ticker}_{interval}_{training_start_date}_{end_date or 'today'}"
        )

        # Check cache first
        if not force_refresh and cache_key in self._dataframe_cache:
            logger.info(f"Using cached training data for {ticker}")
            return self._dataframe_cache[cache_key]

        # Fetch fresh training data
        df = fetch_data_for_training(ticker, training_start_date, end_date, interval)

        # Cache the result if successful
        if df is not None and not df.empty:
            self._dataframe_cache[cache_key] = df
            logger.info(f"Cached {len(df)} rows of training data for {ticker}")

        return df


# Create a singleton instance
data_manager = DataManager()


# Convenient wrapper functions
def get_data(*args, **kwargs):
    return data_manager.get_data(*args, **kwargs)


def get_multiple_tickers(*args, **kwargs):
    return data_manager.get_multiple_tickers(*args, **kwargs)


def clear_cache(*args, **kwargs):
    return data_manager.clear_cache(*args, **kwargs)


def get_training_data(*args, **kwargs):
    return data_manager.get_training_data(*args, **kwargs)
