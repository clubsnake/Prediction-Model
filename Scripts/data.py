# data.py

import datetime
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import logging
import streamlit as st
import time
import requests

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    logging.warning("Module 'finnhub' not installed. Will fall back to yfinance.")
    finnhub = None
    FINNHUB_AVAILABLE = False

from config import TICKER, START_DATE, INTERVAL, USE_CACHING, PROGRESSIVE_LOADING, FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY

def remove_holidays_weekends(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    If the ticker is a stock (no 'USD'), remove weekends & holidays.
    For crypto (containing 'USD'), keep all days.
    """
    ticker = str(ticker)
    if "USD" not in ticker.upper():
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=df["date"].min(), end_date=df["date"].max())
        valid_dates = schedule.index.date
        df["date_only"] = df["date"].dt.date
        df = df[df["date_only"].isin(valid_dates)].copy()
        df.drop(columns=["date_only"], inplace=True)
    return df

# ------------------------------------------------------------------------
# ADJUST THIS PART: Force group_by='column' and auto_adjust=False in yfinance,
# then flatten multi-index columns if needed.
# ------------------------------------------------------------------------
if USE_CACHING:
    @st.cache(show_spinner=True)
    def _download_data_cached(ticker, start, end, interval):
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,  # Keep unadjusted columns
            group_by='column'    # <--- Force "single-level" columns if possible
        )
else:
    def _download_data_cached(ticker, start, end, interval):
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            group_by='column'
        )

def _download_data_finnhub(ticker, start, end, interval):
    """Download data from Finnhub with improved error handling and validation"""
    if not FINNHUB_AVAILABLE:
        logging.warning("Cannot fetch data from Finnhub because the module is not installed.")
        return None
    
    if not FINNHUB_API_KEY:
        logging.warning("No Finnhub API key provided. Skipping Finnhub fetch.")
        return None
    
    # Map yfinance interval to Finnhub resolution
    resolution_map = {
        "1d": "D",
        "1h": "60",
        "30m": "30", 
        "15m": "15",
        "5m": "5",
        "1m": "1",
    }
    
    res = resolution_map.get(interval, "D")
    if res not in resolution_map.values():
        logging.warning(f"Unsupported interval '{interval}' for Finnhub. Falling back to daily ('D').")
        res = "D"
    
    # Convert dates to timestamps
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(time.time()) if end is None else int(pd.Timestamp(end).timestamp())
    
    try:
        logging.info(f"Fetching {ticker} from Finnhub with resolution {res}")
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        resp = client.stock_candles(ticker, res, start_ts, end_ts)
        
        # Validate response
        if resp.get('s') != 'ok':
            error_msg = resp.get('s', 'Unknown error')
            logging.warning(f"Finnhub returned error: {error_msg}")
            return None
        
        # Check if we have data
        if not resp.get('c') or len(resp.get('c', [])) == 0:
            logging.warning(f"Finnhub returned empty data for {ticker}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'Open': resp.get('o', []),
            'High': resp.get('h', []),
            'Low': resp.get('l', []),
            'Close': resp.get('c', []),
            'Volume': resp.get('v', []),
            'date': pd.to_datetime(resp.get('t', []), unit='s')
        })
        
        # Validate data quality
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'date'}
        if not all(col in df.columns for col in required_cols):
            missing = required_cols - set(df.columns)
            logging.warning(f"Finnhub data missing columns: {missing}")
            return None
        
        # Check for empty DataFrame
        if df.empty:
            logging.warning(f"Finnhub returned empty DataFrame for {ticker}")
            return None
            
        # Check for too many NaN values
        nan_pct = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().mean().mean() * 100
        if nan_pct > 25:
            logging.warning(f"Finnhub data has {nan_pct:.1f}% NaN values for {ticker}")
            return None
        
        logging.info(f"Successfully fetched {len(df)} rows from Finnhub for {ticker}")
        return df
        
    except Exception as e:
        logging.error(f"Finnhub error: {type(e).__name__}: {str(e)}")
        return None

def _flatten_yf_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    If yfinance returns a MultiIndex like ("Open","ETH-USD"), flatten it
    to something like "Open_ETH-USD". Then rename to standard "Open", etc.
    """
    # Flatten the columns if it's a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Join each tuple in the multi-index into a single string
        df.columns = ['_'.join(tup).rstrip('_') if isinstance(tup, tuple) else tup
                      for tup in df.columns]
    return df

def _rename_crypto_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If we see columns like 'Open_ETH-USD', rename them to just 'Open', etc.
    Also drop 'Adj Close' if it exists, because we're using auto_adjust=False anyway.
    """
    rename_map = {
        'Open_ETH-USD': 'Open',
        'High_ETH-USD': 'High',
        'Low_ETH-USD': 'Low',
        'Close_ETH-USD': 'Close',
        'Adj Close_ETH-USD': 'Adj Close',
        'Volume_ETH-USD': 'Volume',
        # If the flatten yields e.g. 'OpenETH-USD' (without underscore),
        # we can add those too:
        'OpenETH-USD': 'Open',
        'HighETH-USD': 'High',
        'LowETH-USD': 'Low',
        'CloseETH-USD': 'Close',
        'Adj CloseETH-USD': 'Adj Close',
        'VolumeETH-USD': 'Volume',
    }
    df.rename(columns=rename_map, inplace=True, errors='ignore')
    
    # If we still have "Adj Close", let's drop it since we have "Close"
    if 'Adj Close' in df.columns:
        df.drop(columns=['Adj Close'], inplace=True, errors='ignore')
    
    return df

def _download_data_alphavantage(ticker, start, end, interval):
    """Download data from Alpha Vantage with error handling and validation"""
    if not ALPHAVANTAGE_API_KEY:
        logging.warning("No Alpha Vantage API key provided. Skipping Alpha Vantage fetch.")
        return None
    
    # Map yfinance interval to Alpha Vantage interval
    av_interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
        "1d": "daily",
        "1wk": "weekly",
        "1mo": "monthly"
    }
    
    av_interval = av_interval_map.get(interval, "daily")
    
    # Determine which function to use based on interval
    if av_interval in ["1min", "5min", "15min", "30min", "60min"]:
        function = "TIME_SERIES_INTRADAY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={av_interval}&apikey={ALPHAVANTAGE_API_KEY}&outputsize=full"
    elif av_interval == "daily":
        function = "TIME_SERIES_DAILY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}&outputsize=full"
    elif av_interval == "weekly":
        function = "TIME_SERIES_WEEKLY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    elif av_interval == "monthly":
        function = "TIME_SERIES_MONTHLY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    else:
        logging.warning(f"Unsupported interval '{interval}' for Alpha Vantage. Falling back to daily.")
        function = "TIME_SERIES_DAILY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}&outputsize=full"
    
    try:
        logging.info(f"Fetching {ticker} from Alpha Vantage with function {function}")
        response = requests.get(url)
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            logging.warning(f"Alpha Vantage error: {data['Error Message']}")
            return None
        
        if 'Information' in data:
            logging.info(f"Alpha Vantage info: {data['Information']}")
            # Check if it's a rate limit message
            if 'Thank you for using Alpha Vantage!' in data['Information']:
                logging.warning("Alpha Vantage API call limit reached")
                return None
        
        # Get time series data
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
                
        if not time_series_key or not data.get(time_series_key):
            logging.warning(f"No time series data found in Alpha Vantage response for {ticker}")
            return None
            
        # Convert to DataFrame
        time_series = data[time_series_key]
        df_list = []
        
        for date_str, values in time_series.items():
            row = {
                'date': pd.to_datetime(date_str),
                'Open': float(values.get('1. open', 0)),
                'High': float(values.get('2. high', 0)),
                'Low': float(values.get('3. low', 0)),
                'Close': float(values.get('4. close', 0)),
                'Volume': float(values.get('5. volume', 0) if '5. volume' in values else 0)
            }
            df_list.append(row)
        
        df = pd.DataFrame(df_list)
        
        # Sort by date and filter by date range
        df.sort_values('date', inplace=True)
        if start:
            df = df[df['date'] >= pd.to_datetime(start)]
        if end:
            df = df[df['date'] <= pd.to_datetime(end)]
        
        # Check if DataFrame is valid
        if df.empty:
            logging.warning(f"No data found in date range for {ticker} from Alpha Vantage")
            return None
            
        # Check for data quality
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'date'}
        if not all(col in df.columns for col in required_cols):
            missing = required_cols - set(df.columns)
            logging.warning(f"Alpha Vantage data missing columns: {missing}")
            return None
            
        # Check for too many NaN values
        nan_pct = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().mean().mean() * 100
        if nan_pct > 25:
            logging.warning(f"Alpha Vantage data has {nan_pct:.1f}% NaN values for {ticker}")
            return None
            
        logging.info(f"Successfully fetched {len(df)} rows from Alpha Vantage for {ticker}")
        return df
        
    except Exception as e:
        logging.error(f"Alpha Vantage error: {type(e).__name__}: {str(e)}")
        return None

def fetch_data(ticker: str = TICKER, start: str = START_DATE, end=None, 
               interval: str = INTERVAL, max_retries: int = 3) -> pd.DataFrame:
    """Fetch financial data with three sources: Finnhub -> yfinance -> Alpha Vantage."""
    retry_count = 0
    backoff_factor = 1.5  # Exponential backoff
    current_delay = 1.0  # Starting delay in seconds
    
    # Ensure start date is not in the future
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if start > today:
        logging.warning(f"Start date {start} is in the future. Adjusting to {today}")
        start = today

    while retry_count <= max_retries:
        try:
            # First attempt: Use Finnhub
            logging.info(f"Attempt {retry_count+1}/{max_retries+1}: Fetching {ticker} ({interval}) data from Finnhub")
            df = _download_data_finnhub(ticker, start, end, interval)
            
            # Check if Finnhub returned valid data
            if df is not None and not df.empty and validate_data(df):
                logging.info(f"Successfully fetched {len(df)} rows from Finnhub for {ticker}")
                # Process and return the valid Finnhub data
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                df = remove_holidays_weekends(df, ticker)
                return df
            
            # Second attempt: Fallback to yfinance if Finnhub failed
            logging.info(f"Finnhub fetch failed or returned invalid data. Falling back to yfinance.")
            df = _download_data_cached(ticker, start=start, end=end, interval=interval)
            
            # Process yfinance data
            if df is not None and not df.empty:
                df = df.reset_index()
                df = _flatten_yf_multiindex(df)
                df = _rename_crypto_columns(df)
                
                # Convert date column
                date_cols = ['Date', 'Datetime', 'date']
                for col in date_cols:
                    if col in df.columns:
                        df.rename(columns={col: 'date'}, inplace=True)
                        break
                
                # Validate the processed yfinance data
                if validate_data(df):
                    logging.info(f"Successfully fetched {len(df)} rows from yfinance for {ticker}")
                    df = remove_holidays_weekends(df, ticker)
                    return df
            
            # Third attempt: Try Alpha Vantage as last resort
            logging.info(f"YFinance fetch failed or returned invalid data. Falling back to Alpha Vantage.")
            df = _download_data_alphavantage(ticker, start, end, interval)
            
            # Process Alpha Vantage data
            if df is not None and not df.empty and validate_data(df):
                logging.info(f"Successfully fetched {len(df)} rows from Alpha Vantage for {ticker}")
                df = remove_holidays_weekends(df, ticker)
                return df
            
            # If we get here, all sources failed
            logging.warning(f"Failed to fetch valid data from all three sources (attempt {retry_count+1}/{max_retries+1})")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"Retrying in {current_delay:.2f} seconds...")
                time.sleep(current_delay)
                current_delay *= backoff_factor
                
        except Exception as e:
            error_type = type(e).__name__
            logging.error(f"Error in attempt {retry_count+1}/{max_retries+1}: {error_type}: {str(e)}")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"Retrying in {current_delay:.2f} seconds...")
                time.sleep(current_delay)
                current_delay *= backoff_factor
    
    logging.error(f"All {max_retries+1} attempts to fetch data from all sources failed")
    return None

def validate_data(df: pd.DataFrame) -> bool:
    """
    Check if fetched data is valid, has minimal rows, and required columns.
    """
    if df is None or df.empty:
        return False
        
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'date']
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        logging.warning(f"Data missing required columns: {missing}")
        return False
        
    # Check minimum row count
    if len(df) < 20:
        logging.warning(f"Data has insufficient rows: {len(df)}")
        return False
        
    # Check for excessive NaN values
    nan_pct = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().mean().mean() * 100
    if nan_pct > 10:
        logging.warning(f"Data has high percentage of NaN values: {nan_pct:.1f}%")
        return False
        
    return True

def fetch_data_parallel(tickers, start, end=None, interval="1d"):
    """Fetch data for multiple tickers in parallel, prioritizing Finnhub"""
    from concurrent.futures import ThreadPoolExecutor
    
    def fetch_single(ticker):
        return ticker, fetch_data(ticker, start, end, interval)
    
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as executor:
        for ticker, data in executor.map(fetch_single, tickers):
            results[ticker] = data
    
    return results

# The duplicate fetch_data_with_retry function is no longer needed since
# fetch_data already includes proper retry logic, so we've removed it.
