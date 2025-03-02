# data.py

import datetime
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import logging
import streamlit as st

from config import TICKER, START_DATE, INTERVAL, USE_CACHING, PROGRESSIVE_LOADING

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
    @st.cache_data(show_spinner=True)
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

def fetch_data(ticker: str = TICKER, start: str = START_DATE, end=None, interval: str = INTERVAL) -> pd.DataFrame:
    """
    Download data from yfinance, rename and flatten columns, 
    remove holidays/weekends if needed.
    """
    try:
        if PROGRESSIVE_LOADING:
            # If you want to load in chunks, you'd do it here
            pass

        df = _download_data_cached(ticker, start=start, end=end, interval=interval)
        if df is None or df.empty:
            logging.error("Downloaded data is empty or None.")
            return None
        
        # If there's a date column, reset index
        df = df.reset_index()
        
        # Flatten multi-index columns if needed
        df = _flatten_yf_multiindex(df)
        
        # Because we used group_by='column', we might see "Open_ETH-USD" etc.
        df = _rename_crypto_columns(df)
        
        # Rename 'Date' or 'Datetime' to 'date'
        date_cols = ['Date', 'Datetime', 'date']
        for col in date_cols:
            if col in df.columns:
                df.rename(columns={col: 'date'}, inplace=True)
                break

        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # For stocks, remove weekends/holidays if not a crypto ticker
        df = remove_holidays_weekends(df, ticker)

        # Double-check that we have the usual columns
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logging.error(f"Missing expected OHLC columns. Found columns: {df.columns}")
            return None
        
        return df

    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return None

def fetch_data_parallel(tickers, start, end=None, interval="1d"):
    """Fetch data for multiple tickers in parallel"""
    from concurrent.futures import ThreadPoolExecutor
    
    def fetch_single(ticker):
        return ticker, _download_data_cached(ticker, start, end, interval)
    
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as executor:
        for ticker, data in executor.map(fetch_single, tickers):
            results[ticker] = data
    
    return results

def validate_data(df: pd.DataFrame) -> bool:
    """
    Check if fetched data is valid, has minimal rows, and required columns.
    """
    if df is None or df.empty:
        return False
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'date']
    if not all(col in df.columns for col in required_cols):
        return False
    df.dropna(inplace=True)
    return len(df) >= 50
