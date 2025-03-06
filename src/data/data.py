# data.py

import datetime
import logging
import time

import pandas as pd
import pandas_market_calendars as mcal
import requests
import streamlit as st
import yfinance as yf

try:
    import finnhub

    FINNHUB_AVAILABLE = True
except ImportError:
    logging.warning("Module 'finnhub' not installed. Will fall back to yfinance.")
    finnhub = None
    FINNHUB_AVAILABLE = False

try:
    from pycoingecko import CoinGeckoAPI

    COINGECKO_AVAILABLE = True
except ImportError:
    logging.warning(
        "Module 'pycoingecko' not installed. Will fall back to other sources."
    )
    COINGECKO_AVAILABLE = False

from config import (
    ALPHAVANTAGE_API_KEY,
    COINGECKO_RATE_LIMIT_SLEEP,
    FINNHUB_API_KEY,
    INTERVAL,
    START_DATE,
    TICKER,
    USE_CACHING,
)


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
            group_by="column",  # <--- Force "single-level" columns if possible
        )

else:

    def _download_data_cached(ticker, start, end, interval):
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            group_by="column",
        )


def _download_data_finnhub(ticker, start, end, interval):
    """Download data from Finnhub with improved error handling and validation"""
    if not FINNHUB_AVAILABLE:
        logging.warning(
            "Cannot fetch data from Finnhub because the module is not installed."
        )
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
        logging.warning(
            f"Unsupported interval '{interval}' for Finnhub. Falling back to daily ('D')."
        )
        res = "D"

    # Convert dates to timestamps
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(time.time()) if end is None else int(pd.Timestamp(end).timestamp())

    try:
        logging.info(f"Fetching {ticker} from Finnhub with resolution {res}")
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        resp = client.stock_candles(ticker, res, start_ts, end_ts)

        # Validate response
        if resp.get("s") != "ok":
            error_msg = resp.get("s", "Unknown error")
            logging.warning(f"Finnhub returned error: {error_msg}")
            return None

        # Check if we have data
        if not resp.get("c") or len(resp.get("c", [])) == 0:
            logging.warning(f"Finnhub returned empty data for {ticker}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            {
                "Open": resp.get("o", []),
                "High": resp.get("h", []),
                "Low": resp.get("l", []),
                "Close": resp.get("c", []),
                "Volume": resp.get("v", []),
                "date": pd.to_datetime(resp.get("t", []), unit="s"),
            }
        )

        # Validate data quality
        required_cols = {"Open", "High", "Low", "Close", "Volume", "date"}
        if not all(col in df.columns for col in required_cols):
            missing = required_cols - set(df.columns)
            logging.warning(f"Finnhub data missing columns: {missing}")
            return None

        # Check for empty DataFrame
        if df.empty:
            logging.warning(f"Finnhub returned empty DataFrame for {ticker}")
            return None

        # Check for too many NaN values
        nan_pct = (
            df[["Open", "High", "Low", "Close", "Volume"]].isna().mean().mean() * 100
        )
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
        df.columns = [
            "_".join(tup).rstrip("_") if isinstance(tup, tuple) else tup
            for tup in df.columns
        ]
    return df


def _rename_crypto_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If we see columns like 'Open_ETH-USD', rename them to just 'Open', etc.
    Also drop 'Adj Close' if it exists, because we're using auto_adjust=False anyway.
    """
    rename_map = {
        "Open_ETH-USD": "Open",
        "High_ETH-USD": "High",
        "Low_ETH-USD": "Low",
        "Close_ETH-USD": "Close",
        "Adj Close_ETH-USD": "Adj Close",
        "Volume_ETH-USD": "Volume",
        # If the flatten yields e.g. 'OpenETH-USD' (without underscore),
        # we can add those too:
        "OpenETH-USD": "Open",
        "HighETH-USD": "High",
        "LowETH-USD": "Low",
        "CloseETH-USD": "Close",
        "Adj CloseETH-USD": "Adj Close",
        "VolumeETH-USD": "Volume",
    }
    df.rename(columns=rename_map, inplace=True, errors="ignore")

    # If we still have "Adj Close", let's drop it since we have "Close"
    if "Adj Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True, errors="ignore")

    return df


def _download_data_alphavantage(ticker, start, end, interval):
    """Download data from Alpha Vantage with improved handling for cryptocurrencies"""
    if not ALPHAVANTAGE_API_KEY:
        logging.warning(
            "No Alpha Vantage API key provided. Skipping Alpha Vantage fetch."
        )
        return None

    try:
        # Check if it's a cryptocurrency ticker
        is_crypto = "-" in ticker and ("USD" in ticker or "USDT" in ticker)

        if is_crypto:
            # For cryptocurrencies, use the digital currency API
            symbol = ticker.split("-")[0]  # Extract BTC from BTC-USD
            market = "USD"  # Default market

            # Choose the right function based on interval
            if interval in ["1d", "3d", "1wk", "1mo"]:
                function = "DIGITAL_CURRENCY_DAILY"
                logging.info(
                    f"Fetching crypto {ticker} from Alpha Vantage with function {function}"
                )
                url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&market={market}&apikey={ALPHAVANTAGE_API_KEY}"
            else:
                # Alpha Vantage doesn't support all intraday intervals for crypto
                logging.warning(
                    f"Alpha Vantage doesn't support {interval} for crypto. Falling back to daily."
                )
                function = "DIGITAL_CURRENCY_DAILY"
                url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&market={market}&apikey={ALPHAVANTAGE_API_KEY}"
        else:
            # Map yfinance interval to Alpha Vantage interval
            av_interval_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "60min",
                "1d": "daily",
                "1wk": "weekly",
                "1mo": "monthly",
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
                logging.warning(
                    f"Unsupported interval '{interval}' for Alpha Vantage. Falling back to daily."
                )
                function = "TIME_SERIES_DAILY"
                url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}&outputsize=full"

        # Make the request
        logging.info(f"Requesting data from Alpha Vantage: {url}")
        response = requests.get(url)
        data = response.json()

        # Check for error messages
        if "Error Message" in data:
            logging.warning(f"Alpha Vantage error: {data['Error Message']}")
            return None

        if "Information" in data:
            logging.info(f"Alpha Vantage info: {data['Information']}")
            # Check if it's a rate limit message
            if "Thank you for using Alpha Vantage!" in data["Information"]:
                logging.warning("Alpha Vantage API call limit reached")
                return None

        # Process the response based on function type
        if "DIGITAL_CURRENCY" in function:
            # Handle crypto data format which is different
            time_series_key = f"Time Series (Digital Currency Daily)"
            if time_series_key in data:
                df = _process_alpha_vantage_crypto_data(data, time_series_key)

                # Filter by date range
                if start:
                    df = df[df["date"] >= pd.to_datetime(start)]
                if end:
                    df = df[df["date"] <= pd.to_datetime(end)]

                return df
            else:
                logging.warning(
                    f"Missing time series key '{time_series_key}' in Alpha Vantage response"
                )
                return None
        else:
            # Get time series data
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break

            if not time_series_key or not data.get(time_series_key):
                logging.warning(
                    f"No time series data found in Alpha Vantage response for {ticker}"
                )
                return None

            # Convert to DataFrame
            time_series = data[time_series_key]
            df_list = []

            for date_str, values in time_series.items():
                row = {
                    "date": pd.to_datetime(date_str),
                    "Open": float(values.get("1. open", 0)),
                    "High": float(values.get("2. high", 0)),
                    "Low": float(values.get("3. low", 0)),
                    "Close": float(values.get("4. close", 0)),
                    "Volume": float(
                        values.get("5. volume", 0) if "5. volume" in values else 0
                    ),
                }
                df_list.append(row)

            df = pd.DataFrame(df_list)

            # Sort by date and filter by date range
            df.sort_values("date", inplace=True)
            if start:
                df = df[df["date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["date"] <= pd.to_datetime(end)]

            return df

    except Exception as e:
        logging.error(f"Alpha Vantage error: {type(e).__name__}: {str(e)}")
        return None


def _process_alpha_vantage_crypto_data(data, time_series_key):
    """Process cryptocurrency data from Alpha Vantage"""
    records = []
    for date, values in data[time_series_key].items():
        try:
            record = {
                "date": date,
                "Open": float(values["1a. open (USD)"]),
                "High": float(values["2a. high (USD)"]),
                "Low": float(values["3a. low (USD)"]),
                "Close": float(values["4a. close (USD)"]),
                "Volume": float(values["5. volume"]),
            }
            records.append(record)
        except KeyError as e:
            # Skip this record if keys don't match expected format
            logging.warning(f"Missing key in Alpha Vantage crypto data: {e}")
            continue

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    return df


def _download_data_coingecko(ticker, start, end, interval):
    """Download cryptocurrency data from CoinGecko with error handling and validation"""
    if not COINGECKO_AVAILABLE:
        logging.warning(
            "Cannot fetch data from CoinGecko because pycoingecko is not installed."
        )
        return None

    # Check if it's a cryptocurrency ticker (most likely has '-USD' format)
    if "-" not in ticker or ("USD" not in ticker and "USDT" not in ticker):
        logging.info(f"Skipping CoinGecko for non-crypto ticker: {ticker}")
        return None

    # Extract the coin symbol from the ticker (e.g., "BTC-USD" -> "BTC")
    coin_symbol = ticker.split("-")[0].lower()

    # Map common symbols to CoinGecko IDs
    symbol_to_id = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana",
        "link": "chainlink",
        "matic": "matic-network",
        "avax": "avalanche-2",
        "aave": "aave",
        "ltc": "litecoin",
        "doge": "dogecoin",
        "dot": "polkadot",
        "ada": "cardano",
        "rvn": "ravencoin",
        "xrp": "ripple",
    }

    # Get coin ID or use lowercase symbol if not in mapping
    coin_id = symbol_to_id.get(coin_symbol, coin_symbol)

    # Map yfinance interval to CoinGecko interval (days)
    interval_map = {
        "1d": 1,
        "3d": 3,
        "1wk": 7,
        "1mo": 30,
        # CoinGecko's free API doesn't support hourly granularity
        # so we'll use daily data for all intraday intervals
        "1h": 1,
        "2h": 1,
        "4h": 1,
        "6h": 1,
        "8h": 1,
        "12h": 1,
        "1m": 1,
        "5m": 1,
        "15m": 1,
        "30m": 1,
    }

    days = interval_map.get(interval, 1)

    try:
        logging.info(
            f"Fetching {ticker} (as {coin_id}) from CoinGecko with interval {interval} (mapped to {days} days)"
        )

        # Convert start and end dates to Unix timestamps (milliseconds)
        start_ts = int(pd.Timestamp(start).timestamp()) if start else None
        end_ts = (
            int(pd.Timestamp(end).timestamp())
            if end
            else int(datetime.datetime.now().timestamp())
        )

        # Calculate days from timestamps if both are provided
        if start_ts and end_ts:
            days = min(
                max(1, int((end_ts - start_ts) / (24 * 3600))), 365 * 2
            )  # Cap at 2 years

        # Initialize CoinGecko API
        cg = CoinGeckoAPI()

        # Get market data
        # Note: we'll use vs_currency='usd' since most tickers end with -USD
        market_data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency="usd",
            days=days,
            interval="daily",  # Use daily for all intervals for consistency
        )

        # Respect rate limits to prevent IP bans
        if COINGECKO_RATE_LIMIT_SLEEP > 0:
            time.sleep(COINGECKO_RATE_LIMIT_SLEEP)

        # Create dataframe
        if not market_data or not all(
            key in market_data for key in ["prices", "market_caps", "total_volumes"]
        ):
            logging.warning(f"CoinGecko returned incomplete data for {ticker}")
            return None

        # Extract timestamps and prices
        timestamps = [
            entry[0] / 1000 for entry in market_data["prices"]
        ]  # Convert ms to seconds
        prices = [entry[1] for entry in market_data["prices"]]
        volumes = [entry[1] for entry in market_data["total_volumes"]]

        # Since CoinGecko doesn't provide OHLC directly in the free API,
        # we'll use the daily closing price and estimate other values
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s"),
                "Close": prices,
                "Volume": volumes,
            }
        )

        # Sort by date
        df = df.sort_values("date")

        # Add synthetic OHLC (using close +/- small random variation)
        # This is a simple estimation since we only have closing prices
        price_variations = df["Close"] * 0.02  # 2% variation
        df["Open"] = df["Close"].shift(1).fillna(df["Close"] * 0.995)
        df["High"] = df["Close"] + price_variations.abs()
        df["Low"] = df["Close"] - price_variations.abs()

        # Filter by date range
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]

        # Check if we have enough data
        if len(df) < 5:
            logging.warning(
                f"CoinGecko returned insufficient data points ({len(df)}) for {ticker}"
            )
            return None

        logging.info(f"Successfully fetched {len(df)} rows from CoinGecko for {ticker}")
        return df

    except Exception as e:
        logging.error(f"CoinGecko error: {type(e).__name__}: {str(e)}")
        return None


def fetch_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end=None,
    interval: str = INTERVAL,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch financial data from multiple sources with fallbacks."""
    retry_count = 0
    backoff_factor = 1.5
    current_delay = 1.0

    # Ensure start and end dates are not in the future.
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    if start > today_str:
        logging.warning(
            f"Start date {start} is in the future. Adjusting to {today_str}."
        )
        start = today_str
    if end is not None and end > today_str:
        logging.warning(f"End date {end} is in the future. Adjusting to {today_str}.")
        end = today_str

    while retry_count <= max_retries:
        try:
            df = None
            # For crypto tickers, always use CoinGecko exclusively.
            if "-" in ticker.upper() and (
                "USD" in ticker.upper() or "USDT" in ticker.upper()
            ):
                logging.info(f"Using CoinGecko exclusively for crypto ticker {ticker}")
                df = _download_data_coingecko(ticker, start, end, interval)
            else:
                # For non-crypto tickers, try Finnhub then fall back to Alpha Vantage.
                logging.info(
                    f"Attempt {retry_count+1}: Fetching {ticker} data from Finnhub"
                )
                df = _download_data_finnhub(ticker, start, end, interval)
                if df is None or df.empty or not validate_data(df):
                    logging.info(f"Finnhub failed; Trying Alpha Vantage for {ticker}")
                    df = _download_data_alphavantage(ticker, start, end, interval)

            if df is not None and validate_data(df):
                # ...existing code: process and return df...
                return df
        except Exception as e:
            logging.error(f"Error in fetch_data attempt {retry_count}: {e}")
        retry_count += 1
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

    required_cols = ["Open", "High", "Low", "Close", "Volume", "date"]
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        logging.warning(f"Data missing required columns: {missing}")
        return False

    # Check minimum row count
    if len(df) < 20:
        logging.warning(f"Data has insufficient rows: {len(df)}")
        return False

    # Check for excessive NaN values
    nan_pct = df[["Open", "High", "Low", "Close", "Volume"]].isna().mean().mean() * 100
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
