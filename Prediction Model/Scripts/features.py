# features.py
"""
Technical indicator calculation and feature engineering logic.
"""

import numpy as np
import pandas as pd
import pywt
from hmmlearn import hmm

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add RSI (Relative Strength Index) to df.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean().replace(0, np.nan).ffill()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Add MACD indicators to df.
    """
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Add Bollinger Bands to df.
    """
    sma = df["Close"].rolling(window=window).mean()
    rstd = df["Close"].rolling(window=window).std()
    df["Bollinger_Mid"] = sma
    df["Bollinger_Upper"] = sma + (num_std * rstd)
    df["Bollinger_Lower"] = sma - (num_std * rstd)
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) to df.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    df["ATR"] = tr.rolling(window=period).mean()
    return df

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add On-Balance Volume (OBV) to df.
    """
    close_series = df["Close"]
    volume_series = df["Volume"]
    df["Close"] = pd.to_numeric(close_series, errors="coerce")
    df["Volume"] = pd.to_numeric(volume_series, errors="coerce")
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df

def add_werpi_indicator(df: pd.DataFrame, wavelet_name: str = "db4", level: int = 3,
                        n_states: int = 2, scale_factor: float = 1.0) -> pd.DataFrame:
    """
    Add a wavelet + HMM-based indicator (WERPI) to df.
    """
    if "Close" not in df.columns or len(df["Close"]) < level + 2:
        df["WERPI"] = np.nan
        return df
    
    closes = df["Close"].values
    returns = np.log(closes[1:] / closes[:-1])
    if len(returns) < 2**level:
        df["WERPI"] = np.nan
        return df
    
    coeffs = pywt.wavedec(returns, wavelet_name, level=level)
    reconstructed_levels = []
    for i in range(len(coeffs)):
        level_coeffs = [np.zeros_like(arr) if j != i else arr for j, arr in enumerate(coeffs)]
        x_i = pywt.waverec(level_coeffs, wavelet_name)[:len(returns)]
        reconstructed_levels.append(x_i)
    
    min_length = min(len(arr) for arr in reconstructed_levels)
    reconstructed_levels = [arr[:min_length] for arr in reconstructed_levels]
    returns_trimmed = returns[:min_length]
    
    if "Volume" in df.columns:
        volume_trimmed = df["Volume"].values[1:1+min_length]
        features = np.column_stack(reconstructed_levels + [returns_trimmed, volume_trimmed])
    else:
        features = np.column_stack(reconstructed_levels + [returns_trimmed])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    startprob_prior = np.full(n_states, 1.0 / n_states)
    transmat_prior = np.full((n_states, n_states), 1.0 / n_states)
    
    try:
        model_hmm = hmm.GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=100, 
            random_state=42, startprob_prior=startprob_prior, 
            transmat_prior=transmat_prior
        )
        model_hmm.fit(features)
        epsilon = 1e-6
        transmat = model_hmm.transmat_.copy()
        transmat = np.maximum(transmat, epsilon)
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        model_hmm.transmat_ = transmat
        
        posterior_probs = model_hmm.predict_proba(features)
        werpi_values = scale_factor * (2.0 * posterior_probs[:, 0] - 1.0)
        
        df["WERPI"] = np.nan
        df.loc[1:min_length+1, "WERPI"] = werpi_values
        df["WERPI"].fillna(method='ffill', inplace=True)
        df["WERPI"].fillna(method='bfill', inplace=True)
    except Exception as e:
        print(f"Error in WERPI calculation: {e}")
        df["WERPI"] = np.nan
    return df

def add_weekend_gap_feature(df: pd.DataFrame, apply_gap: bool = True) -> pd.DataFrame:
    """
    Add a 'WeekendGap' column to capture price jumps if the market was closed.
    """
    if not apply_gap:
        df["WeekendGap"] = 0.0
        return df
    df["WeekendGap"] = 0.0
    for i in range(1, len(df)):
        prev_date = df.loc[i-1, "date"]
        current_date = df.loc[i, "date"]
        day_diff = (current_date - prev_date).days
        if day_diff > 1:
            prev_close = df.loc[i-1, "Close"]
            current_open = df.loc[i, "Open"]
            gap = (current_open - prev_close) / max(prev_close, 1e-9)
            df.loc[i, "WeekendGap"] = gap
    return df

def feature_engineering_with_params(
    df: pd.DataFrame,
    ticker: str = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    boll_window: int = 20,
    boll_nstd: float = 2.0,
    atr_period: int = 14,
    werpi_wavelet: str = "db4",
    werpi_level: int = 3,
    werpi_n_states: int = 2,
    werpi_scale: float = 1.0,
    apply_weekend_gap: bool = True
) -> pd.DataFrame:
    """
    Apply multiple technical indicators to the DataFrame according 
    to specified parameters.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_rsi(df, period=rsi_period)
    df = add_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = add_bollinger_bands(df, window=boll_window, num_std=boll_nstd)
    df = add_atr(df, period=atr_period)
    df = add_obv(df)
    df = add_werpi_indicator(df, wavelet_name=werpi_wavelet, level=werpi_level, 
                             n_states=werpi_n_states, scale_factor=werpi_scale)
    
    if ticker is not None:
        is_crypto = isinstance(ticker, str) and ticker.upper().endswith('-USD')
        df["WeekendGap"] = 0.0
        if not is_crypto:
            df = add_weekend_gap_feature(df, apply_gap=apply_weekend_gap)
    else:
        df["WeekendGap"] = 0.0
    
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=10).std()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def feature_engineering(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Feature engineering entry point that reads default parameters from config.
    """
    from config import (
        RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
        BOLL_WINDOW, BOLL_NSTD, ATR_PERIOD,
        WERPI_WAVELET, WERPI_LEVEL, WERPI_N_STATES, WERPI_SCALE,
        APPLY_WEEKEND_GAP
    )
    return feature_engineering_with_params(
        df,
        ticker=ticker,
        rsi_period=RSI_PERIOD,
        macd_fast=MACD_FAST,
        macd_slow=MACD_SLOW,
        macd_signal=MACD_SIGNAL,
        boll_window=BOLL_WINDOW,
        boll_nstd=BOLL_NSTD,
        atr_period=ATR_PERIOD,
        werpi_wavelet=WERPI_WAVELET,
        werpi_level=WERPI_LEVEL,
        werpi_n_states=WERPI_N_STATES,
        werpi_scale=WERPI_SCALE,
        apply_weekend_gap=APPLY_WEEKEND_GAP
    )

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience method to apply all default indicators without custom params.
    """
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_werpi_indicator(df)
    return df
