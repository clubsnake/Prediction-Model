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
    # Make a copy to avoid chained assignment warnings
    df = df.copy()
    
    # Calculate returns if not already present
    if 'returns' not in df.columns:
        df.loc[:, 'returns'] = df['Close'].pct_change()
    
    valid_returns = df['returns'].dropna()
    if len(valid_returns) < 2**level:  # Ensure enough data points
        df["WERPI"] = np.nan
        return df
        
    returns_array = valid_returns.values.reshape(-1, 1)

    try:
        # Initialize and fit the HMM with the specified number of states
        model_hmm = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="diag", 
            n_iter=1000,
            random_state=42
        )
        model_hmm.fit(returns_array)
        
        # Normalize the start probabilities to ensure they sum to 1
        if hasattr(model_hmm, 'startprob_'):
            total = model_hmm.startprob_.sum()
            if total <= 0 or np.isnan(total):
                raise ValueError("Invalid start probabilities in HMM.")
            model_hmm.startprob_ = model_hmm.startprob_ / total
        else:
            raise ValueError("HMM model does not have start probabilities.")

        # Compute the WERPI indicator using state probabilities
        state_probs = model_hmm.predict_proba(returns_array)
        max_probs = np.max(state_probs, axis=1)
        
        # Apply wavelet transform
        coeffs = pywt.wavedec(max_probs, wavelet_name, level=level)
        approx = coeffs[0] * scale_factor
        
        # Interpolate to original length
        x_old = np.linspace(0, len(max_probs)-1, num=len(approx))
        x_new = np.arange(len(max_probs))
        werpi_values = np.interp(x_new, x_old, approx)
        
        # Build an array of NaNs for the full DataFrame length and assign computed values
        werpi_full = np.full((len(df), 1), np.nan)
        werpi_full[valid_returns.index, 0] = werpi_values.flatten()
        df.loc[:, 'WERPI'] = werpi_full
        
        # Fill NaN values
        df["WERPI"].fillna(method='ffill', inplace=True)
        df["WERPI"].fillna(method='bfill', inplace=True)

    except Exception as e:
        print(f"Error in WERPI calculation: {str(e)}")
        df.loc[:, 'WERPI'] = np.nan
        
    return df

def add_weekend_gap_feature(df: pd.DataFrame, apply_gap: bool = True) -> pd.DataFrame:
    """Optimized vectorized implementation"""
    if not apply_gap:
        df["WeekendGap"] = 0.0
        return df
        
    # Calculate day differences vectorized
    df["WeekendGap"] = 0.0
    df["date_diff"] = (df["date"].diff().dt.days).fillna(0)
    
    # Create masks for gaps
    gap_mask = df["date_diff"] > 1
    
    # Calculate gaps vectorized where needed
    if gap_mask.any():
        prev_close = df["Close"].shift(1)
        current_open = df["Open"]
        gaps = (current_open - prev_close) / np.maximum(prev_close, 1e-9)
        df.loc[gap_mask, "WeekendGap"] = gaps[gap_mask]
        
    # Remove temporary column
    df.drop("date_diff", axis=1, inplace=True)
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

def compute_werpi(data, hmm_model, level=3, scale=1.0, wavelet='db4'):
    """
    Parameters:
      data (ndarray): Input data array (e.g. returns) of shape (n_samples, 1)
      hmm_model (GaussianHMM): A fitted HMM model.
      level (int): Decomposition level for the wavelet transform.
      scale (float): Scaling factor applied to the wavelet approximation coefficients.
      wavelet (str): Wavelet name to use (e.g., 'db2').

    Returns:
      werpi_full (ndarray): The computed WERPI indicator with shape (n_samples, 1)
    """
    # Get state probabilities from the HMM
    state_probs = hmm_model.predict_proba(data)
    # Use the maximum state probability per sample as a simple indicator
    max_probs = np.max(state_probs, axis=1)
    # Apply a discrete wavelet transform to the max probabilities
    coeffs = pywt.wavedec(max_probs, wavelet, level=level)
    # Use the approximation coefficients (first element) scaled by 'scale'
    approx = coeffs[0] * scale
    # Interpolate the approximated signal to match the original data length
    x_old = np.linspace(0, len(max_probs)-1, num=len(approx))
    x_new = np.arange(len(max_probs))
    werpi_full = np.interp(x_new, x_old, approx)
    return werpi_full.reshape(-1, 1)
