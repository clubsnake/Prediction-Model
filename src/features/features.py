# features.py
"""
Technical indicator calculation and feature engineering logic.
"""

from typing import Dict

import numpy as np
import pandas as pd
import pywt
from hmmlearn import hmm


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI with improved numerical stability."""
    df = df.copy()  # Make a copy of the input DataFrame

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()

    # Replace zeros with a small number to avoid division by zero
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_loss = avg_loss.replace(0, np.nan)

    # Calculate RS with handling for NaN values
    rs = avg_gain / avg_loss.fillna(0.00001)

    # Calculate RSI
    df["RSI"] = 100 - (100 / (1 + rs))

    # Handle edge cases
    df["RSI"] = df["RSI"].clip(0, 100)  # Ensure RSI is between 0 and 100

    return df


def calculate_emas(df: pd.DataFrame) -> Dict[int, pd.Series]:
    """Calculate and cache EMA series for different periods"""
    common_periods = [9, 12, 26, 50, 200]  # Common EMA periods
    emas = {}
    for period in common_periods:
        emas[period] = df["Close"].ewm(span=period, adjust=False).mean()
    return emas


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    emas: Dict[int, pd.Series] = None,
) -> pd.DataFrame:
    """Add MACD using precalculated EMAs if available"""
    df = df.copy()

    # Calculate EMAs if not provided
    if emas is None:
        emas = calculate_emas(df)

    # Use cached EMAs if available, otherwise calculate
    ema_fast = emas.get(fast, df["Close"].ewm(span=fast, adjust=False).mean())
    ema_slow = emas.get(slow, df["Close"].ewm(span=slow, adjust=False).mean())

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Add Bollinger Bands to df.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

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
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

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
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    close_series = df["Close"]
    volume_series = df["Volume"]
    df["Close"] = pd.to_numeric(close_series, errors="coerce")
    df["Volume"] = pd.to_numeric(volume_series, errors="coerce")
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df


def add_werpi_indicator(
    df: pd.DataFrame,
    wavelet_name: str = "db4",
    level: int = 3,
    n_states: int = 2,
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Add a wavelet + HMM-based indicator (WERPI) to df.
    """
    # Make a copy to avoid chained assignment warnings
    df = df.copy()

    # Calculate returns if not already present
    if "returns" not in df.columns:
        df.loc[:, "returns"] = df["Close"].pct_change()

    valid_returns = df["returns"].dropna()
    if len(valid_returns) < 2**level:  # Ensure enough data points
        df["WERPI"] = np.nan
        return df

    returns_array = valid_returns.values.reshape(-1, 1)

    try:
        # Initialize and fit the HMM with the specified number of states
        model_hmm = hmm.GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42
        )
        model_hmm.fit(returns_array)

        # Normalize the start probabilities to ensure they sum to 1
        if hasattr(model_hmm, "startprob_"):
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
        x_old = np.linspace(0, len(max_probs) - 1, num=len(approx))
        x_new = np.arange(len(max_probs))
        werpi_values = np.interp(x_new, x_old, approx)

        # Build an array of NaNs for the full DataFrame length and assign computed values
        werpi_full = np.full((len(df), 1), np.nan)
        werpi_full[valid_returns.index, 0] = werpi_values.flatten()
        df.loc[:, "WERPI"] = werpi_full

        # Fill NaN values
        df["WERPI"].fillna(method="ffill", inplace=True)
        df["WERPI"].fillna(method="bfill", inplace=True)

    except Exception as e:
        print(f"Error in WERPI calculation: {str(e)}")
        df.loc[:, "WERPI"] = np.nan

    return df


def add_vmli(
    df: pd.DataFrame,
    window_mom: int = 14,
    window_vol: int = 14,
    smooth_period: int = 3,
    winsorize_pct: float = 0.01,
    use_ema: bool = True,
    timeframe: str = "1d",  # Added timeframe parameter
) -> pd.DataFrame:
    """
    Add Volatility-Adjusted Momentum Liquidity Index (VMLI) to df.

    Args:
        df: DataFrame with OHLC and volume data
        window_mom: Lookback window for momentum calculation
        window_vol: Lookback window for volatility calculation
        smooth_period: Period for final smoothing of the indicator
        winsorize_pct: Percentile for winsorizing extreme values (0.01 = 1%)
        use_ema: Whether to use EMA (True) or SMA (False) for smoothing
        timeframe: Timeframe of the data for adaptive parameters

    Returns:
        DataFrame with VMLI indicators added
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    try:
        # Initialize VMLI indicator
        from src.features.vmli_indicator import VMILIndicator

        vmli = VMILIndicator(
            window_mom=window_mom,
            window_vol=window_vol,
            smooth_period=smooth_period,
            winsorize_pct=winsorize_pct,
            use_ema=use_ema,
        )

        # Adjust column names to match expected format in VMLI
        price_col = "Close"  # Your dataframe uses 'Close' instead of 'close'
        volume_col = "Volume"  # Your dataframe uses 'Volume' instead of 'volume'

        # Calculate VMLI with components, passing timeframe
        components = vmli.compute(
            data=df,
            price_col=price_col,
            volume_col=volume_col,
            include_components=True,
            timeframe=timeframe,  # Pass timeframe parameter
        )

        # Add all VMLI components to the dataframe
        df["VMLI"] = components["vmli"]
        df["VMLI_Momentum"] = components["momentum"]
        df["VMLI_Volatility"] = components["volatility"]
        df["VMLI_AdjMomentum"] = components["adj_momentum"]
        df["VMLI_Liquidity"] = components["liquidity"]
        df["VMLI_Raw"] = components["vmli_raw"]

    except Exception as e:
        print(f"Error calculating VMLI: {str(e)}")
        # Set default NaN values if calculation fails
        df["VMLI"] = np.nan
        df["VMLI_Momentum"] = np.nan
        df["VMLI_Volatility"] = np.nan
        df["VMLI_AdjMomentum"] = np.nan
        df["VMLI_Liquidity"] = np.nan
        df["VMLI_Raw"] = np.nan

    return df


def add_weekend_gap_feature(df: pd.DataFrame, apply_gap: bool = True) -> pd.DataFrame:
    """Optimized vectorized implementation"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

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
    apply_weekend_gap: bool = True,
    # VMLI parameters
    use_vmli: bool = True,
    vmli_window_mom: int = 14,
    vmli_window_vol: int = 14,
    vmli_smooth_period: int = 3,
    vmli_winsorize_pct: float = 0.01,
    vmli_use_ema: bool = True,
    # Other advanced indicator flags:
    use_keltner: bool = True,
    use_ichimoku: bool = True,
    use_fibonacci: bool = True,
    use_volatility: bool = True,
    use_momentum: bool = True,
    use_breakout: bool = True,
    use_deep_analytics: bool = True,
    # Add optuna_tuned_params parameter to allow passing in tuned parameters
    optuna_tuned_params: Dict = None,
) -> pd.DataFrame:
    """
    Apply multiple technical indicators to the DataFrame according
    to specified parameters, including advanced indicators.

    Args:
        ...existing args...
        optuna_tuned_params: Dictionary with Optuna-tuned parameters for indicators
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Override parameters with Optuna-tuned params if provided
    if optuna_tuned_params:
        if "werpi" in optuna_tuned_params:
            werpi_params = optuna_tuned_params["werpi"]["params"]
            werpi_wavelet = werpi_params.get("wavelet_name", werpi_wavelet)
            werpi_level = werpi_params.get("level", werpi_level)
            werpi_n_states = werpi_params.get("n_states", werpi_n_states)
            werpi_scale = werpi_params.get("scale_factor", werpi_scale)

        if "vmli" in optuna_tuned_params:
            vmli_params = optuna_tuned_params["vmli"]["params"]
            vmli_window_mom = vmli_params.get("window_mom", vmli_window_mom)
            vmli_window_vol = vmli_params.get("window_vol", vmli_window_vol)
            vmli_smooth_period = vmli_params.get("smooth_period", vmli_smooth_period)
            vmli_winsorize_pct = vmli_params.get("winsorize_pct", vmli_winsorize_pct)
            vmli_use_ema = vmli_params.get("use_ema", vmli_use_ema)

    df = add_rsi(df, period=rsi_period)
    df = add_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = add_bollinger_bands(df, window=boll_window, num_std=boll_nstd)
    df = add_atr(df, period=atr_period)
    df = add_obv(df)
    df = add_werpi_indicator(
        df,
        wavelet_name=werpi_wavelet,
        level=werpi_level,
        n_states=werpi_n_states,
        scale_factor=werpi_scale,
    )

    # Apply VMLI if flagged
    if use_vmli:
        df = add_vmli(
            df,
            window_mom=vmli_window_mom,
            window_vol=vmli_window_vol,
            smooth_period=vmli_smooth_period,
            winsorize_pct=vmli_winsorize_pct,
            use_ema=vmli_use_ema,
        )

    # Apply advanced indicators if flagged
    if use_keltner:
        df = add_keltner_channels(df)
    if use_ichimoku:
        df = add_ichimoku_cloud(df)
    if use_fibonacci:
        df = add_fibonacci_patterns(df)
    if use_volatility:
        df = add_volatility_indicators(df)
    if use_momentum:
        df = add_momentum_indicators(df)
    if use_breakout:
        df = add_breakout_indicators(df)
    if use_deep_analytics:
        df = add_deep_analytics(df)

    if ticker is not None:
        is_crypto = isinstance(ticker, str) and ticker.upper().endswith("-USD")
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


# Add a new function to load and apply Optuna-tuned parameters
def feature_engineering_with_tuned_params(df: pd.DataFrame, ticker: str = None):
    """
    Apply feature engineering using the best parameters found by Optuna tuning.

    Args:
        df: DataFrame with price data
        ticker: Ticker symbol

    Returns:
        DataFrame with technical indicators added
    """
    import json
    import os

    # Try to load tuned parameters
    tuned_params_file = os.path.join("Data", "tuned_indicators.json")
    optuna_tuned_params = None

    if os.path.exists(tuned_params_file):
        try:
            with open(tuned_params_file, "r") as f:
                optuna_tuned_params = json.load(f)
            print(f"Loaded tuned parameters: {optuna_tuned_params.keys()}")
        except Exception as e:
            print(f"Error loading tuned parameters: {e}")

    # Call regular feature engineering with tuned parameters
    return feature_engineering_with_params(
        df, ticker=ticker, optuna_tuned_params=optuna_tuned_params
    )


def feature_engineering(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Feature engineering entry point that reads default parameters from config.
    """
    from config.config_loader import ATR_PERIOD  # VMLI parameters
    from config.config_loader import (
        APPLY_WEEKEND_GAP,
        BOLL_NSTD,
        BOLL_WINDOW,
        MACD_FAST,
        MACD_SIGNAL,
        MACD_SLOW,
        RSI_PERIOD,
        USE_VMLI,
        VMLI_SMOOTH_PERIOD,
        VMLI_USE_EMA,
        VMLI_WINDOW_MOM,
        VMLI_WINDOW_VOL,
        VMLI_WINSORIZE_PCT,
        WERPI_LEVEL,
        WERPI_N_STATES,
        WERPI_SCALE,
        WERPI_WAVELET,
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
        apply_weekend_gap=APPLY_WEEKEND_GAP,
        # VMLI parameters
        use_vmli=USE_VMLI,
        vmli_window_mom=VMLI_WINDOW_MOM,
        vmli_window_vol=VMLI_WINDOW_VOL,
        vmli_smooth_period=VMLI_SMOOTH_PERIOD,
        vmli_winsorize_pct=VMLI_WINSORIZE_PCT,
        vmli_use_ema=VMLI_USE_EMA,
    )


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience method to apply all default indicators without custom params.
    Returns a new DataFrame with indicators added; does not modify the input DataFrame.
    """
    df = df.copy()  # Make a copy of the input DataFrame
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_werpi_indicator(df)
    return df


def compute_werpi(data, hmm_model, level=3, scale=1.0, wavelet="db4"):
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
    x_old = np.linspace(0, len(max_probs) - 1, num=len(approx))
    x_new = np.arange(len(max_probs))
    werpi_full = np.interp(x_new, x_old, approx)
    return werpi_full.reshape(-1, 1)


# Advanced Technical Indicators
import pandas as pd


def add_keltner_channels(
    df: pd.DataFrame, ema_period: int = 20, atr_multiplier: float = 2.0, trial=None
) -> pd.DataFrame:
    """
    Add Keltner Channels with potentially Optuna-tuned parameters.
    """
    if trial is not None:
        ema_period = trial.suggest_int("keltner_ema_period", 5, 30)
        atr_multiplier = trial.suggest_float("keltner_atr_multiplier", 1.0, 3.0)

    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    middle_line = typical_price.ewm(span=ema_period, adjust=False).mean()

    # Calculate ATR if not already added
    if "ATR" not in df.columns:
        df = add_atr(df, period=ema_period)
    atr = df["ATR"]

    df["Keltner_Middle"] = middle_line
    df["Keltner_Upper"] = middle_line + (atr_multiplier * atr)
    df["Keltner_Lower"] = middle_line - (atr_multiplier * atr)
    return df


def add_ichimoku_cloud(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
    trial=None,
) -> pd.DataFrame:
    """
    Add Ichimoku Cloud with potentially Optuna-tuned parameters.
    """
    if trial is not None:
        tenkan_period = trial.suggest_int("ichimoku_tenkan", 5, 20)
        kijun_period = trial.suggest_int("ichimoku_kijun", 15, 40)
        senkou_span_b_period = trial.suggest_int("ichimoku_senkou_b", 30, 80)
        displacement = trial.suggest_int("ichimoku_displacement", 13, 39)

    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    high_tenkan = df["High"].rolling(window=tenkan_period).max()
    low_tenkan = df["Low"].rolling(window=tenkan_period).min()
    df["Ichimoku_Tenkan_Sen"] = (high_tenkan + low_tenkan) / 2

    high_kijun = df["High"].rolling(window=kijun_period).max()
    low_kijun = df["Low"].rolling(window=kijun_period).min()
    df["Ichimoku_Kijun_Sen"] = (high_kijun + low_kijun) / 2

    df["Ichimoku_Senkou_Span_A"] = (
        (df["Ichimoku_Tenkan_Sen"] + df["Ichimoku_Kijun_Sen"]) / 2
    ).shift(displacement)

    high_senkou_b = df["High"].rolling(window=senkou_span_b_period).max()
    low_senkou_b = df["Low"].rolling(window=senkou_span_b_period).min()
    df["Ichimoku_Senkou_Span_B"] = ((high_senkou_b + low_senkou_b) / 2).shift(
        displacement
    )

    df["Ichimoku_Chikou_Span"] = df["Close"].shift(-displacement)

    return df


def add_fibonacci_patterns(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """
    Add Fibonacci-based retracement and extension levels relative to recent price action.

    Args:
        df: DataFrame with OHLC data
        lookback: Period to look back for highs and lows

    Returns:
        DataFrame with additional Fibonacci pattern columns
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Loop over the DataFrame starting at lookback
    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]
        highest_high = window["High"].max()
        lowest_low = window["Low"].min()
        range_size = highest_high - lowest_low
        retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 1.0]

        for level in retracement_levels:
            level_down = f"Fib_Down_{int(level*100)}"
            level_up = f"Fib_Up_{int(level*100)}"
            if level_down not in df.columns:
                df[level_down] = np.nan
            if level_up not in df.columns:
                df[level_up] = np.nan
            df.at[df.index[i], level_down] = highest_high - (range_size * level)
            df.at[df.index[i], level_up] = lowest_low + (range_size * level)

    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add various volatility indicators to the DataFrame.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with additional volatility indicators
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    df["returns"] = df["Close"].pct_change()
    df["HV_10"] = df["returns"].rolling(window=10).std() * np.sqrt(252)
    df["HV_20"] = df["returns"].rolling(window=20).std() * np.sqrt(252)
    df["HV_30"] = df["returns"].rolling(window=30).std() * np.sqrt(252)

    if "ATR" not in df.columns:
        df = add_atr(df)
    df["NATR"] = df["ATR"] / df["Close"] * 100
    df["Volatility_Ratio"] = df["HV_10"] / df["HV_30"]

    if not all(
        col in df.columns
        for col in ["Bollinger_Upper", "Bollinger_Lower", "Bollinger_Mid"]
    ):
        df = add_bollinger_bands(df)
    df["BB_Width"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df[
        "Bollinger_Mid"
    ]

    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced momentum indicators to the DataFrame.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with additional momentum indicators
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    df["ROC_5"] = df["Close"].pct_change(periods=5) * 100
    df["ROC_10"] = df["Close"].pct_change(periods=10) * 100
    df["ROC_20"] = df["Close"].pct_change(periods=20) * 100

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]
    delta = typical_price.diff()
    positive_flow = pd.Series(np.where(delta > 0, money_flow, 0), index=df.index)
    negative_flow = pd.Series(np.where(delta < 0, money_flow, 0), index=df.index)

    for period in [14, 21]:
        pos_flow_sum = positive_flow.rolling(window=period).sum()
        neg_flow_sum = negative_flow.rolling(window=period).sum()
        mf_ratio = np.where(neg_flow_sum != 0, pos_flow_sum / neg_flow_sum, 1)
        df[f"MFI_{period}"] = 100 - (100 / (1 + mf_ratio))

    for period in [14, 21]:
        low_min = df["Low"].rolling(window=period).min()
        high_max = df["High"].rolling(window=period).max()
        df[f"Stoch_{period}_K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
        df[f"Stoch_{period}_D"] = df[f"Stoch_{period}_K"].rolling(window=3).mean()

    return df


def add_breakout_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add breakout detection indicators to the DataFrame.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with additional breakout indicators
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    for period in [20, 50]:
        df[f"Donchian_High_{period}"] = df["High"].rolling(window=period).max()
        df[f"Donchian_Low_{period}"] = df["Low"].rolling(window=period).min()
        df[f"Donchian_Mid_{period}"] = (
            df[f"Donchian_High_{period}"] + df[f"Donchian_Low_{period}"]
        ) / 2

    for ma_period in [20, 50, 200]:
        ma_col = f"MA_{ma_period}"
        if ma_col not in df.columns:
            df[ma_col] = df["Close"].rolling(window=ma_period).mean()
        df[f"Distance_MA_{ma_period}"] = (df["Close"] - df[ma_col]) / df[ma_col] * 100

    # Fix volume ratio calculation
    vol_ma_20 = df["Volume"].rolling(window=20).mean()
    df["Volume_MA_20"] = vol_ma_20
    # Ensure we're dividing Series by Series
    df["Volume_Ratio"] = df["Volume"].div(vol_ma_20)

    df["Range"] = df["High"] - df["Low"]
    df["Avg_Range_20"] = df["Range"].rolling(window=20).mean()
    df["Range_Expansion"] = df["Range"] / df["Avg_Range_20"]

    donchian_break = (
        (df["Close"] > df["Donchian_High_20"].shift(1))
        | (df["Close"] < df["Donchian_Low_20"].shift(1))
    ).astype(float)
    volume_expansion = (df["Volume_Ratio"] > 1.5).astype(float)
    range_expansion = (df["Range_Expansion"] > 1.5).astype(float)

    df["Breakout_Score"] = (
        donchian_break * 0.5 + volume_expansion * 0.3 + range_expansion * 0.2
    )
    return df


def add_deep_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deep analytics features combining multiple indicators and statistical measures.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with additional deep analytics
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    if "RSI" not in df.columns:
        df = add_rsi(df)
    if "MACD" not in df.columns:
        df = add_macd(df)

    df["Close_Mean_20"] = df["Close"].rolling(window=20).mean()
    df["Close_Std_20"] = df["Close"].rolling(window=20).std()
    df["Close_Z_Score"] = (df["Close"] - df["Close_Mean_20"]) / df["Close_Std_20"]

    df["Volume_Mean_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Std_20"] = df["Volume"].rolling(window=20).std()
    df["Volume_Z_Score"] = (df["Volume"] - df["Volume_Mean_20"]) / df["Volume_Std_20"]

    if "RSI" in df.columns and "MACD" in df.columns:
        normalized_rsi = (df["RSI"] - 50) / 50
        macd_std = df["MACD"].rolling(window=20).std()
        normalized_macd = df["MACD"] / (macd_std + 1e-9)
        df["Combined_Momentum"] = (normalized_rsi + normalized_macd) / 2

    if "ATR" in df.columns and "RSI" in df.columns:
        df["Volatility_Adj_Momentum"] = (df["RSI"] - 50) / (
            df["ATR"] / df["Close"] * 100
        )

    for period in [20, 50]:
        ma_col = f"MA_{period}"
        if ma_col not in df.columns:
            df[ma_col] = df["Close"].rolling(window=period).mean()
        df[f"Trend_Strength_{period}"] = (
            abs(df["Close"] - df[ma_col]) / (df["ATR"] + 1e-9) * 100
        )

    return df


# Add a function for Optuna-tuned feature engineering
def feature_engineering_with_optuna_params(df, trial=None, ticker=None):
    """
    Apply feature engineering using parameters suggested by an Optuna trial.
    If trial is None, uses default parameters.

    Args:
        df: DataFrame with price data
        trial: Optuna trial object for hyperparameter suggestion
        ticker: Ticker symbol

    Returns:
        DataFrame with technical indicators added
    """
    if trial is None:
        # If no trial is provided, use default parameters
        return feature_engineering(df, ticker)

    # Suggest parameters via Optuna
    rsi_period = trial.suggest_int("rsi_period", 5, 30)
    macd_fast = trial.suggest_int("macd_fast", 5, 20)
    macd_slow = trial.suggest_int("macd_slow", 15, 40)
    macd_signal = trial.suggest_int("macd_signal", 5, 15)
    boll_window = trial.suggest_int("boll_window", 5, 30)
    boll_nstd = trial.suggest_float("boll_nstd", 1.0, 3.0)
    atr_period = trial.suggest_int("atr_period", 5, 30)

    # Wavelet parameters
    werpi_wavelet = trial.suggest_categorical(
        "werpi_wavelet", ["db1", "db2", "db4", "sym2", "sym4", "haar"]
    )
    werpi_level = trial.suggest_int("werpi_level", 1, 5)
    werpi_n_states = trial.suggest_int("werpi_n_states", 2, 5)
    werpi_scale = trial.suggest_float("werpi_scale", 0.5, 2.0)

    # VMLI parameters
    vmli_window_mom = trial.suggest_int("vmli_window_mom", 5, 30)
    vmli_window_vol = trial.suggest_int("vmli_window_vol", 5, 30)
    vmli_smooth_period = trial.suggest_int("vmli_smooth_period", 1, 10)
    vmli_winsorize_pct = trial.suggest_float("vmli_winsorize_pct", 0.001, 0.05)
    vmli_use_ema = trial.suggest_categorical("vmli_use_ema", [True, False])

    # Which indicators to include
    use_keltner = trial.suggest_categorical("use_keltner", [True, False])
    use_ichimoku = trial.suggest_categorical("use_ichimoku", [True, False])
    use_fibonacci = trial.suggest_categorical("use_fibonacci", [True, False])
    use_volatility = trial.suggest_categorical("use_volatility", [True, False])
    use_momentum = trial.suggest_categorical("use_momentum", [True, False])
    use_breakout = trial.suggest_categorical("use_breakout", [True, False])
    use_deep_analytics = trial.suggest_categorical("use_deep_analytics", [True, False])

    # Apply parameters to feature engineering
    return feature_engineering_with_params(
        df,
        ticker=ticker,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        boll_window=boll_window,
        boll_nstd=boll_nstd,
        atr_period=atr_period,
        werpi_wavelet=werpi_wavelet,
        werpi_level=werpi_level,
        werpi_n_states=werpi_n_states,
        werpi_scale=werpi_scale,
        use_vmli=True,
        vmli_window_mom=vmli_window_mom,
        vmli_window_vol=vmli_window_vol,
        vmli_smooth_period=vmli_smooth_period,
        vmli_winsorize_pct=vmli_winsorize_pct,
        vmli_use_ema=vmli_use_ema,
        use_keltner=use_keltner,
        use_ichimoku=use_ichimoku,
        use_fibonacci=use_fibonacci,
        use_volatility=use_volatility,
        use_momentum=use_momentum,
        use_breakout=use_breakout,
        use_deep_analytics=use_deep_analytics,
    )


import logging
from typing import Dict

import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Import necessary indicator functions
from src.features.indicator_tuning import add_werpi_indicator


def calculate_emas(df: pd.DataFrame) -> Dict[int, pd.Series]:
    """Calculate multiple EMAs once to avoid redundant computation"""
    emas = {}
    for period in [5, 8, 12, 13, 21, 26, 34, 55, 89, 144]:
        emas[period] = df["Close"].ewm(span=period, adjust=False).mean()
    return emas


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index"""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, emas=None
) -> pd.DataFrame:
    """Add Moving Average Convergence Divergence"""
    if emas is not None and fast in emas and slow in emas:
        # Use pre-calculated EMAs if available
        fast_ema = emas[fast]
        slow_ema = emas[slow]
    else:
        fast_ema = df["Close"].ewm(span=fast, adjust=False).mean()
        slow_ema = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = fast_ema - slow_ema
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Add Bollinger Bands"""
    df["BB_Mid"] = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = df["BB_Mid"] + (rolling_std * num_std)
    df["BB_Lower"] = df["BB_Mid"] - (rolling_std * num_std)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range"""
    high_low = df["High"] - df["Low"]
    high_close_prev = abs(df["High"] - df["Close"].shift(1))
    low_close_prev = abs(df["Low"] - df["Close"].shift(1))

    tr = pd.DataFrame(
        {"hl": high_low, "hcp": high_close_prev, "lcp": low_close_prev}
    ).max(axis=1)
    df["ATR"] = tr.rolling(window=period).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume"""
    df["OBV"] = np.where(
        df["Close"] > df["Close"].shift(1),
        df["Volume"],
        np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0),
    ).cumsum()
    return df


def add_weekend_gap_feature(df: pd.DataFrame, apply_gap: bool = True) -> pd.DataFrame:
    """Add weekend gap features"""
    if "date" not in df.columns:
        return df

    df["DayOfWeek"] = df["date"].dt.dayofweek
    df["WeekendGap"] = 0.0

    if apply_gap:
        # Monday has dayofweek = 0, check for gap from previous trading day
        monday_mask = df["DayOfWeek"] == 0
        df.loc[monday_mask, "WeekendGap"] = (
            df.loc[monday_mask, "Open"] / df.loc[monday_mask, "Close"].shift(1) - 1
        )

    return df


# Placeholder for more advanced indicator functions
def add_keltner_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Add Keltner Channels"""
    # Implementation here
    return df


def add_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    """Add Ichimoku Cloud"""
    # Implementation here
    return df


def add_fibonacci_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fibonacci pattern detection"""
    # Implementation here
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add various volatility indicators"""
    # Implementation here
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add various momentum indicators"""
    # Implementation here
    return df


def add_breakout_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add breakout detection indicators"""
    # Implementation here
    return df


def add_deep_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Add more complex indicators based on deep analytics"""
    # Implementation here
    return df


def feature_engineering_with_params(
    df: pd.DataFrame,
    ticker: str = None,
    timeframe: str = "1d",  # Added explicit timeframe parameter
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
    apply_weekend_gap: bool = True,
    # VMLI parameters
    use_vmli: bool = True,
    vmli_window_mom: int = 14,
    vmli_window_vol: int = 14,
    vmli_smooth_period: int = 3,
    vmli_winsorize_pct: float = 0.01,
    vmli_use_ema: bool = True,
    # Advanced indicator flags
    use_advanced_indicators: bool = True,  # Master switch for all advanced indicators
    # Other advanced indicator flags:
    use_keltner: bool = True,
    use_ichimoku: bool = True,
    use_fibonacci: bool = True,
    use_volatility: bool = True,
    use_momentum: bool = True,
    use_breakout: bool = True,
    use_deep_analytics: bool = True,
    # Add optuna_tuned_params parameter to allow passing in tuned parameters
    optuna_tuned_params: Dict = None,
) -> pd.DataFrame:
    """
    Apply multiple technical indicators to the DataFrame according
    to specified parameters, including advanced indicators.

    With improved timeframe adaptation and memory efficiency.
    """
    # Make a copy only of required columns to conserve memory
    required_cols = ["date", "Open", "High", "Low", "Close", "Volume"]
    df_cols = [col for col in required_cols if col in df.columns]
    result_df = df[df_cols].copy()

    if "date" in result_df.columns:
        result_df["date"] = pd.to_datetime(result_df["date"])

    # Override parameters with Optuna-tuned params if provided
    if optuna_tuned_params:
        if "werpi" in optuna_tuned_params:
            werpi_params = optuna_tuned_params["werpi"]["params"]
            werpi_wavelet = werpi_params.get("wavelet_name", werpi_wavelet)
            werpi_level = werpi_params.get("level", werpi_level)
            werpi_n_states = werpi_params.get("n_states", werpi_n_states)
            werpi_scale = werpi_params.get("scale_factor", werpi_scale)

        if "vmli" in optuna_tuned_params:
            vmli_params = optuna_tuned_params["vmli"]["params"]
            vmli_window_mom = vmli_params.get("window_mom", vmli_window_mom)
            vmli_window_vol = vmli_params.get("window_vol", vmli_window_vol)
            vmli_smooth_period = vmli_params.get("smooth_period", vmli_smooth_period)
            vmli_winsorize_pct = vmli_params.get("winsorize_pct", vmli_winsorize_pct)
            vmli_use_ema = vmli_params.get("use_ema", vmli_use_ema)

    # Adjust indicator parameters based on timeframe
    if timeframe != "1d":
        # Scale periods for intraday data
        if timeframe in ["1h", "2h", "4h"]:
            scale_factor = 0.25
        elif timeframe in ["15m", "30m"]:
            scale_factor = 0.1
        elif timeframe in ["1m", "5m"]:
            scale_factor = 0.05
        else:
            scale_factor = 1.0

        # Apply scaling to period parameters
        rsi_period = max(5, int(rsi_period * scale_factor))
        macd_fast = max(3, int(macd_fast * scale_factor))
        macd_slow = max(5, int(macd_slow * scale_factor))
        macd_signal = max(3, int(macd_signal * scale_factor))
        boll_window = max(5, int(boll_window * scale_factor))
        atr_period = max(5, int(atr_period * scale_factor))

    # Pre-calculate common values to avoid redundant computation
    emas = calculate_emas(result_df)  # Calculate EMAs once

    # Add basic indicators (always included)
    result_df = add_rsi(result_df, period=rsi_period)
    result_df = add_macd(
        result_df, fast=macd_fast, slow=macd_slow, signal=macd_signal, emas=emas
    )
    result_df = add_bollinger_bands(result_df, window=boll_window, num_std=boll_nstd)
    result_df = add_atr(result_df, period=atr_period)
    result_df = add_obv(result_df)

    # Add WERPI (always included as it's a core custom indicator)
    result_df = add_werpi_indicator(
        result_df,
        wavelet_name=werpi_wavelet,
        level=werpi_level,
        n_states=werpi_n_states,
        scale_factor=werpi_scale,
    )

    # Apply VMLI if flagged
    if use_vmli:
        try:
            from src.features.vmli_indicator import VMILIndicator

            vmli = VMILIndicator(
                window_mom=vmli_window_mom,
                window_vol=vmli_window_vol,
                smooth_period=vmli_smooth_period,
                winsorize_pct=vmli_winsorize_pct,
                use_ema=vmli_use_ema,
            )

            # Calculate VMLI with timeframe awareness
            components = vmli.compute(
                data=result_df,
                price_col="Close",
                volume_col="Volume" if "Volume" in result_df.columns else None,
                include_components=True,
                timeframe=timeframe,  # Pass timeframe for auto-adjustment
            )

            # Add VMLI components to DataFrame
            result_df["VMLI"] = components["vmli"]
            result_df["VMLI_Momentum"] = components["momentum"]
            result_df["VMLI_Volatility"] = components["volatility"]
            result_df["VMLI_AdjMomentum"] = components["adj_momentum"]
            result_df["VMLI_Liquidity"] = components["liquidity"]
            result_df["VMLI_Raw"] = components["vmli_raw"]

        except Exception as e:
            logger.error(f"Error calculating VMLI: {str(e)}")
            # Set default NaN values if calculation fails
            result_df["VMLI"] = np.nan
            result_df["VMLI_Momentum"] = np.nan
            result_df["VMLI_Volatility"] = np.nan
            result_df["VMLI_AdjMomentum"] = np.nan
            result_df["VMLI_Liquidity"] = np.nan
            result_df["VMLI_Raw"] = np.nan

    # Apply advanced indicators if flagged and master switch is on
    if use_advanced_indicators:
        if use_keltner:
            result_df = add_keltner_channels(result_df)
        if use_ichimoku:
            result_df = add_ichimoku_cloud(result_df)
        if use_fibonacci:
            result_df = add_fibonacci_patterns(result_df)
        if use_volatility:
            result_df = add_volatility_indicators(result_df)
        if use_momentum:
            result_df = add_momentum_indicators(result_df)
        if use_breakout:
            result_df = add_breakout_indicators(result_df)
        if use_deep_analytics:
            result_df = add_deep_analytics(result_df)

    # Add gap features
    if ticker is not None:
        is_crypto = isinstance(ticker, str) and ticker.upper().endswith("-USD")
        result_df["WeekendGap"] = 0.0
        if not is_crypto and apply_weekend_gap:
            result_df = add_weekend_gap_feature(result_df, apply_gap=True)
    else:
        result_df["WeekendGap"] = 0.0

    # Add return and volatility features
    result_df["Returns"] = result_df["Close"].pct_change()
    result_df["Volatility"] = result_df["Returns"].rolling(window=10).std()

    # Clean up and return
    result_df.dropna(inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    return result_df
