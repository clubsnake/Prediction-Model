# preprocessing.py
"""
Data preprocessing utilities with added type hints.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import configuration values
try:
    from config.config_loader import LOOKBACK, PREDICTION_HORIZON  # Corrected import path
except ImportError:
    # Fallback defaults if config cannot be imported
    LOOKBACK = 30
    PREDICTION_HORIZON = 5
    print("Warning: Could not import config values. Using defaults: "
          f"LOOKBACK={LOOKBACK}, PREDICTION_HORIZON={PREDICTION_HORIZON}")


def scale_data(
    df: pd.DataFrame, feature_cols: List[str], target_col: str = "Close"
) -> Tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """
    Scale feature columns and the target column using MinMaxScaler.

    :param df: DataFrame containing at least feature_cols and target_col.
    :param feature_cols: List of columns to scale as features.
    :param target_col: Name of the target column (e.g. 'Close').
    :return: (scaled_df, feature_scaler, target_scaler)
    :raises ValueError: If specified columns don't exist in DataFrame
    """
    # Validate input data
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X = df[feature_cols].values
    y = df[[target_col]].values
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    scaled_df[target_col] = y_scaled.flatten()
    
    # Only add date column if it exists in the original DataFrame
    if "date" in df.columns:
        scaled_df["date"] = df["date"]
    
    return scaled_df, feature_scaler, target_scaler


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Close",
    lookback: int = LOOKBACK,
    horizon: int = PREDICTION_HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time-series sequences for supervised learning.

    :param df: A scaled DataFrame that includes feature_cols and target_col.
    :param feature_cols: Columns used as model features.
    :param target_col: Column used as target variable.
    :param lookback: Number of timesteps to use as input sequence.
    :param horizon: Number of timesteps to predict into the future.
    :return: Tuple of (X, y) where X.shape=(samples, lookback, features),
             and y.shape=(samples, horizon).
    :raises ValueError: If DataFrame length is insufficient for sequence creation
    """
    # Validate DataFrame has sufficient length
    min_required_length = lookback + horizon
    if len(df) < min_required_length:
        raise ValueError(
            f"DataFrame length ({len(df)}) is less than required minimum "
            f"({min_required_length}) for lookback={lookback} and horizon={horizon}"
        )
    
    # Validate all required columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")
    
    df = df.reset_index(drop=True)
    X, y = [], []
    for i in range(len(df) - lookback - horizon + 1):
        X_window = df.loc[i : i + lookback - 1, feature_cols].values
        y_window = df.loc[i + lookback : i + lookback + horizon - 1, target_col].values
        X.append(X_window)
        y.append(y_window)
    
    return np.array(X), np.array(y)
