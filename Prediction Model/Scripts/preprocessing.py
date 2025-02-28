# preprocessing.py
"""
Scaling and sequence generation utilities for time-series.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import LOOKBACK, PREDICTION_HORIZON

def scale_data(df: pd.DataFrame, feature_cols: list, target_col: str = "Close"):
    """
    Scale feature columns and the target column using MinMaxScaler.
    
    :param df: DataFrame containing at least feature_cols and target_col.
    :param feature_cols: List of columns to scale as features.
    :param target_col: Name of the target column (e.g. 'Close').
    :return: (scaled_df, feature_scaler, target_scaler)
    """
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X = df[feature_cols].values
    y = df[[target_col]].values
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    scaled_df[target_col] = y_scaled.flatten()
    scaled_df["date"] = df["date"]
    return scaled_df, feature_scaler, target_scaler

def create_sequences(df: pd.DataFrame, feature_cols: list, 
                     target_col: str = "Close", 
                     lookback: int = LOOKBACK, 
                     horizon: int = PREDICTION_HORIZON):
    """
    Create time-series sequences for supervised learning.

    :param df: A scaled DataFrame that includes feature_cols and target_col.
    :param feature_cols: Columns used as model features.
    :param target_col: Column used as target variable.
    :param lookback: Number of timesteps to use as input sequence.
    :param horizon: Number of timesteps to predict into the future.
    :return: Tuple of (X, y) where X.shape=(samples, lookback, features), 
             and y.shape=(samples, horizon).
    """
    df = df.reset_index(drop=True)
    X, y = [], []
    for i in range(len(df) - lookback - horizon + 1):
        X_window = df.loc[i:i+lookback-1, feature_cols].values
        y_window = df.loc[i+lookback : i+lookback+horizon-1, target_col].values
        X.append(X_window)
        y.append(y_window)
    return np.array(X), np.array(y)
