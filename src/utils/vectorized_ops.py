"""
Provides vectorized operations for high-performance data processing.
For very large datasets, these functions offer significant speed improvements.
"""

import warnings
import pandas as pd
import numpy as np
from numba import jit, njit
import gc

def vectorized_sequence_creation(df, feature_cols, target_col, lookback, horizon):
    """
    Vectorized implementation of sequence creation for time series data.
    Much faster than loop-based approaches for large datasets.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        lookback: Number of lookback steps
        horizon: Number of steps to predict

    Returns:
        X, y numpy arrays
    """
    total_samples = len(df)
    if (total_samples <= lookback):
        return np.array([]), np.array([])

    num_features = len(feature_cols)
    valid_samples = total_samples - lookback - horizon + 1

    if (valid_samples <= 0):
        return np.array([]), np.array([])

    # Pre-allocate arrays (major performance gain for large datasets)
    X = np.zeros((valid_samples, lookback, num_features))
    y = np.zeros((valid_samples, horizon))

    # Extract data once (avoid repeated indexing)
    feature_data = df[feature_cols].values
    target_data = df[target_col].values

    # Create sequences using efficient slicing
    for i in range(valid_samples):
        X[i] = feature_data[i : i + lookback]
        y[i] = target_data[i + lookback : i + lookback + horizon]

    return X, y


@jit(nopython=True)
def numba_mse(predictions, actuals):
    """
    Numba-accelerated MSE calculation.
    Significantly faster for large arrays.

    Args:
        predictions: Numpy array of predictions
        actuals: Numpy array of actual values

    Returns:
        Mean squared error value
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return float("nan")

    squared_errors = np.square(predictions - actuals)
    return np.mean(squared_errors)


@njit
def fast_sequence_creation(feature_data, target_data, lookback, horizon):
    """
    Numba no-Python accelerated sequence creation working directly with numpy arrays.
    Faster than other implementations for very large datasets.

    Args:
        feature_data: Numpy array of feature values (shape: [samples, features])
        target_data: Numpy array of target values (shape: [samples])
        lookback: Number of lookback steps
        horizon: Number of steps to predict

    Returns:
        X, y numpy arrays
    """
    valid_samples = len(feature_data) - lookback - horizon + 1
    X = np.zeros((valid_samples, lookback, feature_data.shape[1]))
    y = np.zeros((valid_samples, horizon))
    
    for i in range(valid_samples):
        X[i] = feature_data[i : i + lookback]
        y[i] = target_data[i + lookback : i + lookback + horizon]
    
    return X, y


def batch_sequence_creation(
    df, feature_cols, target_col, lookback, horizon, batch_size=10000
):
    """
    Memory-efficient batched creation of sequences for very large datasets.
    Processes data in chunks to avoid memory issues.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        lookback: Number of lookback steps
        horizon: Number of steps to predict
        batch_size: Number of sequences to create per batch

    Returns:
        X, y numpy arrays
    """
    total_samples = len(df)
    valid_samples = total_samples - lookback - horizon + 1

    if valid_samples <= 0:
        return np.array([]), np.array([])

    num_features = len(feature_cols)
    X_batches = []
    y_batches = []

    # Process in batches
    for start in range(0, valid_samples, batch_size):
        end = min(start + batch_size, valid_samples)
        batch_size_actual = end - start

        X_batch = np.zeros((batch_size_actual, lookback, num_features))
        y_batch = np.zeros((batch_size_actual, horizon))

        feature_data = df[feature_cols].values
        target_data = df[target_col].values

        for i in range(batch_size_actual):
            idx = start + i
            X_batch[i] = feature_data[idx : idx + lookback]
            y_batch[i] = target_data[idx + lookback : idx + lookback + horizon]

        X_batches.append(X_batch)
        y_batches.append(y_batch)

    # Combine batches
    X = np.vstack(X_batches) if X_batches else np.array([])
    y = np.vstack(y_batches) if y_batches else np.array([])

    return X, y


def preserve_date_column(df, scaled_df, date_col="date"):
    """
    Ensures the date column is preserved after scaling/preprocessing.

    Args:
        df: Original DataFrame with date column
        scaled_df: Scaled DataFrame that might be missing date column
        date_col: Name of date column

    Returns:
        DataFrame with date column restored
    """
    if date_col in df.columns and date_col not in scaled_df.columns:
        # Ensure index alignment
        if len(df) == len(scaled_df):
            scaled_df[date_col] = df[date_col].copy()
        else:
            warnings.warn(f"DataFrame lengths don't match, cannot restore {date_col}")

    return scaled_df
