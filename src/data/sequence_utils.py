"""
Optimized utilities for sequence creation and metric calculation.

This module provides functions for creating sequences from time series data
and calculating metrics like MSE and MAPE with optimized performance.
"""

import numpy as np

# Try to import numba for performance optimization
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    # Define a passthrough decorator if numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    NUMBA_AVAILABLE = False
    print(
        "Warning: Numba not available. Using fallback implementations for optimized functions."
    )


@njit
def numba_mse(y_true, y_pred):
    """
    Calculate MSE with Numba optimization.

    Args:
        y_true: Ground truth values as numpy array
        y_pred: Predicted values as numpy array

    Returns:
        Mean squared error value
    """
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    filtered_y_true = y_true[mask]
    filtered_y_pred = y_pred[mask]

    if len(filtered_y_true) == 0:
        return np.nan

    squared_errors = (filtered_y_true - filtered_y_pred) ** 2
    return np.mean(squared_errors)


@njit
def numba_mape(y_true, y_pred):
    """
    Calculate MAPE with Numba optimization.

    Args:
        y_true: Ground truth values as numpy array
        y_pred: Predicted values as numpy array

    Returns:
        Mean absolute percentage error value
    """
    # Handle NaN and zero values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | (np.abs(y_true) < 1e-10))
    filtered_y_true = y_true[mask]
    filtered_y_pred = y_pred[mask]

    if len(filtered_y_true) == 0:
        return np.nan

    percentage_errors = (
        np.abs((filtered_y_true - filtered_y_pred) / filtered_y_true) * 100
    )
    return np.mean(percentage_errors)


def vectorized_sequence_creation(data, lookback, horizon, step=1, target_col=None):
    """
    Create sequences for time series forecasting using vectorized operations.

    Args:
        data: Input data array/matrix or DataFrame
        lookback: Number of timesteps to look back
        horizon: Number of timesteps to predict forward
        step: Step size between sequences
        target_col: Column name or index to use as target (if data is DataFrame)

    Returns:
        X: Input sequences
        y: Target sequences
    """
    # Handle DataFrame input
    if hasattr(data, "to_numpy"):
        if target_col is not None:
            y_data = data[target_col].to_numpy()
        else:
            y_data = data.iloc[:, 0].to_numpy()  # Default to first column
        data = data.to_numpy()
    else:
        # If data is already numpy array
        if isinstance(data, np.ndarray):
            y_data = data[:, 0] if data.ndim > 1 else data  # Default to first column
        else:
            # Convert to numpy array if not already
            data = np.array(data)
            y_data = data[:, 0] if data.ndim > 1 else data

    length = len(data)

    # Calculate valid indices
    valid_indices = range(lookback, length - horizon + 1, step)
    num_sequences = len(valid_indices)

    # If data is 1D, reshape to 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Get data dimensions
    _, num_features = data.shape

    # Initialize arrays
    X = np.zeros((num_sequences, lookback, num_features))
    y = np.zeros((num_sequences, horizon, 1))  # Assuming single target variable

    # Fill arrays
    for i, idx in enumerate(valid_indices):
        X[i] = data[idx - lookback : idx]
        # Use specified target data
        if y_data.ndim == 1:
            y[i, :, 0] = y_data[idx : idx + horizon]
        else:
            y[i, :, 0] = y_data[idx : idx + horizon, 0]

    return X, y


def batch_sequence_creation(data, lookback, horizon, batch_size=1000, target_col=None):
    """
    Create sequences in batches for large datasets to conserve memory.

    Args:
        data: Input data (array/matrix/DataFrame)
        lookback: Number of timesteps to look back
        horizon: Number of timesteps to predict forward
        batch_size: How many sequences to process at once
        target_col: Column name or index to use as target (if data is DataFrame)

    Returns:
        X: Input sequences
        y: Target sequences
    """
    if hasattr(data, "shape"):
        length = data.shape[0]
    else:
        length = len(data)

    # Calculate number of valid sequences
    num_sequences = length - lookback - horizon + 1

    # If the dataset is small enough, use vectorized approach
    if num_sequences <= batch_size:
        return vectorized_sequence_creation(
            data, lookback, horizon, target_col=target_col
        )

    # Process in batches
    X_batches = []
    y_batches = []

    # Process batches of sequences
    for start_idx in range(0, num_sequences, batch_size):
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_length = end_idx - start_idx

        # Get the relevant portion of data for this batch
        if hasattr(data, "iloc"):
            # For DataFrame
            batch_data = data.iloc[
                start_idx : start_idx + batch_length + lookback + horizon - 1
            ]
        else:
            # For numpy arrays
            batch_data = data[
                start_idx : start_idx + batch_length + lookback + horizon - 1
            ]

        # Create sequences for this batch
        X_batch, y_batch = vectorized_sequence_creation(
            batch_data, lookback, horizon, target_col=target_col
        )

        X_batches.append(X_batch)
        y_batches.append(y_batch)

    # Combine batches
    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)

    return X, y


# Add additional helper function for sequence creation from DataFrames
def create_sequences(df, feature_cols, target_col, lookback, horizon, step=1):
    """
    Create sequences from a DataFrame with specific feature and target columns.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        lookback: Number of timesteps to look back
        horizon: Number of timesteps to predict forward
        step: Step size between sequences

    Returns:
        X: Input sequences
        y: Target sequences
    """
    # Extract features and target
    features = df[feature_cols].values
    target = df[target_col].values.reshape(-1, 1)  # Ensure 2D

    # Get dimensions
    n_samples = len(df)
    n_features = len(feature_cols)

    # Initialize arrays
    X = []
    y = []

    # Create sequences
    for i in range(lookback, n_samples - horizon + 1, step):
        X.append(features[i - lookback : i])
        y.append(target[i : i + horizon])

    return np.array(X), np.array(y)


# Functions to evaluate model performance optimized with numba
@njit
def numba_direction_accuracy(y_true, y_pred):
    """
    Calculate direction accuracy with Numba optimization.

    Args:
        y_true: Ground truth values as numpy array
        y_pred: Predicted values as numpy array

    Returns:
        Direction accuracy as percentage
    """
    if len(y_true) <= 1 or len(y_pred) <= 1:
        return np.nan

    # Calculate directions (positive = up, negative = down)
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0

    # Count matches
    matches = true_direction == pred_direction

    # Return percentage
    return np.mean(matches) * 100
