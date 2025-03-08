# preprocessing.py
"""
Data preprocessing utilities with added type hints.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import logging
from typing import Optional
from datetime import datetime, timedelta

from src.utils.vectorized_ops import (
    vectorized_sequence_creation, 
    batch_sequence_creation
)
from src.utils.memory_utils import log_memory_usage

logger = logging.getLogger(__name__)

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


def ensure_datetime_index(df, date_col='date'):
    """
    Ensure the DataFrame has a proper datetime index or a datetime column.
    
    Args:
        df: DataFrame to process
        date_col: Name of the date column to use if index is not datetime
    
    Returns:
        DataFrame with proper datetime index
    """
    result_df = df.copy()
    
    # Check if the index is already a DatetimeIndex
    if isinstance(result_df.index, pd.DatetimeIndex):
        # We already have a datetime index
        return result_df
    
    # Check if the date column exists
    if date_col in result_df.columns:
        try:
            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Set the date column as index
            result_df = result_df.set_index(date_col)
            logger.info(f"Set index using '{date_col}' column")
            return result_df
            
        except Exception as e:
            logger.error(f"Error setting datetime index: {str(e)}")
    
    # If we get here, we need to create a date column
    try:
        # Create a date range assuming daily data
        start_date = datetime.now() - timedelta(days=len(result_df))
        result_df[date_col] = pd.date_range(start=start_date, periods=len(result_df), freq='D')
        logger.info(f"Fixed missing date column: {[date_col]}")
        
        # Set as index
        result_df = result_df.set_index(date_col)
        return result_df
        
    except Exception as e:
        logger.error(f"Error creating date column: {str(e)}")
        return df  # Return original if all attempts fail


def preprocess_data(df, date_col='date', fill_method='ffill', normalize=False):
    """
    Preprocess DataFrame for modeling.
    
    Args:
        df: DataFrame with price data
        date_col: Name of the date column (if not index)
        fill_method: Method to fill missing values
        normalize: Whether to normalize the data
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Ensure we have a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure datetime index
        result_df = ensure_datetime_index(result_df, date_col)
        
        # Handle missing values
        result_df = result_df.fillna(method=fill_method)
        
        # Drop any remaining NaN values
        result_df = result_df.dropna()
        
        # Normalize if requested
        if normalize:
            for col in result_df.select_dtypes(include=np.number).columns:
                mean = result_df[col].mean()
                std = result_df[col].std()
                if std > 0:
                    result_df[col] = (result_df[col] - mean) / std
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return df  # Return original if preprocessing fails


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Close",
    lookback: int = LOOKBACK,
    horizon: int = PREDICTION_HORIZON,
    use_batched: bool = False,
    batch_size: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time-series sequences for supervised learning using vectorized operations.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        lookback: Number of lookback steps
        horizon: Number of steps to predict
        use_batched: Whether to use memory-efficient batched processing
        batch_size: Batch size for batched processing
        
    Returns:
        X, y numpy arrays
    """
    log_memory_usage("before_sequence_creation")
    
    # Validate data
    if len(df) < lookback + horizon:
        logger.error(
            f"DataFrame length ({len(df)}) is less than required minimum "
            f"({lookback + horizon}) for lookback={lookback} and horizon={horizon}"
        )
        return None, None
    
    # Check for missing columns
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return None, None
        
    try:
        # Use vectorized implementation for standard cases
        if not use_batched or len(df) <= batch_size:
            X, y = vectorized_sequence_creation(df, feature_cols, target_col, lookback, horizon)
        else:
            # For very large datasets, use the batch processing approach
            X, y = batch_sequence_creation(df, feature_cols, target_col, lookback, horizon, batch_size)
        
        log_memory_usage("after_sequence_creation")
        logger.info(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
        
        return X, y
    except Exception as e:
        logger.error(f"Error creating sequences: {e}")
        return None, None
