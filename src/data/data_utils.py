"""
Utilities for loading and preparing data for model training.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.robust_handler import error_boundary
from src.utils.vectorized_ops import vectorized_sequence_creation

logger = logging.getLogger(__name__)


@error_boundary
def load_and_prepare_data(
    file_path: str,
    target_col: str = "Close",
    lookback: int = 30,
    horizon: int = 7,
    test_size: float = 0.2,
    standardize: bool = True,
) -> Dict[str, Any]:
    """
    Load data from file and prepare it for model training.

    Args:
        file_path: Path to data file
        target_col: Target column name
        lookback: Lookback window size
        horizon: Prediction horizon
        test_size: Fraction of data to use for testing
        standardize: Whether to standardize the data

    Returns:
        Dictionary with X_train, y_train, X_test, y_test, and other info
    """
    try:
        # Load data
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".pkl"):
            df = pd.read_pickle(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .pkl")

        # Check if target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Extract date column if present
        date_col = None
        if "date" in df.columns:
            date_col = "date"
        elif "Date" in df.columns:
            date_col = "Date"

        # Handle date column
        if date_col is not None:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])

            # Sort by date
            df = df.sort_values(by=date_col)

        # Get feature columns (all numeric columns except target and date)
        feature_cols = [
            col
            for col in df.columns
            if col != target_col
            and col != date_col
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        # Split into train and test sets
        if date_col is not None:
            # Time-based split
            train_size = int(len(df) * (1 - test_size))
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
        else:
            # Random split
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42
            )

        # Standardize data if requested
        if standardize:
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_df[feature_cols])
            test_features = scaler.transform(test_df[feature_cols])

            # Scale target separately
            target_scaler = StandardScaler()
            train_target = target_scaler.fit_transform(train_df[[target_col]])
            test_target = target_scaler.transform(test_df[[target_col]])

            # Convert back to DataFrame
            train_df_scaled = pd.DataFrame(
                train_features, columns=feature_cols, index=train_df.index
            )
            train_df_scaled[target_col] = train_target
            test_df_scaled = pd.DataFrame(
                test_features, columns=feature_cols, index=test_df.index
            )
            test_df_scaled[target_col] = test_target

            # Preserve date column
            if date_col is not None:
                train_df_scaled[date_col] = train_df[date_col]
                test_df_scaled[date_col] = test_df[date_col]

            # Use scaled DataFrames
            train_df = train_df_scaled
            test_df = test_df_scaled

            # Save scalers
            scalers = {"features": scaler, "target": target_scaler}
        else:
            scalers = None

        # Create sequences
        X_train, y_train = vectorized_sequence_creation(
            train_df, feature_cols, target_col, lookback, horizon
        )

        X_test, y_test = vectorized_sequence_creation(
            test_df, feature_cols, target_col, lookback, horizon
        )

        # Return prepared data
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "scalers": scalers,
            "train_df": train_df,
            "test_df": test_df,
        }

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


def load_data_to_session_state(file_path: str) -> bool:
    """
    Load data and store in session state for use in the application.

    Args:
        file_path: Path to data file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get parameters from user input or defaults
        target_col = st.session_state.get("target_col", "Close")
        lookback = st.session_state.get("lookback", 30)
        horizon = st.session_state.get("horizon", 7)
        test_size = st.session_state.get("test_size", 0.2)

        # Load and prepare data
        data = load_and_prepare_data(
            file_path=file_path,
            target_col=target_col,
            lookback=lookback,
            horizon=horizon,
            test_size=test_size,
        )

        # Store in session state
        for key, value in data.items():
            st.session_state[key] = value

        st.session_state["data_loaded"] = True

        return True

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return False
