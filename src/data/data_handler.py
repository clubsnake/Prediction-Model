from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class DataHandler:
    """
    Unified class for data loading, preprocessing, and feature engineering.
    Combines what might have been separate loader.py and preprocessing.py files.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.features = None
        self.labels = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various sources based on file extension"""
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def preprocess(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply preprocessing steps to the data"""
        df = data if data is not None else self.data

        # Handle missing values
        df = df.fillna(self.config.get("fill_value", 0))

        # Apply normalization if configured
        if self.config.get("normalize", False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[
                numeric_cols
            ].std()

        return df

    def prepare_features_and_labels(
        self, data: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe"""
        df = data if data is not None else self.data

        feature_cols = self.config.get("feature_columns", [])
        label_col = self.config.get("label_column", "target")

        X = df[feature_cols].values
        y = df[label_col].values

        self.features = X
        self.labels = y

        return X, y

    def process_pipeline(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full data processing pipeline"""
        self.data = self.load_data(file_path)
        self.data = self.preprocess(self.data)
        return self.prepare_features_and_labels(self.data)
