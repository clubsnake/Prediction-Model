"""
Unit tests for data preprocessing functions.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_manager import parse_dates
from src.utils.vectorized_ops import preserve_date_column, vectorized_sequence_creation


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""

    def setUp(self) -> None:
        """Set up test environment."""
        # Create a sample dataframe
        dates = pd.date_range(start="2022-01-01", periods=100)
        self.df = pd.DataFrame(
            {
                "date": dates,
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.normal(0, 1, 100),
            }
        )

        self.feature_cols = ["feature1", "feature2"]
        self.target_col = "target"

    def test_vectorized_sequence_creation(self) -> None:
        """Test that sequence creation produces correct shapes."""
        X, y = vectorized_sequence_creation(
            self.df, self.feature_cols, self.target_col, lookback=10, horizon=5
        )

        # Calculate expected shapes
        expected_X_shape = (100 - 10 - 5 + 1, 10, len(self.feature_cols))
        expected_y_shape = (100 - 10 - 5 + 1, 5)

        self.assertEqual(X.shape, expected_X_shape)
        self.assertEqual(y.shape, expected_y_shape)

    def test_empty_sequence_handling(self) -> None:
        """Test handling of sequences that are too short."""
        # Create a very short dataframe
        short_df = self.df.iloc[:5]

        X, y = vectorized_sequence_creation(
            short_df, self.feature_cols, self.target_col, lookback=10, horizon=5
        )

        # Should return empty arrays
        self.assertEqual(X.shape, (0,))
        self.assertEqual(y.shape, (0,))

    def test_date_preservation(self) -> None:
        """Test that date column is preserved after scaling."""
        # Create a scaled dataframe without date column
        scaled_df = pd.DataFrame(
            np.random.normal(0, 1, (100, 2)), columns=self.feature_cols
        )

        # Preserve date column
        result_df = preserve_date_column(self.df, scaled_df)

        # Check that date column was added
        self.assertIn("date", result_df.columns)

        # Check that dates match
        pd.testing.assert_series_equal(result_df["date"], self.df["date"])

    def test_parse_dates(self) -> None:
        """Test date parsing functionality."""
        # Create dataframe with string dates
        string_dates = [f"2022-01-{i+1:02d}" for i in range(10)]
        test_df = pd.DataFrame({"date": string_dates, "value": range(10)})

        # Parse dates
        result_df = parse_dates(test_df)

        # Check result
        self.assertTrue(pd.api.types.is_datetime64_dtype(result_df["date"]))
        self.assertEqual(result_df["date"][0], pd.Timestamp("2022-01-01"))

    def test_preprocessing_with_gaps(self) -> None:
        """Test preprocessing with gaps in data."""
        # Create dataframe with gaps
        dates = list(pd.date_range(start="2022-01-01", periods=50)) + list(
            pd.date_range(start="2022-03-01", periods=50)
        )

        gap_df = pd.DataFrame(
            {
                "date": dates,
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.normal(0, 1, 100),
            }
        )

        # Try to create sequences
        X, y = vectorized_sequence_creation(
            gap_df, self.feature_cols, self.target_col, lookback=10, horizon=5
        )

        # Should still work, but be aware of the gap
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)


if __name__ == "__main__":
    unittest.main()
