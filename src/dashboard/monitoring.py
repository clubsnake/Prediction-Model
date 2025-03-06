# monitoring.py
"""
Provides a class to track and log predictions over time,
and compute accuracy metrics for a defined window.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
from src.utils.threadsafe import (
    AtomicFileWriter,
    convert_to_native_types,
    safe_write_json,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prediction_monitor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PredictionMonitor:
    """
    Monitors predictions, logs them, and calculates basic error metrics
    over a specified window (e.g. 24h).
    """

    def __init__(self, max_log_entries=1000, logs_path=None):
        """
        Initialize the prediction monitor.

        Args:
            max_log_entries: Maximum number of entries to keep in memory
            logs_path: Path to directory where logs will be saved
        """
        self.predictions_log = pd.DataFrame(
            columns=[
                "timestamp",
                "ticker",
                "timeframe",
                "actual_price",
                "predicted_price",
                "prediction_error",
                "horizon",
            ]
        )
        self.metrics_cache = {}
        self.last_cache_update = None
        self.cache_timeout = timedelta(minutes=5)
        self.max_log_entries = max_log_entries

        # Set default logs path if none provided
        self.logs_path = logs_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )

        # Ensure logs directory exists
        os.makedirs(self.logs_path, exist_ok=True)

        # Load existing logs if available
        self.log_file_path = os.path.join(self.logs_path, "prediction_logs.csv")
        self.load_logs()

    def load_logs(self) -> bool:
        """
        Load prediction logs from disk.

        Returns:
            bool: True if logs were successfully loaded
        """
        try:
            if os.path.exists(self.log_file_path):
                loaded_logs = pd.read_csv(self.log_file_path, parse_dates=["timestamp"])
                if not loaded_logs.empty:
                    # If we have more entries than max_log_entries, keep only the most recent
                    if len(loaded_logs) > self.max_log_entries:
                        loaded_logs = loaded_logs.iloc[-self.max_log_entries :]
                    self.predictions_log = loaded_logs
                    logger.info(
                        f"Loaded {len(loaded_logs)} prediction logs from {self.log_file_path}"
                    )
                    return True
            return False
        except Exception as e:
            logger.error(f"Error loading prediction logs: {str(e)}")
            return False

    def save_logs(self) -> bool:
        """
        Save prediction logs to disk.

        Returns:
            bool: True if logs were successfully saved
        """
        try:
            if not self.predictions_log.empty:
                # Use AtomicFileWriter for thread-safe writing
                with AtomicFileWriter(self.log_file_path) as temp_file:
                    self.predictions_log.to_csv(temp_file.name, index=False)
                logger.info(
                    f"Saved {len(self.predictions_log)} prediction logs to {self.log_file_path}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving prediction logs: {str(e)}")
            return False

    def log_prediction(
        self, ticker: str, timeframe: str, actual: float, predicted: float, horizon: int
    ) -> None:
        """
        Log a single prediction result, computing error automatically.
        """
        try:
            actual = float(actual)
            predicted = float(predicted)
            if not (np.isfinite(actual) and np.isfinite(predicted)):
                raise ValueError("Invalid actual or predicted values (NaN/Inf).")

            error = abs((predicted - actual) / actual) * 100
            new_entry = pd.DataFrame(
                {
                    "timestamp": [datetime.now()],
                    "ticker": [ticker],
                    "timeframe": [timeframe],
                    "actual_price": [actual],
                    "predicted_price": [predicted],
                    "prediction_error": [error],
                    "horizon": [horizon],
                }
            )
            self.predictions_log = pd.concat(
                [self.predictions_log, new_entry], ignore_index=True
            )

            # Limit the log size to max_log_entries
            if len(self.predictions_log) > self.max_log_entries:
                self.predictions_log = self.predictions_log.iloc[
                    -self.max_log_entries :
                ]

            self.metrics_cache = {}  # clear cache

            # Save logs after each update
            self.save_logs()
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")

    def get_accuracy_metrics(self, window: str = "24h") -> Dict[str, float]:
        """
        Return aggregated accuracy metrics over the specified window.

        :param window: e.g. '24h', '7d', etc.
        :return: Dict with mean_error, max_error, and correct_direction fraction.
        """
        try:
            now = datetime.now()
            if (
                self.last_cache_update
                and (now - self.last_cache_update < self.cache_timeout)
                and window in self.metrics_cache
            ):
                return self.metrics_cache[window]

            # Vectorized filtering
            recent = self.predictions_log[
                self.predictions_log["timestamp"] > now - pd.Timedelta(window)
            ]
            if recent.empty:
                return {"mean_error": 0.0, "max_error": 0.0, "correct_direction": 0.0}

            # Vectorized calculations
            mean_error = float(recent["prediction_error"].mean())
            max_error = float(recent["prediction_error"].max())

            # Calculate direction accuracy using vectorized operations
            price_diffs = recent["predicted_price"].diff()
            actual_diffs = recent["actual_price"].diff()
            correct_dir = float((price_diffs * actual_diffs > 0).mean())

            metrics = {
                "mean_error": mean_error,
                "max_error": max_error,
                "correct_direction": correct_dir,
            }
            self.metrics_cache[window] = metrics
            self.last_cache_update = now

            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {"mean_error": 0.0, "max_error": 0.0, "correct_direction": 0.0}

    def export_metrics_history(self, path: str = None) -> bool:
        """
        Export metrics history to JSON file.

        Args:
            path: Path to save the metrics history. If None, use default path.

        Returns:
            bool: True if successful
        """
        try:
            if path is None:
                path = os.path.join(self.logs_path, "metrics_history.json")

            # Calculate metrics for different time windows
            metrics = {
                "1h": self.get_accuracy_metrics("1h"),
                "24h": self.get_accuracy_metrics("24h"),
                "7d": self.get_accuracy_metrics("7d"),
                "all": self.get_accuracy_metrics("365d"),  # Effectively all-time
            }

            # Add timestamp
            metrics["timestamp"] = datetime.now().isoformat()

            # Convert to native types for JSON
            metrics = convert_to_native_types(metrics)

            # Save to JSON
            safe_write_json(metrics, path)
            logger.info(f"Exported metrics history to {path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting metrics history: {str(e)}")
            return False
