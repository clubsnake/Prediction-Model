# monitoring.py
"""
Provides a class to track and log predictions over time,
and compute accuracy metrics for a defined window.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prediction_monitor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PredictionMonitor:
    """
    Monitor and evaluate model predictions over time.

    This class tracks predictions, compares them with actual values,
    and calculates accuracy metrics over different time windows.
    """

    def __init__(self, logs_path: str = None):
        """
        Initialize the prediction monitor.

        Args:
            logs_path: Path to store prediction logs
        """
        self.logs_path = logs_path or os.path.join(os.getcwd(), "logs")
        os.makedirs(self.logs_path, exist_ok=True)

        # Create DataFrame to store predictions
        self.predictions_log = pd.DataFrame(
            columns=[
                "timestamp",
                "ticker",
                "timeframe",
                "horizon",
                "predicted",
                "actual",
                "error",
                "pct_error",
            ]
        )

        # Load existing logs if available
        self._load_logs()

    def _load_logs(self):
        """Load existing prediction logs from disk"""
        try:
            log_file = os.path.join(self.logs_path, "predictions.csv")
            if os.path.exists(log_file):
                self.predictions_log = pd.read_csv(log_file)
                # Ensure timestamp is datetime for filtering operations
                self.predictions_log["timestamp"] = pd.to_datetime(
                    self.predictions_log["timestamp"]
                )
                logger.info("Loaded %d prediction logs", len(self.predictions_log))
        except Exception as e:
            logger.error("Error loading prediction logs: %s", e)

    def log_prediction(
        self,
        ticker: str,
        timeframe: str,
        predicted: float,
        actual: Optional[float] = None,
        horizon: int = 1,
        timestamp: Optional[datetime] = None,
    ):
        """
        Log a new prediction with optional actual value.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe (e.g. '1d', '1h')
            predicted: Predicted price value
            actual: Actual price value if known
            horizon: Prediction horizon in periods (days/hours)
            timestamp: Prediction timestamp (defaults to current time)
        """
        try:
            timestamp = timestamp or datetime.now()

            # Calculate error metrics if actual value is provided
            error = None
            pct_error = None
            if actual is not None:
                error = predicted - actual
                pct_error = (abs(error) / actual) * 100 if actual != 0 else None

            # Create new log entry
            new_prediction = pd.DataFrame(
                [
                    {
                        "timestamp": timestamp,
                        "ticker": ticker,
                        "timeframe": timeframe,
                        "horizon": horizon,
                        "predicted": predicted,
                        "actual": actual,
                        "error": error,
                        "pct_error": pct_error,
                    }
                ]
            )

            # Append to log
            self.predictions_log = pd.concat(
                [self.predictions_log, new_prediction], ignore_index=True
            )

            # Save to disk
            self._save_logs()

            logger.debug(
                "Logged prediction for %s/%s: predicted=%.4f, actual=%s",
                ticker, timeframe, predicted, actual if actual is not None else "None"
            )
            return True
        except Exception as e:
            logger.error("Error logging prediction: %s", e)
            return False

    def update_actual(
        self, ticker: str, timeframe: str, timestamp: datetime, actual: float
    ):
        """
        Update a prediction with actual value once known.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            timestamp: Prediction timestamp to update
            actual: Actual price value

        Returns:
            bool: True if updated successfully
        """
        try:
            # Convert timestamp to pd.Timestamp for comparison
            pd_timestamp = pd.Timestamp(timestamp)

            # Locate the prediction to update
            mask = (
                (self.predictions_log["ticker"] == ticker)
                & (self.predictions_log["timeframe"] == timeframe)
                & (self.predictions_log["timestamp"] == pd_timestamp)
                & (self.predictions_log["actual"].isna())
            )

            if not mask.any():
                logger.warning(
                    f"No matching prediction found to update for {ticker}/{timeframe} at {timestamp}"
                )
                return False

            # Update the actual value and calculate error metrics
            idx = mask.idxmax()
            predicted = self.predictions_log.at[idx, "predicted"]

            self.predictions_log.at[idx, "actual"] = actual
            self.predictions_log.at[idx, "error"] = predicted - actual
            self.predictions_log.at[idx, "pct_error"] = (
                (abs(predicted - actual) / actual) * 100 if actual != 0 else None
            )

            # Save updated logs
            self._save_logs()

            logger.debug(
                f"Updated prediction for {ticker}/{timeframe}: actual={actual}, predicted={predicted}"
            )
            return True
        except Exception as e:
            logger.error(f"Error updating prediction: {e}")
            return False

    def _save_logs(self):
        """Save prediction logs to disk"""
        try:
            log_file = os.path.join(self.logs_path, "predictions.csv")
            self.predictions_log.to_csv(log_file, index=False)
            logger.debug("Saved %d prediction logs to %s", len(self.predictions_log), log_file)
        except Exception as e:
            logger.error("Error saving prediction logs: %s", e)

    def get_accuracy_metrics(
        self, window: str = "all", ticker: str = None, timeframe: str = None
    ) -> Dict:
        """
        Calculate accuracy metrics over a time window.

        Args:
            window: Time window for metrics ('all', '24h', '7d', '30d', etc.)
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)

        Returns:
            Dict with accuracy metrics
        """
        try:
            # Create a filtered copy of the log
            filtered_log = self.predictions_log.copy()

            # Apply time window filter
            if window != "all":
                if window == "24h":
                    cutoff = datetime.now() - timedelta(days=1)
                elif window == "7d":
                    cutoff = datetime.now() - timedelta(days=7)
                elif window == "30d":
                    cutoff = datetime.now() - timedelta(days=30)
                else:
                    logger.warning(f"Unknown time window: {window}, using all data")
                    cutoff = None

                if cutoff:
                    filtered_log = filtered_log[filtered_log["timestamp"] >= cutoff]

            # Apply ticker filter
            if ticker:
                filtered_log = filtered_log[filtered_log["ticker"] == ticker]

            # Apply timeframe filter
            if timeframe:
                filtered_log = filtered_log[filtered_log["timeframe"] == timeframe]

            # Filter to entries with both predicted and actual values
            valid_entries = filtered_log.dropna(subset=["predicted", "actual"])

            # Default metrics in case of empty data
            metrics = {
                "mean_error": 0,
                "mean_abs_error": 0,
                "root_mean_sq_error": 0,
                "correct_direction": 0,
                "num_predictions": len(valid_entries),
            }

            if len(valid_entries) == 0:
                return metrics

            # Calculate error metrics
            metrics["mean_error"] = valid_entries["pct_error"].mean()
            metrics["mean_abs_error"] = valid_entries["error"].abs().mean()
            metrics["root_mean_sq_error"] = np.sqrt(
                (valid_entries["error"] ** 2).mean()
            )

            # Calculate direction accuracy if we have enough data
            if len(valid_entries) > 1:
                # Sort by timestamp
                sorted_entries = valid_entries.sort_values("timestamp")

                # Calculate direction changes
                actual_direction = np.diff(sorted_entries["actual"]) > 0
                pred_direction = np.diff(sorted_entries["predicted"]) > 0

                # Direction correct if both have same sign
                correct = (actual_direction == pred_direction).mean()
                metrics["correct_direction"] = correct

            return metrics
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {"error": str(e)}

    def get_recent_predictions(
        self, limit: int = 10, ticker: str = None, timeframe: str = None
    ):
        """
        Get the most recent predictions.

        Args:
            limit: Maximum number of predictions to return
            ticker: Filter by ticker (optional)
            timeframe: Filter by timeframe (optional)

        Returns:
            DataFrame with recent predictions
        """
        try:
            # Create a filtered copy of the log
            filtered_log = self.predictions_log.copy()

            # Apply ticker filter
            if ticker:
                filtered_log = filtered_log[filtered_log["ticker"] == ticker]

            # Apply timeframe filter
            if timeframe:
                filtered_log = filtered_log[filtered_log["timeframe"] == timeframe]

            # Sort by timestamp (newest first) and limit results
            recent = filtered_log.sort_values("timestamp", ascending=False).head(limit)

            return recent
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return pd.DataFrame()

    def export_predictions_to_json(self, file_path: str = None) -> bool:
        """
        Export predictions log to JSON file.

        Args:
            file_path: Path to save the JSON file. If None, saves to logs directory.

        Returns:
            bool: True if export was successful
        """
        try:
            if file_path is None:
                file_path = os.path.join(self.logs_path, "predictions_export.json")

            # Convert DataFrame to JSON-compatible format
            predictions_data = self.predictions_log.to_dict(orient="records")

            # Convert datetime objects to strings
            for entry in predictions_data:
                if isinstance(entry["timestamp"], pd.Timestamp):
                    entry["timestamp"] = entry["timestamp"].isoformat()

            # Write to file
            with open(file_path, "w") as f:
                json.dump(predictions_data, f, indent=2)

            logger.info(f"Exported {len(predictions_data)} predictions to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting predictions to JSON: {e}")
            return False
