# monitoring.py
"""
Provides a class to track and log predictions over time, 
and compute accuracy metrics for a defined window.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict

class PredictionMonitor:
    """
    Monitors predictions, logs them, and calculates basic error metrics 
    over a specified window (e.g. 24h).
    """
    def __init__(self):
        self.predictions_log = pd.DataFrame(columns=[
            'timestamp', 'ticker', 'timeframe', 'actual_price', 
            'predicted_price', 'prediction_error', 'horizon'
        ])
        self.metrics_cache = {}
        self.last_cache_update = None
        self.cache_timeout = timedelta(minutes=5)
    
    def log_prediction(self, ticker: str, timeframe: str, actual: float, predicted: float, horizon: int) -> None:
        """
        Log a single prediction result, computing error automatically.
        """
        try:
            actual = float(actual)
            predicted = float(predicted)
            if not (np.isfinite(actual) and np.isfinite(predicted)):
                raise ValueError("Invalid actual or predicted values (NaN/Inf).")
                
            error = abs((predicted - actual) / actual) * 100
            new_entry = pd.DataFrame({
                'timestamp': [datetime.now()],
                'ticker': [ticker],
                'timeframe': [timeframe],
                'actual_price': [actual],
                'predicted_price': [predicted],
                'prediction_error': [error],
                'horizon': [horizon]
            })
            self.predictions_log = pd.concat([self.predictions_log, new_entry], ignore_index=True)
            self.metrics_cache = {}  # clear cache
        except Exception as e:
            print(f"Error logging prediction: {str(e)}")
            
    def get_accuracy_metrics(self, window: str = '24h') -> Dict[str, float]:
        """
        Return aggregated accuracy metrics over the specified window.

        :param window: e.g. '24h', '7d', etc.
        :return: Dict with mean_error, max_error, and correct_direction fraction.
        """
        try:
            now = datetime.now()
            if (self.last_cache_update and (now - self.last_cache_update < self.cache_timeout)
                and window in self.metrics_cache):
                return self.metrics_cache[window]
            
            recent = self.predictions_log[self.predictions_log['timestamp'] > now - pd.Timedelta(window)]
            if recent.empty:
                return {'mean_error': 0.0, 'max_error': 0.0, 'correct_direction': 0.0}
            
            mean_error = float(recent['prediction_error'].mean())
            max_error = float(recent['prediction_error'].max())
            correct_dir = float((recent['predicted_price'].diff().fillna(0) *
                                 recent['actual_price'].diff().fillna(0) > 0).mean())
            
            metrics = {
                'mean_error': mean_error,
                'max_error': max_error,
                'correct_direction': correct_dir
            }
            self.metrics_cache[window] = metrics
            self.last_cache_update = now
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {'mean_error': 0.0, 'max_error': 0.0, 'correct_direction': 0.0}
