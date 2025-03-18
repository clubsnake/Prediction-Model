"""
Smart adaptive scheduler for drift hyperparameter tuning.
Integrates with existing incremental learning and adaptive parameters systems.
"""

import logging
import numpy as np
import os
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Import existing utility functions
from src.training.adaptive_params import calculate_volatility, adaptive_retraining_threshold
from src.utils.memory_utils import adaptive_memory_clean

logger = logging.getLogger(__name__)

class DriftTuningScheduler:
    """
    Intelligent scheduler for drift hyperparameter optimization that responds
    to system performance, market conditions, and computational resources.
    """
    
    def __init__(self, 
                 registry=None,
                 base_interval_days=7,
                 min_interval_days=1,
                 max_interval_days=30,
                 performance_window=10,
                 volatility_multiplier=0.5,
                 accuracy_threshold=0.1,
                 confidence_threshold=0.15,
                 drift_frequency_threshold=0.3,
                 resource_threshold=0.7,
                 enable_auto_tune=True):
        """
        Initialize the drift tuning scheduler.
        
        Args:
            registry: Model registry for tracking performance
            base_interval_days: Default days between optimizations
            min_interval_days: Minimum days between optimizations
            max_interval_days: Maximum days between optimizations
            performance_window: Window size for performance tracking
            volatility_multiplier: How much volatility affects schedule
            accuracy_threshold: Performance drop to trigger early tuning
            confidence_threshold: Confidence drop to trigger early tuning
            drift_frequency_threshold: Drift detection frequency to trigger tuning
            resource_threshold: CPU/memory threshold for scheduling
            enable_auto_tune: Whether to enable automatic tuning
        """
        self.registry = registry
        self.base_interval_days = base_interval_days
        self.min_interval_days = min_interval_days
        self.max_interval_days = max_interval_days
        self.performance_window = performance_window
        self.volatility_multiplier = volatility_multiplier
        self.accuracy_threshold = accuracy_threshold
        self.confidence_threshold = confidence_threshold
        self.drift_frequency_threshold = drift_frequency_threshold
        self.resource_threshold = resource_threshold
        self.enable_auto_tune = enable_auto_tune
        
        # Internal state tracking
        self.last_tuning_time = {}  # {ticker_timeframe: timestamp}
        self.next_scheduled_time = {}  # {ticker_timeframe: timestamp}
        self.performance_history = {}  # {ticker_timeframe: [{timestamp, metric, value}]}
        self.drift_history = {}  # {ticker_timeframe: [{timestamp, type, score}]}
        self.confidence_history = {}  # {ticker_timeframe: [{timestamp, value}]}
        self.optimization_history = {}  # {ticker_timeframe: [{timestamp, hyperparams, result}]}
        
        # Load previous tuning state if available
        self._load_state()
        
        logger.info(f"Initialized DriftTuningScheduler with base interval of {base_interval_days} days")
    
    def record_performance(self, ticker, timeframe, metric_name, metric_value):
        """Record model performance metrics for tuning decisions"""
        key = f"{ticker}_{timeframe}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append({
            'timestamp': time.time(),
            'metric': metric_name,
            'value': metric_value
        })
        
        # Trim history to keep memory usage reasonable
        max_history = self.performance_window * 3
        if len(self.performance_history[key]) > max_history:
            self.performance_history[key] = self.performance_history[key][-max_history:]
    
    def record_drift_detection(self, ticker, timeframe, drift_type, drift_score):
        """Record detected drift events for tuning decisions"""
        key = f"{ticker}_{timeframe}"
        if key not in self.drift_history:
            self.drift_history[key] = []
        
        self.drift_history[key].append({
            'timestamp': time.time(),
            'type': drift_type,
            'score': drift_score
        })
        
        # Trim history
        max_history = 100  # Keep more drift history
        if len(self.drift_history[key]) > max_history:
            self.drift_history[key] = self.drift_history[key][-max_history:]
    
    def record_confidence(self, ticker, timeframe, confidence_value):
        """Record prediction confidence scores for tuning decisions"""
        key = f"{ticker}_{timeframe}"
        if key not in self.confidence_history:
            self.confidence_history[key] = []
        
        self.confidence_history[key].append({
            'timestamp': time.time(),
            'value': confidence_value
        })
        
        # Trim history
        max_history = self.performance_window * 3
        if len(self.confidence_history[key]) > max_history:
            self.confidence_history[key] = self.confidence_history[key][-max_history:]
    
    def record_optimization(self, ticker, timeframe, hyperparams, result_metrics):
        """Record hyperparameter optimization results"""
        key = f"{ticker}_{timeframe}"
        if key not in self.optimization_history:
            self.optimization_history[key] = []
        
        self.optimization_history[key].append({
            'timestamp': time.time(),
            'hyperparams': hyperparams,
            'result': result_metrics
        })
        
        # Update last tuning time
        self.last_tuning_time[key] = time.time()
        
        # Schedule next tuning based on current conditions
        next_interval = self._calculate_next_interval(ticker, timeframe)
        self.next_scheduled_time[key] = time.time() + (next_interval * 24 * 60 * 60)
        
        # Save state after optimization
        self._save_state()
    
    def should_tune_now(self, ticker, timeframe, market_data=None, system_load=None):
        """
        Determine if drift hyperparameters should be tuned now based on 
        performance metrics, market conditions, and system resources.
        
        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            market_data: DataFrame with market data for volatility calculation
            system_load: Current system load (0-1), or None to check automatically
            
        Returns:
            should_tune: Whether tuning should be performed now
            reason: Reason for tuning decision
        """
        if not self.enable_auto_tune:
            return False, "Auto-tuning disabled"
        
        key = f"{ticker}_{timeframe}"
        
        # Check if this is the first tuning
        if key not in self.last_tuning_time:
            return True, "Initial tuning"
        
        # Check if we're past scheduled time
        current_time = time.time()
        if key in self.next_scheduled_time and current_time > self.next_scheduled_time[key]:
            return True, "Scheduled tuning"
        
        # Don't tune if we've tuned recently (respect minimum interval)
        min_seconds = self.min_interval_days * 24 * 60 * 60
        if key in self.last_tuning_time and (current_time - self.last_tuning_time[key]) < min_seconds:
            return False, "Minimum interval not reached"
        
        # Check system load
        if system_load is None:
            system_load = self._check_system_load()
        
        if system_load > self.resource_threshold:
            logger.info(f"Postponing tuning due to high system load: {system_load:.2f}")
            return False, f"System load too high: {system_load:.2f}"
        
        # Check for performance degradation
        performance_trigger = self._check_performance_degradation(ticker, timeframe)
        if performance_trigger:
            return True, f"Performance degradation: {performance_trigger}"
        
        # Check for confidence degradation
        confidence_trigger = self._check_confidence_degradation(ticker, timeframe)
        if confidence_trigger:
            return True, f"Confidence degradation: {confidence_trigger:.2f}"
        
        # Check for high drift frequency
        drift_trigger = self._check_drift_frequency(ticker, timeframe)
        if drift_trigger:
            return True, f"High drift frequency: {drift_trigger:.2f}"
        
        # Check market volatility if data provided
        if market_data is not None:
            volatility_trigger = self._check_market_volatility(market_data)
            if volatility_trigger:
                return True, f"High market volatility: {volatility_trigger:.2f}"
        
        return False, "No tuning triggers detected"
    
    def get_optimization_config(self, ticker, timeframe):
        """
        Get optimization configuration for this ticker/timeframe,
        using prior optimization results to guide new search.
        
        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            
        Returns:
            config: Dictionary with optimization configuration
        """
        key = f"{ticker}_{timeframe}"
        
        # Default configuration
        config = {
            "n_trials": 30,
            "timeout": 7200,  # 2 hours
            "search_space": {
                "statistical_threshold": (1.0, 3.0),
                "performance_threshold": (0.05, 0.25),
                "distribution_threshold": (0.01, 0.1),
                "window_size_factor": (0.3, 0.8),
                "learning_rate_factor": (1.0, 3.0),
                "retrain_threshold": (0.5, 0.9),
                "ensemble_weight_multiplier": (1.1, 1.5),
                "memory_length": (3, 10)
            },
            "prior_params": None
        }
        
        # If we have prior optimization results, use them to guide search
        if key in self.optimization_history and self.optimization_history[key]:
            # Get most recent optimization
            latest_opt = self.optimization_history[key][-1]
            prior_params = latest_opt.get('hyperparams', {})
            
            if prior_params:
                config["prior_params"] = prior_params
                
                # Narrow search space around prior good values
                # (but still allow exploration to prevent getting stuck)
                for param, value in prior_params.items():
                    if param in config["search_space"]:
                        current_range = config["search_space"][param]
                        
                        # For numerical parameters, narrow the range
                        if isinstance(current_range, tuple) and len(current_range) == 2:
                            # Calculate new bounds: allow 50% movement in either direction
                            # but stay within original bounds
                            orig_min, orig_max = current_range
                            range_size = orig_max - orig_min
                            
                            # New bounds: 50% of original range centered on prior value
                            new_min = max(orig_min, value - range_size * 0.25)
                            new_max = min(orig_max, value + range_size * 0.25)
                            
                            # Update search space
                            config["search_space"][param] = (new_min, new_max)
        
        # Adjust number of trials based on how much we've already explored
        if key in self.optimization_history:
            # Reduce trials for subsequent optimizations
            num_previous = len(self.optimization_history[key])
            if num_previous > 1:
                # Gradually reduce trials as we gain confidence
                config["n_trials"] = max(15, 30 - num_previous * 3)
        
        return config
    
    def get_optimal_hyperparameters(self, ticker, timeframe):
        """
        Get the current optimal hyperparameters for this ticker/timeframe.
        
        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            
        Returns:
            hyperparams: Dictionary with optimal hyperparameters or None
        """
        key = f"{ticker}_{timeframe}"
        
        if key in self.optimization_history and self.optimization_history[key]:
            # Return the most recent optimization result
            return self.optimization_history[key][-1].get('hyperparams')
        
        return None
    
    def _check_performance_degradation(self, ticker, timeframe):
        """Check if there's significant performance degradation"""
        key = f"{ticker}_{timeframe}"
        
        if key not in self.performance_history:
            return False
        
        history = self.performance_history[key]
        if len(history) < self.performance_window:
            return False
        
        # Group by metric
        metrics = {}
        for entry in history:
            metric = entry['metric']
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(entry['value'])
        
        # Check each metric for degradation
        for metric, values in metrics.items():
            if len(values) < self.performance_window:
                continue
                
            # Get recent and older values
            recent = values[-self.performance_window//2:]
            older = values[-(self.performance_window):-self.performance_window//2]
            
            if not recent or not older:
                continue
                
            # Calculate percent change
            recent_avg = np.mean(recent)
            older_avg = np.mean(older)
            
            # For error metrics (lower is better)
            if metric.lower() in ['rmse', 'mse', 'mae', 'mape']:
                pct_change = (recent_avg - older_avg) / (abs(older_avg) + 1e-8)
                
                # If recent performance is worse by threshold
                if pct_change > self.accuracy_threshold:
                    return f"{metric}: {pct_change:.2f}"
            # For score metrics (higher is better)
            else:
                pct_change = (older_avg - recent_avg) / (abs(older_avg) + 1e-8)
                
                # If recent performance is worse by threshold
                if pct_change > self.accuracy_threshold:
                    return f"{metric}: {pct_change:.2f}"
        
        return False
    
    def _check_confidence_degradation(self, ticker, timeframe):
        """Check if there's significant confidence degradation"""
        key = f"{ticker}_{timeframe}"
        
        if key not in self.confidence_history:
            return False
        
        history = self.confidence_history[key]
        if len(history) < self.performance_window:
            return False
        
        # Get recent and older values
        recent = [entry['value'] for entry in history[-self.performance_window//2:]]
        older = [entry['value'] for entry in history[-(self.performance_window):-self.performance_window//2]]
        
        if not recent or not older:
            return False
            
        # Calculate percent change in confidence
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        # For confidence, higher is better
        pct_change = (older_avg - recent_avg) / (abs(older_avg) + 1e-8)
        
        # If confidence has dropped by threshold
        if pct_change > self.confidence_threshold:
            return pct_change
        
        return False
    
    def _check_drift_frequency(self, ticker, timeframe):
        """Check if drift detection frequency is abnormally high"""
        key = f"{ticker}_{timeframe}"
        
        if key not in self.drift_history:
            return False
        
        history = self.drift_history[key]
        if len(history) < 5:  # Need minimum history
            return False
        
        # Count recent drift events
        recent_window = 24 * 60 * 60  # Last 24 hours
        current_time = time.time()
        
        recent_drifts = [
            entry for entry in history 
            if entry['timestamp'] > (current_time - recent_window)
        ]
        
        # Calculate frequency as events per day
        frequency = len(recent_drifts) / 1.0  # per day
        
        # If frequency exceeds threshold
        if frequency > self.drift_frequency_threshold:
            return frequency
        
        return False
    
    def _check_market_volatility(self, market_data):
        """Check if market volatility is high enough to trigger tuning"""
        if market_data is None or 'Close' not in market_data.columns:
            return False
        
        # Use existing volatility calculation from adaptive_params
        try:
            volatility = calculate_volatility(market_data)
            
            # Typical volatility range is 0.1 to 0.4 (10% to 40%)
            # Higher volatility should trigger more frequent tuning
            volatility_trigger = volatility > 0.3  # 30% annualized volatility
            
            if volatility_trigger:
                return volatility
        except Exception as e:
            logger.warning(f"Error calculating market volatility: {e}")
        
        return False
    
    def _calculate_next_interval(self, ticker, timeframe):
        """
        Calculate the next tuning interval based on all factors.
        
        Returns interval in days.
        """
        # Start with base interval
        interval = self.base_interval_days
        
        key = f"{ticker}_{timeframe}"
        
        # 1. Adjust based on performance stability
        performance_factor = 1.0
        if key in self.performance_history and len(self.performance_history[key]) >= self.performance_window:
            # Calculate coefficient of variation for each metric
            metrics = {}
            for entry in self.performance_history[key][-self.performance_window:]:
                metric = entry['metric']
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(entry['value'])
            
            # Average CV across metrics
            cvs = []
            for values in metrics.values():
                if len(values) >= 3:  # Need minimum samples
                    cv = np.std(values) / (abs(np.mean(values)) + 1e-8)
                    cvs.append(cv)
            
            if cvs:
                avg_cv = np.mean(cvs)
                # Higher variation -> shorter interval
                performance_factor = 1.0 / (1.0 + avg_cv * 2.0)
        
        # 2. Adjust based on drift detection frequency
        drift_factor = 1.0
        if key in self.drift_history:
            # Count recent drift events
            recent_window = 7 * 24 * 60 * 60  # Last 7 days
            current_time = time.time()
            
            recent_drifts = [
                entry for entry in self.drift_history[key] 
                if entry['timestamp'] > (current_time - recent_window)
            ]
            
            # Calculate daily frequency
            daily_freq = len(recent_drifts) / 7.0  # per day
            
            # Higher frequency -> shorter interval
            if daily_freq > 0:
                drift_factor = 1.0 / (1.0 + daily_freq * 2.0)
        
        # 3. Adjust based on optimization history - more stable results -> longer intervals
        optimization_factor = 1.0
        if key in self.optimization_history and len(self.optimization_history[key]) >= 2:
            # Compare last two optimization results
            last = self.optimization_history[key][-1].get('hyperparams', {})
            prev = self.optimization_history[key][-2].get('hyperparams', {})
            
            if last and prev:
                # Calculate average parameter change
                param_changes = []
                for param, value in last.items():
                    if param in prev:
                        prev_value = prev[param]
                        if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
                            # Normalize change as percentage
                            change = abs(value - prev_value) / (abs(prev_value) + 1e-8)
                            param_changes.append(change)
                
                if param_changes:
                    avg_change = np.mean(param_changes)
                    # Smaller changes -> longer interval (more stable)
                    optimization_factor = 1.0 + (1.0 - min(1.0, avg_change * 5.0)) * 0.5
        
        # Combine all factors
        adjusted_interval = interval * performance_factor * drift_factor * optimization_factor
        
        # Apply bounds
        final_interval = max(self.min_interval_days, min(self.max_interval_days, adjusted_interval))
        
        logger.info(f"Calculated next tuning interval for {ticker}_{timeframe}: {final_interval:.1f} days")
        logger.debug(f"Adjustment factors - Performance: {performance_factor:.2f}, Drift: {drift_factor:.2f}, Optimization: {optimization_factor:.2f}")
        
        return final_interval
    
    def _check_system_load(self):
        """Check current system load to determine if tuning is feasible"""
        try:
            import psutil
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory_percent = psutil.virtual_memory().percent
            
            # Combined load metric (higher value = higher load)
            system_load = (cpu_percent + memory_percent) / 200.0
            
            return system_load
        except ImportError:
            logger.warning("psutil not available, cannot check system load")
            return 0.5  # Assume moderate load
    
    def _save_state(self):
        """Save scheduler state to disk"""
        try:
            # Create state directory
            from config.config_loader import get_data_dir
            state_dir = os.path.join(get_data_dir(), "DriftTuning")
            os.makedirs(state_dir, exist_ok=True)
            
            # Prepare state to save
            state = {
                "last_tuning_time": self.last_tuning_time,
                "next_scheduled_time": self.next_scheduled_time,
                "optimization_history": {}
            }
            
            # Only save minimal optimization history to reduce file size
            for key, history in self.optimization_history.items():
                # Save only the last 3 optimizations
                state["optimization_history"][key] = history[-3:] if len(history) > 3 else history
            
            # Save state
            state_path = os.path.join(state_dir, "scheduler_state.yaml")
            with open(state_path, "w") as f:
                yaml.dump(state, f)
                
            logger.debug(f"Saved drift tuning scheduler state to {state_path}")
            
        except Exception as e:
            logger.warning(f"Error saving scheduler state: {e}")
    
    def _load_state(self):
        """Load scheduler state from disk"""
        try:
            # Get state file path
            from config.config_loader import get_data_dir
            state_dir = os.path.join(get_data_dir(), "DriftTuning")
            state_path = os.path.join(state_dir, "scheduler_state.yaml")
            
            if os.path.exists(state_path):
                # Load state
                with open(state_path, "r") as f:
                    state = yaml.safe_load(f)
                
                # Restore state
                if state:
                    if "last_tuning_time" in state:
                        self.last_tuning_time = state["last_tuning_time"]
                    if "next_scheduled_time" in state:
                        self.next_scheduled_time = state["next_scheduled_time"]
                    if "optimization_history" in state:
                        self.optimization_history = state["optimization_history"]
                        
                logger.info(f"Loaded drift tuning scheduler state from {state_path}")
            
        except Exception as e:
            logger.warning(f"Error loading scheduler state: {e}")