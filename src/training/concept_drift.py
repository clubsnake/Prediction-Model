"""
Advanced concept drift detection and adaptation module.
Integrates with existing adaptation mechanisms to provide sophisticated drift handling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from sklearn.metrics import mean_squared_error
import time
import logging
import os
import sys

logger = logging.getLogger(__name__)

class DriftHyperparameters:
    """
    Hyperparameters for drift detection that can be tuned.
    """
    def __init__(self, 
                 statistical_threshold=2.0,
                 performance_threshold=0.15,
                 distribution_threshold=0.05,
                 window_size_factor=1.0,
                 learning_rate_factor=1.0,
                 retrain_threshold=0.7,
                 ensemble_weight_multiplier=1.2,
                 memory_length=5):
        """
        Initialize drift hyperparameters.
        
        Args:
            statistical_threshold: Threshold for statistical drift detection (lower = more sensitive)
            performance_threshold: Threshold for performance drift detection (lower = more sensitive)
            distribution_threshold: Threshold for distribution drift detection (lower = more sensitive)
            window_size_factor: How much to adjust window size when drift is detected
            learning_rate_factor: How much to adjust learning rate when drift is detected
            retrain_threshold: Threshold score for triggering retraining
            ensemble_weight_multiplier: Factor to adjust ensemble weights by
            memory_length: How many drift events to remember
        """
        self.statistical_threshold = statistical_threshold
        self.performance_threshold = performance_threshold
        self.distribution_threshold = distribution_threshold
        self.window_size_factor = window_size_factor
        self.learning_rate_factor = learning_rate_factor
        self.retrain_threshold = retrain_threshold
        self.ensemble_weight_multiplier = ensemble_weight_multiplier
        self.memory_length = memory_length




class MultiDetectorDriftSystem:
    """
    Comprehensive drift detection system using multiple detection methods.
    Integrates with existing adaptation mechanisms and provides sophisticated responses.
    """
    
    def __init__(
        self,
        hyperparams=None,
        base_window_size: int = 30,
        statistical_window: int = 20, 
        performance_window: int = 30,
        distribution_window: int = 50,
        statistical_threshold: float = 2.0,
        performance_threshold: float = 0.15,
        distribution_threshold: float = 0.05,
        drift_memory: int = 5,
        confidence_weight: float = 0.7,
        feature_weight: float = 0.5,
        enable_visualization: bool = True
    ):
        """
        Initialize the drift detection system with multiple detectors.
        
        Args:
            base_window_size: Base window size for training
            statistical_window: Window size for statistical detector
            performance_window: Window size for performance detector
            distribution_window: Window size for distribution detector
            statistical_threshold: Threshold for statistical detector
            performance_threshold: Threshold for performance detector
            distribution_threshold: Threshold for distribution detector
            drift_memory: How many recent drift events to remember
            confidence_weight: Weight for confidence adaptation
            feature_weight: Weight for feature importance adaptation
            enable_visualization: Whether to store data for visualization
            Initialize with optional hyperparameters object.
            """
        
        # Use provided hyperparameters or create defaults
        self.hyperparams = hyperparams or DriftHyperparameters()
    
        self.base_window_size = base_window_size
        
        # Detection parameters
        self.windows = {
            'statistical': statistical_window,
            'performance': performance_window,
            'distribution': distribution_window
        }
        
        # Use hyperparameters for thresholds
        self.thresholds = {
            'statistical': self.hyperparams.statistical_threshold,
            'performance': self.hyperparams.performance_threshold,
            'distribution': self.hyperparams.distribution_threshold
        }
        
        # Adaptation parameters
        # Use hyperparameter for memory length
        self.drift_memory = self.hyperparams.memory_length
        self.confidence_weight = confidence_weight
        self.feature_weight = feature_weight
        
        # Internal state
        self.error_history = []
        self.performance_history = {}
        self.feature_distributions = {}
        self.detection_history = []
        self.current_adaptation = {}
        self.last_adaptation_time = 0
        self.enable_visualization = enable_visualization
        self.visualization_data = {
            'timestamps': [],
            'drift_scores': [],
            'drift_types': [],
            'adaptations': []
        }
        
        # Feature importance tracking
        self.feature_importance_history = {}
        
        logger.info(f"Initialized MultiDetectorDriftSystem with {len(self.windows)} detection methods")
    
    def update_error(self, timestamp, actual, predicted, features=None):
        """
        Update with new prediction error.
        
        Args:
            timestamp: Timestamp for this observation
            actual: Actual value
            predicted: Predicted value
            features: Feature values used for prediction (optional)
            
        Returns:
            drift_detected: Whether drift was detected
        """
        # Calculate error
        error = abs(actual - predicted)
        relative_error = error / (abs(actual) + 1e-8)
        
        # Store error with timestamp
        self.error_history.append({
            'timestamp': timestamp,
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'relative_error': relative_error,
            'features': features
        })
        
        # Trim history if needed
        max_history = max(self.windows.values()) * 3
        if len(self.error_history) > max_history:
            self.error_history = self.error_history[-max_history:]
        
        # Check for drift using all available detectors
        drift_results = self._detect_drift()
        
        # If drift detected, trigger adaptation
        if any(drift_results.values()):
            drift_type = next((k for k, v in drift_results.items() if v), None)
            drift_score = self._calculate_drift_score(drift_results)
            self._adapt_to_drift(drift_type, drift_score, features)
            
            # Record drift event
            self.detection_history.append({
                'timestamp': timestamp,
                'drift_type': drift_type,
                'drift_score': drift_score,
                'adaptation': self.current_adaptation.copy()
            })
            
            # Trim detection history
            if len(self.detection_history) > self.drift_memory:
                self.detection_history = self.detection_history[-self.drift_memory:]
            
            # Store visualization data
            if self.enable_visualization:
                self.visualization_data['timestamps'].append(timestamp)
                self.visualization_data['drift_scores'].append(drift_score)
                self.visualization_data['drift_types'].append(drift_type)
                self.visualization_data['adaptations'].append(self.current_adaptation.copy())
            
            logger.info(f"Drift detected: type={drift_type}, score={drift_score:.4f}")
            return True, drift_type, drift_score, self.current_adaptation
            
        return False, None, 0.0, {}
    
    def update_feature_distribution(self, feature_name, values):
        """Update tracking of feature distributions over time"""
        if feature_name not in self.feature_distributions:
            self.feature_distributions[feature_name] = {
                'values': [],
                'mean_history': [],
                'std_history': []
            }
        
        # Store summarized statistics rather than all values to save memory
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        # Store statistics
        self.feature_distributions[feature_name]['mean_history'].append(mean)
        self.feature_distributions[feature_name]['std_history'].append(std)
        
        # Keep a limited sample of actual values
        max_samples = 1000
        if len(values) > max_samples:
            # Store a random sample to preserve distribution
            indices = np.random.choice(len(values), max_samples, replace=False)
            sample = [values[i] for i in indices]
            self.feature_distributions[feature_name]['values'] = sample
        else:
            self.feature_distributions[feature_name]['values'] = values
        
        # Trim history if needed
        max_history = 100
        if len(self.feature_distributions[feature_name]['mean_history']) > max_history:
            self.feature_distributions[feature_name]['mean_history'] = self.feature_distributions[feature_name]['mean_history'][-max_history:]
            self.feature_distributions[feature_name]['std_history'] = self.feature_distributions[feature_name]['std_history'][-max_history:]
    
    def update_model_performance(self, model_type, timestamp, metric_name, metric_value):
        """Update performance history for a specific model type"""
        if model_type not in self.performance_history:
            self.performance_history[model_type] = []
        
        self.performance_history[model_type].append({
            'timestamp': timestamp,
            'metric_name': metric_name,
            'metric_value': metric_value
        })
        
        # Trim history if needed
        max_history = self.windows['performance'] * 3
        if len(self.performance_history[model_type]) > max_history:
            self.performance_history[model_type] = self.performance_history[model_type][-max_history:]
    
    def update_feature_importance(self, feature_importances):
        """Update feature importance tracking"""
        timestamp = time.time()
        
        for feature, importance in feature_importances.items():
            if feature not in self.feature_importance_history:
                self.feature_importance_history[feature] = []
            
            self.feature_importance_history[feature].append({
                'timestamp': timestamp,
                'importance': importance
            })
            
            # Trim history
            max_history = 50
            if len(self.feature_importance_history[feature]) > max_history:
                self.feature_importance_history[feature] = self.feature_importance_history[feature][-max_history:]
    
    def get_adaptive_window_size(self, base_window=None):
        """
        Get the adaptive window size based on current drift status.
        
        Args:
            base_window: Base window size (uses self.base_window_size if None)
            
        Returns:
            Adapted window size
        """
        base = base_window if base_window is not None else self.base_window_size
        
        # If no recent drift, use base window
        if not self.detection_history:
            return base
        
        # Get most recent drift event and its score
        recent_drift = self.detection_history[-1]
        drift_score = recent_drift['drift_score']
        drift_type = recent_drift['drift_type']
        
        # Different adaptation based on drift type
        if drift_type == 'statistical':
            # Sudden changes - reduce window size significantly
            factor = max(0.3, 1.0 - drift_score * 0.5)
        elif drift_type == 'performance':
            # Performance degradation - moderate reduction
            factor = max(0.5, 1.0 - drift_score * 0.3)
        elif drift_type == 'distribution':
            # Distribution shifts - subtle reduction
            factor = max(0.7, 1.0 - drift_score * 0.2)
        else:
            factor = 1.0
        
        # Calculate time decay - effects diminish over time
        time_since_detection = time.time() - self.last_adaptation_time
        decay_factor = 1.0 - min(1.0, time_since_detection / (60 * 60 * 24))  # 24-hour decay
        
        # Apply impact with time decay
        impact = 1.0 - (1.0 - factor) * decay_factor
        
        # Apply bounds
        adapted_window = max(5, min(int(base * impact), base * 2))
        
        # Record in current adaptation
        self.current_adaptation['window_size'] = adapted_window
        
        return adapted_window
    
    def get_ensemble_weight_adjustments(self, base_weights):
        """
        Get adjusted ensemble weights based on model performance during drift.
        
        Args:
            base_weights: Dictionary of base weights by model type
            
        Returns:
            Dictionary of adjusted weights
        """
        # If no adaptation needed, return original weights
        if not self.detection_history or not self.performance_history:
            return base_weights
        
        # Start with original weights
        adjusted_weights = base_weights.copy()
        
        # Get recent performance for each model type
        relative_performance = {}
        for model_type, history in self.performance_history.items():
            if not history:
                continue
                
            # Get recent and older metrics
            recent_window = min(len(history), self.windows['performance'])
            recent_metrics = [entry['metric_value'] for entry in history[-recent_window:]]
            
            if len(history) > recent_window:
                older_metrics = [entry['metric_value'] for entry in history[-(2*recent_window):-recent_window]]
                
                # Compare recent to older performance
                if older_metrics and recent_metrics:
                    recent_avg = np.mean(recent_metrics)
                    older_avg = np.mean(older_metrics)
                    
                    # Lower is better for error metrics
                    relative_performance[model_type] = older_avg / (recent_avg + 1e-8)
            
            # If we don't have enough history, use recent performance directly
            if model_type not in relative_performance and recent_metrics:
                # Calculate coefficient of variation (stability)
                if len(recent_metrics) > 3:
                    cv = np.std(recent_metrics) / (np.mean(recent_metrics) + 1e-8)
                    relative_performance[model_type] = 1.0 / (cv + 1.0)  # Higher stability = higher weight
        
        # If we have performance data, adjust weights
        if relative_performance:
            # Get drift type
            drift_type = self.detection_history[-1]['drift_type'] if self.detection_history else None
            
            # Adjust weights differently based on drift type
            if drift_type == 'statistical':
                # For sudden changes, favor faster models
                fast_models = ['random_forest', 'xgboost']
                for model in fast_models:
                    if model in adjusted_weights:
                        adjusted_weights[model] *= 1.3
            elif drift_type == 'performance':
                # For performance issues, use relative performance
                for model_type, rel_perf in relative_performance.items():
                    if model_type in adjusted_weights:
                        # Cap the adjustment factor
                        factor = min(2.0, max(0.5, rel_perf))
                        adjusted_weights[model_type] *= factor
            
            # Normalize weights
            total = sum(adjusted_weights.values())
            for model_type in adjusted_weights:
                adjusted_weights[model_type] /= total
                
            # Record adaptation
            self.current_adaptation['weight_adjustments'] = {
                k: round(adjusted_weights[k] / base_weights.get(k, 1.0), 2) 
                for k in adjusted_weights if k in base_weights
            }
        
        return adjusted_weights
    
    def get_feature_importance_shifts(self):
        """
        Detect shifts in feature importance over time.
        
        Returns:
            Dictionary of features with importance shift score
        """
        if not self.feature_importance_history:
            return {}
        
        importance_shifts = {}
        
        for feature, history in self.feature_importance_history.items():
            if len(history) < 5:  # Need minimum history
                continue
                
            # Get recent and older importance values
            recent = [entry['importance'] for entry in history[-5:]]
            
            if len(history) >= 10:
                older = [entry['importance'] for entry in history[-10:-5]]
                
                # Calculate shift in importance
                recent_avg = np.mean(recent)
                older_avg = np.mean(older)
                
                # Calculate relative change
                if older_avg > 0:
                    relative_change = (recent_avg - older_avg) / older_avg
                    importance_shifts[feature] = relative_change
        
        return importance_shifts
    
    def get_visualization_data(self):
        """Get data for visualization of drift detection"""
        if not self.enable_visualization:
            return None
            
        return self.visualization_data
    
    def _detect_drift(self) -> Dict[str, bool]:
        """
        Run all drift detection methods and return results.
        
        Returns:
            Dictionary of detection results by method
        """
        results = {
            'statistical': False,
            'performance': False,
            'distribution': False
        }
        
        # 1. Statistical drift detection (error distribution changes)
        if len(self.error_history) >= self.windows['statistical'] * 2:
            window = self.windows['statistical']
            recent_errors = [entry['relative_error'] for entry in self.error_history[-window:]]
            previous_errors = [entry['relative_error'] for entry in self.error_history[-(window*2):-window]]
            
            # Check for distribution change using Mann-Whitney U test (nonparametric)
            try:
                u_stat, p_value = stats.mannwhitneyu(recent_errors, previous_errors)
                results['statistical'] = p_value < self.thresholds['statistical']
                
                # Alternative: check for significant mean shift
                if not results['statistical']:
                    recent_mean = np.mean(recent_errors)
                    prev_mean = np.mean(previous_errors)
                    pooled_std = np.sqrt((np.std(recent_errors)**2 + np.std(previous_errors)**2) / 2)
                    z_score = abs(recent_mean - prev_mean) / (pooled_std / np.sqrt(window) + 1e-8)
                    results['statistical'] = z_score > self.thresholds['statistical']
            except Exception as e:
                logger.warning(f"Statistical drift detection error: {e}")
        
        # 2. Performance-based drift detection (for each model type)
        if self.performance_history:
            any_perf_drift = False
            
            for model_type, history in self.performance_history.items():
                window = min(len(history), self.windows['performance'] // 2)
                if len(history) >= window * 2:
                    recent = [entry['metric_value'] for entry in history[-window:]]
                    prev = [entry['metric_value'] for entry in history[-(window*2):-window]]
                    
                    if recent and prev:
                        # Calculate percent change in performance
                        recent_avg = np.mean(recent)
                        prev_avg = np.mean(prev)
                        pct_change = (recent_avg - prev_avg) / (prev_avg + 1e-8)
                        
                        # Detect significant degradation
                        if pct_change > self.thresholds['performance']:
                            any_perf_drift = True
                            break
            
            results['performance'] = any_perf_drift
        
        # 3. Distribution-based drift detection (feature distribution changes)
        if self.feature_distributions:
            distribution_shifts = 0
            features_checked = 0
            
            for feature, distribution in self.feature_distributions.items():
                if len(distribution['mean_history']) >= 4:
                    # Check for shift in mean or variance
                    recent_mean = distribution['mean_history'][-1]
                    recent_std = distribution['std_history'][-1]
                    
                    prev_means = distribution['mean_history'][:-1]
                    prev_stds = distribution['std_history'][:-1]
                    
                    # Calculate z-score for current mean
                    mean_z = abs(recent_mean - np.mean(prev_means)) / (np.std(prev_means) + 1e-8)
                    
                    # Calculate z-score for current std
                    std_z = abs(recent_std - np.mean(prev_stds)) / (np.std(prev_stds) + 1e-8)
                    
                    # Count significant shifts
                    if mean_z > 2.0 or std_z > 2.0:
                        distribution_shifts += 1
                    
                    features_checked += 1
            
            # If we have enough features and significant proportion show shifts
            if features_checked >= 3:
                shift_ratio = distribution_shifts / features_checked
                results['distribution'] = shift_ratio > self.thresholds['distribution']
        
        return results
    
    def _calculate_drift_score(self, drift_results):
        """Calculate an overall drift score based on detection results"""
        # Different weights for different drift types
        weights = {
            'statistical': 0.5,  # Higher weight for error distribution changes
            'performance': 0.3,  # Medium weight for performance issues
            'distribution': 0.2   # Lower weight for feature distribution changes
        }
        
        # Calculate weighted score
        score = 0.0
        total_weight = 0.0
        
        for drift_type, detected in drift_results.items():
            if detected:
                score += weights[drift_type]
                total_weight += weights[drift_type]
        
        # Normalize
        if total_weight > 0:
            return score / total_weight
        return 0.0
    
    def _adapt_to_drift(self, drift_type, drift_score, features=None):
        """Generate adaptation response to drift"""
        # Record adaptation time
        self.last_adaptation_time = time.time()
        
        # Start with empty adaptation
        self.current_adaptation = {
            'drift_type': drift_type,
            'drift_score': drift_score,
            'timestamp': self.last_adaptation_time
        }
        
        # Different adaptation strategies based on drift type
        if drift_type == 'statistical':
            # For statistical drift, adjust window size and learning rate
            self.current_adaptation['window_size_factor'] = max(0.3, 1.0 - drift_score * 0.5)
            self.current_adaptation['learning_rate_factor'] = min(3.0, 1.0 + drift_score * 2.0)
            
            # Record volatility adaptation
            self.current_adaptation['volatility_factor'] = min(2.0, 1.0 + drift_score)
        
        elif drift_type == 'performance':
            # For performance drift, adjust model weights
            self.current_adaptation['ensemble_weight_strategy'] = 'performance_based'
            
            # Record retraining trigger
            self.current_adaptation['retrain_triggered'] = drift_score > 0.7
        
        elif drift_type == 'distribution':
            # For distribution drift, focus on feature relationships
            self.current_adaptation['feature_importance_update'] = True
            
            # Suggest feature engineering reevaluation
            if drift_score > 0.6 and features is not None:
                self.current_adaptation['feature_engineering_review'] = True
                
                # Store feature distribution stats if available
                if self.feature_distributions:
                    shifted_features = []
                    
                    for feature, dist in self.feature_distributions.items():
                        if len(dist['mean_history']) >= 3:
                            recent = dist['mean_history'][-1]
                            prev_avg = np.mean(dist['mean_history'][:-1])
                            
                            # Calculate percent shift
                            pct_shift = abs(recent - prev_avg) / (abs(prev_avg) + 1e-8)
                            
                            if pct_shift > 0.2:  # 20% shift threshold
                                shifted_features.append(feature)
                    
                    if shifted_features:
                        self.current_adaptation['shifted_features'] = shifted_features
        
        logger.info(f"Adaptation generated for {drift_type} drift: {self.current_adaptation}")