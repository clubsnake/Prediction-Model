"""
Provides comprehensive prediction confidence metrics for ensemble models.
Combines multiple approaches to estimate forecast reliability.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ConfidenceCalculator:
    """
    Calculates prediction confidence using multiple metrics:
    1. Ensemble agreement: Variance among ensemble models
    2. Historical accuracy: Performance on similar past conditions
    3. Market volatility: Current market condition adjustment
    4. Model-specific uncertainty: Leveraging inherent model uncertainty
    """
    
    def __init__(self, 
                 volatility_window: int = 20,
                 weight_ensemble: float = 0.4,
                 weight_historical: float = 0.3,
                 weight_volatility: float = 0.2,
                 weight_model: float = 0.1,
                 scale_factor: float = 1.0):
        """
        Initialize the confidence calculator.
        
        Args:
            volatility_window: Window size for volatility calculation
            weight_ensemble: Weight for ensemble agreement confidence
            weight_historical: Weight for historical accuracy confidence
            weight_volatility: Weight for volatility-based confidence
            weight_model: Weight for model-specific confidence
            scale_factor: Scaling factor for overall confidence
        """
        self.volatility_window = volatility_window
        self.weights = {
            'ensemble': weight_ensemble,
            'historical': weight_historical,
            'volatility': weight_volatility,
            'model': weight_model
        }
        self.scale_factor = scale_factor
        self.error_history = {}
        
    def calculate_confidence(self,
                             predictions: Dict[str, np.ndarray],
                             weights: Dict[str, float],
                             historical_data: pd.DataFrame,
                             past_predictions: Optional[Dict[str, Dict]] = None,
                             model_uncertainties: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Calculate comprehensive confidence scores for predictions.
        
        Args:
            predictions: Dictionary of model predictions by model type
            weights: Dictionary of ensemble weights by model type
            historical_data: DataFrame with historical price data
            past_predictions: Dictionary of past predictions with actuals
            model_uncertainties: Model-specific uncertainty metrics (optional)
            
        Returns:
            Tuple of (confidence_scores, confidence_components)
        """
        # Initialize confidence arrays (one score per prediction timestep)
        if not predictions or len(predictions) == 0:
            logger.warning("No predictions available for confidence calculation")
            return np.zeros(1), {}
            
        # Get shape from first available prediction
        for model_type, preds in predictions.items():
            if preds is not None and len(preds) > 0:
                prediction_length = preds.shape[0]
                break
        else:
            logger.warning("Could not determine prediction shape")
            return np.zeros(1), {}
        
        # Initialize confidence components
        ensemble_confidence = np.ones(prediction_length)
        historical_confidence = np.ones(prediction_length)
        volatility_confidence = np.ones(prediction_length)
        model_confidence = np.ones(prediction_length)
        
        # 1. Calculate ensemble agreement confidence
        ensemble_confidence = self._calculate_ensemble_agreement(predictions, weights)
        
        # 2. Calculate historical accuracy confidence
        if past_predictions:
            historical_confidence = self._calculate_historical_confidence(
                past_predictions, prediction_length)
        
        # 3. Calculate volatility-based confidence
        if historical_data is not None and 'Close' in historical_data.columns:
            volatility_confidence = self._calculate_volatility_confidence(
                historical_data, prediction_length)
        
        # 4. Incorporate model-specific uncertainty if available
        if model_uncertainties:
            model_confidence = self._calculate_model_confidence(
                model_uncertainties, weights, prediction_length)
        
        # Combine confidence components with weights
        combined_confidence = (
            self.weights['ensemble'] * ensemble_confidence +
            self.weights['historical'] * historical_confidence +
            self.weights['volatility'] * volatility_confidence +
            self.weights['model'] * model_confidence
        )
        
        # Scale confidence to 0-100 range and apply scaling factor
        confidence_scores = np.clip(combined_confidence * 100 * self.scale_factor, 0, 100)
        
        # Create dictionary of confidence components for analysis
        confidence_components = {
            'ensemble': ensemble_confidence,
            'historical': historical_confidence,
            'volatility': volatility_confidence,
            'model': model_confidence,
            'combined': combined_confidence,
            'final': confidence_scores
        }
        
        # Record confidence with scheduler if available in streamlit session state
        try:
            import streamlit as st
            if "drift_tuning_scheduler" in st.session_state:
                # Use ticker and timeframe from past_predictions if available
                if past_predictions and len(past_predictions) > 0:
                    latest_date = max(past_predictions.keys())
                    prediction_info = past_predictions[latest_date]
                    
                    if "ticker" in prediction_info and "timeframe" in prediction_info:
                        ticker = prediction_info["ticker"]
                        timeframe = prediction_info["timeframe"]
                        
                        # Record average confidence
                        avg_confidence = float(np.mean(confidence_scores))
                        st.session_state["drift_tuning_scheduler"].record_confidence(
                            ticker, timeframe, avg_confidence
                        )
        except Exception as e:
            logger.debug(f"Could not record confidence with scheduler: {e}")
        
        # Check for drift and adjust confidence if necessary
        drift_detected = False
        drift_score = 0.0
        
        try:
            import streamlit as st
            import time
            
            # Use drift detector if available in session state
            if "drift_detector" in st.session_state:
                drift_detector = st.session_state["drift_detector"]
                # Check for drift using first prediction/actual pair
                if prediction_length > 0 and predictions and len(predictions) > 0:
                    first_key = list(predictions.keys())[0]
                    first_pred = predictions[first_key][0]
                    first_actual = None
                    
                    # Try to get actual value from past_predictions
                    if past_predictions and len(past_predictions) > 0:
                        latest_date = max(past_predictions.keys())
                        if latest_date in past_predictions and 'actual' in past_predictions[latest_date]:
                            first_actual = past_predictions[latest_date]['actual']
                    
                    # If we have both prediction and actual, check for drift
                    if first_actual is not None:
                        drift_result, drift_type, drift_score, _ = drift_detector.update_error(
                            timestamp=time.time(),
                            actual=first_actual,
                            predicted=first_pred
                        )
                        drift_detected = drift_result
            else:
                # Fallback to simple detection if no detector available
                drift_detected, drift_score = self.detect_drift_from_confidence(
                    lookback=min(20, prediction_length),
                    threshold_factor=2.0
                )
        except Exception as e:
            logger.debug(f"Could not perform drift detection: {e}")
            # Fallback to internal drift detection
            try:
                drift_detected, drift_score = self.detect_drift_from_confidence(
                    lookback=10,
                    threshold_factor=2.0
                )
            except Exception as inner_e:
                logger.error(f"Fallback drift detection failed: {inner_e}")
        
        if drift_detected:
            # Adjust confidence based on drift
            confidence_reducer = max(0.5, 1.0 - drift_score * 0.3)
            confidence_scores = confidence_scores * confidence_reducer
            
            # Add drift information to components
            confidence_components['drift_detected'] = drift_detected
            confidence_components['drift_score'] = drift_score
            confidence_components['drift_adjuster'] = confidence_reducer
        
        return confidence_scores, confidence_components
        
    def _calculate_ensemble_agreement(self, 
                                     predictions: Dict[str, np.ndarray],
                                     weights: Dict[str, float]) -> np.ndarray:
        """
        Calculate confidence based on agreement between ensemble models.
        Lower variance = higher confidence.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            
        Returns:
            Array of confidence scores (0-1) based on ensemble agreement
        """
        try:
            # Collect all predictions in a list
            all_preds = []
            all_weights = []
            
            for model_type, preds in predictions.items():
                if preds is None or model_type not in weights or weights[model_type] <= 0:
                    continue
                    
                # Ensure prediction is 2D and has correct shape
                if len(preds.shape) == 1:
                    preds = preds.reshape(-1, 1)
                
                all_preds.append(preds)
                all_weights.append(weights[model_type])
            
            if not all_preds:
                return np.ones(1)  # Default confidence if no predictions
                
            # Stack predictions and calculate weighted variance
            stacked_preds = np.stack(all_preds, axis=-1)  # shape: (time_steps, horizon, n_models)
            
            # Reshape to handle different input shapes
            if len(stacked_preds.shape) == 3:
                # For multiple time steps and horizon
                n_timesteps, horizon, n_models = stacked_preds.shape
                # Calculate variance across models for each timestep and horizon
                variance = np.var(stacked_preds, axis=2)  # Shape: (time_steps, horizon)
                # Average variance across horizon for each timestep
                avg_variance = np.mean(variance, axis=1)  # Shape: (time_steps,)
            else:
                # For single dimension predictions
                avg_variance = np.var(stacked_preds, axis=1)
            
            # Normalize predictions to make variance comparable
            mean_preds = np.mean(stacked_preds, axis=-1)
            normalized_variance = avg_variance / (np.abs(mean_preds) + 1e-8)
            
            # Convert variance to confidence (higher variance = lower confidence)
            # Using exponential decay function: conf = exp(-k * norm_var)
            confidence = np.exp(-2.0 * normalized_variance)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating ensemble agreement: {e}")
            return np.ones(1)  # Default to neutral confidence
            
    def _calculate_historical_confidence(self, 
                                       past_predictions: Dict[str, Dict],
                                       prediction_length: int) -> np.ndarray:
        """
        Calculate confidence based on historical prediction accuracy.
        
        Args:
            past_predictions: Dictionary of past predictions with actuals
            prediction_length: Length of prediction horizon
            
        Returns:
            Array of confidence scores based on historical performance
        """
        try:
            if not past_predictions:
                return np.ones(prediction_length)
                
            # Extract errors from past predictions
            errors = []
            dates = []
            
            for date_str, pred_info in past_predictions.items():
                if pred_info.get('actual') is not None and pred_info.get('predicted') is not None:
                    error_pct = abs(pred_info['predicted'] - pred_info['actual']) / pred_info['actual']
                    errors.append(error_pct)
                    dates.append(date_str)
            
            if not errors:
                return np.ones(prediction_length)
                
            # Calculate historical error statistics
            mean_error = np.mean(errors)
            
            # More recent errors are more relevant - apply exponential decay
            if len(errors) > 1:
                recent_errors = errors[-min(10, len(errors)):]
                weighted_mean_error = np.mean(recent_errors)
                mean_error = (weighted_mean_error + mean_error) / 2
            
            # Convert error to confidence
            # Using exponential function: conf = exp(-k * error)
            base_confidence = np.exp(-5.0 * mean_error)
            
            # Decrease confidence as horizon increases
            horizon_decay = np.exp(-0.05 * np.arange(prediction_length))
            confidence = base_confidence * horizon_decay
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating historical confidence: {e}")
            return np.ones(prediction_length)
            
    def _calculate_volatility_confidence(self, 
                                       historical_data: pd.DataFrame,
                                       prediction_length: int) -> np.ndarray:
        """
        Calculate confidence based on market volatility.
        Higher volatility = lower confidence.
        
        Args:
            historical_data: DataFrame with historical price data
            prediction_length: Length of prediction horizon
            
        Returns:
            Array of confidence scores based on volatility
        """
        try:
            # Calculate historical volatility
            if 'Close' not in historical_data.columns:
                return np.ones(prediction_length)
                
            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
            
            # Annualize the volatility
            annualized_vol = volatility * np.sqrt(252)  # 252 trading days in a year
            
            # Convert volatility to confidence
            # Higher volatility = lower confidence
            # Using exponential function: conf = exp(-k * vol)
            base_confidence = np.exp(-10.0 * annualized_vol)
            
            # Decrease confidence as horizon increases
            horizon_decay = np.exp(-0.03 * np.arange(prediction_length))
            confidence = base_confidence * horizon_decay
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating volatility confidence: {e}")
            return np.ones(prediction_length)
            
    def _calculate_model_confidence(self,
                                  model_uncertainties: Dict[str, np.ndarray],
                                  weights: Dict[str, float],
                                  prediction_length: int) -> np.ndarray:
        """
        Calculate confidence from model-specific uncertainty metrics.
        
        Args:
            model_uncertainties: Model-specific uncertainty values
            weights: Dictionary of model weights
            prediction_length: Length of prediction horizon
            
        Returns:
            Array of confidence scores based on model uncertainties
        """
        try:
            if not model_uncertainties:
                return np.ones(prediction_length)
                
            # Combine model uncertainties using weights
            weighted_uncertainty = np.zeros(prediction_length)
            total_weight = 0
            
            for model_type, uncertainty in model_uncertainties.items():
                if model_type in weights and weights[model_type] > 0:
                    # Ensure uncertainty array matches prediction length
                    if len(uncertainty) < prediction_length:
                        padded = np.pad(uncertainty, (0, prediction_length - len(uncertainty)), 
                                       mode='edge')
                        uncertainty = padded
                    elif len(uncertainty) > prediction_length:
                        uncertainty = uncertainty[:prediction_length]
                        
                    weighted_uncertainty += weights[model_type] * uncertainty
                    total_weight += weights[model_type]
            
            if total_weight > 0:
                weighted_uncertainty /= total_weight
                
            # Convert uncertainty to confidence
            confidence = 1.0 - np.clip(weighted_uncertainty, 0, 1)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating model confidence: {e}")
            return np.ones(prediction_length)
    
    def update_error_history(self, prediction_date: str, predicted: float, actual: float):
        """
        Update the error history with a new prediction-actual pair.
        
        Args:
            prediction_date: Date string for the prediction
            predicted: Predicted value
            actual: Actual observed value
        """
        if actual is not None and predicted is not None:
            error = abs(predicted - actual)
            error_pct = error / (abs(actual) + 1e-8)
            
            self.error_history[prediction_date] = {
                'predicted': predicted,
                'actual': actual,
                'error': error,
                'error_pct': error_pct
            }


# Confidence Drift Detection
def detect_drift_from_confidence(self, lookback=10, threshold_factor=2.0):
    """
    Use confidence metrics to detect drift in model performance.
    
    Args:
        lookback: Number of recent predictions to analyze
        threshold_factor: Threshold multiplier for drift detection
        
    Returns:
        (drift_detected, drift_score): Whether drift was detected and its intensity
    """
    # Need sufficient history
    if len(self.error_history) < lookback * 2:
        return False, 0.0
    
    # Sort by date
    sorted_dates = sorted(self.error_history.keys())
    recent = sorted_dates[-lookback:]
    previous = sorted_dates[-2*lookback:-lookback]
    
    # Get error values
    recent_errors = [self.error_history[date]['error_pct'] for date in recent]
    prev_errors = [self.error_history[date]['error_pct'] for date in previous]
    
    # Calculate statistics
    recent_mean = np.mean(recent_errors)
    prev_mean = np.mean(prev_errors)
    recent_std = np.std(recent_errors)
    prev_std = np.std(prev_errors)
    
    # Check for significant changes in error distribution
    # 1. Mean shift
    pooled_std = np.sqrt((recent_std**2 + prev_std**2) / 2)
    mean_shift_score = abs(recent_mean - prev_mean) / (pooled_std + 1e-8)
    
    # 2. Volatility change
    volatility_ratio = recent_std / (prev_std + 1e-8)
    vol_shift_score = abs(volatility_ratio - 1.0)
    
    # 3. Combined score
    drift_score = (mean_shift_score + vol_shift_score) / 2
    
    # Detect drift if score exceeds threshold
    drift_detected = drift_score > threshold_factor
    
    return drift_detected, drift_score
def adjust_confidence_for_drift(self, drift_detected, drift_score, confidence_scores):
    """
    Adjust confidence scores when drift is detected.
    
    Args:
        drift_detected: Whether drift was detected
        drift_score: Intensity of detected drift
        confidence_scores: Original confidence scores
        
    Returns:
        Adjusted confidence scores
    """
    if not drift_detected or drift_score <= 0:
        return confidence_scores
    
    # Scale confidence based on drift intensity
    # More intense drift = lower confidence
    reduction_factor = max(0.5, 1.0 - drift_score * 0.3)
    
    # Apply reduction to confidence scores
    adjusted_scores = confidence_scores * reduction_factor
    
    return adjusted_scores