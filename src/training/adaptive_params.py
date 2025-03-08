"""
Utilities for adaptive parameter adjustment based on market conditions and model performance.
This module provides functions to dynamically adjust hyperparameters based on data characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def calculate_volatility(data: pd.DataFrame, column: str = 'Close', window: int = 20) -> float:
    """
    Calculate market volatility from price data.
    
    Args:
        data: DataFrame containing price data
        column: Column name containing price data
        window: Window size for volatility calculation
        
    Returns:
        Volatility estimate (standard deviation of returns)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    # Calculate returns
    returns = data[column].pct_change().dropna()
    
    # Calculate volatility (standard deviation of returns)
    return returns.rolling(window=window).std().mean() * np.sqrt(252)  # Annualized volatility

def adaptive_window_size(data: pd.DataFrame, 
                        base_window: int = 30,
                        min_window: int = 5, 
                        max_window: int = 60,
                        column: str = 'Close') -> int:
    """
    Calculate adaptive window size based on market volatility.
    
    Args:
        data: DataFrame containing price data
        base_window: Base window size
        min_window: Minimum window size allowed
        max_window: Maximum window size allowed
        column: Column name containing price data
        
    Returns:
        Adapted window size
    """
    try:
        # Calculate market volatility
        volatility = calculate_volatility(data, column)
        
        # Default to base_window if volatility calculation fails
        if pd.isna(volatility) or volatility <= 0:
            return base_window
        
        # Scale window size inversely with volatility:
        # - Higher volatility → smaller window (more adaptability)
        # - Lower volatility → larger window (more stability)
        
        # Typical volatility range for financial markets: 0.1 to 0.4 (10% to 40%)
        # Normalize volatility to 0-1 range for reasonable window scaling
        normalized_vol = min(max(volatility / 0.4, 0), 1)
        
        # Calculate adaptive window: large when volatility is low, small when volatility is high
        window_size = max_window - (max_window - min_window) * normalized_vol
        
        # Round to nearest integer and enforce min/max bounds
        window_size = int(round(window_size))
        window_size = max(min_window, min(window_size, max_window))
        
        logger.info(f"Adaptive window size: {window_size} (volatility: {volatility:.4f})")
        return window_size
    
    except Exception as e:
        logger.warning(f"Error calculating adaptive window size: {e}. Using base_window {base_window}.")
        return base_window

def adaptive_retraining_threshold(performance_history: Dict[str, float],
                                 volatility: Optional[float] = None,
                                 base_threshold: float = 0.1,
                                 min_threshold: float = 0.02,
                                 max_threshold: float = 0.3) -> float:
    """
    Calculate adaptive retraining threshold based on performance history and market volatility.
    
    Args:
        performance_history: Dict containing model performance metrics over time
        volatility: Market volatility (optional, will be used if provided)
        base_threshold: Base retraining threshold
        min_threshold: Minimum threshold allowed
        max_threshold: Maximum threshold allowed
        
    Returns:
        Adapted retraining threshold
    """
    try:
        # If we have performance history, analyze variation in performance
        if performance_history and len(performance_history) > 3:
            # Extract performance metrics
            metrics = list(performance_history.values())
            
            # Calculate performance volatility
            performance_std = np.std(metrics)
            performance_mean = np.mean(metrics)
            
            # Calculate coefficient of variation
            cv = performance_std / performance_mean if performance_mean > 0 else 1.0
            
            # Normalize CV to 0-1 range for threshold scaling
            normalized_cv = min(max(cv / 0.5, 0), 1)
            
            # Start with base threshold
            threshold = base_threshold
            
            # Adjust based on performance volatility
            # Higher CV → lower threshold (more sensitive to changes)
            # Lower CV → higher threshold (more stable)
            threshold_range = max_threshold - min_threshold
            threshold = max_threshold - threshold_range * normalized_cv
            
            # Further adjust based on market volatility if provided
            if volatility is not None and not pd.isna(volatility):
                # Normalize market volatility to 0-1 range
                norm_market_vol = min(max(volatility / 0.4, 0), 1)
                
                # Lower threshold when market is more volatile
                threshold *= (1.0 - 0.5 * norm_market_vol)
        else:
            # Not enough history, use base threshold
            threshold = base_threshold
            
        # Ensure threshold is within allowed range
        threshold = max(min_threshold, min(threshold, max_threshold))
        
        return threshold
        
    except Exception as e:
        logger.warning(f"Error calculating adaptive retraining threshold: {e}. Using base_threshold {base_threshold}.")
        return base_threshold

def adaptive_learning_rate(current_lr: float, 
                         performance_history: Dict[str, float],
                         iterations_without_improvement: int = 0,
                         min_lr: float = 1e-6,
                         max_lr: float = 1e-2) -> float:
    """
    Adaptively adjust learning rate based on performance history and improvement trends.
    
    Args:
        current_lr: Current learning rate
        performance_history: Dict containing model performance metrics over time
        iterations_without_improvement: Number of iterations without improvement
        min_lr: Minimum learning rate allowed
        max_lr: Maximum learning rate allowed
        
    Returns:
        Adapted learning rate
    """
    try:
        # Base case - no adaptation needed
        if not performance_history or len(performance_history) < 3:
            return current_lr
            
        # Extract recent performance values
        recent_values = list(performance_history.values())[-3:]
        
        # Check for plateau (similar values)
        differences = [abs(recent_values[i] - recent_values[i-1]) for i in range(1, len(recent_values))]
        avg_diff = sum(differences) / len(differences)
        
        # If we're plateauing and not improving enough
        if avg_diff < 0.01 * abs(np.mean(recent_values)):
            # Reduce learning rate
            new_lr = current_lr * 0.5
        elif iterations_without_improvement > 5:
            # Reduce learning rate if no improvement for several iterations
            new_lr = current_lr * 0.7
        else:
            # Keep current learning rate
            new_lr = current_lr
            
        # Ensure learning rate is within allowed range
        new_lr = max(min_lr, min(new_lr, max_lr))
        
        # Log if learning rate changed
        if new_lr != current_lr:
            logger.info(f"Adaptive learning rate adjusted: {current_lr:.2e} → {new_lr:.2e}")
            
        return new_lr
        
    except Exception as e:
        logger.warning(f"Error adjusting learning rate: {e}. Keeping current_lr {current_lr}.")
        return current_lr
