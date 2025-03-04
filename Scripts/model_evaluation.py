"""
Utility functions for evaluating and comparing models.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger(__name__)

def measure_model_inference_time(
    model: Any, 
    X: np.ndarray, 
    n_runs: int = 5, 
    warmup: bool = True
) -> float:
    """
    Measure average inference time (in seconds) for a given model.
    
    Args:
        model: A trained model with a predict() method
        X: Input data for prediction
        n_runs: Number of runs to average over
        warmup: Whether to run a warmup prediction before timing
        
    Returns:
        Average inference time per prediction (seconds)
    """
    # Optional warmup call to initialize any lazy loading
    if warmup and hasattr(model, 'predict'):
        try:
            _ = model.predict(X)
        except Exception as e:
            logger.warning(f"Warmup prediction failed: {str(e)}")
    
    # Measure prediction time
    times = []
    for _ in range(n_runs):
        try:
            start = time.time()
            _ = model.predict(X)
            end = time.time()
            times.append(end - start)
        except Exception as e:
            logger.error(f"Error during inference timing: {str(e)}")
            return float('nan')
    
    # Return average time
    return np.mean(times)

def benchmark_models(
    models: Dict[str, Any], 
    X: np.ndarray, 
    n_runs: int = 5
) -> pd.DataFrame:
    """
    Benchmark multiple models and compare their inference times.
    
    Args:
        models: Dictionary mapping model names to model objects
        X: Input data for prediction
        n_runs: Number of runs to average over
        
    Returns:
        DataFrame with model names and inference times
    """
    results = []
    
    for model_name, model in models.items():
        try:
            # Measure inference time
            avg_time = measure_model_inference_time(model, X, n_runs)
            
            # Store results
            results.append({
                'model_name': model_name,
                'avg_inference_time': avg_time,
                'runs': n_runs
            })
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {str(e)}")
            results.append({
                'model_name': model_name,
                'avg_inference_time': float('nan'),
                'runs': 0,
                'error': str(e)
            })
    
    # Create DataFrame from results
    return pd.DataFrame(results)

def plot_inference_times(inference_times: Dict[str, float]) -> plt.Figure:
    """
    Create a bar chart of model inference times.
    
    Args:
        inference_times: Dictionary mapping model names to inference times
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort models by inference time
    sorted_items = sorted(inference_times.items(), key=lambda x: x[1])
    model_names = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]
    
    # Create bars with custom colors
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(
        model_names, 
        times, 
        color=bar_colors[:len(model_names)]
    )
    
    # Add data labels to bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height * 1.01,
            f'{height:.5f}s',
            ha='center', va='bottom', 
            rotation=0
        )
    
    # Customize plot
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Average Inference Time (seconds)")
    ax.set_title("Inference Time Comparison Between Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def calculate_model_metrics(
    y_true: np.ndarray, 
    y_pred_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Calculate various metrics for multiple models.
    
    Args:
        y_true: Ground truth values
        y_pred_dict: Dictionary mapping model names to predictions
        
    Returns:
        DataFrame with model metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = []
    
    for model_name, y_pred in y_pred_dict.items():
        try:
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # MAPE calculation with handling for zeros
            epsilon = 1e-10  # Small value to avoid division by zero
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
            
            r2 = r2_score(y_true, y_pred)
            
            # Store in metrics list
            metrics.append({
                'model_name': model_name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R²': r2
            })
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
            metrics.append({
                'model_name': model_name,
                'error': str(e)
            })
    
    # Create DataFrame from metrics
    return pd.DataFrame(metrics)

def load_example_data() -> pd.DataFrame:
    """
    Load example data for demonstration purposes.
    In a real application, this would load from CSV or API.
    """
    # Create a simple simulated price series
    np.random.seed(42)
    n = 500  # fixed the undefined variable (was "a500")
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Generate random walk with drift for price
    returns = np.random.normal(0.0005, 0.015, n)
    price = 100 * (1 + returns).cumprod()
    
    # Generate volume data
    volume = np.random.lognormal(10, 1, n) * (1 + abs(returns) * 10)
    
    df = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.normal(0, 0.005, n)),
        'high': price * (1 + abs(np.random.normal(0, 0.012, n))),
        'low': price * (1 - abs(np.random.normal(0, 0.012, n))),
        'close': price,
        'volume': volume
    })
    
    df = df.set_index('date')
    return df
