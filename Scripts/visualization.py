# visualization.py
"""
Matplotlib-based visualizations for training history, weights, and predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from config import SHOW_TRAINING_HISTORY, SHOW_WEIGHT_HISTOGRAMS, SHOW_PREDICTION_PLOTS

def visualize_training_history(history):
    """
    Create training/validation loss plot using Matplotlib.
    
    Returns:
        matplotlib figure that can be displayed with plt.show() or st.pyplot()
    """
    if not SHOW_TRAINING_HISTORY:
        print("Training history visualization is disabled by config.")
        return None
        
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    
    return fig

def visualize_weight_histograms(model):
    """
    Create histograms of model weights if SHOW_WEIGHT_HISTOGRAMS is enabled.
    
    Returns:
        list of matplotlib figures that can be displayed with plt.show() or st.pyplot()
    """
    if not SHOW_WEIGHT_HISTOGRAMS:
        print("Weight histogram visualization is disabled by config.")
        return []
    
    figures = []
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for idx, w in enumerate(weights):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(w.flatten(), bins=50)
                ax.set_title(f"Histogram of weights for layer {layer.name} (Weight {idx})")
                ax.set_xlabel("Weight value")
                ax.set_ylabel("Frequency")
                figures.append(fig)
                
    return figures

def visualize_predictions(y_true, y_pred, sample_idx=0):
    """
    Create plot of predicted vs. actual series for a single sample.
    
    Returns:
        matplotlib figure that can be displayed with plt.show() or st.pyplot()
    """
    if not SHOW_PREDICTION_PLOTS:
        print("Prediction visualization is disabled by config.")
        return None
        
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true[sample_idx], label="True Prices")
    ax.plot(y_pred[sample_idx], label="Predicted Prices")
    ax.set_title(f"Price Predictions vs True Prices (Sample {sample_idx})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized Price")
    ax.legend()
    
    return fig

def plot_combined_validation_and_forecast(true_prices, predicted_prices, forecast_prices, 
                                          dates, forecast_dates, validation_rmse, validation_mape):
    """
    Combine validation and forecast data into a single line plot.
    
    Returns:
        matplotlib figure that can be displayed with plt.show() or st.pyplot()
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, true_prices, marker="o", linestyle="-", label="True (Validation)")
    ax.plot(dates, predicted_prices, marker="x", linestyle="--", label="Predicted (Validation)")
    ax.plot(forecast_dates, forecast_prices, linestyle="dashed", label="Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Combined Validation and Future Forecast")
    ax.legend()
    error_text = f"Validation RMSE: {validation_rmse:.4f}\nValidation MAPE: {validation_mape:.2f}%"
    ax.text(dates[0], max(true_prices) * 0.95, error_text, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7))
    ax.grid(True)
    
    # Return the figure instead of showing it
    return fig
