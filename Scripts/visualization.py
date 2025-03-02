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
    Show training/validation loss using Matplotlib, 
    if SHOW_TRAINING_HISTORY is enabled in config.
    """
    if not SHOW_TRAINING_HISTORY:
        print("Training history visualization is disabled by config.")
        return
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def visualize_weight_histograms(model):
    """
    Show histograms of model weights if SHOW_WEIGHT_HISTOGRAMS is enabled.
    """
    if not SHOW_WEIGHT_HISTOGRAMS:
        print("Weight histogram visualization is disabled by config.")
        return
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for idx, w in enumerate(weights):
                plt.figure(figsize=(8, 4))
                plt.hist(w.flatten(), bins=50)
                plt.title(f"Histogram of weights for layer {layer.name} (Weight {idx})")
                plt.xlabel("Weight value")
                plt.ylabel("Frequency")
                plt.show()

def visualize_predictions(y_true, y_pred, sample_idx=0):
    """
    Plot predicted vs. actual series for a single sample in the batch if enabled.
    """
    if not SHOW_PREDICTION_PLOTS:
        print("Prediction visualization is disabled by config.")
        return
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[sample_idx], label="True Prices")
    plt.plot(y_pred[sample_idx], label="Predicted Prices")
    plt.title(f"Price Predictions vs True Prices (Sample {sample_idx})")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.show()

def plot_combined_validation_and_forecast(true_prices, predicted_prices, forecast_prices, 
                                          dates, forecast_dates, validation_rmse, validation_mape):
    """
    Combine validation and forecast data into a single line plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, true_prices, marker="o", linestyle="-", label="True (Validation)")
    plt.plot(dates, predicted_prices, marker="x", linestyle="--", label="Predicted (Validation)")
    plt.plot(forecast_dates, forecast_prices, linestyle="dashed", label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Combined Validation and Future Forecast")
    plt.legend()
    error_text = f"Validation RMSE: {validation_rmse:.4f}\nValidation MAPE: {validation_mape:.2f}%"
    plt.text(dates[0], max(true_prices) * 0.95, error_text, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7))
    plt.grid(True)
    plt.show()
