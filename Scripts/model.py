# model.py
"""
Core model-building logic for LSTM/RNN/Tree-based approaches,
as well as utility for evaluating predictions and saving/loading weights.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, InputLayer, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Scripts.callbacks import DynamicLearningRateScheduler
from config import LOOKBACK

class FeatureWeightingLayer(tf.keras.layers.Layer):
    """
    A custom layer that multiplies each input feature by a specified weight.
    """
    def __init__(self, weights, **kwargs):
        super(FeatureWeightingLayer, self).__init__(**kwargs)
        self.weights_vector = tf.constant(weights, dtype=tf.float32)

    def call(self, inputs):
        return inputs * self.weights_vector

def build_lstm_model(num_features, horizon, learning_rate=0.001, dropout_rate=0.2,
                     loss_function="mean_squared_error", lookback=LOOKBACK,
                     units_per_layer=[64], feature_weights=None):
    model = Sequential()
    model.add(InputLayer(input_shape=(lookback, num_features)))
    if feature_weights is not None:
        model.add(FeatureWeightingLayer(feature_weights))
    for i, units in enumerate(units_per_layer):
        return_seq = (i < len(units_per_layer) - 1)
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    model.add(Dense(horizon))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function)
    return model

def build_rnn_model(num_features, horizon, learning_rate=0.001, dropout_rate=0.2,
                    loss_function="mean_squared_error", lookback=LOOKBACK,
                    units_per_layer=[64], feature_weights=None):
    model = Sequential()
    model.add(InputLayer(input_shape=(lookback, num_features)))
    if feature_weights is not None:
        model.add(FeatureWeightingLayer(feature_weights))
    for i, units in enumerate(units_per_layer):
        return_seq = (i < len(units_per_layer) - 1)
        model.add(SimpleRNN(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    model.add(Dense(horizon))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function)
    return model

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def build_random_forest_model(num_features, horizon, n_estimators=100, max_depth=None):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

def build_xgboost_model(num_features, horizon, learning_rate=0.001, n_estimators=100):
    return xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)

def build_tft_model(num_features, horizon, learning_rate=0.001, dropout_rate=0.2,
                    loss_function="mean_squared_error", lookback=LOOKBACK,
                    units_per_layer=[64], feature_weights=None):
    """
    Placeholder example if you want to integrate a TFT (Temporal Fusion Transformer).
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(lookback, num_features)))
    if feature_weights is not None:
        model.add(FeatureWeightingLayer(feature_weights))
    for i, units in enumerate(units_per_layer):
        return_seq = (i < len(units_per_layer) - 1)
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    model.add(Dense(horizon))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function)
    return model

def build_model_by_type(model_type, num_features, horizon, learning_rate, dropout_rate, loss_function, lookback, architecture_params=None):
    if architecture_params is None:
        architecture_params = {}
    
    units_per_layer = architecture_params.get("units_per_layer", [64, 32])
    use_batch_norm = architecture_params.get("use_batch_norm", False)  # Get batch norm param
    
    model = Sequential()
    
    # First layer
    if model_type == "lstm":
        model.add(LSTM(units_per_layer[0], activation="relu", return_sequences=len(units_per_layer) > 1, 
                      input_shape=(lookback, num_features)))
    elif model_type == "rnn":
        model.add(SimpleRNN(units_per_layer[0], activation="relu", return_sequences=len(units_per_layer) > 1,
                           input_shape=(lookback, num_features)))
                           
    # Add batch normalization if enabled
    if use_batch_norm:
        model.add(BatchNormalization())
        
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for i in range(1, len(units_per_layer)):
        is_last_layer = (i == len(units_per_layer) - 1)
        if model_type == "lstm":
            model.add(LSTM(units_per_layer[i], activation="relu", return_sequences=not is_last_layer))
        elif model_type == "rnn":
            model.add(SimpleRNN(units_per_layer[i], activation="relu", return_sequences=not is_last_layer))
            
        # Add batch normalization after each layer if enabled
        if use_batch_norm:
            model.add(BatchNormalization())
            
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(horizon))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function)
    
    return model

def evaluate_predictions(true_values: np.ndarray, predicted_values: np.ndarray):
    """
    Compute MSE and MAPE metrics.
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    mse = np.mean((true_values - predicted_values) ** 2)
    epsilon = 1e-7
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + epsilon))) * 100
    return mse, mape

def save_model_weights(model, ticker, timeframe, range_cat, horizon=30):  # <-- CHANGED to include horizon
    """
    Save model weights to an H5 file named with the ticker/timeframe/range/horizon.
    """
    os.makedirs("saved_weights", exist_ok=True)
    filename = f"saved_weights/best_weights_{ticker}_{timeframe}_{range_cat}_H{horizon}.h5"  # <-- CHANGED
    if hasattr(model, "save_weights"):
        model.save_weights(filename)

def load_model_weights(model, ticker, timeframe, range_cat, horizon=30):  # <-- CHANGED
    """
    Load model weights from an H5 file if available.
    """
    filename = f"saved_weights/best_weights_{ticker}_{timeframe}_{range_cat}_H{horizon}.h5"
    if os.path.exists(filename) and hasattr(model, "load_weights"):
        model.load_weights(filename)

def train_on_new_data(old_data, new_data, feature_cols, target_col, lookback, horizon, epochs, batch_size, reshuffle_amount, reshuffle_frequency):
    """
    Example placeholder for incremental training logic.
    """
    return None
