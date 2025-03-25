# model.py
"""
Core model-building logic for LSTM/RNN/Tree-based approaches,
as well as utility for evaluating predictions and saving/loading weights.
"""

import os
import sys

# Instead of multiple sys.path.append(), ensure the project root is added only once.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.regularizers import l2  # type: ignore

from src.models.nbeats_model import build_nbeats_model
from src.models.temporal_fusion_transformer import TemporalFusionTransformer

# Fix the import from config
try:
    from config.config_loader import (
        get_value,
    )  # Import get_value to use for configuration

    # Get LOOKBACK from config
    LOOKBACK = get_value("time_series.lookback", 30)
except ImportError:
    # Fallback value if import fails
    LOOKBACK = 30
    import logging

    logging.warning(
        "Could not import from config module. Using fallback LOOKBACK value."
    )

# Import optuna for trial parameter handling
try:
    import optuna
except ImportError:
    pass  # Handle gracefully if optuna isn't available


class FeatureWeightingLayer(tf.keras.layers.Layer):
    """
    A custom layer that multiplies each input feature by a specified weight.
    """

    def __init__(self, weights, **kwargs):
        super(FeatureWeightingLayer, self).__init__(**kwargs)
        self.weights_vector = tf.constant(weights, dtype=tf.float32)

    def call(self, inputs):
        return inputs * self.weights_vector


class AttentionLayer(tf.keras.layers.Layer):
    """Tunable attention layer with multiple attention types."""

    def __init__(
        self,
        attention_type="dot",  # "dot", "multiplicative", "additive"
        attention_size=64,  # Size of attention projection
        num_heads=1,  # For multi-head attention
        use_scale=True,  # Whether to scale attention scores
        dropout_rate=0.0,  # Attention dropout
        **kwargs,
    ):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_type = attention_type
        self.attention_size = attention_size
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention_size = self.attention_size or input_shape[-1]

        if self.attention_type == "dot":
            # Simple dot-product attention
            pass  # No weights needed

        elif self.attention_type == "multiplicative":
            # Multiplicative attention (Luong)
            self.W = self.add_weight(
                name="att_weight",
                shape=(input_shape[-1], input_shape[-1]),
                initializer="glorot_uniform",
            )

        elif self.attention_type == "additive":
            # Additive attention (Bahdanau)
            self.W1 = self.add_weight(
                name="att_weight1",
                shape=(input_shape[-1], self.attention_size),
                initializer="glorot_uniform",
            )
            self.W2 = self.add_weight(
                name="att_weight2",
                shape=(self.attention_size, 1),
                initializer="glorot_uniform",
            )
            self.b = self.add_weight(
                name="att_bias", shape=(self.attention_size,), initializer="zeros"
            )

        if self.num_heads > 1:
            # Multi-head attention components
            self.query_dense = tf.keras.layers.Dense(
                self.attention_size, activation="linear", use_bias=False
            )
            self.key_dense = tf.keras.layers.Dense(
                self.attention_size, activation="linear", use_bias=False
            )
            self.value_dense = tf.keras.layers.Dense(
                self.attention_size, activation="linear", use_bias=False
            )
            self.output_dense = tf.keras.layers.Dense(
                input_shape[-1], activation="linear", use_bias=False
            )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # Implementation varies by attention type
        if self.num_heads > 1:
            return self._multi_head_attention(inputs, training)

        if self.attention_type == "dot":
            return self._dot_attention(inputs, training)
        elif self.attention_type == "multiplicative":
            return self._multiplicative_attention(inputs, training)
        elif self.attention_type == "additive":
            return self._additive_attention(inputs, training)

    def _dot_attention(self, inputs, training=None):
        # Simple dot-product attention
        # inputs shape: [batch_size, time_steps, features]

        # Calculate attention scores
        attention_scores = tf.matmul(
            inputs, inputs, transpose_b=True
        )  # [batch, time, time]

        # Scale if requested
        if self.use_scale:
            attention_scores = attention_scores / tf.math.sqrt(
                tf.cast(self.attention_size, tf.float32)
            )

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(
            attention_scores, axis=-1
        )  # [batch, time, time]

        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention weights to get context vector
        context_vector = tf.matmul(attention_weights, inputs)  # [batch, time, features]

        return context_vector

    def _multiplicative_attention(self, inputs, training=None):
        # Multiplicative attention (Luong)
        # inputs shape: [batch_size, time_steps, features]

        # Project inputs with learned weights
        projection = tf.matmul(inputs, self.W)  # [batch, time, features]

        # Calculate attention scores
        attention_scores = tf.matmul(
            inputs, projection, transpose_b=True
        )  # [batch, time, time]

        # Scale if requested
        if self.use_scale:
            attention_scores = attention_scores / tf.math.sqrt(
                tf.cast(self.attention_size, tf.float32)
            )

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(
            attention_scores, axis=-1
        )  # [batch, time, time]

        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention weights to get context vector
        context_vector = tf.matmul(attention_weights, inputs)  # [batch, time, features]

        return context_vector

    def _additive_attention(self, inputs, training=None):
        # Additive attention (Bahdanau)
        # inputs shape: [batch_size, time_steps, features]

        # Get useful dimensions
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]

        # Reshape inputs for broadcasting
        # [batch, time, 1, features]
        query = tf.expand_dims(inputs, 2)
        # [batch, 1, time, features]
        key = tf.expand_dims(inputs, 1)

        # Create broadcast shapes
        # [batch, time, time, features]
        query_tiled = tf.tile(query, [1, 1, time_steps, 1])
        key_tiled = tf.tile(key, [1, time_steps, 1, 1])

        # Additive attention formula
        # First linear transform
        W1_broadcast = tf.nn.tanh(
            tf.matmul(query_tiled, self.W1) + tf.matmul(key_tiled, self.W1) + self.b
        )

        # Second linear transform to get attention scores [batch, time, time, 1]
        attention_scores = tf.matmul(W1_broadcast, tf.reshape(self.W2, [-1, 1]))
        attention_scores = tf.squeeze(attention_scores, -1)  # [batch, time, time]

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(
            attention_scores, axis=-1
        )  # [batch, time, time]

        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention weights to get context vector
        context_vector = tf.matmul(attention_weights, inputs)  # [batch, time, features]

        return context_vector

    def _multi_head_attention(self, inputs, training=None):
        # Multi-head attention implementation
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]

        # Split attention into multiple heads
        head_size = self.attention_size // self.num_heads

        # Linear projections
        q = self.query_dense(inputs)  # [batch, time, att_size]
        k = self.key_dense(inputs)  # [batch, time, att_size]
        v = self.value_dense(inputs)  # [batch, time, att_size]

        # Reshape for multi-head
        q = tf.reshape(q, [batch_size, time_steps, self.num_heads, head_size])
        k = tf.reshape(k, [batch_size, time_steps, self.num_heads, head_size])
        v = tf.reshape(v, [batch_size, time_steps, self.num_heads, head_size])

        # Transpose to [batch, heads, time, head_size]
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Calculate attention scores
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # [batch, heads, time, time]

        # Scale attention scores
        scale = tf.math.sqrt(tf.cast(head_size, tf.float32))
        matmul_qk = matmul_qk / scale

        # Apply softmax
        attention_weights = tf.nn.softmax(
            matmul_qk, axis=-1
        )  # [batch, heads, time, time]

        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention weights
        output = tf.matmul(attention_weights, v)  # [batch, heads, time, head_size]

        # Transpose back to original shape
        output = tf.transpose(output, [0, 2, 1, 3])  # [batch, time, heads, head_size]

        # Concatenate heads
        output = tf.reshape(output, [batch_size, time_steps, self.attention_size])

        # Final linear projection
        output = self.output_dense(output)

        return output


def build_lstm_model(
    num_features,
    horizon,
    learning_rate=0.001,
    dropout_rate=0.2,
    loss_function="mean_squared_error",
    lookback=LOOKBACK,
    units_per_layer=[64],
    feature_weights=None,
):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))
    if feature_weights is not None:
        model.add(FeatureWeightingLayer(feature_weights))
    for i, units in enumerate(units_per_layer):
        return_seq = i < len(units_per_layer) - 1
        model.add(tf.keras.layers.LSTM(units, return_sequences=return_seq))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(horizon))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function)
    return model


def build_rnn_model(
    num_features,
    horizon,
    learning_rate=0.001,
    dropout_rate=0.2,
    loss_function="mean_squared_error",
    lookback=LOOKBACK,
    units_per_layer=[64],
    feature_weights=None,
):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))
    if feature_weights is not None:
        model.add(FeatureWeightingLayer(feature_weights))
    for i, units in enumerate(units_per_layer):
        return_seq = i < len(units_per_layer) - 1
        model.add(tf.keras.layers.SimpleRNN(units, return_sequences=return_seq))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(horizon))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function)
    return model


def build_random_forest_model(num_features, horizon, n_estimators=100, max_depth=None):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )


def build_xgboost_model(num_features, horizon, learning_rate=0.001, n_estimators=100):
    return xgb.XGBRegressor(
        learning_rate=learning_rate, n_estimators=n_estimators, random_state=42
    )


def build_tft_model(
    num_features,
    horizon,
    learning_rate=0.001,
    dropout_rate=0.2,
    loss_function="mean_squared_error",
    lookback=LOOKBACK,
    hidden_size=64,
    lstm_units=128,
    num_heads=4,
    units_per_layer=[128],
    max_positions=1000,
    feature_weights=None,
):
    # Build the Temporal Fusion Transformer model using all provided hyperparameters
    model = TemporalFusionTransformer(
        num_features=num_features,
        forecast_horizon=horizon,
        hidden_size=hidden_size,
        lstm_units=lstm_units,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        max_positions=max_positions,  # Passing the max_positions parameter
    )

    # Perform a forward pass to build the model structure
    import tensorflow as tf

    sample_input = tf.random.normal((1, lookback, num_features))
    model(sample_input)

    # Compile the model with the chosen optimizer and loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
    )

    return model


def build_ltc_model(
    num_features,
    horizon,
    learning_rate=0.001,
    loss_function="mse",
    lookback=30,
    units=64,
):
    """
    Build a Liquid Time Constant (LTC) model.

    Args:
        num_features: Number of input features
        horizon: Forecast horizon (steps to predict)
        learning_rate: Learning rate for optimizer
        loss_function: Loss function to use
        lookback: Length of input sequence
        units: Number of LTC units

    Returns:
        Compiled LTC model
    """
    # Import here to avoid circular imports
    try:
        from ltc_model import LTCCell
    except ImportError:
        # Fallback implementation if LTC module is not available
        class LTCCell(tf.keras.layers.AbstractRNNCell):
            """Simplified Liquid Time Constant cell implementation."""

            def __init__(self, units, **kwargs):
                super(LTCCell, self).__init__(**kwargs)
                self.units = units
                self.state_size = units
                self.output_size = units

            def build(self, input_shape):
                # Input kernel
                self.kernel = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer="glorot_uniform",
                    name="kernel",
                )
                # Recurrent kernel
                self.recurrent_kernel = self.add_weight(
                    shape=(self.units, self.units),
                    initializer="orthogonal",
                    name="recurrent_kernel",
                )
                # Bias
                self.bias = self.add_weight(
                    shape=(self.units,), initializer="zeros", name="bias"
                )
                self.built = True

            def call(self, inputs, states):
                # Previous state
                prev_state = states[0]

                # Calculate gate values (simplified version)
                z = (
                    tf.matmul(inputs, self.kernel)
                    + tf.matmul(prev_state, self.recurrent_kernel)
                    + self.bias
                )

                # Apply activation
                output = tf.nn.tanh(z)

                # Use output as new state
                new_state = output

                return output, [new_state]

    # Create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))
    model.add(tf.keras.layers.RNN(LTCCell(units), return_sequences=False))
    model.add(tf.keras.layers.Dense(horizon))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
    )

    return model


def build_model_by_type(
    model_type,
    num_features,
    horizon,
    learning_rate,
    dropout_rate,
    loss_function,
    lookback,
    architecture_params=None,
    trial=None,
):
    """
    Build a model of the specified type with parameters either provided directly or suggested by Optuna.

    Args:
        model_type: Type of model to build ('lstm', 'rnn', 'tft', etc.)
        num_features: Number of input features
        horizon: Number of future time steps to predict
        learning_rate: Learning rate for the optimizer
        dropout_rate: Dropout rate for regularization
        loss_function: Loss function to use
        lookback: Number of past time steps to use
        architecture_params: Additional architecture parameters
        trial: Optional Optuna trial object for hyperparameter tuning

    Returns:
        Compiled Keras model
    """
    # Ensure GPU settings are configured properly before building model
    from src.utils.gpu_memory_management import configure_mixed_precision

    configure_mixed_precision()

    tf.keras.backend.clear_session()

    # Default architecture params if none provided
    if architecture_params is None:
        architecture_params = {}

    # If trial is provided, use study_manager to suggest hyperparameters
    if trial is not None:
        # Import the study_manager function for hyperparameter suggestion
        from src.tuning.study_manager import suggest_model_hyperparameters
        
        # Get model-specific parameters
        model_params = suggest_model_hyperparameters(trial, model_type)
        
        # Update architecture_params with suggested parameters
        if "units_per_layer" in model_params:
            architecture_params["units_per_layer"] = model_params["units_per_layer"]
        if "dropout" in model_params:
            dropout_rate = model_params["dropout"]
        if "lr" in model_params:
            learning_rate = model_params["lr"]
        if "loss_function" in model_params:
            loss_function = model_params["loss_function"]
        if "lookback" in model_params:
            lookback = model_params["lookback"]
            
        # Add any other parameters
        architecture_params.update(model_params)

    # Extract architecture parameters
    units_per_layer = architecture_params.get("units_per_layer", [64, 32])
    use_batch_norm = architecture_params.get("use_batch_norm", False)
    l2_reg = architecture_params.get("l2_reg", 0.0)

    # Attention parameters
    use_attention = architecture_params.get("use_attention", False)
    attention_type = architecture_params.get("attention_type", "dot")
    attention_size = architecture_params.get("attention_size", 64)
    attention_heads = architecture_params.get("attention_heads", 1)
    attention_dropout = architecture_params.get("attention_dropout", 0.0)

    # If loss_function is a string, convert to actual function if needed
    if isinstance(loss_function, str):
        loss_function = get_loss_function(loss_function)

    # Initialize model to None
    model = None

    # Model architecture based on type
    if model_type == "lstm":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))

        for i, units in enumerate(units_per_layer):
            return_sequences = i < len(units_per_layer) - 1

            # Add LSTM layer
            lstm_layer = tf.keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=0.0,  # Avoid using recurrent dropout for better performance
                kernel_regularizer=(
                    tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
                ),
            )
            model.add(lstm_layer)

            # Add batch normalization if requested
            if use_batch_norm:
                model.add(tf.keras.layers.BatchNormalization())

            # Add attention if requested and this is the last LSTM layer
            if use_attention and i == len(units_per_layer) - 1:
                attention_layer = AttentionLayer(
                    attention_type=attention_type,
                    attention_size=attention_size,
                    num_heads=attention_heads,
                    dropout_rate=attention_dropout,
                )
                model.add(attention_layer)

    elif model_type == "rnn":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))

        for i, units in enumerate(units_per_layer):
            return_sequences = i < len(units_per_layer) - 1

            # Add SimpleRNN layer
            rnn_layer = tf.keras.layers.SimpleRNN(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                kernel_regularizer=(
                    tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
                ),
            )
            model.add(rnn_layer)

            # Add batch normalization if requested
            if use_batch_norm:
                model.add(tf.keras.layers.BatchNormalization())

    elif model_type == "tft":
        from src.models.temporal_fusion_transformer import build_tft_model

        model = build_tft_model(
            num_features=num_features,
            horizon=horizon,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            loss_function=loss_function,
            lookback=lookback,
            hidden_size=architecture_params.get("hidden_size", 64),
            lstm_units=architecture_params.get("lstm_units", 128),
            num_heads=architecture_params.get("num_heads", 4),
            units_per_layer=units_per_layer,
            max_positions=architecture_params.get(
                "max_positions", 1000
            ),  # Added max_positions parameter
        )

    elif model_type == "nbeats":
        from src.models.nbeats_model import build_nbeats_model

        model = build_nbeats_model(
            lookback=lookback,
            horizon=horizon,
            num_features=num_features,
            learning_rate=learning_rate,
            stack_types=architecture_params.get(
                "stack_types", ["trend", "seasonality"]
            ),
            num_blocks=architecture_params.get("num_blocks", [3, 3]),
            num_layers=architecture_params.get("num_layers", [4, 4]),
            layer_width=architecture_params.get("layer_width", 256),
        )

    elif model_type == "ltc":
        from src.models.ltc_model import build_ltc_model

        model = build_ltc_model(
            num_features=num_features,
            horizon=horizon,
            learning_rate=learning_rate,
            loss_function=loss_function,
            lookback=lookback,
            units=architecture_params.get("units", 64),
            num_layers=architecture_params.get("num_layers", 1),
            use_attention=use_attention,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=architecture_params.get(
                "recurrent_dropout_rate", 0.0
            ),
            timescale_min=architecture_params.get("timescale_min", 0.1),
            timescale_max=architecture_params.get("timescale_max", 10.0),
            use_mixed_precision=None,  # Use system configuration
            adaptive_params=architecture_params.get("adaptive_params", None),
        )
        return model  # LTC model is already compiled

    elif model_type == "cnn":
        from src.models.cnn_model import CNNPricePredictor

        model = CNNPricePredictor(
            lookback=lookback,
            num_features=num_features,
            horizon=horizon,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            num_filters=architecture_params.get("num_filters", 64),
            kernel_size=architecture_params.get("kernel_size", 3),
            num_conv_layers=architecture_params.get("num_conv_layers", 2),
            fc_layers=architecture_params.get("fc_layers", [64]),
        )
        model.build((None, lookback, num_features))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_function,
            metrics=["mae", "mse"],
        )
        return model

    elif model_type == "tabnet":
        from src.models.tabnet_model import TabNetPricePredictor

        model = TabNetPricePredictor(
            n_d=architecture_params.get("n_d", 64),
            n_a=architecture_params.get("n_a", 64),
            n_steps=architecture_params.get("n_steps", 3),
            gamma=architecture_params.get("gamma", 1.5),
            lambda_sparse=architecture_params.get("lambda_sparse", 0.001),
            optimizer_fn=lambda lr: tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            optimizer_params={"lr": learning_rate},
            verbose=0,
        )
        return model  # TabNet has its own compilation

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # If we've reached here and model is still None, something went wrong
    if model is None:
        raise ValueError(f"Failed to build model of type {model_type}")

    # Add output layer for sequential models (LSTM, RNN, TFT)
    if isinstance(model, tf.keras.Sequential):
        model.add(tf.keras.layers.Dense(horizon))

    # Compile model with the resolved loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=["mae", "mse"],
    )

    return model


def evaluate_predictions(true_values: np.ndarray, predicted_values: np.ndarray):
    """
    Compute MSE and MAPE metrics.
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    mse = np.mean((true_values - predicted_values) ** 2)
    epsilon = 1e-7
    mape = (
        np.mean(np.abs((true_values - predicted_values) / (true_values + epsilon)))
        * 100
    )
    return mse, mape


def save_model_weights(
    model, ticker, timeframe, range_cat, horizon=30
):  # <-- CHANGED to include horizon
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
    filename = (
        f"saved_weights/best_weights_{ticker}_{timeframe}_{range_cat}_H{horizon}.h5"
    )
    if os.path.exists(filename) and hasattr(model, "load_weights"):
        model.load_weights(filename)


def train_on_new_data(
    old_data,
    new_data,
    feature_cols,
    target_col,
    lookback,
    horizon,
    epochs,
    batch_size,
    reshuffle_amount,
    reshuffle_frequency,
):
    """
    Example placeholder for incremental training logic.
    """
    return None


# Add snapshot helpers if not already defined
def record_weight_snapshot(model, epoch):
    """Record a snapshot of current weights for neural network visualization."""
    if "weight_history" not in st.session_state:
        st.session_state["weight_history"] = []
    snapshot = {
        "epoch": epoch,
        "weights": [layer.get_weights() for layer in model.layers if layer.weights],
    }
    st.session_state["weight_history"].append(snapshot)


def extract_tree_structure(tree):
    """Extract structure from a sklearn decision tree."""
    tree_struct = {"nodes": {}}

    if hasattr(tree, "tree_"):
        # Handle sklearn tree
        for i in range(tree.tree_.node_count):
            if tree.tree_.children_left[i] == -1:
                tree_struct["nodes"][i] = {
                    "is_leaf": True,
                    "value": float(tree.tree_.value[i][0][0]),
                }
            else:
                tree_struct["nodes"][i] = {
                    "is_leaf": False,
                    "feature_idx": int(tree.tree_.feature[i]),
                    "threshold": float(tree.tree_.threshold[i]),
                    "left_child": int(tree.tree_.children_left[i]),
                    "right_child": int(tree.tree_.children_right[i]),
                }
    elif hasattr(tree, "get_dump"):
        # Handle XGBoost tree
        try:
            tree_dump = tree.get_dump(dump_format="json")
            if isinstance(tree_dump, list) and len(tree_dump) > 0:
                tree_json = json.loads(tree_dump[0])
                tree_struct = parse_xgboost_tree(tree_json)
        except Exception as e:
            print(f"Error parsing XGBoost tree: {e}")
    else:
        print(f"Unknown tree type: {type(tree).__name__}")

    return tree_struct


def record_tree_snapshot(model, iteration):
    """Record a snapshot of the tree structure for visualization."""
    if "tree_models" not in st.session_state:
        st.session_state["tree_models"] = {}
    # Use a timestamp-based model key to group snapshots for a given run
    model_key = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if model_key not in st.session_state["tree_models"]:
        st.session_state["tree_models"][model_key] = {"growth_history": []}
    snapshot = {"iteration": iteration, "trees": extract_tree_structure(model)}
    st.session_state["tree_models"][model_key]["growth_history"].append(snapshot)


# Update your neural network training function to add snapshots
def train_neural_network(model, train_data, epochs=10):
    for epoch in range(epochs):
        # ...existing training code, e.g., model.fit(train_data, epochs=1, verbose=0)...
        model.fit(train_data, epochs=1, verbose=0)
        # RECORD WEIGHT SNAPSHOT after each epoch
        record_weight_snapshot(model, epoch)
    return model


# Update your tree-based model training function to add snapshots
def train_tree_based_model(model, X_train, y_train, iterations=10):
    for iteration in range(iterations):
        # ...existing training code, e.g., model.fit(X_train, y_train)...
        model.fit(X_train, y_train)
        # RECORD TREE SNAPSHOT after each iteration
        record_tree_snapshot(model, iteration)
    return model


def extract_tree_structure(tree):
    """Extract structure from a sklearn decision tree."""
    tree_struct = {"nodes": {}}

    for i in range(tree.tree_.node_count):
        if tree.tree_.children_left[i] == -1:
            tree_struct["nodes"][i] = {
                "is_leaf": True,
                "value": tree.tree_.value[i][0][0],
            }
        else:
            tree_struct["nodes"][i] = {
                "is_leaf": False,
                "feature_idx": tree.tree_.feature[i],
                "threshold": tree.tree_.threshold[i],
                "left_child": tree.tree_.children_left[i],
                "right_child": tree.tree_.children_right[i],
            }
    return tree_struct


def parse_xgboost_tree(tree_json):
    """Parse XGBoost tree from JSON."""
    tree_struct = {"nodes": {}}

    def parse_node(node, node_id):
        if "leaf" in node:
            tree_struct["nodes"][node_id] = {"is_leaf": True, "value": node["leaf"]}
        else:
            tree_struct["nodes"][node_id] = {
                "is_leaf": False,
                "feature_idx": int(node["split"].replace("f", "")),
                "threshold": node["split_condition"],
                "left_child": node_id * 2 + 1,
                "right_child": node_id * 2 + 2,
            }
            parse_node(node["children"][0], node_id * 2 + 1)
            parse_node(node["children"][1], node_id * 2 + 2)

    parse_node(tree_json, 0)
    return tree_struct


def record_tree_snapshot(model, iteration):
    """Record a snapshot of tree structure for visualization."""
    if "tree_models" not in st.session_state:
        st.session_state["tree_models"] = {}

    model_key = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if model_key not in st.session_state["tree_models"]:
        st.session_state["tree_models"][model_key] = {
            "type": type(model).__name__,
            "growth_history": [],
            "feature_names": (
                model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
            ),
        }

    trees = []
    if hasattr(model, "estimators_"):
        # For RandomForest, GradientBoosting
        for t in model.estimators_:
            if hasattr(t, "tree_"):  # sklearn trees
                trees.append(extract_tree_structure(t))
    elif hasattr(model, "get_booster"):
        # For XGBoost
        booster = model.get_booster()
        tree_dumps = booster.get_dump(dump_format="json")
        for d in tree_dumps:
            trees.append(parse_xgboost_tree(json.loads(d)))

    snapshot = {"iteration": iteration, "trees": trees}
    st.session_state["tree_models"][model_key]["growth_history"].append(snapshot)


# Update training functions to include snapshots
def train_neural_network(model, X_train, y_train, epochs=10, batch_size=32, verbose=0):
    """Train neural network model with weight recording."""
    history = None
    for epoch in range(epochs):
        hist = model.fit(
            X_train, y_train, epochs=1, batch_size=batch_size, verbose=verbose
        )
        # Record weights after each epoch
        record_weight_snapshot(model, epoch)

        # Store history
        if history is None:
            history = hist.history
        else:
            for k, v in hist.history.items():
                history[k].extend(v)

    return model, history


def train_tree_based_model(model, X_train, y_train, iterations=1, verbose=0):
    """Train tree-based model with structure recording."""
    # Initial fit
    model.fit(X_train, y_train)
    # Record initial tree structure
    record_tree_snapshot(model, 0)

    # For iterative models like GradientBoosting or XGBoost
    if hasattr(model, "n_estimators") and iterations > 1:
        # This is a simplified approach - actual implementation would depend on the model
        step_size = model.n_estimators // iterations
        for i in range(1, iterations):
            # Record tree structure at this stage
            record_tree_snapshot(model, i)

    return model


# Add more custom loss functions that can be tuned by Optuna
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE) for TensorFlow.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAPE loss value
    """
    epsilon = tf.keras.backend.epsilon()
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))
    return 100.0 * tf.reduce_mean(diff, axis=-1)


def quantile_loss(quantile):
    """
    Create a quantile loss function with the specified quantile value.

    Args:
        quantile: Quantile value (between 0 and 1)

    Returns:
        Quantile loss function
    """

    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            np.maximum(quantile * error, (quantile - 1) * error), axis=-1
        )

    # Set a name for the loss function
    loss.__name__ = f"quantile_loss_{quantile}"
    return loss


def huber_with_delta(delta=1.0):
    """
    Create a Huber loss function with configurable delta parameter.

    Args:
        delta: Threshold parameter for Huber loss

    Returns:
        Huber loss function with specified delta
    """

    def loss(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear, axis=-1)

    loss.__name__ = f"huber_delta_{delta}"
    return loss


def log_cosh_with_scaling(scaling=1.0):
    """
    Create a Log-Cosh loss function with configurable scaling.

    Args:
        scaling: Scaling factor for the predictions

    Returns:
        Log-Cosh loss function with specified scaling
    """

    def loss(y_true, y_pred):
        scaled_error = (y_true - y_pred) * scaling
        return tf.reduce_mean(tf.math.log(tf.math.cosh(scaled_error)), axis=-1)

    loss.__name__ = f"log_cosh_scale_{scaling}"
    return loss


def get_loss_function(loss_name, params=None):
    """
    Get a loss function by name or create a parameterized loss function.

    Args:
        loss_name: Name of the loss function
        params: Additional parameters for configurable loss functions

    Returns:
        Loss function (either string name or callable)
    """
    # If params is None, initialize as empty dict
    params = params or {}

    # Handle custom parametrized loss functions
    if loss_name == "mean_absolute_percentage_error":
        return mean_absolute_percentage_error
    elif loss_name == "quantile_loss":
        quantile = params.get("quantile", 0.5)
        return quantile_loss(quantile)
    elif loss_name == "huber_loss" and "delta" in params:
        return huber_with_delta(params["delta"])
    elif loss_name == "log_cosh" and "scaling" in params:
        return log_cosh_with_scaling(params["scaling"])

    # Return the loss name for TensorFlow's built-in losses
    return loss_name


# Update build_model_by_type to handle more complex loss function configurations
def build_model_by_type(
    model_type,
    num_features,
    horizon,
    learning_rate,
    dropout_rate,
    loss_function,
    lookback,
    architecture_params=None,
    trial=None,
):
    """
    Build a model of the specified type with parameters either provided directly or suggested by Optuna.

    Args:
        model_type: Type of model to build ('lstm', 'rnn', 'tft', etc.)
        num_features: Number of input features
        horizon: Number of future time steps to predict
        learning_rate: Learning rate for the optimizer
        dropout_rate: Dropout rate for regularization
        loss_function: Loss function name or callable
        lookback: Number of past time steps to use
        architecture_params: Additional architecture parameters
        trial: Optional Optuna trial object for hyperparameter tuning

    Returns:
        Compiled Keras model
    """
    tf.keras.backend.clear_session()

    # Default architecture params if none provided
    if architecture_params is None:
        architecture_params = {
            "units_per_layer": [256, 128, 64],  # Increased layer sizes
            "use_batch_norm": True,
            "l2_reg": 0.001,
            "use_attention": True,
            "attention_type": "dot",
            "attention_size": 128,
            "attention_heads": 4,
            "attention_dropout": 0.3,
        }

    # If trial is provided, use it to suggest hyperparameters
    if trial is not None:
        # Override provided parameters with Optuna suggestions
        if model_type == "lstm" or model_type == "rnn":
            learning_rate = trial.suggest_float(
                f"{model_type}_lr", 1e-5, 1e-2, log=True
            )
            dropout_rate = trial.suggest_float(f"{model_type}_dropout", 0.0, 0.5)

            # Architecture parameters
            units_per_layer = [
                trial.suggest_int(f"{model_type}_units_1", 16, 256, log=True),
                trial.suggest_int(f"{model_type}_units_2", 8, 128, log=True),
            ]
            use_batch_norm = trial.suggest_categorical(
                f"{model_type}_batch_norm", [True, False]
            )
            l2_reg = trial.suggest_float(f"{model_type}_l2_reg", 0.0, 0.01)

            # Attention parameters
            use_attention = trial.suggest_categorical(
                f"{model_type}_use_attention", [True, False]
            )
            attention_type = trial.suggest_categorical(
                f"{model_type}_attention_type", ["dot", "multiplicative", "additive"]
            )
            attention_size = trial.suggest_int(f"{model_type}_attention_size", 32, 128)
            attention_heads = trial.suggest_int(f"{model_type}_attention_heads", 1, 4)
            attention_dropout = trial.suggest_float(
                f"{model_type}_attention_dropout", 0.0, 0.5
            )

            # Loss function suggestion
            from config.config_loader import get_value

            available_loss_fns = get_value(
                "loss_functions.available",
                ["mean_squared_error", "mean_absolute_error", "huber_loss"],
            )

            loss_function = trial.suggest_categorical(
                f"{model_type}_loss", available_loss_fns
            )

            # Handle custom loss function parameters
            loss_params = {}
            if loss_function == "quantile_loss":
                quantile_values = get_value(
                    "loss_functions.optimization.quantile_values", [0.1, 0.5, 0.9]
                )
                quantile = trial.suggest_categorical(
                    f"{model_type}_quantile", quantile_values
                )
                loss_params["quantile"] = quantile
            elif loss_function == "huber_loss":
                delta = trial.suggest_float(
                    f"{model_type}_huber_delta", 0.1, 10.0, log=True
                )
                loss_params["delta"] = delta
            elif loss_function == "log_cosh":
                scaling = trial.suggest_float(
                    f"{model_type}_log_cosh_scaling", 0.1, 10.0, log=True
                )
                loss_params["scaling"] = scaling

            # Create final loss function
            loss_function = get_loss_function(loss_function, loss_params)

            # Update architecture parameters
            architecture_params.update(
                {
                    "units_per_layer": units_per_layer,
                    "use_batch_norm": use_batch_norm,
                    "l2_reg": l2_reg,
                    "use_attention": use_attention,
                    "attention_type": attention_type,
                    "attention_size": attention_size,
                    "attention_heads": attention_heads,
                    "attention_dropout": attention_dropout,
                }
            )

    # Extract architecture parameters
    units_per_layer = architecture_params.get("units_per_layer", [64, 32])
    use_batch_norm = architecture_params.get("use_batch_norm", False)
    l2_reg = architecture_params.get("l2_reg", 0.0)

    # If loss_function is a string, convert to actual function if needed
    if isinstance(loss_function, str):
        loss_function = get_loss_function(
            loss_function, architecture_params.get("loss_params", {})
        )

    # Initialize model to None

    # Model architecture based on type
    if model_type == "lstm":
        # Create sequential model for LSTM
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))

        # Add LSTM layers
        for i, units in enumerate(units_per_layer):
            return_sequences = i < len(units_per_layer) - 1 or use_attention
            model.add(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
                    recurrent_dropout=dropout_rate,
                )
            )
            model.add(tf.keras.layers.Dropout(dropout_rate))
            if use_batch_norm:
                model.add(tf.keras.layers.BatchNormalization())

        # Add attention layer if requested
        if use_attention:
            model.add(
                AttentionLayer(
                    attention_type=attention_type,
                    attention_size=attention_size,
                    num_heads=attention_heads,
                    dropout_rate=attention_dropout,
                )
            )

    elif model_type == "rnn":
        # Create sequential model for RNN
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))

        # Add SimpleRNN layers
        for i, units in enumerate(units_per_layer):
            return_sequences = i < len(units_per_layer) - 1 or use_attention
            model.add(
                tf.keras.layers.SimpleRNN(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
                )
            )
            model.add(tf.keras.layers.Dropout(dropout_rate))
            if use_batch_norm:
                model.add(tf.keras.layers.BatchNormalization())

        # Add attention layer if requested
        if use_attention:
            model.add(
                AttentionLayer(
                    attention_type=attention_type,
                    attention_size=attention_size,
                    num_heads=attention_heads,
                    dropout_rate=attention_dropout,
                )
            )

    elif model_type == "tft":
        # Create sequential model for TFT
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(lookback, num_features)))

        # Extract TFT-specific parameters
        hidden_size = architecture_params.get("hidden_size", 64)
        lstm_units = architecture_params.get("lstm_units", 128)
        num_heads = architecture_params.get("num_heads", 4)

        # Implement TFT (simplified version using attention)
        # Input gate
        model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))

        # Self-attention module (always used in TFT)
        model.add(
            AttentionLayer(
                attention_type="multiplicative",
                attention_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )
        )

        # Gated residual network
        model.add(tf.keras.layers.Dense(hidden_size, activation="relu"))
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))

    elif model_type == "nbeats":
        # N-BEATS returns a complete model (already has output layer and is compiled)
        # Extract N‑BEATS–specific parameters; provide defaults if not specified
        num_blocks = architecture_params.get("num_blocks", 3)
        num_layers = architecture_params.get("num_layers", 4)
        layer_width = architecture_params.get("layer_width", 256)

        # Build and return N-BEATS model directly
        return build_nbeats_model(
            num_features,
            horizon,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_width=layer_width,
            learning_rate=learning_rate,
            loss_function=loss_function,
        )

    elif model_type == "ltc":
        # LTC returns a complete model (already has output layer and is compiled)
        # LTC-specific parameters: default units, but can be tuned
        units = architecture_params.get("units", 64)

        # Build and return LTC model directly
        return build_ltc_model(
            num_features, horizon, learning_rate, loss_function, lookback, units
        )

    else:
        # Handle unknown model type
        raise ValueError(f"Unknown model type: {model_type}")

    # If we've reached here and model is still None, something went wrong
    if model is None:
        return None

    # Add output layer for sequential models (LSTM, RNN, TFT)
    model.add(tf.keras.layers.Dense(horizon))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=["mae", "mse"],
    )

    return model


# Add a custom model metrics function that includes MAPE
def model_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for model evaluation.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Ensure arrays are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE
    epsilon = np.finfo(np.float64).eps  # Small value to avoid division by zero
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    )

    # Calculate Huber loss (with delta=1.0)
    delta = 1.0
    huber_errors = np.where(
        np.abs(y_true - y_pred) < delta,
        0.5 * np.square(y_true - y_pred),
        delta * (np.abs(y_true - y_pred) - 0.5 * delta),
    )
    huber = np.mean(huber_errors)

    # Calculate quantile losses for different quantiles
    quantiles = [0.1, 0.5, 0.9]
    quantile_losses = {}
    for q in quantiles:
        errors = y_true - y_pred
        quantile_loss_value = np.mean(np.maximum(q * errors, (q - 1) * errors))
        quantile_losses[f"q{int(q*100)}"] = quantile_loss_value

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "huber": huber,
        "quantile_losses": quantile_losses,
    }


# Extend model evaluation to include all new metrics
def evaluate_model_with_all_metrics(model, X_test, y_test):
    """
    Evaluate a model using multiple metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True test values

    Returns:
        Dictionary of evaluation metrics
    """
    # Predict values
    y_pred = model.predict(X_test)

    # Calculate comprehensive metrics
    return model_metrics(y_test, y_pred)


# Add a multi-objective loss function class for weighted combination of losses
class WeightedMultiLoss(tf.keras.losses.Loss):
    """
    A weighted combination of multiple loss functions.

    This allows optimizing for multiple objectives simultaneously with configurable weights.
    """

    def __init__(self, loss_functions, weights, name="weighted_multi_loss"):
        """
        Initialize a weighted multi-loss function.

        Args:
            loss_functions: List of loss functions
            weights: List of weights for each loss function
            name: Name of the loss function
        """
        super().__init__(name=name)
        self.loss_functions = loss_functions
        self.weights = weights

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in weights]

    def call(self, y_true, y_pred):
        """
        Calculate the weighted combination of losses.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Weighted loss value
        """
        loss_value = 0.0
        for i, loss_fn in enumerate(self.loss_functions):
            # Handle both string and callable loss functions
            if isinstance(loss_fn, str):
                loss_obj = tf.keras.losses.get(loss_fn)
                curr_loss = loss_obj(y_true, y_pred)
            else:
                curr_loss = loss_fn(y_true, y_pred)

            loss_value += self.weights[i] * curr_loss

        return loss_value


# Function to create a multi-objective loss from configuration
def create_multi_objective_loss(config=None, trial=None):
    """
    Create a multi-objective loss function based on configuration or Optuna trial.

    Args:
        config: Configuration dictionary (optional)
        trial: Optuna trial object (optional)

    Returns:
        Multi-objective loss function
    """

    # Default loss functions and weights
    loss_fns = ["mean_squared_error", "mean_absolute_error"]
    weights = [0.7, 0.3]

    # If using Optuna trial
    if trial:
        # Suggest which losses to use
        use_mse = trial.suggest_categorical("use_mse", [True, False])
        use_mae = trial.suggest_categorical("use_mae", [True, False])
        use_mape = trial.suggest_categorical("use_mape", [True, False])
        use_huber = trial.suggest_categorical("use_huber", [True, False])
        use_quantile = trial.suggest_categorical("use_quantile", [True, False])

        # Build list of losses
        selected_losses = []
        if use_mse:
            selected_losses.append("mean_squared_error")
        if use_mae:
            selected_losses.append("mean_absolute_error")
        if use_mape:
            selected_losses.append(mean_absolute_percentage_error)
        if use_huber:
            delta = trial.suggest_float("huber_delta", 0.1, 10.0, log=True)
            selected_losses.append(huber_with_delta(delta))
        if use_quantile:
            quantile = trial.suggest_float("quantile_value", 0.1, 0.9)
            selected_losses.append(quantile_loss(quantile))

        # Ensure at least one loss is selected
        if not selected_losses:
            selected_losses = ["mean_squared_error"]

        # Suggest weights
        raw_weights = []
        for i, loss in enumerate(selected_losses):
            loss_name = loss if isinstance(loss, str) else loss.__name__
            w = trial.suggest_float(f"weight_{loss_name}", 0.1, 1.0)
            raw_weights.append(w)

        # Normalize weights
        total_weight = sum(raw_weights)
        if total_weight > 0:
            weights = [w / total_weight for w in raw_weights]
        else:
            weights = [1.0 / len(selected_losses)] * len(selected_losses)

        # Return multi-objective loss
        return WeightedMultiLoss(selected_losses, weights)

    # If using configuration
    elif config:
        # Get loss functions from config
        loss_fns = config.get("loss_functions", loss_fns)
        weights = config.get("weights", weights)

        # Process any custom loss functions
        processed_losses = []
        for loss in loss_fns:
            if isinstance(loss, str):
                if loss == "mean_absolute_percentage_error":
                    processed_losses.append(mean_absolute_percentage_error)
                elif loss == "quantile_loss":
                    quantile = config.get("quantile", 0.5)
                    processed_losses.append(quantile_loss(quantile))
                elif loss == "huber_loss":
                    delta = config.get("huber_delta", 1.0)
                    processed_losses.append(huber_with_delta(delta))
                else:
                    processed_losses.append(loss)
            else:
                processed_losses.append(loss)

        return WeightedMultiLoss(processed_losses, weights)

    # Default case - just MSE and MAE with default weights
    return WeightedMultiLoss(loss_fns, weights)


# Helper function to expose all available custom loss functions to Optuna
def register_loss_functions_with_optuna(trial, model_type, available_losses=None):
    """
    Register loss functions with Optuna for tuning.

    Args:
        trial: Optuna trial object
        model_type: Model type string (e.g., 'lstm')
        available_losses: Optional list of available loss functions

    Returns:
        Tuple of (loss_function, loss_params)
    """
    from config.config_loader import get_value

    # Get available losses from config or use defaults
    if available_losses is None:
        available_losses = get_value(
            "loss_functions.available",
            [
                "mean_squared_error",
                "mean_absolute_error",
                "huber_loss",
                "mean_absolute_percentage_error",
                "quantile_loss",
            ],
        )

    # Suggest which loss function to use
    loss_function = trial.suggest_categorical(f"{model_type}_loss", available_losses)

    # Handle custom loss function parameters
    loss_params = {}

    if loss_function == "quantile_loss":
        # For quantile loss, suggest the quantile value
        quantile_values = get_value(
            "loss_functions.optimization.quantile_values", [0.1, 0.5, 0.9]
        )
        if len(quantile_values) > 1:
            quantile = trial.suggest_categorical(
                f"{model_type}_quantile", quantile_values
            )
        else:
            quantile = quantile_values[0] if quantile_values else 0.5
        loss_params["quantile"] = quantile

    elif loss_function == "huber_loss":
        # For Huber loss, suggest the delta value
        delta = trial.suggest_float(f"{model_type}_huber_delta", 0.1, 10.0, log=True)
        loss_params["delta"] = delta

    elif loss_function == "log_cosh":
        # For Log-Cosh loss, suggest the scaling factor
        scaling = trial.suggest_float(
            f"{model_type}_log_cosh_scaling", 0.1, 10.0, log=True
        )
        loss_params["scaling"] = scaling

    # Return both the loss function name and any parameters
    return loss_function, loss_params


# Function to build optimization target metrics based on loss functions
def build_optimization_metrics(trial=None):
    """
    Build optimization metrics for model evaluation/optimization.

    Args:
        trial: Optional Optuna trial object for suggesting metric weights

    Returns:
        Dictionary of metrics and their weights
    """
    from config.config_loader import get_value

    # Get default metric weights from config
    default_weights = get_value(
        "loss_functions.optimization.weighted_metrics",
        {"rmse": 1.0, "mape": 1.0, "huber": 0.8, "quantile": 1.0},
    )

    if trial:
        # Allow Optuna to suggest metric weights
        weights = {
            "rmse": trial.suggest_float("weight_rmse", 0.1, 5.0),
            "mape": trial.suggest_float("weight_mape", 0.1, 5.0),
            "huber": trial.suggest_float("weight_huber", 0.1, 5.0),
            "quantile": trial.suggest_float("weight_quantile", 0.1, 5.0),
        }
    else:
        # Use default weights
        weights = default_weights

    return weights
