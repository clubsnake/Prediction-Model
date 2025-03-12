# ltc_model.py
"""
Liquid Time Constant (LTC) model implementation.
This model extends traditional RNNs by introducing learnable time constants
that allow different neurons to operate at different timescales.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Add optuna import for Optuna trial handling
try:
    import optuna
except ImportError:
    import logging

    logging.warning(
        "Optuna not available - hyperparameter optimization will be disabled"
    )

# Make sure os is imported for file paths
import os

# Add datetime import for timestamp generation
from datetime import datetime

# Add training optimizer import
from src.utils.training_optimizer import get_training_optimizer

# Initialize the training optimizer
training_optimizer = None
try:
    training_optimizer = get_training_optimizer()
except Exception as e:
    logger.warning(f"Could not initialize training optimizer: {e}")


class LTCCell(tf.keras.layers.AbstractRNNCell):
    """
    Liquid Time Constant Cell - an advanced RNN cell with learnable timescales.

    The LTC model introduces individual time constants for each hidden unit,
    allowing the network to learn optimal timescales for different features
    and improving the model's ability to capture multi-scale temporal dependencies.
    """

    def __init__(
        self,
        units: int,
        timescale_min: float = 0.1,
        timescale_max: float = 10.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        timescale_initializer: str = "ones",
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        timescale_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the LTC cell with advanced configuration options.

        Args:
            units: Number of hidden units in the cell
            timescale_min: Minimum value for time constants
            timescale_max: Maximum value for time constants
            activation: Activation function for hidden state
            recurrent_activation: Activation function for gates
            use_bias: Whether to use bias terms
            kernel_initializer: Initializer for input weights
            recurrent_initializer: Initializer for recurrent weights
            bias_initializer: Initializer for bias terms
            timescale_initializer: Initializer for time constants
            kernel_regularizer: Regularizer for input weights
            recurrent_regularizer: Regularizer for recurrent weights
            bias_regularizer: Regularizer for bias terms
            timescale_regularizer: Regularizer for time constants
            dropout: Input dropout rate
            recurrent_dropout: Recurrent dropout rate
        """
        super(LTCCell, self).__init__(**kwargs)
        self.units = units
        self.timescale_min = timescale_min
        self.timescale_max = timescale_max
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias

        # Initializers
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.timescale_initializer = tf.keras.initializers.get(timescale_initializer)

        # Regularizers
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.timescale_regularizer = tf.keras.regularizers.get(timescale_regularizer)

        # Dropout
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))

        # Dropout masks (will be created in call if needed)
        self.dropout_mask = None
        self.recurrent_dropout_mask = None

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        """Build the cell weights and biases."""
        input_dim = input_shape[-1]

        # Input weights for update gate
        self.kernel_z = self.add_weight(
            shape=(input_dim, self.units),
            name="kernel_z",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
        )

        # Recurrent weights for update gate
        self.recurrent_kernel_z = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_kernel_z",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
        )

        # Input weights for candidate state
        self.kernel_h = self.add_weight(
            shape=(input_dim, self.units),
            name="kernel_h",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
        )

        # Recurrent weights for candidate state
        self.recurrent_kernel_h = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_kernel_h",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
        )

        # Liquid time constants - the key innovation of the LTC model
        # These are learnable parameters that control the timescale of each unit
        self.tau = self.add_weight(
            shape=(self.units,),
            name="tau",
            initializer=self.timescale_initializer,
            regularizer=self.timescale_regularizer,
            constraint=lambda x: tf.clip_by_value(
                x, self.timescale_min, self.timescale_max
            ),
        )

        # Biases
        if self.use_bias:
            self.bias_z = self.add_weight(
                shape=(self.units,),
                name="bias_z",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
            )
            self.bias_h = self.add_weight(
                shape=(self.units,),
                name="bias_h",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
            )
        else:
            self.bias_z = None
            self.bias_h = None

        self.built = True

    def call(self, inputs, states, training=None):
        """
        Forward pass for the LTC cell.

        Args:
            inputs: Input tensor.
            states: Previous state tensor.
            training: Whether in training mode (for dropout).

        Returns:
            output: The output tensor.
            new_states: The updated cell state.
        """
        # Get previous state
        h_prev = states[0]

        # Apply dropout for inputs if needed
        if (training is not None) and (self.dropout > 0):
            if self.dropout_mask is None:
                self.dropout_mask = self._generate_dropout_mask(inputs, training)
            inputs = inputs * self.dropout_mask

        # Apply recurrent dropout if needed
        if (training is not None) and (self.recurrent_dropout > 0):
            if self.recurrent_dropout_mask is None:
                self.recurrent_dropout_mask = self._generate_dropout_mask(
                    h_prev, training
                )
            h_prev_dropped = h_prev * self.recurrent_dropout_mask
        else:
            h_prev_dropped = h_prev

        # Calculate update gate
        z = tf.matmul(inputs, self.kernel_z) + tf.matmul(
            h_prev_dropped, self.recurrent_kernel_z
        )
        if self.use_bias:
            z = tf.nn.bias_add(z, self.bias_z)
        z = self.recurrent_activation(z)

        # Calculate candidate state
        h_candidate = tf.matmul(inputs, self.kernel_h) + tf.matmul(
            h_prev_dropped, self.recurrent_kernel_h
        )
        if self.use_bias:
            h_candidate = tf.nn.bias_add(h_candidate, self.bias_h)
        h_candidate = self.activation(h_candidate)

        # Apply liquid time constant update
        # This is the core LTC mechanism - each unit has its own timescale
        dt = 1.0  # Time step size (fixed for simplicity)
        decay = tf.exp(-dt / self.tau)
        h_new = decay * h_prev + (1 - decay) * h_candidate

        return h_new, [h_new]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Generate initial state for the cell."""
        return [tf.zeros([batch_size, self.units], dtype=dtype or tf.float32)]

    def _generate_dropout_mask(self, inputs, training):
        """Generate a dropout mask for the inputs."""
        if training:
            return tf.keras.backend.dropout(tf.ones_like(inputs), self.dropout)
        else:
            return tf.ones_like(inputs)

    def get_config(self):
        """Return the cell configuration for serialization."""
        config = {
            "units": self.units,
            "timescale_min": self.timescale_min,
            "timescale_max": self.timescale_max,
            "activation": tf.keras.activations.serialize(self.activation),
            "recurrent_activation": tf.keras.activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": tf.keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "timescale_initializer": tf.keras.initializers.serialize(
                self.timescale_initializer
            ),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": tf.keras.regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "timescale_regularizer": tf.keras.regularizers.serialize(
                self.timescale_regularizer
            ),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        }
        base_config = super(LTCCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LTCAttentionCell(LTCCell):
    """
    LTC cell with attention mechanism for improved sequence modeling.
    This extended version adds attention to focus on relevant parts of input sequences.
    """

    def __init__(
        self,
        units: int,
        attention_units: int = None,
        attention_activation: str = "tanh",
        **kwargs,
    ):
        """
        Initialize an LTC cell with attention mechanism.

        Args:
            units: Number of hidden units
            attention_units: Number of attention units (defaults to units)
            attention_activation: Activation function for attention
            **kwargs: Additional arguments for the base LTC cell
        """
        super(LTCAttentionCell, self).__init__(units, **kwargs)
        self.attention_units = attention_units or units
        self.attention_activation = tf.keras.activations.get(attention_activation)

    def build(self, input_shape):
        """Build the cell with additional attention weights."""
        super(LTCAttentionCell, self).build(input_shape)

        input_dim = input_shape[-1]

        # Attention weights
        self.attention_w = self.add_weight(
            shape=(input_dim, self.attention_units),
            name="attention_w",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
        )

        self.attention_u = self.add_weight(
            shape=(self.units, self.attention_units),
            name="attention_u",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
        )

        self.attention_v = self.add_weight(
            shape=(self.attention_units, 1),
            name="attention_v",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
        )

        if self.use_bias:
            self.attention_bias = self.add_weight(
                shape=(self.attention_units,),
                name="attention_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
            )
        else:
            self.attention_bias = None

    def call(self, inputs, states, training=None):
        """Forward pass with attention mechanism."""
        h_prev = states[0]

        # Calculate attention score
        attention = tf.matmul(inputs, self.attention_w) + tf.matmul(
            h_prev, self.attention_u
        )
        if self.use_bias:
            attention = tf.nn.bias_add(attention, self.attention_bias)
        attention = self.attention_activation(attention)
        score = tf.matmul(attention, self.attention_v)

        # Apply attention to input
        attention_weights = tf.nn.sigmoid(score)
        attended_input = inputs * attention_weights

        # Continue with normal LTC update using attended input
        return super(LTCAttentionCell, self).call(attended_input, states, training)

    def get_config(self):
        """Return configuration including attention parameters."""
        config = {
            "attention_units": self.attention_units,
            "attention_activation": tf.keras.activations.serialize(
                self.attention_activation
            ),
        }
        base_config = super(LTCAttentionCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_ltc_model(
    num_features: int,
    horizon: int,
    learning_rate: float = 0.001,
    loss_function: str = "mse",
    lookback: int = 30,
    units: int = 64,
    num_layers: int = 1,
    use_attention: bool = False,
    dropout_rate: float = 0.1,
    recurrent_dropout_rate: float = 0.0,
    timescale_min: float = 0.1,
    timescale_max: float = 10.0,
    use_mixed_precision: bool = None,
    adaptive_params: dict = None,
) -> tf.keras.Model:
    """
    Build and compile a Liquid Time Constant (LTC) model with GPU optimizations.

    Includes support for mixed precision and adaptive parameters.

    Args:
        num_features: Number of input features
        horizon: Forecast horizon (number of future steps to predict)
        learning_rate: Learning rate for the optimizer
        loss_function: Loss function to use
        lookback: Number of historical time steps to consider
        units: Number of hidden units in each LTC layer
        num_layers: Number of LTC layers to stack
        use_attention: Whether to use attention mechanism
        dropout_rate: Input dropout rate
        recurrent_dropout_rate: Recurrent dropout rate
        timescale_min: Minimum timescale value
        timescale_max: Maximum timescale value
        use_mixed_precision: Whether to use mixed precision (float16)
        adaptive_params: Dictionary of adaptive parameters

    Returns:
        Compiled Keras model ready for training
    """
    # Configure GPU memory and precision settings
    try:
        from src.utils.gpu_memory_management import configure_mixed_precision

        precision_policy = configure_mixed_precision(use_mixed_precision)
        logger.info(f"Using precision policy: {precision_policy}")
    except ImportError:
        logger.warning("GPU memory management module not found")

    # Apply adaptive parameters if provided
    if adaptive_params is None:
        # Try to load from config
        try:
            from config.config_loader import get_adaptive_params

            adaptive_params = get_adaptive_params()
        except ImportError:
            logger.warning("Could not load adaptive parameters from config")
            adaptive_params = {}

    # Apply adaptive window size if enabled
    if adaptive_params.get("use_adaptive_window", False):
        logger.info(f"Using adaptive window size (original: {lookback})")
        # Implementation would depend on your adaptive_window_size function
        # Here we just use the adaptive parameter directly if provided
        lookback = adaptive_params.get("walk_forward_window", lookback)
        logger.info(f"Adaptive window size: {lookback}")

    # Input layer with shape [batch, lookback, features]
    inputs = tf.keras.layers.Input(shape=(lookback, num_features))

    # Create the LTC cell with or without attention
    if use_attention:
        cell_class = LTCAttentionCell
    else:
        cell_class = LTCCell

    # Apply dropout to inputs if specified
    x = tf.keras.layers.Dropout(dropout_rate)(inputs) if dropout_rate > 0 else inputs

    # Stack multiple LTC layers if requested
    for i in range(num_layers):
        return_sequences = i < num_layers - 1  # Return sequences for all but last layer

        ltc_cell = cell_class(
            units=units,
            dropout=(
                dropout_rate if i == 0 else 0
            ),  # Only apply dropout to first layer inputs
            recurrent_dropout=recurrent_dropout_rate,
            timescale_min=timescale_min,
            timescale_max=timescale_max,
        )

        layer = tf.keras.layers.RNN(
            ltc_cell, return_sequences=return_sequences, name=f"ltc_layer_{i+1}"
        )

        x = layer(x)

        # Add intermediate dropout between layers if needed
        if i < num_layers - 1 and dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(horizon)(x)

    # Create and compile model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Adjust learning rate if adaptive learning rate is enabled
    if adaptive_params.get("adaptive_learning_rate", False):
        # Use the adaptive learning rate from parameters or base learning rate
        learning_rate = adaptive_params.get("learning_rate", learning_rate)
        logger.info(f"Using adaptive learning rate: {learning_rate}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
    )

    return model


def create_ltc_ensemble(
    num_features: int,
    horizon: int,
    learning_rate: float = 0.001,
    loss_function: str = "mse",
    lookback: int = 30,
    configs: List[Dict[str, Any]] = None,
) -> List[tf.keras.Model]:
    """
    Create an ensemble of LTC models with different configurations.

    Args:
        num_features: Number of input features
        horizon: Forecast horizon
        learning_rate: Base learning rate
        loss_function: Loss function
        lookback: Number of past time steps
        configs: List of model configuration dictionaries

    Returns:
        List of compiled LTC models
    """
    # Default configurations if none provided
    if configs is None:
        configs = [
            # Base LTC
            {"units": 64, "num_layers": 1, "use_attention": False},
            # Deep LTC
            {"units": 32, "num_layers": 2, "use_attention": False},
            # LTC with attention
            {"units": 64, "num_layers": 1, "use_attention": True},
        ]

    models = []
    for i, config in enumerate(configs):
        # Create model with specified config
        model = build_ltc_model(
            num_features=num_features,
            horizon=horizon,
            learning_rate=learning_rate,
            loss_function=loss_function,
            lookback=lookback,
            **config,
        )
        model.name = f"ltc_ensemble_{i+1}"
        models.append(model)

    return models


def visualize_timescales(model: tf.keras.Model) -> Dict[str, List[float]]:
    """
    Extract and visualize the learned timescales from an LTC model.

    Args:
        model: Trained LTC model

    Returns:
        Dictionary with extracted timescales
    """
    timescales = {}

    # Find LTC layers
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.RNN):
            cell = layer.cell
            if isinstance(cell, LTCCell):
                # Extract tau values (time constants)
                tau_values = cell.tau.numpy()
                layer_name = layer.name
                timescales[layer_name] = tau_values.tolist()

    return timescales


# Helper function to get memory usage
def get_model_memory_usage(model: tf.keras.Model) -> float:
    """
    Estimate memory usage of an LTC model in MB.

    Args:
        model: LTC model

    Returns:
        Estimated memory usage in MB
    """
    # Count parameters
    trainable_count = np.sum(
        [tf.keras.backend.count_params(w) for w in model.trainable_weights]
    )
    non_trainable_count = np.sum(
        [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
    )

    # Estimate using 4 bytes per parameter (float32)
    param_bytes = 4 * (trainable_count + non_trainable_count)

    # Add buffer for computation (rough estimate)
    buffer_mb = 50

    # Convert to MB
    total_mb = (param_bytes / (1024 * 1024)) + buffer_mb

    return total_mb


def calculate_forecast_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for forecast evaluation.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Ensure arrays are the right shape
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # For multi-step forecasts, average metrics across horizons
        mse_values = []
        mae_values = []
        mape_values = []
        r2_values = []

        for i in range(y_true.shape[1]):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]

            mse_values.append(mean_squared_error(y_t, y_p))
            mae_values.append(mean_absolute_error(y_t, y_p))

            # Handle division by zero in MAPE
            with np.errstate(divide="ignore", invalid="ignore"):
                mape = np.mean(np.abs((y_t - y_p) / np.abs(y_t + 1e-10))) * 100
                mape_values.append(np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0))

            r2_values.append(r2_score(y_t, y_p))

        # Average metrics across horizons
        mse = np.mean(mse_values)
        rmse = np.sqrt(mse)
        mae = np.mean(mae_values)
        mape = np.mean(mape_values)
        r2 = np.mean(r2_values)
    else:
        # For single-step forecasts
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # Handle division by zero in MAPE
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true + 1e-10))) * 100
            mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)

        r2 = r2_score(y_true, y_pred)

    # Calculate direction accuracy
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        direction_match = np.equal(true_direction, pred_direction)
        direction_accuracy = np.mean(direction_match) * 100
    else:
        direction_accuracy = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
    }


def get_ltc_hyperparameter_ranges():
    """
    Define hyperparameter ranges for Optuna optimization of LTC models.

    These ranges match the dashboard UI and are optimized for financial time series.
    """
    return {
        # Core architecture
        "units": {"type": "int", "low": 32, "high": 512, "log": True},
        "hidden_size": {"type": "int", "low": 32, "high": 256, "log": True},
        "num_layers": {"type": "int", "low": 1, "high": 4, "step": 1},
        "activation": {
            "type": "categorical",
            "choices": ["tanh", "relu", "sigmoid", "gelu"],
        },
        # Timescales - key parameters for LTC
        "timescale_min": {"type": "float", "low": 0.0001, "high": 0.1, "log": True},
        "timescale_max": {"type": "float", "low": 1.0, "high": 100.0, "log": True},
        # Training parameters
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
        "recurrent_dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
        "l2_reg": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        # Architecture options
        "use_bidirectional": {"type": "categorical", "choices": [True, False]},
        "use_dense_after_ltc": {"type": "categorical", "choices": [True, False]},
        "use_batch_normalization": {"type": "categorical", "choices": [True, False]},
        "use_layer_normalization": {"type": "categorical", "choices": [True, False]},
        "use_residual_connections": {"type": "categorical", "choices": [True, False]},
        # Advanced features
        "use_attention": {"type": "categorical", "choices": [True, False]},
        "attention_size": {"type": "int", "low": 32, "high": 128, "log": True},
        "include_market_features": {"type": "categorical", "choices": [True, False]},
        # Optimizer settings
        "optimizer_type": {
            "type": "categorical",
            "choices": ["adam", "adamw", "rmsprop"],
        },
        "use_lr_schedule": {"type": "categorical", "choices": [True, False]},
        # Loss function
        "loss_function": {
            "type": "categorical",
            "choices": ["mse", "mae", "huber_loss"],
        },
        # NEW: Incremental learning parameters
        "retraining_threshold": {
            "type": "float",
            "low": 0.01,
            "high": 0.5,
            "log": True,
        },
        "walk_forward_window": {"type": "int", "low": 3, "high": 60, "step": 1},
        "update_frequency": {"type": "int", "low": 1, "high": 30, "step": 1},
        "reuse_weight_ratio": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
        "early_stopping_patience": {"type": "int", "low": 5, "high": 50, "step": 5},
    }


# Add TimescaleAnalysisCallback class to monitor timescale learning during training
class TimescaleAnalysisCallback(tf.keras.callbacks.Callback):
    """
    Callback to analyze learned timescales during training.

    This helps understand what temporal patterns the model is learning.
    """

    def __init__(self, logging_freq=10):
        super(TimescaleAnalysisCallback, self).__init__()
        self.logging_freq = logging_freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.logging_freq != 0:
            return

        # Find LTC layers
        ltc_layers = [
            layer
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.RNN)
            and isinstance(layer.cell, LTCCell)
        ]

        for i, layer in enumerate(ltc_layers):
            tau = layer.cell.tau.numpy()
            logger.info(f"Epoch {epoch}, LTC layer {i+1} timescales:")
            logger.info(f"  Min timescale: {np.min(tau):.4f}")
            logger.info(f"  Max timescale: {np.max(tau):.4f}")
            logger.info(f"  Mean timescale: {np.mean(tau)::.4f}")
            logger.info(f"  Median timescale: {np.median(tau)::.4f}")


def get_callbacks(
    patience=20,
    reduce_lr_factor=0.2,
    monitor="val_loss",
    mode="min",
    timescale_logging=True,
):
    """Get training callbacks for LTC model."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, mode=mode
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=patience // 2,
            min_lr=1e-6,
            mode=mode,
        ),
    ]

    # Add timescale analysis if requested
    if timescale_logging:
        callbacks.append(TimescaleAnalysisCallback(logging_freq=10))

    return callbacks


def visualize_ltc_dynamics(model, sample_input, save_path=None):
    """
    Visualize the dynamics of an LTC model.

    Args:
        model: Trained LTC model
        sample_input: Sample input for visualization
        save_path: Path to save visualization (if None, display instead)

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Find LTC layers
    ltc_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.RNN) and hasattr(layer.cell, "tau")
    ]

    if not ltc_layers:
        logger.warning("No LTC layers found in model")
        return

    # Process sample input
    sample_prediction = model.predict(sample_input[np.newaxis])[0]

    # Create a model that outputs the LTC states
    ltc_layer = ltc_layers[0]
    state_model = tf.keras.Model(inputs=model.inputs, outputs=ltc_layer.output)

    # Get states for sample input
    states = state_model.predict(sample_input[np.newaxis])[0]

    # Get timescales
    tau_values = ltc_layer.cell.tau.numpy()

    # Plot timescale distribution
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Timescale distribution plot
    axes[0].hist(tau_values, bins=30, alpha=0.7, color="blue")
    axes[0].set_xlabel("Timescale Value")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Learned Timescales")
    axes[0].grid(True, alpha=0.3)

    # State dynamics plot - show a subset of units
    num_units_to_plot = min(10, states.shape[1])
    for i in range(num_units_to_plot):
        axes[1].plot(states[:, i], label=f"Unit {i} (τ={tau_values[i]:.3f})")

    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Unit Activation")
    axes[1].set_title("LTC Unit Dynamics Over Time")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def export_ltc_model(model, path):
    """
    Export an LTC model for deployment.

    Args:
        model: Trained LTC model
        path: Export path

    Returns:
        None
    """
    # Try to extract important timescale information before saving
    ltc_params = {}

    # Find LTC layers and extract parameters
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.RNN) and hasattr(layer.cell, "tau"):
            ltc_params[f"ltc_layer_{i}"] = {
                "tau_min": float(layer.cell.tau.numpy().min()),
                "tau_max": float(layer.cell.tau.numpy().max()),
                "tau_mean": float(layer.cell.tau.numpy().mean()),
                "tau_median": float(np.median(layer.cell.tau.numpy())),
            }

    # Save the model with SavedModel format
    model.save(path, save_format="tf")

    # Save additional metadata
    import json

    with open(os.path.join(path, "ltc_metadata.json"), "w") as f:
        json.dump(
            {
                "model_type": "LTC",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timescale_params": ltc_params,
            },
            f,
            indent=4,
        )

    logger.info(f"LTC model exported to {path}")


def load_ltc_model(path):
    """
    Load an LTC model from disk.

    Args:
        path: Path to saved model

    Returns:
        Loaded model
    """
    # Custom object scope to ensure LTC cells are loaded properly
    custom_objects = {
        "LTCCell": LTCCell,
        "TimescaleAnalysisCallback": TimescaleAnalysisCallback,
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(path)

    return model


if __name__ == "__main__":
    import numpy as np

    # Simple test to verify functionality
    print("Testing LTC model implementation...")

    # Generate some test data (sine wave with noise)
    t = np.linspace(0, 20 * np.pi, 1000)
    data = np.sin(t) + 0.1 * np.random.randn(1000)

    # Create sequences
    lookback = 50
    horizon = 10
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)

    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and train a simple LTC model
    model = build_ltc_model(
        num_features=1,
        horizon=horizon,
        lookback=lookback,
        units=64,
        num_layers=2,
        hidden_size=64,
        dropout_rate=0.1,
    )

    print(model.summary())

    # Train the model for a few epochs
    model.fit(
        X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test)
    )

    # Evaluate
    mse = model.evaluate(X_test, y_test)
    print(f"Test MSE: {mse}")

    # Make predictions
    predictions = model.predict(X_test[:5])
    print(f"Predictions shape: {predictions.shape}")

    print("LTC model implementation test completed!")
