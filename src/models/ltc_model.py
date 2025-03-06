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
) -> tf.keras.Model:
    """
    Build and compile a Liquid Time Constant (LTC) model.

    This function creates a model compatible with your existing pipeline,
    supporting both single-layer and multi-layer configurations with optional attention.

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

    Returns:
        Compiled Keras model ready for training
    """
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
