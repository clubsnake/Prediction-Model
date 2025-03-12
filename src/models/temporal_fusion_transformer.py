# Add this to a new file Scripts/models/transformer_models.py

import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Dense,  
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for transformers.
    Uses sine and cosine functions of different frequencies.
    """

    def __init__(self, max_positions=100, d_model=128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_positions = max_positions
        self.d_model = d_model

    def build(self, input_shape):
        """Build the positional encoding matrix."""
        # Get the actual sequence length from input shape
        seq_len = input_shape[1]
        # Limit maximum positions to actual sequence length + small buffer
        actual_max_pos = min(seq_len + 50, self.max_positions)

        positions = tf.range(actual_max_pos, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32)
            * (-tf.math.log(10000.0) / self.d_model)
        )

        pos_encoding = tf.zeros((actual_max_pos, self.d_model))
        indices = tf.range(0, self.d_model, 2)
        updates_sin = tf.sin(positions * div_term)
        updates_cos = tf.cos(positions * div_term)

        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack(
                [
                    tf.repeat(tf.range(actual_max_pos), len(indices)),
                    tf.tile(indices, [actual_max_pos]),
                ],
                axis=1,
            ),
            tf.reshape(updates_sin, [-1]),
        )

        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack(
                [
                    tf.repeat(tf.range(actual_max_pos), len(indices)),
                    tf.tile(indices + 1, [actual_max_pos]),
                ],
                axis=1,
            ),
            tf.reshape(updates_cos, [-1]),
        )

        self.pos_encoding = tf.expand_dims(pos_encoding, 0)
        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs):
        """Add positional encoding to the input."""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"max_positions": self.max_positions, "d_model": self.d_model})
        return config


class VariableSelectionNetwork(tf.keras.layers.Layer):
    """
    Network that selects the most relevant features using a gating mechanism.
    """

    def __init__(self, num_features, hidden_size=64, dropout_rate=0.1, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Layers
        self.grn_layers = [
            GatedResidualNetwork(hidden_size, dropout_rate) for _ in range(num_features)
        ]
        self.softmax_layer = Dense(num_features, activation="softmax")

    def call(self, inputs, training=None):
        """
        Apply variable selection.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            Selected features weighted by importance
        """
        # Individual feature processing
        transformed_features = []
        for i in range(self.num_features):
            # Extract single feature across all timesteps [batch, time, 1]
            feature = tf.expand_dims(inputs[:, :, i], -1)
            # Apply GRN
            transformed = self.grn_layers[i](feature, training=training)
            transformed_features.append(transformed)

        # Stack to [batch, time, num_features, hidden_size]
        feature_stack = tf.stack(transformed_features, axis=2)

        # Calculate weights (gates) for each feature
        flat_inputs = tf.reshape(inputs, [-1, self.num_features])
        weights = self.softmax_layer(flat_inputs)
        weights = tf.reshape(weights, [-1, tf.shape(inputs)[1], self.num_features, 1])

        # Apply weights and sum
        x = tf.reduce_sum(feature_stack * weights, axis=2)
        return x

    def get_config(self):
        config = super(VariableSelectionNetwork, self).get_config()
        config.update(
            {
                "num_features": self.num_features,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    Gated Residual Network as described in the TFT paper.
    """

    def __init__(
        self, hidden_size, dropout_rate=0.1, use_time_distributed=False, **kwargs
    ):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed

        if use_time_distributed:
            self.dense1 = tf.keras.layers.TimeDistributed(
                Dense(hidden_size, activation=None)
            )
            self.dense2 = tf.keras.layers.TimeDistributed(
                Dense(hidden_size, activation=None)
            )
            self.dense3 = tf.keras.layers.TimeDistributed(
                Dense(hidden_size, activation=None)
            )
            self.gate = tf.keras.layers.TimeDistributed(
                Dense(hidden_size, activation="sigmoid")
            )
            self.layer_norm = tf.keras.layers.TimeDistributed(LayerNormalization())
        else:
            self.dense1 = Dense(hidden_size, activation=None)
            self.dense2 = Dense(hidden_size, activation=None)
            self.dense3 = Dense(hidden_size, activation=None)
            self.gate = Dense(hidden_size, activation="sigmoid")
            self.layer_norm = LayerNormalization()

        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, context=None, training=None):
        """
        Apply gated residual network.

        Args:
            inputs: Input tensor
            context: Optional context tensor for conditioning
            training: Training mode flag

        Returns:
            Output tensor after gating and residual connection
        """
        x = self.dense1(inputs)

        if context is not None:
            context_hidden = self.dense3(context)
            x = x + context_hidden

        x = tf.keras.activations.elu(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)

        # Gating mechanism
        gate = self.gate(inputs)
        x = gate * x + (1 - gate) * inputs

        # Residual connection and layer normalization
        return self.layer_norm(x)

    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
                "use_time_distributed": self.use_time_distributed,
            }
        )
        return config


class TemporalFusionTransformer(tf.keras.Model):
    """
    Implementation of Temporal Fusion Transformer (TFT) for time series forecasting.

    Based on the paper "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    by Bryan Lim et al.
    """

    def __init__(
        self,
        num_features,
        forecast_horizon=1,
        hidden_size=64,
        lstm_units=128,
        num_heads=4,
        dropout_rate=0.1,
        max_positions=100,
        **kwargs,
    ):
        super(TemporalFusionTransformer, self).__init__(**kwargs)
        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_positions = max_positions

        # Feature extraction and selection
        self.input_layer = Dense(hidden_size)
        self.variable_selection = VariableSelectionNetwork(
            num_features, hidden_size, dropout_rate
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            max_positions=max_positions, d_model=hidden_size
        )

        # Sequence processing
        self.encoder_lstm = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            activation="tanh",
            recurrent_dropout=0.0,
            unroll=False,
            use_bias=True,
        )

        self.decoder_lstm = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            activation="tanh",
            recurrent_dropout=0.0,
            unroll=False,
            use_bias=True,
        )

        # Self-attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size // num_heads
        )

        # Gated skip connections and layer norms
        self.post_attention_grn = GatedResidualNetwork(
            hidden_size, dropout_rate, use_time_distributed=True
        )

        self.output_layer = Dense(forecast_horizon)

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass of the TFT model.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, num_features]
            training: Boolean indicating whether in training mode
            mask: Input mask tensor (optional)

        Returns:
            Forecasted values
        """
        # Input transformation and variable selection
        x = self.variable_selection(inputs, training=training)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Process with LSTM encoder
        lstm_out = self.encoder_lstm(x)

        # Apply self-attention mechanism
        attn_out = self.attention(
            query=lstm_out, value=lstm_out, key=lstm_out, training=training
        )

        # Apply final gated residual network
        x = self.post_attention_grn(attn_out, training=training)

        # Output layer for predictions
        outputs = self.output_layer(x)

        # Return only the last timestep for predictions
        return outputs[:, -1, :]

    def get_config(self):
        """Get model configuration for serialization."""
        config = super(TemporalFusionTransformer, self).get_config()
        config.update(
            {
                "num_features": self.num_features,
                "forecast_horizon": self.forecast_horizon,
                "hidden_size": self.hidden_size,
                "lstm_units": self.lstm_units,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "max_positions": self.max_positions,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create model instance from saved configuration.

        Args:
            config: Dictionary containing model configuration
            custom_objects: Dictionary mapping names to custom classes or functions

        Returns:
            A model instance.
        """
        return cls(**config)


def build_tft_model(
    num_features,
    horizon,
    learning_rate=0.001,
    hidden_size=64,
    lstm_units=128,
    num_heads=4,
    dropout_rate=0.1,
    loss_function="mse",
    max_positions=100,  # new hyperparameter added
):
    """
    Build and compile a Temporal Fusion Transformer model.

    Args:
        num_features: Number of input features
        horizon: Forecast horizon (number of timesteps to predict)
        learning_rate: Learning rate for the Adam optimizer
        hidden_size: Hidden size for the model
        lstm_units: Number of LSTM units
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
        loss_function: Loss function to use
        max_positions: Maximum number of positions for positional encoding

    Returns:
        Compiled TFT model
    """
    model = TemporalFusionTransformer(
        num_features=num_features,
        forecast_horizon=horizon,
        hidden_size=hidden_size,
        lstm_units=lstm_units,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        max_positions=max_positions,  # now tunable
    )

    # Define input shape and do a forward pass to build the model
    sample_input = tf.random.normal(
        (1, 30, num_features)
    )  # Batch size of 1, sequence length of 30
    model(sample_input)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
    )

    return model


def add_tft_to_model_types():
    """
    Update the model_types config to include TFT.

    This function should be called to register the TFT model
    with the existing framework.
    """
    # Import required modules with proper path handling
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    # Add the model type to the global model types list
    try:
        from config import MODEL_TYPES

        if "tft" not in MODEL_TYPES:
            MODEL_TYPES.append("tft")
    except ImportError:
        print("Warning: Could not import MODEL_TYPES from config")
        return

    # Update the build_model_by_type function to include TFT
    try:
        from model import build_model_by_type as original_build_model

        def extended_build_model_by_type(
            model_type,
            num_features,
            horizon,
            learning_rate,
            dropout_rate,
            loss_function,
            lookback,
            architecture_params=None,
        ):
            """
            Extended version of build_model_by_type that includes TFT.
            """
            if model_type.lower() == "tft":
                # Extract TFT-specific parameters from architecture_params
                if architecture_params is None:
                    architecture_params = {}

                hidden_size = architecture_params.get("hidden_size", 64)
                lstm_units = architecture_params.get("lstm_units", 128)
                num_heads = architecture_params.get("num_heads", 4)
                max_positions = architecture_params.get(
                    "max_positions", 100
                )  # new hyperparameter

                return build_tft_model(
                    num_features=num_features,
                    horizon=horizon,
                    learning_rate=learning_rate,
                    hidden_size=hidden_size,
                    lstm_units=lstm_units,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    loss_function=loss_function,
                    max_positions=max_positions,
                )
            else:
                # Call the original function for other model types
                return original_build_model(
                    model_type,
                    num_features,
                    horizon,
                    learning_rate,
                    dropout_rate,
                    loss_function,
                    lookback,
                    architecture_params,
                )

        # Replace the original function with the extended version
        import model

        model.build_model_by_type = extended_build_model_by_type

        print("Temporal Fusion Transformer model type added successfully.")
    except ImportError:
        print("Warning: Could not update build_model_by_type function")


# Update the Optuna parameter search space to include TFT
def add_tft_to_optuna_search(meta_tuning_module):
    """
    Extend the Optuna parameter search to include TFT hyperparameters.

    Args:
        meta_tuning_module: The meta_tuning module to extend
    """
    # Check if the required attributes exist
    if not hasattr(meta_tuning_module, "ensemble_with_walkforward_objective"):
        print(
            "Warning: meta_tuning_module does not have ensemble_with_walkforward_objective"
        )
        return

    if not hasattr(meta_tuning_module, "MODEL_TYPES"):
        print("Warning: meta_tuning_module does not have MODEL_TYPES")
        return

    if not hasattr(meta_tuning_module, "submodel_params"):
        # Create it if it doesn't exist
        meta_tuning_module.submodel_params = {}

    # Get the original objective function
    original_objective = meta_tuning_module.ensemble_with_walkforward_objective

    def extended_objective(trial, ticker, timeframe, range_cat):
        """
        Extended objective function that includes TFT parameters.
        """
        # Call the original objective to get all other parameters
        result = original_objective(trial, ticker, timeframe, range_cat)

        # Add TFT-specific parameters
        if "tft" in meta_tuning_module.MODEL_TYPES:
            # Add TFT loss function
            loss_functions = getattr(
                meta_tuning_module, "LOSS_FUNCTIONS", ["mae", "mse"]
            )
            tft_loss_fn = trial.suggest_categorical("tft_loss_function", loss_functions)

            # Add TFT hyperparameters
            tft_hidden_size = trial.suggest_categorical(
                "tft_hidden_size", [32, 64, 128, 256]
            )
            tft_lstm_units = trial.suggest_categorical(
                "tft_lstm_units", [64, 128, 256, 512]
            )
            tft_num_heads = trial.suggest_categorical("tft_num_heads", [1, 2, 4, 8])
            tft_dropout = trial.suggest_float("tft_dropout", 0.0, 0.5)

            # Add TFT learning rate
            tft_lr = trial.suggest_float("tft_lr", 1e-5, 1e-2, log=True)

            # Add parameters to submodel_params
            meta_tuning_module.submodel_params["tft"] = {
                "loss_function": tft_loss_fn,
                "hidden_size": tft_hidden_size,
                "lstm_units": tft_lstm_units,
                "num_heads": tft_num_heads,
                "dropout": tft_dropout,
                "lr": tft_lr,
            }

        return result

    # Replace the original objective with the extended version
    meta_tuning_module.ensemble_with_walkforward_objective = extended_objective

    print("TFT hyperparameters added to Optuna search space.")
