# nbeats_model.py
from tensorflow.keras import layers, models, optimizers  # type: ignore


def build_nbeats_model(
    num_features,
    horizon,
    num_blocks=3,
    num_layers=4,
    layer_width=256,
    learning_rate=0.001,
    loss_function="mse",
    lookback=30,
):
    """
    Build a simple N‑BEATS model for time-series forecasting.

    Args:
        num_features: Number of input features.
        horizon: Forecast horizon (number of timesteps to predict).
        num_blocks: Number of fully connected blocks.
        num_layers: Number of layers per block.
        layer_width: Number of neurons per layer.
        learning_rate: Learning rate for the optimizer.
        loss_function: Loss function to use.
        lookback: Length of the input sequence.

    Returns:
        A compiled Keras model.
    """
    # Input layer: shape [batch_size, lookback, num_features]
    inputs = layers.Input(shape=(lookback, num_features))
    # Flatten the time dimension: [batch_size, lookback*num_features]
    x = layers.Flatten()(inputs)

    # Initialize a list to accumulate forecasts from each block
    forecast_outputs = []

    # Loop over blocks; each block is a fully connected stack with backcast adjustment
    for _ in range(num_blocks):
        block = x
        for _ in range(num_layers):
            block = layers.Dense(layer_width, activation="relu")(block)
        # Forecast branch: predict future values
        forecast = layers.Dense(horizon, activation="linear")(block)
        forecast_outputs.append(forecast)
        # Backcast branch: reconstruct input and update residual
        backcast = layers.Dense(lookback * num_features, activation="linear")(block)
        x = layers.Subtract()([x, backcast])

    # Sum forecasts from all blocks to form the final prediction
    if len(forecast_outputs) > 1:
        final_forecast = layers.Add()(forecast_outputs)
    else:
        final_forecast = forecast_outputs[0]

    model = models.Model(inputs=inputs, outputs=final_forecast)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate), loss=loss_function
    )
    return model
