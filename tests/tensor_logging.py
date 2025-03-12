"""
Utility for logging tensor shapes during model execution, especially useful for debugging
DirectML compatibility issues or dimensional mismatches.
"""

import logging
import os

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Check if we're using DirectML
is_directml = (
    "TENSORFLOW_USE_DIRECTML" in os.environ or "DML_VISIBLE_DEVICES" in os.environ
)


def log_tensor_shape(tensor, name: str, log_level: str = "debug"):
    """
    Log the shape and basic statistics of a tensor.

    Args:
        tensor: TensorFlow tensor or NumPy array
        name: Name to identify the tensor in logs
        log_level: Logging level ('debug', 'info', 'warning')
    """
    try:
        # Handle both TF tensors and NumPy arrays
        if tensor is None:
            logger.warning(f"Tensor '{name}' is None")
            return

        # Convert to numpy if it's a TensorFlow tensor
        if hasattr(tensor, "numpy"):
            try:
                # For TF tensors/EagerTensor
                np_tensor = tensor.numpy()
            except:
                # Some TF objects can't be converted directly
                logger.debug(f"Could not convert {name} to numpy, logging shape only")
                shape_str = f"Shape: {tensor.shape}"
                _log_at_level(log_level, f"Tensor '{name}': {shape_str}")
                return
        else:
            np_tensor = tensor

        # Log shape and basic stats
        shape_str = f"Shape: {np_tensor.shape}"

        # Only compute stats for numeric data that's not too large
        if np.issubdtype(np_tensor.dtype, np.number) and np_tensor.size < 1000000:
            try:
                stats_str = (
                    f"Min: {np.min(np_tensor):.4f}, Max: {np.max(np_tensor):.4f}, "
                    f"Mean: {np.mean(np_tensor):.4f}, Std: {np.std(np_tensor):.4f}"
                )
                _log_at_level(log_level, f"Tensor '{name}': {shape_str}, {stats_str}")
            except:
                # Fall back to just shape if stats fail
                _log_at_level(log_level, f"Tensor '{name}': {shape_str}")
        else:
            _log_at_level(
                log_level, f"Tensor '{name}': {shape_str} (dtype: {np_tensor.dtype})"
            )

    except Exception as e:
        logger.warning(f"Error logging tensor '{name}': {e}")


def _log_at_level(level: str, message: str):
    """Log at specified level."""
    if level.lower() == "debug":
        logger.debug(message)
    elif level.lower() == "info":
        logger.info(message)
    elif level.lower() == "warning":
        logger.warning(message)
    else:
        logger.debug(message)


class TensorShapeLogger(tf.keras.callbacks.Callback):
    """
    Keras callback to log tensor shapes during training.
    Useful for debugging DirectML compatibility issues.
    """

    def __init__(self, log_frequency=50, log_inputs=True, log_outputs=True):
        super().__init__()
        self.log_frequency = log_frequency
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_frequency != 0:
            return

        logger.debug(f"Batch {batch} shapes:")

        if hasattr(self.model, "inputs") and self.log_inputs:
            for i, inp in enumerate(self.model.inputs):
                log_tensor_shape(inp, f"input_{i}")

        if hasattr(self.model, "outputs") and self.log_outputs:
            for i, out in enumerate(self.model.outputs):
                log_tensor_shape(out, f"output_{i}")


def wrap_layer_for_shape_logging(layer, name=None):
    """
    Wrap a Keras layer to log input and output shapes during forward pass.

    Args:
        layer: Keras layer to wrap
        name: Name prefix for the logs

    Returns:
        Wrapped layer with same functionality but added logging
    """
    original_call = layer.call
    layer_name = name or layer.name

    def logged_call(self, inputs, *args, **kwargs):
        # Log input shape(s)
        if isinstance(inputs, list):
            for i, inp in enumerate(inputs):
                log_tensor_shape(inp, f"{layer_name}_input_{i}", "debug")
        else:
            log_tensor_shape(inputs, f"{layer_name}_input", "debug")

        # Original call
        outputs = original_call(inputs, *args, **kwargs)

        # Log output shape(s)
        if isinstance(outputs, list):
            for i, out in enumerate(outputs):
                log_tensor_shape(out, f"{layer_name}_output_{i}", "debug")
        else:
            log_tensor_shape(outputs, f"{layer_name}_output", "debug")

        return outputs

    # Replace the call method
    layer.call = logged_call.__get__(layer)
    return layer


def log_all_layer_shapes(model, input_shape):
    """
    Run a forward pass through the model with dummy data and log all layer shapes.

    Args:
        model: Keras model to analyze
        input_shape: Input shape to use (without batch dimension)
    """
    logger.info(f"Logging shapes for model: {model.name}")

    # Create dummy input data
    if isinstance(input_shape, list):
        # Handle multiple inputs
        dummy_inputs = [tf.ones((1,) + shape) for shape in input_shape]
    else:
        # Single input
        dummy_inputs = tf.ones((1,) + input_shape)

    # Create intermediate model for each layer to get its output
    for i, layer in enumerate(model.layers):
        try:
            # Create intermediate model up to this layer
            intermediate_model = tf.keras.Model(
                inputs=model.inputs, outputs=layer.output
            )

            # Run forward pass
            intermediate_output = intermediate_model(dummy_inputs)

            # Log the layer output shape
            if isinstance(intermediate_output, list):
                for j, out in enumerate(intermediate_output):
                    log_tensor_shape(
                        out, f"Layer {i}: {layer.name} (output {j})", "info"
                    )
            else:
                log_tensor_shape(
                    intermediate_output, f"Layer {i}: {layer.name}", "info"
                )

        except Exception as e:
            logger.warning(f"Could not log shapes for layer {layer.name}: {e}")


def instrument_model_for_shape_logging(model):
    """
    Add shape logging to all layers in a model.

    Args:
        model: Keras model to instrument

    Returns:
        Model with shape logging added
    """
    # Add logging to each layer
    for layer in model.layers:
        wrap_layer_for_shape_logging(layer)

    logger.info(
        f"Added shape logging to all {len(model.layers)} layers in model {model.name}"
    )
    return model


# Configure basic setup if running on DirectML
if is_directml:
    logger.info(
        "DirectML detected - enabling tensor shape logging for compatibility debugging"
    )
    # Set this variable to True to enable detailed shape logging in your models
    ENABLE_SHAPE_LOGGING = True
else:
    ENABLE_SHAPE_LOGGING = False
