"""
This module provides functions for training an ensemble
of models and creates a weighted ensemble predictor.
"""

import logging
import threading
import traceback
from typing import Dict

import numpy as np

from src.utils.training_optimizer import get_training_optimizer

# Set up logging
logger = logging.getLogger(__name__)

# Import the LazyImportManager to avoid circular imports
from meta_tuning import LazyImportManager

# Initialize training optimizer
training_optimizer = get_training_optimizer()


def combine_predictions(
    predictions: Dict[str, np.ndarray], weights: Dict[str, float]
) -> np.ndarray:
    """
    Combine predictions from multiple models using weighted average.

    Args:
        predictions: Dictionary mapping model type to prediction arrays
        weights: Dictionary mapping model type to weights

    Returns:
        Weighted average of predictions
    """
    # Filter out models with no predictions
    valid = [
        (mtype, pred)
        for mtype, pred in predictions.items()
        if pred is not None and weights.get(mtype, 0) > 0
    ]

    if not valid:
        return None

    # Calculate total weight for normalization
    total_weight = sum(weights.get(mtype, 0) for mtype, _ in valid)

    if total_weight == 0:
        return None

    # Create weighted sum
    result = None
    for mtype, pred in valid:
        weight = weights.get(mtype, 0) / total_weight  # Normalized weight

        if result is None:
            result = weight * pred
        else:
            result += weight * pred

    return result


def train_ensemble_models(submodel_params, X_train, y_train, feature_cols, horizon):
    """Train multiple models and return a weighted ensemble predictor"""
    models = {}
    model_lock = threading.RLock()  # Thread-safe model access

    # Get model builder using LazyImportManager
    build_model_by_type = LazyImportManager.get_model_builder()

    # Group models for efficient parallel training
    model_configs = []
    for model_type, params in submodel_params.items():
        model_configs.append({"model_type": model_type, "params": params})

    # Use training optimizer to determine optimal grouping
    training_groups = training_optimizer.parallel_training_groups(model_configs)

    # Train each group in sequence
    for group_idx, group in enumerate(training_groups):
        logger.info(
            f"Training ensemble model group {group_idx+1}/{len(training_groups)}"
        )
        threads = []

        for model_config in group:
            thread = threading.Thread(
                target=_train_ensemble_model,
                args=(
                    model_config["model_type"],
                    model_config["params"],
                    X_train,
                    y_train,
                    feature_cols,
                    horizon,
                    models,
                    model_lock
                )
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads in this group to complete
        for thread in threads:
            thread.join()

        # Clean GPU memory between groups
        if training_optimizer.has_gpu:
            from src.utils.memory_utils import cleanup_tf_session
            cleanup_tf_session()

    # Return ensemble predictor function
    def ensemble_predict(X, weights=None):
        if weights is None:
            weights = {mtype: 1.0 for mtype in models.keys()}

        predictions = {}
        for mtype, model in models.items():
            try:
                pred = model.predict(X, verbose=0)
                predictions[mtype] = pred
            except Exception as e:
                logger.error(f"Error predicting with {mtype} model: {e}")

        # Combine predictions
        return combine_predictions(predictions, weights)

    return ensemble_predict, models


def _train_ensemble_model(
    model_type, params, X_train, y_train, feature_cols, horizon, models, model_lock
):
    """Helper function to train a single ensemble model in a thread"""
    try:
        # Get model builder using LazyImportManager
        build_model_by_type = LazyImportManager.get_model_builder()

        # Get optimal configuration for this model type
        model_config = training_optimizer.get_model_config(model_type, "medium")

        # Special handling for nbeats model
        if model_type == "nbeats":
            from src.models.nbeats_model import build_nbeats_model
            model = build_nbeats_model(
                lookback=params.get("lookback", 30),
                horizon=horizon,
                num_features=len(feature_cols),
                learning_rate=params.get("lr", 0.001),
                layer_width=params.get("layer_width", 256),
                num_blocks=params.get("num_blocks", [3, 3]),
                num_layers=params.get("num_layers", [4, 4]),
                thetas_dim=params.get("thetas_dim", 10),
                include_price_specific_stack=params.get("include_price_specific_stack", True),
                dropout_rate=params.get("dropout_rate", 0.1),
                use_batch_norm=params.get("use_batch_norm", True),
            )
        elif model_type == "ltc":
            from src.models.ltc_model import build_ltc_model
            model = build_ltc_model(
                num_features=len(feature_cols),
                horizon=horizon,
                learning_rate=params.get("lr", 0.001),
                loss_function=params.get("loss_function", "mse"),
                lookback=params.get("lookback", 30),
                units=params.get("units", 64),
                num_layers=params.get("num_layers", 1),
                use_attention=params.get("use_attention", False),
                dropout_rate=params.get("dropout_rate", 0.1),
                recurrent_dropout_rate=params.get("recurrent_dropout_rate", 0.0),
            )
        else:
            # Create model using architecture parameters
            model = build_model_by_type(
                model_type=model_type,
                num_features=len(feature_cols),
                horizon=horizon,
                learning_rate=params.get("lr", 0.001),
                dropout_rate=params.get("dropout", 0.2),
                loss_function=params.get("loss_function", "mean_squared_error"),
                lookback=params.get("lookback", 30),
                architecture_params=params,
            )

        # Extract training parameters with default values from training_optimizer
        batch_size = params.get("batch_size", model_config.get("batch_size", 32))
        epochs = params.get("epochs", 10)
        
        # Train model with optimized parameters
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Store model in shared dictionary with thread safety
        with model_lock:
            models[model_type] = model
            
        logger.info(f"Successfully trained {model_type} model")

    except Exception as e:
        logger.error(f"Error training {model_type} model: {e}")
        logger.error(traceback.format_exc())
