"""
This module provides functions for training an ensemble
of models and creates a weighted ensemble predictor.
"""

import logging
from typing import Dict

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Import the LazyImportManager to avoid circular imports
from meta_tuning import LazyImportManager


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

    # Get model builder using LazyImportManager
    build_model_by_type = LazyImportManager.get_model_builder()

    # Train each model type
    for model_type, params in submodel_params.items():
        try:
            # Build the model using builder from LazyImportManager
            models[model_type] = build_model_by_type(
                model_type=model_type,
                num_features=len(feature_cols),
                horizon=horizon,
                learning_rate=params.get("lr", 0.001),
                dropout_rate=params.get("dropout", 0.2),
                loss_function=params.get("loss_function", "mean_squared_error"),
                lookback=params.get("lookback", 30),
                architecture_params=params.get("architecture_params", {}),
            )

            # Extract training parameters
            training_params = {
                "epochs": params.get("epochs", 10),
                "batch_size": params.get("batch_size", 32),
                "verbose": 0,
            }

            # Train the model
            logger.info(f"Training {model_type} model...")
            models[model_type].fit(X_train, y_train, **training_params)
            logger.info(f"Completed training {model_type} model")

        except Exception as e:
            logger.error(f"Error training {repr(model_type)}: {e}")
            # Skip this model in the ensemble

    # Return ensemble predictor function
    def ensemble_predict(X, weights=None):
        if weights is None:
            # Equal weighting by default if no weights provided
            weights = {mtype: 1.0 / len(models) for mtype in models}

        predictions = {}
        for mtype, model in models.items():
            if weights.get(mtype, 0) > 0:
                try:
                    predictions[mtype] = model.predict(X)
                except Exception as e:
                    logger.error(f"Error predicting with {repr(mtype)}: {e}")

        # Combine predictions
        return combine_predictions(predictions, weights)

    return ensemble_predict, models
