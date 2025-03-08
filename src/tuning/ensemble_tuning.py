"""
This module provides functions for training an ensemble
of models and creates a weighted ensemble predictor.
"""

import logging
from typing import Dict
import threading
from src.utils.training_optimizer import get_training_optimizer

import numpy as np

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
        model_configs.append({
            "model_type": model_type,
            "params": params
        })
    
    # Use training optimizer to determine optimal grouping
    training_groups = training_optimizer.parallel_training_groups(model_configs)
    
    # Train each group in sequence
    for group_idx, group in enumerate(training_groups):
        logger.info(f"Training ensemble model group {group_idx+1}/{len(training_groups)}")
        threads = []
        
        for model_config in group:
            model_type = model_config["model_type"]
            params = model_config["params"]
            
            # Create training thread for this model
            thread = threading.Thread(
                target=_train_ensemble_model,
                args=(model_type, params, X_train, y_train, feature_cols, horizon, models, model_lock)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads in this group to complete
        for thread in threads:
            thread.join()
            
        # Clean GPU memory between groups
        if training_optimizer.has_gpu:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
            except Exception as e:
                logger.warning(f"Error cleaning GPU memory: {e}")

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

def _train_ensemble_model(model_type, params, X_train, y_train, feature_cols, horizon, models, model_lock):
    """Helper function to train a single ensemble model in a thread"""
    try:
        # Get model builder using LazyImportManager
        build_model_by_type = LazyImportManager.get_model_builder()
        
        # Get optimal configuration for this model type
        model_config = training_optimizer.get_model_config(model_type, "medium")
        
        # Build the model using builder from LazyImportManager
        model = build_model_by_type(
            model_type=model_type,
            num_features=len(feature_cols),
            horizon=horizon,
            learning_rate=params.get("lr", model_config["learning_rate"]),
            dropout_rate=params.get("dropout", 0.2),
            loss_function=params.get("loss_function", "mean_squared_error"),
            lookback=params.get("lookback", 30),
            architecture_params=params.get("architecture_params", {}),
        )

        # Extract training parameters with default values from training_optimizer
        training_params = {
            "epochs": params.get("epochs", 10),
            "batch_size": params.get("batch_size", model_config["batch_size"]),
            "verbose": 0,
        }

        # Train the model
        logger.info(f"Training {model_type} model...")
        model.fit(X_train, y_train, **training_params)
        logger.info(f"Completed training {model_type} model")
        
        # Store the model in the shared models dictionary
        with model_lock:
            models[model_type] = model

    except Exception as e:
        logger.error(f"Error training {repr(model_type)}: {e}")
        # Skip this model in the ensemble
