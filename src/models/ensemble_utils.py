"""
Utility functions for working with ensemble models.
Provides weight calculation, combining predictions, and performance tracking.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def create_default_ensemble_weights(model_types: List[str]) -> Dict[str, float]:
    """
    Create equal weights for all model types.

    Args:
        model_types: List of model types

    Returns:
        Dictionary mapping model types to weights
    """
    if not model_types:
        return {}

    weight = 1.0 / len(model_types)
    return {mtype: weight for mtype in model_types}


def get_ensemble_weighter(strategy: str = "equal"):
    """
    Get a basic ensemble weighting function.

    Args:
        strategy: Weighting strategy ("equal", "performance", or "adaptive")

    Returns:
        Function that calculates ensemble weights
    """
    if strategy == "equal":
        return lambda model_types, _: create_default_ensemble_weights(model_types)

    elif strategy == "performance":

        def performance_weighter(model_types, metrics=None):
            if not metrics or not model_types:
                return create_default_ensemble_weights(model_types)

            # Calculate inverse RMSE weights (lower RMSE = higher weight)
            weights = {}
            total_inverse = 0.0

            for mtype in model_types:
                if mtype in metrics and metrics[mtype] > 0:
                    # Use inverse of RMSE as weight
                    weights[mtype] = 1.0 / metrics[mtype]
                    total_inverse += weights[mtype]
                else:
                    weights[mtype] = 0.0

            # Normalize weights
            if total_inverse > 0:
                for mtype in weights:
                    weights[mtype] /= total_inverse
            else:
                # Fallback to equal weights
                return create_default_ensemble_weights(model_types)

            return weights

        return performance_weighter

    elif strategy == "adaptive":

        def adaptive_weighter(model_types, history=None):
            # Default to equal weights if no history
            if not history or not model_types:
                return create_default_ensemble_weights(model_types)

            # Calculate weights based on moving average of performance
            # (Implementation would depend on how history is structured)
            return create_default_ensemble_weights(model_types)  # Fallback for now

        return adaptive_weighter

    # Default to equal weights for unknown strategies
    return lambda model_types, _: create_default_ensemble_weights(model_types)


def get_advanced_ensemble_weighter(config=None):
    """
    Get an advanced ensemble weighter with memory and exploration.

    Args:
        config: Configuration dictionary

    Returns:
        Function that calculates ensemble weights with memory and exploration
    """
    if not config:
        config = {
            "memory_factor": 0.5,  # How much to remember previous weights
            "min_weight": 0.05,  # Minimum weight for any model
            "exploration_factor": 0.1,  # Weight added for exploration
        }

    def advanced_weighter(model_types, metrics=None, previous_weights=None):
        # Start with performance-based weights
        base_weighter = get_ensemble_weighter("performance")
        new_weights = base_weighter(model_types, metrics)

        # If we have previous weights, blend with them
        if previous_weights:
            memory_factor = config.get("memory_factor", 0.5)
            for mtype in model_types:
                if mtype in previous_weights:
                    new_weights[mtype] = memory_factor * previous_weights[mtype] + (
                        1 - memory_factor
                    ) * new_weights.get(mtype, 0.0)

        # Apply minimum weight
        min_weight = config.get("min_weight", 0.05)
        for mtype in model_types:
            new_weights[mtype] = max(min_weight, new_weights.get(mtype, min_weight))

        # Add exploration factor
        exploration = config.get("exploration_factor", 0.1)
        if exploration > 0:
            for mtype in model_types:
                new_weights[mtype] += exploration

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            for mtype in model_types:
                new_weights[mtype] /= total

        return new_weights

    return advanced_weighter


def combine_predictions(
    predictions: Dict[str, np.ndarray], weights: Dict[str, float]
) -> np.ndarray:
    """
    Combine predictions from multiple models using weights.

    Args:
        predictions: Dictionary mapping model types to prediction arrays
        weights: Dictionary mapping model types to weights

    Returns:
        Combined predictions
    """
    if not predictions or not weights:
        return None

    combined = None
    total_weight = 0.0

    for model_type, preds in predictions.items():
        if preds is None or model_type not in weights:
            continue

        weight = weights[model_type]
        if weight <= 0:
            continue

        # Add weighted prediction
        if combined is None:
            combined = weight * preds
        else:
            combined += weight * preds

        total_weight += weight

    # Normalize by total weight
    if combined is not None and total_weight > 0:
        combined /= total_weight

    return combined
