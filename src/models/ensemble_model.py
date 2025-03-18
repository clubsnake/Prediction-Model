"""
Enhanced ensemble model implementation that supports all nine submodels.
This module:
  - Groups models into categories (neural, tree-based, statistical)
  - Prepares data as needed for each group
  - Generates weighted ensemble predictions
  - Calculates model contribution percentages
  - Supports dynamic weight updates
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class EnhancedEnsembleModel:
    """
    Improved ensemble model that efficiently combines predictions
    from a diverse set of models (e.g. lstm, rnn, tft, cnn, ltc, nbeats, tabnet,
    random_forest, xgboost, lightgbm).
    """
    
    def __init__(
        self,
        models: Dict[str, Any] = None,
        weights: Dict[str, float] = None,
        feature_cols: List[str] = None
    ):
        """
        Initialize the ensemble model.
        
        Args:
            models: Dictionary mapping model type to trained model instance.
            weights: Dictionary mapping model type to its ensemble weight.
            feature_cols: List of feature column names used for prediction.
        """
        self.models = models or {}
        self.weights = weights or {}
        self.feature_cols = feature_cols or []
        self.model_groups = self._group_models_by_type()
        
        # Ensure weights exist for all models and are normalized
        self._normalize_weights()

    def _group_models_by_type(self) -> Dict[str, List[str]]:
        """
        Group models into categories for optimized processing:
         - 'neural': lstm, rnn, tft, cnn, ltc, nbeats, tabnet
         - 'tree': random_forest, xgboost, lightgbm
         - 'statistical': any others
        """
        neural_types = {"lstm", "rnn", "tft", "cnn", "ltc", "nbeats", "tabnet"}
        tree_types = {"random_forest", "xgboost", "lightgbm"}
        groups = {"neural": [], "tree": [], "statistical": []}
        for model_type in self.models.keys():
            if model_type in neural_types:
                groups["neural"].append(model_type)
            elif model_type in tree_types:
                groups["tree"].append(model_type)
            else:
                groups["statistical"].append(model_type)
        return groups

    def _normalize_weights(self) -> None:
        """
        Ensure all models have weights and normalize them to sum to 1.0.
        """
        # Add default weights for any models without them
        for model_type in self.models.keys():
            if model_type not in self.weights:
                self.weights[model_type] = 1.0  # Default equal weight
                
        # Calculate total weight
        total_weight = sum(self.weights.values())
        
        # Normalize weights if total is not zero
        if total_weight > 0:
            for model_type in self.weights:
                self.weights[model_type] /= total_weight
        # If total is zero, set equal weights
        elif self.weights:
            equal_weight = 1.0 / len(self.weights)
            for model_type in self.weights:
                self.weights[model_type] = equal_weight

    def _prepare_data(self, X: np.ndarray, group_type: str) -> Any:
        """
        Prepare input data based on model group expectations.
        
        Neural models expect 3D input (samples, timesteps, features).
        Tree-based models expect 2D input.
        Statistical models use the original format.
        """
        if group_type == "neural":
            if len(X.shape) == 2:
                return X.reshape((X.shape[0], 1, X.shape[1]))
            return X
        elif group_type == "tree":
            if len(X.shape) == 3:
                return X.reshape((X.shape[0], -1))
            return X
        else:
            return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate weighted ensemble predictions.
        
        Args:
            X: Input data as a NumPy array.
        Returns:
            Ensemble predictions as a NumPy array.
        """
        if not self.models:
            logger.error("No submodels available for ensemble prediction.")
            return np.array([])
        
        # Store individual model predictions for later analysis
        self.model_predictions = {}
        
        ensemble_pred = None
        total_weight = 0.0
        
        # Process predictions by grouping models
        for group, model_types in self.model_groups.items():
            if not model_types:
                continue
            prepared_data = self._prepare_data(X, group)
            for mtype in model_types:
                model = self.models.get(mtype)
                weight = self.weights.get(mtype, 0)
                if model is None or weight <= 0:
                    continue
                try:
                    pred = model.predict(prepared_data)
                    pred = np.array(pred)  # Ensure prediction is an array
                    
                    # Store individual model prediction
                    self.model_predictions[mtype] = pred
                    
                    if ensemble_pred is None:
                        ensemble_pred = weight * pred
                    else:
                        ensemble_pred += weight * pred
                    total_weight += weight
                except Exception as e:
                    logger.error(f"Error predicting with model '{mtype}': {e}", exc_info=True)
                    
        if ensemble_pred is not None and total_weight > 0:
            return ensemble_pred / total_weight
        return np.array([])

    def calculate_metrics(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Calculate RMSE, MAPE, and directional accuracy for the ensemble and individual models.
        
        Args:
            X: Input features
            y_true: True target values
            
        Returns:
            Dictionary with metrics by model type and ensemble metrics
        """
        # Make predictions with the ensemble
        ensemble_pred = self.predict(X)
        
        # Calculate metrics for the ensemble
        metrics = {
            "ensemble": self._calculate_model_metrics(ensemble_pred, y_true),
            "models": {}
        }
        
        # Calculate metrics for each individual model
        for model_type, predictions in self.model_predictions.items():
            metrics["models"][model_type] = {
                **self._calculate_model_metrics(predictions, y_true),
                "weight": self.weights.get(model_type, 0.0)
            }
        
        # Calculate weighted metrics
        if metrics["models"]:
            weighted_metrics = self._calculate_weighted_metrics(metrics["models"])
            metrics["weighted"] = weighted_metrics
        
        return metrics
    
    def _calculate_model_metrics(self, predictions: np.ndarray, true_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard metrics for a single set of predictions.
        
        Args:
            predictions: Model predictions
            true_values: True target values
            
        Returns:
            Dictionary with metrics
        """
        # Check for empty predictions
        if len(predictions) == 0:
            return {
                "rmse": float('inf'),
                "mape": float('inf'),
                "directional_accuracy": 0.0
            }
        
        # Calculate RMSE
        mse = np.mean(np.square(predictions - true_values))
        rmse = np.sqrt(mse)
        
        # Calculate MAPE
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        mape = np.mean(np.abs((true_values - predictions) / (np.abs(true_values) + epsilon))) * 100
        
        # Calculate directional accuracy
        if len(predictions) > 1:
            direction_actual = np.sign(np.diff(true_values, axis=0))
            direction_pred = np.sign(np.diff(predictions, axis=0))
            
            # Count matches
            matches = np.sum(direction_actual == direction_pred)
            total = len(direction_actual)
            
            if total > 0:
                da = (matches / total) * 100
            else:
                da = 0.0
        else:
            da = 0.0  # Not enough data points for directional accuracy
        
        return {
            "rmse": float(rmse),
            "mape": float(mape),
            "directional_accuracy": float(da)
        }
    
    def _calculate_weighted_metrics(self, model_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate weighted ensemble metrics based on individual model metrics and weights.
        
        Args:
            model_metrics: Dictionary mapping model types to their metrics
            
        Returns:
            Dictionary with weighted ensemble metrics
        """
        # Initialize weighted metrics
        weighted_metrics = {
            "rmse": 0.0,
            "mape": 0.0,
            "directional_accuracy": 0.0
        }
        
        # Calculate weighted sum
        total_weight = 0.0
        for model_type, metrics in model_metrics.items():
            weight = metrics.get("weight", 0.0)
            if weight > 0:
                weighted_metrics["rmse"] += weight * metrics.get("rmse", 0.0)
                weighted_metrics["mape"] += weight * metrics.get("mape", 0.0)
                weighted_metrics["directional_accuracy"] += weight * metrics.get("directional_accuracy", 0.0)
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for metric in weighted_metrics:
                weighted_metrics[metric] /= total_weight
        
        return weighted_metrics

    def get_model_contributions(self, X: np.ndarray) -> Dict[str, float]:
        """
        Calculate the contribution of each submodel to the final prediction.
        
        Args:
            X: Input data.
        Returns:
            Dictionary mapping each model type to its percentage contribution.
        """
        contributions = {}
        total_weight = sum(self.weights.get(m, 0) for m in self.models.keys())
        if total_weight > 0:
            for mtype in self.models.keys():
                contributions[mtype] = (self.weights.get(mtype, 0) / total_weight) * 100
        return contributions

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update the ensemble weights.
        
        Args:
            new_weights: Dictionary mapping model types to new weights.
        """
        self.weights = new_weights
        # Re-normalize weights
        self._normalize_weights()
        logger.info(f"Ensemble weights updated: {self.weights}")

    def info(self) -> Dict[str, Any]:
        """
        Retrieve metadata about the ensemble model.
        
        Returns:
            Dictionary with information on submodel types, weights, groups, and features.
        """
        return {
            "model_types": list(self.models.keys()),
            "weights": self.weights,
            "model_groups": self.model_groups,
            "feature_columns": self.feature_cols
        }

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble model and all submodels on the provided data.
        
        Args:
            X: Input features
            y_true: True target values
            
        Returns:
            Dictionary with evaluation metrics for ensemble and individual models
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        
        results = {
            "ensemble": {},
            "models": {}
        }
        
        # Get ensemble prediction
        y_pred_ensemble = self.predict(X)
        
        # Calculate ensemble metrics
        if len(y_pred_ensemble) > 0:
            # RMSE for ensemble
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_ensemble))
            results["ensemble"]["rmse"] = float(rmse)
            
            # MAPE for ensemble
            try:
                mape = mean_absolute_percentage_error(y_true, y_pred_ensemble) * 100
                results["ensemble"]["mape"] = float(mape)
            except Exception as e:
                logger.warning(f"Error calculating MAPE: {e}")
                results["ensemble"]["mape"] = float('inf')
            
            # Directional Accuracy
            direction_actual = np.sign(np.diff(y_true, axis=0))
            direction_pred = np.sign(np.diff(y_pred_ensemble, axis=0))
            matching_directions = np.sum(direction_actual == direction_pred)
            total_directions = len(direction_actual)
            
            if total_directions > 0:
                da_pct = (matching_directions / total_directions) * 100
                results["ensemble"]["directional_accuracy"] = float(da_pct)
            else:
                results["ensemble"]["directional_accuracy"] = 0.0
        
        # Calculate metrics for each individual model
        for model_type, model in self.models.items():
            if model is None:
                continue
                
            try:
                # Prepare data for this model type
                group = next((g for g, types in self.model_groups.items() if model_type in types), "statistical")
                prepared_data = self._prepare_data(X, group)
                
                # Get prediction from this model
                y_pred = model.predict(prepared_data)
                
                # Calculate metrics
                model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                results["models"][model_type] = {
                    "rmse": float(model_rmse),
                    "weight": self.weights.get(model_type, 0.0)
                }
                
                try:
                    model_mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    results["models"][model_type]["mape"] = float(model_mape)
                except:
                    results["models"][model_type]["mape"] = float('inf')
                    
                # Calculate directional accuracy for this model
                direction_pred_model = np.sign(np.diff(y_pred, axis=0))
                matching_dir_model = np.sum(direction_actual == direction_pred_model)
                
                if total_directions > 0:
                    model_da = (matching_dir_model / total_directions) * 100
                    results["models"][model_type]["directional_accuracy"] = float(model_da)
                else:
                    results["models"][model_type]["directional_accuracy"] = 0.0
                    
            except Exception as e:
                logger.error(f"Error evaluating model {model_type}: {e}")
                results["models"][model_type] = {
                    "rmse": float('inf'),
                    "mape": float('inf'),
                    "directional_accuracy": 0.0,
                    "weight": self.weights.get(model_type, 0.0),
                    "error": str(e)
                }
                
        return results
        
    def save_weights(self, filepath: str) -> bool:
        """
        Save the current ensemble weights to a file.
        
        Args:
            filepath: Path to save the weights
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump({
                    "weights": self.weights,
                    "timestamp": str(pd.Timestamp.now())
                }, f, indent=2)
            logger.info(f"Saved ensemble weights to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving ensemble weights: {e}")
            return False
            
    def load_weights(self, filepath: str) -> bool:
        """
        Load ensemble weights from a file.
        
        Args:
            filepath: Path to load the weights from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            import os
            if not os.path.exists(filepath):
                logger.warning(f"Weights file not found: {filepath}")
                return False
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if "weights" in data:
                self.weights = data["weights"]
                # Re-normalize weights
                self._normalize_weights()
                logger.info(f"Loaded ensemble weights from {filepath}")
                return True
            else:
                logger.warning(f"No weights found in {filepath}")
                return False
        except Exception as e:
            logger.error(f"Error loading ensemble weights: {e}")
            return False

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

def calculate_ensemble_metrics(
    predictions: Dict[str, np.ndarray], 
    weights: Dict[str, float],
    y_true: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate metrics for individual models and the weighted ensemble.
    
    Args:
        predictions: Dictionary mapping model type to prediction arrays
        weights: Dictionary mapping model type to weights
        y_true: True target values
        
    Returns:
        Dictionary with metrics for each model and the ensemble
    """
    metrics = {
        "models": {},
        "ensemble": {}
    }
    
    # Normalize weights for consistency
    total_weight = sum(weights.values())
    normalized_weights = {
        model: weight/total_weight for model, weight in weights.items()
    } if total_weight > 0 else weights
    
    # Calculate metrics for each model
    for model_type, pred in predictions.items():
        if pred is None:
            continue
            
        # Skip models with zero weight
        if weights.get(model_type, 0) <= 0:
            continue
            
        try:
            # Calculate RMSE
            mse = np.mean(np.square(pred - y_true))
            rmse = np.sqrt(mse)
            
            # Calculate MAPE with epsilon to avoid division by zero
            epsilon = 1e-10
            mape = np.mean(np.abs((y_true - pred) / (np.abs(y_true) + epsilon))) * 100
            
            # Calculate directional accuracy
            if len(pred) > 1:
                direction_actual = np.sign(np.diff(y_true, axis=0))
                direction_pred = np.sign(np.diff(pred, axis=0))
                
                # Count matches
                matches = np.sum(direction_actual == direction_pred)
                total = len(direction_actual)
                
                if total > 0:
                    da = (matches / total) * 100
                else:
                    da = 0.0
            else:
                da = 0.0
                
            # Store metrics for this model
            metrics["models"][model_type] = {
                "rmse": float(rmse),
                "mape": float(mape),
                "directional_accuracy": float(da),
                "weight": normalized_weights.get(model_type, 0.0)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_type}: {e}")
    
    # Calculate ensemble prediction
    ensemble_pred = combine_predictions(predictions, weights)
    
    # Calculate ensemble metrics
    if ensemble_pred is not None:
        try:
            # RMSE
            mse = np.mean(np.square(ensemble_pred - y_true))
            rmse = np.sqrt(mse)
            
            # MAPE
            epsilon = 1e-10
            mape = np.mean(np.abs((y_true - ensemble_pred) / (np.abs(y_true) + epsilon))) * 100
            
            # Directional accuracy
            if len(ensemble_pred) > 1:
                direction_actual = np.sign(np.diff(y_true, axis=0))
                direction_pred = np.sign(np.diff(ensemble_pred, axis=0))
                
                # Count matches
                matches = np.sum(direction_actual == direction_pred)
                total = len(direction_actual)
                
                if total > 0:
                    da = (matches / total) * 100
                else:
                    da = 0.0
            else:
                da = 0.0
                
            # Store ensemble metrics
            metrics["ensemble"] = {
                "rmse": float(rmse),
                "mape": float(mape),
                "directional_accuracy": float(da)
            }
        except Exception as e:
            logger.error(f"Error calculating ensemble metrics: {e}")
    
    # Calculate weighted metrics (theoretical combination based on model metrics)
    if metrics["models"]:
        weighted_metrics = {
            "rmse": 0.0,
            "mape": 0.0,
            "directional_accuracy": 0.0
        }
        
        # Calculate weighted sum
        total_weight = 0.0
        for model_type, model_metrics in metrics["models"].items():
            weight = normalized_weights.get(model_type, 0.0)
            if weight > 0:
                weighted_metrics["rmse"] += weight * model_metrics.get("rmse", 0.0)
                weighted_metrics["mape"] += weight * model_metrics.get("mape", 0.0)
                weighted_metrics["directional_accuracy"] += weight * model_metrics.get("directional_accuracy", 0.0)
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for metric in weighted_metrics:
                weighted_metrics[metric] /= total_weight
        
        metrics["weighted"] = weighted_metrics
    
    return metrics