"""
Robust model serialization, versioning and management.

This module provides functions for saving and loading models with proper versioning,
metadata tracking, and integrity verification.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import tensorflow as tf

# Import the file locking utilities from your threadsafe module
from src.utils.threadsafe import safe_read_json, safe_write_json

# Configure logging
logger = logging.getLogger("ModelManager")


class ModelSerializationError(Exception):
    """Exception raised for model serialization errors."""


class ModelVersionManager:
    """
    Manages versioned models with metadata and incremental updates.
    """

    def __init__(self, base_dir: str = "models"):
        """
        Initialize the model version manager.

        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_model(
        self,
        model: tf.keras.Model,
        ticker: str,
        timeframe: str,
        metrics: Dict[str, float] = None,
        hyperparams: Dict[str, Any] = None,
        feature_importance: Dict[str, float] = None,
        training_history: Dict[str, List[float]] = None,
        incremental: bool = True,
        save_format: str = "h5",
    ) -> str:
        """
        Save a model with versioning and metadata.

        Args:
            model: The TensorFlow model to save
            ticker: Ticker symbol the model is for
            timeframe: Timeframe the model is for
            metrics: Performance metrics for the model
            hyperparams: Hyperparameters used to train the model
            feature_importance: Feature importance scores
            training_history: Training history dictionary
            incremental: Whether to create a new version or update the latest
            save_format: Format to save in ('h5' or 'saved_model')

        Returns:
            Version identifier of the saved model
        """
        # Determine model directory based on ticker and timeframe
        model_key = f"{ticker}_{timeframe}"
        model_dir = os.path.join(self.base_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)

        # Load existing metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata = safe_read_json(metadata_path, default={"versions": []})

        # Determine new version number
        current_version = 0
        for version_info in metadata["versions"]:
            version_num = version_info.get("version", 0)
            if version_num > current_version:
                current_version = version_num

        if incremental and metadata["versions"]:
            # Create a new incremented version
            new_version = current_version + 1
        else:
            # Otherwise start at version 1
            new_version = 1

        # Create version directory
        version_dir = os.path.join(model_dir, f"v{new_version}")
        os.makedirs(version_dir, exist_ok=True)

        # Save model file using the chosen format
        model_filename = "model.h5" if save_format == "h5" else "saved_model"
        model_path = os.path.join(version_dir, model_filename)
        try:
            if save_format == "h5":
                model.save(model_path, save_format="h5")
            elif save_format == "saved_model":
                model.save(model_path, save_format="tf")
            else:
                raise ModelSerializationError("Unsupported save format")
        except Exception as e:
            raise ModelSerializationError(f"Error saving model: {e}")

        # Prepare parameters and metadata for this version
        params = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparams": hyperparams,
            "feature_importance": feature_importance,
            "training_history": training_history,
            "model_path": model_path,
        }
        params_path = os.path.join(version_dir, "params.json")
        success = safe_write_json(params_path, params)
        if not success:
            raise ModelSerializationError("Failed to write model parameters")

        # Update overall metadata with this new version
        metadata["versions"].append(params)
        safe_write_json(metadata_path, metadata)

        # Optionally, update a symlink to the latest version
        latest_link = os.path.join(model_dir, "latest")
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            os.remove(latest_link)
        os.symlink(version_dir, latest_link)

        logger.info(f"Model saved as version {new_version} at {model_path}")
        return str(new_version)
