"""
Robust model serialization, versioning and management.

This module provides functions for saving and loading models with proper versioning,
metadata tracking, and integrity verification.
"""

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Union

import tensorflow as tf

# Import the file locking utilities from your threadsafe module
from src.utils.threadsafe import safe_read_json, safe_write_json

# Configure logging
logger = logging.getLogger("ModelManager")


class ModelSerializationError(Exception):
    """Exception raised for model serialization errors."""

    pass


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""

    pass


class ModelVersionManager:
    """
    Manages versioned models with metadata and incremental updates.
    """

    def __init__(self, base_dir: str = "models"):
        """
        Initialize the model version manager.

        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Initialized ModelVersionManager with base directory: {base_dir}")

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
            model: The model to save
            ticker: Ticker symbol the model was trained on
            timeframe: Timeframe the model was trained on
            metrics: Performance metrics for the model
            hyperparams: Hyperparameters used for training
            feature_importance: Feature importance scores
            training_history: Training history data
            incremental: Whether to increment version if model exists
            save_format: Format to save model in ('h5' or 'saved_model')

        Returns:
            Path to the saved model
        """
        # Create directory structure
        model_dir = os.path.join(self.base_dir, ticker, timeframe)
        os.makedirs(model_dir, exist_ok=True)

        # Get current version or start at 1
        current_version = self.get_latest_version(ticker, timeframe)
        if incremental:
            version = current_version + 1
        else:
            version = current_version

        # Create version directory
        version_dir = os.path.join(model_dir, f"v{version}")
        if os.path.exists(version_dir) and incremental:
            # Remove old version if overwriting
            shutil.rmtree(version_dir)
        os.makedirs(version_dir, exist_ok=True)

        # Save the model
        try:
            if save_format == "h5":
                model_path = os.path.join(version_dir, "model.h5")
                model.save(model_path, save_format="h5")
            else:
                model_path = os.path.join(version_dir, "saved_model")
                model.save(model_path, save_format="tf")
        except Exception as e:
            raise ModelSerializationError(f"Failed to save model: {str(e)}")

        # Create metadata
        metadata = {
            "version": version,
            "ticker": ticker,
            "timeframe": timeframe,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "hyperparams": hyperparams or {},
            "feature_importance": feature_importance or {},
        }

        # Save metadata
        metadata_path = os.path.join(version_dir, "metadata.json")
        safe_write_json(metadata, metadata_path)

        # Save training history if provided
        if training_history:
            history_path = os.path.join(version_dir, "history.json")
            safe_write_json(training_history, history_path)

        logger.info(f"Saved model {ticker}/{timeframe}/v{version}")
        return model_path

    def load_model(
        self,
        ticker: str,
        timeframe: str,
        version: Union[int, str] = "latest",
        with_metadata: bool = True,
    ) -> Union[tf.keras.Model, tuple]:
        """
        Load a model with optional metadata.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            version: Version to load (int or 'latest')
            with_metadata: Whether to return metadata with the model

        Returns:
            The loaded model, or a tuple of (model, metadata) if with_metadata=True

        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        # Resolve version number if 'latest'
        if version == "latest":
            version = self.get_latest_version(ticker, timeframe)
            if version == 0:
                raise ModelLoadError(f"No models found for {ticker}/{timeframe}")

        # Construct paths
        version_dir = os.path.join(self.base_dir, ticker, timeframe, f"v{version}")
        if not os.path.exists(version_dir):
            raise ModelLoadError(
                f"Model version {version} not found for {ticker}/{timeframe}"
            )

        # Check for model format (h5 or saved_model)
        model_h5_path = os.path.join(version_dir, "model.h5")
        model_saved_path = os.path.join(version_dir, "saved_model")

        # Load the model
        try:
            if os.path.exists(model_h5_path):
                model = tf.keras.models.load_model(model_h5_path)
            elif os.path.exists(model_saved_path):
                model = tf.keras.models.load_model(model_saved_path)
            else:
                raise ModelLoadError(f"No model file found in {version_dir}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

        # Load metadata if requested
        if with_metadata:
            metadata_path = os.path.join(version_dir, "metadata.json")
            metadata = (
                safe_read_json(metadata_path) if os.path.exists(metadata_path) else {}
            )
            return model, metadata
        else:
            return model

    def list_versions(self, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        List all versions for a ticker and timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe

        Returns:
            List of metadata dictionaries for each version
        """
        model_dir = os.path.join(self.base_dir, ticker, timeframe)
        if not os.path.exists(model_dir):
            return []

        versions = []
        for item in os.listdir(model_dir):
            if item.startswith("v") and os.path.isdir(os.path.join(model_dir, item)):
                version_num = int(item[1:])  # Remove 'v' prefix
                metadata_path = os.path.join(model_dir, item, "metadata.json")

                if os.path.exists(metadata_path):
                    metadata = safe_read_json(metadata_path)
                    versions.append(metadata)
                else:
                    # Create minimal metadata if file doesn't exist
                    versions.append(
                        {
                            "version": version_num,
                            "ticker": ticker,
                            "timeframe": timeframe,
                        }
                    )

        # Sort by version number
        versions.sort(key=lambda x: x.get("version", 0))
        return versions

    def get_latest_version(self, ticker: str, timeframe: str) -> int:
        """
        Get the latest version number for a ticker and timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe

        Returns:
            Latest version number or 0 if none exists
        """
        versions = self.list_versions(ticker, timeframe)
        if not versions:
            return 0

        return max(version.get("version", 0) for version in versions)

    def delete_version(self, ticker: str, timeframe: str, version: int) -> bool:
        """
        Delete a specific version of a model.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            version: Version to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        version_dir = os.path.join(self.base_dir, ticker, timeframe, f"v{version}")
        if not os.path.exists(version_dir):
            logger.warning(f"Version {version} not found for {ticker}/{timeframe}")
            return False

        try:
            shutil.rmtree(version_dir)
            logger.info(f"Deleted model version {version} for {ticker}/{timeframe}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False
