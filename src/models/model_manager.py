"""
Robust model serialization, versioning and management.

This module provides functions for saving and loading models with proper versioning,
metadata tracking, and integrity verification.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

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

        Raises:
            ModelSerializationError: If model saving fails
        """
        if model is None:
            raise ModelSerializationError("Cannot save None model")

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
            raise ModelSerializationError(f"Error saving model: {str(e)}")

        # Prepare parameters and metadata for this version
        params = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "hyperparams": hyperparams or {},
            "feature_importance": feature_importance or {},
            "training_history": training_history or {},
            "model_path": model_path,
        }
        params_path = os.path.join(version_dir, "params.json")
        success = safe_write_json(params_path, params)
        if not success:
            raise ModelSerializationError("Failed to write model parameters")

        # Update overall metadata with this new version
        metadata["versions"].append(params)
        if not safe_write_json(metadata_path, metadata):
            raise ModelSerializationError("Failed to update model metadata")

        # Optionally, update a symlink to the latest version
        latest_link = os.path.join(model_dir, "latest")
        try:
            if os.path.exists(latest_link) or os.path.islink(latest_link):
                os.remove(latest_link)
            os.symlink(version_dir, latest_link)
        except Exception as e:
            logger.warning(f"Could not update latest symlink: {str(e)}")

        logger.info(f"Model saved as version {new_version} at {model_path}")
        return str(new_version)

    def load_model(
        self, 
        ticker: str, 
        timeframe: str, 
        version: Union[int, str] = "latest", 
        with_metadata: bool = True
    ) -> Union[tf.keras.Model, tuple]:
        """
        Load a model with optional metadata.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            version: Version to load (int, str or "latest")
            with_metadata: Whether to return metadata with the model

        Returns:
            If with_metadata=True: Tuple of (model, metadata)
            If with_metadata=False: Model only

        Raises:
            ModelLoadError: If model loading fails
        """
        model_key = f"{ticker}_{timeframe}"
        model_dir = os.path.join(self.base_dir, model_key)

        if not os.path.exists(model_dir):
            raise ModelLoadError(f"No models found for {ticker} {timeframe}")

        # Handle "latest" version
        if version == "latest":
            latest_link = os.path.join(model_dir, "latest")
            if os.path.islink(latest_link) and os.path.exists(latest_link):
                version_dir = latest_link
            else:
                # Find highest version number if link doesn't exist
                metadata_path = os.path.join(model_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    raise ModelLoadError(f"Metadata not found for {ticker} {timeframe}")

                metadata = safe_read_json(metadata_path)
                if not metadata or "versions" not in metadata or not metadata["versions"]:
                    raise ModelLoadError(f"No versions found for {ticker} {timeframe}")

                # Get the latest version
                latest_version = max(v.get("version", 0) for v in metadata["versions"])
                version_dir = os.path.join(model_dir, f"v{latest_version}")
        else:
            # Convert to int if it's a string number
            if isinstance(version, str) and version.isdigit():
                version = int(version)
            
            version_dir = os.path.join(model_dir, f"v{version}")

        if not os.path.exists(version_dir):
            raise ModelLoadError(f"Version {version} not found for {ticker} {timeframe}")

        # Check for h5 or saved_model
        h5_path = os.path.join(version_dir, "model.h5")
        sm_path = os.path.join(version_dir, "saved_model")

        model = None
        try:
            if os.path.exists(h5_path):
                model = tf.keras.models.load_model(h5_path)
            elif os.path.exists(sm_path):
                model = tf.keras.models.load_model(sm_path)
            else:
                raise ModelLoadError(f"No model file found in {version_dir}")
        except Exception as e:
            raise ModelLoadError(f"Error loading model: {str(e)}")

        if model is None:
            raise ModelLoadError(f"Failed to load model from {version_dir}")

        # Load metadata if requested
        if with_metadata:
            params_path = os.path.join(version_dir, "params.json")
            if not os.path.exists(params_path):
                logger.warning(f"Metadata not found for {version_dir}")
                metadata = {}
            else:
                metadata = safe_read_json(params_path, default={})
            return model, metadata
        else:
            return model

    def list_versions(self, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        List all versions for a specific ticker and timeframe.
        
        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            
        Returns:
            List of version information dictionaries
            
        Raises:
            ModelLoadError: If no models found or metadata is corrupted
        """
        model_key = f"{ticker}_{timeframe}"
        model_dir = os.path.join(self.base_dir, model_key)
        
        if not os.path.exists(model_dir):
            raise ModelLoadError(f"No models found for {ticker} {timeframe}")
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ModelLoadError(f"Metadata file not found for {ticker} {timeframe}")
        
        metadata = safe_read_json(metadata_path)
        if not metadata or "versions" not in metadata:
            raise ModelLoadError(f"Invalid metadata format for {ticker} {timeframe}")
        
        return metadata["versions"]
    
    def get_latest_version(self, ticker: str, timeframe: str) -> int:
        """
        Get the latest version number for a specific ticker and timeframe.
        
        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            
        Returns:
            Latest version number as integer
            
        Raises:
            ModelLoadError: If no models found
        """
        versions = self.list_versions(ticker, timeframe)
        if not versions:
            raise ModelLoadError(f"No versions found for {ticker} {timeframe}")
        
        latest_version = max(v.get("version", 0) for v in versions)
        return latest_version
    
    def delete_version(self, ticker: str, timeframe: str, version: int) -> bool:
        """
        Delete a specific model version.
        
        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            version: Version number to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            ModelLoadError: If version doesn't exist
        """
        model_key = f"{ticker}_{timeframe}"
        model_dir = os.path.join(self.base_dir, model_key)
        version_dir = os.path.join(model_dir, f"v{version}")
        
        if not os.path.exists(version_dir):
            raise ModelLoadError(f"Version {version} not found for {ticker} {timeframe}")
        
        # Update metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata = safe_read_json(metadata_path)
        
        if metadata and "versions" in metadata:
            metadata["versions"] = [v for v in metadata["versions"] if v.get("version") != version]
            safe_write_json(metadata_path, metadata)
        
        # Delete the version directory
        try:
            import shutil
            shutil.rmtree(version_dir)
            
            # Update latest link if needed
            latest_link = os.path.join(model_dir, "latest")
            if os.path.islink(latest_link) and os.readlink(latest_link) == version_dir:
                os.remove(latest_link)
                # Point to new latest if available
                if metadata["versions"]:
                    latest_version = max(v.get("version", 0) for v in metadata["versions"])
                    new_latest_dir = os.path.join(model_dir, f"v{latest_version}")
                    if os.path.exists(new_latest_dir):
                        os.symlink(new_latest_dir, latest_link)
            
            logger.info(f"Deleted model version {version} for {ticker} {timeframe}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete version {version}: {str(e)}")
            return False