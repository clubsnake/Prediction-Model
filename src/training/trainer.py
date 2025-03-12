import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
import tensorflow as tf

# Add project root to sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.dashboard import visualization
from src.models.model_factory import BaseModel
from src.utils.training_optimizer import get_training_optimizer


class ModelTrainer:
    """
    Unified class for model training, validation, and evaluation.
    Combines training and evaluation logic that might be spread across files.
    """

    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.history = {"train_loss": [], "val_loss": [], "train_time": []}

        # Initialize training optimizer
        self.training_optimizer = get_training_optimizer()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train the model and track metrics"""
        epochs = self.config.get("epochs", 10)

        # Get optimized batch size from training optimizer
        model_type = self.config.get("model_type", "generic")
        model_complexity = self.config.get("model_complexity", "medium")
        optimized_config = self.training_optimizer.get_model_config(
            model_type, model_complexity
        )

        # Use batch size from config if specified, otherwise use optimized batch size
        batch_size = self.config.get("batch_size", optimized_config["batch_size"])

        for epoch in range(epochs):
            start_time = time.time()

            # Train the model for one epoch
            self.model.train(X_train, y_train)

            # Calculate training loss
            train_metrics = self.model.evaluate(X_train, y_train)
            self.history["train_loss"].append(train_metrics["mse"])

            # Calculate validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_metrics = self.model.evaluate(X_val, y_val)
                self.history["val_loss"].append(val_metrics["mse"])

            # Track training time
            self.history["train_time"].append(time.time() - start_time)

            # Print progress
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_metrics['mse']:.4f}, "
                f"Time: {self.history['train_time'][-1]:.2f}s"
            )

            if X_val is not None and y_val is not None:
                print(f"Validation Loss: {val_metrics['mse']:.4f}")

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data"""
        return self.model.evaluate(X_test, y_test)

    def visualize_training(self, save_path: Optional[str] = None) -> None:
        """Visualize training progress"""
        metrics = {"Training Loss": self.history["train_loss"]}

        if "val_loss" in self.history and self.history["val_loss"]:
            metrics["Validation Loss"] = self.history["val_loss"]

        visualization.plot_training_history(metrics, save_path)

    def visualize_predictions(
        self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None
    ) -> None:
        """Visualize model predictions"""
        predictions = self.model.predict(X)
        visualization.plot_predictions(y, predictions, save_path)

    def train_model(
        self, model_type, X_train, y_train, X_val=None, y_val=None, trial=None
    ):
        """
        Train a single model of the specified type.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            trial: Optuna trial object (optional)

        Returns:
            Trained model
        """
        # Check if model type is active
        if model_type not in st.session_state.get("ACTIVE_MODEL_TYPES", []):
            logging.warning(
                f"Model type {model_type} is not active. Skipping training."
            )
            return None

        # Get optimized configuration for this model type
        model_complexity = "medium"
        optimized_config = self.training_optimizer.get_model_config(
            model_type, model_complexity
        )

        # Apply optimized settings if not explicitly specified in params
        params = {}
        params["batch_size"] = params.get("batch_size", optimized_config["batch_size"])
        params["learning_rate"] = params.get(
            "learning_rate", optimized_config["learning_rate"]
        )

        # Enable mixed precision if available
        if optimized_config["mixed_precision"]:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logging.info(f"Mixed precision enabled for {model_type} model")
            except Exception as e:
                logging.warning(f"Could not enable mixed precision: {e}")

        # ...existing model training code...
