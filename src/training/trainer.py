import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.models.model_factory import BaseModel
from src.utils.utils import Visualization
from src.dashboard.visualization import Visualization


class ModelTrainer:
    """
    Unified class for model training, validation, and evaluation.
    Combines training and evaluation logic that might be spread across files.
    """

    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.history = {"train_loss": [], "val_loss": [], "train_time": []}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train the model and track metrics"""
        epochs = self.config.get("epochs", 10)

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

        Visualization.plot_training_history(metrics, save_path)

    def visualize_predictions(
        self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None
    ) -> None:
        """Visualize model predictions"""
        predictions = self.model.predict(X)
        Visualization.plot_predictions(y, predictions, save_path)
