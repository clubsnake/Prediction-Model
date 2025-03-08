import concurrent.futures
import datetime
import json
import logging
import multiprocessing
import os
import pickle
import threading
import uuid
from functools import partial

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger  # type: ignore
from tensorflow.keras.models import Model, load_model, model_from_json  # type: ignore

# Import optimized utilities
from src.data.sequence_utils import batch_sequence_creation
from src.utils.memory_utils import adaptive_memory_clean

# Add training optimizer import
from src.utils.training_optimizer import get_training_optimizer

# Define register_shutdown_callback if it's not available
def register_shutdown_callback(callback_func):
    """
    Register a function to be called during application shutdown.
    This is a fallback implementation if the actual function is not available.

    Args:
        callback_func: Function to call at shutdown
    """
    import atexit

    atexit.register(callback_func)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IncrementalLearning")


class ModelLoadError(Exception):
    """Exception raised when a model cannot be loaded."""
    pass


class ModelRegistry:
    """
    Registry for managing models with versioning, metadata tracking, and serialization.

    Supports:
    - Multiple model types (TensorFlow, scikit-learn, ensemble)
    - Model versioning and lineage tracking
    - Performance metrics storage
    - Metadata and hyperparameter tracking
    - Model serialization and deserialization
    - A/B testing and model comparison
    """

    def __init__(
        self,
        registry_dir="model_registry",
        create_if_missing=True,
        max_models_per_type=10,
    ):
        """
        Initialize the model registry.

        Args:
            registry_dir: Base directory for the model registry
            create_if_missing: Whether to create the directory if it doesn't exist
            max_models_per_type: Maximum number of models to keep per type
        """
        self.registry_dir = registry_dir
        self.max_models_per_type = max_models_per_type
        self.logger = logging.getLogger("ModelRegistry")

        # Create registry directory if needed
        if create_if_missing and not os.path.exists(registry_dir):
            os.makedirs(registry_dir)
            self.logger.info(f"Created model registry directory: {registry_dir}")

        # Initialize the registry
        self.model_index = {}
        self.load_index()

    def load_index(self):
        """
        Load the model index from the registry.
        """
        index_path = os.path.join(self.registry_dir, "model_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    self.model_index = json.load(f)
                self.logger.info(
                    f"Loaded model index with {len(self.model_index)} entries"
                )
            except Exception as e:
                self.logger.warning(f"Could not load model index: {e}")
                self.model_index = {}
        else:
            self.logger.info("Model index not found, creating new index")
            self.model_index = {}

    def save_index(self):
        """
        Save the model index to the registry.
        """
        index_path = os.path.join(self.registry_dir, "model_index.json")
        try:
            with open(index_path, "w") as f:
                json.dump(self.model_index, f, indent=2)
            self.logger.info(f"Saved model index with {len(self.model_index)} entries")
        except Exception as e:
            self.logger.error(f"Could not save model index: {e}")

    def register_model(
        self,
        model,
        model_type="tensorflow",
        ticker=None,
        timeframe=None,
        metrics=None,
        hyperparams=None,
        parent_id=None,
        tags=None,
        description=None,
        save_format="h5",
    ):
        """
        Register a model in the registry.

        Args:
            model: The model to register
            model_type: Type of model ('tensorflow', 'sklearn', 'xgboost', 'ensemble')
            ticker: Ticker symbol the model was trained on
            timeframe: Timeframe the model was trained on
            metrics: Dictionary of performance metrics
            hyperparams: Dictionary of hyperparameters
            parent_id: ID of parent model (for incremental learning)
            tags: List of tags for the model
            description: Description of the model
            save_format: Format to save TensorFlow models ('h5' or 'saved_model')

        Returns:
            model_id: The ID of the registered model
        """
        # Generate a model ID
        model_id = str(uuid.uuid4())[:8]

        # Create model directory
        model_dir = os.path.join(self.registry_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Create metadata
        timestamp = datetime.datetime.now().isoformat()

        metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "created_at": timestamp,
            "ticker": ticker,
            "timeframe": timeframe,
            "metrics": metrics or {},
            "hyperparams": hyperparams or {},
            "parent_id": parent_id,
            "tags": tags or [],
            "description": description or "",
            "status": "active",
        }

        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save the model
        model_path = self._save_model(model, model_type, model_dir, save_format)

        # Update the metadata with the path
        metadata["model_path"] = model_path

        # Update the index
        self.model_index[model_id] = metadata
        self.save_index()

        self.logger.info(f"Registered model {model_id} of type {model_type}")

        # Prune old models if needed
        self._prune_old_models(model_type)

        return model_id

    def load_model(self, model_id, with_metadata=True):
        """
        Load a model from the registry.

        Args:
            model_id: ID of the model to load
            with_metadata: Whether to return metadata along with the model

        Returns:
            model: The loaded model
            metadata: Metadata dictionary (if with_metadata=True)
            
        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        if model_id not in self.model_index:
            error_msg = f"Model {model_id} not found in registry"
            self.logger.error(error_msg)
            raise ModelLoadError(error_msg)

        metadata = self.model_index[model_id]
        model_type = metadata["model_type"]
        model_path = metadata.get("model_path")

        if not model_path or not os.path.exists(model_path):
            error_msg = f"Model path not found: {model_path}"
            self.logger.error(error_msg)
            raise ModelLoadError(error_msg)

        try:
            # Load the model based on its type
            if model_type == "tensorflow":
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            elif model_type == "sklearn":
                import joblib
                model = joblib.load(model_path)
            elif model_type == "xgboost":
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(model_path)
            elif model_type == "ensemble":
                import pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                error_msg = f"Unsupported model type: {model_type}"
                self.logger.error(error_msg)
                raise ModelLoadError(error_msg)

            self.logger.info(f"Loaded model {model_id} of type {model_type}")

            if with_metadata:
                return model, metadata
            else:
                return model

        except Exception as e:
            error_msg = f"Error loading model {model_id}: {str(e)}"
            self.logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def update_model_metrics(self, model_id, metrics):
        """
        Update the metrics for a model.

        Args:
            model_id: ID of the model to update
            metrics: Dictionary of metrics to update

        Returns:
            success: Whether the update was successful
        """
        if model_id not in self.model_index:
            self.logger.error(f"Model {model_id} not found in registry")
            return False

        # Update the metrics
        self.model_index[model_id]["metrics"].update(metrics)

        # Update the metadata file
        model_dir = os.path.join(self.registry_dir, model_id)
        metadata_path = os.path.join(model_dir, "metadata.json")

        try:
            with open(metadata_path, "w") as f:
                json.dump(self.model_index[model_id], f, indent=2)

            self.save_index()
            self.logger.info(f"Updated metrics for model {model_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating metrics for model {model_id}: {e}")
            return False

    def add_model_tag(self, model_id, tag):
        """
        Add a tag to a model.

        Args:
            model_id: ID of the model to tag
            tag: Tag to add

        Returns:
            success: Whether the update was successful
        """
        if model_id not in self.model_index:
            self.logger.error(f"Model {model_id} not found in registry")
            return False

        # Add the tag if it doesn't exist
        if tag not in self.model_index[model_id]["tags"]:
            self.model_index[model_id]["tags"].append(tag)

            # Update the metadata file
            model_dir = os.path.join(self.registry_dir, model_id)
            metadata_path = os.path.join(model_dir, "metadata.json")

            try:
                with open(metadata_path, "w") as f:
                    json.dump(self.model_index[model_id], f, indent=2)

                self.save_index()
                self.logger.info(f"Added tag '{tag}' to model {model_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error adding tag to model {model_id}: {e}")
                return False
        return True

    def update_model_status(self, model_id, status):
        """
        Update the status of a model.

        Args:
            model_id: ID of the model to update
            status: New status ('active', 'archived', 'deprecated')

        Returns:
            success: Whether the update was successful
        """
        if model_id not in self.model_index:
            self.logger.error(f"Model {model_id} not found in registry")
            return False

        # Update the status
        self.model_index[model_id]["status"] = status

        # Update the metadata file
        model_dir = os.path.join(self.registry_dir, model_id)
        metadata_path = os.path.join(model_dir, "metadata.json")

        try:
            with open(metadata_path, "w") as f:
                json.dump(self.model_index[model_id], f, indent=2)

            self.save_index()
            self.logger.info(f"Updated status for model {model_id} to '{status}'")
            return True
        except Exception as e:
            self.logger.error(f"Error updating status for model {model_id}: {e}")
            return False

    def delete_model(self, model_id):
        """
        Delete a model from the registry.

        Args:
            model_id: ID of the model to delete

        Returns:
            success: Whether the deletion was successful
        """
        if model_id not in self.model_index:
            self.logger.error(f"Model {model_id} not found in registry")
            return False

        # Get the model directory
        model_dir = os.path.join(self.registry_dir, model_id)

        try:
            # Remove the directory and all contents
            import shutil

            shutil.rmtree(model_dir)

            # Remove from the index
            del self.model_index[model_id]
            self.save_index()

            self.logger.info(f"Deleted model {model_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting model {model_id}: {e}")
            return False

    def get_models_by_tags(self, tags, require_all=False):
        """
        Get models that have the specified tags.

        Args:
            tags: List of tags to search for
            require_all: Whether all tags must be present (AND) or any tag (OR)

        Returns:
            models: List of model IDs that match the criteria
        """
        if not tags:
            return []

        matching_models = []

        for model_id, metadata in self.model_index.items():
            model_tags = metadata.get("tags", [])

            if require_all:
                # All tags must be present
                if all(tag in model_tags for tag in tags):
                    matching_models.append(model_id)
            else:
                # Any tag can be present
                if any(tag in model_tags for tag in tags):
                    matching_models.append(model_id)

        return matching_models

    def get_best_model(
        self, metric="rmse", ascending=True, filter_tags=None, ticker=None
    ):
        """
        Get the best model based on a metric.

        Args:
            metric: Metric to use for comparison
            ascending: Whether lower is better (True) or higher is better (False)
            filter_tags: List of tags to filter models by
            ticker: Ticker to filter models by

        Returns:
            model_id: ID of the best model
        """
        # Filter models
        filtered_models = {}

        for model_id, metadata in self.model_index.items():
            # Skip inactive models
            if metadata.get("status") != "active":
                continue

            # Filter by tags if specified
            if filter_tags:
                model_tags = metadata.get("tags", [])
                if not all(tag in model_tags for tag in filter_tags):
                    continue

            # Filter by ticker if specified
            if ticker and metadata.get("ticker") != ticker:
                continue

            # Check if the model has the metric
            metrics = metadata.get("metrics", {})
            if metric not in metrics:
                continue

            # Add to filtered models
            filtered_models[model_id] = metrics[metric]

        if not filtered_models:
            self.logger.warning(
                f"No models match the criteria (metric={metric}, tags={filter_tags}, ticker={ticker})"
            )
            return None

        # Find the best model
        if ascending:
            best_model_id = min(filtered_models, key=filtered_models.get)
        else:
            best_model_id = max(filtered_models, key=filtered_models.get)

        self.logger.info(
            f"Best model for metric {metric}: {best_model_id} with value {filtered_models[best_model_id]}"
        )
        return best_model_id

    def get_model_lineage(self, model_id):
        """
        Get the lineage of a model.

        Args:
            model_id: ID of the model to get lineage for

        Returns:
            lineage: List of model IDs in the lineage, from earliest to latest
        """
        if model_id not in self.model_index:
            self.logger.error(f"Model {model_id} not found in registry")
            return []

        lineage = []
        current_id = model_id

        # Traverse backwards to find all ancestors
        while current_id:
            lineage.append(current_id)
            metadata = self.model_index.get(current_id, {})
            current_id = metadata.get("parent_id")

        # Reverse to get from earliest to latest
        lineage.reverse()

        return lineage

    def get_model_children(self, model_id):
        """
        Get the direct children of a model.

        Args:
            model_id: ID of the model to get children for

        Returns:
            children: List of model IDs that are direct children
        """
        if model_id not in self.model_index:
            self.logger.error(f"Model {model_id} not found in registry")
            return []

        children = []

        for id, metadata in self.model_index.items():
            if metadata.get("parent_id") == model_id:
                children.append(id)

        return children

    def compare_models(self, model_ids, metrics=None):
        """
        Compare multiple models based on various metrics.

        Args:
            model_ids: List of model IDs to compare
            metrics: List of metrics to compare (None for all available)

        Returns:
            comparison: DataFrame with comparison results
        """
        if not model_ids:
            return None

        # Collect metadata for each model
        model_data = []

        for model_id in model_ids:
            if model_id not in self.model_index:
                self.logger.warning(f"Model {model_id} not found in registry")
                continue

            metadata = self.model_index[model_id]
            model_metrics = metadata.get("metrics", {})

            # Create a row for this model
            row = {
                "model_id": model_id,
                "model_type": metadata.get("model_type"),
                "ticker": metadata.get("ticker"),
                "timeframe": metadata.get("timeframe"),
                "created_at": metadata.get("created_at"),
                "status": metadata.get("status"),
                "tags": ", ".join(metadata.get("tags", [])),
            }

            # Add metrics
            if metrics:
                for metric in metrics:
                    row[metric] = model_metrics.get(metric, None)
            else:
                for metric, value in model_metrics.items():
                    row[metric] = value

            model_data.append(row)

        if not model_data:
            return None

        # Create DataFrame
        df = pd.DataFrame(model_data)

        return df

    def list_models(self, ticker=None, status="active", tag=None, limit=None):
        """
        List models in the registry with filtering options.

        Args:
            ticker: Filter by ticker
            status: Filter by status
            tag: Filter by tag
            limit: Maximum number of models to return

        Returns:
            models: List of model metadata dictionaries
        """
        models = []

        for model_id, metadata in self.model_index.items():
            # Apply filters
            if ticker and metadata.get("ticker") != ticker:
                continue

            if status and metadata.get("status") != status:
                continue

            if tag and tag not in metadata.get("tags", []):
                continue

            # Add to results
            models.append(metadata)

            # Apply limit
            if limit and len(models) >= limit:
                break

        return models

    def _save_model(self, model, model_type, model_dir, save_format="h5"):
        """
        Save a model to the registry.

        Args:
            model: The model to save
            model_type: Type of model
            model_dir: Directory to save the model in
            save_format: Format for TensorFlow models

        Returns:
            model_path: Path to the saved model
        """
        if model_type == "tensorflow":
            if save_format == "h5":
                model_path = os.path.join(model_dir, "model.h5")
                model.save(model_path, save_format="h5")
            else:
                model_path = os.path.join(model_dir, "saved_model")
                model.save(model_path, save_format="tf")
        elif model_type == "sklearn":
            model_path = os.path.join(model_dir, "model.joblib")
            joblib.dump(model, model_path)
        elif model_type == "xgboost":
            model_path = os.path.join(model_dir, "model.xgb")
            model.save_model(model_path)
        elif model_type == "ensemble":
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_path

    def _prune_old_models(self, model_type):
        """
        Prune old models to keep only a maximum number per type.
        """
        # Get all models of this type
        models = [
            (model_id, meta)
            for model_id, meta in self.model_index.items()
            if meta["model_type"] == model_type
        ]

        # If we're under the limit, no pruning needed
        if len(models) <= self.max_models_per_type:
            return

        # Sort by creation date (oldest first)
        models.sort(key=lambda x: x[1].get("created_at", ""))

        # Delete oldest models until we're back under the limit
        models_to_delete = models[: len(models) - self.max_models_per_type]

        for model_id, _ in models_to_delete:
            self.logger.info(f"Pruning old model {model_id} to stay under limit")
            self.delete_model(model_id)

    def visualize_metrics_history(
        self, model_ids=None, metric="rmse", save_path=None, show=True
    ):
        """
        Visualize the performance history of models.

        Args:
            model_ids: List of model IDs to include
            metric: Metric to visualize
            save_path: Path to save the visualization
            show: Whether to display the plot

        Returns:
            fig: Matplotlib figure
        """
        # If no model IDs provided, use all active models
        if not model_ids:
            model_ids = [
                mid
                for mid, meta in self.model_index.items()
                if meta.get("status") == "active"
            ]

        # Filter models that have this metric
        valid_models = []
        for mid in model_ids:
            if mid in self.model_index and metric in self.model_index[mid].get(
                "metrics", {}
            ):
                valid_models.append(mid)

        if not valid_models:
            self.logger.warning(f"No models found with metric: {metric}")
            return None

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Add each model as a point
        x = []  # Creation timestamps
        y = []  # Metric values
        labels = []  # Model IDs
        colors = []  # Colors for different model types

        for mid in valid_models:
            meta = self.model_index[mid]

            # Parse timestamp
            try:
                timestamp = datetime.datetime.fromisoformat(meta.get("created_at", ""))
            except:
                continue

            # Get metric value
            metric_value = meta.get("metrics", {}).get(metric)
            if metric_value is None:
                continue

            x.append(timestamp)
            y.append(metric_value)
            labels.append(mid)

            # Assign color based on model type
            if meta.get("model_type") == "tensorflow":
                colors.append("blue")
            elif meta.get("model_type") == "sklearn":
                colors.append("green")
            elif meta.get("model_type") == "xgboost":
                colors.append("red")
            elif meta.get("model_type") == "ensemble":
                colors.append("purple")
            else:
                colors.append("gray")

        # Plot the points
        scatter = ax.scatter(x, y, c=colors, alpha=0.7)

        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (x[i], y[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Add trendline if we have enough points
        if len(x) > 1:
            z = np.polyfit([t.timestamp() for t in x], y, 1)
            p = np.poly1d(z)

            # Convert datetime to timestamp for line plotting
            x_timestamps = [t.timestamp() for t in x]
            x_range = np.linspace(min(x_timestamps), max(x_timestamps), 100)
            ax.plot(
                [datetime.datetime.fromtimestamp(t) for t in x_range],
                p(x_range),
                "r--",
                alpha=0.7,
            )

        # Add legend for model types
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="TensorFlow",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=10,
                label="scikit-learn",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="XGBoost",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="purple",
                markersize=10,
                label="Ensemble",
            ),
        ]

        ax.legend(handles=legend_elements, loc="upper right")

        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        ax.set_title(f"Model Performance History: {metric}")

        # Add grid
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path)

        # Show if requested
        if show:
            plt.show()

        return fig


class IncrementalLearner:
    """
    Handles incremental learning, model persistence, and continuous updating.

    Supports:
    - Multiple model types (TensorFlow, ensemble, tree-based)
    - Warm start training from previous model weights
    - Schedule-based model updating
    - Performance monitoring and automatic retraining
    - Training data versioning
    - Asynchronous training
    """

    def __init__(
        self,
        model_registry=None,
        checkpoint_dir="checkpoints",
        data_cache_dir="data_cache",
        max_data_age_days=30,
    ):
        """
        Initialize the incremental learner.

        Args:
            model_registry: ModelRegistry instance or path to registry
            checkpoint_dir: Directory to save checkpoints
            data_cache_dir: Directory to cache training data
            max_data_age_days: Maximum age of data to keep in cache
        """
        # Set up model registry
        if model_registry is None:
            self.registry = ModelRegistry()
        elif isinstance(model_registry, str):
            self.registry = ModelRegistry(registry_dir=model_registry)
        else:
            self.registry = model_registry

        # Set up directories
        self.checkpoint_dir = checkpoint_dir
        self.data_cache_dir = data_cache_dir
        self.max_data_age_days = max_data_age_days

        # Create directories if needed
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(data_cache_dir, exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger("IncrementalLearner")

        # Training state
        self.training_threads = {}
        self.futures = {}
        self.data_versions = {}

        # Initialize training optimizer
        self.training_optimizer = get_training_optimizer()
        logger.info("Training optimizer initialized for IncrementalLearner")

    def cache_training_data(self, ticker, timeframe, df, version=None):
        """
        Cache training data for later use.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            df: DataFrame with training data
            version: Version string (default: current timestamp)

        Returns:
            version: Version of the cached data
        """
        # Generate version if not provided
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create cache key and directory
        cache_key = f"{ticker}_{timeframe}"
        ticker_dir = os.path.join(self.data_cache_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        # Save the data
        data_path = os.path.join(ticker_dir, f"{timeframe}_v{version}.pkl")
        df.to_pickle(data_path)

        # Update data version tracking
        if cache_key not in self.data_versions:
            self.data_versions[cache_key] = []

        self.data_versions[cache_key].append(version)

        # Prune old versions
        self._prune_old_data(ticker, timeframe)

        self.logger.info(
            f"Cached training data for {ticker} ({timeframe}) version {version}"
        )
        return version

    def get_cached_data(self, ticker, timeframe, version=None):
        """
        Get cached training data.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            version: Version to retrieve (None for latest)

        Returns:
            df: DataFrame with training data or None if not found
        """
        cache_key = f"{ticker}_{timeframe}"
        ticker_dir = os.path.join(self.data_cache_dir, ticker)

        if not os.path.exists(ticker_dir):
            self.logger.warning(f"No cached data found for {ticker}")
            return None

        # Determine version to load
        if version is None:
            # Get the latest version
            if cache_key in self.data_versions and self.data_versions[cache_key]:
                version = self.data_versions[cache_key][-1]
            else:
                # Scan directory for versions
                versions = []
                for filename in os.listdir(ticker_dir):
                    if filename.startswith(f"{timeframe}_v") and filename.endswith(
                        ".pkl"
                    ):
                        versions.append(
                            filename.replace(f"{timeframe}_v", "").replace(".pkl", "")
                        )

                if versions:
                    versions.sort()
                    version = versions[-1]
                else:
                    self.logger.warning(
                        f"No cached data found for {ticker} ({timeframe})"
                    )
                    return None

        # Load the data
        data_path = os.path.join(ticker_dir, f"{timeframe}_v{version}.pkl")
        if not os.path.exists(data_path):
            self.logger.warning(f"Cached data not found: {data_path}")
            return None

        try:
            df = pd.read_pickle(data_path)
            self.logger.info(
                f"Loaded cached data for {ticker} ({timeframe}) version {version}"
            )
            return df
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            return None

    def train_incrementally(
        self,
        model,
        df,
        ticker,
        timeframe,
        previous_model_id=None,
        batch_size=None,
        epochs=10,
        validation_split=0.2,
        callbacks=None,
        verbose=1,
        save_best_only=True,
        model_type="tensorflow",
        hyperparams=None,
        return_metrics=True,
        target_col="Close",
        feature_cols=None,
        train_on_all=False,
        **kwargs,
    ):
        """
        Train a model incrementally from a previous state.

        Args:
            model: Model to train (or None to load from previous_model_id)
            df: DataFrame with training data
            ticker: Ticker symbol
            timeframe: Timeframe
            previous_model_id: ID of previous model to start from
            batch_size: Batch size for training
            epochs: Number of epochs to train
            validation_split: Fraction of data to use for validation
            callbacks: List of callbacks for training
            verbose: Verbosity level
            save_best_only: Whether to save only the best model
            model_type: Type of model
            hyperparams: Dictionary of hyperparameters
            return_metrics: Whether to return metrics after training
            target_col: Target column name
            feature_cols: List of feature columns
            train_on_all: Whether to train on all data (no validation)
            **kwargs: Additional arguments for model.fit

        Returns:
            model_id: ID of the trained model
            metrics: Dictionary of performance metrics (if return_metrics=True)
        """
        # Load previous model if provided
        if model is None and previous_model_id:
            model = self.registry.load_model(previous_model_id, with_metadata=False)
            if model is None:
                self.logger.error(f"Could not load previous model: {previous_model_id}")
                return None
            self.logger.info(f"Loaded previous model: {previous_model_id}")

        if model is None:
            self.logger.error("No model provided and no previous model ID")
            return None

        # Cache the training data
        self.cache_training_data(ticker, timeframe, df)

        # Extract features and target
        if feature_cols is None:
            # Exclude date and target columns
            feature_cols = [
                col
                for col in df.columns
                if col != target_col and col != "date" and col != "Date"
            ]

        # Clean memory before data processing
        adaptive_memory_clean("small")
        
        # Prepare data for training using optimized sequence creation
        X = df[feature_cols].values
        y = df[target_col].values

        if model_type == "tensorflow":
            # For TensorFlow/Keras models that expect sequential data
            if len(model.input_shape) > 2:  # Sequential data expected
                # Replace this block with optimized version
                X_reshaped, y_reshaped = batch_sequence_creation(
                    np.column_stack((y.reshape(-1, 1), X)), 
                    lookback=model.input_shape[1], 
                    horizon=model.output_shape[1]
                )
                X = X_reshaped
                y = y_reshaped

            # Set up checkpointing
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{ticker}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5",
            )

            # Create callbacks if not provided
            if callbacks is None:
                callbacks = []

            # Add checkpoint callback
            if save_best_only:
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                    verbose=1,
                )
                callbacks.append(checkpoint_callback)

            # Add early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            )
            callbacks.append(early_stopping)

            # Add CSV logger
            log_dir = os.path.join(self.checkpoint_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(
                log_dir,
                f"{ticker}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
            csv_logger = tf.keras.callbacks.CSVLogger(log_path)
            callbacks.append(csv_logger)

            # Determine data shape and reshape if needed
            if len(model.input_shape) > 2:  # Sequential data expected
                # Check if we need to reshape X
                if len(X.shape) < 3:
                    # Assume the model expects [samples, time_steps, features]
                    time_steps = model.input_shape[1]
                    n_features = X.shape[1]
                    X_reshaped = np.zeros(
                        (X.shape[0] - time_steps + 1, time_steps, n_features)
                    )

                    for i in range(X_reshaped.shape[0]):
                        X_reshaped[i] = X[i : i + time_steps, :]

                    X = X_reshaped
                    y = y[time_steps - 1 :]

            # Train the model
            history = model.fit(
                X,
                y,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0 if train_on_all else validation_split,
                callbacks=callbacks,
                verbose=verbose,
                **kwargs,
            )

            # Calculate metrics
            metrics = {}

            if return_metrics:
                if train_on_all:
                    # Use the last 20% for testing
                    split_idx = int(len(X) * 0.8)
                    X_test = X[split_idx:]
                    y_test = y[split_idx:]
                else:
                    # Use the validation set
                    split_idx = int(len(X) * (1 - validation_split))
                    X_test = X[split_idx:]
                    y_test = y[split_idx:]

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))
                metrics["rmse"] = float(rmse)

                # Calculate MAPE
                mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
                metrics["mape"] = float(mape)

                self.logger.info(f"Metrics - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        elif model_type in ["sklearn", "xgboost"]:
            # For scikit-learn and XGBoost models

            # Split the data
            split_idx = int(len(X) * (1 - validation_split))
            if train_on_all:
                X_train, y_train = X, y
                X_test, y_test = (
                    X[split_idx:],
                    y[split_idx:],
                )  # Still need test data for metrics
            else:
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

            # Train the model
            model.fit(X_train, y_train)

            # Calculate metrics
            metrics = {}

            if return_metrics:
                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                metrics["rmse"] = float(rmse)

                # Calculate MAPE
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                metrics["mape"] = float(mape)

                self.logger.info(f"Metrics - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        else:
            self.logger.error(f"Unsupported model type: {model_type}")
            return None

        # Register the model
        model_id = self.registry.register_model(
            model=model,
            model_type=model_type,
            ticker=ticker,
            timeframe=timeframe,
            metrics=metrics,
            hyperparams=hyperparams,
            parent_id=previous_model_id,
            tags=["incremental"],
        )

        self.logger.info(f"Registered incremental model: {model_id}")

        if return_metrics:
            return model_id, metrics
        else:
            return model_id

    def train_incrementally_async(self, *args, **kwargs):
        """
        Train a model incrementally in a separate thread.

        Args:
            Same as train_incrementally

        Returns:
            thread_id: ID of the training thread
        """
        # Generate a thread ID
        thread_id = str(uuid.uuid4())[:8]

        # Create a thread for training
        thread = threading.Thread(
            target=self._train_thread, args=(thread_id, args, kwargs)
        )
        thread.daemon = True

        # Store the thread
        self.training_threads[thread_id] = {
            "thread": thread,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "model_id": None,
            "metrics": None,
            "error": None,
        }

        # Start the thread
        thread.start()

        self.logger.info(f"Started async training thread: {thread_id}")
        return thread_id

    def train_incrementally_parallel(self, configs, max_workers=None):
        """
        Train multiple models in parallel.

        Args:
            configs: List of dictionaries with training configurations
            max_workers: Maximum number of parallel workers

        Returns:
            future_ids: Dictionary mapping future IDs to futures
        """
        # Use the training optimizer to group configurations efficiently
        try:
            # Group configurations for optimal resource utilization
            training_groups = self.training_optimizer.parallel_training_groups(configs)
            
            future_ids = {}
            
            # Process each group sequentially to avoid resource contention
            for group_idx, group in enumerate(training_groups):
                self.logger.info(f"Training group {group_idx+1}/{len(training_groups)} with {len(group)} models")
                
                # Create a thread pool for this group
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
                    group_futures = {}
                    
                    for config in group:
                        # Create a partial function with the configuration
                        train_fn = partial(self.train_incrementally, **config)
                        
                        # Submit the task
                        future = executor.submit(train_fn)
                        
                        # Generate a future ID
                        future_id = str(uuid.uuid4())[:8]
                        
                        # Store the future
                        self.futures[future_id] = {
                            "future": future,
                            "config": config,
                            "status": "pending",
                        }
                        
                        group_futures[future_id] = future
                        future_ids[future_id] = future
                
                # Wait for all futures in this group to complete before moving to next group
                for future_id, future in group_futures.items():
                    try:
                        future.result()  # This blocks until the future completes
                        self.futures[future_id]["status"] = "completed"
                    except Exception as e:
                        self.logger.error(f"Training error in future {future_id}: {str(e)}")
                        self.futures[future_id]["status"] = "error"
                        self.futures[future_id]["error"] = str(e)
                
                # Clean GPU memory between groups
                if self.training_optimizer.has_gpu:
                    self._clean_gpu_memory()
            
            return future_ids
            
        except Exception as e:
            self.logger.error(f"Error in parallel training: {str(e)}")
            
            # Fall back to original implementation
            # Use a thread pool with configured max_workers
            if max_workers is None:
                max_workers = self.training_optimizer.config.get("num_parallel_models", 
                                                             multiprocessing.cpu_count())
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_ids = {}
                
                for config in configs:
                    # Create a partial function with the configuration
                    train_fn = partial(self.train_incrementally, **config)
                    
                    # Submit the task
                    future = executor.submit(train_fn)
                    
                    # Generate a future ID
                    future_id = str(uuid.uuid4())[:8]
                    
                    # Store the future
                    self.futures[future_id] = {
                        "future": future,
                        "config": config,
                        "status": "pending",
                    }
                    
                    future_ids[future_id] = future
                
            return future_ids
    
    def _clean_gpu_memory(self):
        """Clean GPU memory between training runs"""
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            self.logger.debug("GPU memory cleaned")
        except Exception as e:
            self.logger.warning(f"Error cleaning GPU memory: {e}")

    def get_training_status(self, thread_id=None):
        """
        Get the status of a training thread.

        Args:
            thread_id: ID of the thread to check (None for all)

        Returns:
            status: Dictionary with thread status information
        """
        if thread_id is None:
            # Return status of all threads
            return {
                tid: {
                    "status": info["status"],
                    "model_id": info["model_id"],
                    "metrics": info["metrics"],
                    "error": info["error"],
                }
                for tid, info in self.training_threads.items()
            }
        elif thread_id in self.training_threads:
            # Return status of specific thread
            info = self.training_threads[thread_id]
            return {
                "status": info["status"],
                "model_id": info["model_id"],
                "metrics": info["metrics"],
                "error": info["error"],
            }
        else:
            self.logger.warning(f"Thread ID not found: {thread_id}")
            return None

    def get_future_status(self, future_id=None):
        """
        Get the status of a parallel training future.

        Args:
            future_id: ID of the future to check (None for all)

        Returns:
            status: Dictionary with future status information
        """
        if future_id is None:
            # Return status of all futures
            return {
                fid: {
                    "status": info["status"],
                    "result": info.get("result"),
                    "error": info.get("error"),
                }
                for fid, info in self.futures.items()
            }
        elif future_id in self.futures:
            # Return status of specific future
            info = self.futures[future_id]
            return {
                "status": info["status"],
                "result": info.get("result"),
                "error": info.get("error"),
            }
        else:
            self.logger.warning(f"Future ID not found: {future_id}")
            return None

    def wait_for_training(self, thread_id, timeout=None):
        """
        Wait for a training thread to complete.

        Args:
            thread_id: ID of the thread to wait for
            timeout: Maximum time to wait (None for no timeout)

        Returns:
            status: Thread status after waiting
        """
        if thread_id not in self.training_threads:
            self.logger.warning(f"Thread ID not found: {thread_id}")
            return None

        thread_info = self.training_threads[thread_id]
        thread = thread_info["thread"]

        # Wait for the thread to complete
        thread.join(timeout=timeout)

        # Update status if the thread completed
        if not thread.is_alive():
            thread_info["status"] = (
                "completed" if thread_info["error"] is None else "error"
            )

        return self.get_training_status(thread_id)

    def wait_for_all_futures(self, timeout=None):
        """
        Wait for all parallel training futures to complete.

        Args:
            timeout: Maximum time to wait (None for no timeout)

        Returns:
            statuses: Dictionary mapping future IDs to status information
        """
        futures = [
            info["future"]
            for info in self.futures.values()
            if info["status"] == "pending"
        ]

        if not futures:
            self.logger.info("No pending futures to wait for")
            return self.get_future_status()

        try:
            # Wait for futures to complete
            concurrent.futures.wait(futures, timeout=timeout)

            # Update statuses
            for future_id, info in self.futures.items():
                if info["status"] == "pending":
                    future = info["future"]

                    if future.done():
                        try:
                            info["result"] = future.result()
                            info["status"] = "completed"
                        except Exception as e:
                            info["error"] = str(e)
                            info["status"] = "error"

            return self.get_future_status()

        except Exception as e:
            self.logger.error(f"Error waiting for futures: {e}")
            return self.get_future_status()

    def update_model_on_schedule(
        self,
        ticker,
        timeframe,
        model_id,
        schedule="daily",
        fetch_data_fn=None,
        feature_engineering_fn=None,
        **kwargs,
    ):
        """
        Schedule automatic model updates.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            model_id: ID of the model to update
            schedule: Update schedule ("hourly", "daily", "weekly", or cron string)
            fetch_data_fn: Function to fetch new data
            feature_engineering_fn: Function to perform feature engineering
            **kwargs: Additional arguments for train_incrementally

        Returns:
            scheduler: Scheduler object (for cancellation)
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            self.logger.error(
                "APScheduler is required for scheduling. Install with: pip install apscheduler"
            )

            # Attempt to install the package automatically
            try:
                import subprocess

                self.logger.info("Attempting to install APScheduler automatically...")
                subprocess.check_call(["pip", "install", "apscheduler"])

                # Try importing again after installation
                from apscheduler.schedulers.background import BackgroundScheduler
                from apscheduler.triggers.cron import CronTrigger

                self.logger.info("APScheduler installed successfully.")
            except Exception as e:
                self.logger.error(f"Failed to automatically install APScheduler: {e}")
                self.logger.error("Please run 'pip install apscheduler' manually.")
                return None

        if fetch_data_fn is None:
            self.logger.error("fetch_data_fn is required for scheduling")
            return None

        # Set up the scheduler
        scheduler = BackgroundScheduler()

        # Define the update function
        def update_model():
            try:
                self.logger.info(f"Scheduled update for model {model_id}")

                # Fetch new data
                df = fetch_data_fn(ticker=ticker, timeframe=timeframe)

                # Apply feature engineering if provided
                if feature_engineering_fn:
                    df = feature_engineering_fn(df)

                # Train incrementally
                update_kwargs = kwargs.copy()
                update_kwargs.update(
                    {
                        "model": None,
                        "df": df,
                        "ticker": ticker,
                        "timeframe": timeframe,
                        "previous_model_id": model_id,
                    }
                )

                new_model_id, metrics = self.train_incrementally(**update_kwargs)

                # Log the update
                self.logger.info(f"Updated model {model_id} -> {new_model_id}")
                self.logger.info(f"Metrics: {metrics}")

                # Return the new model ID for reference
                return new_model_id

            except Exception as e:
                self.logger.error(f"Error in scheduled update: {e}")
                return None

        # Set up the schedule
        if schedule == "hourly":
            scheduler.add_job(update_model, "interval", hours=1)
        elif schedule == "daily":
            scheduler.add_job(update_model, "interval", days=1)
        elif schedule == "weekly":
            scheduler.add_job(update_model, "interval", weeks=1)
        else:
            # Assume it's a cron string
            scheduler.add_job(update_model, CronTrigger.from_crontab(schedule))

        # Start the scheduler
        scheduler.start()

        self.logger.info(
            f"Started scheduler for model {model_id} with schedule: {schedule}"
        )
        return scheduler

    def monitor_model_performance(
        self,
        model_id,
        ticker,
        timeframe,
        fetch_data_fn=None,
        feature_engineering_fn=None,
        metric="rmse",
        threshold=0.1,
        retrain=True,
        **retrain_kwargs,
    ):
        """
        Monitor model performance and retrain if it degrades.

        Args:
            model_id: ID of the model to monitor
            ticker: Ticker symbol
            timeframe: Timeframe
            fetch_data_fn: Function to fetch new data
            feature_engineering_fn: Function to perform feature engineering
            metric: Metric to monitor ("rmse", "mape")
            threshold: Relative threshold for performance degradation
            retrain: Whether to automatically retrain on degradation
            **retrain_kwargs: Additional arguments for train_incrementally

        Returns:
            metrics: Dictionary with the latest performance metrics
        """
        if fetch_data_fn is None:
            self.logger.error("fetch_data_fn is required for monitoring")
            return None

        try:
            # Load the model
            model, metadata = self.registry.load_model(model_id, with_metadata=True)
            if model is None:
                self.logger.error(f"Could not load model: {model_id}")
                return None

            # Get the baseline performance
            baseline = metadata.get("metrics", {}).get(metric)
            if baseline is None:
                self.logger.warning(f"No baseline {metric} found for model {model_id}")
                baseline = float("inf")

            # Fetch new data
            df = fetch_data_fn(ticker=ticker, timeframe=timeframe)

            # Apply feature engineering if provided
            if feature_engineering_fn:
                df = feature_engineering_fn(df)

            # Extract features and target
            feature_cols = [
                col
                for col in df.columns
                if col != "Close" and col != "date" and col != "Date"
            ]
            X = df[feature_cols].values
            y = df["Close"].values

            # Make predictions and calculate metrics
            model_type = metadata.get("model_type")

            if model_type == "tensorflow":
                # Handle reshaping for sequential models
                if len(model.input_shape) > 2:
                    time_steps = model.input_shape[1]
                    n_features = X.shape[1]
                    X_reshaped = np.zeros(
                        (X.shape[0] - time_steps + 1, time_steps, n_features)
                    )

                    for i in range(X_reshaped.shape[0]):
                        X_reshaped[i] = X[i : i + time_steps, :]

                    X = X_reshaped
                    y = y[time_steps - 1 :]

                # Make predictions
                y_pred = model.predict(X)

                # Calculate metrics
                rmse = np.sqrt(np.mean((y - y_pred.flatten()) ** 2))
                mape = np.mean(np.abs((y - y_pred.flatten()) / y)) * 100

            else:
                # For sklearn/XGBoost models
                y_pred = model.predict(X)

                # Calculate metrics
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                mape = np.mean(np.abs((y - y_pred) / y)) * 100

            metrics = {"rmse": float(rmse), "mape": float(mape)}

            # Check for performance degradation
            current = metrics.get(metric)
            relative_change = (current - baseline) / baseline

            if relative_change > threshold:
                self.logger.warning(
                    f"Performance degradation detected for model {model_id}"
                )
                self.logger.warning(
                    f"Baseline {metric}: {baseline}, Current: {current}"
                )
                self.logger.warning(
                    f"Relative change: {relative_change:.2f} (threshold: {threshold})"
                )

                # Retrain if requested
                if retrain:
                    self.logger.info(f"Retraining model {model_id}...")

                    # Set up training parameters
                    train_kwargs = retrain_kwargs.copy()
                    train_kwargs.update(
                        {
                            "model": None,
                            "df": df,
                            "ticker": ticker,
                            "timeframe": timeframe,
                            "previous_model_id": model_id,
                        }
                    )

                    # Train incrementally
                    new_model_id, new_metrics = self.train_incrementally(**train_kwargs)

                    self.logger.info(f"Retrained model {model_id} -> {new_model_id}")
                    self.logger.info(f"New metrics: {new_metrics}")

                    # Update the return metrics
                    metrics["retrained"] = True
                    metrics["new_model_id"] = new_model_id
                    metrics["new_metrics"] = new_metrics

            return metrics

        except Exception as e:
            self.logger.error(f"Error monitoring model performance: {e}")
            return None

    def _train_thread(self, thread_id, args, kwargs):
        """
        Thread function for asynchronous training.
        """
        info = self.training_threads[thread_id]
        info["start_time"] = datetime.datetime.now()
        info["status"] = "running"

        try:
            result = self.train_incrementally(*args, **kwargs)

            if isinstance(result, tuple) and len(result) == 2:
                model_id, metrics = result
                info["model_id"] = model_id
                info["metrics"] = metrics
            else:
                info["model_id"] = result

            info["status"] = "completed"

        except Exception as e:
            self.logger.error(f"Error in training thread {thread_id}: {e}")
            info["error"] = str(e)
            info["status"] = "error"

        finally:
            info["end_time"] = datetime.datetime.now()

    def _prune_old_data(self, ticker, timeframe):
        """
        Prune old data that exceeds the maximum age.
        """
        cache_key = f"{ticker}_{timeframe}"
        ticker_dir = os.path.join(self.data_cache_dir, ticker)

        if not os.path.exists(ticker_dir):
            return

        # Get all cached versions
        versions = []

        for filename in os.listdir(ticker_dir):
            if filename.startswith(f"{timeframe}_v") and filename.endswith(".pkl"):
                version = filename.replace(f"{timeframe}_v", "").replace(".pkl", "")
                versions.append(version)

        if not versions:
            return

        # Sort versions
        versions.sort()

        # Keep track of versions
        self.data_versions[cache_key] = versions

        # Get the cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(
            days=self.max_data_age_days
        )

        # Check each version to see if it's older than the cutoff
        for version in versions:
            try:
                # Parse the timestamp from the version
                version_date = datetime.datetime.strptime(
                    version.split("_")[0], "%Y%m%d"
                )

                if version_date < cutoff_date:
                    # Delete the file
                    data_path = os.path.join(ticker_dir, f"{timeframe}_v{version}.pkl")
                    os.remove(data_path)

                    # Remove from versions
                    self.data_versions[cache_key].remove(version)

                    self.logger.info(f"Pruned old data: {data_path}")
            except Exception as e:
                self.logger.warning(f"Error pruning old data: {e}")

        # Ensure we keep at least one version
        if not self.data_versions[cache_key] and versions:
            # Keep the most recent version
            keep_version = versions[-1]
            self.data_versions[cache_key] = [keep_version]

            self.logger.info(f"Keeping most recent version: {keep_version}")


# Example usage
def example_usage():
    # Initialize model registry
    registry = ModelRegistry("model_registry")

    # Initialize incremental learner
    learner = IncrementalLearner(
        model_registry=registry,
        checkpoint_dir="checkpoints",
        data_cache_dir="data_cache",
    )

    from src.features.features import feature_engineering

    from data.data import fetch_data

    # Fetch data
    ticker = "ETH-USD"
    timeframe = "1d"
    df = fetch_data(ticker=ticker, start="2023-01-01", interval=timeframe)
    df = feature_engineering(df)

    # Create a simple model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                64, activation="relu", input_shape=(len(df.columns) - 2,)
            ),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model
    model_id = learner.train_incrementally(
        model=model,
        df=df,
        ticker=ticker,
        timeframe=timeframe,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
        verbose=1,
    )

    print(f"Trained model: {model_id}")

    # Load the model
    loaded_model = registry.load_model(model_id, with_metadata=False)

    # Make the model artificially worse for testing
    loaded_model.layers[-1].kernel.assign(loaded_model.layers[-1].kernel * 1.1)

    # Monitor performance
    metrics = learner.monitor_model_performance(
        model_id=model_id,
        ticker=ticker,
        timeframe=timeframe,
        fetch_data_fn=lambda ticker, timeframe: fetch_data(
            ticker=ticker, start="2023-01-01", interval=timeframe
        ),
        feature_engineering_fn=feature_engineering,
        metric="rmse",
        threshold=0.05,
        retrain=True,
    )

    print(f"Monitoring metrics: {metrics}")

    # Schedule updates
    scheduler = learner.update_model_on_schedule(
        ticker=ticker,
        timeframe=timeframe,
        model_id=model_id,
        schedule="0 0 * * *",  # Daily at midnight
        fetch_data_fn=lambda ticker, timeframe: fetch_data(
            ticker=ticker, start="2023-01-01", interval=timeframe
        ),
        feature_engineering_fn=feature_engineering,
    )

    print("Scheduled model updates")


if __name__ == "__main__":
    example_usage()
