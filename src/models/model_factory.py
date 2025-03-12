import logging
import os
import sys
from typing import Any, Dict

# Add project root to Python path for proper imports
current_file = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(models_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)

# Import centralized GPU memory management
try:
    from src.utils.gpu_memory_management import (
        configure_gpu_memory,
        configure_mixed_precision,
    )

    logger.info("Successfully imported GPU memory management")
except ImportError as e:
    logger.warning(
        f"GPU memory management not found, performance optimizations will be disabled: {e}"
    )
    logger.warning(f"Current sys.path: {sys.path}")

# Check for optional dependencies
# TabNet
try:
    from src.models.tabnet_model import TabNetPricePredictor

    HAS_TABNET = True
    logger.info("Successfully imported TabNet module")
except ImportError as e:
    HAS_TABNET = False
    logger.warning(
        f"TabNet module not available. TabNet models will not be supported: {e}"
    )
    # Log more details about why the import failed
    logger.warning(f"Current directory: {os.getcwd()}")
    logger.warning(f"__file__ is: {__file__}")

    # Check if file exists
    tabnet_path = os.path.join(src_dir, "models", "tabnet_model.py")
    if os.path.exists(tabnet_path):
        logger.info(f"TabNet file exists at {tabnet_path}")
    else:
        logger.warning(f"TabNet file not found at {tabnet_path}")


class ModelFactory:
    """
    Factory class to create and configure different types of models.
    Centralizes model creation and ensures consistent configuration.
    """

    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any] = None) -> Any:
        """
        Create and return a model instance based on the model type and parameters.

        Args:
            model_type: Type of model to create ('lstm', 'rnn', 'tft', etc.)
            params: Dictionary of parameters for model configuration

        Returns:
            A model instance

        Raises:
            ValueError: If model type is not supported
            ImportError: If required dependencies are missing
        """
        if params is None:
            params = {}

        logger.info(f"Creating model of type '{model_type}'")

        try:
            # Convert model type to lowercase for case-insensitive matching
            model_type = model_type.lower()

            # Map model types to their creation functions
            creation_map = {
                "linear": ModelFactory._create_linear_model,
                "lstm": ModelFactory._create_neural_network_model,
                "rnn": ModelFactory._create_neural_network_model,
                "tft": ModelFactory._create_tft_model,
                "nbeats": ModelFactory._create_nbeats_model,
                "tabnet": ModelFactory._create_tabnet_model,
                "random_forest": ModelFactory._create_random_forest_model,
                "xgboost": ModelFactory._create_xgboost_model,
                "ensemble": ModelFactory._create_ensemble_model,
                "ltc": ModelFactory._create_ltc_model,
                "cnn": ModelFactory._create_cnn_model,
            }

            # Check if model type is supported
            if model_type not in creation_map:
                raise ValueError(
                    f"Model type '{model_type}' not supported. Supported types are: {list(creation_map.keys())}"
                )

            # Create model using appropriate function
            model = creation_map[model_type](params)

            return model

        except KeyError:
            error_msg = f"Model type '{model_type}' not supported"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except ImportError as e:
            error_msg = f"Required dependencies for model type '{model_type}' not available: {e}"
            logger.error(error_msg)
            raise

        except Exception as e:
            logger.error(f"Failed to create model of type {model_type}: {e}")
            raise

    @staticmethod
    def _create_linear_model(params: Dict[str, Any]) -> Any:
        """Create a linear regression model"""

        return LinearRegressionModel(params)

    @staticmethod
    def _create_neural_network_model(params: Dict[str, Any]) -> Any:
        """Create a neural network model (LSTM, RNN)"""
        # Delegate to build_model_by_type in model.py
        from src.models.model import build_model_by_type

        # Extract required parameters
        model_type = params.get("model_type", "lstm")
        num_features = params.get("num_features", 1)
        horizon = params.get("horizon", 1)
        learning_rate = params.get("learning_rate", 0.001)
        dropout_rate = params.get("dropout_rate", 0.2)
        loss_function = params.get("loss_function", "mean_squared_error")
        lookback = params.get("lookback", 30)

        # Architecture parameters can be passed directly
        architecture_params = params.get("architecture_params", {})

        # Create the model
        model = build_model_by_type(
            model_type=model_type,
            num_features=num_features,
            horizon=horizon,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            loss_function=loss_function,
            lookback=lookback,
            architecture_params=architecture_params,
        )

        return model

    @staticmethod
    def _create_tft_model(params: Dict[str, Any]) -> Any:
        """Create a Temporal Fusion Transformer model"""
        # Delegate to build_model_by_type
        from src.models.model import build_model_by_type

        # Extract parameters with defaults
        num_features = params.get("num_features", 1)
        horizon = params.get("horizon", 1)
        learning_rate = params.get("learning_rate", 0.001)
        dropout_rate = params.get("dropout_rate", 0.2)
        loss_function = params.get("loss_function", "mean_squared_error")
        lookback = params.get("lookback", 30)
        hidden_size = params.get("hidden_size", 64)
        lstm_units = params.get("lstm_units", 128)
        num_heads = params.get("num_heads", 4)

        # Create architecture params dict
        architecture_params = {
            "hidden_size": hidden_size,
            "lstm_units": lstm_units,
            "num_heads": num_heads,
            "units_per_layer": params.get("units_per_layer", [64, 32]),
        }

        # Create the model
        model = build_model_by_type(
            model_type="tft",
            num_features=num_features,
            horizon=horizon,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            loss_function=loss_function,
            lookback=lookback,
            architecture_params=architecture_params,
        )

        return model

    @staticmethod
    def _create_nbeats_model(params: Dict[str, Any]) -> tf.keras.Model:
        """Create an N-BEATS model"""
        try:
            from src.models.nbeats_model import build_nbeats_model

            # Extract parameters with defaults
            lookback = params.get("lookback", 30)
            horizon = params.get("horizon", 1)
            num_features = params.get("num_features", 1)
            learning_rate = params.get("learning_rate", 0.001)
            stack_types = params.get("stack_types", ["trend", "seasonality"])
            num_blocks = params.get("num_blocks", [3, 3])
            num_layers = params.get("num_layers", [4, 4])
            layer_width = params.get("layer_width", 256)

            return build_nbeats_model(
                lookback=lookback,
                horizon=horizon,
                num_features=num_features,
                learning_rate=learning_rate,
                stack_types=stack_types,
                num_blocks=num_blocks,
                num_layers=num_layers,
                layer_width=layer_width,
            )
        except ImportError:
            logger.error("N-BEATS model module not available")
            raise ImportError("N-BEATS model module not available")

    @staticmethod
    def _create_tabnet_model(params: Dict[str, Any]) -> Any:
        """Create a TabNet model"""
        if not HAS_TABNET:
            logger.error("TabNet module not available")
            raise ImportError("TabNet module is not available")

        try:
            return TabNetModel(params)
        except Exception as e:
            logger.error(f"Failed to create TabNet model: {e}")
            raise

    @staticmethod
    def _create_random_forest_model(params: Dict[str, Any]) -> Any:
        """Create a Random Forest model"""
        try:
            from sklearn.ensemble import RandomForestRegressor

            # Extract parameters with defaults
            n_estimators = params.get("n_estimators", 100)
            max_depth = params.get("max_depth", None)
            min_samples_split = params.get("min_samples_split", 2)
            min_samples_leaf = params.get("min_samples_leaf", 1)
            random_state = params.get("random_state", 42)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )

            return model
        except ImportError:
            logger.error("scikit-learn module not available")
            raise ImportError("scikit-learn module not available")

    @staticmethod
    def _create_xgboost_model(params: Dict[str, Any]) -> Any:
        """Create an XGBoost model"""
        try:
            import xgboost as xgb

            # Extract parameters with defaults
            n_estimators = params.get("n_estimators", 100)
            learning_rate = params.get("learning_rate", 0.1)
            max_depth = params.get("max_depth", 6)
            subsample = params.get("subsample", 0.8)
            colsample_bytree = params.get("colsample_bytree", 0.8)

            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
            )

            return model
        except ImportError:
            logger.error("XGBoost module not available")
            raise ImportError("XGBoost module not available")

    @staticmethod
    def _create_ensemble_model(params: Dict[str, Any]) -> Any:
        """Create an ensemble of models"""
        try:
            from src.models.ensemble_utils import create_ensemble_model

            return create_ensemble_model(params)
        except ImportError:
            logger.error("Ensemble module not available")
            raise ImportError("Ensemble module not available")

    @staticmethod
    def _create_ltc_model(params: Dict[str, Any]) -> Any:
        """Create a Liquid Time-Constant (LTC) model"""
        try:
            from src.models.ltc_model import build_ltc_model

            # Extract parameters with defaults
            num_features = params.get("num_features", 1)
            horizon = params.get("horizon", 1)
            learning_rate = params.get("learning_rate", 0.001)
            loss_function = params.get("loss_function", "mean_squared_error")
            lookback = params.get("lookback", 30)
            units = params.get("units", 64)

            model = build_ltc_model(
                num_features=num_features,
                horizon=horizon,
                learning_rate=learning_rate,
                loss_function=loss_function,
                lookback=lookback,
                units=units,
                dropout_rate=params.get("dropout_rate", 0.2),
            )

            return model
        except ImportError:
            logger.error("LTC model module not available")
            raise ImportError("LTC model module not available")

    @staticmethod
    def _create_cnn_model(params: Dict[str, Any]) -> Any:
        """Create a CNN model"""
        try:
            from src.models.cnn_model import CNNPricePredictor

            # Extract parameters with defaults
            input_dim = params.get("num_features", 1)
            output_dim = params.get("horizon", 1)
            lookback = params.get("lookback", 30)

            model = CNNPricePredictor(
                input_dim=input_dim,
                output_dim=output_dim,
                num_conv_layers=params.get("num_conv_layers", 3),
                num_filters=params.get("num_filters", 64),
                kernel_size=params.get("kernel_size", 3),
                stride=params.get("stride", 1),
                dropout_rate=params.get("dropout_rate", 0.2),
                activation=params.get("activation", "relu"),
                use_adaptive_pooling=params.get("use_adaptive_pooling", True),
                fc_layers=params.get("fc_layers", [128, 64]),
                lookback=lookback,
                learning_rate=params.get("learning_rate", 0.001),
                batch_size=params.get("batch_size", 32),
                epochs=params.get("epochs", 10),
                early_stopping_patience=params.get("early_stopping_patience", 5),
            )

            return model
        except ImportError:
            logger.error("CNN model module not available")
            raise ImportError("CNN model module not available")


# Create an instance for convenience
model_factory = ModelFactory()


class BaseModel:
    """Base class for all prediction models"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() instead.")
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")
        raise NotImplementedError("Subclasses must implement predict()")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")
        raise NotImplementedError("Subclasses must implement evaluate()")

    def fit(self, X, y):
        """Fit the model to the data"""
        raise NotImplementedError("Subclasses must implement fit()")

    def save(self, path):
        """Save the model to disk"""
        raise NotImplementedError("Subclasses must implement save()")

    def load(self, path):
        """Load a model from disk"""
        raise NotImplementedError("Subclasses must implement load()")

    def get_feature_importance(self):
        """Get feature importances if applicable"""
        raise NotImplementedError(
            "Feature importance not implemented for this model type"
        )


class LinearRegressionModel(BaseModel):
    """Linear regression model implementation"""

    def __init__(self, params=None):
        super().__init__(params)
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

    def fit(self, X, y):
        """Fit the linear regression model"""
        if len(y.shape) > 1 and y.shape[1] > 1:
            # For multi-output, we'll predict only the first horizon
            y = y[:, 0]

        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        return self

    def predict(self, X):
        """Make predictions"""
        X_flat = X.reshape(X.shape[0], -1)
        preds = self.model.predict(X_flat)

        # Reshape to match expected output format if needed
        if len(preds.shape) == 1:
            # Assume we want same prediction for all horizons
            preds = np.tile(preds.reshape(-1, 1), (1, self.config.get("horizon", 1)))

        return preds

    def evaluate(self, X, y):
        """Evaluate model performance"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Get predictions
        preds = self.predict(X)

        # Calculate metrics
        if len(y.shape) > 1 and y.shape[1] > 1:
            # For multi-horizon targets, calculate metrics for first horizon
            mse = mean_squared_error(y[:, 0], preds[:, 0])
            mae = mean_absolute_error(y[:, 0], preds[:, 0])
        else:
            mse = mean_squared_error(y, preds)
            mae = mean_absolute_error(y, preds)

        return {"mse": mse, "rmse": np.sqrt(mse), "mae": mae}

    def save(self, path):
        """Save model to disk"""
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        """Load model from disk"""
        import joblib

        self.model = joblib.load(path)
        return self

    def get_feature_importance(self):
        """Get feature importances for linear model (coefficients)"""
        if not hasattr(self.model, "coef_"):
            return None
        return self.model.coef_


class NeuralNetworkModel(BaseModel):
    """Simple neural network model implementation"""

    def __init__(self, params=None):
        super().__init__(params)
        self.model = None
        self.model_type = params.get("model_type", "lstm") if params else "lstm"
        self.history = None

    def fit(self, X, y, **kwargs):
        """Build and train the model"""
        from src.models.model import build_model_by_type

        # Extract parameters from config
        num_features = X.shape[-1] if len(X.shape) > 1 else 1
        horizon = y.shape[-1] if len(y.shape) > 1 else 1
        lookback = X.shape[1] if len(X.shape) > 2 else self.config.get("lookback", 30)

        # Override with specific params if provided
        num_features = self.config.get("num_features", num_features)
        horizon = self.config.get("horizon", horizon)
        lookback = self.config.get("lookback", lookback)
        learning_rate = self.config.get("learning_rate", 0.001)
        dropout_rate = self.config.get("dropout_rate", 0.2)
        loss_function = self.config.get("loss_function", "mean_squared_error")
        architecture_params = self.config.get("architecture_params", {})

        # Build model if not already built
        if self.model is None:
            self.model = build_model_by_type(
                model_type=self.model_type,
                num_features=num_features,
                horizon=horizon,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                loss_function=loss_function,
                lookback=lookback,
                architecture_params=architecture_params,
            )

        # Configure training parameters
        epochs = kwargs.get("epochs", self.config.get("epochs", 10))
        batch_size = kwargs.get("batch_size", self.config.get("batch_size", 32))
        verbose = kwargs.get("verbose", self.config.get("verbose", 0))
        validation_split = kwargs.get(
            "validation_split", self.config.get("validation_split", 0.1)
        )

        # Train the model
        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["epochs", "batch_size", "verbose", "validation_split"]
            },
        )

        return self

    def predict(self, X):
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")

        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")

        # Use model's evaluate method
        results = self.model.evaluate(X, y, verbose=0)

        # Format results based on metrics used during compilation
        metrics = {}
        if isinstance(results, list) and hasattr(self.model, "metrics_names"):
            metrics = {
                name: value for name, value in zip(self.model.metrics_names, results)
            }
        else:
            metrics["loss"] = results

        # Add RMSE if MSE is available
        if "mse" in metrics:
            metrics["rmse"] = np.sqrt(metrics["mse"])

        return metrics

    def save(self, path):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path):
        """Load model from disk"""
        import tensorflow as tf

        self.model = tf.keras.models.load_model(path)
        return self


class TabNetModel(BaseModel):
    """
    TabNet model implementation for the ensemble pipeline.
    Incorporates TabNetPricePredictor for deep learning on tabular data with
    explainability and feature selection capabilities.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.model = None
        self.feature_names = None
        self.model_type = "tabnet"

        # Default TabNet parameters if not specified
        self.default_params = {
            "n_d": 64,  # Width of decision prediction layer
            "n_a": 64,  # Width of attention layer
            "n_steps": 5,  # Number of steps in the architecture
            "gamma": 1.5,  # Scaling factor for attention
            "lambda_sparse": 0.001,  # Sparsity regularization
            "max_epochs": 200,  # Maximum training epochs
            "patience": 15,  # Early stopping patience
            "batch_size": 1024,  # Batch size
            "virtual_batch_size": 128,  # Ghost batch normalization size
            "momentum": 0.02,  # Momentum for batch normalization
            "device_name": "auto",  # Device (auto, cpu, cuda)
        }

        # Update defaults with provided parameters
        for key, value in self.default_params.items():
            if key not in self.params:
                self.params[key] = value

    def fit(self, X, y):
        """
        Fit the TabNet model to training data.

        Args:
            X: Training features (DataFrame or numpy array)
            y: Target values

        Returns:
            self: Trained model instance
        """
        if not HAS_TABNET:
            raise ImportError(
                "TabNet module is not available. Cannot train TabNet model."
            )

        # Get feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        # Create and train TabNet model
        self.model = TabNetPricePredictor(
            n_d=self.params["n_d"],
            n_a=self.params["n_a"],
            n_steps=self.params["n_steps"],
            gamma=self.params["gamma"],
            lambda_sparse=self.params["lambda_sparse"],
            feature_names=self.feature_names,
            max_epochs=self.params["max_epochs"],
            patience=self.params["patience"],
            batch_size=self.params["batch_size"],
            virtual_batch_size=self.params["virtual_batch_size"],
            momentum=self.params["momentum"],
            device_name=self.params["device_name"],
        )

        # Create validation split
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        # Train the model
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return self

    def predict(self, X):
        """Make predictions using the trained TabNet model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature importances from TabNet model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.feature_importances()

    def explain(self, X):
        """Get explanations for predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        try:
            return self.model.explain(X)
        except (AttributeError, Exception) as e:
            logger.error(f"Explain method not available: {e}")
            return None

    def save(self, path):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path):
        """Load a model from disk"""
        if not HAS_TABNET:
            raise ImportError(
                "TabNet module is not available. Cannot load TabNet model."
            )

        self.model = TabNetPricePredictor.load(path)
        return self
