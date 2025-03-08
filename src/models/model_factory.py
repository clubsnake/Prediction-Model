from typing import Any, Dict, List, Optional, Union
import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)

# Import centralized GPU memory management
try:
    from src.utils.gpu_memory_management import configure_gpu_memory, configure_mixed_precision
except ImportError:
    logger.warning("GPU memory management not found, performance optimizations will be disabled")

# Import TabNet module
try:
    from tabnet_model import TabNetPricePredictor
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    print("Warning: TabNet module not available. TabNet models will not be supported.")


class ModelFactory:
    """
    Centralized factory for creating different prediction models.
    Provides a single interface to create any supported model type.
    """

    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any] = None) -> Any:
        """
        Creates and returns a model of the specified type with given parameters.
        
        Args:
            model_type: Type of model to create
            params: Dictionary of parameters for the model
            
        Returns:
            Initialized model of the requested type
        """
        # Normalize model type string
        model_type = model_type.lower()
        
        # Initialize default params if not provided
        if params is None:
            params = {}
            
        # Configure GPU before model creation
        try:
            configure_mixed_precision(params.get("use_mixed_precision"))
        except:
            pass
        
        # Create requested model
        if model_type == "tabnet":
            return ModelFactory._create_tabnet_model(params)
        elif model_type == "lstm":
            return ModelFactory._create_lstm_model(params)
        elif model_type == "rnn":
            return ModelFactory._create_rnn_model(params)
        elif model_type == "cnn":
            return ModelFactory._create_cnn_model(params)
        elif model_type == "ltc":
            return ModelFactory._create_ltc_model(params)
        elif model_type == "tft":
            return ModelFactory._create_tft_model(params)
        elif model_type == "nbeats":
            return ModelFactory._create_nbeats_model(params)
        elif model_type == "random_forest":
            return ModelFactory._create_random_forest_model(params)
        elif model_type == "xgboost":
            return ModelFactory._create_xgboost_model(params)
        elif model_type == "ensemble":
            return ModelFactory._create_ensemble_model(params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def _create_tabnet_model(params: Dict[str, Any]) -> Any:
        """Create a TabNet model"""
        try:
            from src.models.tabnet_model import TabNetPricePredictor
            
            # Extract parameters with defaults
            n_d = params.get("n_d", 64)
            n_a = params.get("n_a", 64)
            n_steps = params.get("n_steps", 3)
            gamma = params.get("gamma", 1.5)
            lambda_sparse = params.get("lambda_sparse", 0.001)
            learning_rate = params.get("learning_rate", 0.001)
            
            # Create the model
            model = TabNetPricePredictor(
                n_d=n_d,
                n_a=n_a,
                n_steps=n_steps,
                gamma=gamma,
                lambda_sparse=lambda_sparse,
                optimizer_fn=lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
                optimizer_params={"lr": learning_rate}
            )
            
            return model
        except ImportError:
            logger.error("TabNet module not available")
            raise ImportError("TabNet module is not available")
    
    @staticmethod
    def _create_lstm_model(params: Dict[str, Any]) -> tf.keras.Model:
        """Create an LSTM model using build_model_by_type"""
        try:
            from src.models.model import build_model_by_type
            
            return build_model_by_type(
                model_type="lstm",
                num_features=params.get("num_features", 1),
                horizon=params.get("horizon", 1),
                learning_rate=params.get("learning_rate", 0.001),
                dropout_rate=params.get("dropout_rate", 0.2),
                loss_function=params.get("loss_function", "mse"),
                lookback=params.get("lookback", 30),
                architecture_params=params.get("architecture_params", None)
            )
        except ImportError:
            logger.error("Could not import build_model_by_type")
            raise ImportError("Model building module not available")
    
    @staticmethod
    def _create_rnn_model(params: Dict[str, Any]) -> tf.keras.Model:
        """Create a SimpleRNN model"""
        try:
            from src.models.model import build_model_by_type
            
            return build_model_by_type(
                model_type="rnn",
                num_features=params.get("num_features", 1),
                horizon=params.get("horizon", 1),
                learning_rate=params.get("learning_rate", 0.001),
                dropout_rate=params.get("dropout_rate", 0.2),
                loss_function=params.get("loss_function", "mse"),
                lookback=params.get("lookback", 30),
                architecture_params=params.get("architecture_params", None)
            )
        except ImportError:
            logger.error("Could not import build_model_by_type")
            raise ImportError("Model building module not available")
    
    @staticmethod
    def _create_cnn_model(params: Dict[str, Any]) -> Any:
        """Create a CNN model"""
        try:
            from src.models.cnn_model import CNNPricePredictor
            
            # Extract parameters with defaults
            lookback = params.get("lookback", 30)
            num_features = params.get("num_features", 1)
            horizon = params.get("horizon", 1)
            num_filters = params.get("num_filters", 64)
            kernel_size = params.get("kernel_size", 3)
            learning_rate = params.get("learning_rate", 0.001)
            
            # Create model
            model = CNNPricePredictor(
                lookback=lookback,
                num_features=num_features,
                horizon=horizon,
                num_filters=num_filters,
                kernel_size=kernel_size,
                learning_rate=learning_rate
            )
            
            return model
        except ImportError:
            logger.error("CNN model module not available")
            raise ImportError("CNN model module not available")
    
    @staticmethod
    def _create_ltc_model(params: Dict[str, Any]) -> tf.keras.Model:
        """Create a Liquid Time Constant (LTC) model"""
        try:
            from src.models.ltc_model import build_ltc_model
            
            return build_ltc_model(
                num_features=params.get("num_features", 1),
                horizon=params.get("horizon", 1),
                learning_rate=params.get("learning_rate", 0.001),
                loss_function=params.get("loss_function", "mse"),
                lookback=params.get("lookback", 30),
                units=params.get("units", 64),
                num_layers=params.get("num_layers", 1),
                use_attention=params.get("use_attention", False),
                dropout_rate=params.get("dropout_rate", 0.1),
                recurrent_dropout_rate=params.get("recurrent_dropout_rate", 0.0),
                timescale_min=params.get("timescale_min", 0.1),
                timescale_max=params.get("timescale_max", 10.0)
            )
        except ImportError:
            logger.error("LTC model module not available")
            raise ImportError("LTC model module not available")
    
    @staticmethod
    def _create_tft_model(params: Dict[str, Any]) -> tf.keras.Model:
        """Create a Temporal Fusion Transformer model"""
        try:
            from src.models.temporal_fusion_transformer import build_tft_model
            
            return build_tft_model(
                num_features=params.get("num_features", 1),
                horizon=params.get("horizon", 1),
                learning_rate=params.get("learning_rate", 0.001),
                dropout_rate=params.get("dropout_rate", 0.1),
                loss_function=params.get("loss_function", "mse"),
                lookback=params.get("lookback", 30),
                hidden_size=params.get("hidden_size", 64),
                lstm_units=params.get("lstm_units", 128),
                num_heads=params.get("num_heads", 4)
            )
        except ImportError:
            logger.error("TFT model module not available")
            raise ImportError("TFT model module not available")
    
    @staticmethod
    def _create_nbeats_model(params: Dict[str, Any]) -> tf.keras.Model:
        """Create an N-BEATS model"""
        try:
            from src.models.nbeats_model import build_nbeats_model
            
            return build_nbeats_model(
                lookback=params.get("lookback", 30),
                horizon=params.get("horizon", 1),
                num_features=params.get("num_features", 1),
                learning_rate=params.get("learning_rate", 0.001),
                stack_types=params.get("stack_types", ["trend", "seasonality"]),
                num_blocks=params.get("num_blocks", [3, 3]),
                num_layers=params.get("num_layers", [4, 4]),
                layer_width=params.get("layer_width", 256)
            )
        except ImportError:
            logger.error("N-BEATS model module not available")
            raise ImportError("N-BEATS model module not available")
    
    @staticmethod
    def _create_random_forest_model(params: Dict[str, Any]) -> Any:
        """Create a Random Forest model"""
        from sklearn.ensemble import RandomForestRegressor
        
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=params.get("random_state", 42)
        )
    
    @staticmethod
    def _create_xgboost_model(params: Dict[str, Any]) -> Any:
        """Create an XGBoost model"""
        try:
            import xgboost as xgb
            
            return xgb.XGBRegressor(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 6),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                random_state=params.get("random_state", 42),
                n_jobs=params.get("n_jobs", -1)
            )
        except ImportError:
            logger.error("XGBoost not available")
            raise ImportError("XGBoost module not available")
    
    @staticmethod
    def _create_ensemble_model(params: Dict[str, Any]) -> Any:
        """Create an ensemble of models"""
        try:
            from src.models.ensemble_weighting import AdvancedEnsembleWeighter
            
            # Create submodels
            submodel_params = params.get("submodels", {})
            models = {}
            
            for model_type, model_params in submodel_params.items():
                try:
                    models[model_type] = ModelFactory.create_model(model_type, model_params)
                except Exception as e:
                    logger.warning(f"Failed to create {model_type} for ensemble: {e}")
            
            # Create ensemble weighter
            weights = {model_type: 1.0/len(models) for model_type in models}
            ensemble = AdvancedEnsembleWeighter(
                base_weights=weights,
                adaptation_rate=params.get("adaptation_rate", 0.05)
            )
            
            # Return both models and weighter
            return {"models": models, "weighter": ensemble}
            
        except ImportError:
            logger.error("Ensemble module not available")
            raise ImportError("Ensemble module not available")

# Create an instance for convenience
model_factory = ModelFactory()


class BaseModel:
    """Base class for all prediction models"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        raise NotImplementedError("Subclasses must implement predict()")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance"""
        raise NotImplementedError("Subclasses must implement evaluate()")


class LinearRegressionModel(BaseModel):
    """Linear regression model implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.coefficients = None
        self.intercept = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a simple linear regression model"""
        # Simplified implementation for demonstration
        X_mean = X.mean(axis=0)
        y_mean = y.mean()

        numerator = np.sum((X - X_mean) * (y - y_mean)[:, np.newaxis], axis=0)
        denominator = np.sum((X - X_mean) ** 2, axis=0)

        self.coefficients = numerator / denominator
        self.intercept = y_mean - np.dot(X_mean, self.coefficients)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained coefficients"""
        return np.dot(X, self.coefficients) + self.intercept

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}


class NeuralNetworkModel(BaseModel):
    """Simple neural network model implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weights = []
        self.biases = []

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a simple neural network"""
        # Simplified implementation for demonstration
        input_size = X.shape[1]
        hidden_size = self.config.get("hidden_size", 10)
        output_size = 1 if len(y.shape) == 1 else y.shape[1]

        # Initialize weights and biases
        np.random.seed(self.config.get("random_seed", 42))
        self.weights = [
            np.random.randn(input_size, hidden_size) * 0.01,
            np.random.randn(hidden_size, output_size) * 0.01,
        ]
        self.biases = [np.zeros((1, hidden_size)), np.zeros((1, output_size))]

        # Actual training would implement backpropagation here
        print("Neural network training placeholder")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the neural network"""
        # Forward pass
        layer1 = np.maximum(0, np.dot(X, self.weights[0]) + self.biases[0])  # ReLU
        output = np.dot(layer1, self.weights[1]) + self.biases[1]
        return output

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}


class TabNetModel(BaseModel):
    """
    TabNet model implementation for the ensemble pipeline.
    Incorporates TabNetPricePredictor for deep learning on tabular data with
    explainability and feature selection capabilities.
    """
    
    def __init__(self, params=None):
        super().__init__()
        self.model = None
        self.params = params or {}
        self.feature_names = None
        self.model_type = "tabnet"
        
        # Default TabNet parameters if not specified
        self.default_params = {
            "n_d": 64,            # Width of decision prediction layer
            "n_a": 64,            # Width of attention layer
            "n_steps": 5,         # Number of steps in the architecture
            "gamma": 1.5,         # Scaling factor for attention
            "lambda_sparse": 0.001, # Sparsity regularization
            "max_epochs": 200,    # Maximum training epochs
            "patience": 15,       # Early stopping patience
            "batch_size": 1024,   # Batch size
            "virtual_batch_size": 128, # Ghost batch normalization size
            "momentum": 0.02,     # Momentum for batch normalization
            "device_name": "auto" # Device (auto, cpu, cuda)
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
            raise ImportError("TabNet module is not available. Cannot train TabNet model.")
        
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
            device_name=self.params["device_name"]
        )
        
        # Create validation split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Train the model
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return self
    
    def predict(self, X):
        """
        Generate predictions using the trained TabNet model.
        
        Args:
            X: Features for prediction
        
        Returns:
            Numpy array of predictions
        """
        if not self.model:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """
        Get feature importance from the TabNet model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.model:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        # Get feature importance
        importances = self.model.feature_importances()
        
        # Map to feature names if available
        if self.feature_names and len(importances) == len(self.feature_names):
            return dict(zip(self.feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
    
    def explain(self, X):
        """
        Generate model explanations for predictions.
        
        Args:
            X: Features to explain
        
        Returns:
            Dictionary with explanation data
        """
        if not self.model:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        return self.model.explain(X)
    
    def save(self, path):
        """Save the model to disk"""
        if not self.model:
            raise RuntimeError("No trained model to save.")
        
        self.model.save(path)
    
    def load(self, path):
        """Load a model from disk"""
        if not HAS_TABNET:
            raise ImportError("TabNet module is not available. Cannot load TabNet model.")
        
        self.model = TabNetPricePredictor.load(path)
        return self
