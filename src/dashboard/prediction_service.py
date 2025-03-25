"""
Provides interfaces for both programmatic and web-based prediction.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default paths - these will be overridden if config module is available
DEFAULT_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_DATA_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "data")
DEFAULT_MODELS_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "Models")
DEFAULT_BEST_PARAMS_FILE = os.path.join(
    DEFAULT_PROJECT_ROOT, "config", "best_params.yaml"
)

# Try to import config using a more robust approach
try:
    # Add project root to path for more reliable imports
    if DEFAULT_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, DEFAULT_PROJECT_ROOT)

    # First try to import from config module
    try:
        from config.config_loader import get_config

        config = get_config()
        PROJECT_ROOT = config.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
        DATA_DIR = config.get("DATA_DIR", DEFAULT_DATA_DIR)
        MODELS_DIR = config.get("MODELS_DIR", DEFAULT_MODELS_DIR)
        BEST_PARAMS_FILE = config.get("BEST_PARAMS_FILE", DEFAULT_BEST_PARAMS_FILE)
        logger.info("Successfully loaded configuration from config_loader")
    except ImportError:
        # Fall back to direct config import
        try:
            from config import config

            PROJECT_ROOT = getattr(config, "PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
            DATA_DIR = getattr(config, "DATA_DIR", DEFAULT_DATA_DIR)
            MODELS_DIR = getattr(config, "MODELS_DIR", DEFAULT_MODELS_DIR)
            BEST_PARAMS_FILE = getattr(
                config, "BEST_PARAMS_FILE", DEFAULT_BEST_PARAMS_FILE
            )
            logger.info("Successfully loaded configuration from config module")
        except ImportError:
            raise ImportError("Could not import configuration from any source")
except ImportError as e:
    logger.warning(f"Config not found, using default paths: {e}")
    PROJECT_ROOT = DEFAULT_PROJECT_ROOT
    DATA_DIR = DEFAULT_DATA_DIR
    MODELS_DIR = DEFAULT_MODELS_DIR
    BEST_PARAMS_FILE = DEFAULT_BEST_PARAMS_FILE

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Log the paths being used
logger.info(f"Using PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"Using DATA_DIR: {DATA_DIR}")
logger.info(f"Using MODELS_DIR: {MODELS_DIR}")

# Add proper import for PredictionMonitor
try:
    from src.dashboard.monitoring import PredictionMonitor

    HAS_MONITORING = True
except ImportError:
    logger.warning("PredictionMonitor not available")
    HAS_MONITORING = False


class PredictionService:
    """
    Provides a service to load models and data, and make predictions.
    Supports both file-based and in-memory models.
    """

    def __init__(
        self,
        data_dir: str = None,
        models_dir: str = None,
        model_file: str = None,
        model_instance=None,
        monitor=None,
        ticker=None,
        timeframe=None,
    ):
        """
        Initialize the prediction service.

        Args:
            data_dir: Directory containing data files
            models_dir: Directory containing model files
            model_file: Optional specific model file to load
            model_instance: Optional pre-loaded model instance
            monitor: Optional PredictionMonitor instance for tracking predictions
            ticker: Current ticker symbol being analyzed
            timeframe: Current timeframe being analyzed
        """
        self.data_dir = data_dir or DATA_DIR
        self.models_dir = models_dir or MODELS_DIR
        self.model_file = model_file
        self.model = model_instance
        self.monitor = monitor
        self.current_ticker = ticker
        self.current_timeframe = timeframe

        # If model_file is provided, load the model
        if self.model_file and not self.model:
            self._load_model()

    def _load_model(self, model_file: str = None) -> bool:
        """
        Load a model from a file.

        Args:
            model_file: Path to model file, relative to models_dir or absolute

        Returns:
            bool: True if model loaded successfully
        """
        model_file = model_file or self.model_file
        if not model_file:
            logger.error("No model file specified")
            return False

        try:
            # Check if path is absolute
            if os.path.isabs(model_file):
                model_path = model_file
            else:
                model_path = os.path.join(self.models_dir, model_file)

            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            self.model_file = model_file
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found at: {model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load data from a CSV file.

        Args:
            filename: Filename (relative to data_dir) or absolute path

        Returns:
            DataFrame or None if failed
        """
        try:
            # Check if path is absolute
            if os.path.isabs(filename):
                file_path = filename
            else:
                file_path = os.path.join(self.data_dir, filename)

            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
            elif file_path.endswith((".xls", ".xlsx")):
                data = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return None

            logger.info(f"Data loaded successfully from: {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        feature_names: List[str] = None,
    ) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Args:
            data: Input data for prediction (DataFrame, numpy array, or list)
            feature_names: Optional feature names when data is not a DataFrame

        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            logger.error("No model loaded. Call load_model() first.")
            return np.array([])

        try:
            # Preprocess data if needed
            processed_data = self._preprocess_input(data, feature_names)

            # Make prediction
            predictions = self.model.predict(processed_data)
            logger.info(
                f"Prediction successful: shape={predictions.shape if hasattr(predictions, 'shape') else 'scalar'}"
            )
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return np.array([])

    def predict_with_confidence(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        feature_names: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores using the loaded model.

        Args:
            data: Input data for prediction (DataFrame, numpy array, or list)
            feature_names: Optional feature names when data is not a DataFrame

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if self.model is None:
            logger.error("No model loaded. Call load_model() first.")
            return np.array([]), np.array([])

        try:
            # Preprocess data if needed
            processed_data = self._preprocess_input(data, feature_names)

            # Check if model has confidence prediction method
            if hasattr(self.model, 'predict_with_confidence'):
                # Ensemble models with confidence support
                predictions, confidence_scores, _ = self.model.predict_with_confidence(processed_data)
                logger.info(
                    f"Prediction with confidence successful: shape={predictions.shape if hasattr(predictions, 'shape') else 'scalar'}"
                )
                return predictions, confidence_scores
            else:
                # Regular prediction without confidence
                predictions = self.model.predict(processed_data)
                
                # Generate default confidence scores (lower for longer horizons)
                if hasattr(predictions, 'shape') and len(predictions.shape) > 0:
                    # Decaying confidence based on horizon
                    horizon = predictions.shape[0] if len(predictions.shape) == 1 else predictions.shape[1]
                    confidence_scores = np.ones_like(predictions) * 75 * np.exp(-0.1 * np.arange(horizon))
                else:
                    confidence_scores = np.array([75.0])  # Default confidence level
                    
                return predictions, confidence_scores
        except Exception as e:
            logger.error(f"Error during prediction with confidence: {e}")
            return np.array([]), np.array([])

    def predict_and_log(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        actual=None,
        feature_names: List[str] = None,
        horizon: int = 1,
    ) -> np.ndarray:
        """Make a prediction and log it if a monitor is available."""
        
        # Get predictions
        predictions = self.predict(data, feature_names)
        
        # Log prediction if we have a monitor
        if self.monitor is not None and HAS_MONITORING:
            try:
                # Get timestamp (current time by default)
                timestamp = datetime.now()
                
                # Extract prediction value (first value of the first prediction)
                pred_value = float(predictions[0][0]) if predictions.size > 0 else None
                
                # Extract actual value if provided
                actual_value = float(actual[0][0]) if actual is not None and hasattr(actual, 'size') and actual.size > 0 else actual
                
                # Log the prediction
                self.monitor.log_prediction(
                    ticker=self.ticker,
                    timeframe=self.timeframe,
                    predicted=pred_value,
                    actual=actual_value,
                    horizon=horizon,
                )
                logger.debug("Logged prediction - ticker: %s, value: %.4f", self.ticker, pred_value)
            except Exception as e:
                logger.error("Error logging prediction: %s", e)

        return predictions

    def _preprocess_input(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        feature_names: List[str] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Preprocess input data for prediction.

        Args:
            data: Input data (DataFrame, numpy array, or list)
            feature_names: Feature names when data is not a DataFrame

        Returns:
            Processed data ready for model prediction
        """
        # If data is a DataFrame, use it as is
        if isinstance(data, pd.DataFrame):
            return data

        # If data is a single sample (1D array or list), reshape to 2D
        if isinstance(data, list) or (
            isinstance(data, np.ndarray) and len(data.shape) == 1
        ):
            data_array = np.array(data).reshape(1, -1)
        else:
            data_array = np.array(data)

        # Convert to DataFrame if feature names are provided
        if feature_names:
            if len(feature_names) != data_array.shape[1]:
                logger.warning(
                    f"Number of feature names ({len(feature_names)}) does not match data shape ({data_array.shape[1]})"
                )
            # Use available feature names, pad with generic names if needed
            columns = feature_names[: data_array.shape[1]]
            if len(columns) < data_array.shape[1]:
                columns += [
                    f"feature_{i}" for i in range(len(columns), data_array.shape[1])
                ]
            return pd.DataFrame(data_array, columns=columns)

        return data_array

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dict with model metadata
        """
        if self.model is None:
            return {"status": "No model loaded"}

        info = {
            "model_type": type(self.model).__name__,
            "model_file": self.model_file,
        }

        # Add scikit-learn specific info
        if hasattr(self.model, "get_params"):
            info["params"] = self.model.get_params()

        # Add feature names if available
        if hasattr(self.model, "feature_names_in_"):
            info["feature_names"] = self.model.feature_names_in_.tolist()

        # Add keras/tensorflow specific info
        if hasattr(self.model, "summary"):
            info["is_neural_network"] = True
            try:
                # Get layer info
                layers = []
                for layer in self.model.layers:
                    layer_info = {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                        "output_shape": str(layer.output_shape),
                    }
                    if hasattr(layer, "units"):
                        layer_info["units"] = layer.units
                    layers.append(layer_info)
                info["layers"] = layers
            except Exception as e:
                info["layers_error"] = str(e)

        # Add ensemble info if this is an ensemble model
        if hasattr(self.model, "models") and hasattr(self.model, "weights"):
            info["is_ensemble"] = True
            info["ensemble_models"] = list(self.model.models.keys())
            info["ensemble_weights"] = {
                k: float(v) for k, v in self.model.weights.items()
            }

        return info

    def load_best_parameters(self, params_file: str = None) -> Dict:
        """
        Load best parameters from YAML file.

        Args:
            params_file: Optional path to parameters file

        Returns:
            Dict of parameters or empty dict if failed
        """
        params_file = params_file or BEST_PARAMS_FILE
        try:
            with open(params_file, "r") as f:
                params = yaml.safe_load(f)
            logger.info(f"Loaded parameters from {params_file}")
            return params
        except FileNotFoundError:
            logger.error(f"Parameters file not found at {params_file}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing parameters file: {e}")
            return {}

def generate_forecast(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int = 30,
    horizon: int = 30,
    with_confidence: bool = False,
    market_regime: bool = False
) -> Union[List[float], Tuple[List[float], List[float], Dict]]:
    """
    Generate forecast for future time periods with optional confidence scores.

    Args:
        df: DataFrame with historical data
        feature_cols: Features to use for prediction
        lookback: Number of past days to use for input
        horizon: Number of days to forecast
        with_confidence: Whether to include confidence scores
        market_regime: Whether to detect market regime for enhanced forecasting

    Returns:
        If with_confidence=False: List of forecasted values
        If with_confidence=True: Tuple of (forecast_values, confidence_scores, confidence_components)
    """
    if self.model is None:
        logger.error("No model loaded for forecasting")
        return ([], [], {}) if with_confidence else []

    try:
        # Apply market regime analysis if requested
        current_regime = "unknown"
        context_similarity = 0.5
        original_weights = None
        
        if market_regime and hasattr(self, 'market_regime_system'):
            # Initialize market regime system if needed
            if self.market_regime_system is None:
                from src.training.market_regime import MarketRegimeSystem
                regime_memory_file = os.path.join(DATA_DIR, "market_regime_memory.json")
                self.market_regime_system = MarketRegimeSystem(regime_memory_file=regime_memory_file)
                
            # Use market regime information if available for ensemble models
            if hasattr(self.model, 'weights') and hasattr(self.model, 'models'):
                original_weights = self.model.weights.copy()
                current_features = self.market_regime_system.extract_market_features(df)
                current_regime = self.market_regime_system.detect_regime(df)
                context_similarity = self.market_regime_system.calculate_context_similarity(current_features)
                
                # Get optimized weights for current regime
                optimized_weights = self.market_regime_system.get_optimal_weights(
                    regime=current_regime,
                    base_weights=original_weights,
                    context_similarity=context_similarity
                )
                
                # Update model weights for prediction
                self.model.weights = optimized_weights
                logger.info(f"Using regime-optimized weights for {current_regime} regime (similarity: {context_similarity:.2f})")

        # Check if model supports confidence prediction directly
        use_model_confidence = with_confidence and hasattr(self.model, 'predict_with_confidence')
        
        # Get the last 'lookback' days of data for input
        if len(df) < lookback:
            logger.error(f"DataFrame has fewer rows ({len(df)}) than lookback ({lookback})")
            return ([], [], {}) if with_confidence else []

        last_data = df.iloc[-lookback:].copy()

        # Create a scaler for feature normalization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(last_data[feature_cols])

        # Initialize arrays to store predictions and confidence
        future_prices = []
        confidence_scores = []
        confidence_components = {}
        current_data = last_data.copy()

        try:
            # Import create_sequences once
            from src.data.preprocessing import create_sequences

            # Generate input sequence for prediction
            X_input, _ = create_sequences(current_data, feature_cols, "Close", lookback, 1)

            # Make prediction with confidence if requested and supported
            if use_model_confidence:
                future_prices, confidence_scores, confidence_components = self.model.predict_with_confidence(X_input, verbose=0)
                
                # Process outputs to ensure proper format
                if isinstance(future_prices, np.ndarray):
                    future_prices = future_prices.flatten().tolist()[:horizon]
                if isinstance(confidence_scores, np.ndarray):
                    confidence_scores = confidence_scores.flatten().tolist()[:horizon]
                    
                # Pad or truncate to match horizon
                if len(future_prices) < horizon:
                    future_prices = future_prices + [future_prices[-1]] * (horizon - len(future_prices))
                if len(confidence_scores) < horizon:
                    confidence_scores = confidence_scores + [confidence_scores[-1]] * (horizon - len(confidence_scores))
                
                # Add market regime info to confidence components
                if market_regime and hasattr(self, 'market_regime_system'):
                    confidence_components['market_regime'] = current_regime
                    confidence_components['context_similarity'] = context_similarity
                    
            else:
                # Standard iterative prediction for each day in the horizon
                for i in range(horizon):
                    # Use model to predict
                    preds = self.model.predict(X_input, verbose=0)

                    # Get the predicted price (first value if multiple outputs)
                    if hasattr(preds, "shape") and len(preds.shape) > 1:
                        next_price = float(preds[0][0])
                    else:
                        next_price = float(preds[0])

                    future_prices.append(next_price)
                    
                    # Generate default confidence scores if needed
                    if with_confidence:
                        confidence_scores.append(max(30, 80 - i * 1.5))

                    # Update input data with the prediction for next iteration
                    next_row = current_data.iloc[-1:].copy()
                    if isinstance(next_row.index[0], pd.Timestamp):
                        next_row.index = [next_row.index[0] + pd.Timedelta(days=1)]
                    else:
                        next_row.index = [next_row.index[0] + 1]

                    next_row["Close"] = next_price
                    current_data = pd.concat([current_data.iloc[1:], next_row])

                    # Scale features for consistent input
                    current_scaled = current_data.copy()
                    current_scaled[feature_cols] = scaler.transform(
                        current_data[feature_cols]
                    )

                    # Create new sequence for next prediction
                    X_input, _ = create_sequences(
                        current_scaled, feature_cols, "Close", lookback, 1
                    )

        except ImportError:
            logger.warning("Using simplified forecast approach without create_sequences")
            # Simplified approach without create_sequences
            feature_data = last_data[feature_cols].values
            X_input = np.array([feature_data])

            for i in range(horizon):
                # Use model to predict
                preds = self.model.predict(X_input, verbose=0)

                # Get the predicted price
                if hasattr(preds, "shape") and len(preds.shape) > 1:
                    next_price = float(preds[0][0])
                else:
                    next_price = float(preds[0])

                future_prices.append(next_price)
                
                # Generate default confidence scores if needed
                if with_confidence:
                    confidence_scores.append(max(30, 80 - i * 1.5))

                # Update input data with the prediction
                next_row = current_data.iloc[-1:].copy()
                if isinstance(next_row.index[0], pd.Timestamp):
                    next_row.index = [next_row.index[0] + pd.Timedelta(days=1)]
                else:
                    next_row.index = [next_row.index[0] + 1]

                next_row["Close"] = next_price
                current_data = pd.concat([current_data.iloc[1:], next_row])

                # Get new feature data for next prediction
                feature_data = current_data[feature_cols].values
                X_input = np.array([feature_data])

        # Restore original weights if modified
        if original_weights is not None:
            self.model.weights = original_weights

        logger.info(f"Generated {len(future_prices)} day forecast" + 
                    (" with confidence scores" if with_confidence else ""))
                    
        # Return based on with_confidence parameter
        if with_confidence:
            return future_prices, confidence_scores, confidence_components
        else:
            return future_prices

    except Exception as e:
        logger.error(f"Error generating forecast: {e}", exc_info=True)
        return ([], [], {}) if with_confidence else []
    

def generate_forecast_with_confidence(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int = 30,
    horizon: int = 30,
) -> Tuple[List[float], List[float], Dict]:
    """
    Generate forecast with confidence scores (wrapper for unified function).
    
    Args:
        df: DataFrame with historical data
        feature_cols: Features to use for prediction
        lookback: Number of past days to use for input
        horizon: Number of days to forecast
        
    Returns:
        Tuple of (forecast_values, confidence_scores, confidence_components)
    """
    return self.generate_forecast(
        df=df, 
        feature_cols=feature_cols, 
        lookback=lookback, 
        horizon=horizon, 
        with_confidence=True,
        market_regime=True  # Enable market regime analysis
    )

def generate_predictions(
    model, df, feature_cols, lookback=30, horizon=30, return_sequences=False
):
    """
    Generate predictions using the provided model.
    Wrapper that uses the consolidated forecast generator.

    Args:
        model: Trained prediction model
        df: DataFrame with historical data
        feature_cols: Features to use for prediction
        lookback: Number of past days to use for input
        horizon: Number of days to forecast
        return_sequences: Whether to return the full sequence or just predictions

    Returns:
        Predicted values
    """
    try:
        # Use PredictionService for all prediction functionality
        service = PredictionService(model_instance=model)
        forecast = service.generate_forecast(df, feature_cols, lookback, horizon)

        if forecast and len(forecast) > 0:
            return forecast

        logger.warning(
            "Unable to generate predictions using PredictionService"
        )
        return []

    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        return []


def update_dashboard_forecast(model, df, feature_cols, ensemble_weights=None):
    """
    Unified function to update the dashboard forecast with confidence scores.
    
    Args:
        model: The trained model
        df: DataFrame with historical data
        feature_cols: Feature columns to use
        ensemble_weights: Optional weights for ensemble models
        
    Returns:
        List of forecast values or None if failed
    """
    try:
        from datetime import datetime
        import streamlit as st

        # Get context information from session state or directly passed
        ticker = st.session_state.get("selected_ticker")
        timeframe = st.session_state.get("selected_timeframe")

        # Get monitor from session state
        monitor = st.session_state.get("prediction_monitor")

        # Get parameters or use defaults
        lookback = st.session_state.get("lookback", 30)
        forecast_window = st.session_state.get("forecast_window", 30)

        # Create prediction service if needed
        service = PredictionService(
            model_instance=model, ticker=ticker, timeframe=timeframe, monitor=monitor
        )

        # Generate forecast with confidence
        forecast, confidence_scores, confidence_components = service.generate_forecast(
            df, feature_cols, lookback, forecast_window, with_confidence=True, market_regime=True
        )

        # Update session state with forecast results
        if forecast and len(forecast) > 0:
            st.session_state["future_forecast"] = forecast
            st.session_state["forecast_confidence"] = confidence_scores
            st.session_state["confidence_components"] = confidence_components
            st.session_state["last_forecast_update"] = datetime.now()
            
            # Store metadata for dashboard display
            if not hasattr(service, 'metadata'):
                service.metadata = {}
            
            service.metadata['current_regime'] = confidence_components.get('market_regime', 'unknown')
            service.metadata['context_similarity'] = confidence_components.get('context_similarity', 0.5)
            
            if hasattr(service, 'market_regime_system'):
                service.metadata['regime_stats'] = service.market_regime_system.get_regime_stats()
                
            st.session_state['metadata'] = service.metadata

            # Store ensemble weights if provided
            if ensemble_weights:
                st.session_state["ensemble_weights"] = ensemble_weights

            # Log this forecast to monitor
            if monitor is not None:
                try:
                    # Record first day forecast for monitoring
                    first_day_forecast = forecast[0] if forecast else None

                    # Get the most recent close price as reference
                    last_price = df["Close"].iloc[-1] if not df.empty else None

                    if first_day_forecast is not None:
                        monitor.log_prediction(
                            ticker=ticker,
                            timeframe=timeframe,
                            predicted=first_day_forecast,
                            actual=None,  # Will be updated later
                            horizon=1,
                        )
                        logger.info(f"Logged new forecast: {first_day_forecast:.2f}")
                except Exception as e:
                    logger.error(f"Error logging forecast: {e}")

            # Try to save prediction history
            try:
                from src.dashboard.dashboard.dashboard_visualization import (
                    save_best_prediction,
                )

                save_best_prediction(df, forecast)
            except ImportError:
                logger.debug("Could not import save_best_prediction")

            logger.info(f"Updated dashboard forecast with {len(forecast)} days and confidence scores")
            return forecast
        else:
            logger.warning("Generated forecast was empty")
            return None

    except Exception as e:
        logger.error(f"Error updating dashboard forecast: {e}", exc_info=True)
        return None


# ===== FLASK WEB INTERFACE =====


def create_flask_app(prediction_service=None):
    """
    Create a Flask app for serving predictions via web interface.

    Args:
        prediction_service: Optional PredictionService instance

    Returns:
        Flask app
    """
    from flask import Flask, jsonify, render_template, request

    app = Flask(__name__)

    # Use provided prediction service or create a new one
    if prediction_service is None:
        prediction_service = PredictionService()

    @app.route("/", methods=["GET", "POST"])
    def index():
        """Handle the home page and prediction logic."""
        prediction = None
        error = None

        if request.method == "POST":
            try:
                # Get input values from form
                feature_values = []
                feature_names = []

                for key, value in request.form.items():
                    if key.startswith("feature_"):
                        try:
                            feature_values.append(float(value))
                            feature_names.append(key)
                        except ValueError:
                            error = f"Invalid value for {key}: {value}"
                            break

                if not error:
                    # Make prediction
                    prediction = prediction_service.predict(
                        feature_values, feature_names
                    )
                    if len(prediction) > 0:
                        prediction = float(prediction[0])
                    else:
                        error = "No prediction returned from model"

            except Exception as e:
                error = f"Error during prediction: {str(e)}"

        # Get model info for display
        model_info = prediction_service.get_model_info()

        # Render template with result
        return render_template(
            "index.html", prediction=prediction, error=error, model_info=model_info
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """API endpoint for predictions."""
        try:
            # Get input data from JSON
            data = request.get_json()

            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Extract features and names
            features = data.get("features", [])
            feature_names = data.get("feature_names", None)

            # Make prediction
            prediction = prediction_service.predict(features, feature_names)

            # Return result
            return jsonify(
                {
                    "prediction": (
                        prediction.tolist()
                        if isinstance(prediction, np.ndarray)
                        else prediction
                    ),
                    "status": "success",
                }
            )

        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    @app.route("/api/model/info", methods=["GET"])
    def api_model_info():
        """API endpoint for model information."""
        return jsonify(prediction_service.get_model_info())

    @app.route("/api/forecast", methods=["POST"])
    def api_forecast():
        """API endpoint for generating forecasts."""
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Extract DataFrame from JSON
            df_data = data.get("data", [])
            if not df_data:
                return jsonify({"error": "No data provided in 'data' field"}), 400

            feature_cols = data.get("feature_cols", [])
            if not feature_cols:
                return jsonify({"error": "No feature columns specified"}), 400

            lookback = data.get("lookback", 30)
            horizon = data.get("horizon", 30)

            # Convert to DataFrame
            df = pd.DataFrame(df_data)

            # Generate forecast
            forecast = prediction_service.generate_forecast(
                df, feature_cols, lookback, horizon
            )

            return jsonify({"forecast": forecast, "status": "success"})

        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

    @app.route("/templates", methods=["GET"])
    def list_templates():
        """Show available prediction templates."""
        templates = [
            {
                "name": "Simple Prediction",
                "description": "Basic prediction with minimal features",
            },
            {
                "name": "Full Analysis",
                "description": "Comprehensive analysis with all features",
            },
        ]
        return render_template("templates.html", templates=templates)

    return app


# ===== COMMAND LINE INTERFACE =====


def run_cli():
    """
    Run the prediction service from the command line.

    Usage:
        python prediction_service.py [--model MODEL_FILE] [--data DATA_FILE] [--output OUTPUT_FILE]

    Returns:
        Exit code (0 for success)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Prediction Service CLI")
    parser.add_argument("--model", help="Path to model file")
    parser.add_argument("--data", help="Path to data file")
    parser.add_argument("--output", help="Path to output file")
    parser.add_argument("--params", help="Path to parameters file")
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Generate forecast instead of prediction",
    )
    args = parser.parse_args()

    # Initialize prediction service
    service = PredictionService(model_file=args.model)

    # Load parameters if specified
    if args.params:
        params = service.load_best_parameters(args.params)
        if params:
            logger.info(f"Loaded parameters: {params}")

    # Load model if not already loaded
    if service.model is None and not args.model:
        # Try to get model file from parameters
        model_file = service.load_best_parameters().get(
            "model_file", "trained_model.pkl"
        )
        service._load_model(model_file)

    # Check if model is loaded
    if service.model is None:
        logger.error("No model loaded. Exiting.")
        return 1

    # Load data
    data_file = args.data
    if not data_file:
        # Try to get data file from parameters
        data_file = service.load_best_parameters().get("data_file", "data.csv")

    data = service.load_data(data_file)
    if data is None:
        logger.error(f"Could not load data from {data_file}. Exiting.")
        return 1

    # Generate forecast or make predictions
    if args.forecast:
        logger.info("Generating forecast...")
        feature_cols = [col for col in data.columns if col != "Close"]
        forecast = service.generate_forecast(data, feature_cols)
        result = pd.DataFrame({"forecast": forecast})
    else:
        logger.info("Performing predictions...")
        predictions = service.predict(data)

        if len(predictions) == 0:
            logger.error("Prediction failed. Exiting.")
            return 1

        result = pd.DataFrame({"prediction": predictions})

    # Save or display results
    if args.output:
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save results
            if args.output.endswith(".csv"):
                result.to_csv(args.output, index=False)
            elif args.output.endswith((".xls", ".xlsx")):
                result.to_excel(args.output, index=False)
            elif args.output.endswith(".json"):
                with open(args.output, "w") as f:
                    json.dump(result.to_dict(orient="records"), f)
            else:
                np.save(args.output, result.values)

            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return 1
    else:
        # Display first 10 values
        display_count = min(10, len(result))
        logger.info(f"First {display_count} results:")
        for i in range(display_count):
            for col in result.columns:
                logger.info(f"{col}[{i}]: {result.iloc[i][col]}")

    logger.info("Processing completed successfully.")
    return 0


# ===== MODULE MAIN FUNCTION =====


def main():
    """
    Main function to run the prediction service.
    Determines whether to run CLI or Flask app based on arguments.
    """
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Run Flask app if first argument doesn't start with --

        prediction_service = PredictionService()
        app = create_flask_app(prediction_service)

        # Get port from arguments or use default
        port = 5000
        if len(sys.argv) > 2 and sys.argv[1] == "run":
            try:
                port = int(sys.argv[2])
            except ValueError:
                pass

        logger.info(f"Starting Flask app on port {port}")
        app.run(debug=False, port=port)
    else:
        # Run CLI
        return run_cli()


if __name__ == "__main__":
    sys.exit(main())
