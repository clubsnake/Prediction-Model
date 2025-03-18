import os
import argparse
import sys

import yaml
import numpy as np  # Added import for RMSE calculation
from sklearn.metrics import mean_squared_error  # Added import for evaluation

# Fix import path
try:
    from config import __init__ as config_init
    from config.config_loader import get_data_dir

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except ImportError:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(PROJECT_ROOT)
from config.logger_config import logger

# These constants may not exist directly in config
try:
    from config.config_loader import get_data_dir

    MODELS_DIR = get_data_dir("Models")
    BEST_PARAMS_FILE = os.path.join(get_data_dir("Hyperparameters"), "best_params.yaml")
except ImportError:
    # Define fallbacks
    MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "Models")
    BEST_PARAMS_FILE = os.path.join(
        PROJECT_ROOT, "data", "Hyperparameters", "best_params.yaml"
    )
    logger.warning("Could not import from config.config_loader, using fallback paths")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train prediction models")

    parser.add_argument(
        "--ticker", type=str, default="BTC-USD", help="Ticker symbol to train on"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d", help="Timeframe to train on"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble",
        choices=["ensemble", "lstm", "xgboost", "random_forest", "tabnet"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Window size for walk-forward validation",
    )
    parser.add_argument(
        "--use-registry",
        action="store_true",
        help="Use the model registry for model management",
    )
    # Add command line argument for drift hyperparameter tuning
    parser.add_argument(
        "--tune-drift",
        action="store_true",
        help="Tune drift detection hyperparameters"
    )

    return parser.parse_args()


def train_model():
    """Train the model using centralized configurations and ModelTrainer."""
    logger.info("Starting model training...")

    # Parse command line arguments
    args = parse_arguments()

    # Load best parameters from YAML file
    try:
        with open(BEST_PARAMS_FILE, "r") as f:
            best_params = yaml.safe_load(f)
            if best_params is None:
                logger.warning(f"Empty or invalid YAML file: {BEST_PARAMS_FILE}")
                best_params = {}
    except FileNotFoundError:
        logger.error(f"Best parameters file not found at {BEST_PARAMS_FILE}")
        best_params = {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing best parameters file: {e}")
        best_params = {}

    # Get parameters for the selected model
    model_params = best_params.get(args.model_type, {})
    logger.info(f"Using parameters for model type: {args.model_type}")

    # Get data for training
    from src.data.data import fetch_data
    from src.features.features import feature_engineering

    try:
        # Fetch and prepare data
        df = fetch_data(ticker=args.ticker, interval=args.timeframe)
        df = feature_engineering(df)

        feature_cols = [
            col for col in df.columns if col not in ["date", "Date", "Close"]
        ]

        logger.info(f"Data prepared with {len(feature_cols)} features")

        # Handle drift hyperparameter tuning if requested
        if args.tune_drift:
            logger.info("Tuning drift detection hyperparameters...")
            
            from src.tuning.drift_optimizer import optimize_drift_hyperparameters
            from src.data.preprocessing import create_sequences
            
            # Set up submodel parameters
            submodel_params_dict = {
                model_type: params
                for model_type, params in best_params.items()
                if model_type in ["lstm", "random_forest", "xgboost", "tabnet"]
            }
            
            # Define training function with drift hyperparameters
            def train_with_drift_params(drift_hyperparams):
                # Create drift detector with custom hyperparameters
                from src.training.concept_drift import MultiDetectorDriftSystem
                drift_detector = MultiDetectorDriftSystem(hyperparams=drift_hyperparams)
                
                # Run walk-forward training with this detector
                from src.training.walk_forward import unified_walk_forward
                ensemble_model, _ = unified_walk_forward(
                    df=df,
                    feature_cols=feature_cols,
                    submodel_params_dict=submodel_params_dict,
                    window_size=args.window_size,
                    update_dashboard=False,
                    drift_detector=drift_detector
                )
                
                return ensemble_model
            
            # Define evaluation function
            def evaluate_model(model):
                # Create test window
                test_size = min(30, len(df) // 5)
                test_df = df.iloc[-test_size:].copy()
                
                # Prepare test data
                X_test, y_test = create_sequences(
                    test_df, feature_cols, "Close", lookback=30, horizon=1
                )
                
                # Make predictions
                preds = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
                
                return {"rmse": rmse}
            
            # Run optimization
            best_hyperparams = optimize_drift_hyperparameters(
                train_function=train_with_drift_params,
                eval_function=evaluate_model,
                n_trials=30,  # Adjust based on available time
                timeout=7200  # 2 hours timeout
            )
            
            # Save best hyperparameters
            hyperparams_dir = os.path.join(get_data_dir(), "Hyperparameters")
            os.makedirs(hyperparams_dir, exist_ok=True)
            
            drift_params_file = os.path.join(hyperparams_dir, f"drift_params_{args.ticker}_{args.timeframe}.yaml")
            with open(drift_params_file, "w") as f:
                yaml.dump(best_hyperparams, f)
            
            logger.info(f"Saved best drift hyperparameters to {drift_params_file}")
            
            # Continue with normal training but using the optimized drift parameters
            logger.info("Proceeding with normal training using optimized drift parameters")

        # Continue with regular model training logic
        if args.use_registry:
            # Use ModelRegistry from incremental_learning.py
            from src.training.incremental_learning import ModelRegistry

            registry = ModelRegistry(os.path.join(MODELS_DIR, "registry"))
            logger.info(
                f"Using model registry at {os.path.join(MODELS_DIR, 'registry')}"
            )

            # Use the modern walk_forward training approach
            from src.training.walk_forward import unified_walk_forward

            # Set up submodel parameters
            submodel_params_dict = {
                model_type: params
                for model_type, params in best_params.items()
                if model_type in ["lstm", "random_forest", "xgboost", "tabnet"]
            }

            # Train model
            ensemble_model, metrics = unified_walk_forward(
                df=df,
                feature_cols=feature_cols,
                submodel_params_dict=submodel_params_dict,
                window_size=args.window_size,
                update_dashboard=True,
            )

            # Register the trained model
            model_id = registry.register_model(
                model=ensemble_model,
                model_type="ensemble",
                ticker=args.ticker,
                timeframe=args.timeframe,
                metrics=metrics,
                hyperparams=submodel_params_dict,
                tags=["production", "walkforward"],
            )

            logger.info(f"Model trained and registered with ID: {model_id}")
            logger.info(
                f"Metrics: RMSE={metrics.get('rmse', 'N/A')}, MAPE={metrics.get('mape', 'N/A')}"
            )

        else:
            # Use the ModelTrainer from trainer.py
            from src.models.model import build_model_by_type
            from src.training.trainer import ModelTrainer

            # Build and train a specific model type
            model = build_model_by_type(
                model_type=args.model_type,
                num_features=len(feature_cols),
                **model_params,
            )

            trainer = ModelTrainer(model, config=model_params)

            # Prepare training data
            from src.data.preprocessing import create_sequences

            X, y = create_sequences(df, feature_cols, "Close", lookback=30, horizon=1)

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]

            # Train the model
            history = trainer.train(X_train, y_train, X_val, y_val)

            # Evaluate
            eval_metrics = trainer.evaluate(X_val, y_val)
            logger.info(f"Model trained with metrics: {eval_metrics}")

            # Save the model
            import joblib

            model_path = os.path.join(
                MODELS_DIR, f"{args.ticker}_{args.timeframe}_{args.model_type}.joblib"
            )
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")

    except Exception as e:
        logger.error(f"Error in training process: {e}", exc_info=True)

    logger.info("Model training completed.")


if __name__ == "__main__":
    train_model()
