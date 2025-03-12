"""
Main entry point. Sets environment variables, launches Streamlit dashboard,
and optionally performs real-time updates or auto-tuning.
"""

import logging
import os
import subprocess
import sys
import threading

# Add project root to sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import TensorFlow safely
try:
    pass
except ImportError:
    print(
        "Error: TensorFlow is not installed. Install it with 'pip install tensorflow'."
    )

# Configure GPU first (ensure resource_config exists and is correct)
try:
    pass
except Exception as e:
    print(f"Error importing resource_config: {e}")

# Set environment variables from config; ensure they exist in config.py
try:
    from config.config_loader import AUTO_RUN_TUNING, OMP_NUM_THREADS, REALTIME_UPDATE
except Exception as e:
    print(f"Error importing configuration: {e}")
    sys.exit(1)

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"  # Enable XLA devices

# Ensure the Logs directory exists
LOGS_DIR = os.path.join("Data", "Logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Update logging configuration to use the new directory
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.path.join(LOGS_DIR, "prediction_model.log"),
)
logger = logging.getLogger()

# Import and register TFT model safely
try:
    from src.models.temporal_fusion_transformer import (
        add_tft_to_model_types,
        add_tft_to_optuna_search,
    )
    from src.tuning import meta_tuning

    # Register TFT with model system
    add_tft_to_model_types()

    # Add TFT to Optuna search
    add_tft_to_optuna_search(meta_tuning)

    # Update MODEL_TYPES in config if needed
    from config.config_loader import MODEL_TYPES

    if "tft" not in MODEL_TYPES:
        MODEL_TYPES.append("tft")

except Exception as e:
    logger.error(f"Error initializing TFT model: {e}")

# Fix Streamlit cache issue with backward compatibility
try:
    import streamlit as st

    # Handle Streamlit caching issue dynamically
    if hasattr(st, "cache_data"):
        cache_function = st.cache_data
    elif hasattr(st, "cache_resource"):
        cache_function = st.cache_resource
    else:
        cache_function = st.cache

    # For model loading specifically
    if hasattr(st, "cache_resource"):
        cache_model = st.cache_resource
    else:
        cache_model = st.cache(allow_output_mutation=True)

except ImportError:
    logger.error(
        "Streamlit is not installed. Please install it using 'pip install streamlit'."
    )

    # Define fallback functions to avoid errors if streamlit isn't available
    def cache_function(func):
        return func

    def cache_model(func):
        return func


# Launch Dashboard
def launch_dashboard(mode="full") -> None:
    """
    Launch the appropriate dashboard in a separate subprocess.

    Args:
        mode: 'full' for main dashboard, 'enhanced' for the enhanced version
    """
    try:
        if mode == "enhanced":
            script_path = os.path.join("dashboard_core,py")
            logger.info("Launching enhanced dashboard...")
        else:
            possible_paths = [
                os.path.join("src", "dashboard", "dashboard", "dashboard_core.py"),
            ]

            script_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    script_path = path
                    break

            if not script_path:
                logger.error("Dashboard file not found in any expected location")
                return

            logger.info(f"Launching standard dashboard from {script_path}...")

        command = [sys.executable, "-m", "streamlit", "run", script_path]
        subprocess.Popen(command, cwd=os.getcwd())
        logger.info(
            f"Dashboard ({mode}) launched successfully with command: %s",
            " ".join(command),
        )
    except Exception as e:
        logger.error(f"Error launching {mode} dashboard: {e}")


from config import Config
from src.dashboard.prediction_service import PredictionService
from src.data.data_handler import DataHandler
from src.data.data_loader import DataLoader
from src.data.data_utils import train_test_split
from src.models.model_factory import ModelFactory
from src.models.model_trainer import ModelTrainer


def main() -> None:
    """
    Main entry for the script. Optionally performs real-time data update
    and triggers the auto-tuning if configured.
    """
    logger.info("Starting main program...")

    data_dir = Config.DATA_DIR
    data_loader = DataLoader(data_dir)
    ModelTrainer(data_dir)
    PredictionService(data_dir)

    # Example usage:
    data = data_loader.load_data("your_data.csv")
    if data is not None:
        # Train your model
        pass

        # Save your model
        # model_trainer.save_model(model, "your_model.joblib")

        # Load your model
        # loaded_model = model_trainer.load_model("your_model.joblib")

        # Make predictions
        # predictions = prediction_service.predict(loaded_model, data)

    # Launch Streamlit dashboard in a separate thread
    try:
        # You can change this to "enhanced" to use the enhanced dashboard
        dashboard_mode = os.environ.get("DASHBOARD_MODE", "full")
        dashboard_thread = threading.Thread(
            target=lambda: launch_dashboard(dashboard_mode), daemon=True
        )
        dashboard_thread.start()
    except Exception as e:
        logger.error(f"Error starting dashboard thread: {e}")

    # Example placeholder for real-time updates
    if REALTIME_UPDATE:
        logger.info("Real-time update logic would go here. (placeholder)")

    # Auto-run tuning if desired
    if AUTO_RUN_TUNING:
        try:
            from src.tuning import meta_tuning

            meta_tuning.main()
        except Exception as e:
            logger.error(f"Error in auto-run tuning: {e}")

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        # Create default config if it doesn't exist
        default_config = {
            "data_path": "./data/example_data.csv",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "label_column": "target",
            "model_type": "linear_regression",
            "normalize": True,
            "test_size": 0.2,
            "random_seed": 42,
            "epochs": 10,
            "save_path": "./results",
        }
        Config.save_config(default_config, config_path)

    config = Config.load_config(config_path)

    # Process data
    data_handler = DataHandler(config)
    X, y = data_handler.process_pipeline(config["data_path"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.get("test_size", 0.2),
        random_seed=config.get("random_seed", 42),
    )

    # Create model
    model = ModelFactory.create_model(config["model_type"], config)

    # Train and evaluate model
    trainer = ModelTrainer(model, config)
    trainer.train(X_train, y_train, X_test, y_test)

    # Evaluate on test data
    test_metrics = trainer.evaluate(X_test, y_test)
    print(f"Test Metrics: {test_metrics}")

    # Visualize results
    save_path = config.get("save_path", "./results")
    os.makedirs(save_path, exist_ok=True)

    trainer.visualize_training(os.path.join(save_path, "training_history.png"))
    trainer.visualize_predictions(
        X_test, y_test, os.path.join(save_path, "predictions.png")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dashboard",
        choices=["full", "enhanced"],
        default="full",
        help="Select which dashboard to launch",
    )
    args = parser.parse_args()

    os.environ["DASHBOARD_MODE"] = args.dashboard
    main()
