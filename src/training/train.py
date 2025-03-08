import sys
import os

# Fix import path
try:
    from config.config_loader import get_data_dir
    from config import __init__ as config_init
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
    BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "data", "Hyperparameters", "best_params.yaml")
    logger.warning("Could not import from config.config_loader, using fallback paths")


def train_model():
    """Train the model using centralized configurations."""
    logger.info("Starting model training...")

    # Example usage of centralized file paths
    best_params_path = BEST_PARAMS_FILE
    logger.info(f"Loading best parameters from: {best_params_path}")

    # Load best parameters from YAML file
    try:
        with open(best_params_path, "r") as f:
            yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Best parameters file not found at {best_params_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing best parameters file: {e}")
        return

    # Example usage of centralized directories
    models_dir = MODELS_DIR
    logger.info(f"Saving trained model to: {models_dir}")

    # Add your model training logic here, using best_params and models_dir
    # ...

    logger.info("Model training completed.")


if __name__ == "__main__":
    train_model()
