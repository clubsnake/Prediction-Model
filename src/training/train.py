import sys

from config import PROJECT_ROOT

sys.path.append(PROJECT_ROOT)
import yaml
from config.logger_config import logger

from config import BEST_PARAMS_FILE, MODELS_DIR


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
