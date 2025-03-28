"""
Configuration package initialization.

This module initializes the configuration package and provides convenient access
to configuration functions and constants. It handles:

1. Adding the project root to the Python path
2. Importing and exposing key configuration functions from config_loader.py
3. Providing fallback values and functions if imports fail
4. Initializing the logging system

The config package is a critical dependency for all other modules in the project,
providing centralized configuration management and logging setup.
"""

import os
import sys

# Add the project root to the Python path
config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(config_dir)
PROJECT_ROOT = project_root  # Expose PROJECT_ROOT as top-level constant
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Function to get system configuration
def get_system_config():
    """Get the current system configuration."""
    from .resource_config import load_system_config

    return load_system_config()


# Import key components from config_loader
try:
    from .config_loader import (
        DATA_DIR,
        DB_DIR,  # Functions; Constants
        HYPERPARAMS_DIR,
        LOGS_DIR,
        MODELS_DIR,
        N_STARTUP_TRIALS,
        PROGRESS_FILE,
        TESTED_MODELS_FILE,
        TICKER,
        TICKERS,
        TIMEFRAMES,
        TUNING_STATUS_FILE,
        get_active_feature_names,
        get_config,
        get_data_dir,
        get_random_state,
        get_value,
        set_value,
    )
except ImportError as e:
    print(f"Error importing from config_loader: {e}")

    # Define fallback constants if imports fail
    import os

    # Base directories
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    LOGS_DIR = os.path.join(DATA_DIR, "logs")
    DB_DIR = os.path.join(DATA_DIR, "db")
    MODELS_DIR = os.path.join(DATA_DIR, "models")  # Standardized to lowercase
    HYPERPARAMS_DIR = os.path.join(MODELS_DIR, "hyperparams")
    WEIGHTS_DIR = os.path.join(MODELS_DIR, "weights")
    VISUALIZATIONS_DIR = os.path.join(MODELS_DIR, "visualizations")

    # Ensure directories exist
    for directory in [
        DATA_DIR,
        LOGS_DIR,
        DB_DIR,
        MODELS_DIR,
        HYPERPARAMS_DIR,
        WEIGHTS_DIR,
        VISUALIZATIONS_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    # Files
    PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
    TESTED_MODELS_FILE = os.path.join(MODELS_DIR, "tested_models.yaml")
    TUNING_STATUS_FILE = os.path.join(MODELS_DIR, "tuning_status.yaml")
    CYCLE_METRICS_FILE = os.path.join(MODELS_DIR, "cycle_metrics.yaml")
    BEST_PARAMS_FILE = os.path.join(HYPERPARAMS_DIR, "best_params.yaml")

    # Default parameters
    DEFAULT_TICKER = "BTC-USD"
    DEFAULT_TIMEFRAMES = ["1d", "1h", "15m"]

    # Define fallback functions
    def get_config():
        return {}

    def get_value(path, default=None):
        return default

    def set_value(path, value, save=True, target="user"):
        return False

    def get_random_state():
        return 42

    def get_data_dir(subdir=None):
        if subdir:
            path = os.path.join(DATA_DIR, subdir)
            os.makedirs(path, exist_ok=True)
            return path
        return DATA_DIR

    def get_active_feature_names():
        return ["Open", "High", "Low", "Close", "Volume"]


# Import logger
try:
    from .logger_config import logger, setup_logger
except ImportError as e:
    print(f"Error importing from logger_config: {e}")

    # Define a basic logger if imports fail
    import logging

    logger = logging.getLogger("prediction_model")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    def setup_logger(name, level=None, **kwargs):
        new_logger = logging.getLogger(name)
        new_logger.setLevel(level or logging.INFO)
        if not new_logger.handlers:
            new_logger.addHandler(handler)
        return new_logger


MODEL_TYPES = ["lstm", "rnn", "xgboost", "random_forest", "nbeats", "ltc", "tft"]
ACTIVE_MODEL_TYPES = ["lstm", "rnn", "xgboost", "random_forest", "nbeats", "ltc", "tft"]
MAPE_THRESHOLD = 5.0
RMSE_THRESHOLD = 0.05
TUNING_TRIALS_PER_CYCLE_min = 10
TUNING_TRIALS_PER_CYCLE_max = 5000
