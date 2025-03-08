"""
Config package initialization.
Imports and exposes key configuration functions and constants.
"""

import os
import sys

# Add the project root to the Python path
config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(config_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import key components from config_loader
try:
    from .config_loader import (
        # Functions
        get_config,
        get_value,
        set_value,
        get_random_state,
        get_data_dir,
        get_active_feature_names,
        # Constants
        DATA_DIR,
        DB_DIR,
        LOGS_DIR,
        MODELS_DIR,
        HYPERPARAMS_DIR,
        PROGRESS_FILE,
        TESTED_MODELS_FILE,
        TUNING_STATUS_FILE,
        TICKER,
        TICKERS,
        TIMEFRAMES,
        N_STARTUP_TRIALS,
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
    for directory in [DATA_DIR, LOGS_DIR, DB_DIR, MODELS_DIR, HYPERPARAMS_DIR, WEIGHTS_DIR, VISUALIZATIONS_DIR]:
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
    logger = logging.getLogger('prediction_model')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    
    def setup_logger(name, level=None, **kwargs):
        new_logger = logging.getLogger(name)
        new_logger.setLevel(level or logging.INFO)
        if not new_logger.handlers:
            new_logger.addHandler(handler)
        return new_logger