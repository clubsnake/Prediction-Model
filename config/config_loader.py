"""
Unified configuration loader module.

This module provides a centralized system for accessing configuration settings across the project.
It loads settings from user_config.yaml (user-editable parameters) and system_config.json
(system-managed settings), and makes them accessible through consistent interfaces.

Key functionalities:
1. Loading configuration from YAML and JSON files
2. Providing dot notation access to nested settings
3. Resolving file paths with project root placeholders
4. Exposing commonly used constants to the rest of the application

This module is imported by most other modules in the project that need access to
configuration settings, making it a foundational component of the application.
"""

import json
import logging
import os
from typing import Any, Dict

import yaml

# Configure logger
logger = logging.getLogger(__name__)

# Initialize the hyperparameter registry
HYPERPARAMETER_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Determine the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)

# Configuration file paths
USER_CONFIG_PATH = os.path.join(script_dir, "user_config.yaml")
SYSTEM_CONFIG_PATH = os.path.join(script_dir, "system_config.json")


# Load configuration files
def load_config_file(filepath, is_yaml=False):
    """Load a configuration file"""
    try:
        with open(filepath, "r") as f:
            if is_yaml:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {filepath}: {e}")
        return {}


# NEW: Load API keys from api_keys.yaml
API_KEYS_FILE = os.path.join(script_dir, "api_keys.yaml")
api_keys_config = load_config_file(API_KEYS_FILE, is_yaml=True)
if not api_keys_config:
    api_keys_config = {}

# Load configurations
user_config = load_config_file(USER_CONFIG_PATH, is_yaml=True)
system_config = load_config_file(SYSTEM_CONFIG_PATH, is_yaml=False)

# Make system_config available for import
__all__ = ["system_config", "user_config", "API_KEYS", "APPLY_WEEKEND_GAP"]


# Function to resolve paths that use __project_root__ placeholder
def resolve_path(path_str):
    """Replace __project_root__ placeholder with actual project root path"""
    if isinstance(path_str, str) and "__project_root__" in path_str:
        return path_str.replace("__project_root__", PROJECT_ROOT)
    return path_str


# Process all project paths to resolve placeholders
if "project" in system_config:
    if "data_dirs" in system_config["project"]:
        for key, path in system_config["project"]["data_dirs"].items():
            system_config["project"]["data_dirs"][key] = resolve_path(path)

    if "files" in system_config["project"]:
        for key, path in system_config["project"]["files"].items():
            system_config["project"]["files"][key] = resolve_path(path)

# Extract commonly used constants
TICKER = user_config.get("tickers", {}).get("default", "ETH-USD")
TICKERS = user_config.get("tickers", {}).get("symbols", ["ETH-USD"])
TIMEFRAMES = user_config.get("time_series", {}).get("timeframes", ["1d"])
MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost", "tft", "ltc", "tabnet", "cnn", "nbeats"]
ACTIVE_MODEL_TYPES = []

for model_type in MODEL_TYPES:
    if (
        user_config.get("model", {})
        .get("active_model_types", {})
        .get(model_type, False)
    ):
        ACTIVE_MODEL_TYPES.append(model_type)

# If CNN is not in active models, add it (to ensure backward compatibility)
if "cnn" not in ACTIVE_MODEL_TYPES and "cnn" in MODEL_TYPES:
    ACTIVE_MODEL_TYPES.append("cnn")

# Add the missing AUTO_RUN_TUNING configuration
AUTO_RUN_TUNING = user_config.get("tuning", {}).get("auto_run", False)

# Constants for walk forward validation
WALK_FORWARD_MIN = (
    system_config.get("time_series", {}).get("walk_forward", {}).get("min_window", 3)
)
WALK_FORWARD_MAX = (
    system_config.get("time_series", {}).get("walk_forward", {}).get("max_window", 180)
)
WALK_FORWARD_DEFAULT = (
    system_config.get("time_series", {})
    .get("walk_forward", {})
    .get("default_window", 30)
)

# Constants for prediction settings
PREDICTION_HORIZON = user_config.get("time_series", {}).get("prediction_horizon", 30)
START_DATE = system_config.get("time_series", {}).get("start_date", "auto")
LOOKBACK = user_config.get("model", {}).get("default_lookback", 30)

# Constants for hyperparameter tuning
N_STARTUP_TRIALS = user_config.get("hyperparameter", {}).get(
    "n_startup_trials",
    system_config.get("hyperparameter", {}).get("n_startup_trials", 10000),
)
TUNING_TRIALS_PER_CYCLE_min = (
    system_config.get("hyperparameter", {}).get("trials_per_cycle", {}).get("min", 10)
)
TUNING_TRIALS_PER_CYCLE_max = (
    system_config.get("hyperparameter", {}).get("trials_per_cycle", {}).get("max", 5000)
)
TUNING_LOOP = system_config.get("hyperparameter", {}).get("tuning_loop", True)

# Constants for random seed
RANDOM_SEED_VALUE = user_config.get("random_seed", {}).get("value", 42)

# Constants for pruning settings
PRUNING_ENABLED = (
    system_config.get("hyperparameter", {}).get("pruning", {}).get("enabled", True)
)
PRUNING_MEDIAN_FACTOR = (
    system_config.get("hyperparameter", {}).get("pruning", {}).get("median_factor", 1.9)
)
PRUNING_MIN_TRIALS = (
    system_config.get("hyperparameter", {}).get("pruning", {}).get("min_trials", 10)
)
PRUNING_ABSOLUTE_RMSE_FACTOR = (
    system_config.get("hyperparameter", {})
    .get("pruning", {})
    .get("absolute_rmse_factor", 2.0)
)
PRUNING_ABSOLUTE_MAPE_FACTOR = (
    system_config.get("hyperparameter", {})
    .get("pruning", {})
    .get("absolute_mape_factor", 3.0)
)

# Constants for loss functions
LOSS_FUNCTIONS = system_config.get("loss_functions", {}).get(
    "available", ["mean_squared_error", "mean_absolute_error"]
)
RMSE_THRESHOLD = (
    user_config.get("incremental_learning", {})
    .get("model_saving", {})
    .get("rmse_threshold", 5.0)
)
MAPE_THRESHOLD = (
    user_config.get("incremental_learning", {})
    .get("model_saving", {})
    .get("mape_threshold", 8.0)
)

# Incremental learning constants
INCREMENTAL_RETRAINING_THRESHOLD = (
    user_config.get("incremental_learning", {})
    .get("retraining", {})
    .get("threshold", 0.1)
)
MAX_SAVED_MODELS = user_config.get("incremental_learning", {}).get(
    "max_saved_models", 100
)

# Get centralized directory paths from system_config.json
DATA_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("data", os.path.join(PROJECT_ROOT, "data"))  # Ensure lowercase "data"
)
DB_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("db", os.path.join(DATA_DIR, "db"))  # Ensure lowercase "db"
)
MODELS_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("models", os.path.join(DATA_DIR, "models"))  # Ensure lowercase "models"
)
HYPERPARAMS_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("hyperparams", os.path.join(MODELS_DIR, "hyperparams"))  # Use lowercase "hyperparams"
)
RAW_DATA_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("raw_data", os.path.join(DATA_DIR, "raw"))  # Ensure lowercase "raw"
)
PROCESSED_DATA_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("processed_data", os.path.join(DATA_DIR, "processed"))  # Ensure lowercase "processed"
)
LOGS_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("logs", os.path.join(DATA_DIR, "logs"))  # Ensure lowercase "logs"
)

# Add new registry directory constants
REGISTRY_DIR = (
    system_config.get("project", {})
    .get("data_dirs", {})
    .get("registry", os.path.join(MODELS_DIR, "registry"))  # Ensure lowercase "registry"
)

# Add registry configuration constants
REGISTRY_MAX_MODELS = system_config.get("registry", {}).get(
    "max_saved_models", MAX_SAVED_MODELS
)
REGISTRY_AUTO_CLEAN = system_config.get("registry", {}).get("auto_clean", True)
REGISTRY_VERSION_TIMEOUT = system_config.get("registry", {}).get("version_timeout", 3600)

# Get centralized file paths from system_config.json
PROGRESS_FILE = (
    system_config.get("project", {})
    .get("files", {})
    .get("progress_file", os.path.join(DATA_DIR, "progress.yaml"))
)
TUNING_STATUS_FILE = (
    system_config.get("project", {})
    .get("files", {})
    .get("tuning_status_file", os.path.join(DATA_DIR, "tuning_status.txt"))
)
TESTED_MODELS_FILE = (
    system_config.get("project", {})
    .get("files", {})
    .get("tested_models_file", os.path.join(DATA_DIR, "tested_models.yaml"))
)
CYCLE_METRICS_FILE = (
    system_config.get("project", {})
    .get("files", {})
    .get("cycle_metrics_file", os.path.join(DATA_DIR, "cycle_metrics.yaml"))
)
BEST_PARAMS_FILE = (
    system_config.get("project", {})
    .get("files", {})
    .get("best_params_file", os.path.join(HYPERPARAMS_DIR, "best_params.yaml"))
)

# Add model progress directory path
MODEL_PROGRESS_DIR = os.path.join(DATA_DIR, "model_progress")  # Maintain lowercase

# Constants for visualization
SHOW_PREDICTION_PLOTS = user_config.get("dashboard", {}).get(
    "show_prediction_plots", True
)
SHOW_TRAINING_HISTORY = user_config.get("dashboard", {}).get(
    "show_training_history", True
)
SHOW_WEIGHT_HISTOGRAMS = user_config.get("dashboard", {}).get(
    "show_weight_histograms", True
)
UPDATE_DURING_WALK_FORWARD = user_config.get("dashboard", {}).get(
    "update_during_walk_forward", True
)

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    HYPERPARAMS_DIR,
    DB_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REGISTRY_DIR, 
    MODEL_PROGRESS_DIR,  # Add this to ensure it's created
]:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logger.warning(f"Error creating directory {directory}: {e}")

# Create registry directory
try:
    os.makedirs(REGISTRY_DIR, exist_ok=True)
except Exception as e:
    logger.warning(f"Error creating registry directory {REGISTRY_DIR}: {e}")

# Set default interval
INTERVAL = user_config.get("time_series", {}).get("default_interval", "1d")

# Define the TFT grid for grid search (placeholder)
TFT_GRID = {}

# Caching settings
USE_CACHING = system_config.get("data", {}).get("use_caching", True)

# Use API keys from api_keys.yaml
API_KEYS = {
    "alphavantage": api_keys_config.get("alphavantage_api_key", "demo"),
    "finnhub": api_keys_config.get("finnhub_api_key", "demo"),
    "coingecko": api_keys_config.get(
        "coingecko_api_key", ""
    ),  # CoinGecko has a free tier without API key
}

# Load APPLY_WEEKEND_GAP from user_config.yaml
APPLY_WEEKEND_GAP = user_config.get("toggles", {}).get("apply_weekend_gap", True)

# Add threshold configurations
THRESHOLDS = {
    'confidence_threshold': 0.75,
    'anomaly_threshold': 3.0,
    'time_series_min_points': 30,
    'correlation_significance': 0.05,
    'feature_importance_threshold': 0.01,
    'tabnet': {
        'n_d': 64,
        'n_a': 64,
        'n_steps': 5,
        'gamma': 1.5,
        'lambda_sparse': 0.001
    }
}

# Add update interval for walk forward
UPDATE_DURING_WALK_FORWARD_INTERVAL = 60  # Default update interval in seconds

# Function to get data directory path
def get_data_dir(subdir=None):
    """
    Get the path to a data directory.

    Args:
        subdir (str, optional): Subdirectory within the data directory

    Returns:
        str: Path to the requested directory
    """
    base_dir = DATA_DIR

    if subdir:
        dir_path = os.path.join(base_dir, subdir)
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    return base_dir


# Get config value with dotted path notation
def get_value(path, default=None, target=None):
    """
    Get a configuration value using a dot notation path.

    Args:
        path: Dot separated path to the config value (e.g., "hardware.use_gpu")
        default: Default value to return if path not found
        target: Specific config to search ('user', 'system', or None for both)

    Returns:
        The configured value or default if not found
    """
    parts = path.split(".")

    # Search in user config first (if not restricted)
    if target is None or target == "user":
        current = user_config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break

        if current is not None:
            return current

    # Search in system config if not found in user config
    if target is None or target == "system":
        current = system_config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break

        if current is not None:
            return current

    # Return default if not found
    return default


# Set config value with dotted path notation
def set_value(path, value, target="user"):
    """
    Set a configuration value using a dot notation path.

    Args:
        path: Dot separated path to the config value (e.g., "hardware.use_gpu")
        value: Value to set
        target: Specific config to update ('user' or 'system')

    Returns:
        Success boolean
    """
    parts = path.split(".")
    config_file = USER_CONFIG_PATH if target == "user" else SYSTEM_CONFIG_PATH
    config = user_config if target == "user" else system_config

    # Navigate to the nested location
    current = config
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]

    # Set the value
    current[parts[-1]] = value

    # Save the updated config
    try:
        with open(config_file, "w") as f:
            if target == "user":
                yaml.dump(config, f)
            else:
                json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config to {config_file}: {e}")
        return False


# Get the entire config as a dictionary
def get_config():
    """
    Get the merged configuration dictionary.

    Returns:
        Dictionary with all configuration values
    """
    # Start with system config as base
    merged = system_config.copy()

    # Recursively update with user config
    def update_recursive(base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                update_recursive(base[key], value)
            else:
                base[key] = value

    update_recursive(merged, user_config)

    # Ensure all constants are directly available
    merged["TICKER"] = TICKER
    merged["TICKERS"] = TICKERS
    merged["TIMEFRAMES"] = TIMEFRAMES
    merged["MODEL_TYPES"] = MODEL_TYPES
    merged["ACTIVE_MODEL_TYPES"] = ACTIVE_MODEL_TYPES
    merged["WALK_FORWARD_MIN"] = WALK_FORWARD_MIN
    merged["WALK_FORWARD_MAX"] = WALK_FORWARD_MAX
    merged["WALK_FORWARD_DEFAULT"] = WALK_FORWARD_DEFAULT
    merged["PREDICTION_HORIZON"] = PREDICTION_HORIZON
    merged["START_DATE"] = START_DATE
    merged["UPDATE_DURING_WALK_FORWARD"] = UPDATE_DURING_WALK_FORWARD

    return merged


def get_horizon_for_category(category=None):
    """
    Get the prediction horizon for a specific market category.

    Args:
        category: Market category (e.g., 'crypto', 'stocks', 'forex')

    Returns:
        int: Prediction horizon in days
    """
    # Default horizon if category not specified
    if category is None:
        return PREDICTION_HORIZON

    # Get category-specific horizons from config
    category_horizons = user_config.get("time_series", {}).get("category_horizons", {})

    # Return the category horizon or default if not found
    return category_horizons.get(category.lower(), PREDICTION_HORIZON)


def initialize_random_seed():
    """
    Initialize random seed based on configuration.

    Modes:
    - "fixed": Use a fixed seed value from configuration
    - "off": Don't set any seed (random behavior)
    - "random": Use a random seed between 1-100 for each run

    Returns:
        int or None: The seed that was set, or None if no seed was set
    """
    import random

    import numpy as np
    import tensorflow as tf

    # Get random seed mode from user configuration
    mode = get_value("random_seed.mode", default="fixed")

    # No seed setting (random behavior)
    if mode.lower() == "off":
        return None

    # Fixed seed (reproducible behavior)
    elif mode.lower() == "fixed":
        seed_value = get_value("random_seed.value", default=42)

    # Random seed (changes each run but is documented)
    elif mode.lower() == "random":
        import random as py_random

        seed_value = py_random.randint(1, 100)
        logger.info(f"Generated random seed: {seed_value}")

    # Invalid mode, default to fixed
    else:
        logger.warning(f"Invalid random_seed.mode '{mode}', using 'fixed' mode")
        seed_value = get_value("random_seed.value", default=42)

    # Set seeds for different libraries
    try:
        # Python's built-in random
        random.seed(seed_value)

        # NumPy
        np.random.seed(seed_value)

        # TensorFlow
        tf.random.set_seed(seed_value)

        logger.info(f"Random seed set to {seed_value}")
        return seed_value

    except Exception as e:
        logger.error(f"Error setting random seed: {e}")
        return None


# Add this function to get the random seed
def get_random_state():
    """
    Get the random state value from the configuration.

    Returns:
        int: The random state value
    """
    return get_value("random_seed.value", default=42)


def get_active_feature_names():
    """
    Get the list of active features based on configuration.

    Returns:
        List of active feature names
    """
    active_features = []

    # Add base features
    base_features = system_config.get("features", {}).get("base_features", ["Open", "High", "Low", "Close", "Volume"])
    active_features.extend(base_features)
    
    # Add technical indicators if enabled
    tech_indicators = system_config.get("features", {}).get("technical_indicators", {})
    for indicator, settings in tech_indicators.items():
        if user_config.get("features", {}).get("technical_indicators", {}).get(indicator, True):
            indicator_name = indicator.upper() if indicator in ["rsi", "macd", "atr"] else indicator.capitalize()
            active_features.append(indicator_name)
            
            # Add sub-components for indicators like MACD that have multiple outputs
            if indicator == "macd":
                active_features.extend(["MACD_Signal", "MACD_Hist"])
            elif indicator == "bollinger_bands":
                active_features.extend(["BB_Upper", "BB_Lower"])
                
    # Add any custom features from user config
    custom_features = user_config.get("features", {}).get("custom_features", [])
    active_features.extend(custom_features)
    
    return active_features


def get_gpu_settings():
    """Get a dictionary of all GPU-related settings"""
    return {
        "use_mixed_precision": get_value("hardware.use_mixed_precision", False),
        "use_xla": get_value("hardware.use_xla", True),
        "memory_growth": get_value("hardware.memory_growth", True)
    }

def set_all_random_states(seed):
    """
    Set random seeds for reproducibility using:
    - Python random,
    - NumPy,
    - TensorFlow,
    - PyTorch (if available)
    Returns the seed set.
    """
    import random
    import numpy as np
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
