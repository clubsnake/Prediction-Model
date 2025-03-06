"""
Unified configuration loader module.

Provides a streamlined interface to access configuration settings with a clear hierarchy:
1. User-editable settings (user_config.yaml) - for human-editable parameters
2. System-managed settings (system_config.json) - for Optuna and runtime values

Configuration values can be accessed using dot notation paths.
"""

import json
import logging
import os
import random
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta

import yaml

# Initialize the hyperparameter registry
HYPERPARAMETER_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Determine the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configuration file paths
USER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/user_config.yaml")
SYSTEM_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/system_config.json")

# Define constants needed by dashboard
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
DB_DIR = os.path.join(DATA_DIR, "DB")
LOGS_DIR = os.path.join(DATA_DIR, "Logs")
MODELS_DIR = os.path.join(DATA_DIR, "Models")
HYPERPARAMS_DIR = os.path.join(DATA_DIR, "Hyperparameters")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")

# Cache for loaded configurations
_config_cache = {"user": None, "system": None, "combined": None}

# Default configuration values
DEFAULT_CONFIG = {
    "data": {
        "base_path": "data/",
        "log_path": "data/Logs/",
        "model_path": "models/",
        "results_path": "results/"
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "early_stopping": 10
    },
    "tuning": {
        "trials_per_cycle": 20,
        "max_trials": 100
    },
    "dashboard": {
        "default_ticker": "AAPL",
        "default_timeframe": "1d",
        "theme": "light"
    }
}

# Create constants for easy access - these prevent the ImportError
TUNING_TRIALS_PER_CYCLE = DEFAULT_CONFIG["tuning"]["trials_per_cycle"]
TUNING_TRIALS_PER_CYCLE_min = TUNING_TRIALS_PER_CYCLE // 2
TUNING_TRIALS_PER_CYCLE_max = TUNING_TRIALS_PER_CYCLE * 2

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file with fallback to defaults"""
    config = DEFAULT_CONFIG.copy()
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                user_config = yaml.safe_load(file)
                
                # Update default config with user settings
                if user_config:
                    for section, values in user_config.items():
                        if section in config:
                            config[section].update(values)
                        else:
                            config[section] = values
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        
    return config

# Load config at module import time
CONFIG = load_config()

# Update the constants with values from loaded config
TUNING_TRIALS_PER_CYCLE = CONFIG["tuning"].get("trials_per_cycle", TUNING_TRIALS_PER_CYCLE)
TUNING_TRIALS_PER_CYCLE_min = CONFIG["tuning"].get("min_trials_per_cycle", TUNING_TRIALS_PER_CYCLE // 2)
TUNING_TRIALS_PER_CYCLE_max = CONFIG["tuning"].get("max_trials_per_cycle", TUNING_TRIALS_PER_CYCLE * 2)

def _replace_path_placeholders(config: Dict) -> Dict:
    """
    Recursively replace __project_root__ placeholders with actual project root path.

    Args:
        config: Dictionary containing configuration values

    Returns:
        Updated configuration with replaced paths
    """
    if not isinstance(config, dict):
        return config

    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = _replace_path_placeholders(value)
        elif isinstance(value, str) and "__project_root__" in value:
            config[key] = value.replace("__project_root__", PROJECT_ROOT)

    return config


def load_user_config(reload: bool = False) -> Dict:
    """
    Load the user configuration from YAML.

    Args:
        reload: Force reload from disk instead of using cache

    Returns:
        Dictionary containing user configuration
    """
    if _config_cache["user"] is None or reload:
        try:
            with open(USER_CONFIG_PATH, "r") as f:
                _config_cache["user"] = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading user config: {e}")
            # Return empty dict if file doesn't exist or has errors
            _config_cache["user"] = {}

    return _config_cache["user"]


def load_system_config(reload: bool = False) -> Dict:
    """
    Load the system configuration from JSON.

    Args:
        reload: Force reload from disk instead of using cache

    Returns:
        Dictionary containing system configuration
    """
    if _config_cache["system"] is None or reload:
        try:
            with open(SYSTEM_CONFIG_PATH, "r") as f:
                config = json.load(f)
                # Replace path placeholders
                _config_cache["system"] = _replace_path_placeholders(config)
        except Exception as e:
            print(f"Error loading system config: {e}")
            # Return empty dict if file doesn't exist or has errors
            _config_cache["system"] = {}

    return _config_cache["system"]


def get_config(reload: bool = False) -> Dict:
    """
    Get the combined configuration from both user and system configs.
    User config takes precedence over system config.

    Args:
        reload: Force reload from disk instead of using cache

    Returns:
        Combined dictionary containing all configuration
    """
    if _config_cache["combined"] is None or reload:
        # Load system config first (lower priority)
        system = load_system_config(reload)
        
        # Load user config (higher priority)
        user = load_user_config(reload)
        
        # Deep merge configs with user config taking precedence
        _config_cache["combined"] = deep_merge(system, user)

    return _config_cache["combined"]


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two dictionaries with the second taking precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary with higher precedence
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result


def get_value(path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation path.

    Args:
        path: Dot-separated path to the config value (e.g., "hardware.use_gpu")
        default: Default value to return if path doesn't exist

    Returns:
        The configuration value or default if not found
    """
    config = get_config()
    keys = path.split(".")

    # Navigate through nested dictionary
    current = config
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_value(path: str, value: Any, save: bool = True, target: str = "user") -> bool:
    """
    Set a configuration value using dot notation path.
    By default, this updates the user config as it's intended to be human-editable.

    Args:
        path: Dot-separated path to the config value (e.g., "hardware.use_gpu")
        value: New value to set
        save: Whether to save changes to disk
        target: Which config to update ("user" or "system")

    Returns:
        True if successful, False otherwise
    """
    # Determine which config to update
    if target.lower() == "user":
        config_path = USER_CONFIG_PATH
        config_data = load_user_config(True)  # Force reload to get fresh data
    elif target.lower() == "system":
        config_path = SYSTEM_CONFIG_PATH
        config_data = load_system_config(True)  # Force reload to get fresh data
    else:
        raise ValueError(f"Invalid target: {target}. Must be 'user' or 'system'")
    
    keys = path.split(".")

    # Navigate to the right nested level
    current = config_data
    for i, key in enumerate(keys[:-1]):
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the value
    current[keys[-1]] = value

    # Update cache
    _config_cache[target] = config_data
    _config_cache["combined"] = None  # Force recalculation of combined config

    # Save to disk if requested
    if save:
        try:
            if target.lower() == "user":
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            else:
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2, sort_keys=False)
            return True
        except Exception as e:
            print(f"Error saving {target} config: {e}")
            return False

    return True


def get_system_value(path: str, default: Any = None) -> Any:
    """
    Get a value specifically from the system config.
    Useful when you want to bypass the combined config.

    Args:
        path: Dot-separated path to the config value
        default: Default value to return if path doesn't exist

    Returns:
        The configuration value or default if not found
    """
    config = load_system_config()
    keys = path.split(".")

    # Navigate through nested dictionary
    current = config
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def get_user_value(path: str, default: Any = None) -> Any:
    """
    Get a value specifically from the user config.
    Useful when you want to bypass the combined config.

    Args:
        path: Dot-separated path to the config value
        default: Default value to return if path doesn't exist

    Returns:
        The configuration value or default if not found
    """
    config = load_user_config()
    keys = path.split(".")

    # Navigate through nested dictionary
    current = config
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def ensure_dir_exists(path_key: str) -> str:
    """
    Ensure that a directory specified in the config exists, and create it if not.

    Args:
        path_key: Dot-separated path to the directory config value

    Returns:
        The actual directory path
    """
    dir_path = get_value(path_key)
    if dir_path and isinstance(dir_path, str):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_data_dir(subdir: Optional[str] = None) -> str:
    """
    Helper function to get a data directory path.

    Args:
        subdir: Subdirectory name under the data directory

    Returns:
        Path to the requested directory
    """
    data_dir = DATA_DIR  # Use the constant defined at module level
    
    # Also try to get from config if available
    config_data_dir = get_value("project.data_dirs.data")
    if config_data_dir:
        data_dir = config_data_dir
        
    if subdir:
        path = os.path.join(data_dir, subdir)
        os.makedirs(path, exist_ok=True)
        return path
    return data_dir


def get_random_state():
    """
    Get the random state based on configuration settings.

    Returns:
        Either a fixed number, None, or a random number between 1-100
        depending on configuration.
    """
    random_state_config = get_value(
        "training.random_state", {"mode": "fixed", "value": 42}
    )
    mode = random_state_config.get("mode", "fixed")

    if mode == "fixed":
        return random_state_config.get("value", 42)
    elif mode == "off":
        return None
    elif mode == "random":
        return random.randint(1, 100)
    else:
        # Default to fixed if mode is unknown
        return random_state_config.get("value", 42)


def save_optuna_result(path: str, value: Any) -> bool:
    """
    Save an Optuna optimization result to the system config.
    
    Args:
        path: Dot-separated path to the config value
        value: Value to save (typically from Optuna optimization)
        
    Returns:
        True if successful, False otherwise
    """
    return set_value(path, value, save=True, target="system")


def get_active_feature_names():
    """Helper function to get active feature names from configuration"""
    features = get_value("features", {})
    base_features = features.get("base_features", ["Open", "High", "Low", "Close", "Volume"])
    
    # Check which feature toggles are on
    toggles = features.get("toggles", {})
    active_features = base_features.copy()
    
    if toggles.get("rsi", True):
        active_features.append("RSI")
    if toggles.get("macd", True):
        active_features.extend(["MACD", "MACD_Signal", "MACD_Hist"])
    if toggles.get("bollinger_bands", True):
        active_features.extend(["BB_Upper", "BB_Middle", "BB_Lower"])
    if toggles.get("atr", True):
        active_features.append("ATR")
    if toggles.get("obv", True):
        active_features.append("OBV")
    if toggles.get("werpi", True):
        active_features.append("WERPI")
    if toggles.get("vmli", True):
        active_features.append("VMLI")
    
    return active_features


def get_horizon_for_category(category="all"):
    """Get prediction horizon based on category"""
    default_horizon = get_value('time_series.prediction_horizon', 30)
    
    if category == "short":
        return max(1, int(default_horizon * 0.25))
    elif category == "medium":
        return int(default_horizon * 0.5)
    elif category == "long":
        return int(default_horizon * 1.5)
    else:  # "all" or any other value
        return default_horizon


def get_hyperparameter_ranges():
    """Get hyperparameter ranges for tuning"""
    # This would return the appropriate ranges for hyperparameters
    # based on the configuration
    return get_value('hyperparameter.ranges', {})


# Initialize directories based on configuration
def initialize_directories():
    """
    Create all directories specified in the configuration.
    """
    system_config = load_system_config()
    if "project" in system_config and "data_dirs" in system_config["project"]:
        for _, path in system_config["project"]["data_dirs"].items():
            os.makedirs(path, exist_ok=True)
    
    # Also ensure our direct constant paths exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HYPERPARAMS_DIR, exist_ok=True)


# Initialize on module import
initialize_directories()

# Define additional constants (extracted from config)
# These are used in various parts of the application
TICKER = get_value("tickers.default", "ETH-USD")  # Default ticker
INTERVAL = get_value("time_series.default_interval", "1d")  # Default timeframe
START_DATE = get_value("time_series.start_date", "auto")  # Default start date
PREDICTION_HORIZON = get_value("time_series.prediction_horizon", 30)  # Default horizon
WALK_FORWARD_DEFAULT = get_value("time_series.walk_forward.default_days", 30)  # Default walk forward size

# Get training-related constants
BATCH_SIZE = get_value("training.batch_size", 16)
LEARNING_RATE = get_value("training.learning_rate", 0.001)
VALIDATION_SPLIT = get_value("training.validation_split", 0.2)
SHUFFLE = get_value("training.shuffle", True)

# Get model types and active models from configuration
MODEL_TYPES = get_value("model.model_types", 
                      ["lstm", "rnn", "random_forest", "xgboost", "tft", "ltc", "tabnet", "cnn"])
                      
# Active model types
_active_model_config = get_value("model.active_model_types", {})
ACTIVE_MODEL_TYPES = [model_type for model_type, is_active in _active_model_config.items() if is_active]

# If empty, default to all model types
if not ACTIVE_MODEL_TYPES:
    ACTIVE_MODEL_TYPES = MODEL_TYPES

# Hyperparameter tuning constants
N_STARTUP_TRIALS = get_value("hyperparameter.n_startup_trials", 5000)
PRUNING_ENABLED = get_value("hyperparameter.pruning.enabled", True)
PRUNING_MEDIAN_FACTOR = get_value("hyperparameter.pruning.median_factor", 1.9)
PRUNING_MIN_TRIALS = get_value("hyperparameter.pruning.min_trials", 10)
PRUNING_ABSOLUTE_RMSE_FACTOR = get_value("hyperparameter.pruning.absolute_rmse_factor", 2.0)
PRUNING_ABSOLUTE_MAPE_FACTOR = get_value("hyperparameter.pruning.absolute_mape_factor", 3.0)

# Thresholds for stopping tuning
RMSE_THRESHOLD = get_value("hyperparameter.thresholds.rmse", 5.0)
MAPE_THRESHOLD = get_value("hyperparameter.thresholds.mape", 5.0)

# Tuning trials per cycle
TUNING_TRIALS_PER_CYCLE_min = get_value("hyperparameter.trials_per_cycle.min", 10)
TUNING_TRIALS_PER_CYCLE_max = get_value("hyperparameter.trials_per_cycle.max", 5000)
TUNING_LOOP = get_value("hyperparameter.tuning_loop", True)

# Get available loss functions
LOSS_FUNCTIONS = get_value("loss_functions.available", 
                         ["mean_squared_error", "mean_absolute_error", "huber_loss"])

# Create TFT grid (Temporal Fusion Transformer)
TFT_GRID = {
    "hidden_size": [32, 64, 128, 256],
    "lstm_units": [32, 64, 128, 256],
    "num_heads": [1, 2, 4, 8],
    "dropout": [0.1, 0.2, 0.3],
    "attention_dropout": [0.1, 0.2, 0.3]
}

# List of supported timeframes
TIMEFRAMES = get_value("time_series.timeframes",
                     ["1d", "3d", "1wk", "1h", "2h", "4h", "6h", "8h", "12h", "1mo", "1m", "5m", "15m", "30m"])

# List of supported tickers
TICKERS = get_value("tickers.symbols",
                  ["ETH-USD", "BTC-USD"])

def get_env_setting(name, default=None):
    """
    Get a setting from environment variable with fallback to config
    """
    import os
    env_value = os.environ.get(name)
    if env_value is not None:
        return env_value
    
    # Convert env var name to config path (e.g. TF_XLA_ENABLED -> tensorflow.xla_enabled)
    config_path = name.lower().replace('_', '.')
    return get_value(config_path, default)

# Return model config settings
def get_model_config(model_type):
    """Get specific configuration for a model type"""
    return get_value(f"models.{model_type}", {})

# Create logger settings
def get_logger_config():
    """Get logger configuration"""
    return {
        "level": get_value("logger.default_level", "INFO"),
        "format": get_value("logger.default_format", 
                         "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    }

# Define default CNN model settings
DEFAULT_CNN_PARAMS = {
    "num_conv_layers": 3,
    "num_filters": 64,
    "kernel_size": 3,
    "stride": 1, 
    "dropout_rate": 0.2,
    "activation": "relu",
    "use_adaptive_pooling": True,
    "fc_layers": [128, 64],
    "lookback": 30,
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "early_stopping_patience": 10
}

# Add CNN hyperparameters to the registry
HYPERPARAMETER_REGISTRY.update({
    "cnn_num_conv_layers": {
        "type": "int",
        "range": [1, 5],
        "default": 3,
        "description": "Number of convolutional layers in the CNN"
    },
    "cnn_num_filters": {
        "type": "int",
        "range": [16, 256],
        "default": 64,
        "log_scale": True,
        "description": "Number of filters in convolutional layers"
    },
    "cnn_kernel_size": {
        "type": "int",
        "range": [2, 7],
        "default": 3,
        "description": "Size of the convolutional kernel"
    },
    "cnn_stride": {
        "type": "int",
        "range": [1, 2],
        "default": 1,
        "description": "Stride of the convolution"
    },
    "cnn_dropout_rate": {
        "type": "float",
        "range": [0.0, 0.5],
        "default": 0.2,
        "description": "Dropout rate for CNN layers"
    },
    "cnn_activation": {
        "type": "categorical",
        "range": ["relu", "leaky_relu", "elu"],
        "default": "relu",
        "description": "Activation function for CNN layers"
    },
    "cnn_use_adaptive_pooling": {
        "type": "bool",
        "range": None,
        "default": True,
        "description": "Whether to use adaptive pooling in the CNN"
    },
    "cnn_fc_layer_1": {
        "type": "int",
        "range": [32, 256],
        "default": 128,
        "log_scale": True,
        "description": "Size of the first fully connected layer"
    },
    "cnn_fc_layer_2": {
        "type": "int",
        "range": [16, 128],
        "default": 64,
        "log_scale": True,
        "description": "Size of the second fully connected layer"
    },
    "cnn_lookback": {
        "type": "int",
        "range": [7, 90],
        "default": 30,
        "description": "Number of past time steps to use as input"
    },
    "cnn_lr": {
        "type": "float",
        "range": [1e-5, 1e-2],
        "default": 0.001,
        "log_scale": True,
        "description": "Learning rate for CNN model"
    },
    "cnn_batch_size": {
        "type": "categorical",
        "range": [16, 32, 64, 128],
        "default": 32,
        "description": "Batch size for CNN training"
    },
    "cnn_base_epochs": {
        "type": "int",
        "range": [1, 10],
        "default": 5,
        "description": "Base number of epochs (will be multiplied by epochs_multiplier)"
    },
    "cnn_early_stopping_patience": {
        "type": "int",
        "range": [3, 10],
        "default": 5,
        "description": "Patience for early stopping during CNN training"
    }
})