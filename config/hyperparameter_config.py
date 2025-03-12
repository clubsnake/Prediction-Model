# hyperparameter_config.py
"""
Hyperparameter configuration and settings module.

This includes hyperparameter search options and specific tuning parameters.
"""
import os
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fix the import path - use relative import since we're already in config package
from config.config_loader import N_STARTUP_TRIALS, TUNING_TRIALS_PER_CYCLE_max, TUNING_TRIALS_PER_CYCLE_min

# Hyperparameter Registry System
HYPERPARAMETER_REGISTRY = {}


def register_param(
    name, default, type_hint="categorical", range_values=None, group=None, **kwargs
):
    """
    Register a parameter for potential hyperparameter tuning.

    Args:
        name: Parameter name
        default: Default value
        type_hint: Type of parameter ("int", "float", "categorical", "bool")
        range_values: Possible values or [min, max] for numerical types
        group: Optional group name for parameter organization
        **kwargs: Additional parameter attributes (log_scale, step, etc.)

    Returns:
        The default value for immediate use
    """
    HYPERPARAMETER_REGISTRY[name] = {
        "default": default,
        "type": type_hint,
        "range": range_values,
        "group": group or "misc",
        **kwargs,
    }
    return default


# Pruning settings
PRUNING_ENABLED = register_param("pruning_enabled", True, "bool", None, "pruning")
PRUNING_MEDIAN_FACTOR = register_param(
    "pruning_median_factor", 1.9, "float", [1.1, 5.0], "pruning"
)
PRUNING_MIN_TRIALS = register_param(
    "pruning_min_trials", 10, "int", [5, 100], "pruning"
)
PRUNING_ABSOLUTE_RMSE_FACTOR = register_param(
    "pruning_absolute_rmse_factor", 2.0, "float", [1.1, 5.0], "pruning"
)
PRUNING_ABSOLUTE_MAPE_FACTOR = register_param(
    "pruning_absolute_mape_factor", 3.0, "float", [1.1, 10.0], "pruning"
)

# Desired thresholds – tuning will continue until these are met (or manual stop).
RMSE_THRESHOLD = 1500
MAPE_THRESHOLD = 5.0

AUTO_RUN_TUNING = False  # Start tuning automatically on startup

# Meta-tuning parameters with min/max ranges - use values from config_loader for consistency
META_TUNING_PARAMS = {
    "TOTAL_STEPS": {"min": 1, "max": 500},  # Steps for splitting evaluation
    "N_STARTUP_TRIALS": {
        "min": 5,
        "max": 10000,
        "default": N_STARTUP_TRIALS,
    },  # Trials before pruning starts
    "N_WARMUP_STEPS": {"min": 5, "max": 500},  # Steps before pruning enabled
    "INTERVAL_STEPS": {"min": 1, "max": 50},  # Interval between pruning checks
    "PRUNING_PERCENTILE": {"min": 1, "max": 90},  # Percentile for pruning
    "META_TUNING_ENABLED": True,  # Toggle for meta-tuning
    "TRIALS_PER_CYCLE": {
        "min": TUNING_TRIALS_PER_CYCLE_min,
        "max": TUNING_TRIALS_PER_CYCLE_max,
    },  # Min/max trials per cycle from config_loader
}

# Define model types
MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost", "tft", "ltc", "tabnet"]

# Set search method to Optuna only
HYPERPARAM_SEARCH_METHOD = "optuna"

# Add LTC-specific hyperparameters to the registry
HYPERPARAMETER_REGISTRY.update(
    {
        "ltc_units": {
            "type": "int",
            "default": 64,
            "range": [32, 512],
            "group": "ltc",
            "log_scale": True,
        },
        "ltc_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-5, 1e-2],
            "group": "ltc",
            "log_scale": True,
        },
        "ltc_lookback": {
            "type": "int",
            "default": 30,
            "range": [7, 90],
            "group": "ltc",
        },
        "ltc_epochs": {"type": "int", "default": 25, "range": [5, 100], "group": "ltc"},
        "ltc_batch_size": {
            "type": "int",
            "default": 32,
            "range": [16, 128],
            "group": "ltc",
        },
        "ltc_loss": {
            "type": "categorical",
            "default": "mean_squared_error",
            "range": ["mean_squared_error", "mean_absolute_error", "huber_loss"],
            "group": "ltc",
        },
    }
)

# TabNet hyperparameters
HYPERPARAMETER_REGISTRY.update(
    {
        # Architecture parameters
        "n_d": {
            "type": "int",
            "default": 64,
            "range": [8, 256],
            "group": "tabnet",
            "log_scale": True,
        },
        "n_a": {
            "type": "int",
            "default": 64,
            "range": [8, 256],
            "group": "tabnet",
            "log_scale": True,
        },
        "n_steps": {"type": "int", "default": 5, "range": [1, 15], "group": "tabnet"},
        "gamma": {
            "type": "float",
            "default": 1.5,
            "range": [0.5, 3.0],
            "group": "tabnet",
        },
        "lambda_sparse": {
            "type": "float",
            "default": 0.001,
            "range": [1e-7, 1e-1],
            "group": "tabnet",
            "log_scale": True,
        },
        # Training parameters
        "tabnet_max_epochs": {
            "type": "int",
            "default": 200,
            "range": [50, 500],
            "group": "tabnet",
        },
        "tabnet_patience": {
            "type": "int",
            "default": 15,
            "range": [5, 50],
            "group": "tabnet",
        },
        "tabnet_batch_size": {
            "type": "categorical",
            "default": 1024,
            "range": [128, 256, 512, 1024, 2048, 4096],
            "group": "tabnet",
        },
        "tabnet_virtual_batch_size": {
            "type": "int",
            "default": 128,
            "range": [16, 1024],
            "group": "tabnet",
            "log_scale": True,
        },
        "tabnet_momentum": {
            "type": "float",
            "default": 0.02,
            "range": [0.005, 0.5],
            "group": "tabnet",
        },
        # Optimizer parameters
        "tabnet_optimizer_lr": {
            "type": "float",
            "default": 0.02,
            "range": [1e-5, 0.5],
            "group": "tabnet",
            "log_scale": True,
        },
    }
)

# LSTM hyperparameters
HYPERPARAMETER_REGISTRY.update(
    {
        "lstm_units_1": {
            "type": "int",
            "default": 128,
            "range": [16, 512],
            "group": "lstm",
            "log_scale": True,
        },
        "lstm_units_2": {
            "type": "int",
            "default": 64,
            "range": [16, 256],
            "group": "lstm",
            "log_scale": True,
        },
        "lstm_units_3": {
            "type": "int",
            "default": 32,
            "range": [16, 128],
            "group": "lstm",
        },
        "lstm_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-5, 1e-2],
            "group": "lstm",
            "log_scale": True,
        },
        "lstm_dropout": {
            "type": "float",
            "default": 0.2,
            "range": [0.0, 0.5],
            "group": "lstm",
        },
        "lstm_batch_norm": {
            "type": "bool",
            "default": True,
            "range": None,
            "group": "lstm",
        },
        "lstm_attention": {
            "type": "bool",
            "default": True,
            "range": None,
            "group": "lstm",
        },
    }
)

# TFT hyperparameters
HYPERPARAMETER_REGISTRY.update(
    {
        "tft_hidden_size": {
            "type": "int",
            "default": 64,
            "range": [32, 256],
            "group": "tft",
            "log_scale": True,
        },
        "tft_lstm_units": {
            "type": "int",
            "default": 128,
            "range": [64, 512],
            "group": "tft",
            "log_scale": True,
        },
        "tft_num_heads": {"type": "int", "default": 4, "range": [1, 8], "group": "tft"},
        "tft_dropout": {
            "type": "float",
            "default": 0.2,
            "range": [0.0, 0.5],
            "group": "tft",
        },
        "tft_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-5, 1e-2],
            "group": "tft",
            "log_scale": True,
        },
        "tft_max_positions": {
            "type": "int",
            "default": 1000,
            "range": [100, 5000],
            "group": "tft",
        },
        "tft_batch_size": {
            "type": "categorical",
            "default": 64,
            "range": [16, 32, 64, 128, 256],
            "group": "tft",
        },
        "tft_epochs": {
            "type": "int",
            "default": 50,
            "range": [10, 200],
            "group": "tft",
        },
    }
)
