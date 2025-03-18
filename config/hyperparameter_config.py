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


def suggest_param(trial, name, override_range=None):
    """
    Suggest a parameter value using an Optuna trial.
    Utilizes the HYPERPARAMETER_REGISTRY.
    """
    if name not in HYPERPARAMETER_REGISTRY:
        raise KeyError(f"Parameter '{name}' not registered")
    config = HYPERPARAMETER_REGISTRY[name]
    param_type = config["type"]
    param_range = override_range or config.get("range")
    use_log = config.get("log_scale", False)
    if param_type == "int":
        if param_range:
            return trial.suggest_int(name, param_range[0], param_range[1], log=use_log)
        else:
            return config["default"]
    elif param_type == "float":
        if param_range:
            return trial.suggest_float(name, param_range[0], param_range[1], log=use_log)
        else:
            return config["default"]
    elif param_type == "categorical":
        if param_range:
            return trial.suggest_categorical(name, param_range)
        else:
            return config["default"]
    elif param_type == "bool":
        return trial.suggest_categorical(name, [True, False])
    return config["default"]


# Pruning settings
PRUNING_ENABLED = register_param("pruning_enabled", True, "bool", None, "pruning")
PRUNING_MEDIAN_FACTOR = register_param(
    "pruning_median_factor", 1.9, "float", [1.1, 5.0], "pruning"
)
PRUNING_MIN_TRIALS = register_param(
    "pruning_min_trials", 10, "int", [5, 1000], "pruning"
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
    "TOTAL_STEPS": {"min": 1, "max": 1000},  # Steps for splitting evaluation
    "N_STARTUP_TRIALS": {
        "min": 5,
        "max": 10000,
        "default": N_STARTUP_TRIALS,
    },  # Trials before pruning starts
    "N_WARMUP_STEPS": {"min": 500, "max": 5000},  # Steps before pruning enabled
    "INTERVAL_STEPS": {"min": 1, "max": 500},  # Interval between pruning checks
    "PRUNING_PERCENTILE": {"min": 1, "max": 90},  # Percentile for pruning
    "META_TUNING_ENABLED": True,  # Toggle for meta-tuning
    "TRIALS_PER_CYCLE": {
        "min": TUNING_TRIALS_PER_CYCLE_min,
        "max": TUNING_TRIALS_PER_CYCLE_max,
    },  # Min/max trials per cycle from config_loader
}

# Define model types
MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost", "tft", "ltc", "tabnet", "nbeats", "cnn"]

# Set search method to Optuna only
HYPERPARAM_SEARCH_METHOD = "optuna"

# CNN hyperparameters
HYPERPARAMETER_REGISTRY.update(
    {
        "cnn_num_conv_layers": {
            "type": "int",
            "default": 3,
            "range": [1, 10],
            "group": "cnn",
        },
        "cnn_num_filters": {
            "type": "int",
            "default": 64,
            "range": [16, 2048],
            "group": "cnn",
            "log_scale": True,
        },
        "cnn_kernel_size": {
            "type": "int",
            "default": 3,
            "range": [2, 14],
            "group": "cnn",
        },
        "cnn_stride": {
            "type": "int",
            "default": 1,
            "range": [1, 6],
            "group": "cnn",
        },
        "cnn_dropout_rate": {
            "type": "float",
            "default": 0.2,
            "range": [0.0, 0.8],
            "group": "cnn",
        },
        "cnn_activation": {
            "type": "categorical",
            "default": "relu",
            "range": ["relu", "leaky_relu", "elu"],
            "group": "cnn",
        },
        "cnn_use_adaptive_pooling": {
            "type": "bool",
            "default": True,
            "range": [True, False],
            "group": "cnn",
        },
        "cnn_fc_layer_1": {
            "type": "int",
            "default": 128,
            "range": [32, 2048],
            "group": "cnn",
            "log_scale": True,
        },
        "cnn_fc_layer_2": {
            "type": "int",
            "default": 64,
            "range": [16, 2048],
            "group": "cnn",
            "log_scale": True,
        },
        "cnn_lookback": {
            "type": "int",
            "default": 30,
            "range": [7, 120],
            "group": "cnn",
        },
        "cnn_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-6, 1e-1],
            "group": "cnn",
            "log_scale": True,
        },
        "cnn_batch_size": {
            "type": "categorical",
            "default": 64,
            "range": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            "group": "cnn",
        },
        "cnn_early_stopping_patience": {
            "type": "int",
            "default": 5,
            "range": [3, 20],
            "group": "cnn",
        },
        "cnn_base_epochs": {
            "type": "int",
            "default": 5,
            "range": [1, 1000],
            "group": "cnn",
        },
    }
)

#LTC-specific hyperparameters to the registry
HYPERPARAMETER_REGISTRY.update(
    {
        "ltc_units": {
            "type": "int",
            "default": 256,
            "range": [32, 2048],
            "group": "ltc",
            "log_scale": True,
        },
        "ltc_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-6, 1e-1],
            "group": "ltc",
            "log_scale": True,
        },
        "ltc_lookback": {
            "type": "int",
            "default": 30,
            "range": [7, 120],
            "group": "ltc",
        },
        "ltc_epochs": {"type": "int", "default": 100, "range": [5, 5000], "group": "ltc"},
        "ltc_batch_size": {
            "type": "int",
            "default": 256,
            "range": [16, 4096],
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
            "default": 256,
            "range": [8, 2048],
            "group": "tabnet",
            "log_scale": True,
        },
        "n_a": {
            "type": "int",
            "default": 256,
            "range": [8, 2048],
            "group": "tabnet",
            "log_scale": True,
        },
        "n_steps": {"type": "int", "default": 8, "range": [1, 20], "group": "tabnet"},
        "gamma": {
            "type": "float",
            "default": 1.5,
            "range": [0.5, 3.5],
            "group": "tabnet",
        },
        "lambda_sparse": {
            "type": "float",
            "default": 0.0001,
            "range": [1e-7, 1e-1],
            "group": "tabnet",
            "log_scale": True,
        },
        # Training parameters
        "tabnet_max_epochs": {
            "type": "int",
            "default": 500,
            "range": [10, 5000],
            "group": "tabnet",
        },
        "tabnet_patience": {
            "type": "int",
            "default": 50,
            "range": [5, 500],
            "group": "tabnet",
        },
        "tabnet_batch_size": {
            "type": "categorical",
            "default": 1024,
            "range": [64,128, 256, 512, 1024, 2048, 4096],
            "group": "tabnet",
        },
        "tabnet_virtual_batch_size": {
            "type": "int",
            "default": 128,
            "range": [8, 512],
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
            "range": [1e-6, 0.1],
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
            "range": [32, 4096],
            "group": "lstm",
            "log_scale": True,
        },
        "lstm_units_2": {
            "type": "int",
            "default": 64,
            "range": [16, 2048],
            "group": "lstm",
            "log_scale": True,
        },
        "lstm_units_3": {
            "type": "int",
            "default": 32,
            "range": [8, 1048],
            "group": "lstm",
        },
        "lstm_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-6, 1e-1],
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
            "default": 256,
            "range": [32, 2048],
            "group": "tft",
            "log_scale": True,
        },
        "tft_lstm_units": {
            "type": "int",
            "default": 1024,
            "range": [64, 4096],
            "group": "tft",
            "log_scale": True,
        },
        "tft_num_heads": {"type": "int", "default": 6, "range": [2, 8], "group": "tft"},
        "tft_dropout": {
            "type": "float",
            "default": 0.2,
            "range": [0.0, 0.5],
            "group": "tft",
        },
        "tft_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-6, 1e-1],
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
            "range": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            "group": "tft",
        },
        "tft_epochs": {
            "type": "int",
            "default": 300,
            "range": [10, 5000],
            "group": "tft",
        },
    }
)
