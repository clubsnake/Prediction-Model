# hyperparameter_config.py
"""
Hyperparameter configuration and settings module.

This includes hyperparameter search options and specific tuning parameters.
"""

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
        **kwargs
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

# Meta-tuning parameters with min/max ranges
META_TUNING_PARAMS = {
    "TOTAL_STEPS": {"min": 1, "max": 500},  # Steps for splitting evaluation
    "N_STARTUP_TRIALS": {"min": 5, "max": 500},  # Trials before pruning starts
    "N_WARMUP_STEPS": {"min": 5, "max": 500},  # Steps before pruning enabled
    "INTERVAL_STEPS": {"min": 1, "max": 50},  # Interval between pruning checks
    "PRUNING_PERCENTILE": {"min": 1, "max": 90},  # Percentile for pruning
    "META_TUNING_ENABLED": True,  # Toggle for meta-tuning
}

# Define model types
MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost", "tft", "ltc", "tabnet"]

# Grid search configuration
HYPERPARAM_SEARCH_METHOD = "optuna"  # Options: "optuna", "grid", or "both"
# Select grid option: "normal", "thorough", "full"
GRID_SEARCH_TYPE = "normal"

NORMAL_GRID = {
    "epochs": [25, 50, 75],
    "batch_size": [64, 128, 256],
    "learning_rate": [0.001, 0.0005],
    "lookback": [14, 30, 60],
    "dropout_rate": [0.1, 0.2, 0.3],
}
THOROUGH_GRID = {
    "epochs": [25, 50, 75, 100],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "lookback": [14, 30, 60, 90],
    "dropout_rate": [0.1, 0.2, 0.3, 0.4],
}
FULL_GRID = {
    "epochs": [25, 50, 75, 100, 125],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "learning_rate": [0.001, 0.0005, 0.0001, 0.00005],
    "lookback": [14, 30, 60, 90, 120],
    "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
}


def get_hyperparameter_ranges():
    """
    Return hyperparameter ranges based on GRID_SEARCH_TYPE setting.
    """
    grid_type = GRID_SEARCH_TYPE.lower()
    if grid_type == "normal":
        return NORMAL_GRID
    elif grid_type == "thorough":
        return THOROUGH_GRID
    elif grid_type == "full":
        return FULL_GRID
    else:
        return NORMAL_GRID


# Add LTC-specific hyperparameters to the registry
HYPERPARAMETER_REGISTRY.update(
    {
        "ltc_units": {"type": "int", "default": 64, "range": [32, 512], "group": "ltc", "log_scale": True},
        "ltc_lr": {
            "type": "float",
            "default": 0.001,
            "range": [1e-5, 1e-2],
            "group": "ltc",
            "log_scale": True
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
HYPERPARAMETER_REGISTRY.update({
    # Architecture parameters
    "n_d": {
        "type": "int", 
        "default": 64, 
        "range": [8, 256], 
        "group": "tabnet",
        "log_scale": True
    },
    "n_a": {
        "type": "int", 
        "default": 64, 
        "range": [8, 256], 
        "group": "tabnet",
        "log_scale": True
    },
    "n_steps": {
        "type": "int", 
        "default": 5, 
        "range": [1, 15], 
        "group": "tabnet"
    },
    "gamma": {
        "type": "float", 
        "default": 1.5, 
        "range": [0.5, 3.0], 
        "group": "tabnet"
    },
    "lambda_sparse": {
        "type": "float", 
        "default": 0.001, 
        "range": [1e-7, 1e-1], 
        "group": "tabnet",
        "log_scale": True
    },
    
    # Training parameters
    "tabnet_max_epochs": {
        "type": "int", 
        "default": 200, 
        "range": [50, 500], 
        "group": "tabnet"
    },
    "tabnet_patience": {
        "type": "int", 
        "default": 15, 
        "range": [5, 50], 
        "group": "tabnet"
    },
    "tabnet_batch_size": {
        "type": "categorical", 
        "default": 1024, 
        "range": [128, 256, 512, 1024, 2048, 4096], 
        "group": "tabnet"
    },
    "tabnet_virtual_batch_size": {
        "type": "int", 
        "default": 128, 
        "range": [16, 1024], 
        "group": "tabnet",
        "log_scale": True
    },
    "tabnet_momentum": {
        "type": "float", 
        "default": 0.02, 
        "range": [0.005, 0.5], 
        "group": "tabnet"
    },
    
    # Optimizer parameters
    "tabnet_optimizer_lr": {
        "type": "float", 
        "default": 0.02, 
        "range": [1e-5, 0.5], 
        "group": "tabnet",
        "log_scale": True
    }
})

# LSTM hyperparameters
HYPERPARAMETER_REGISTRY.update({
    "lstm_units_1": {
        "type": "int",
        "default": 128,
        "range": [16, 512],
        "group": "lstm",
        "log_scale": True
    },
    "lstm_units_2": {
        "type": "int",
        "default": 64,
        "range": [16, 256],
        "group": "lstm",
        "log_scale": True
    },
    "lstm_units_3": {
        "type": "int",
        "default": 32,
        "range": [16, 128],
        "group": "lstm"
    },
    "lstm_lr": {
        "type": "float",
        "default": 0.001,
        "range": [1e-5, 1e-2],
        "group": "lstm",
        "log_scale": True
    },
    "lstm_dropout": {
        "type": "float",
        "default": 0.2,
        "range": [0.0, 0.5],
        "group": "lstm"
    },
    "lstm_batch_norm": {
        "type": "bool",
        "default": True,
        "range": None,
        "group": "lstm"
    },
    "lstm_attention": {
        "type": "bool",
        "default": True,
        "range": None,
        "group": "lstm"
    }
})
