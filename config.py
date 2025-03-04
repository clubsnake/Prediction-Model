# config.py
"""
Global configuration and settings module.

This includes GPU settings, hyperparameter search options, 
and general toggles for various dashboard features.
"""

import datetime
import yaml
import random
import numpy as np

# Hyperparameter Registry System
HYPERPARAMETER_REGISTRY = {}

def register_param(name, default, type_hint="categorical", range_values=None, group=None):
    """
    Register a parameter for potential hyperparameter tuning.
    
    Args:
        name: Parameter name
        default: Default value
        type_hint: Type of parameter ("int", "float", "categorical", "bool")
        range_values: Possible values or [min, max] for numerical types
        group: Optional group name for parameter organization
        
    Returns:
        The default value for immediate use
    """
    HYPERPARAMETER_REGISTRY[name] = {
        "default": default,
        "type": type_hint,
        "range": range_values,
        "group": group or "misc"
    }
    return default

# Set random seed for reproducibility
RANDOM_STATE = register_param("random_state", 42, "int", [0, 100], "system")

#######################
# GPU & general settings
#######################
USE_GPU = True
GPU_MEMORY_FRACTION = register_param("gpu_memory_fraction", 0.6, "float", [0.1, 1.0], "system")
OMP_NUM_THREADS = register_param("omp_num_threads", 4, "int", [1, 32], "system")
USE_XLA = register_param("use_xla", True, "bool", None, "system")
USE_MIXED_PRECISION = register_param("use_mixed_precision", False, "bool", None, "system")
USE_DIRECTML = True  # Use DirectML for AMD GPUs if available
TF_ENABLE_ONEDNN_OPTS = True  # Enable Intel MKL-DNN optimizations if available
FINNHUB_API_KEY = "cv2lrk1r01qhefsl3gm0cv2lrk1r01qhefsl3gmg"  
ALPHAVANTAGE_API_KEY = "4S361Y7KXRYY914J"

#########################
# Tuning method & Grid
#########################
HYPERPARAM_SEARCH_METHOD = "optuna"  # Options: "optuna", "grid", or "both"
# Select grid option: "normal", "thorough", "full"
GRID_SEARCH_TYPE = "normal"

NORMAL_GRID = {
    "epochs": [25, 50, 75],
    "batch_size": [64, 128, 256],
    "learning_rate": [0.001, 0.0005],
    "lookback": [14, 30, 60],
    "dropout_rate": [0.1, 0.2, 0.3]
}
THOROUGH_GRID = {
    "epochs": [25, 50, 75, 100],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "lookback": [14, 30, 60, 90],
    "dropout_rate": [0.1, 0.2, 0.3, 0.4]
}
FULL_GRID = {
    "epochs": [25, 50, 75, 100, 125],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "learning_rate": [0.001, 0.0005, 0.0001, 0.00005],
    "lookback": [14, 30, 60, 90, 120],
    "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5]
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


###########################
# Model & training settings
###########################

#Model Types
MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost", "tft", "nbeats", "ltc"]
ACTIVE_MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost","tft", "nbeats", "ltc"]
LOSS_FUNCTIONS = ["mean_squared_error", "mean_absolute_error", "huber_loss"]


# TFT hyperparameter grid (tune these ranges as needed)
TFT_GRID = {
    "hidden_size": [32, 64, 128, 256, 512],
    "lstm_units": [32, 64, 128, 256, 512],
    "num_heads": [2, 4, 8, 16, 32],
    "lr": [0.001, 0.0005, 0.0001, 0.00005],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "epochs": [10, 100, 250, 500, 1000, 2500, 5000],
    "batch_size": [32, 64, 128, 256, 512, 1024],
    "lookback": [1, 2, 3, 5, 7, 14, 30, 60, 90, 120],
}

# Add LTC-specific hyperparameters to the registry
HYPERPARAMETER_REGISTRY.update({
    "ltc_units": {
        "type": "int", 
        "default": 64,
        "range": [32, 512],
        "group": "ltc"
    },
    "ltc_lr": {
        "type": "float",
        "default": 0.001,
        "range": [1e-5, 1e-2],
        "group": "ltc"
    },
    "ltc_lookback": {
        "type": "int",
        "default": 30,
        "range": [7, 90],
        "group": "ltc"
    },
    "ltc_epochs": {
        "type": "int",
        "default": 25,
        "range": [5, 100],
        "group": "ltc"
    },
    "ltc_batch_size": {
        "type": "int",
        "default": 32,
        "range": [16, 128],
        "group": "ltc"
    },
    "ltc_loss": {
        "type": "categorical",
        "default": "mean_squared_error",
        "range": ["mean_squared_error", "mean_absolute_error", "huber_loss"],
        "group": "ltc"
    }
})

#########################
# Tuning configuration
#########################
N_STARTUP_TRIALS = 5000 # Number of initial random (startup) trials
TUNING_TRIALS_PER_CYCLE_min = 10
TUNING_TRIALS_PER_CYCLE_max = 5000
TUNING_LOOP = True  # If True, keep repeating cycles until threshold or stop.
 

  # Pruning settings
PRUNING_ENABLED = register_param("pruning_enabled", True, "bool", None, "pruning")
PRUNING_MEDIAN_FACTOR = register_param("pruning_median_factor", 1.9, "float", [1.1, 5.0], "pruning")
PRUNING_MIN_TRIALS = register_param("pruning_min_trials", 10, "int", [5, 100], "pruning")  
PRUNING_ABSOLUTE_RMSE_FACTOR = register_param("pruning_absolute_rmse_factor", 2.0, "float", [1.1, 5.0], "pruning")
PRUNING_ABSOLUTE_MAPE_FACTOR = register_param("pruning_absolute_mape_factor", 3.0, "float", [1.1, 10.0], "pruning")

# Desired thresholds – tuning will continue until these are met (or manual stop).
RMSE_THRESHOLD = 1500
MAPE_THRESHOLD = 5.0

AUTO_RUN_TUNING = False  # Start tuning automatically on startup

# Meta-tuning parameters with min/max ranges
META_TUNING_PARAMS = {
    "TOTAL_STEPS": {"min": 1, "max": 500},     # Steps for splitting evaluation
    "N_STARTUP_TRIALS": {"min": 5, "max": 500}, # Trials before pruning starts
    "N_WARMUP_STEPS": {"min": 5, "max": 500},    # Steps before pruning enabled
    "INTERVAL_STEPS": {"min": 1, "max": 50},    # Interval between pruning checks
    "PRUNING_PERCENTILE": {"min": 1, "max": 90}, # Percentile for pruning
    "META_TUNING_ENABLED": True                              # Toggle for meta-tuning
}


########################
# Time-series parameters
########################
TICKER = "ETH-USD"
TICKERS = [
    "ETH-USD", "BTC-USD", "LINK-USD", "LTC-USD",
    "AAVE-USD", "AVAX-USD", "SOL-USD", "MATIC-USD", "RVN-USD",
      "GOOGL", "T", "AMZN", "NVDA", "MSFT", "META",
      "STLA", "AMD", "INTC", "SQ", "NET", "ZM", "ROKU", "AAPL", "TSLA", "NFLX",
      "SPY", "ARK", "KBWB", "XLC"
]

TIMEFRAMES = [ "1d", "3d", "1wk", "1h", "2h", "4h", "6h", "8h", "12h", "1mo","1m", "5m", "15m", "30m"]
START_DATE = "2024-01-01"
INTERVAL = "1d"
LOOKBACK = register_param("lookback", 30, "int", [5, 120], "time_series")
PREDICTION_HORIZON = register_param("prediction_horizon", 30, "int", [1, 90], "time_series")

EPOCHS = register_param("epochs", 50, "int", [10, 500], "training")
BATCH_SIZE = register_param("batch_size", 16, "int", [8, 512], "training")
LEARNING_RATE = register_param("learning_rate", 0.001, "float", [1e-5, 1e-2], "training")
VALIDATION_SPLIT = register_param("validation_split", 0.2, "float", [0.1, 0.4], "training")
SHUFFLE = True

WALK_FORWARD_ON = register_param("walk_forward_on", True, "bool", None, "walk_forward")
WALK_FORWARD_DEFAULT = register_param("walk_forward_default", 30, "int", [3, 180], "walk_forward")
WALK_FORWARD_MIN = 3
WALK_FORWARD_MAX = 180

CONTINUE_FROM_WEIGHTS = False
CLEANUP_OLD_LOGS = True

#########################
# Real-time / Dashboard
#########################
REALTIME_UPDATE = True
AUTOREFRESH_INTERVAL_SEC = 60
SHOW_PREDICTION_PLOTS = True
SHOW_PREDICTION_PLOTS_INTERVAL = 5
SHOW_TRAINING_HISTORY = True
SHOW_WEIGHT_HISTOGRAMS = True

#########################
# Feature toggles
#########################
USE_RSI = register_param("use_rsi", True, "bool", None, "features")
USE_MACD = register_param("use_macd", True, "bool", None, "features")
USE_BOLLINGER_BANDS = register_param("use_bollinger_bands", True, "bool", None, "features")
USE_ATR = register_param("use_atr", True, "bool", None, "features")
USE_OBV = register_param("use_obv", True, "bool", None, "features")
USE_WERPI = register_param("use_werpi", True, "bool", None, "features")
USE_SENTIMENT = register_param("use_sentiment", False, "bool", None, "features")

BASE_FEATURES = ["Open", "High", "Low", "Volume"]

# Feature Engineering Defaults
RSI_PERIOD = register_param("rsi_period", 14, "int", [5, 60], "technical_indicators")
MACD_FAST = register_param("macd_fast", 12, "int", [5, 20], "technical_indicators")
MACD_SLOW = register_param("macd_slow", 26, "int", [15, 50], "technical_indicators")
MACD_SIGNAL = register_param("macd_signal", 9, "int", [5, 20], "technical_indicators")
BOLL_WINDOW = register_param("boll_window", 20, "int", [10, 50], "technical_indicators")
BOLL_NSTD = register_param("boll_nstd", 2.0, "float", [0.5, 4.0], "technical_indicators")
ATR_PERIOD = register_param("atr_period", 14, "int", [7, 30], "technical_indicators")
WERPI_WAVELET = register_param("werpi_wavelet", "db4", "categorical", ["haar", "db1", "db4", "sym5"], "technical_indicators")
WERPI_LEVEL = register_param("werpi_level", 3, "int", [1, 6], "technical_indicators")
WERPI_N_STATES = register_param("werpi_n_states", 2, "int", [2, 10], "technical_indicators")
WERPI_SCALE = register_param("werpi_scale", 1.0, "float", [0.1, 10.0], "technical_indicators")

#####################
# Additional toggles
#####################
USE_CACHING = True  # Enable @st.cache_data or caching in data fetch
PROGRESSIVE_LOADING = False  # Placeholder for chunked data loading logic
ENABLE_EMAIL_ALERTS = False  # For future email notifications


# VMLI indicator parameters
USE_VMLI = register_param("use_vmli", True, "bool", None, "indicators")
VMLI_WINDOW_MOM = register_param("vmli_window_mom", 14, "int", [1, 120], "indicators")
VMLI_WINDOW_VOL = register_param("vmli_window_vol", 14, "int", [1, 120], "indicators")
VMLI_SMOOTH_PERIOD = register_param("vmli_smooth_period", 3, "int", [1, 16], "indicators")
VMLI_WINSORIZE_PCT = register_param("vmli_winsorize_pct", 0.01, "float", [0.001, 0.1], "indicators")
VMLI_USE_EMA = register_param("vmli_use_ema", True, "bool", None, "indicators")

# Add indicator tuning parameters
INDICATOR_TUNING_SETTINGS = {
    'n_trials': 50,
    'eval_method': 'returns',  # 'returns', 'correlation', or 'signal_quality'
    'werpi': {
        'upper_threshold': 70,
        'lower_threshold': 30
    },
    'vmli': {
        'buy_threshold': 0.8,
        'sell_threshold': -0.8
    }
}

# Load tuned parameters if available
def load_tuned_indicator_params():
    """Load Optuna-tuned indicator parameters from file"""
    import os
    import json
    
    tuned_params_file = os.path.join("Data", "tuned_indicators.json")
    if os.path.exists(tuned_params_file):
        try:
            with open(tuned_params_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading tuned parameters: {e}")
    return None

# Make tuned parameters available
TUNED_INDICATOR_PARAMS = load_tuned_indicator_params()

def get_active_feature_names():
    """
    Return the active feature names (core features + any technical indicators).
    This is simplified, but can be expanded to read from toggles above.
    """
    return ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "ATR", "OBV", "WERPI"]    

def get_horizon_for_category(range_cat: str) -> int:
    """
    Map textual horizon categories (e.g. "1w", "1m") to integer horizon days.
    """
    if range_cat.lower() == "1w":
        return 7
    elif range_cat.lower() == "2w":
        return 14
    elif range_cat.lower() == "1m":
        return 30
    elif range_cat.lower() == "all":
        return 30
    return 30

def load_user_config():
    """
    Load user overrides from a YAML file if present.
    """
    try:
        with open('user_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

