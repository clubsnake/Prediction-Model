# config.py
"""
Global configuration and settings module.

This includes GPU settings, hyperparameter search options, 
and general toggles for various dashboard features.
"""

import datetime
import yaml

#######################
# GPU & general settings
#######################
USE_GPU = True
GPU_MEMORY_FRACTION = 0.9
OMP_NUM_THREADS = 8  # up to 8 threads

#########################
# Tuning method & Grid
#########################
HYPERPARAM_SEARCH_METHOD = "optuna"  # Options: "optuna", "grid", or "both"
# Select grid option: "normal", "thorough", "full"
GRID_SEARCH_TYPE = "normal"

NORMAL_GRID = {
    "epochs": [25, 50, 75],
    "batch_size": [16, 32],
    "learning_rate": [0.001, 0.0005],
    "lookback": [14, 30, 60],
    "dropout_rate": [0.1, 0.2, 0.3]
}
THOROUGH_GRID = {
    "epochs": [25, 50, 75, 100],
    "batch_size": [16, 32, 64],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "lookback": [14, 30, 60, 90],
    "dropout_rate": [0.1, 0.2, 0.3, 0.4]
}
FULL_GRID = {
    "epochs": [25, 50, 75, 100, 125],
    "batch_size": [16, 32, 64, 128],
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

MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost"]
ACTIVE_MODEL_TYPES = ["lstm", "rnn", "random_forest", "xgboost"]
LOSS_FUNCTIONS = ["mean_squared_error", "mean_absolute_error", "huber_loss"]

#########################
# Tuning configuration
#########################
N_STARTUP_TRIALS = 100  # Number of initial random (startup) trials
TUNING_TRIALS_PER_CYCLE_min = 20
TUNING_TRIALS_PER_CYCLE_max = 100
TUNING_LOOP = True  # If True, keep repeating cycles until threshold or stop.

# Desired thresholds – tuning will continue until these are met (or manual stop).
RMSE_THRESHOLD = 1500
MAPE_THRESHOLD = 5.0

AUTO_RUN_TUNING = False  # Start tuning automatically on startup

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
START_DATE = "2020-01-01"
INTERVAL = "1d"
LOOKBACK = 60
PREDICTION_HORIZON = 180

EPOCHS = 50
BATCH_SIZE = 16
ORIGINAL_LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
SHUFFLE = True

WALK_FORWARD_ON = True
WALK_FORWARD_DEFAULT = 30
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
USE_RSI = True
USE_MACD = True
USE_BOLLINGER_BANDS = True
USE_ATR = True
USE_OBV = True
USE_WERPI = True
USE_SENTIMENT = False  # Example placeholder

BASE_FEATURES = ["Open", "High", "Low", "Volume"]

# Feature Engineering Defaults
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLL_WINDOW = 20
BOLL_NSTD = 2.0
ATR_PERIOD = 14
WERPI_WAVELET = "db4"
WERPI_LEVEL = 3
WERPI_N_STATES = 2
WERPI_SCALE = 1.0
APPLY_WEEKEND_GAP = False

#####################
# Additional toggles
#####################
USE_CACHING = True  # Enable @st.cache_data or caching in data fetch
PROGRESSIVE_LOADING = False  # Placeholder for chunked data loading logic
ENABLE_EMAIL_ALERTS = False  # For future email notifications

def get_active_feature_names():
    """
    Return the active feature names (core features + any technical indicators).
    This is simplified, but can be expanded to read from toggles above.
    """
    return ["Open", "High", "Low", "Close", "Volume"]

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
