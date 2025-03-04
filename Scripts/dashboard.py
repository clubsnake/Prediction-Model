"""
unified_dashboard.py

A comprehensive dashboard that combines the best features from the original dashboard,
advanced_dashboard, and enhanced_dashboard into a single, maintainable solution.

Key features:
- Modern UI with tabs, metrics, and interactive charts
- Real-time model tuning controls
- Technical indicators and WERPI visualization
- Model architecture and performance visualization
- Prediction explorer with error analysis
- Comparative model analysis
- Feature importance explorer
- Data and model export functionality
"""

import os
import sys
import time
import threading
import traceback
from datetime import datetime, timedelta
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yaml
import json
import io
import base64
import tempfile


# Configure exception handling before anything else
def global_exception_handler(exctype, value, tb):
    """Global exception handler to catch and log unhandled exceptions"""
    logger.error(f"Unhandled exception: {value}", exc_info=(exctype, value, tb))
    if 'error_log' in st.session_state:
        st.session_state['error_log'].append({
            'timestamp': datetime.now(),
            'function': 'global',
            'error': str(value),
            'traceback': traceback.format_exc()
        })
    # Call the original exception handler
    sys.__excepthook__(exctype, value, tb)

# Set the global exception handler
sys.excepthook = global_exception_handler

# Create directories safely
def safe_mkdir(directory):
    """Create directory safely with error handling"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False

# Set up directory structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
DB_DIR = os.path.join(DATA_DIR, "DB")
LOGS_DIR = os.path.join(DATA_DIR, "Logs")
MODELS_DIR = os.path.join(DATA_DIR, "Models")
HYPERPARAMS_DIR = os.path.join(DATA_DIR, "Hyperparameters")

# Create all necessary directories
for directory in [DATA_DIR, DB_DIR, LOGS_DIR, MODELS_DIR, HYPERPARAMS_DIR]:
    safe_mkdir(directory)

# Append paths for modules
sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)

# Configure logger
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR, "dashboard.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("unified_dashboard")
except Exception as e:
    print(f"Error configuring logger: {e}")
    # Fallback logger
    logger = logging.getLogger("unified_dashboard")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

# File paths
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")
CYCLE_METRICS_FILE = os.path.join(DATA_DIR, "cycle_metrics.yaml")
BEST_PARAMS_FILE = os.path.join(DATA_DIR, "best_params.yaml")

# Import modules with robust error handling
def safe_import(module_name):
    """Safely import a module with fallback mechanisms"""
    try:
        return __import__(module_name)
    except ImportError:
        logger.warning(f"Module {module_name} not found. Some functionality may be limited.")
        return None
    except Exception as e:
        logger.error(f"Error importing {module_name}: {e}")
        return None

# Set up Streamlit cache function based on available version
def get_cache_function():
    """Get appropriate Streamlit cache function based on available version"""
    try:
        if hasattr(st, "cache_data"):
            return st.cache_data
        elif hasattr(st, "cache_resource"):
            return st.cache_resource
        else:
            return st.cache
    except Exception as e:
        logger.error(f"Error setting up cache function: {e}")
        return lambda f: f  # Return identity function as fallback

# Set up memory management functions
def clean_memory(force_gc=False):
    """Clean up memory to avoid out-of-memory issues"""
    try:
        import gc
        if force_gc:
            gc.collect()
    except Exception as e:
        logger.error(f"Error cleaning memory: {e}")

# Enhanced error boundary with detailed logging and recovery
def robust_error_boundary(func):
    """Enhanced decorator for robust error handling in dashboard functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Store error in session state for history/debugging
            if 'error_log' not in st.session_state:
                st.session_state['error_log'] = []
                
            st.session_state['error_log'].append({
                'timestamp': datetime.now(),
                'function': func.__name__,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Display error to user
            try:
                st.error(error_msg)
                st.warning("Attempting to continue dashboard operation...")
            except:
                # If we can't even show the error (might happen during Streamlit reloading)
                pass
                
            return None
    return wrapper

@robust_error_boundary
def init_session_state():
    """Initialize session state variables with defensive checks"""
    # Check if already initialized to avoid resetting values
    if "initialized" in st.session_state:
        return

# Try to import project modules with error handling
try:
    # Import resource configuration first
    resource_config = safe_import("resource_config")
    
    # Try to import remaining modules
    modules = {
        "meta_tuning": safe_import("meta_tuning"),
        "config": safe_import("config"),
        "data": safe_import("data"),
        "features": safe_import("features"),
        "model_visualization": safe_import("model_visualization"),
        "visualization": safe_import("visualization"),
        "preprocessing": safe_import("preprocessing"),
        "utils": safe_import("utils"),
        "threadsafe": safe_import("Scripts.threadsafe"),
        "model": safe_import("Scripts.model"),
        "dashboard_shutdown": safe_import("Scripts.dashboard_shutdown")
    }
    
    # Set defaults from config if available
    if modules["config"]:
        try:
            TICKERS = modules["config"].TICKERS
            TIMEFRAMES = modules["config"].TIMEFRAMES
            START_DATE = modules["config"].START_DATE
            RMSE_THRESHOLD = modules["config"].RMSE_THRESHOLD
            MAPE_THRESHOLD = modules["config"].MAPE_THRESHOLD
            N_STARTUP_TRIALS = modules["config"].N_STARTUP_TRIALS
            TUNING_TRIALS_PER_CYCLE_max = modules["config"].TUNING_TRIALS_PER_CYCLE_max
            TICKER = modules["config"].TICKER
        except AttributeError as e:
            logger.error(f"Error accessing config attributes: {e}")
            # Set fallback defaults
            TICKERS = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "GOOGL"]
            TIMEFRAMES = ["1d", "1h", "4h"]
            START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            RMSE_THRESHOLD = 100
            MAPE_THRESHOLD = 10
            N_STARTUP_TRIALS = 2000
            TUNING_TRIALS_PER_CYCLE_max = 1000
            TICKER = "ETH-USD"
    else:
        # Set fallback defaults if config not available
        TICKERS = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "GOOGL"]
        TIMEFRAMES = ["1d", "1h", "4h"]
        START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        RMSE_THRESHOLD = 100
        MAPE_THRESHOLD = 10
        N_STARTUP_TRIALS = 2000
        TUNING_TRIALS_PER_CYCLE_max = 1000
        TICKER = "ETH-USD"
        
except Exception as e:
    logger.error(f"Error during module imports: {e}", exc_info=True)
    st.error(f"Critical error during initialization: {e}")

@robust_error_boundary
def display_system_diagnostics():
    """Display system diagnostics and error monitoring"""
    with st.expander("System Diagnostics", expanded=False):
        # Display basic system info
        st.subheader("System Information")
        
        try:
            import platform
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Python Version: {platform.python_version()}")
                st.write(f"Operating System: {platform.system()} {platform.release()}")
                st.write(f"Streamlit Version: {st.__version__}")
            
            with col2:
                # Memory info - try to get but don't fail if not available
                try:
                    import psutil
                    st.write(f"Memory Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
                    st.write(f"CPU Cores: {psutil.cpu_count()}")
                except ImportError:
                    st.write("Memory info: psutil not available")
        except Exception as e:
            st.error(f"Error getting system info: {e}")
            
        # Display module status
        st.subheader("Module Status")
        import_status = {name: mod is not None for name, mod in modules.items()}
        st.table(pd.DataFrame({"Module": list(import_status.keys()), 
                              "Loaded": list(import_status.values())}))
        
        # Path verification
        st.subheader("Path Verification")
        paths = {
            "DATA_DIR": os.path.exists(DATA_DIR),
            "DB_DIR": os.path.exists(DB_DIR),
            "LOGS_DIR": os.path.exists(LOGS_DIR),
            "MODELS_DIR": os.path.exists(MODELS_DIR),
            "HYPERPARAMS_DIR": os.path.exists(HYPERPARAMS_DIR)
        }
        st.table(pd.DataFrame({"Path": list(paths.keys()), 
                              "Exists": list(paths.values())}))

# Get cache functions
cache_function = get_cache_function()

# For model loading specifically
if hasattr(st, "cache_resource"):
    cache_model = st.cache_resource
else:
    cache_model = st.cache(allow_output_mutation=True)

###############################################################################
#                           MAIN DASHBOARD FUNCTION                           #
###############################################################################:
    
def debug_imports():
    """Debug module import status"""
    import_status = {}
    for module_name, module_obj in modules.items():
        import_status[module_name] = module_obj is not None
    
    # Display import status as a table
    st.sidebar.expander("Module Import Status", expanded=False).table(
        pd.DataFrame({"Module": import_status.keys(), "Loaded": import_status.values()})
    )
    
    # Log detailed import status
    logger.info(f"Module import status: {import_status}")

@robust_error_boundary
def handle_auto_refresh():
    """More efficient auto-refresh with memory management"""
    if 'auto_refresh' not in st.session_state:
        st.session_state['auto_refresh'] = False
        
    if not st.session_state.get('auto_refresh', False):
        return False

    now = time.time()

    # Get refresh intervals with defaults
    progress_refresh_sec = st.session_state.get('progress_refresh_sec', 5)
    full_refresh_sec = st.session_state.get('full_refresh_sec', 30)

    # Get last refresh times with defaults
    last_progress_refresh = st.session_state.get('last_progress_refresh', 0)
    last_full_refresh = st.session_state.get('last_full_refresh', 0)

    # Check if progress refresh is needed
    time_since_progress = now - last_progress_refresh
    if time_since_progress >= progress_refresh_sec:
        st.session_state['last_progress_refresh'] = now

        # Clean up memory before updating display
        clean_memory(force_gc=False)

        # Update only the progress components
        update_progress_display()

    # Check if full refresh is needed
    time_since_full = now - last_full_refresh
    if time_since_full >= full_refresh_sec:
        st.session_state['last_full_refresh'] = now

        # Clean up memory before full refresh
        clean_memory(force_gc=True)

        return True  # Signal for full refresh

    return False  # No full refresh needed

@robust_error_boundary
def generate_future_forecast(model, df, feature_cols, lookback=30, horizon=30):
    """Generate predictions for the next 'horizon' days into the future"""
    try:
        if model is None or df is None or df.empty:
            return []
            
        # Use values from session state if not provided
        lookback = lookback or st.session_state.get("lookback", 30)
        horizon = horizon or st.session_state.get("forecast_window", 30)
        
        # Get the last 'lookback' days of data for input
        last_data = df.iloc[-lookback:].copy()
        
        # Create a scaler for feature normalization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(last_data[feature_cols])
        
        # Initialize array to store predictions
        future_prices = []
        current_data = last_data.copy()
        
        # Create sequences for prediction
        try:
            from preprocessing import create_sequences
            X_input, _ = create_sequences(current_data, feature_cols, "Close", lookback, 1)
            
            # Make prediction for full horizon at once if model supports it
            if hasattr(model, 'predict') and callable(model.predict):
                preds = model.predict(X_input, verbose=0)
                if isinstance(preds, np.ndarray) and preds.shape[1] >= horizon:
                    # If model can predict full horizon at once
                    return preds[0, :horizon].tolist()
                
                # Otherwise, predict one day at a time
                next_price = float(preds[0][0])
                future_prices.append(next_price)
                
                # Continue with iterative prediction for remaining days
                for i in range(1, horizon):
                    # Update data with previous prediction
                    next_row = current_data.iloc[-1:].copy()
                    if isinstance(next_row.index[0], pd.Timestamp):
                        next_row.index = [next_row.index[0] + pd.Timedelta(days=1)]
                    else:
                        next_row.index = [next_row.index[0] + 1]
                    
                    next_row["Close"] = next_price
                    current_data = pd.concat([current_data.iloc[1:], next_row])
                    
                    # Rescale features
                    current_scaled = current_data.copy()
                    current_scaled[feature_cols] = scaler.transform(current_data[feature_cols])
                    
                    # Create new sequence and predict
                    X_input, _ = create_sequences(current_scaled, feature_cols, "Close", lookback, 1)
                    preds = model.predict(X_input, verbose=0)
                    next_price = float(preds[0][0])
                    future_prices.append(next_price)
            
            return future_prices
        except Exception as e:
            logger.error(f"Error in sequence prediction: {e}", exc_info=True)
            return []
            
    except Exception as e:
        logger.error(f"Error generating future forecast: {e}", exc_info=True)
        return []

@robust_error_boundary
def update_progress_display():
    """Update just the progress display without a full dashboard rerun"""
    # Create dedicated containers if not already set
    if 'progress_container' not in st.session_state:
        st.session_state['progress_container'] = st.empty()
        st.session_state['progress_bar'] = st.empty()
        st.session_state['metrics_container'] = st.empty()
        
    progress = load_latest_progress()
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get("total_trials", 0)
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)
    
    # Update progress bar and container
    if total_trials > 0:
        percent = int((current_trial / total_trials) * 100)
        st.session_state['progress_bar'].progress(current_trial / total_trials)
        st.session_state['progress_container'].markdown(
            f"### Trial Progress: {current_trial}/{total_trials} ({percent}%)"
        )
        
    with st.session_state['metrics_container'].container():
        cols = st.columns(3)
        with cols[0]:
            st.metric("Current Trial", current_trial)
        with cols[1]:
            st.metric("Current RMSE", f"{current_rmse:.4f}" if current_rmse is not None else "N/A")
        with cols[2]:
            st.metric("Current MAPE", f"{current_mape:.2f}%" if current_mape is not None else "N/A")

@robust_error_boundary
def display_tested_models():
    """Display models from the tested_models.yaml file"""
    if not os.path.exists(TESTED_MODELS_FILE):
        st.info("No tested models available yet.")
        return
        
    try:
        if modules["threadsafe"]:
            tested_models = modules["threadsafe"].safe_read_yaml(TESTED_MODELS_FILE, default=[])
        else:
            # Fallback if threadsafe module is not available
            with open(TESTED_MODELS_FILE, 'r') as f:
                tested_models = yaml.safe_load(f) or []
    except Exception as e:
        st.error(f"Error loading tested models: {e}")
        tested_models = []
        
    if tested_models:
        st.subheader("Tested Models")
        try:
            df_tested = pd.DataFrame(tested_models)
            if "trial_number" in df_tested.columns:
                df_tested.sort_values("trial_number", inplace=True)
            st.dataframe(df_tested)
        except Exception as e:
            st.error(f"Error displaying tested models: {e}")
            # Fallback display for tested models
            st.json(tested_models)
    else:
        st.info("No tested models available yet.")

@robust_error_boundary
def model_save_load_controls():
    """Controls for saving and loading models"""
    st.subheader("Model Save/Load Options")
    
    # Use session state with default to prevent KeyError
    if 'saved_model_dir' not in st.session_state:
        st.session_state['saved_model_dir'] = "saved_models"
        
    st.session_state['saved_model_dir'] = st.text_input(
        "Models Directory",
        value=st.session_state['saved_model_dir']
    )
    
    if 'continue_from_old_weights' not in st.session_state:
        st.session_state['continue_from_old_weights'] = False
        
    st.session_state['continue_from_old_weights'] = st.checkbox(
        "Continue from old weights if available?",
        value=st.session_state['continue_from_old_weights']
    )
    
    if st.button("Save Current Model"):
        model = st.session_state.get('current_model')
        if model is None:
            st.warning("No model in session to save.")
        else:
            save_dir = st.session_state['saved_model_dir']
            try:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(save_dir, f"model_{timestamp}")
                
                # Try to import tensorflow directly if needed
                try:
                    # First try using the model's save method
                    model.save(model_path)
                    st.success(f"Model saved at: {model_path}")
                except AttributeError:
                    # If the model doesn't have a save method, try importing tensorflow
                    import tensorflow as tf
                    tf.keras.models.save_model(model, model_path)
                    st.success(f"Model saved at: {model_path}")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
                logger.error(f"Error saving model: {str(e)}", exc_info=True)
    
    st.write("### Load a Previously Saved Model")
    load_path = st.text_input("Path to model folder/file", key="load_model_path")
    if st.button("Load Model"):
        if not load_path:
            st.warning("Please enter a valid model path to load.")
            return
            
        if not os.path.exists(load_path):
            st.error(f"Path does not exist: {load_path}")
            return
            
        try:
            # Try to import tensorflow directly
            try:
                import tensorflow as tf
                loaded_model = tf.keras.models.load_model(load_path)
                st.session_state['current_model'] = loaded_model
                st.session_state['model_loaded'] = True
                st.success(f"Model loaded from: {load_path}")
            except ImportError:
                st.error("TensorFlow is not available. Cannot load model.")
        except Exception as e:
            st.error(f"Error loading model from {load_path}: {str(e)}")
            logger.error(f"Error loading model: {str(e)}", exc_info=True)

@robust_error_boundary
def start_tuning(ticker, timeframe):
    """Start the tuning process in a background thread"""
    # Check if meta_tuning module is available
    if not modules["meta_tuning"]:
        st.error("Meta tuning module not available. Cannot start tuning.")
        return
    
    # Reset all stop flags to ensure clean start
    try:
        modules["meta_tuning"].set_stop_requested(False)
    except Exception as e:
        st.error(f"Error resetting stop flags: {e}")
        return
    
    # Update all state tracking
    st.session_state["tuning_in_progress"] = True
    write_tuning_status(True)
    
    def tuning_thread_inner():
        try:
            # Set thread name for debugging
            threading.current_thread().name = f"Tuning-{ticker}-{timeframe}"
            logger.info(f"Starting tuning thread for {ticker} on {timeframe}")
            
            # Call the main tuning function
            modules["meta_tuning"].main()
        except Exception as e:
            logger.error(f"Error in tuning thread: {str(e)}", exc_info=True)
            if "error_log" not in st.session_state:
                st.session_state["error_log"] = []
            st.session_state["error_log"].append({
                'timestamp': datetime.now(),
                'function': 'tuning_thread',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        finally:
            # Always mark as finished when thread ends (for any reason)
            logger.info("Tuning thread finished")
            st.session_state['tuning_in_progress'] = False
            write_tuning_status(False)
    
    # Create and start the thread
    try:
        thread = threading.Thread(target=tuning_thread_inner)
        thread.daemon = True
        st.session_state["tuning_thread"] = thread
        thread.start()
        
        st.success(f"Tuning started for {ticker} on {timeframe}. Check logs for progress.")
    except Exception as e:
        st.error(f"Error starting tuning thread: {e}")
        logger.error(f"Error starting tuning thread: {e}", exc_info=True)
        st.session_state["tuning_in_progress"] = False
        write_tuning_status(False)

@robust_error_boundary
def stop_tuning():
    """Thread-safe way to stop tuning process"""
    if not modules["meta_tuning"]:
        st.error("Meta tuning module not available. Cannot stop tuning.")
        return
        
    try:
        # Use a single consistent method to request stop
        modules["meta_tuning"].set_stop_requested(True)
        
        # Update UI state
        st.warning("⚠️ Stop requested! Waiting for current trial to complete... This may take several minutes.")
        st.session_state['stop_request_time'] = datetime.now().strftime("%H:%M:%S")
        st.session_state['tuning_in_progress'] = False  # Mark as stopped in UI immediately
        
        # Update file-based flag
        write_tuning_status(False)
    except Exception as e:
        st.error(f"Error stopping tuning: {e}")
        logger.error(f"Error stopping tuning: {e}", exc_info=True)

@robust_error_boundary
def write_tuning_status(status: bool):
    """Write tuning status to file"""
    try:
        with open(TUNING_STATUS_FILE, "w") as f:
            f.write(str(status))
    except Exception as e:
        logger.error(f"Error writing tuning status: {e}")

@robust_error_boundary
def read_tuning_status() -> bool:
    """Read tuning status from file"""
    try:
        if os.path.exists(TUNING_STATUS_FILE):
            with open(TUNING_STATUS_FILE, "r") as f:
                status = f.read().strip()
                return status.lower() == "true"
        return False
    except Exception as e:
        logger.error(f"Error reading tuning status: {e}")
        return False

@robust_error_boundary
def load_latest_progress():
    """Get most up-to-date progress from either session state or file"""
    # First check session state (fastest)
    if 'live_progress' in st.session_state:
        return st.session_state['live_progress']
        
    # Fall back to file if no session state data or dashboard just loaded
    try:
        if os.path.exists(PROGRESS_FILE):
            if modules["threadsafe"]:
                return modules["threadsafe"].safe_read_yaml(PROGRESS_FILE, default={})
            else:
                # Fallback if threadsafe module is not available
                with open(PROGRESS_FILE, 'r') as f:
                    try:
                        return yaml.safe_load(f) or {}
                    except Exception:
                        return {}
        return {}
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
        return {}

@cache_function(ttl=600)  # Cache for 10 minutes
def load_data(ticker, start_date, end_date=None, interval="1d"):
    """Load market data with caching"""
    try:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if modules["data"] and hasattr(modules["data"], "fetch_data"):
            df = modules["data"].fetch_data(ticker, start=start_date, end=end_date, interval=interval)
        else:
            st.warning("Data module not available. Using placeholder data.")
            # Create placeholder data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            df = pd.DataFrame({
                'date': date_range,
                'Open': np.random.normal(100, 10, len(date_range)),
                'High': np.random.normal(105, 10, len(date_range)),
                'Low': np.random.normal(95, 10, len(date_range)),
                'Close': np.random.normal(100, 10, len(date_range)),
                'Volume': np.random.normal(1000000, 200000, len(date_range))
            })
            # Ensure values make sense (High > Open > Low, etc.)
            for i in range(len(df)):
                o, h, l = df.loc[i, ['Open', 'High', 'Low']]
                df.loc[i, 'High'] = max(o, h, l)
                df.loc[i, 'Low'] = min(o, h, l)
                df.loc[i, 'Close'] = np.random.uniform(df.loc[i, 'Low'], df.loc[i, 'High'])
                
        if df is not None and not df.empty:
            if modules["features"] and hasattr(modules["features"], "feature_engineering"):
                return modules["features"].feature_engineering(df)
            else:
                # Add basic features if feature_engineering not available
                return calculate_indicators(df)
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return None

@robust_error_boundary
def calculate_indicators(data):
    """Calculate technical indicators for the data"""
    if data is None or data.empty:
        return data
        
    try:
        # Ensure there's a Date column with consistent name
        if 'Date' in data.columns:
            data = data.rename(columns={'Date': 'date'})
        elif 'date' not in data.columns:
            # If neither exists, create from index if possible
            if isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index
            else:
                # Last resort: create a dummy date column
                data['date'] = pd.date_range(start='2020-01-01', periods=len(data))
        
        # Calculate moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Volume indicators
        if 'Volume' in data.columns:
            data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        
        return data
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        # Return the original data if calculation fails
        return data
    

@robust_error_boundary
def main_dashboard():
    """Main dashboard entry point with robust error handling"""
    try:
        # Set up the page
        set_page_config()
        
        # Initialize session state
        init_session_state()
        
        # Create header
        create_header()
        
        # Create control panel
        params = create_control_panel()
        
        # Create metrics cards if tuning is in progress or we have metrics
        if st.session_state.get('tuning_in_progress', False) or st.session_state.get('best_metrics'):
            create_metrics_cards()
        
        # Create progress bar for tuning
        if st.session_state.get('tuning_in_progress', False):
            create_progress_bar()
        
        # Load data based on selected parameters
        ticker = params["ticker"]
        timeframe = params["timeframe"]
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        
        # Handle potential issues with date parameters
        if not isinstance(start_date, datetime):
            if isinstance(start_date, str):
                try:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                except ValueError:
                    start_date = datetime.now() - timedelta(days=30)
            else:
                start_date = datetime.now() - timedelta(days=30)
                
        if not isinstance(end_date, datetime):
            if isinstance(end_date, str):
                try:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                except ValueError:
                    end_date = datetime.now()
            else:
                end_date = datetime.now()
        
        # Try to load data
        df = load_data(ticker, start_date.strftime("%Y-%m-%d"), 
                   end_date.strftime("%Y-%m-%d"), interval=timeframe)
        
        if df is not None and not df.empty:
            st.session_state["df_raw"] = df
        elif 'df_raw' not in st.session_state or st.session_state["df_raw"] is None:
            st.warning(f"No data available for {ticker} with the selected parameters.")
            # Create an empty placeholder
            st.empty()
            return
        else:
            df = st.session_state["df_raw"]
            st.warning("Using cached data as new data could not be fetched.")
        
        # Generate future forecast if we have a model
        if 'current_model' in st.session_state and st.session_state['current_model'] is not None and df is not None and not df.empty:
            model = st.session_state['current_model']
            
            # Get feature columns safely
            if modules["config"] and hasattr(modules["config"], "get_active_feature_names"):
                feature_cols = modules["config"].get_active_feature_names()
            else:
                # Fallback to extracting columns from dataframe
                feature_cols = [col for col in df.columns if col not in ['date', 'Date', 'Close']]
                
            # Make sure we have features to work with
            if feature_cols:
                # Use session state values for lookback and forecast window
                lookback = st.session_state.get("lookback", 30)
                horizon = st.session_state.get("forecast_window", 30)
                
                try:
                    # Display a status message during forecast generation
                    forecast_status = st.empty()
                    forecast_status.info("Generating price forecast...")
                    
                    # Generate the forecast using the updated function
                    future_forecast = generate_future_forecast(model, df, feature_cols, lookback, horizon)
                    
                    # Store the forecast and update timestamp
                    if future_forecast and len(future_forecast) > 0:
                        st.session_state['future_forecast'] = future_forecast
                        st.session_state['last_forecast_update'] = datetime.now()
                        forecast_status.success(f"Forecast generated for {horizon} days")
                    else:
                        forecast_status.warning("Forecast generation returned no data")
                        # Set empty forecast to avoid errors
                        st.session_state['future_forecast'] = []
                        
                except Exception as e:
                    logger.error(f"Error generating forecast: {e}", exc_info=True)
                    st.warning(f"Could not generate price forecast: {str(e)}")
                    # Set empty forecast to avoid errors in visualization
                    st.session_state['future_forecast'] = []
            else:
                logger.warning("No feature columns available for forecast")
                st.session_state['future_forecast'] = []
        
        # Create tabs for the main content
        main_tabs = st.tabs([
            "Price Prediction", 
            "Technical Indicators", 
            "Model Visualization",
            "Prediction Explorer",
            "Model Comparison",
            "Feature Importance",
            "Hyperparameter Tuning",
            "Downloads & Settings"
        ])
        
        with main_tabs[0]:
            # Show price prediction chart
            create_interactive_price_chart(
                df, 
                params,
                st.session_state.get('ensemble_predictions_log'),
                st.session_state.get('future_forecast')
            )
            
            # Add tuning controls
            tuning_col1, tuning_col2 = st.columns(2)
            with tuning_col1:
                if st.button("▶️ Start Tuning", key="start_tuning_main_tab"):
                    start_tuning(ticker, timeframe)
            with tuning_col2:
                if st.button("⏹️ Stop Tuning", key="stop_tuning_main_tab"):
                    stop_tuning()
        
        with main_tabs[1]:
            # Show technical indicators
            create_technical_indicators_chart(df, params)
        
        with main_tabs[2]:
            # Show model visualization
            create_model_visualization()
        
        with main_tabs[3]:
            # Show prediction explorer
            create_prediction_explorer(st.session_state.get('ensemble_predictions_log', []))
        
        with main_tabs[4]:
            # Show model comparison
            create_model_comparison()
        
        with main_tabs[5]:
            # Show feature importance explorer
            create_feature_importance_explorer()
            
        with main_tabs[6]:
            # Try to integrate with hyperparameter dashboard
            try:
                # Import hyperparameter dashboard dynamically
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                import hyperparameter_dashboard
                hyperparameter_dashboard.main()
            except ImportError:
                st.warning("Hyperparameter dashboard module not found.")
                st.write("Please make sure `hyperparameter_dashboard.py` is in the same directory as this script.")
            except Exception as e:
                st.error(f"Error loading hyperparameter dashboard: {e}")
                logger.error(f"Error loading hyperparameter dashboard: {e}", exc_info=True)
        
        with main_tabs[7]:
            # Show download section and model save/load controls
            settings_tabs = st.tabs(["Downloads", "Model Save/Load", "Tested Models", "Error Log"])
            
            with settings_tabs[0]:
                create_download_section()
            with settings_tabs[1]:
                model_save_load_controls()
            with settings_tabs[2]:
                display_tested_models()
            with settings_tabs[3]:
                show_error_log()
                
        # Additional section for advanced indicators and analysis
        with st.expander("Advanced Analysis", expanded=False):
            show_advanced_dashboard_tabs(df)
        
        # Auto-refresh logic
        if params.get("auto_refresh") and params.get("refresh_interval"):
            time_since_refresh = time.time() - st.session_state.get("last_refresh", 0)
            if time_since_refresh >= params["refresh_interval"]:
                st.session_state["last_refresh"] = time.time()
                st.experimental_rerun()
                
    except Exception as e:
        st.error(f"Critical error in main dashboard: {e}")
        logger.error(f"Critical error in main dashboard: {e}", exc_info=True)
        
        # Try to provide some recovery options
        if st.button("Reset Session State and Reload"):
            for key in list(st.session_state.keys()):
                if key != 'df_raw':  # Preserve data if available
                    del st.session_state[key]
            st.experimental_rerun()

@robust_error_boundary
def show_error_log():
    """Display error log for debugging"""
    st.header("Error Log")
    
    if 'error_log' not in st.session_state or not st.session_state['error_log']:
        st.info("No errors recorded in this session.")
        return
        
    if st.button("Clear Error Log"):
        st.session_state['error_log'] = []
        st.success("Error log cleared.")
        return
        
    # Convert to DataFrame for easier display
    try:
        error_data = []
        for error in st.session_state['error_log']:
            error_data.append({
                'Time': error.get('timestamp', datetime.now()),
                'Function': error.get('function', 'Unknown'),
                'Error': error.get('error', 'Unknown error')
            })
            
        df_errors = pd.DataFrame(error_data)
        st.dataframe(df_errors)
        
        # Show detailed traceback for selected error
        if len(error_data) > 0:
            selected_index = st.selectbox(
                "Select error to view details",
                range(len(st.session_state['error_log'])),
                format_func=lambda i: f"{error_data[i]['Time']} - {error_data[i]['Function']}"
            )
            
            selected_error = st.session_state['error_log'][selected_index]
            if 'traceback' in selected_error:
                st.code(selected_error['traceback'], language="python")
    except Exception as e:
        st.error(f"Error displaying error log: {e}")
        # Fallback to simpler display
        for i, error in enumerate(st.session_state['error_log']):
            st.write(f"**Error {i+1}:** {error.get('timestamp')} - {error.get('function')} - {error.get('error')}")
            
@robust_error_boundary
def show_advanced_dashboard_tabs(data):
    """Show tabs with advanced indicators and analysis"""
    if data is None or data.empty:
        st.warning("No data available for advanced analysis.")
        return
            
    adv_tabs = st.tabs([
        "WERPI Indicator", 
        "VMLI Analysis", 
        "Correlation Matrix"
    ])
    
    with adv_tabs[0]:
        st.subheader("Wavelet-based Encoded Relative Price Indicator (WERPI)")
        st.info("WERPI is a custom indicator that uses wavelet transforms to identify potential price movements.")
        
        # Create placeholder WERPI data using simpler calculations for reliability
        try:
            # Simple placeholder calculation
            werpi_values = (data['Close'] - data['Close'].rolling(window=20).mean()) / data['Close'].rolling(window=20).std()
            werpi_values = werpi_values.rolling(window=5).mean()  # Smooth the values
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['date'], y=data['Close'], name='Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data['date'], y=werpi_values, name='WERPI', line=dict(color='red')))
            fig.update_layout(
                title="Price vs WERPI Indicator",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot (removed as it seems to be part of a different section)
            # This was likely accidentally pasted here
            fig.add_trace(go.Scatter(x=data['date'], y=werpi_values, name='WERPI', line=dict(color='red')))
            fig.update_layout(title="Price vs WERPI Indicator", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating WERPI: {e}")
    
    with adv_tabs[1]:
        st.subheader("Volatility-Momentum-Liquidity Indicator (VMLI)")
        st.info("VMLI combines volatility, momentum and liquidity metrics into a single indicator.")
        
        # Create simplified VMLI calculation for reliability
        try:
            # Calculate simple components
            volatility = data['Close'].pct_change().rolling(window=14).std() * 100
            momentum = data['Close'].pct_change(periods=14) * 100
            
            # Calculate simple VMLI
            vmli_values = (momentum / volatility).fillna(0)
            vmli_values = 50 + (vmli_values * 10)  # Scale to a more readable range
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['date'], y=data['Close'], name='Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data['date'], y=vmli_values, name='VMLI', line=dict(color='purple')))
            fig.update_layout(title="Price vs VMLI Indicator", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating VMLI: {e}")
    
    with adv_tabs[2]:
        st.subheader("Correlation Matrix")
        
        # Create correlation matrix for key features
        try:
            # Safely select only numeric columns
            numeric_cols = []
            for col in data.columns:
                if col not in ['date', 'Date'] and pd.api.types.is_numeric_dtype(data[col]):
                    numeric_cols.append(col)
            
            # Limit to most important columns
            if len(numeric_cols) > 15:
                numeric_cols = numeric_cols[:15]
            
            if len(numeric_cols) < 2:
                st.warning("Not enough numeric columns for correlation analysis.")
                return
                
            corr_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r"
            )
            fig.update_layout(title="Feature Correlation Matrix", height=600)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating correlation matrix: {e}")

###############################################################################
#                      PAGE CONFIG AND INITIALIZATION                         #
###############################################################################

@robust_error_boundary
def set_page_config():
    """Configure the Streamlit page settings"""
    try:
        st.set_page_config(
            page_title="AI Price Prediction Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        # This might fail if called twice
        logger.warning(f"Error setting page config: {e}")
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4CAF50, #8BC34A, #CDDC39);
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    .metric-card:hover {
        transform: translateY(-0.25rem);
        box-shadow: 0 0.5rem 2rem 0 rgba(58, 59, 69, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stPlotlyChart {
        padding: 1rem 0;
    }
    .dashboard-header {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
        
    default_values = {
        "initialized": True,
        "tuning_in_progress": read_tuning_status(),  # Read from file for persistence
        "best_metrics": {},
        "model_history": [],
        "last_refresh": time.time(),
        "prediction_history": [],
        "accuracy_metrics_history": {},
        "neural_network_state": {},
        "current_model": None,
        "model_loaded": False,
        "error_log": [],
        "training_history": None,
        "ensemble_predictions_log": [],
        "df_raw": None,
        "historical_window": 30,
        "forecast_window": 30,
        "trials_per_cycle": N_STARTUP_TRIALS,
        "initial_trials": N_STARTUP_TRIALS,
        "saved_model_dir": "saved_models",
        # Auto-refresh intervals (in seconds)
        "progress_refresh_sec": 2,
        "full_refresh_sec": 30,
        "last_progress_refresh": time.time(),
        "last_full_refresh": time.time(),
        "update_heavy_components": True,
        # UI Control state
        "selected_ticker": TICKER,
        "selected_timeframe": TIMEFRAMES[0] if TIMEFRAMES else "1d",
        "start_date": datetime.now() - timedelta(days=30),
        "end_date": datetime.now(),
        "auto_refresh": True,
        "refresh_interval": 30,
        # Model data structures
        "model_comparison_data": [],
        "feature_importance_data": {},
        "model_weights_history": [],
        # Containers for progressive display
        "progress_container": None,
        "progress_bar": None,
        "metrics_container": None,
        # Status tracking
        "live_progress": {}
    }
    
    # Set defaults only for keys not already in session_state
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

###############################################################################
#                           UI COMPONENT FUNCTIONS                            #
###############################################################################

@robust_error_boundary
def create_progress_bar():
    """Create an enhanced progress bar"""
    progress = load_latest_progress()
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get("total_trials", 1)  # Avoid division by zero
    
    if total_trials > 0:
        progress_value = current_trial / total_trials
        
        # Ensure progress value is between 0 and 1
        progress_value = max(0, min(1, progress_value))
        
        # Progress bar with custom styling
        st.progress(progress_value)
        
        # Add textual description below
        col1, col2 = st.columns([3, 1])
        with col1:
            if progress_value < 0.3:
                st.info(f"Exploring model hyperparameters: {current_trial}/{total_trials} trials ({int(progress_value*100)}%)")
            elif progress_value < 0.7:
                st.info(f"Refining promising configurations: {current_trial}/{total_trials} trials ({int(progress_value*100)}%)")
            else:
                st.success(f"Finalizing optimal model: {current_trial}/{total_trials} trials ({int(progress_value*100)}%)")
        
        with col2:
            # Estimate time remaining
            if 'start_time' in st.session_state and current_trial > 0:
                elapsed_time = time.time() - st.session_state['start_time']
                time_per_trial = elapsed_time / current_trial
                trials_remaining = total_trials - current_trial
                estimated_time = time_per_trial * trials_remaining
                
                # Format as hours:minutes:seconds
                hours, remainder = divmod(estimated_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                st.text(f"Est. time: {int(hours)}h {int(minutes)}m")

@robust_error_boundary
def create_header():
    """Create a visually appealing header section"""
    with st.container():
        st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.title("🤖 AI Price Prediction Dashboard")
            st.markdown("""
            This interactive dashboard displays real-time price predictions, model training metrics, 
            and advanced technical indicators for financial markets.
            """)
        
        with col2:
            # Add a status indicator
            if st.session_state.get('tuning_in_progress', False):
                st.success("🔄 Tuning Active")
            else:
                st.info("⏸️ Tuning Inactive")
            
            # Add a last updated timestamp
            st.text(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        st.markdown('</div>', unsafe_allow_html=True)

@robust_error_boundary
def create_metrics_cards():
    """Create a row of key metrics cards with icons"""
    progress = load_latest_progress()
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get("total_trials", 0)
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)
    cycle = progress.get("cycle", 1)
    
    # Calculate best metrics if available
    best_rmse = st.session_state.get('best_metrics', {}).get('rmse', float('inf'))
    best_mape = st.session_state.get('best_metrics', {}).get('mape', float('inf'))
    
    # Handle potential NaN or infinity values
    if not isinstance(best_rmse, (int, float)) or np.isnan(best_rmse) or np.isinf(best_rmse):
        best_rmse = float('inf')
    if not isinstance(best_mape, (int, float)) or np.isnan(best_mape) or np.isinf(best_mape):
        best_mape = float('inf')
    
    # Delta values (improvement from best to current)
    rmse_delta = None if current_rmse is None else best_rmse - current_rmse
    mape_delta = None if current_mape is not None else best_mape - current_mape
    
    # Handle potential NaN or infinity values in deltas
    if rmse_delta is not None and (np.isnan(rmse_delta) or np.isinf(rmse_delta)):
        rmse_delta = None
    if mape_delta is not None and (np.isnan(mape_delta) or np.isinf(mape_delta)):
        mape_delta = None
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="🔄 Current Cycle",
            value=cycle,
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if total_trials > 0:
            progress_pct = int((current_trial / total_trials) * 100)
        else:
            progress_pct = 0
            
        st.metric(
            label="📊 Trials Progress",
            value=f"{current_trial}/{total_trials}",
            delta=f"{progress_pct}%" if total_trials > 0 else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="📉 Current RMSE",
            value=f"{current_rmse:.2f}" if current_rmse is not None else "N/A",
            delta=f"{rmse_delta:.2f}" if rmse_delta is not None else None,
            delta_color="inverse"  # Lower is better for RMSE
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="📊 Current MAPE",
            value=f"{current_mape:.2f}%" if current_mape is not None else "N/A",
            delta=f"{mape_delta:.2f}%" if mape_delta is not None else None,
            delta_color="inverse"  # Lower is better for MAPE
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        # Calculate success rate if we have predictions
        if 'ensemble_predictions_log' in st.session_state and st.session_state['ensemble_predictions_log']:
            predictions = st.session_state['ensemble_predictions_log']
            if len(predictions) > 1:
                try:
                    # Calculate direction accuracy
                    correct_direction = 0
                    for i in range(1, len(predictions)):
                        actual_direction = predictions[i]['actual'] > predictions[i-1]['actual']
                        pred_direction = predictions[i]['predicted'] > predictions[i-1]['predicted']
                        if actual_direction == pred_direction:
                            correct_direction += 1
                    
                    success_rate = (correct_direction / (len(predictions) - 1)) * 100
                    st.metric(
                        label="🎯 Direction Accuracy",
                        value=f"{success_rate:.1f}%",
                        delta=None
                    )
                except (KeyError, TypeError) as e:
                    st.metric(label="🎯 Direction Accuracy", value="Error", delta=None)
                    logger.error(f"Error calculating direction accuracy: {e}")
            else:
                st.metric(label="🎯 Direction Accuracy", value="N/A", delta=None)
        else:
            st.metric(label="🎯 Direction Accuracy", value="N/A", delta=None)
        
        st.markdown('</div>', unsafe_allow_html=True)

@robust_error_boundary
def create_interactive_price_chart(df, params, predictions_log=None, future_forecast=None):
    """Create an enhanced interactive price chart with predictions"""
    if df is None or df.empty:
        st.warning("No data available to display.")
        return
    
    # Defensive type checking
    if not isinstance(df, pd.DataFrame):
        st.error("Invalid data format. Expected DataFrame.")
        return
        
    # Ensure required columns exist
    required_cols = ['date', 'Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return
    
    try:
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Drop rows with NaN dates
        df = df.dropna(subset=['date'])
        
        # Prepare the data
        df = df.sort_values('date')
        
        # Filter to historical window
        historical_window = params.get("historical_window", 30)
        if len(df) > historical_window:
            df = df.iloc[-historical_window:]
        
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.8, 0.2],
            subplot_titles=(f"{params['ticker']} Price with Predictions", "Volume")
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC",
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add volume chart if volume exists
        if 'Volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['Volume'],
                    name="Volume",
                    marker_color='rgba(0, 0, 255, 0.5)'
                ),
                row=2, col=1
            )
        
        # Add moving averages if enough data
        for period in [20, 50, 200]:
            # Skip if we don't have enough data
            if len(df) >= period:
                ma_col = f'MA_{period}'
                if ma_col not in df.columns:
                    df[ma_col] = df['Close'].rolling(window=period).mean()
                    
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[ma_col],
                        name=f"{period}-day MA",
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Add past predictions if available
        if predictions_log:
            # Convert to DataFrame
            try:
                pred_df = pd.DataFrame(predictions_log)
                
                # Ensure date column is datetime
                if 'date' in pred_df.columns:
                    pred_df['date'] = pd.to_datetime(pred_df['date'], errors='coerce')
                    
                    # Filter to dates in our chart
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    pred_df = pred_df[(pred_df['date'] >= min_date) & (pred_df['date'] <= max_date)]
                    
                    if not pred_df.empty:
                        # Add past predictions
                        fig.add_trace(
                            go.Scatter(
                                x=pred_df['date'],
                                y=pred_df['predicted'],
                                mode='markers',
                                name='Past Predictions',
                                marker=dict(
                                    size=8,
                                    color='blue',
                                    symbol='circle'
                                )
                            ),
                            row=1, col=1
                        )
                        
                        # Add prediction errors
                        if 'actual' in pred_df.columns:
                            try:
                                pred_df['error'] = ((pred_df['predicted'] - pred_df['actual']) / pred_df['actual']) * 100
                                error_colors = ['red' if err > 0 else 'green' for err in pred_df['error']]
                                
                                # Use a scatter plot with hover info
                                fig.add_trace(
                                    go.Scatter(
                                        x=pred_df['date'],
                                        y=pred_df['error'],
                                        mode='markers',
                                        name='Prediction Error %',
                                        marker=dict(
                                            size=8,
                                            color=error_colors,
                                            symbol='circle'
                                        ),
                                        hovertemplate='Date: %{x}<br>Error: %{y:.2f}%<br>'
                                    ),
                                    row=2, col=1
                                )
                            except Exception as e:
                                logger.error(f"Error plotting prediction errors: {e}")
            except Exception as e:
                logger.error(f"Error processing predictions: {e}")
        
        # Add future forecast if available
        if isinstance(future_forecast, (list, np.ndarray)) and len(future_forecast) > 0:
            try:
                # Generate future dates
                last_date = df['date'].iloc[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=len(future_forecast)
                )
                
                # Add the forecast line
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=future_forecast,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(
                            color='red',
                            width=2,
                            dash='dash'
                        ),
                        marker=dict(
                            size=6,
                            color='red',
                            symbol='circle'
                        )
                    ),
                    row=1, col=1
                )
                
                # Add confidence interval if we have enough past predictions
                if predictions_log and len(predictions_log) >= 5:
                    try:
                        # Calculate RMSE from past predictions
                        pred_array = np.array([p.get('predicted', 0) for p in predictions_log])
                        actual_array = np.array([p.get('actual', 0) for p in predictions_log])
                        
                        # Filter out any NaN values
                        mask = ~(np.isnan(pred_array) | np.isnan(actual_array))
                        pred_array = pred_array[mask]
                        actual_array = actual_array[mask]
                        
                        if len(pred_array) > 0 and len(actual_array) > 0:
                            rmse = np.sqrt(np.mean((pred_array - actual_array)**2))
                            
                            # Create upper and lower bounds (95% confidence interval)
                            upper_bound = np.array(future_forecast) + 1.96 * rmse
                            lower_bound = np.array(future_forecast) - 1.96 * rmse
                            
                            # Add confidence interval as a filled area
                            fig.add_trace(
                                go.Scatter(
                                    x=future_dates,
                                    y=upper_bound,
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=future_dates,
                                    y=lower_bound,
                                    mode='lines',
                                    line=dict(width=0),
                                    fillcolor='rgba(255, 0, 0, 0.1)',
                                    fill='tonexty',
                                    name='95% Confidence'
                                ),
                                row=1, col=1
                            )
                    except Exception as e:
                        logger.error(f"Error adding confidence interval: {e}")
            except Exception as e:
                logger.error(f"Error adding forecast: {e}")
        
        # Update layout for better appearance
        fig.update_layout(
            height=700,
            title=f"{params['ticker']} Price Chart with Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Update y-axis for the volume subplot
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Add current stats as annotations
        if len(df) >= 2:
            try:
                last_price = df['Close'].iloc[-1]
                change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                change_pct = (change / df['Close'].iloc[-2]) * 100
                
                fig.add_annotation(
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Last Price: ${last_price:.2f}<br>Change: {change:.2f} ({change_pct:.2f}%)",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color="white"
                    ),
                    align="left",
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4,
                    xanchor="left",
                    yanchor="top"
                )
            except Exception as e:
                logger.error(f"Error adding price annotation: {e}")
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating price chart: {e}")
        logger.error(f"Error in create_interactive_price_chart: {e}", exc_info=True)

@robust_error_boundary
def create_technical_indicators_chart(df, params):
    """Create an interactive technical indicators chart"""
    if df is None or df.empty:
        st.warning("No technical indicators data available.")
        return
    
    # Ensure 'date' column exists (defensive coding)
    if 'date' not in df.columns:
        if 'Date' in df.columns:
            # Rename for consistency
            df = df.rename(columns={'Date': 'date'})
        else:
            st.error("Missing date column in dataframe")
            return
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Create tabs for different indicator groups
    indicator_tabs = st.tabs([
        "Price & Bands", 
        "Momentum", 
        "MACD", 
        "Volume"
    ])
    
    with indicator_tabs[0]:
        try:
            # Price with Bollinger Bands
            fig1 = go.Figure()
            
            fig1.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['Close'],
                    name="Price",
                    line=dict(color='blue', width=2)
                )
            )
            
            # Check if BB columns exist before plotting
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                fig1.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['BB_Upper'],
                        name="Upper Band",
                        line=dict(color='green', width=1, dash='dash')
                    )
                )
                fig1.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['BB_Lower'],
                        name="Lower Band",
                        line=dict(color='red', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.05)'
                    )
                )
            
            fig1.update_layout(
                title="Price with Bollinger Bands",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating Bollinger Bands chart: {e}")
    
    with indicator_tabs[1]:
        try:
            # RSI Chart
            fig2 = go.Figure()
            if 'RSI' in df.columns:
                fig2.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['RSI'],
                        name="RSI",
                        line=dict(color='purple', width=2)
                    )
                )
                # Add overbought/oversold lines
                fig2.add_shape(
                    type="line",
                    x0=df['date'].iloc[0],
                    x1=df['date'].iloc[-1],
                    y0=70,
                    y1=70,
                    line=dict(color="red", width=1, dash="dash")
                )
                fig2.add_shape(
                    type="line",
                    x0=df['date'].iloc[0],
                    x1=df['date'].iloc[-1],
                    y0=30,
                    y1=30,
                    line=dict(color="green", width=1, dash="dash")
                )
                fig2.update_layout(
                    title="Relative Strength Index (RSI)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100]),
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("RSI indicator not available in dataset.")
        except Exception as e:
            st.error(f"Error creating RSI chart: {e}")
    
    with indicator_tabs[2]:
        try:
            # MACD Chart
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns and 'MACD_Hist' in df.columns:
                fig3 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Price", "MACD")
                )
                fig3.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['Close'],
                        name="Price",
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                fig3.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['MACD'],
                        name="MACD",
                        line=dict(color='blue', width=1.5)
                    ),
                    row=2, col=1
                )
                fig3.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['MACD_Signal'],
                        name="Signal",
                        line=dict(color='red', width=1.5)
                    ),
                    row=2, col=1
                )
                colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
                fig3.add_trace(
                    go.Bar(
                        x=df['date'],
                        y=df['MACD_Hist'],
                        name="Histogram",
                        marker_color=colors
                    ),
                    row=2, col=1
                )
                fig3.update_layout(
                    height=600,
                    hovermode="x unified",
                    title="Moving Average Convergence Divergence (MACD)"
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("MACD indicators not available in dataset.")
        except Exception as e:
            st.error(f"Error creating MACD chart: {e}")
    
    with indicator_tabs[3]:
        try:
            # Volume Analysis
            if 'Volume' in df.columns:
                fig4 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Price", "Volume")
                )
                fig4.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['Close'],
                        name="Price",
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
                fig4.add_trace(
                    go.Bar(
                        x=df['date'],
                        y=df['Volume'],
                        name="Volume",
                        marker_color=colors
                    ),
                    row=2, col=1
                )
                
                # Add volume moving average if not already calculated
                if 'Volume_MA_20' not in df.columns:
                    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
                    
                fig4.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['Volume_MA_20'],
                        name="Volume MA (20)",
                        line=dict(color='purple', width=1.5)
                    ),
                    row=2, col=1
                )
                fig4.update_layout(
                    height=600,
                    hovermode="x unified",
                    title="Volume Analysis"
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Volume data not available in dataset.")
        except Exception as e:
            st.error(f"Error creating Volume chart: {e}")

@robust_error_boundary
def create_model_visualization():
    """Create visualizations for the current model"""
    model = st.session_state.get('current_model')
    
    if model is None:
        st.info("No model available for visualization. Please train or load a model first.")
        return
    
    # Create tabs for different visualizations
    model_tabs = st.tabs([
        "Architecture", 
        "Training History", 
        "Feature Importance",
        "Layer Weights"
    ])
    
    with model_tabs[0]:
        try:
            # Try to get feature names
            feature_names = []
            if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
                df = st.session_state['df_raw']
                feature_names = [col for col in df.columns if col not in ['date', 'Date', 'Close']]
            
            # Create architecture visualization
            if modules["model_visualization"] and hasattr(modules["model_visualization"], "visualize_neural_network"):
                fig = modules["model_visualization"].visualize_neural_network(model, feature_names)
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    
                    # Display the image
                    st.image(buf, caption="Model Architecture", use_column_width=True)
                else:
                    # Fallback to text-based model summary
                    st.warning("Could not create architecture visualization, showing summary instead.")
                    summary_str = []
                    model.summary(print_fn=lambda x: summary_str.append(x))
                    st.text("\n".join(summary_str))
            else:
                # Fallback to text-based model summary
                st.warning("Model visualization module not available, showing summary instead.")
                summary_str = []
                model.summary(print_fn=lambda x: summary_str.append(x))
                st.text("\n".join(summary_str))
        except Exception as e:
            st.error(f"Error visualizing model architecture: {e}")
            logger.error(f"Error visualizing model architecture: {e}", exc_info=True)
    
    with model_tabs[1]:
        try:
            # Display training history if available
            if 'training_history' in st.session_state and st.session_state['training_history']:
                history = st.session_state['training_history']
                
                # Create training history plot
                if modules["model_visualization"] and hasattr(modules["model_visualization"], "plot_training_history"):
                    hist_fig = modules["model_visualization"].plot_training_history(history)
                    if hist_fig:
                        st.plotly_chart(hist_fig, use_container_width=True)
                    else:
                        # Fallback to manual plotting
                        st.warning("Could not create training history visualization, showing data instead.")
                        
                        # Create a simple plot
                        fig = go.Figure()
                        for metric in history.keys():
                            fig.add_trace(go.Scatter(y=history[metric], name=metric))
                        fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Value")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to manual plotting
                    fig = go.Figure()
                    for metric in history.keys():
                        fig.add_trace(go.Scatter(y=history[metric], name=metric))
                    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Value")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training history available.")
        except Exception as e:
            st.error(f"Error visualizing training history: {e}")
            logger.error(f"Error visualizing training history: {e}", exc_info=True)
    
    with model_tabs[2]:
        try:
            # Try to visualize feature importance
            if 'feature_importance' in st.session_state:
                feature_importance = st.session_state['feature_importance']
                feature_names = list(feature_importance.keys())
                importance_values = list(feature_importance.values())
                
                if modules["model_visualization"] and hasattr(modules["model_visualization"], "plot_feature_importance"):
                    fig = modules["model_visualization"].plot_feature_importance(importance_values, feature_names)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to manual plotting
                        st.warning("Could not create feature importance visualization, showing data instead.")
                        
                        # Create a simple bar chart
                        fig = px.bar(
                            x=feature_names, 
                            y=importance_values,
                            title="Feature Importance"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to manual plotting
                    fig = px.bar(
                        x=feature_names, 
                        y=importance_values,
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance data available.")
        except Exception as e:
            st.error(f"Error visualizing feature importance: {e}")
            logger.error(f"Error visualizing feature importance: {e}", exc_info=True)
    
    with model_tabs[3]:
        try:
            # Visualize layer weights
            if hasattr(model, 'layers'):
                # Select layer
                layers_with_weights = [l.name for l in model.layers if len(l.weights) > 0]
                if layers_with_weights:
                    selected_layer = st.selectbox(
                        "Select layer to visualize weights:",
                        layers_with_weights
                    )
                    
                    # Get layer by name
                    layer = None
                    for l in model.layers:
                        if l.name == selected_layer:
                            layer = l
                            break
                    
                    if layer and len(layer.weights) > 0:
                        # Create weight visualization
                        st.write(f"Layer: {layer.name}, Shape: {layer.weights[0].shape}")
                        
                        # Plot weights as heatmap
                        weights = layer.weights[0].numpy()
                        
                        # Reshape if needed for visualization
                        if len(weights.shape) > 2:
                            weights = weights.reshape(-1, weights.shape[-1])
                        
                        # Create heatmap
                        fig = px.imshow(
                            weights, 
                            title=f"Weights for layer: {layer.name}",
                            color_continuous_scale="RdBu_r",
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot histogram of weights
                        hist_fig = px.histogram(
                            weights.flatten(), 
                            title=f"Weight Distribution for layer: {layer.name}",
                            nbins=50
                        )
                        st.plotly_chart(hist_fig, use_container_width=True)
                    else:
                        st.warning("Selected layer has no weights.")
                else:
                    st.info("No layers with weights found in the model.")
            else:
                st.info("Model does not have layers to visualize.")
        except Exception as e:
            st.error(f"Error visualizing layer weights: {e}")
            logger.error(f"Error visualizing layer weights: {e}", exc_info=True)

@robust_error_boundary
def create_feature_importance_explorer():
    """Create an explorer for feature importance across models"""
    if 'feature_importance_data' not in st.session_state:
        # Initialize with an empty dict if it doesn't exist
        st.session_state['feature_importance_data'] = {}
        
    feature_data = st.session_state['feature_importance_data']
    if not feature_data:
        st.info("No feature importance data available.")
        return
    
    try:
        # Allow selection of different models
        model_ids = list(feature_data.keys())
        selected_model = st.selectbox(
            "Select model to view feature importance:",
            model_ids
        )
        
        if selected_model not in feature_data:
            st.warning(f"Model {selected_model} not found in feature importance data.")
            return
        
        # Display feature importance for selected model
        model_features = feature_data[selected_model]
        
        # Convert to DataFrame and sort
        df_features = pd.DataFrame({
            'Feature': list(model_features.keys()),
            'Importance': list(model_features.values())
        }).sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            df_features,
            x='Feature',
            y='Importance',
            title=f"Feature Importance for Model {selected_model}",
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add feature descriptions and explanations
        st.subheader("Feature Explanations")
        
        # Define explanations for common features
        feature_explanations = {
            'Open': 'Opening price of the asset for the period.',
            'High': 'Highest price reached during the period.',
            'Low': 'Lowest price reached during the period.',
            'Close': 'Closing price of the asset for the period.',
            'Volume': 'Trading volume (number of shares/contracts traded).',
            'RSI': 'Relative Strength Index - momentum oscillator measuring speed and change of price movements.',
            'MACD': 'Moving Average Convergence Divergence - trend-following momentum indicator.',
            'MACD_Signal': 'Signal line for the MACD indicator.',
            'MACD_Hist': 'Histogram representing difference between MACD and Signal line.',
            'BB_Upper': 'Upper band of the Bollinger Bands indicator.',
            'BB_Lower': 'Lower band of the Bollinger Bands indicator.',
            'BB_Middle': 'Middle band (simple moving average) of Bollinger Bands.',
            'ATR': 'Average True Range - market volatility indicator.',
            'OBV': 'On-Balance Volume - momentum indicator using volume flow.',
            'WERPI': 'Wavelet-Encoded Relative Price Indicator - custom indicator using wavelets.',
            'WeekendGap': 'Price gap between Friday close and Monday open.',
            'Returns': 'Percentage price change from previous period.',
            'Volatility': 'Standard deviation of returns over a period.',
            'MA_20': '20-day Moving Average of closing price.',
            'MA_50': '50-day Moving Average of closing price.',
            'MA_200': '200-day Moving Average of closing price.'
        }
        
        # Display explanations for top features
        for feature in df_features['Feature'].head(10):
            if feature in feature_explanations:
                st.markdown(f"**{feature}**: {feature_explanations[feature]}")
            else:
                st.markdown(f"**{feature}**: No explanation available.")
        
        # Add correlation heatmap
        if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
            df = st.session_state['df_raw']
            
            # Get top features
            top_features = df_features['Feature'][:10].tolist()
            
            # Filter DataFrame to include only these features
            available_features = [f for f in top_features if f in df.columns]
            
            if len(available_features) >= 2:
                st.subheader("Feature Correlation Heatmap")
                
                # Create correlation matrix
                corr_matrix = df[available_features].corr()
                
                # Create heatmap
                corr_fig = px.imshow(
                    corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    title="Feature Correlation Heatmap",
                    text_auto=True
                )
                
                st.plotly_chart(corr_fig, use_container_width=True)
                
                st.info("""
                The correlation heatmap shows how the top features relate to each other. 
                Values close to 1 indicate strong positive correlation, 
                values close to -1 indicate strong negative correlation, 
                and values close to 0 indicate little or no correlation.
                """)
    except Exception as e:
        st.error(f"Error in feature importance explorer: {e}")
        logger.error(f"Error in feature importance explorer: {e}", exc_info=True)

@robust_error_boundary
def create_prediction_explorer(predictions_log):
    """Create an explorer for past predictions and performance"""
    if not predictions_log:
        st.info("No prediction data available yet.")
        return
    
    try:
        # Convert to DataFrame
        df_pred = pd.DataFrame(predictions_log)
        
        # Calculate additional metrics
        try:
            df_pred['error_pct'] = ((df_pred['predicted'] - df_pred['actual']) / df_pred['actual']) * 100
            df_pred['abs_error_pct'] = df_pred['error_pct'].abs()
        except KeyError:
            st.warning("Missing actual or predicted values in prediction log.")
            return
        
        # Add date info
        try:
            df_pred['date'] = pd.to_datetime(df_pred['date'])
            df_pred['day'] = df_pred['date'].dt.day_name()
            df_pred['month'] = df_pred['date'].dt.month_name()
            df_pred['year'] = df_pred['date'].dt.year
        except KeyError:
            st.warning("Missing date information in prediction log.")
            return
        
        # Create tabs for different analyses
        pred_tabs = st.tabs([
            "Prediction Performance", 
            "Error Analysis", 
            "Time Patterns",
            "Prediction Log"
        ])
        
        with pred_tabs[0]:
            # Overall metrics
            st.subheader("Overall Prediction Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Average RMSE",
                    f"{df_pred['rmse'].mean():.4f}" if 'rmse' in df_pred.columns else "N/A"
                )
            
            with col2:
                st.metric(
                    "Average MAPE",
                    f"{df_pred['mape'].mean():.2f}%" if 'mape' in df_pred.columns else "N/A"
                )
            
            with col3:
                # Direction accuracy
                if len(df_pred) > 1:
                    df_pred['actual_direction'] = df_pred['actual'].diff().fillna(0) > 0
                    df_pred['pred_direction'] = df_pred['predicted'].diff().fillna(0) > 0
                    direction_accuracy = (df_pred['actual_direction'] == df_pred['pred_direction']).mean() * 100
                    st.metric("Direction Accuracy", f"{direction_accuracy:.1f}%")
                else:
                    st.metric("Direction Accuracy", "N/A")
            
            # Time series plot of actual vs predicted
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=df_pred['date'],
                    y=df_pred['actual'],
                    name="Actual",
                    line=dict(color='blue', width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_pred['date'],
                    y=df_pred['predicted'],
                    name="Predicted",
                    line=dict(color='red', width=2, dash='dash')
                )
            )
            
            fig.update_layout(
                title="Actual vs Predicted Prices Over Time",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            scatter_fig = px.scatter(
                df_pred,
                x='actual',
                y='predicted',
                hover_data=['date', 'error_pct'],
                title="Predicted vs Actual Scatter Plot",
                labels={'actual': 'Actual Price', 'predicted': 'Predicted Price'}
            )
            
            # Add perfect prediction line
            scatter_fig.add_trace(
                go.Scatter(
                    x=[df_pred['actual'].min(), df_pred['actual'].max()],
                    y=[df_pred['actual'].min(), df_pred['actual'].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='green', dash='dash')
                )
            )
            
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        with pred_tabs[1]:
            # Error analysis
            st.subheader("Prediction Error Analysis")
            
            # Error distribution
            error_fig = px.histogram(
                df_pred,
                x='error_pct',
                nbins=50,
                title="Prediction Error Distribution",
                labels={'error_pct': 'Prediction Error (%)'}
            )
            st.plotly_chart(error_fig, use_container_width=True)
            
            # Error over time
            error_time_fig = go.Figure()
            error_time_fig.add_trace(
                go.Scatter(
                    x=df_pred['date'],
                    y=df_pred['error_pct'],
                    mode='lines+markers',
                    name='Error %',
                    line=dict(color='red', width=2),
                    marker=dict(size=6, color='red', symbol='circle')
                )
            )
            error_time_fig.update_layout(
                title="Prediction Error Over Time",
                xaxis_title="Date",
                yaxis_title="Error (%)",
                hovermode="x unified"
            )
            st.plotly_chart(error_time_fig, use_container_width=True)
            
            # Error by day of week
            error_day_fig = px.box(
                df_pred,
                x='day',
                y='error_pct',
                title="Prediction Error by Day of Week",
                labels={'day': 'Day of Week', 'error_pct': 'Prediction Error (%)'}
            )
            st.plotly_chart(error_day_fig, use_container_width=True)
            
            # Error by month
            error_month_fig = px.box(
                df_pred,
                x='month',
                y='error_pct',
                title="Prediction Error by Month",
                labels={'month': 'Month', 'error_pct': 'Prediction Error (%)'}
            )
            st.plotly_chart(error_month_fig, use_container_width=True)
        
        with pred_tabs[2]:
            # Time patterns
            st.subheader("Time-Based Patterns in Predictions")
            
            # Average error by day of week
            avg_error_day = df_pred.groupby('day')['error_pct'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            avg_error_day_fig = px.bar(
                avg_error_day,
                title="Average Prediction Error by Day of Week",
                labels={'index': 'Day of Week', 'value': 'Average Error (%)'}
            )
            st.plotly_chart(avg_error_day_fig, use_container_width=True)
            
            # Average error by month
            avg_error_month = df_pred.groupby('month')['error_pct'].mean().reindex([
                'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
            ])
            avg_error_month_fig = px.bar(
                avg_error_month,
                title="Average Prediction Error by Month",
                labels={'index': 'Month', 'value': 'Average Error (%)'}
            )
            st.plotly_chart(avg_error_month_fig, use_container_width=True)
        
        with pred_tabs[3]:
            # Prediction log
            st.subheader("Detailed Prediction Log")
            
            # Display the DataFrame
            st.dataframe(df_pred)
            
            # Allow download of the log
            csv = df_pred.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_log.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in prediction explorer: {e}")
        logger.error(f"Error in prediction explorer: {e}", exc_info=True)

@robust_error_boundary
def create_model_comparison():
    """Create a comparison of different models"""
    if 'model_comparison_data' not in st.session_state:
        st.session_state['model_comparison_data'] = []
    
    comparison_data = st.session_state['model_comparison_data']
    if not comparison_data:
        st.info("No model comparison data available.")
        return
    
    try:
        # Convert to DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display the DataFrame
        st.dataframe(df_comparison)
        
        # Create comparison plots
        comparison_tabs = st.tabs([
            "RMSE Comparison", 
            "MAPE Comparison", 
            "Training Time Comparison"
        ])
        
        with comparison_tabs[0]:
            # RMSE comparison
            rmse_fig = px.bar(
                df_comparison,
                x='model_name',
                y='rmse',
                title="RMSE Comparison",
                labels={'model_name': 'Model', 'rmse': 'RMSE'}
            )
            st.plotly_chart(rmse_fig, use_container_width=True)
        
        with comparison_tabs[1]:
            # MAPE comparison
            mape_fig = px.bar(
                df_comparison,
                x='model_name',
                y='mape',
                title="MAPE Comparison",
                labels={'model_name': 'Model', 'mape': 'MAPE (%)'}
            )
            st.plotly_chart(mape_fig, use_container_width=True)
        
        with comparison_tabs[2]:
            # Training time comparison
            time_fig = px.bar(
                df_comparison,
                x='model_name',
                y='training_time',
                title="Training Time Comparison",
                labels={'model_name': 'Model', 'training_time': 'Training Time (s)'}
            )
            st.plotly_chart(time_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in model comparison: {e}")
        logger.error(f"Error in model comparison: {e}", exc_info=True)

@robust_error_boundary
def create_control_panel():
    """Create a control panel for user inputs"""
    st.sidebar.header("Control Panel")
    
    # Select ticker
    ticker = st.sidebar.selectbox(
        "Select Ticker",
        TICKERS,
        index=TICKERS.index(st.session_state.get("selected_ticker", TICKER))
    )
    st.session_state["selected_ticker"] = ticker
    
    # Select timeframe
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        TIMEFRAMES,
        index=TIMEFRAMES.index(st.session_state.get("selected_timeframe", TIMEFRAMES[0]))
    )
    st.session_state["selected_timeframe"] = timeframe
    
    # Select date range
    start_date = st.sidebar.date_input(
        "Start Date",
        value=st.session_state.get("start_date", datetime.now() - timedelta(days=30))
    )
    st.session_state["start_date"] = start_date
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=st.session_state.get("end_date", datetime.now())
    )
    st.session_state["end_date"] = end_date
    
    # Training Start Date
    st.sidebar.subheader("Model Training Settings")

    training_start_date = st.sidebar.date_input(
    "Training Start Date",
    value=st.session_state.get("training_start_date", datetime.now() - timedelta(days=365))
    )
    st.session_state["training_start_date"] = training_start_date

    # Add a note about what this date means
    st.sidebar.info("Training Start Date defines the earliest data point used for model training.")
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=st.session_state.get("auto_refresh", True)
    )
    st.session_state["auto_refresh"] = auto_refresh
    
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=10,
        max_value=300,
        value=st.session_state.get("refresh_interval", 30)
    )
    st.session_state["refresh_interval"] = refresh_interval
    
    # Return selected parameters
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval
    }

@robust_error_boundary
def create_download_section():
    """Create a section for downloading data and models"""
    st.subheader("Download Data and Models")
    
    # Download raw data
    if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
        df_raw = st.session_state['df_raw']
        csv = df_raw.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="raw_data.csv">Download Raw Data (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No raw data available for download.")
    
    # Download model weights
    if 'current_model' in st.session_state and st.session_state['current_model'] is not None:
        model = st.session_state['current_model']
        model_json = model.to_json()
        b64 = base64.b64encode(model_json.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="model.json">Download Model Architecture (JSON)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Save weights to a temporary file
        weights_path = os.path.join(tempfile.gettempdir(), "model_weights.h5")
        model.save_weights(weights_path)
        
        with open(weights_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/h5;base64,{b64}" download="model_weights.h5">Download Model Weights (H5)</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No model available for download.")

if __name__ == "__main__":
    main_dashboard()