"""
dashboard_error.py

Error handling and logging functionality for dashboard components.
"""

import os
import sys
import traceback
import yaml
from datetime import datetime
from functools import wraps

import streamlit as st

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from config
try:
    from config.config_loader import PROGRESS_FILE, DATA_DIR
    from config.logger_config import logger
except ImportError:
    # Fallback if import fails
    print("Warning: Could not import from config module. Using fallback values.")
    DATA_DIR = os.path.join(project_root, "data")
    PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
    
    # Set up basic logger as fallback
    import logging
    logger = logging.getLogger("dashboard")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)


def log_exception(message):
    """Log an exception with details"""
    logger.error(message)
    logger.error(traceback.format_exc())

def error_handler(func):
    """Decorator to handle errors in functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_exception(f"Error in {func.__name__}: {str(e)}")
            
            # Initialize error log if it doesn't exist
            if "error_log" not in st.session_state:
                st.session_state["error_log"] = []
                
            # Add error to log
            st.session_state["error_log"].append({
                "function": func.__name__,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            return None
    return wrapper


@error_handler
def load_latest_progress(ticker=None, timeframe=None):
    """
    Load the most recent model progress from the YAML status file.
    
    Args:
        ticker (str, optional): Specific ticker to get progress for
        timeframe (str, optional): Specific timeframe to get progress for
    
    Returns:
        dict: Progress information
    """
    try:
        # Define the progress file path
        progress_file = os.path.join(DATA_DIR, "progress.yaml")
        
        # Default progress values
        default_progress = {
            "current_trial": 0,
            "total_trials": 0,
            "current_rmse": None,
            "current_mape": None,
            "cycle": 1
        }
        
        # Check if file exists
        if not os.path.exists(progress_file):
            logger.warning(f"Progress file not found: {progress_file}")
            return default_progress
            
        # Load progress data from YAML file
        with open(progress_file, "r") as f:
            try:
                status_info = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error parsing progress file: {e}")
                return default_progress
                
        # If YAML is empty or not a dict
        if not status_info or not isinstance(status_info, dict):
            return default_progress
        
        # If ticker and timeframe specified, try to find matching progress
        if ticker and timeframe:
            models = status_info.get("models", [])
            
            # If models is not a list, convert to list with single entry
            if not isinstance(models, list):
                models = [models] if models else []
                
            for model_info in models:
                if isinstance(model_info, dict) and model_info.get("ticker") == ticker and model_info.get("timeframe") == timeframe:
                    return {
                        "current_trial": model_info.get("current_trial", 0),
                        "total_trials": model_info.get("total_trials", 0),
                        "current_rmse": model_info.get("current_rmse"),
                        "current_mape": model_info.get("current_mape"),
                        "cycle": model_info.get("cycle", 1)
                    }
        
        # Fall back to first model or global info
        if "models" in status_info and isinstance(status_info["models"], list) and status_info["models"]:
            model = status_info["models"][0]
            return {
                "current_trial": model.get("current_trial", 0),
                "total_trials": model.get("total_trials", 0),
                "current_rmse": model.get("current_rmse"),
                "current_mape": model.get("current_mape"),
                "cycle": model.get("cycle", 1)
            }
        
        # Default fallback using global data if available
        return {
            "current_trial": status_info.get("current_trial", 0),
            "total_trials": status_info.get("total_trials", 0),
            "current_rmse": status_info.get("current_rmse"),
            "current_mape": status_info.get("current_mape"),
            "cycle": status_info.get("cycle", 1)
        }
    except Exception as e:
        logger.error(f"Error loading progress information: {e}")
        return default_progress


@error_handler
def write_tuning_status(status: bool):
    """Write tuning status to file"""
    tuning_status_file = os.path.join(DATA_DIR, "tuning_status.txt")
    try:
        with open(tuning_status_file, "w") as f:
            f.write(str(status))
    except Exception as e:
        logger.error(f"Error writing tuning status: {e}")


@error_handler
def read_tuning_status() -> bool:
    """Read tuning status from file"""
    tuning_status_file = os.path.join(DATA_DIR, "tuning_status.txt")
    try:
        if os.path.exists(tuning_status_file):
            with open(tuning_status_file, "r") as f:
                status = f.read().strip()
                return status.lower() == "true"
        return False
    except Exception as e:
        logger.error(f"Error reading tuning status: {e}")
        return False


@error_handler
def global_exception_handler(exctype, value, tb):
    """Global exception handler to catch and log unhandled exceptions"""
    logger.error(f"Unhandled exception: {value}", exc_info=(exctype, value, tb))
    if "error_log" in st.session_state:
        st.session_state["error_log"].append(
            {
                "timestamp": datetime.now(),
                "function": "global",
                "error": str(value),
                "traceback": traceback.format_exc(),
            }
        )
    # Call the original exception handler
    import sys
    sys.__excepthook__(exctype, value, tb)


# Set the global exception handler
import sys
sys.excepthook = global_exception_handler