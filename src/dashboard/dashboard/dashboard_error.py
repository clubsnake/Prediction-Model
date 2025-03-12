"""
Centralized error handling utilities for the dashboard.

This module provides error boundaries and status tracking functions
to ensure the dashboard remains operational even when components fail.
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

# Add project root to Python path for reliable imports
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logger
logger = logging.getLogger("dashboard_error")

# Path configurations
try:
    # Only import at module level what we truly need
    from config.config_loader import get_config
    config = get_config()
    DATA_DIR = config.get("DATA_DIR", os.path.join(project_root, "data"))
except ImportError:
    # Fallback if config import fails
    DATA_DIR = os.path.join(project_root, "data")
    logger.warning("Could not import config, using default DATA_DIR")


def robust_error_boundary(func: Callable) -> Callable:
    """
    A decorator that creates a robust error boundary around any function.
    Catches exceptions, logs them, and provides helpful error messages.
    
    Args:
        func: The function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get function name and error details
            func_name = getattr(func, "__name__", "unknown_function")
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            
            # Log the error
            logger.error(
                f"Error in {func_name}: {error_msg}\n{error_traceback}"
            )
            
            # If streamlit is available, display error to user
            try:
                import streamlit as st
                st.error(f"Error in {func_name}: {error_msg}")
                
                # Show traceback in expander for debugging
                with st.expander("Show detailed error information"):
                    st.code(error_traceback)
                    
            except ImportError:
                # If streamlit not available, just print to console
                print(f"ERROR in {func_name}: {error_msg}")
                
            # Return None or appropriate fallback value
            return None
    
    return wrapper


def validate_file_path(file_path: str) -> bool:
    """
    Validate that a file path exists and is accessible.
    Creates any necessary parent directories.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error validating path {file_path}: {e}")
        return False


@robust_error_boundary
def read_tuning_status() -> Dict[str, Any]:
    """
    Read current tuning status from file.
    
    Returns:
        Dictionary with status information
    """
    # Import inside function to avoid circular imports
    import yaml
    
    status_file = os.path.join(DATA_DIR, "tuning_status.txt")
    
    if not os.path.exists(status_file):
        # Return default status if file doesn't exist
        return {
            "status": "idle",
            "is_running": False,
            "ticker": None,
            "timeframe": None,
            "last_updated": datetime.now().isoformat()
        }
    
    try:
        with open(status_file, "r") as f:
            status_data = yaml.safe_load(f)
            
        # Ensure is_running is a boolean
        if "is_running" in status_data:
            if isinstance(status_data["is_running"], str):
                status_data["is_running"] = (status_data["is_running"].lower() == "true")
        else:
            status_data["is_running"] = False
            
        return status_data
        
    except Exception as e:
        logger.error(f"Error reading tuning status: {e}")
        return {
            "status": "unknown",
            "is_running": False,
            "error": str(e)
        }


@robust_error_boundary
def write_tuning_status(ticker: str, timeframe: str, is_running: bool = False) -> bool:
    """
    Write current tuning status to file.
    
    Args:
        ticker: Ticker symbol being tuned
        timeframe: Timeframe being tuned
        is_running: Whether tuning is currently running
        
    Returns:
        True if successful, False otherwise
    """
    # Import inside function to avoid circular imports
    import yaml
    from src.utils.threadsafe import safe_file_write
    
    status_file = os.path.join(DATA_DIR, "tuning_status.txt")
    
    # Create data directory if it doesn't exist
    validate_file_path(status_file)
    
    status_data = {
        "status": "running" if is_running else "idle",
        "is_running": is_running,
        "ticker": ticker,
        "timeframe": timeframe,
        "start_time": datetime.now().isoformat() if is_running else None,
        "last_updated": datetime.now().isoformat()
    }
    
    # Use thread-safe file writing
    try:
        yaml_content = yaml.dump(status_data)
        safe_file_write(status_file, yaml_content)
        
        # Also update streamlit session state if available
        try:
            import streamlit as st
            st.session_state["tuning_in_progress"] = is_running
            st.session_state["tuning_ticker"] = ticker
            st.session_state["tuning_timeframe"] = timeframe
        except Exception:
            # If not in a streamlit context, just ignore
            pass
            
        return True
    except Exception as e:
        logger.error(f"Error writing tuning status: {e}")
        return False


@robust_error_boundary
def load_latest_progress(
    ticker: Optional[str] = None, 
    timeframe: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load the latest tuning progress from YAML file.
    
    Args:
        ticker: Optional ticker to filter progress
        timeframe: Optional timeframe to filter progress
        
    Returns:
        Dictionary with progress information
    """
    # Import inside function to avoid circular imports
    import yaml
    
    progress_file = os.path.join(DATA_DIR, "progress.yaml")
    
    if not os.path.exists(progress_file):
        # Return default progress if file doesn't exist
        return {
            "current_trial": 0,
            "total_trials": 1,
            "current_rmse": None,
            "current_mape": None,
            "best_rmse": None,
            "best_mape": None,
            "cycle": 1,
        }
    
    try:
        with open(progress_file, "r") as f:
            progress_data = yaml.safe_load(f) or {}
        
        # If ticker and timeframe specified, check if they match
        if ticker and timeframe:
            file_ticker = progress_data.get("ticker")
            file_timeframe = progress_data.get("timeframe")
            
            if (file_ticker != ticker or file_timeframe != timeframe) and "cycle" in progress_data:
                logger.info(
                    f"Progress file contains different ticker/timeframe: "
                    f"{file_ticker}/{file_timeframe} vs requested {ticker}/{timeframe}"
                )
                
        # Ensure numeric values are proper numbers
        for key in ["current_trial", "total_trials", "cycle"]:
            if key in progress_data and not isinstance(progress_data[key], (int, float)):
                try:
                    progress_data[key] = int(progress_data[key])
                except (TypeError, ValueError):
                    progress_data[key] = 0
        
        for key in ["current_rmse", "current_mape", "best_rmse", "best_mape"]:
            if key in progress_data and progress_data[key] is not None:
                try:
                    progress_data[key] = float(progress_data[key])
                except (TypeError, ValueError):
                    progress_data[key] = None
        
        return progress_data
        
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
        return {
            "current_trial": 0,
            "total_trials": 1,
            "current_rmse": None,
            "current_mape": None,
            "cycle": 1,
            "error": str(e)
        }


@robust_error_boundary
def write_progress(progress_data: Dict[str, Any]) -> bool:
    """
    Write progress data to YAML file.
    
    Args:
        progress_data: Dictionary of progress data
        
    Returns:
        True if successful, False otherwise
    """
    # Import inside function to avoid circular imports
    import yaml
    from src.utils.threadsafe import safe_file_write
    
    progress_file = os.path.join(DATA_DIR, "progress.yaml")
    
    # Create data directory if it doesn't exist
    validate_file_path(progress_file)
    
    # Add timestamp
    progress_data["timestamp"] = datetime.now().isoformat()
    
    # Use thread-safe file writing
    try:
        yaml_content = yaml.dump(progress_data)
        safe_file_write(progress_file, yaml_content)
        
        # Also update streamlit session state if available
        try:
            import streamlit as st
            if "best_rmse" in progress_data and progress_data["best_rmse"] is not None:
                if "best_metrics" not in st.session_state:
                    st.session_state["best_metrics"] = {}
                st.session_state["best_metrics"]["rmse"] = progress_data["best_rmse"]
                
            if "best_mape" in progress_data and progress_data["best_mape"] is not None:
                if "best_metrics" not in st.session_state:
                    st.session_state["best_metrics"] = {}
                st.session_state["best_metrics"]["mape"] = progress_data["best_mape"]
        except Exception:
            # If not in a streamlit context, just ignore
            pass
            
        return True
    except Exception as e:
        logger.error(f"Error writing progress: {e}")
        return False


@robust_error_boundary
def log_exception(error: Exception, context: str = "") -> None:
    """
    Log an exception with context.
    
    Args:
        error: The exception to log
        context: Additional context about where the error occurred
    """
    error_msg = str(error)
    error_traceback = traceback.format_exc()
    
    # Log to file
    logger.error(f"Exception in {context}: {error_msg}\n{error_traceback}")
    
    # Try to display in streamlit if available
    try:
        import streamlit as st
        if "error_log" not in st.session_state:
            st.session_state["error_log"] = []
            
        # Add to session state error log
        st.session_state["error_log"].append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "message": error_msg,
            "traceback": error_traceback
        })
        
        # Keep only the last 20 errors
        if len(st.session_state["error_log"]) > 20:
            st.session_state["error_log"] = st.session_state["error_log"][-20:]
    except ImportError:
        # If not in streamlit context, just continue
        pass


@robust_error_boundary
def display_error_log() -> None:
    """
    Display the error log in streamlit.
    """
    try:
        import streamlit as st
        
        if "error_log" not in st.session_state or not st.session_state["error_log"]:
            st.info("No errors logged.")
            return
            
        error_log = st.session_state["error_log"]
        
        st.markdown("## Recent Errors")
        
        for i, error in enumerate(reversed(error_log)):
            with st.expander(f"{error['timestamp']} - {error['context']}"):
                st.error(error["message"])
                st.code(error["traceback"])
                
        # Add button to clear errors
        if st.button("Clear Error Log"):
            st.session_state["error_log"] = []
            st.success("Error log cleared!")
            st.experimental_rerun()
                
    except ImportError:
        print("Streamlit not available for displaying error log")
