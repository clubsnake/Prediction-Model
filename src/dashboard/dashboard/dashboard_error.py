"""
Error handling and logging functionality for dashboard components.
"""

import os
import sys
import traceback
import yaml
import functools
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import streamlit as st
import tensorflow as tf

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

# Type variable for function return values
T = TypeVar('T')

# Custom exception classes
class DashboardError(Exception):
    """Base exception for dashboard errors."""
    pass


class DataFetchError(DashboardError):
    """Exception raised when data fetching fails."""
    pass


class ValidationError(DashboardError):
    """Exception raised for data validation failures."""
    pass


class RenderingError(DashboardError):
    """Exception raised when dashboard rendering fails."""
    pass


class ModelError(DashboardError):
    """Exception raised for model-related failures."""
    pass


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log error details with context information.
    
    Args:
        error: The exception to log
        context: Additional context information about when the error occurred
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
    
    # Format context info for logging
    context_str = ""
    if context:
        context_str = "\nContext:\n" + "\n".join(f"    {k}: {v}" for k, v in context.items())
    
    # Log with stack trace and context
    logger.error(
        f"Error: {str(error)}{context_str}\n"
        f"Stack Trace:\n{''.join(stack_trace)}"
    )


def display_error(e, error_type="Error"):
    """Display a user-friendly error message with details available on demand."""
    st.error(f"⚠️ {error_type}: {str(e)}")
    
    # Add timestamp to key to make it unique across error displays
    error_details_key = f"show_error_details_{datetime.now().timestamp()}"
    
    if st.checkbox("Show technical details", key=error_details_key):
        with st.expander("Technical Details", expanded=True):
            st.code(traceback.format_exc())
            
            if hasattr(e, "__cause__") and e.__cause__ is not None:
                st.write("#### Caused by:")
                st.code(str(e.__cause__))
                
    st.warning("Please try again or contact support if the issue persists.")


def robust_error_boundary(
    func=None, 
    fallback_value=None,
    error_type="Error",
    show_error=True,
    log_errors=True,
    context=None
):
    """
    Decorator to create a robust error boundary around any function.
    
    This wraps function execution in a try/except block and handles errors
    by displaying user-friendly messages and optionally providing fallback values.
    
    Args:
        func: Function to wrap with error handling
        fallback_value: Value to return if function fails
        error_type: Category of error for display
        show_error: Whether to display the error in the UI
        log_errors: Whether to log the error
        context: Additional context information about the function
        
    Returns:
        Wrapped function that handles errors gracefully
        
    Example:
        @robust_error_boundary(fallback_value=pd.DataFrame(), error_type="Data Error")
        def load_data(url):
            return pd.read_csv(url)
    """
    # Support both @robust_error_boundary and @robust_error_boundary()
    if func is None:
        # Called with parentheses and arguments
        return lambda f: robust_error_boundary(
            f, fallback_value, error_type, show_error, log_errors, context
        )
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Add context about the function that failed
            error_context = context or {}
            error_context.update({
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            
            # Log the error
            if log_errors:
                log_error(e, error_context)
                
            # Display the error
            if show_error:
                display_error(e, error_type)
                
            # Return fallback value if provided
            return fallback_value
            
    return wrapper


def handle_api_errors(
    show_error: bool = True,
    error_message: str = "API request failed",
    fallback_value: Any = None
) -> Callable:
    """
    Decorator for handling API-related errors.
    
    Args:
        show_error: Whether to show error in UI
        error_message: Custom error message to display
        fallback_value: Value to return if API call fails
        
    Returns:
        Decorated function with API error handling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                log_error(e, {"api_function": func.__name__})
                
                if show_error:
                    st.error(f"⚠️ {error_message}: {str(e)}")
                    
                return fallback_value
        return wrapper
    return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    fallback_value: Optional[T] = None,
    error_type: str = "Error",
    show_error: bool = True,
    **kwargs
) -> T:
    """
    Safely execute a function with error handling.
    
    This is a non-decorator version of robust_error_boundary for one-off calls.
    
    Args:
        func: Function to execute safely
        *args: Args to pass to the function
        fallback_value: Value to return if function fails
        error_type: Category of error for display
        show_error: Whether to display the error in the UI
        **kwargs: Kwargs to pass to the function
        
    Returns:
        Result of the function or fallback value if it fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Log the error
        log_error(e, {"function": func.__name__})
        
        # Display the error
        if show_error:
            display_error(e, error_type)
            
        # Return fallback value if provided
        return fallback_value


class section_error_boundary:
    """
    Context manager for creating error boundaries around dashboard sections.
    
    Use with a "with" statement to catch errors for a section of the UI:
    
    Example:
        with section_error_boundary("Data Visualization"):
            plot_complex_visualization(data)
    """
    
    def __init__(self, section_name: str):
        self.section_name = section_name
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log_error(exc_val, {"section": self.section_name})
            st.error(f"⚠️ Error in {self.section_name} section")
            with st.expander("Error Details"):
                st.write(f"**Error**: {str(exc_val)}")
                if st.checkbox(f"Show technical details for {self.section_name}"):
                    st.code(traceback.format_exc(), language="python")
            return True  # Suppress the exception
        return False


@robust_error_boundary
def load_latest_progress(ticker=None, timeframe=None):
    """
    Load the most recent model progress from the YAML status file.
    
    Args:
        ticker (str, optional): Specific ticker to get progress for
        timeframe (str, optional): Specific timeframe to get progress for
    
    Returns:
        dict: Progress information
    """
    # Default progress values
    default_progress = {
        "current_trial": 0,
        "total_trials": 0,
        "current_rmse": None,
        "current_mape": None,
        "cycle": 1
    }
    
    # Define the progress file path
    progress_file = os.path.join(DATA_DIR, "progress.yaml")
    
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


@robust_error_boundary
def write_tuning_status(status: bool):
    """
    Write tuning status to file.
    
    Args:
        status: Boolean indicating whether tuning is active
    """
    tuning_status_file = os.path.join(DATA_DIR, "tuning_status.txt")
    with open(tuning_status_file, "w") as f:
        f.write(str(status))


@robust_error_boundary
def read_tuning_status() -> bool:
    """
    Read tuning status from file.
    
    Returns:
        Boolean indicating whether tuning is active
    """
    tuning_status_file = os.path.join(DATA_DIR, "tuning_status.txt")
    if os.path.exists(tuning_status_file):
        with open(tuning_status_file, "r") as f:
            status = f.read().strip()
            return status.lower() == "true"
    return False


@st.cache_data
def load_data(ticker=None, timeframe=None):
    """
    Load data for a ticker and timeframe with proper caching.
    
    Args:
        ticker: The ticker symbol to load data for
        timeframe: The timeframe to load data for
        
    Returns:
        DataFrame: The loaded data
    """
    try:
        from src.data.data_loader import load_market_data
        return load_market_data(ticker=ticker, timeframe=timeframe)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None


def validate_date_range(start_date, end_date):
    """
    Validate date ranges to ensure they're not in the future.
    
    Args:
        start_date: Start date to validate
        end_date: End date to validate
        
    Returns:
        Tuple of validated (start_date, end_date)
    """
    today = datetime.now().date()
    
    # Convert strings to dates if needed
    if isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid start date format: {start_date}, using today - 365 days")
            start_date = today - timedelta(days=365)
            
    if isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid end date format: {end_date}, using today")
            end_date = today
    
    # Check if dates are in the future and adjust if needed
    if start_date > today:
        logger.warning(f"Start date {start_date} is in the future, adjusting to today - 365 days")
        start_date = today - timedelta(days=365)
        
    if end_date > today:
        logger.warning(f"End date {end_date} is in the future, adjusting to today")
        end_date = today
        
    # Ensure start_date is before end_date
    if start_date > end_date:
        logger.warning(f"Start date {start_date} is after end date {end_date}, swapping")
        start_date, end_date = end_date, start_date
        
    return start_date, end_date


def get_callbacks(patience=20, reduce_lr_factor=0.2, monitor='val_loss', mode='min', timescale_logging=True):
    """
    Get training callbacks for model training.
    
    Args:
        patience: Patience parameter for early stopping
        reduce_lr_factor: Factor to reduce learning rate by
        monitor: Metric to monitor
        mode: Direction of improvement ('min' or 'max')
        timescale_logging: Whether to log timescale information
        
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # Add early stopping callback
    try:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                mode=mode
            )
        )
    except Exception as e:
        logger.warning(f"Could not add EarlyStopping callback: {e}")
    
    # Add learning rate reduction callback
    try:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=reduce_lr_factor,
                patience=patience // 2,
                min_lr=1e-6,
                mode=mode
            )
        )
    except Exception as e:
        logger.warning(f"Could not add ReduceLROnPlateau callback: {e}")
    
    # Add TimescaleAnalysis callback if requested and available
    if timescale_logging:
        try:
            from src.models.ltc_model import TimescaleAnalysisCallback
            callbacks.append(TimescaleAnalysisCallback(logging_freq=10))
        except ImportError:
            logger.warning("Could not import TimescaleAnalysisCallback - skipping timescale logging")
        except Exception as e:
            logger.warning(f"Could not add TimescaleAnalysisCallback: {e}")
    
    return callbacks


def global_exception_handler(exctype, value, tb):
    """
    Global exception handler to catch and log unhandled exceptions.
    
    Args:
        exctype: Exception type
        value: Exception value
        tb: Traceback object
    """
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
    sys.__excepthook__(exctype, value, tb)


# Set the global exception handler
sys.excepthook = global_exception_handler


# Function to optimize parallel model training
def configure_parallel_training(num_parallel_models=None):
    """
    Configure system for efficient parallel model training.
    
    Args:
        num_parallel_models: Number of models to train in parallel (None = auto-detect)
        
    Returns:
        Dictionary with configuration settings
    """
    import multiprocessing
    import tensorflow as tf
    
    # Auto-detect optimal settings if not specified
    cpu_count = multiprocessing.cpu_count()
    
    if num_parallel_models is None:
        # Set based on CPU count (guideline: use ~75% of available cores)
        num_parallel_models = max(1, int(cpu_count * 0.75))
    
    # Configure TensorFlow for parallel processing
    gpus = tf.config.list_physical_devices('GPU')
    
    config = {
        "num_parallel_models": num_parallel_models,
        "cpu_count": cpu_count,
        "gpu_count": len(gpus),
        "thread_count": max(1, int(cpu_count / num_parallel_models)),
        "inter_op_parallelism": min(4, max(1, int(cpu_count / (2 * num_parallel_models))))
    }
    
    # Apply configuration
    tf.config.threading.set_inter_op_parallelism_threads(config["inter_op_parallelism"])
    tf.config.threading.set_intra_op_parallelism_threads(config["thread_count"])
    
    logger.info(f"Configured for parallel training: {config}")
    return config
