"""
Progress tracking utilities for hyperparameter tuning and model training.

This module provides functions to track and manage progress information during:
1. Hyperparameter tuning with Optuna
2. Individual model training
3. Ensemble model training and weighting
4. Adaptation from market regime and incremental learning

The progress information is stored in:
- progress.yaml: Overall progress including ensemble metrics and completion
- model_progress/[model_type]_progress.yaml: Individual model metrics
- tested_models/[model_type]_tested_models.yaml: Models tested during tuning
- cycle_metrics.yaml: Metrics for each tuning cycle
"""

import os
import sys
import time
import logging
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import yaml
import numpy as np

# Import the robust FileLock implementation
from src.utils.threadsafe import FileLock

# Configure logging
logger = logging.getLogger(__name__)

# Import file paths from config_loader instead of redefining them
try:
    from config.config_loader import (
        DATA_DIR, 
        PROGRESS_FILE,
        TUNING_STATUS_FILE,
        CYCLE_METRICS_FILE,
        BEST_PARAMS_FILE,
        MODEL_PROGRESS_DIR,
        MODELS_DIR,
        HYPERPARAMS_DIR,
        TESTED_MODELS_FILE,  # Add this import 
    )
except ImportError:
    # Fallback to original definitions if import fails
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    DATA_DIR = os.path.join(project_root, "data")  # Use lowercase for consistency
    PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
    TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")
    CYCLE_METRICS_FILE = os.path.join(DATA_DIR, "cycle_metrics.yaml")
    MODELS_DIR = os.path.join(DATA_DIR, "models")  # Use lowercase for consistency
    HYPERPARAMS_DIR = os.path.join(MODELS_DIR, "hyperparams")  # Use lowercase for consistency
    BEST_PARAMS_FILE = os.path.join(HYPERPARAMS_DIR, "best_params.yaml")
    MODEL_PROGRESS_DIR = os.path.join(DATA_DIR, "model_progress")
    TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")  # Add this fallback definition
    logger.warning("Could not import file paths from config_loader. Using fallback definitions.")

# Create directories for model-specific files
TESTED_MODELS_DIR = os.path.join(DATA_DIR, "tested_models")
os.makedirs(TESTED_MODELS_DIR, exist_ok=True)
os.makedirs(MODEL_PROGRESS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)

# Thread safety helpers
_lock = threading.RLock()
stop_event = threading.Event()
_current_cycle = 1  # Global tracking of current cycle

# Export TESTED_MODELS_FILE at module level so it can be imported by other modules
__all__ = [
    "PROGRESS_FILE", 
    "TUNING_STATUS_FILE", 
    "CYCLE_METRICS_FILE", 
    "BEST_PARAMS_FILE", 
    "MODEL_PROGRESS_DIR", 
    "TESTED_MODELS_DIR", 
    "TESTED_MODELS_FILE",  # Add this to exports
    "DATA_DIR"
]

# ==========================================
# Progress File Path Management
# ==========================================

def get_model_progress_file(model_type: str) -> str:
    """Get path to the progress file for a specific model type."""
    return os.path.join(MODEL_PROGRESS_DIR, f"{model_type}_progress.yaml")

def get_model_tested_file(model_type: str) -> str:
    """Get path to the tested models file for a specific model type."""
    return os.path.join(TESTED_MODELS_DIR, f"{model_type}_tested_models.yaml")

# ==========================================
# File Operations with Thread Safety
# ==========================================

def safe_read_yaml(file_path: str, default: Any = None) -> Any:
    """Read YAML file with proper error handling and thread safety."""
    if not os.path.exists(file_path):
        return default if default is not None else {}
    
    try:
        with FileLock(file_path):
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or default
    except (yaml.YAMLError, TimeoutError, FileNotFoundError) as e:
        logger.error(f"Error reading {file_path}: {e}")
        return default if default is not None else {}

def safe_write_yaml(file_path: str, data: Any) -> bool:
    """Write data to YAML file with proper error handling and thread safety."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Validate and clean data before serialization to prevent corruption
        data = sanitize_yaml_data(data)
        
        # Create temp file first
        temp_file = f"{file_path}.tmp.{int(time.time())}"
        with open(temp_file, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Use atomic rename for thread safety
        with FileLock(file_path):
            os.replace(temp_file, file_path)
        
        return True
    except (yaml.YAMLError, TimeoutError, OSError) as e:
        logger.error(f"Error writing {file_path}: {e}")
        return False

def sanitize_yaml_data(data: Any) -> Any:
    """
    Sanitize data before YAML serialization to prevent corruption.
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized data safe for YAML serialization
    """
    # Convert to native types first
    data = convert_to_native_types(data)
    
    if isinstance(data, dict):
        # Create a new dict with sanitized keys and values
        result = {}
        for k, v in data.items():
            # Ensure keys are strings, not None or other types
            if k is None:
                k = "null_key"
            elif not isinstance(k, str):
                k = str(k)
                
            # Remove problematic characters from keys
            if isinstance(k, str):
                # Replace any characters that might cause YAML issues
                k = k.replace('\t', ' ').replace('\n', ' ')
                
            # Recursively sanitize values
            v = sanitize_yaml_data(v)
            
            result[k] = v
        return result
    elif isinstance(data, list):
        # Recursively sanitize list items
        return [sanitize_yaml_data(item) for item in data]
    elif isinstance(data, (str, bytes)):
        # Ensure string values don't have problematic characters
        if isinstance(data, bytes):
            try:
                data = data.decode('utf-8')
            except UnicodeDecodeError:
                data = str(data)
        
        # Replace tabs and newlines in strings as they can cause indentation issues
        return data.replace('\t', ' ').replace('\r\n', '\n')
    elif data is None or isinstance(data, (int, float, bool)):
        # These types are safe for YAML
        return data
    else:
        # Convert any other types to strings
        return str(data)

def append_to_yaml_list(file_path: str, item: Any) -> bool:
    """Append an item to a YAML list with thread safety."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with FileLock(file_path):
            # Read existing data
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f) or []
            else:
                data = []
                
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
                
            # Sanitize the item before appending
            item = sanitize_yaml_data(item)
            
            # Append new item
            data.append(item)
            
            # Write back to file using safer serialization
            with open(file_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                f.flush()
                os.fsync(f.fileno())
                
        return True
    except (yaml.YAMLError, TimeoutError, OSError) as e:
        logger.error(f"Error appending to {file_path}: {e}")
        return False

# ==========================================
# Stop Event Management
# ==========================================

def set_stop_requested(val: bool) -> None:
    """Set or clear the global stop request flag."""
    global stop_event
    with _lock:
        if val:
            logger.info("Stop requested via progress_helper")
            stop_event.set()
        else:
            stop_event.clear()

def is_stop_requested() -> bool:
    """Check if stop has been requested."""
    global stop_event
    with _lock:
        return stop_event.is_set()

# ==========================================
# Cycle Management
# ==========================================

def get_current_cycle() -> int:
    """Get the current tuning cycle number."""
    global _current_cycle
    with _lock:
        return _current_cycle

def set_current_cycle(cycle_num: int) -> None:
    """Set the current cycle number globally."""
    global _current_cycle
    with _lock:
        _current_cycle = cycle_num
        
        # Update progress file with new cycle
        progress_data = read_progress_from_yaml()
        progress_data["cycle"] = cycle_num
        update_progress_in_yaml(progress_data)
        
        logger.info(f"Set current cycle to {cycle_num}")

# ==========================================
# Core Progress Tracking
# ==========================================

def read_progress_from_yaml() -> Dict[str, Any]:
    """Read current progress information from progress.yaml file."""
    global _current_cycle
    
    progress_data = safe_read_yaml(PROGRESS_FILE, default={})
    
    # Add default values for missing fields
    if not progress_data:
        # Try to get ticker/timeframe from tuning status
        ticker = None
        timeframe = None
        try:
            status = read_tuning_status()
            ticker = status.get("ticker")
            timeframe = status.get("timeframe")
        except Exception:
            pass
            
        progress_data = {
            "current_trial": 0,
            "total_trials": 1,
            "current_rmse": None,
            "current_mape": None,
            "best_rmse": None,
            "best_mape": None,
            "cycle": _current_cycle,
            "timestamp": time.time(),
            "ticker": ticker,
            "timeframe": timeframe
        }
    
    # Validate numeric fields
    for key in ["current_trial", "total_trials", "cycle"]:
        if key not in progress_data:
            progress_data[key] = 0 if key == "current_trial" else 1
        elif not isinstance(progress_data[key], (int, float)):
            try:
                progress_data[key] = int(progress_data[key])
            except (ValueError, TypeError):
                progress_data[key] = 0 if key == "current_trial" else 1
    
    # Ensure model_trials exists if not present
    if "model_trials" not in progress_data:
        progress_data["model_trials"] = {}
    
    # IMPROVED: Make sure current_trial and total_trials always use aggregated values if available
    if "aggregated_current_trial" in progress_data and "aggregated_total_trials" in progress_data:
        progress_data["current_trial"] = progress_data["aggregated_current_trial"]
        progress_data["total_trials"] = progress_data["aggregated_total_trials"]
    
    # Update global cycle
    _current_cycle = progress_data.get("cycle", 1)
    
    # Check if ticker/timeframe are present, if not add from tuning status
    if "ticker" not in progress_data or "timeframe" not in progress_data:
        try:
            status = read_tuning_status()
            if "ticker" not in progress_data and "ticker" in status:
                progress_data["ticker"] = status.get("ticker")
            if "timeframe" not in progress_data and "timeframe" in status:
                progress_data["timeframe"] = status.get("timeframe")
        except Exception:
            pass
    
    return progress_data

# New function to provide consistent file path access
def get_progress_file_path(file_type: str, model_type: Optional[str] = None) -> str:
    """
    Get the correct file path for various progress-related files.
    
    Args:
        file_type: Type of file ('progress', 'status', 'metrics', 'best_params', etc.)
        model_type: Model type for model-specific files
        
    Returns:
        str: Path to the requested file
    """
    if file_type == 'progress':
        return PROGRESS_FILE
    elif file_type == 'status':
        return TUNING_STATUS_FILE
    elif file_type == 'metrics':
        return CYCLE_METRICS_FILE
    elif file_type == 'best_params':
        return BEST_PARAMS_FILE
    elif file_type == 'model_progress' and model_type:
        return get_model_progress_file(model_type)
    elif file_type == 'tested_models' and model_type:
        return get_model_tested_file(model_type)
    else:
        logger.warning(f"Unknown file type: {file_type}")
        return os.path.join(DATA_DIR, f"{file_type}.yaml")

# Add a debug logging function to track progress file changes
def _log_progress_change(file_path, message, data=None):
    """Log when progress data is being modified for debugging."""
    logger.debug(f"PROGRESS UPDATE [{os.path.basename(file_path)}]: {message}")
    if data and logger.isEnabledFor(logging.DEBUG):
        # Only log data details if debug is enabled
        if isinstance(data, dict):
            # Log key fields for easier debugging
            fields_to_log = {}
            for key in ["ticker", "timeframe", "cycle", "current_trial", "total_trials", 
                      "aggregated_current_trial", "aggregated_total_trials", "model_type"]:
                if key in data:
                    fields_to_log[key] = data[key]
            logger.debug(f"PROGRESS DATA: {fields_to_log}")
        else:
            logger.debug(f"PROGRESS DATA: {type(data)}")

# Enhanced file lock with progress change tracking
class ProgressFileLock(FileLock):
    """Enhanced file lock that tracks progress changes."""
    
    def __init__(self, filepath, operation="update", timeout=10.0, retry_delay=0.1):
        super().__init__(filepath)
        self.operation = operation
        self.filepath = filepath
        
    def __enter__(self):
        result = super().__enter__()
        _log_progress_change(self.filepath, f"Acquired lock for {self.operation}")
        return result
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        _log_progress_change(self.filepath, f"Released lock for {self.operation}")
        return super().__exit__(exc_type, exc_val, exc_tb)

def update_progress_in_yaml(progress_data: Dict[str, Any], model_type: Optional[str] = None, aggregate_trials: bool = True) -> bool:
    """
    Update progress data in the appropriate YAML file.
    
    Args:
        progress_data: Dictionary with progress information
        model_type: If provided, updates model-specific progress file
        aggregate_trials: Whether to trigger aggregation after model update
    
    Returns:
        bool: Success status
    """
    # Ensure a timestamp is present
    if "timestamp" not in progress_data:
        progress_data["timestamp"] = time.time()
    
    # Use the global cycle
    progress_data["cycle"] = get_current_cycle()
    
    # Check if we have ticker/timeframe - if not, try to get from existing data
    if "ticker" not in progress_data or "timeframe" not in progress_data:
        try:
            # Try to get from tuning status
            current_status = read_tuning_status()
            if "ticker" not in progress_data and "ticker" in current_status:
                progress_data["ticker"] = current_status["ticker"]
            if "timeframe" not in progress_data and "timeframe" in current_status:
                progress_data["timeframe"] = current_status["timeframe"]
        except Exception as e:
            logger.warning(f"Could not get ticker/timeframe from status: {e}")
    
    # Handle model-specific update
    if model_type:
        # Write to model-specific progress file
        model_progress_file = get_model_progress_file(model_type)
        _log_progress_change(model_progress_file, f"Updating model-specific progress for {model_type}", progress_data)
        
        # Use enhanced lock specifically for model progress
        with ProgressFileLock(model_progress_file, operation=f"update_model_{model_type}"):
            # Read existing data to preserve fields
            existing_data = safe_read_yaml(model_progress_file, default={})
            
            # Make sure we preserve ticker/timeframe from existing data if not in new data
            if "ticker" not in progress_data and "ticker" in existing_data:
                progress_data["ticker"] = existing_data["ticker"]
            if "timeframe" not in progress_data and "timeframe" in existing_data:
                progress_data["timeframe"] = existing_data["timeframe"]
                
            # Update with new data
            existing_data.update(progress_data)
            
            # Add model type to data
            existing_data["model_type"] = model_type
            
            # Calculate completion percentage
            if "current_trial" in existing_data and "total_trials" in existing_data:
                existing_data["completion_percentage"] = (
                    existing_data["current_trial"] / max(1, existing_data["total_trials"])
                ) * 100
            
            # Write to model-specific file
            model_success = safe_write_yaml(model_progress_file, existing_data)
        
        # FIXED: Always perform aggregation after a model update
        # We do this outside the lock to avoid deadlocks
        if model_success and aggregate_trials:
            _log_progress_change(PROGRESS_FILE, f"Triggering aggregation after model update for {model_type}")
            aggregation_success = _update_aggregated_progress()
            return aggregation_success  # Return the aggregation result since that's most important
        return model_success
    
    # Handle main progress file update
    _log_progress_change(PROGRESS_FILE, "Updating main progress file", progress_data)
    
    # Use enhanced lock for main progress file
    with ProgressFileLock(PROGRESS_FILE, operation="update_main_progress"):
        existing_data = safe_read_yaml(PROGRESS_FILE, default={})
        
        # Preserve important fields
        for field in ["ticker", "timeframe", "model_trials", "aggregated_total_trials", "aggregated_current_trial"]:
            if field not in progress_data and field in existing_data:
                progress_data[field] = existing_data[field]
        
        # Merge data
        existing_data.update(progress_data)
        
        # IMPROVED: Ensure aggregated values are consistently used as the main values
        # If we have aggregated values, use them as the main trial counts
        if "aggregated_total_trials" in existing_data:
            existing_data["total_trials"] = existing_data["aggregated_total_trials"]
        if "aggregated_current_trial" in existing_data:
            existing_data["current_trial"] = existing_data["aggregated_current_trial"]
        
        # Write merged data
        result = safe_write_yaml(PROGRESS_FILE, existing_data)
    
    # Debug logging
    logger.debug(f"Progress update result for main progress file: {result}")
    logger.debug(f"Progress data contains ticker={existing_data.get('ticker', 'None')}, timeframe={existing_data.get('timeframe', 'None')}")
    
    return result


def _update_aggregated_progress():
    """
    Update aggregated progress by combining metrics from all model types.
    Uses the new AtomicMultiFileUpdate to ensure consistency.
    """
    try:
        logger.info("Starting aggregated progress update")
        
        # Get active model types
        try:
            from config.config_loader import ACTIVE_MODEL_TYPES
            model_types = ACTIVE_MODEL_TYPES
        except ImportError:
            # Scan the model_progress directory
            model_types = []
            for file in os.listdir(MODEL_PROGRESS_DIR):
                if file.endswith("_progress.yaml"):
                    model_type = file.replace("_progress.yaml", "")
                    model_types.append(model_type)
        
        # Get model progress data
        model_progress_files = [get_model_progress_file(model_type) for model_type in model_types]
        model_data_list = {}
        
        # Read all model progress files first
        for model_type, progress_file in zip(model_types, model_progress_files):
            if os.path.exists(progress_file):
                model_data = safe_read_yaml(progress_file, default={})
                if model_data:
                    model_data_list[model_type] = model_data
        
        # If no data found, nothing to aggregate
        if not model_data_list:
            logger.info("No model progress data found, skipping aggregation")
            return True
        
        # Extract ticker/timeframe from first available model's data
        ticker = None
        timeframe = None
        for data in model_data_list.values():
            if "ticker" in data:
                ticker = data["ticker"]
            if "timeframe" in data:
                timeframe = data["timeframe"]
            if ticker and timeframe:
                break
                
        # Prepare aggregated data
        aggregated_data = {
            "aggregated_total_trials": 0,
            "aggregated_current_trial": 0,
            "model_trials": {},
            "cycle": get_current_cycle(),
            "timestamp": time.time(),
            "update_id": int(time.time() * 1000),
            "ticker": ticker,
            "timeframe": timeframe
        }
        
        # Calculate aggregated metrics across all models
        best_rmse = float('inf')
        best_mape = float('inf')
        best_model = None
        weighted_metrics = {"rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0}
        ensemble_weights = {}
        total_weight = 0.0
        
        # Process each model's data
        for model_type, data in model_data_list.items():
            # Get trial counts
            total_trials = data.get("total_trials", 0)
            current_trial = data.get("current_trial", 0)
            
            # Add to aggregated totals
            aggregated_data["aggregated_total_trials"] += total_trials
            aggregated_data["aggregated_current_trial"] += current_trial
            
            # Store in model_trials dict
            aggregated_data["model_trials"][model_type] = {
                "total_trials": total_trials,
                "current_trial": current_trial,
                "completion_percentage": (current_trial / max(1, total_trials)) * 100,
                "rmse": data.get("current_rmse"),
                "mape": data.get("current_mape"),
                "directional_accuracy": data.get("directional_accuracy", 0.0),
                "cycle": data.get("cycle", get_current_cycle())
            }
            
            # Track best metrics
            rmse = data.get("current_rmse")
            mape = data.get("current_mape")
            
            if rmse is not None and rmse < best_rmse:
                best_rmse = rmse
                best_model = model_type
                
            if mape is not None and mape < best_mape:
                best_mape = mape
            
            # Get model weight for ensemble calculations
            weight = data.get("weight", 0.0)
            if weight > 0:
                ensemble_weights[model_type] = weight
                total_weight += weight
                
                # Add to weighted metrics
                if rmse is not None:
                    weighted_metrics["rmse"] += rmse * weight
                if mape is not None:
                    weighted_metrics["mape"] += mape * weight
                da = data.get("directional_accuracy", 0.0)
                if da is not None:
                    weighted_metrics["directional_accuracy"] += da * weight
        
        # Normalize weighted metrics if we have weights
        if total_weight > 0:
            normalized_weights = {
                model: weight/total_weight for model, weight in ensemble_weights.items()
            }
            
            # Add to aggregated data
            aggregated_data["ensemble_weights"] = ensemble_weights
            aggregated_data["normalized_weights"] = normalized_weights
            
            # Normalize metrics
            for metric in weighted_metrics:
                weighted_metrics[metric] /= total_weight
                
            aggregated_data["weighted_metrics"] = weighted_metrics
            aggregated_data["ensemble_rmse"] = weighted_metrics["rmse"]
            aggregated_data["ensemble_mape"] = weighted_metrics["mape"]
            aggregated_data["ensemble_directional_accuracy"] = weighted_metrics["directional_accuracy"]
        
        # Add best metrics to aggregated data
        aggregated_data["best_rmse"] = best_rmse if best_rmse != float('inf') else None
        aggregated_data["best_mape"] = best_mape if best_mape != float('inf') else None
        aggregated_data["best_model"] = best_model
        
        # Set main trial counts for consistency
        aggregated_data["total_trials"] = aggregated_data["aggregated_total_trials"]
        aggregated_data["current_trial"] = aggregated_data["aggregated_current_trial"]
        
        # Also get tuning status to keep it in sync
        tuning_status = read_tuning_status()
        
        # Prepare updated status that matches aggregated data
        updated_status = {
            "is_running": tuning_status.get("is_running", False),
            "status": tuning_status.get("status", "idle"),
            "ticker": ticker,
            "timeframe": timeframe,
            "current_trial": aggregated_data["current_trial"],
            "total_trials": aggregated_data["total_trials"],
            "timestamp": datetime.now().isoformat(),
            "update_id": int(time.time() * 1000)
        }
        
        # Update both progress and status files atomically
        from src.utils.threadsafe import AtomicMultiFileUpdate
        
        with AtomicMultiFileUpdate([PROGRESS_FILE, TUNING_STATUS_FILE], timeout=10.0) as update:
            update.write_yaml(PROGRESS_FILE, aggregated_data)
            update.write_yaml(TUNING_STATUS_FILE, updated_status)
        
        logger.info(f"Updated aggregated progress with {aggregated_data['current_trial']}/{aggregated_data['total_trials']} trials")
        return True
        
    except Exception as e:
        logger.error(f"Error updating aggregated progress: {e}", exc_info=True)
        return False

def update_trial_info_in_yaml(file_path: str, trial_info: Dict[str, Any], model_type: Optional[str] = None) -> bool:
    """
    Store information about completed trials.
    
    Args:
        file_path: Path to store trial information (or uses model-specific file if model_type provided)
        trial_info: Dictionary with trial information
        model_type: Optional model type to use model-specific file
    
    Returns:
        bool: Success status
    """
    # Use model-specific file if model_type is provided
    if model_type:
        file_path = get_model_tested_file(model_type)
        _log_progress_change(file_path, f"Updating trial info for {model_type}")
        
        # Add model type to trial info if not present
        if "model_type" not in trial_info:
            trial_info["model_type"] = model_type
    
    # Add timestamp if not present
    if "timestamp" not in trial_info:
        trial_info["timestamp"] = datetime.now().isoformat()
    
    # Add cycle information
    if "cycle" not in trial_info:
        trial_info["cycle"] = get_current_cycle()
    
    # IMPROVED: More thorough sanitization for trial_info to prevent YAML corruption
    trial_info = sanitize_yaml_data(trial_info)
    
    # Ensure params is a proper dictionary
    if "params" in trial_info and not isinstance(trial_info["params"], dict):
        logger.warning(f"Converting non-dict params to dict: {type(trial_info['params'])}")
        try:
            if hasattr(trial_info["params"], "__dict__"):
                trial_info["params"] = trial_info["params"].__dict__
            else:
                trial_info["params"] = {"value": str(trial_info["params"])}
        except Exception as e:
            logger.error(f"Could not convert params to dict: {e}")
            trial_info["params"] = {"error": str(e)}
    
    # Append to YAML list
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Use the enhanced lock for consistent operations
        with ProgressFileLock(file_path, operation="update_trial_info"):
            # Read existing data first
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f) or []
            else:
                data = []
            
            # Ensure data is a list
            if not isinstance(data, list):
                logger.warning(f"Expected list in {file_path}, got {type(data)}. Converting.")
                data = [data] if data else []
            
            # Add the new trial info
            data.append(trial_info)
            
            # Write back to file with safer serialization settings
            with open(file_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                f.flush()
                os.fsync(f.fileno())
        
        success = True
    except Exception as e:
        logger.error(f"Error appending to {file_path}: {e}")
        logger.error(traceback.format_exc())
        success = False
    
    # If successful and model_type provided, update model progress
    if success and model_type:
        # Extract progress information
        model_progress = {
            "model_type": model_type,
            "current_trial": trial_info.get("number", 0),
            "timestamp": time.time(),
            "cycle": get_current_cycle()
        }
        
        # Add metrics if available
        if "metrics" in trial_info:
            metrics = trial_info["metrics"]
            if "rmse" in metrics:
                model_progress["current_rmse"] = metrics["rmse"]
            if "mape" in metrics:
                model_progress["current_mape"] = metrics["mape"]
            if "directional_accuracy" in metrics:
                model_progress["directional_accuracy"] = metrics["directional_accuracy"]
        elif any(k in trial_info for k in ["rmse", "mape", "directional_accuracy"]):
            if "rmse" in trial_info:
                model_progress["current_rmse"] = trial_info["rmse"]
            if "mape" in trial_info:
                model_progress["current_mape"] = trial_info["mape"]
            if "directional_accuracy" in trial_info:
                model_progress["directional_accuracy"] = trial_info["directional_accuracy"]
        
        # Update ensemble weight if available
        if "weight" in trial_info:
            model_progress["weight"] = trial_info["weight"]
        elif "params" in trial_info and "ensemble_weight" in trial_info["params"]:
            model_progress["weight"] = trial_info["params"]["ensemble_weight"]
        
        # Update model progress
        _log_progress_change(get_model_progress_file(model_type), f"Updating model progress for {model_type} from trial info")
        update_model_progress(model_type, model_progress)
    
    return success

# ==========================================
# Tuning Status Management
# ==========================================

def write_tuning_status(status_dict: Dict[str, Any]) -> None:
    """
    Write tuning status to file.
    
    Args:
        status_dict: Dictionary with status information
    """
    with _lock:
        try:
            _log_progress_change(TUNING_STATUS_FILE, "Updating tuning status", status_dict)
            
            # If we're updating an existing status file, preserve values that aren't being updated
            if os.path.exists(TUNING_STATUS_FILE) and not status_dict.get('reset', False):
                current_status = {}
                try:
                    with open(TUNING_STATUS_FILE, "r") as f:
                        lines = f.readlines()
                    
                    # Parse key:value format
                    for line in lines:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            current_status[key.strip()] = value.strip()
                except Exception as e:
                    logger.warning(f"Could not read existing tuning status: {e}")
                
                # Merge dictionaries, preserving existing values not explicitly overwritten
                merged_status = current_status.copy()
                merged_status.update(status_dict)
                
                # Don't preserve error message if we're setting is_running=True
                if status_dict.get('is_running', False) and 'error' in merged_status:
                    del merged_status['error']
                
                # If we're starting a new tuning session, include ticker/timeframe
                if status_dict.get('is_running', False):
                    if 'ticker' in status_dict and 'timeframe' in status_dict:
                        # Make sure we explicitly keep the new ticker/timeframe values
                        merged_status['ticker'] = status_dict['ticker']
                        merged_status['timeframe'] = status_dict['timeframe']
                
                # Always update timestamp
                if 'timestamp' not in status_dict:
                    from datetime import datetime
                    merged_status['timestamp'] = datetime.now().isoformat()
                    
                # Remove reset flag to avoid confusion
                if 'reset' in merged_status:
                    del merged_status['reset']
                    
                # Use the merged status
                status_dict = merged_status
                
            # Remove any None values for ticker/timeframe to avoid "None" strings
            if 'ticker' in status_dict and status_dict['ticker'] is None:
                del status_dict['ticker']
            if 'timeframe' in status_dict and status_dict['timeframe'] is None:
                del status_dict['timeframe']
            
            # FIXED: Make sure trial counts are consistent with progress file
            # If we're writing a running status, include current/total trial counts from progress
            if status_dict.get('is_running', False) and ('current_trial' not in status_dict or 'total_trials' not in status_dict):
                try:
                    progress_data = read_progress_from_yaml()
                    if 'current_trial' not in status_dict and 'current_trial' in progress_data:
                        status_dict['current_trial'] = progress_data['current_trial']
                    if 'total_trials' not in status_dict and 'total_trials' in progress_data:
                        status_dict['total_trials'] = progress_data['total_trials']
                except Exception as e:
                    logger.warning(f"Could not read progress file for trial counts: {e}")
                    
            # Write the status to file with lock
            with ProgressFileLock(TUNING_STATUS_FILE, operation="write_status"):
                with open(TUNING_STATUS_FILE, "w") as f:
                    for key, value in status_dict.items():
                        f.write(f"{key}: {value}\n")
                    
            # Enhanced Debug logging
            if 'ticker' in status_dict or 'timeframe' in status_dict:
                logger.debug(f"Tuning status updated with ticker={status_dict.get('ticker', 'None')}, timeframe={status_dict.get('timeframe', 'None')}")
                
            # FIXED: After writing status, ensure progress file is consistent
            # This helps keep the two files in sync
            if status_dict.get('is_running', False) and ('ticker' in status_dict or 'timeframe' in status_dict):
                try:
                    progress_data = read_progress_from_yaml()
                    update_needed = False
                    
                    # Check if we need to update ticker/timeframe
                    if 'ticker' in status_dict and progress_data.get('ticker') != status_dict['ticker']:
                        progress_data['ticker'] = status_dict['ticker']
                        update_needed = True
                    if 'timeframe' in status_dict and progress_data.get('timeframe') != status_dict['timeframe']:
                        progress_data['timeframe'] = status_dict['timeframe']
                        update_needed = True
                        
                    # Check if we need to update trial counts
                    if 'current_trial' in status_dict and progress_data.get('current_trial') != status_dict['current_trial']:
                        progress_data['current_trial'] = status_dict['current_trial']
                        update_needed = True
                    if 'total_trials' in status_dict and progress_data.get('total_trials') != status_dict['total_trials']:
                        progress_data['total_trials'] = status_dict['total_trials']
                        update_needed = True
                        
                    # If we need to update, write to progress file
                    if update_needed:
                        _log_progress_change(PROGRESS_FILE, "Syncing progress file with tuning status")
                        update_progress_in_yaml(progress_data, aggregate_trials=False)
                except Exception as e:
                    logger.warning(f"Could not sync progress file with tuning status: {e}")
                
        except Exception as e:
            logger.error(f"Error writing tuning status: {e}")

def read_tuning_status() -> Dict[str, Any]:
    """
    Read the current tuning status from file.
    
    Returns:
        Dict with status information
    """
    status = {"is_running": False, "status": "idle"}
    
    if os.path.exists(TUNING_STATUS_FILE):
        try:
            # Check if the file is stale
            file_age = time.time() - os.path.getmtime(TUNING_STATUS_FILE)
            if file_age > 3600:  # 1 hour
                logger.warning(f"Tuning status file is stale ({file_age:.1f} seconds old)")
                return {"is_running": False, "status": "idle", "stale": True}
            
            with open(TUNING_STATUS_FILE, "r") as f:
                lines = f.readlines()
            
            # Parse key:value format
            for line in lines:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert "true"/"false" strings to boolean
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    
                    status[key] = value
                    
            # Log ticker/timeframe for debugging
            if "ticker" in status or "timeframe" in status:
                logger.debug(f"Read tuning status with ticker={status.get('ticker', 'None')}, timeframe={status.get('timeframe', 'None')}")
                
            # Throttle repeated calls with a small sleep
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error reading tuning status: {e}")
            status = {"is_running": False, "status": "idle", "error": str(e)}
    
    return status

# ==========================================
# Cycle Metrics Management
# ==========================================

def update_cycle_metrics(cycle_data: Dict[str, Any], cycle_num: Optional[int] = None) -> bool:
    """
    Update the cycle metrics in the cycle_metrics.yaml file.
    
    Args:
        cycle_data: Dictionary with cycle metrics
        cycle_num: Cycle number to use as an index
    
    Returns:
        bool: Success status
    """
    # Use global cycle if not provided
    if cycle_num is None:
        cycle_num = get_current_cycle()
    
    # Read existing cycle metrics
    existing_cycles = safe_read_yaml(CYCLE_METRICS_FILE, default=[])
    
    # If file is empty or not a list, initialize with empty list
    if not existing_cycles or not isinstance(existing_cycles, list):
        existing_cycles = []
    
    # If cycle_num is provided, try to update that specific cycle
    if cycle_num is not None:
        # Find matching cycle or append new one
        found = False
        for i, cycle in enumerate(existing_cycles):
            if cycle.get("cycle") == cycle_num:
                # Update existing cycle
                for key, value in cycle_data.items():
                    existing_cycles[i][key] = value
                found = True
                break
        
        if not found:
            # Add cycle_num to the data
            if "cycle" not in cycle_data:
                cycle_data["cycle"] = cycle_num
            existing_cycles.append(cycle_data)
        
        # Enhanced handling for model-specific metrics
        if "model_type" in cycle_data and "model_metrics" not in cycle_data:
            model_type = cycle_data.get("model_type")
            
            # Find the cycle we just updated to add model metrics
            for i, cycle in enumerate(existing_cycles):
                if cycle.get("cycle") == cycle_num:
                    # Initialize model_metrics dict if it doesn't exist
                    if "model_metrics" not in existing_cycles[i]:
                        existing_cycles[i]["model_metrics"] = {}
                    
                    # Extract metrics relevant to this model
                    metrics_to_track = ["rmse", "mape", "best_value", "completed_trials", 
                                       "directional_accuracy", "combined_score"]
                    model_metrics = {k: cycle_data.get(k) for k in metrics_to_track if k in cycle_data}
                    
                    # Add trial counts if available
                    if "total_trials" in cycle_data:
                        model_metrics["total_trials"] = cycle_data.get("total_trials")
                    if "current_trial" in cycle_data:
                        model_metrics["current_trial"] = cycle_data.get("current_trial")
                    
                    # Store the metrics under this model type
                    if model_metrics:
                        existing_cycles[i]["model_metrics"][model_type] = model_metrics
                    break
                    
        # Add weighted metrics if available
        if "weighted_metrics" in cycle_data:
            for i, cycle in enumerate(existing_cycles):
                if cycle.get("cycle") == cycle_num:
                    existing_cycles[i]["weighted_metrics"] = cycle_data["weighted_metrics"]
                    break
    else:
        # Just append the new cycle data
        existing_cycles.append(cycle_data)
    
    # Add timestamp if not present
    if "timestamp" not in cycle_data:
        cycle_data["timestamp"] = time.time()
    
    # Limit to 50 most recent cycles
    if len(existing_cycles) > 50:
        existing_cycles = existing_cycles[-50:]
    
    # Write back to file
    return safe_write_yaml(CYCLE_METRICS_FILE, existing_cycles)

def get_cycle_metrics(cycle_num: Optional[int] = None) -> Dict[str, Any]:
    """
    Get metrics for a specific cycle or the latest cycle.
    
    Args:
        cycle_num: Cycle number to get metrics for, or None for latest
    
    Returns:
        Dict with cycle metrics
    """
    # Use global cycle if not provided
    if cycle_num is None:
        cycle_num = get_current_cycle()
    
    # Read all cycle metrics
    all_cycles = safe_read_yaml(CYCLE_METRICS_FILE, default=[])
    
    if not all_cycles:
        return {}
    
    # If cycle_num specified, find that cycle
    if cycle_num is not None:
        for cycle in all_cycles:
            if cycle.get("cycle") == cycle_num:
                return cycle
        return {}  # Cycle not found
    
    # Otherwise return the latest cycle
    return all_cycles[-1]

def update_model_progress(model_type: str, progress_data: Dict[str, Any]) -> bool:
    """
    Update progress information for a specific model type.
    
    Args:
        model_type: The type of model (e.g., 'lstm', 'cnn', etc.)
        progress_data: Dictionary with progress information
        
    Returns:
        bool: Success status
    """
    # Make sure model_type is included in the data
    progress_data["model_type"] = model_type
    
    # Ensure cycle information is included in the model progress data
    progress_data["cycle"] = get_current_cycle()
    
    # Update the model-specific progress file
    result = update_progress_in_yaml(progress_data, model_type=model_type)
    
    # Store cycle history if enabled in config
    try:
        # Try to get config settings
        store_cycle_history = False
        cycle_history_path = os.path.join(MODEL_PROGRESS_DIR, f"{model_type}_history.yaml")
        
        try:
            from config.config_loader import TUNING_CONFIG
            if isinstance(TUNING_CONFIG, dict):
                store_cycle_history = TUNING_CONFIG.get("store_cycle_history", False)
                if "cycle_history_path" in TUNING_CONFIG:
                    cycle_history_path = TUNING_CONFIG["cycle_history_path"].format(model=model_type)
        except ImportError:
            # Fall back to default settings if config can't be imported
            logger.debug("Could not import TUNING_CONFIG, using default cycle history settings")
        
        if store_cycle_history:
            cycle = progress_data.get("cycle", get_current_cycle())
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(cycle_history_path), exist_ok=True)
            
            # Use our existing file lock mechanism
            with ProgressFileLock(cycle_history_path, operation=f"update_history_{model_type}", timeout=5.0):
                # Load existing history or initialize
                history = safe_read_yaml(cycle_history_path, default={})
                history.setdefault("cycles", {})
                
                # Add this cycle's data
                history["cycles"][str(cycle)] = {
                    "rmse": progress_data.get("current_rmse"),
                    "mape": progress_data.get("current_mape"),
                    "directional_accuracy": progress_data.get("directional_accuracy"),
                    "trials": progress_data.get("current_trial"),
                    "timestamp": datetime.now().isoformat(),
                    "ticker": progress_data.get("ticker"),
                    "timeframe": progress_data.get("timeframe")
                }
                
                # Write updated history
                safe_write_yaml(cycle_history_path, history)
                logger.debug(f"Updated cycle history for {model_type} at cycle {cycle}")
    except Exception as e:
        logger.warning(f"Failed to update cycle history for {model_type}: {e}")
        # Don't fail the whole operation if just the history update fails
    
    return result

def get_individual_model_progress() -> Dict[str, Dict[str, Any]]:
    """
    Get progress information for all individual model types.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model types to progress info
    """
    try:
        # Get active model types
        try:
            from config.config_loader import ACTIVE_MODEL_TYPES
            model_types = ACTIVE_MODEL_TYPES
        except ImportError:
            # Scan directory for model types
            model_types = []
            for file in os.listdir(MODEL_PROGRESS_DIR):
                if file.endswith("_progress.yaml"):
                    model_type = file.replace("_progress.yaml", "")
                    model_types.append(model_type)
        
        progress_data = {}
        
        # Check if any model progress files exist
        any_files_exist = False
        for model_type in model_types:
            progress_file = get_model_progress_file(model_type)
            if os.path.exists(progress_file):
                any_files_exist = True
                break
        
        # If no files exist, return empty dictionary
        if not any_files_exist:
            return {}
        
        # Read progress files for each model type
        for model_type in model_types:
            progress_file = get_model_progress_file(model_type)
            if os.path.exists(progress_file):
                model_progress = safe_read_yaml(progress_file, default={})
                if model_progress:
                    progress_data[model_type] = model_progress
        
        return progress_data
    except Exception as e:
        logger.error(f"Error getting individual model progress: {e}")
        return {}

# ==========================================
# Helper Functions for Optuna Integration
# ==========================================

def create_progress_callback(cycle=1, model_type=None, ticker=None, timeframe=None, range_cat=None):
    """
    Create a callback function for Optuna to update progress information.
    
    Args:
        cycle: Current cycle number
        model_type: The model type being optimized
        ticker: The ticker symbol being used
        timeframe: The timeframe being used
        range_cat: The range category being used
    
    Returns:
        callable: Callback function for Optuna
    """
    def progress_callback(study, trial):
        """Update progress information during optimization."""
        # Extract model_type from study if not provided
        nonlocal model_type
        local_model_type = model_type
        
        # Try to extract model type from trial
        if not local_model_type:
            if hasattr(trial, "params") and "model_type" in trial.params:
                local_model_type = trial.params.get("model_type")
            elif hasattr(study, "user_attrs") and "model_type" in study.user_attrs:
                local_model_type = study.user_attrs.get("model_type")
            # Try from study name
            elif study.study_name:
                try:
                    # Properly import ACTIVE_MODEL_TYPES from config
                    from config.config_loader import ACTIVE_MODEL_TYPES
                    for mt in ACTIVE_MODEL_TYPES:
                        if mt in study.study_name:
                            local_model_type = mt
                            break
                except (ImportError, Exception):
                    # Fallback if import fails
                    for mt in ["lstm", "rnn", "cnn", "xgboost", "random_forest", "ltc", "nbeats", "tabnet", "tft"]:
                        if mt in study.study_name:
                            local_model_type = mt
                            break
        
        # Get current trial info
        current_trial = len(study.trials)
        total_trials = study.user_attrs.get("n_trials", 10000)
        
        # Create progress data
        progress_data = {
            "current_trial": current_trial,
            "total_trials": total_trials,
            "cycle": cycle,
            "timestamp": time.time(),
            "is_running": True,
        }
        
        # Add model type if available
        if local_model_type:
            progress_data["model_type"] = local_model_type
        
        # Add ticker, timeframe and range_cat if provided
        if ticker:
            progress_data["ticker"] = ticker
        if timeframe:
            progress_data["timeframe"] = timeframe
        if range_cat:
            progress_data["range_cat"] = range_cat
        
        # Add best metrics if available
        try:
            if study.trials:
                best_trial = study.best_trial
                progress_data["best_rmse"] = best_trial.user_attrs.get("rmse")
                progress_data["best_mape"] = best_trial.user_attrs.get("mape")
                progress_data["best_model"] = best_trial.params.get("model_type", local_model_type)
        except (ValueError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get best trial: {e}")
        
        # Add current trial info
        if hasattr(trial, "user_attrs") and trial.user_attrs:
            progress_data["current_rmse"] = trial.user_attrs.get("rmse")
            progress_data["current_mape"] = trial.user_attrs.get("mape")
            progress_data["directional_accuracy"] = trial.user_attrs.get("directional_accuracy", 0.0)
        
        if hasattr(trial, "value"):
            progress_data["current_trial_value"] = trial.value
        
        # Update progress
        if local_model_type:
            update_model_progress(local_model_type, progress_data)
        else:
            update_progress_in_yaml(progress_data)
        
        # Update tuning status
        write_tuning_status({
            "is_running": True,
            "status": "running",
            "current_trial": current_trial,
            "total_trials": total_trials,
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "timeframe": timeframe
        })
    
    return progress_callback

# ==========================================
# Utility Functions
# ==========================================

def convert_to_native_types(data):
    """Convert numpy values to native Python types for serialization."""
    # Reuse the implementation from threadsafe module
    from src.utils.threadsafe import convert_to_native_types as threadsafe_convert
    return threadsafe_convert(data)

def prune_old_cycles(max_cycles: int = 50) -> bool:
    """
    Prune old cycle metrics, keeping only the most recent.
    
    Args:
        max_cycles: Maximum number of cycle metrics to keep
    
    Returns:
        bool: Success status
    """
    # Read existing cycle metrics
    existing_cycles = safe_read_yaml(CYCLE_METRICS_FILE, default=[])
    
    # If file is empty or not a list, nothing to do
    if not existing_cycles or not isinstance(existing_cycles, list):
        return True
    
    # Keep only the most recent cycles
    if len(existing_cycles) > max_cycles:
        pruned_cycles = existing_cycles[-max_cycles:]
        return safe_write_yaml(CYCLE_METRICS_FILE, pruned_cycles)
    
    return True

def cleanup_stale_files():
    """
    Cleanup stale lock files and temporary files.
    """
    # Use the implementation from threadsafe module
    from src.utils.threadsafe import cleanup_stale_locks, cleanup_stale_temp_files
    
    # Clean lock files with shorter timeout for critical files
    cleaned_locks = cleanup_stale_locks(max_age=300)
    
    # Clean temporary files
    cleaned_temps = cleanup_stale_temp_files(max_age=3600)
    
    if cleaned_locks or cleaned_temps:
        logger.info(f"Cleaned up {cleaned_locks} stale lock files and {cleaned_temps} temporary files")

def synchronize_paths_with_config():
    """
    Ensures all module paths match those in config_loader by re-importing.
    
    This can be called to refresh paths if they've been updated.
    """
    try:
        from importlib import reload
        import config.config_loader
        reload(config.config_loader)
        
        # Update global variables with new paths
        global DATA_DIR, PROGRESS_FILE, TUNING_STATUS_FILE, CYCLE_METRICS_FILE
        global BEST_PARAMS_FILE, MODEL_PROGRESS_DIR, TESTED_MODELS_DIR
        
        from config.config_loader import (
            DATA_DIR, PROGRESS_FILE, TUNING_STATUS_FILE, 
            CYCLE_METRICS_FILE, BEST_PARAMS_FILE, MODEL_PROGRESS_DIR
        )
        
        # Keep TESTED_MODELS_DIR definition consistent
        TESTED_MODELS_DIR = os.path.join(DATA_DIR, "tested_models")
        
        # Ensure directories exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_PROGRESS_DIR, exist_ok=True)
        os.makedirs(TESTED_MODELS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)
        
        logger.info("Successfully synchronized paths with config_loader")
        return True
    except Exception as e:
        logger.error(f"Error synchronizing paths: {e}")
        return False

def verify_progress_consistency():
    """
    Verify consistency between progress files and fix if needed.
    Returns True if files were consistent or fixed, False if unrecoverable errors.
    """
    inconsistencies = 0
    fixes = 0
    
    try:
        # Read main progress file
        progress_data = safe_read_yaml(PROGRESS_FILE, default={})
        
        # Read tuning status
        status_data = read_tuning_status()
        
        # Check ticker/timeframe consistency
        if "ticker" in progress_data and "ticker" in status_data:
            if progress_data["ticker"] != status_data["ticker"]:
                logger.warning(f"Ticker mismatch: {progress_data['ticker']} vs {status_data['ticker']}")
                inconsistencies += 1
                
                # Trust the status file for ticker/timeframe
                progress_data["ticker"] = status_data["ticker"]
                fixes += 1
                
        if "timeframe" in progress_data and "timeframe" in status_data:
            if progress_data["timeframe"] != status_data["timeframe"]:
                logger.warning(f"Timeframe mismatch: {progress_data['timeframe']} vs {status_data['timeframe']}")
                inconsistencies += 1
                
                # Trust the status file for ticker/timeframe
                progress_data["timeframe"] = status_data["timeframe"]
                fixes += 1
        
        # Check trial count consistency
        if "current_trial" in progress_data and "current_trial" in status_data:
            try:
                progress_trial = int(progress_data["current_trial"])
                status_trial = int(status_data["current_trial"])
                
                if abs(progress_trial - status_trial) > 10:
                    logger.warning(f"Trial count mismatch: {progress_trial} vs {status_trial}")
                    inconsistencies += 1
                    
                    # Use the maximum value
                    max_trial = max(progress_trial, status_trial)
                    progress_data["current_trial"] = max_trial
                    status_data["current_trial"] = str(max_trial)
                    fixes += 1
            except (ValueError, TypeError):
                pass
                
        # Check model progress files for consistency with main progress
        try:
            # Get model types from config or from existing directories
            try:
                from config.config_loader import ACTIVE_MODEL_TYPES
                model_types = ACTIVE_MODEL_TYPES
            except ImportError:
                # Fallback to scanning directory
                model_types = []
                for file in os.listdir(MODEL_PROGRESS_DIR):
                    if file.endswith("_progress.yaml"):
                        model_type = file.replace("_progress.yaml", "")
                        model_types.append(model_type)
            
            # Check each model's progress file
            for model_type in model_types:
                model_file = get_model_progress_file(model_type)
                if os.path.exists(model_file):
                    model_data = safe_read_yaml(model_file, default={})
                    
                    # Check ticker/timeframe consistency
                    if "ticker" in model_data and "ticker" in progress_data:
                        if model_data["ticker"] != progress_data["ticker"]:
                            logger.warning(f"Model {model_type} ticker mismatch: {model_data['ticker']} vs {progress_data['ticker']}")
                            inconsistencies += 1
                            
                            # Update model file with main progress ticker
                            model_data["ticker"] = progress_data["ticker"]
                            safe_write_yaml(model_file, model_data)
                            fixes += 1
                    
                    # Check timeframe consistency
                    if "timeframe" in model_data and "timeframe" in progress_data:
                        if model_data["timeframe"] != progress_data["timeframe"]:
                            logger.warning(f"Model {model_type} timeframe mismatch: {model_data['timeframe']} vs {progress_data['timeframe']}")
                            inconsistencies += 1
                            
                            # Update model file with main progress timeframe
                            model_data["timeframe"] = progress_data["timeframe"]
                            safe_write_yaml(model_file, model_data)
                            fixes += 1
        except Exception as model_err:
            logger.error(f"Error checking model progress files: {model_err}")
        
        # If we fixed anything, write back the files
        if fixes > 0:
            logger.info(f"Fixed {fixes} consistency issues")
            
            # Write progress data
            safe_write_yaml(PROGRESS_FILE, progress_data)
            
            # Update status data with new values
            if "current_trial" in status_data and isinstance(status_data["current_trial"], int):
                status_update = {
                    "current_trial": status_data["current_trial"],
                    "timestamp": datetime.now().isoformat()
                }
                write_tuning_status(status_update)
        
        return True
    except Exception as e:
        logger.error(f"Error checking consistency: {e}")
        return False

# Update the existing check_file_consistency function to use our new method
def check_file_consistency():
    """Check consistency between progress and status files at module load time."""
    try:
        status_file_exists = os.path.exists(TUNING_STATUS_FILE)
        progress_file_exists = os.path.exists(PROGRESS_FILE)
        
        # If either file is missing, don't need to check consistency
        if not status_file_exists or not progress_file_exists:
            return
        
        # Use our new comprehensive verification function
        verify_progress_consistency()
            
    except Exception as e:
        logger.error(f"Error checking file consistency: {e}")

# Add initialization code at module level
try:
    # Initialize directories
    for directory in [MODEL_PROGRESS_DIR, TESTED_MODELS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Make sure BEST_PARAMS_FILE parent directory exists
    os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)
    
    # Clean up stale files on module import
    cleanup_stale_files()
    
    # Now check file consistency 
    check_file_consistency()
    
    # Import config properly
    try:
        from config.config_loader import ACTIVE_MODEL_TYPES
        logger.info(f"Successfully imported config with {len(ACTIVE_MODEL_TYPES)} active model types")
    except ImportError:
        logger.warning("Could not import configuration, using defaults")
except Exception as e:
    logger.error(f"Error during module initialization: {e}")