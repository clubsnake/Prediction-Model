"""
Helper functions for tracking and managing tuning progress.
"""
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

# Import file paths from config
from config.config_loader import (
    CYCLE_METRICS_FILE,
    PROGRESS_FILE,
    TESTED_MODELS_FILE,
    TUNING_STATUS_FILE,
)

# Import proper thread-safe utilities
from src.utils.threadsafe import (
    FileLock,
    append_to_yaml_list,
    safe_read_yaml,
    safe_write_yaml,
    emergency_cleanup_for_file,
    cleanup_stale_locks,
)

# Define individual model progress file paths
MODEL_PROGRESS_DIR = os.path.dirname(os.path.abspath(PROGRESS_FILE))
MODEL_PROGRESS_FILES = {
    model_type: os.path.join(MODEL_PROGRESS_DIR, f"{model_type}_progress.yaml")
    for model_type in [
        "lstm", "rnn", "cnn", "xgboost", "random_forest", 
        "ltc", "nbeats", "tabnet", "tft"
    ]
}

# Make sure directories exist
os.makedirs(os.path.dirname(os.path.abspath(PROGRESS_FILE)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(TESTED_MODELS_FILE)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(CYCLE_METRICS_FILE)), exist_ok=True)

# Add thread-safe locks for global variables
_lock = threading.RLock()

# Global stop control via threading.Event
stop_event = threading.Event()

# Global cycle tracking
current_cycle = 1


def set_stop_requested(val: bool):
    """Set or clear the stop request flag in a thread-safe manner"""
    with _lock:
        if val:
            print("Stop requested via progress_helper")
            stop_event.set()
        else:
            stop_event.clear()


def is_stop_requested():
    """Check if stop has been requested in a thread-safe manner"""
    with _lock:
        return stop_event.is_set()


def is_progress_file_writable() -> bool:
    """Check if progress.yaml or its directory is writable."""
    if os.path.exists(PROGRESS_FILE):
        return os.access(PROGRESS_FILE, os.W_OK)
    else:
        directory = os.path.dirname(PROGRESS_FILE)
        return os.access(directory, os.W_OK)


def force_clear_lock_files():
    """
    Forcibly clear lock files for critical files like progress.yaml.
    Call this before critical operations that must succeed.
    """
    critical_files = [PROGRESS_FILE, TESTED_MODELS_FILE, TUNING_STATUS_FILE]
    
    for file_path in critical_files:
        lock_file = f"{file_path}.lock"
        if os.path.exists(lock_file):
            try:
                # For extreme cases, try to force unlock using direct file removal
                os.remove(lock_file)
                print(f"Forcibly removed lock file before critical operation: {lock_file}")
            except PermissionError:
                print(f"Warning: Cannot remove lock file {lock_file} - in use by another process")
            except Exception as e:
                print(f"Error removing lock file {lock_file}: {e}")


def update_progress_in_yaml(progress_data: Dict[str, Any], aggregate_trials: bool = True) -> bool:
    """
    Update the progress.yaml file with the current progress.
    Improved to properly aggregate trial counts across models.

    Args:
        progress_data: The progress data to update
        aggregate_trials: Whether to aggregate trials from all models

    Returns:
        bool: True if successful, False otherwise
    """
    global current_cycle
    max_attempts = 3
    attempt = 0
    
    # First check if the file is writable
    if not is_progress_file_writable():
        print(f"Progress file {PROGRESS_FILE} is not writable. Check permissions.")
        return False

    # Ensure a timestamp is present
    if "timestamp" not in progress_data:
        progress_data["timestamp"] = time.time()
    
    # Update global cycle if provided
    if "cycle" in progress_data:
        current_cycle = progress_data["cycle"]
    else:
        # Use the global cycle
        progress_data["cycle"] = current_cycle
    
    # If we're aggregating trials, read the existing data first
    if aggregate_trials and os.path.exists(PROGRESS_FILE):
        try:
            existing_data = safe_read_yaml(PROGRESS_FILE, default={})
            
            # If we have existing data, aggregate trial counts
            if existing_data:
                # Get the current model type being updated
                current_model = progress_data.get("model_type")
                
                # Create or update the model_trials dictionary
                if "model_trials" not in existing_data:
                    existing_data["model_trials"] = {}
                
                # Update the trials for the current model
                if current_model:
                    existing_data["model_trials"][current_model] = {
                        "current_trial": progress_data.get("current_trial", 0),
                        "total_trials": progress_data.get("total_trials", 1),
                        "cycle": progress_data.get("cycle", current_cycle)
                    }
                
                # Calculate aggregated trial counts across all models
                # Only count models in the current cycle
                current_trial_sum = 0
                total_trials_sum = 0
                
                for model, trials in existing_data.get("model_trials", {}).items():
                    if trials.get("cycle", 1) == current_cycle:
                        current_trial_sum += trials.get("current_trial", 0)
                        total_trials_sum += trials.get("total_trials", 1)
                
                # Update the aggregated counts in the progress data
                progress_data["aggregated_current_trial"] = current_trial_sum
                progress_data["aggregated_total_trials"] = total_trials_sum
                
                # For UI display, use the aggregated trials
                # Only override if not explicitly provided
                if "current_trial" not in progress_data or "model_type" in progress_data:
                    progress_data["current_trial"] = current_trial_sum
                if "total_trials" not in progress_data or "model_type" in progress_data:
                    progress_data["total_trials"] = total_trials_sum

                # Add weighted metrics if available in existing data
                if "weighted_metrics" in existing_data and "weighted_metrics" not in progress_data:
                    progress_data["weighted_metrics"] = existing_data["weighted_metrics"]
                
                # Make sure we keep the existing cycle value if not provided
                if "cycle" in existing_data and "cycle" not in progress_data:
                    progress_data["cycle"] = existing_data["cycle"]
                
                # Keep the best metrics if we don't have new ones
                for key in ["best_rmse", "best_mape", "best_model"]:
                    if key not in progress_data and key in existing_data:
                        progress_data[key] = existing_data[key]
        except Exception as e:
            print(f"Error aggregating trial data: {e}")
            # Continue with the update without aggregation

    # Compute remaining_trials from global totals
    if "total_trials" in progress_data and "current_trial" in progress_data:
        progress_data["remaining_trials"] = progress_data["total_trials"] - progress_data["current_trial"]

    while attempt < max_attempts:
        try:
            # Try using safe_write_yaml first
            result = safe_write_yaml(PROGRESS_FILE, progress_data)
            if result:
                print(f"Successfully updated progress.yaml (attempt {attempt+1})")
                return True
                
            # If that fails, try direct write as fallback
            if attempt == max_attempts - 1:
                # On last attempt, try to force clear any locks first
                force_clear_lock_files()
                
                # Now try emergency direct write
                print("Attempting emergency direct write to progress.yaml")
                temp_file = f"{PROGRESS_FILE}.tmp.{int(time.time())}"
                with open(temp_file, "w") as f:
                    yaml.safe_dump(progress_data, f, default_flow_style=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                    
                # Use os.replace for atomic operation
                os.replace(temp_file, PROGRESS_FILE)
                print("Emergency direct write to progress.yaml succeeded")
                return True
        except FileLock as e:
            attempt += 1
            print(f"Lock error writing YAML (attempt {attempt}/{max_attempts}): {e}")
            
            if attempt >= max_attempts - 1:
                # On the last retry, attempt emergency cleanup first
                print(f"Attempting emergency cleanup for {PROGRESS_FILE} before final attempt")
                emergency_cleanup_for_file(PROGRESS_FILE)
                
        except Exception as e:
            attempt += 1
            print(f"Error updating progress.yaml (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(0.5)  # Short delay before retry
            
    print(f"Failed to update progress.yaml after {max_attempts} attempts")
    return False

def update_model_progress(model_type: str, model_data: Dict[str, Any]) -> bool:
    """
    Update the progress file for a specific model type.
    
    Args:
        model_type: The type of model (e.g., 'lstm', 'xgboost')
        model_data: The progress data to update for this model
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Get the model-specific progress file path
    if model_type not in MODEL_PROGRESS_FILES:
        print(f"Unknown model type: {model_type}")
        return False
        
    file_path = MODEL_PROGRESS_FILES[model_type]
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Ensure timestamp is present
    if "timestamp" not in model_data:
        model_data["timestamp"] = time.time()
        
    # Add model_type to the data
    model_data["model_type"] = model_type
    
    # Add cycle if not present
    if "cycle" not in model_data:
        model_data["cycle"] = current_cycle
        
    try:
        # Use safe_write_yaml to update the model-specific file
        result = safe_write_yaml(file_path, model_data)
        if result:
            print(f"Successfully updated {model_type} progress file")
            
            # Also update the main progress file with this model's data
            # First read current progress
            main_progress = read_progress_from_yaml()
            
            # Update model-specific entry in model_trials
            if "model_trials" not in main_progress:
                main_progress["model_trials"] = {}
                
            main_progress["model_trials"][model_type] = {
                "current_trial": model_data.get("current_trial", 0),
                "total_trials": model_data.get("total_trials", 1),
                "cycle": model_data.get("cycle", current_cycle),
                "rmse": model_data.get("rmse"),
                "mape": model_data.get("mape"),
                "directional_accuracy": model_data.get("directional_accuracy")
            }
            
            # Update main progress file (but don't aggregate again as we just did it)
            update_progress_in_yaml(main_progress, aggregate_trials=False)
            return True
        else:
            print(f"Failed to update {model_type} progress file")
            return False
    except Exception as e:
        print(f"Error updating {model_type} progress file: {e}")
        return False

def read_model_progress(model_type: str) -> Dict[str, Any]:
    """
    Read the progress file for a specific model type.
    
    Args:
        model_type: The type of model (e.g., 'lstm', 'xgboost')
    
    Returns:
        Dict with model progress information
    """
    # Get the model-specific progress file path
    if model_type not in MODEL_PROGRESS_FILES:
        print(f"Unknown model type: {model_type}")
        return {}
        
    file_path = MODEL_PROGRESS_FILES[model_type]
    
    # If file doesn't exist, return empty dict
    if not os.path.exists(file_path):
        return {"model_type": model_type, "cycle": current_cycle}
        
    try:
        # Use safe_read_yaml to read the model-specific file
        model_data = safe_read_yaml(file_path, default={})
        
        # Add model_type if not present
        if "model_type" not in model_data:
            model_data["model_type"] = model_type
            
        return model_data
    except Exception as e:
        print(f"Error reading {model_type} progress file: {e}")
        return {"model_type": model_type, "error": str(e)}

def update_weighted_metrics(weighted_metrics: Dict[str, Any]) -> bool:
    """
    Update the weighted metrics in the progress.yaml file.
    
    Args:
        weighted_metrics: Dictionary with weighted metrics
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Read current progress
    progress_data = read_progress_from_yaml()
    
    # Add weighted metrics
    progress_data["weighted_metrics"] = weighted_metrics
    
    # Add timestamp
    progress_data["weighted_metrics"]["timestamp"] = time.time()
    
    # Update progress file
    return update_progress_in_yaml(progress_data)

def read_weighted_metrics() -> Dict[str, Any]:
    """
    Read the weighted metrics from the progress.yaml file.
    
    Returns:
        Dict with weighted metrics
    """
    progress_data = read_progress_from_yaml()
    return progress_data.get("weighted_metrics", {})

def get_all_model_progress() -> Dict[str, Dict[str, Any]]:
    """
    Get progress information for all models.
    
    Returns:
        Dictionary mapping model types to their progress information
    """
    all_model_progress = {}
    
    for model_type in MODEL_PROGRESS_FILES.keys():
        model_progress = read_model_progress(model_type)
        all_model_progress[model_type] = model_progress
        
    return all_model_progress

def read_progress_from_yaml() -> Dict[str, Any]:
    """
    Read the current progress from the progress.yaml file.
    Enhanced with better error handling and fallbacks.

    Returns:
        Dict with progress information
    """
    global current_cycle
    
    # Try to clean up any stale locks first
    cleanup_stale_locks(max_age=60, force=False)  # Less aggressive initial cleanup
    
    try:
        progress_data = safe_read_yaml(PROGRESS_FILE, default={})
        
        # Add default values for missing fields
        if not progress_data:
            progress_data = {
                "current_trial": 0,
                "total_trials": 1,
                "current_rmse": None,
                "current_mape": None,
                "best_rmse": None,
                "best_mape": None,
                "cycle": current_cycle,
                "timestamp": time.time()
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
            
        # Update global cycle
        current_cycle = progress_data.get("cycle", 1)
        
        return progress_data
    except Exception as e:
        print(f"Error reading progress.yaml, returning defaults: {e}")
        return {
            "current_trial": 0,
            "total_trials": 1,
            "current_rmse": None,
            "current_mape": None,
            "best_rmse": None,
            "best_mape": None,
            "cycle": current_cycle,
            "timestamp": time.time(),
            "error": str(e),
            "model_trials": {}
        }


def write_tuning_status(status_info: Dict[str, Any]) -> bool:
    """
    Write the current tuning status to a file for coordination between processes.
    Enhanced with better error handling and emergency fallbacks.

    Args:
        status_info: Dictionary with status information
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Add timestamp if not present
        if "timestamp" not in status_info:
            status_info["timestamp"] = datetime.now().isoformat()
            
        print(f"Writing tuning status to {TUNING_STATUS_FILE}: {status_info}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(TUNING_STATUS_FILE)), exist_ok=True)
        
        # Use FileLock for thread-safety when writing the status file
        try:
            with FileLock(TUNING_STATUS_FILE, timeout=5.0):  # Shorter timeout
                # Write in simple text format for better compatibility
                with open(TUNING_STATUS_FILE, "w") as f:
                    for key, value in status_info.items():
                        f.write(f"{key}: {value}\n")
                print(f"Successfully wrote tuning status to text file: {TUNING_STATUS_FILE}")
                return True
        except Exception as lock_error:
            print(f"Error acquiring lock for tuning status: {lock_error}")
            
            # Emergency cleanup and retry
            emergency_cleanup_for_file(TUNING_STATUS_FILE)
            
            # Direct write attempt
            print("Attempting emergency direct write to tuning status file")
            temp_file = f"{TUNING_STATUS_FILE}.tmp.{int(time.time())}"
            with open(temp_file, "w") as f:
                for key, value in status_info.items():
                    f.write(f"{key}: {value}\n")
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
                
            # Use os.replace for atomic operation
            os.replace(temp_file, TUNING_STATUS_FILE)
            print("Emergency direct write to tuning status file succeeded")
            return True
    except Exception as e:
        print(f"Error writing tuning status: {e}")
        traceback.print_exc()
        return False


def read_tuning_status() -> Dict[str, Any]:
    """
    Read the current tuning status from file.
    Enhanced with better error detection and handling of stale status.

    Returns:
        Dict with status information
    """
    status = {"is_running": False, "status": "idle"}
    
    if os.path.exists(TUNING_STATUS_FILE):
        try:
            # Check if the file is stale
            file_age = time.time() - os.path.getmtime(TUNING_STATUS_FILE)
            if file_age > 3600:  # 1 hour
                print(f"Warning: Tuning status file is stale ({file_age:.1f} seconds old)")
                # Don't return stale status that says tuning is running
                return {"is_running": False, "status": "idle", "stale": True}
                
            with open(TUNING_STATUS_FILE, "r") as f:
                lines = f.readlines()
                
            # Parse simple key:value format
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
                    
            print(f"Read tuning status: {status}")
        except Exception as e:
            print(f"Error reading tuning status: {e}")
            # Reset to not running if there was an error
            status = {"is_running": False, "status": "idle", "error": str(e)}
            
    return status


def update_trial_info_in_yaml(file_path: str, trial_info: Dict[str, Any], model_type: str = None) -> bool:
    """
    Append trial information to a YAML file with improved error handling.
    If model_type is provided, the trial is logged only in the model-specific file.
    Otherwise, it is written to the global trials file.

    Args:
        file_path: Path to the YAML file
        trial_info: Trial information to append
        model_type: Optional model type identifier

    Returns:
        bool: True if successful, False otherwise
    """
    global current_cycle
    
    try:
        # Add model type to trial info if provided
        if model_type and "model_type" not in trial_info:
            trial_info["model_type"] = model_type
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Enhanced lock file handling
        lock_file = f"{file_path}.lock"
        if os.path.exists(lock_file):
            lock_age = time.time() - os.path.getmtime(lock_file)
            # More aggressive lock clearing - reduce from 60 to 30 seconds
            if lock_age > 30:
                try:
                    os.remove(lock_file)
                    print(f"Removed stale lock file before updating trial: {lock_file}")
                except Exception as e:
                    print(f"Could not remove stale lock file: {e}")
                    traceback.print_exc()

        # Log attempt to write trial info
        print(f"Writing trial info to {file_path} - Trial {trial_info.get('number', 'unknown')}")

        # Add cycle information to the trial info
        if "cycle" not in trial_info:
            trial_info["cycle"] = current_cycle
            
        # If model_type is provided, update model-specific file only
        if model_type:
            result = append_to_yaml_list(file_path, trial_info)
            print(f"Model-specific trial update result: {result}")
            
            # Also update progress.yaml with the latest trial info
            current_progress = read_progress_from_yaml()
            
            # Create model-specific progress update
            model_progress = {
                "model_type": model_type,
                "current_trial": trial_info.get("number", 0),
                "timestamp": time.time(),
                "cycle": current_cycle  # Use the global cycle number
            }
            
            # Add trial count and cycle to the model_trials dictionary
            if "model_trials" not in current_progress:
                current_progress["model_trials"] = {}
                
            # Preserve total trials if available, otherwise use a default
            total_trials = 0
            if model_type in current_progress.get("model_trials", {}):
                total_trials = current_progress["model_trials"][model_type].get("total_trials", 0)
                
            current_progress["model_trials"][model_type] = {
                "current_trial": trial_info.get("number", 0),
                "total_trials": trial_info.get("total_trials", total_trials),
                "cycle": current_cycle
            }
            
            # Update metrics if available
            if "rmse" in trial_info or "metrics" in trial_info:
                # Get metrics from the metrics dictionary if available
                metrics = trial_info.get("metrics", {})
                rmse = metrics.get("rmse", trial_info.get("rmse"))
                mape = metrics.get("mape", trial_info.get("mape"))
                
                if rmse is not None:
                    current_progress["current_rmse"] = rmse
                    # Update best RMSE if this is better
                    if (current_progress.get("best_rmse") is None or 
                        rmse < current_progress["best_rmse"]):
                        current_progress["best_rmse"] = rmse
                        current_progress["best_model"] = model_type
                
                if mape is not None:
                    current_progress["current_mape"] = mape
                    # Update best MAPE if this is better
                    if (current_progress.get("best_mape") is None or 
                        mape < current_progress["best_mape"]):
                        current_progress["best_mape"] = mape
            
            # Write the updated progress
            update_progress_in_yaml(current_progress)
            return result

        # If no model_type is provided, proceed with global trial logging
        result = append_to_yaml_list(file_path, trial_info)

        if not result:
            try:
                # Read existing data first
                existing_data = []
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            existing_data = yaml.safe_load(f) or []
                    except Exception as read_err:
                        print(f"Error reading existing data, starting fresh: {read_err}")
                        existing_data = []

                # Ensure existing_data is a list
                if not isinstance(existing_data, list):
                    print(f"Warning: {file_path} contained non-list data, resetting to empty list")
                    existing_data = []

                if "timestamp" not in trial_info:
                    trial_info["timestamp"] = datetime.now().isoformat()

                existing_data.append(trial_info)

                with open(file_path, "w") as f:
                    yaml.safe_dump(existing_data, f)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

                print(f"Successfully wrote trial {trial_info.get('number', 'unknown')} to {file_path} (direct method)")
                result = True
            except Exception as e:
                print(f"Error directly appending to {file_path}: {e}")
                traceback.print_exc()

                try:
                    backup_path = f"{file_path}.{int(time.time())}.bak"
                    print(f"Attempting to write to alternate file: {backup_path}")
                    with open(backup_path, "w") as f:
                        yaml.safe_dump([trial_info], f)
                    result = False
                except Exception as e2:
                    print(f"Failed even backup write attempt: {e2}")
                    result = False

        print(f"Successfully wrote trial {trial_info.get('number', 'unknown')} to {file_path}")
        return result
    except Exception as e:
        print(f"Error updating trial info: {e}")
        traceback.print_exc()
        return False


def update_cycle_metrics(cycle_data: Dict[str, Any], cycle_num: int = None) -> bool:
    """
    Update the cycle metrics in the cycle_metrics.yaml file.
    Enhanced with better handling of model-specific cycle metrics.

    Args:
        cycle_data: Dictionary with cycle metrics
        cycle_num: Cycle number to use as an index

    Returns:
        bool: True if successful, False otherwise
    """
    global current_cycle
    
    try:
        # Use global cycle if not provided
        if cycle_num is None:
            cycle_num = current_cycle
            
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
            # First check if this update contains model_type
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

    except Exception as e:
        print(f"Error updating cycle metrics: {e}")
        return False


def get_cycle_metrics(cycle_num: Optional[int] = None) -> Dict[str, Any]:
    """
    Get metrics for a specific cycle or the latest cycle.
    
    Args:
        cycle_num: Cycle number to get metrics for, or None for latest
        
    Returns:
        Dict with cycle metrics
    """
    try:
        # Use global cycle if not provided
        if cycle_num is None:
            cycle_num = current_cycle
            
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
    except Exception as e:
        print(f"Error getting cycle metrics: {e}")
        return {}


def prune_old_cycles(max_cycles: int = 50) -> bool:
    """
    Prune old cycle metrics from the YAML file, keeping only the most recent.

    Args:
        max_cycles: Maximum number of cycle metrics to keep

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read existing cycle metrics
        existing_cycles = safe_read_yaml(CYCLE_METRICS_FILE)

        # If file is empty or not a list, nothing to do
        if not existing_cycles or not isinstance(existing_cycles, list):
            return True

        # Keep only the most recent cycles
        if len(existing_cycles) > max_cycles:
            pruned_cycles = existing_cycles[-max_cycles:]
            return safe_write_yaml(CYCLE_METRICS_FILE, pruned_cycles)

        return True

    except Exception as e:
        print(f"Error pruning old cycles: {e}")
        return False


def get_individual_model_progress() -> Dict[str, Dict[str, Any]]:
    """
    Get progress information for individual models from progress.yaml.
    
    Returns:
        Dictionary mapping model types to their progress information
    """
    progress_data = read_progress_from_yaml()
    model_progress = {}
    
    # Extract model-specific trial information
    for model_type, trials in progress_data.get("model_trials", {}).items():
        # Only include models in the current cycle
        if trials.get("cycle", 1) == current_cycle:
            model_progress[model_type] = {
                "current_trial": trials.get("current_trial", 0),
                "total_trials": trials.get("total_trials", 1),
                "completion_percentage": (trials.get("current_trial", 0) / max(1, trials.get("total_trials", 1))) * 100
            }
    
    return model_progress


def set_current_cycle(cycle_num: int) -> None:
    """
    Set the current cycle number globally.
    
    Args:
        cycle_num: New cycle number
    """
    global current_cycle
    current_cycle = cycle_num
    
    # Update the progress file with the new cycle
    progress_data = read_progress_from_yaml()
    progress_data["cycle"] = cycle_num
    update_progress_in_yaml(progress_data)
    
    print(f"Set current cycle to {cycle_num}")


def get_current_cycle() -> int:
    """
    Get the current tuning cycle number.
    
    Returns:
        Current cycle number
    """
    return current_cycle