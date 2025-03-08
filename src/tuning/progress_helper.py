# progress_helper.py
"""
Helper functions to update and save progress metrics to a YAML file.
Used by meta_tuning.py to log current tuning progress.
"""

import os
import threading
from datetime import datetime

import yaml

# Add thread-safe locks for global variables
_lock = threading.RLock()


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


# Data directory structure and file paths
DATA_DIR = "Data"
DB_DIR = os.path.join(DATA_DIR, "DB")
LOGS_DIR = os.path.join(DATA_DIR, "Logs")
MODELS_DIR = os.path.join(DATA_DIR, "Models")

# Create all directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Define all file paths
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")
CYCLE_METRICS_FILE = os.path.join(DATA_DIR, "cycle_metrics.yaml")
BEST_PARAMS_FILE = os.path.join(DATA_DIR, "best_params.yaml")

# Global stop control via threading.Event
stop_event = threading.Event()


def set_stop_requested(val: bool):
    """Set or clear the stop request flag"""
    if val:
        print("Stop requested via progress_helper")
        stop_event.set()
    else:
        stop_event.clear()


def is_stop_requested():
    """Check if stop has been requested"""
    return stop_event.is_set()


def update_trial_info_in_yaml(progress_file, trial_info):
    """
    Update the progress tracking YAML file with information from a trial.

    Args:
        progress_file: Path to the progress YAML file
        trial_info: Dictionary containing trial information to log
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    # Read existing data
    try:
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
    except Exception as e:
        print(f"Error reading progress file: {e}")
        data = {}

    # Add timestamp if not provided
    if "timestamp" not in trial_info:
        trial_info["timestamp"] = datetime.now().isoformat()

    # Add/update trial info
    trial_number = trial_info.get("number", len(data) + 1)
    data[f"trial_{trial_number}"] = trial_info

    # Write updated data
    try:
        with open(progress_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
            return True
    except Exception as e:
        print(f"Error writing progress file: {e}")
        return False


def update_progress_in_yaml(progress_data, filepath=None):
    """
    Update progress in a YAML file.
    
    Args:
        progress_data: Dictionary with progress information
        filepath: Path to YAML file (optional)
        
    Returns:
        Boolean indicating success
    """
    try:
        # Get project root directory to resolve relative paths
        if filepath is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tuning_dir = os.path.dirname(script_dir)
            src_dir = os.path.dirname(tuning_dir)
            project_root = os.path.dirname(src_dir)
            filepath = os.path.join(project_root, "Data", "progress.yaml")
        
        # Make sure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Add timestamp if not present
        if 'timestamp' not in progress_data:
            progress_data['timestamp'] = datetime.now().timestamp()
        
        # Write to file
        with open(filepath, 'w') as f:
            yaml.dump(progress_data, f, default_flow_style=False)
        
        return True
    except Exception as e:
        print(f"Error updating progress file: {e}")
        return False


def read_tuning_status():
    """Read tuning status from file"""
    try:
        if os.path.exists(TUNING_STATUS_FILE):
            with open(TUNING_STATUS_FILE, "r") as f:
                status = f.read().strip()
                return status.lower() == "true"
        return False
    except Exception as e:
        print(f"Error reading tuning status: {e}")
        return False


def write_tuning_status(status: bool):
    """Write tuning status to file"""
    try:
        with open(TUNING_STATUS_FILE, "w") as f:
            f.write(str(status))
    except Exception as e:
        print(f"Error writing tuning status: {e}")
