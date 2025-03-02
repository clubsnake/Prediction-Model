# progress_helper.py
"""
Helper functions to update and save progress metrics to a YAML file.
Used by meta_tuning.py to log current tuning progress.
"""

import yaml
import os
import threading

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

def update_progress_in_yaml(cycle=None, current_trial=None, total_trials=None, current_rmse=None, current_mape=None):
    """
    Update progress metrics in the progress file.
    
    Parameters:
        cycle (int): Current tuning cycle number
        current_trial (int): Current trial number
        total_trials (int): Total number of trials in the current cycle
        current_rmse (float): The RMSE value from the current trial
        current_mape (float): The MAPE value from the current trial
    """
    # No need to create directory again since we do it at module level
    
    data = {}
    if cycle is not None:
        data["cycle"] = int(cycle)
    if current_trial is not None:
        data["current_trial"] = int(current_trial)
    if total_trials is not None:
        data["total_trials"] = int(total_trials)
        data["trial_progress"] = (float(current_trial) / float(total_trials)) if total_trials > 0 else 0.0
    if current_rmse is not None:
        data["current_rmse"] = float(current_rmse)
    if current_mape is not None:
        data["current_mape"] = float(current_mape)
    
    with open(PROGRESS_FILE, "w") as f:
        yaml.safe_dump(data, f)

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
