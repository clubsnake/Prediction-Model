# progress_helper.py
"""
Helper functions for tracking and managing tuning progress.
"""
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict

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
)

# Make sure directories exist
os.makedirs(os.path.dirname(os.path.abspath(PROGRESS_FILE)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(TESTED_MODELS_FILE)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(CYCLE_METRICS_FILE)), exist_ok=True)

# Add thread-safe locks for global variables
_lock = threading.RLock()

# Global stop control via threading.Event
stop_event = threading.Event()


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


def update_progress_in_yaml(progress_data: Dict[str, Any]) -> bool:
    """Update the progress.yaml file with the current progress."""
    max_attempts = 5
    attempt = 0

    # First validate that we have a path
    if not PROGRESS_FILE:
        print("ERROR: PROGRESS_FILE path is empty or not defined")
        return False

    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(PROGRESS_FILE)), exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create directory for {PROGRESS_FILE}: {e}")
        return False

    while attempt < max_attempts:
        try:
            # Ensure data is clean
            clean_data = {k: v for k, v in progress_data.items() if v is not None}

            # Add timestamp if not present
            if "timestamp" not in clean_data:
                clean_data["timestamp"] = time.time()

            print(
                f"Writing progress to {PROGRESS_FILE} (attempt {attempt+1}/{max_attempts})"
            )

            # Write directly with minimal complexity
            temp_path = f"{PROGRESS_FILE}.tmp.{int(time.time())}"
            with open(temp_path, "w") as f:
                yaml.safe_dump(clean_data, f)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Rename temp file to target file (atomic operation)
            os.replace(temp_path, PROGRESS_FILE)
            print(f"Successfully updated progress file: {PROGRESS_FILE}")
            return True

        except Exception as e:
            attempt += 1
            print(f"Error updating progress (attempt {attempt}/{max_attempts}): {e}")
            import traceback

            traceback.print_exc()
            time.sleep(0.5)  # Short delay before retry

    print(f"CRITICAL: Failed to update progress file after {max_attempts} attempts")
    return False


def read_progress_from_yaml() -> Dict[str, Any]:
    """
    Read the current progress from the progress.yaml file.

    Returns:
        Dict with progress information
    """
    return safe_read_yaml(PROGRESS_FILE)


def write_tuning_status(status_info: Dict[str, Any]) -> bool:
    """
    Write the current tuning status to a file for coordination between processes.

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
        with FileLock(TUNING_STATUS_FILE):
            # Write in simple text format for better compatibility
            with open(TUNING_STATUS_FILE, "w") as f:
                for key, value in status_info.items():
                    f.write(f"{key}: {value}\n")
            print(
                f"Successfully wrote tuning status to text file: {TUNING_STATUS_FILE}"
            )
            return True
    except Exception as e:
        print(f"Error writing tuning status: {e}")
        traceback.print_exc()
        return False


def read_tuning_status() -> Dict[str, Any]:
    """
    Read the current tuning status from file.

    Returns:
        Dict with status information
    """
    status = {"is_running": False}

    if os.path.exists(TUNING_STATUS_FILE):
        try:
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

    return status


def update_trial_info_in_yaml(file_path: str, trial_info: Dict[str, Any]) -> bool:
    """
    Append trial information to the tested_models.yaml file.

    Args:
        file_path: Path to the YAML file (typically TESTED_MODELS_FILE)
        trial_info: Dictionary with trial information

    Returns:
        bool: True if successful, False otherwise
    """
    try:
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
        print(
            f"Writing trial info to {file_path} - Trial {trial_info.get('number', 'unknown')}"
        )

        # Try to append using threadsafe utility
        result = append_to_yaml_list(file_path, trial_info)

        # If that fails, try direct append with more robust error handling
        if not result:
            try:
                # Read existing data first
                existing_data = []
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            existing_data = yaml.safe_load(f) or []
                    except Exception as read_err:
                        print(
                            f"Error reading existing data, starting fresh: {read_err}"
                        )
                        existing_data = []

                # Ensure existing_data is a list
                if not isinstance(existing_data, list):
                    print(
                        f"Warning: {file_path} contained non-list data, resetting to empty list"
                    )
                    existing_data = []

                # Append the new trial with timestamp if not present
                if "timestamp" not in trial_info:
                    trial_info["timestamp"] = datetime.now().isoformat()

                existing_data.append(trial_info)

                # Write back to file with direct approach
                with open(file_path, "w") as f:
                    yaml.safe_dump(existing_data, f)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

                print(
                    f"Successfully wrote trial {trial_info.get('number', 'unknown')} to {file_path} (direct method)"
                )
                return True
            except Exception as e:
                print(f"Error directly appending to {file_path}: {e}")
                traceback.print_exc()

                # Ultimate fallback: Try writing to a new file
                try:
                    backup_path = f"{file_path}.{int(time.time())}.bak"
                    print(f"Attempting to write to alternate file: {backup_path}")
                    with open(backup_path, "w") as f:
                        yaml.safe_dump([trial_info], f)
                    return False
                except Exception as e2:
                    print(f"Failed even backup write attempt: {e2}")
                    return False

        print(
            f"Successfully wrote trial {trial_info.get('number', 'unknown')} to {file_path}"
        )
        return result
    except Exception as e:
        print(f"Error updating trial info: {e}")
        traceback.print_exc()
        return False


def update_cycle_metrics(cycle_data: Dict[str, Any], cycle_num: int = None) -> bool:
    """
    Update the cycle metrics in the cycle_metrics.yaml file.

    Args:
        cycle_data: Dictionary with cycle metrics
        cycle_num: Cycle number to use as an index

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read existing cycle metrics
        existing_cycles = safe_read_yaml(CYCLE_METRICS_FILE)

        # If file is empty or not a list, initialize with empty list
        if not existing_cycles or not isinstance(existing_cycles, list):
            existing_cycles = []

        # If cycle_num is provided, try to update that specific cycle
        if cycle_num is not None:
            # Find matching cycle or append new one
            found = False
            for i, cycle in enumerate(existing_cycles):
                if cycle.get("cycle") == cycle_num:
                    existing_cycles[i] = cycle_data
                    found = True
                    break

            if not found:
                existing_cycles.append(cycle_data)
        else:
            # Just append the new cycle data
            existing_cycles.append(cycle_data)

        # Limit to 50 most recent cycles
        if len(existing_cycles) > 50:
            existing_cycles = existing_cycles[-50:]

        # Write back to file
        return safe_write_yaml(CYCLE_METRICS_FILE, existing_cycles)

    except Exception as e:
        print(f"Error updating cycle metrics: {e}")
        return False


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
