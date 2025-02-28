# progress_helper.py
"""
Helper functions to update and save progress metrics to a YAML file.
This file is used by meta_tuning.py to log the current tuning progress,
including cycle, current trial, trial progress fraction, RMSE, and MAPE.
"""

import yaml

def update_progress_in_yaml(cycle=None, current_trial=None, total_trials=None, current_rmse=None, current_mape=None):
    """
    Update progress metrics in 'progress.yaml'.

    Parameters:
        cycle (int): Current tuning cycle number.
        current_trial (int): Current trial number.
        total_trials (int): Total number of trials in the current cycle.
        current_rmse (float): The RMSE value from the current trial.
        current_mape (float): The MAPE value from the current trial.
    """
    data = {}
    if cycle is not None:
        data["cycle"] = cycle
    if current_trial is not None:
        data["current_trial"] = current_trial
    if total_trials is not None:
        data["total_trials"] = total_trials
        if total_trials > 0 and current_trial is not None:
            data["trial_progress"] = float(current_trial) / float(total_trials)
        else:
            data["trial_progress"] = 0.0
    if current_rmse is not None:
        data["current_rmse"] = current_rmse
    if current_mape is not None:
        data["current_mape"] = current_mape
    with open("progress.yaml", "w") as f:
        yaml.safe_dump(data, f)
