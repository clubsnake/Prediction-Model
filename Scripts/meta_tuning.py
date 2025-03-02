"""
Implements an Optuna-based hyperparameter search with ensemble approach and walk-forward.
Logs detailed trial information live to session state and YAML files:
- progress.yaml for current progress,
- best_params.yaml when thresholds are met,
- tested_models.yaml with details for each trial,
- cycle_metrics.yaml with cycle-level summaries.
"""

import numpy as np
import yaml
import sys
import os
import signal
import optuna
import logging
from datetime import datetime
import streamlit as st
import threading

from config import (
    N_STARTUP_TRIALS,
    TUNING_TRIALS_PER_CYCLE_max,
    TUNING_TRIALS_PER_CYCLE_min,
    TUNING_LOOP,
    RMSE_THRESHOLD,
    MAPE_THRESHOLD,
    MODEL_TYPES,
    LOSS_FUNCTIONS,
    START_DATE,
    get_active_feature_names,
    get_horizon_for_category,
    get_hyperparameter_ranges,
    TICKER,
    TIMEFRAMES
)


import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from progress_helper import update_progress_in_yaml, set_stop_requested, is_stop_requested


if "trial_logs" not in st.session_state:
    st.session_state["trial_logs"] = []

stop_requested = False
def set_stop_requested(val: bool):
    global stop_requested
    stop_requested = val

if sys.platform != "win32":
    def signal_handler(sig, frame):
        global stop_requested
        stop_requested = True
        print("\nManual stop requested. Exiting tuning loop.")
    signal.signal(signal.SIGINT, signal_handler)

from data import fetch_data
from features import feature_engineering_with_params
from preprocessing import scale_data, create_sequences
from model import evaluate_predictions, build_model_by_type
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils import clean_memory

# Data directories and file paths
DATA_DIR = "Data"
DB_DIR = os.path.join(DATA_DIR, "DB")
LOGS_DIR = os.path.join(DATA_DIR, "Logs")
MODELS_DIR = os.path.join(DATA_DIR, "Models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(DATA_DIR, "progress.yaml")
TESTED_MODELS_FILE = os.path.join(DATA_DIR, "tested_models.yaml")
TUNING_STATUS_FILE = os.path.join(DATA_DIR, "tuning_status.txt")
CYCLE_METRICS_FILE = os.path.join(DATA_DIR, "cycle_metrics.yaml")
BEST_PARAMS_FILE = os.path.join(DATA_DIR, "best_params.yaml")

# Global stop control via event
stop_event = threading.Event()

def set_stop_requested(val: bool):
    if val:
        stop_event.set()
    else:
        stop_event.clear()

def is_stop_requested():
    return stop_event.is_set()



def convert_to_builtin_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    else:
        return obj

def reset_progress():
    import yaml
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        yaml.safe_dump({}, f)

def prune_old_cycles(filename=CYCLE_METRICS_FILE, max_cycles=50):
    """
    Prune old cycle metrics from the YAML file if more than max_cycles exist.
    Keeps only the most recent max_cycles cycles.
    """
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                cycles = yaml.safe_load(f) or []
            if len(cycles) > max_cycles:
                # Keep only the last max_cycles cycles
                cycles = cycles[-max_cycles:]
                with open(filename, "w") as f:
                    yaml.safe_dump(convert_to_builtin_type(cycles), f)
                print(f"Pruned old cycles. Now storing {len(cycles)} cycles.")
    except Exception as e:
        print(f"Error pruning old cycles: {e}")


def get_model_prediction(mtype, submodel_params, X_train, y_train, X_test, horizon, unified_lookback, feature_cols):
    if (mtype in ["lstm", "rnn"]):
        arch_params = {
            "units_per_layer": submodel_params[mtype]["units_per_layer"],
            "use_batch_norm": submodel_params[mtype].get("use_batch_norm", False)  # Add batch norm param
        }
        model = build_model_by_type(
            model_type=mtype,
            num_features=len(feature_cols),
            horizon=horizon,
            learning_rate=submodel_params[mtype]["lr"],
            dropout_rate=submodel_params[mtype]["dropout"],
            loss_function=submodel_params[mtype]["loss_function"],
            lookback=unified_lookback,
            architecture_params=arch_params
        )
        # Use the tuned epochs and batch size
        epochs = submodel_params[mtype].get("epochs", 1)
        batch_size = submodel_params[mtype].get("batch_size", 32)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        preds = model.predict(X_test)
    elif mtype == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=submodel_params[mtype]["n_est"],
            max_depth=submodel_params[mtype]["mdepth"],
            min_samples_split=submodel_params[mtype].get("min_samples_split", 2),  # Use new param
            min_samples_leaf=submodel_params[mtype].get("min_samples_leaf", 1),    # Use new param
            random_state=42
        )
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        y_tr_flat = y_train[:, 0]
        rf.fit(X_tr_flat, y_tr_flat)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)
        preds_1d = rf.predict(X_te_flat)
        preds = np.tile(preds_1d.reshape(-1,1), (1, horizon))
    elif mtype == "xgboost":
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=submodel_params[mtype]["n_est"],
            learning_rate=submodel_params[mtype]["lr"],
            max_depth=submodel_params[mtype].get("max_depth", 6),           # Use new param
            subsample=submodel_params[mtype].get("subsample", 1.0),         # Use new param
            colsample_bytree=submodel_params[mtype].get("colsample_bytree", 1.0),  # Use new param
            random_state=42
        )
        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        y_tr_flat = y_train[:, 0]
        xgb_model.fit(X_tr_flat, y_tr_flat)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)
        preds_1d = xgb_model.predict(X_te_flat)
        preds = np.tile(preds_1d.reshape(-1,1), (1, horizon))
    return preds

def train_parallel_models(submodel_params_dict, X_train, y_train, X_test, feature_cols, horizon):
    """
    Train multiple models in parallel with a hybrid approach:
    - ThreadPoolExecutor for neural network models (TensorFlow)
    - ProcessPoolExecutor for tree-based models (sklearn/XGBoost)
    
    This design optimizes CPU usage based on model type.
    """
    predictions = {}
    
    # Set up shared parameters for all models
    unified_lookback = X_train.shape[1] if len(X_train.shape) > 2 else 1
    
    # Separate models by type for optimal parallelization
    nn_models = {}
    tree_models = {}
    
    for mtype, params in submodel_params_dict.items():
        if mtype in ["lstm", "rnn"]:
            nn_models[mtype] = params
        else:  # RandomForest, XGBoost
            tree_models[mtype] = params
    
    # Dynamic resource allocation
    import psutil
    total_cpu = psutil.cpu_count(logical=True)
    physical_cpu = psutil.cpu_count(logical=False) or 4
    
    # Neural network models with threads (better for TensorFlow)
    if nn_models:
        nn_workers = min(len(nn_models), max(1, total_cpu // 4))
        with ThreadPoolExecutor(max_workers=nn_workers) as executor:
            future_to_mtype = {}
            for mtype in nn_models:
                future = executor.submit(
                    get_model_prediction, 
                    mtype, submodel_params_dict, X_train, y_train, X_test, 
                    horizon, unified_lookback, feature_cols
                )
                future_to_mtype[future] = mtype
                
            for future in future_to_mtype:
                mtype = future_to_mtype[future]
                try:
                    predictions[mtype] = future.result()
                except Exception as e:
                    print(f"Error training {mtype} model: {e}")
                    predictions[mtype] = None
    
    # Train tree models with process pool (overcomes Python's GIL for CPU-bound work)
    if tree_models:
        # Convert numpy arrays to lists for pickling
        X_train_list = X_train.tolist() if isinstance(X_train, np.ndarray) else X_train
        y_train_list = y_train.tolist() if isinstance(y_train, np.ndarray) else y_train
        X_test_list = X_test.tolist() if isinstance(X_test, np.ndarray) else X_test
        
        # Optimal process count based on system resources
        tree_workers = min(len(tree_models), max(1, physical_cpu - 1))  # Leave 1 core free
        
        with ProcessPoolExecutor(max_workers=tree_workers) as executor:
            future_to_mtype = {}
            for mtype in tree_models:
                future = executor.submit(
                    _train_tree_model, 
                    mtype, submodel_params_dict[mtype], X_train_list, y_train_list, X_test_list, 
                    horizon
                )
                future_to_mtype[future] = mtype
                
            for future in future_to_mtype:
                mtype = future_to_mtype[future]
                try:
                    predictions[mtype] = np.array(future.result())
                except Exception as e:
                    print(f"Error training {mtype} model: {e}")
                    predictions[mtype] = None
    
    return predictions

def _train_tree_model(mtype, params, X_train_list, y_train_list, X_test_list, horizon):
    """Process-safe wrapper for training tree models"""
    # Convert lists back to numpy arrays
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    
    # Flatten for tree models
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    y_tr_flat = y_train[:, 0]
    X_te_flat = X_test.reshape(X_test.shape[0], -1)
    
    if mtype == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=params["n_est"],
            max_depth=params["mdepth"],
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            n_jobs=1,  # Important: Use 1 job within process pool
            random_state=42
        )
    elif mtype == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=params["n_est"],
            learning_rate=params["lr"],
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            tree_method='hist',  # Faster histogram-based algorithm
            n_jobs=1,  # Important: Use 1 job within process pool
            random_state=42
        )
    
    model.fit(X_tr_flat, y_tr_flat)
    preds_1d = model.predict(X_te_flat)
    preds_2d = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
    
    return preds_2d.tolist()  # Return as list for pickling

# Update the walk_forward_ensemble_eval function to check stop_requested:
def walk_forward_ensemble_eval(df, feature_cols, horizon, wf_size, submodel_params_dict, ensemble_weights):
    global stop_requested  # Add this line
    lookback_list = []
    for mp in submodel_params_dict.values():
        if "lookback" in mp:
            lookback_list.append(mp["lookback"])
    unified_lookback = max(lookback_list) if lookback_list else 30

    scaled_df, _, _ = scale_data(df, feature_cols, "Close")
    all_step_mse = []
    all_step_mape = []
    total_samples = len(scaled_df)
    idx = 0
    while True:
        # Add this stop check inside the loop
        if stop_requested:
            print("Stop requested during walk-forward evaluation")
            break
            
        if idx + unified_lookback + horizon > total_samples:
            break
        train_slice = scaled_df.iloc[: idx + unified_lookback]
        test_slice  = scaled_df.iloc[idx + unified_lookback : idx + unified_lookback + horizon]
        X_train, y_train = create_sequences(train_slice, feature_cols, "Close", unified_lookback, horizon)
        X_test, y_test  = create_sequences(test_slice, feature_cols, "Close", unified_lookback, horizon)
        if len(X_train) == 0 or len(X_test) == 0:
            break
        ensemble_preds = np.zeros_like(y_test, dtype=float)
        model_configs = {mtype: params for mtype, params in submodel_params_dict.items() 
                         if ensemble_weights.get(mtype, 0) >= 1e-9}
        
        if model_configs:
            predictions = train_parallel_models(
                model_configs, X_train, y_train, X_test, feature_cols, horizon
            )
            
            ensemble_preds = np.zeros_like(y_test, dtype=float)
            for mtype, weight in ensemble_weights.items():
                if weight < 1e-9 or mtype not in predictions or predictions[mtype] is None:
                    continue
                ensemble_preds += weight * predictions[mtype]
            
            # Clean memory after each step
            clean_memory(force_gc=False)  # No need to force GC every step
        step_mse, step_mape = evaluate_predictions(y_test, ensemble_preds)
        all_step_mse.append(step_mse)
        all_step_mape.append(step_mape)
        idx += wf_size
    if len(all_step_mse) == 0:
        return 1e9, 999
    avg_mse = np.mean(all_step_mse)
    avg_mape = np.mean(all_step_mape)
    return avg_mse, avg_mape

def ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat):
    global stop_requested
    if stop_requested:
        raise optuna.exceptions.TrialPruned("Stop requested globally.")
    
    # --- Hyperparameter suggestions with WIDER RANGES ---
    rsi_period      = trial.suggest_int("rsi_period", 5, 40)  # Expanded from 30
    macd_fast       = trial.suggest_int("macd_fast", 5, 20)   # Expanded from 8-15
    macd_slow       = trial.suggest_int("macd_slow", 15, 50)  # Expanded from 20-40
    macd_signal     = trial.suggest_int("macd_signal", 3, 20) # Expanded from 5-15
    boll_window     = trial.suggest_int("boll_window", 5, 60) # Expanded from 10-40
    boll_nstd       = trial.suggest_float("boll_nstd", 0.5, 4.0) # Expanded from 1.0-3.0
    atr_period      = trial.suggest_int("atr_period", 3, 40)  # Expanded from 5-30
    werpi_level     = trial.suggest_int("werpi_level", 1, 7)  # Expanded from 2-5
    werpi_n_states  = trial.suggest_int("werpi_n_states", 2, 8) # Expanded from 2-5
    werpi_scale     = trial.suggest_float("werpi_scale", 0.1, 4.0) # Expanded from 0.5-2.0
    werpi_wavelet   = trial.suggest_categorical("werpi_wavelet", ["db1", "db2", "db3", "db4", "db5", "db6", 
                                                                 "sym2", "sym3", "sym4", "sym5", 
                                                                 "coif1", "coif2", "coif3", 
                                                                 "haar", "dmey"])  # Added more wavelets
    wf_size         = trial.suggest_int("wf_size", 3, 90)     # Expanded from 5-60
    
    # --- Ensemble weights ---
    raw_weights = {}
    total_w = 0.0
    for mtype in MODEL_TYPES:
        w = trial.suggest_float(f"w_{mtype}", 0.0, 1.0)
        raw_weights[mtype] = w
        total_w += w
    if total_w < 1e-9:
        total_w = float(len(MODEL_TYPES))
        for mtype in MODEL_TYPES:
            raw_weights[mtype] = 1.0
    ensemble_weights = {m: raw_weights[m] / total_w for m in MODEL_TYPES}
    
    # --- Submodel hyperparameters with WIDER RANGES ---
    horizon = get_horizon_for_category(range_cat)
    submodel_params = {}
    for mtype in MODEL_TYPES:
        if mtype in ["lstm", "rnn"]:
            # WIDER RANGES FOR NEURAL NETWORKS
            loss_fn = trial.suggest_categorical(f"{mtype}_loss_function", LOSS_FUNCTIONS)
            lookb = trial.suggest_int(f"{mtype}_lookback", 1, 120)  # Expanded from 10-90
            lr = trial.suggest_float(f"{mtype}_lr", 1e-6, 1e-1, log=True)  # Expanded from 1e-5-1e-2
            dropout = trial.suggest_float(f"{mtype}_dropout", 0.01, 0.5, log=True)  # Expanded from 0.0-0.5
            
            # ADDED TUNING FOR EPOCHS AND BATCH SIZE
            epochs = trial.suggest_int(f"{mtype}_epochs", 1, 100)  # New parameter
            batch_size = trial.suggest_categorical(f"{mtype}_batch_size", 
                                     [16, 32, 64, 128, 256, 512, 1024, 2048])  # New parameter
            
            # ADDED BATCH NORMALIZATION
            use_batch_norm = trial.suggest_categorical(f"{mtype}_batch_norm", [True, False])
            
            # INCREASED NUMBER OF LAYERS
            num_layers = trial.suggest_int(f"{mtype}_num_layers", 1, 10)  # Expanded from 1-3
            
            units_list = []
            for i in range(num_layers):
                # WIDER RANGE OF UNITS PER LAYER
                layer_units = trial.suggest_categorical(f"{mtype}_units_layer_{i}", [16, 32, 64, 128, 256, 512])  # Added 256, 512
                units_list.append(layer_units)
                
            submodel_params[mtype] = {
                "loss_function": loss_fn,
                "lookback": lookb,
                "lr": lr,
                "dropout": dropout,
                "units_per_layer": units_list,
                "use_batch_norm": use_batch_norm,  # New parameter
                "epochs": epochs,  # New parameter
                "batch_size": batch_size  # New parameter
            }
        elif mtype == "random_forest":
            submodel_params[mtype] = {
                "n_est": trial.suggest_int("rf_n_est", 50, 1500),  # Expanded from 50-300
                "mdepth": trial.suggest_int("rf_mdepth", 3, 40),   # Expanded from 3-20
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 30),  # New parameter
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 20)     # New parameter
            }
        elif mtype == "xgboost":
            submodel_params[mtype] = {
                "n_est": trial.suggest_int("xgb_n_est", 50, 1500),         # Expanded from 50-300
                "lr": trial.suggest_float("xgb_lr", 1e-5, 0.5, log=True),  # Expanded from 1e-4-1e-1
                "max_depth": trial.suggest_int("xgb_max_depth", 2, 25),     # New parameter
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),# New parameter
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0)  # New parameter
            }
    
    # --- Data fetching and feature engineering ---
    df_raw = fetch_data(ticker, start=START_DATE, interval=timeframe)
    df_raw = feature_engineering_with_params(
        df_raw,
        ticker=ticker,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        boll_window=boll_window,
        boll_nstd=boll_nstd,
        atr_period=atr_period,
        werpi_wavelet=werpi_wavelet,
        werpi_level=werpi_level,
        werpi_n_states=werpi_n_states,
        werpi_scale=werpi_scale
    )
    feature_cols = get_active_feature_names()
    if df_raw is None or len(df_raw) < 50:
        return 1e9

    mse_val, mape_val = walk_forward_ensemble_eval(
        df=df_raw,
        feature_cols=feature_cols,
        horizon=horizon,
        wf_size=wf_size,
        submodel_params_dict=submodel_params,
        ensemble_weights=ensemble_weights
    )
    rmse = np.sqrt(mse_val)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("mape", mape_val)
    print(f"Trial {trial.number}: rmse = {rmse:.4f}, mape = {mape_val:.2f}%")
    
    # Record trial log for live display
    trial_logs = st.session_state.get('trial_logs', [])
    trial_logs.append({
        'trial_number': trial.number,
        'params': convert_to_builtin_type(trial.params),
        'rmse': float(rmse),
        'mape': float(mape_val),
        'ensemble_weights': ensemble_weights,
        'feature_params': {
            'rsi_period': rsi_period,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'boll_window': boll_window,
            'boll_nstd': boll_nstd,
            'atr_period': atr_period,
            'werpi_level': werpi_level,
            'werpi_n_states': werpi_n_states,
            'werpi_scale': werpi_scale,
            'werpi_wavelet': werpi_wavelet
        },
        'submodel_params': submodel_params,
        'timestamp': datetime.now().isoformat()
    })
    st.session_state['trial_logs'] = trial_logs

    if rmse <= RMSE_THRESHOLD and mape_val <= MAPE_THRESHOLD:
        os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)
        with open(BEST_PARAMS_FILE, "w") as f:
            yaml.safe_dump(convert_to_builtin_type(trial.params), f)
        print(f"Thresholds met (RMSE={rmse:.4f}, MAPE={mape_val:.2f}%)!")
    
    tested_models = []
    if os.path.exists(TESTED_MODELS_FILE):
        try:
            with open(TESTED_MODELS_FILE, "r") as f:
                tested_models = yaml.safe_load(f) or []
        except Exception as e:
            print(f"Error loading tested_models.yaml: {e}. Starting with empty list.")
            tested_models = []
    tested_models.append({
        'trial_number': trial.number,
        'params': convert_to_builtin_type(trial.params),
        'rmse': float(rmse),
        'mape': float(mape_val),
        'timestamp': datetime.now().isoformat()
    })
    os.makedirs(os.path.dirname(TESTED_MODELS_FILE), exist_ok=True)
    with open(TESTED_MODELS_FILE, "w") as f:
        yaml.safe_dump(convert_to_builtin_type(tested_models), f)

    objective_val = rmse + (mape_val / 100.0)
    return objective_val

def create_or_load_study(study_name):
    """Create or load an Optuna study with persistent storage."""
    try:
        # Use SQLite for persistent storage
        db_path = os.path.join(DB_DIR, f"{study_name}.db")
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True
        )
        return study
    except Exception as e:
        logging.error(f"Error creating/loading study: {e}")
        # Fallback to in-memory storage
        return optuna.create_study(direction="minimize")

# Add this callback to tune_for_combo function
def tune_for_combo(ticker, timeframe, range_cat="all", n_trials=None, cycle=1):
    """
    Perform hyperparameter tuning for a specific combination.
    """
    study_name = f"study_{ticker}_{timeframe}_{range_cat}"
    study = create_or_load_study(study_name)
    
    # Store trial count in study user attributes
    if n_trials is None:
        n_trials = TUNING_TRIALS_PER_CYCLE_max
    study.set_user_attr("n_trials", n_trials)
    
    # Debug output
    print(f"Starting tuning with {n_trials} trials (cycle {cycle})")
    
    # Create combined stop callback
    def stop_callback(study, trial):
        # Always check both mechanisms for compatibility
        if is_stop_requested() or stop_requested:
            print(f"Stop requested after trial {trial.number}. Stopping study.")
            try:
                # Update dashboard state
                st.session_state['tuning_in_progress'] = False
                # Update the file flag
                with open(TUNING_STATUS_FILE, "w") as f:
                    f.write("False")
                    f.flush()
                    os.fsync(f.fileno())
                st.session_state['tuning_recently_stopped'] = True
                st.session_state['stop_time'] = datetime.now()
            except:
                pass
            study.stop()  # This immediately stops Optuna
    
    # Run the optimization with both callbacks
    study.optimize(
        lambda trial: ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat),
        n_trials=n_trials,
        callbacks=[progress_callback, stop_callback]
    )
    return study

def main():
    from config import TICKER, TIMEFRAMES, TUNING_TRIALS_PER_CYCLE_max, TUNING_LOOP
    ticker = TICKER
    timeframe = TIMEFRAMES[0]

    # Determine if we should resume: if a progress file exists and the ticker/timeframe match.
    resume_mode = False
    saved_progress = {}
    progress_file = "progress.yaml"
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                saved_progress = yaml.safe_load(f) or {}
            # Check if the saved progress corresponds to the same ticker and timeframe.
            if saved_progress.get("ticker") == ticker and saved_progress.get("timeframe") == timeframe:
                resume_mode = True
        except Exception as e:
            print(f"Error reading progress file: {e}")
            # If error occurs, assume fresh start.
            resume_mode = False

    # If not resuming, clear progress and initialize with current ticker/timeframe.
    if not resume_mode:
        reset_progress()
        saved_progress = {
            "ticker": ticker,
            "timeframe": timeframe,
            "cycle": 1
        }
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            yaml.safe_dump(saved_progress, f)
        st.session_state["cycle"] = 1
        print("Starting fresh tuning session.")
    else:
        # If resuming, use the saved cycle (default to 1 if not found)
        st.session_state["cycle"] = saved_progress.get("cycle", 1)
        print(f"Resuming tuning session: cycle {st.session_state['cycle']} for {ticker} on {timeframe}.")

    # Always reset the stop flag at the start.
    set_stop_requested(False)
    cycle = st.session_state["cycle"]

    while TUNING_LOOP and not stop_requested:
        print(f"Starting cycle {cycle}, stop_requested = {stop_requested}")
        # Do not reset progress if resuming; it persists the current progress.
        horizon = get_horizon_for_category("all")
        trials = st.session_state.get("trials_per_cycle", TUNING_TRIALS_PER_CYCLE_max)
        study = tune_for_combo(ticker, timeframe, horizon, n_trials=trials, cycle=cycle)

        # Clear trial logs for the next cycle.
        st.session_state["trial_logs"] = []
        prune_old_cycles(filename="cycle_metrics.yaml", max_cycles=50)

        print(f"After cycle {cycle}, stop_requested = {stop_requested}")
        if stop_requested:
            print(f"Stop requested after cycle {cycle}. Exiting tuning loop.")
            break

        cycle += 1
        st.session_state["cycle"] = cycle
        # Update the progress file with the new cycle number.
        saved_progress["cycle"] = cycle
        with open(progress_file, "w") as f:
            yaml.safe_dump(saved_progress, f)
        # Reset the stop flag for the next cycle.
        set_stop_requested(False)
        clean_memory(force_gc=True)  # Force GC between cycles
    print("Continuous tuning loop finished.")
    
    # Add the metrics saving that was in your original code
    # Get the best parameters and values from the last study (if available)
    if 'study' in locals():
        print("Best Parameters:", study.best_params)
        print("Best RMSE+MAPE combo:", study.best_value)

    if st.session_state.get("trial_logs"):
        rmse_values = [log["rmse"] for log in st.session_state["trial_logs"] if log.get("rmse") is not None]
        mape_values = [log["mape"] for log in st.session_state["trial_logs"] if log.get("mape") is not None]
        if rmse_values and mape_values:
            cycle_metrics = {
                "cycle": cycle,  # Use the current cycle number
                "average_rmse": float(np.mean(rmse_values)),
                "average_mape": float(np.mean(mape_values)),
                "timestamp": datetime.now().isoformat()
            }
            cycle_metrics_list = []
            if os.path.exists("cycle_metrics.yaml"):
                with open("cycle_metrics.yaml", "r") as f:
                    cycle_metrics_list = yaml.safe_load(f) or []
            cycle_metrics_list.append(cycle_metrics)
            with open("cycle_metrics.yaml", "w") as f:
                yaml.safe_dump(convert_to_builtin_type(cycle_metrics_list), f)

if __name__ == "__main__":
    main()

def get_horizon_for_category(range_cat):
    """
    Convert a range category to number of days.
    
    Args:
        range_cat: A string like "1w", "2w", "1m", "3m", "all" or an integer.
    
    Returns:
        int: Number of days for the horizon.
    """
    # Handle the case where range_cat is already an integer
    if isinstance(range_cat, int):
        return range_cat
    
    # Otherwise, try to process it as a string
    try:
        range_cat_str = str(range_cat).lower()
        if range_cat_str == "1w":
            return 7
        elif range_cat_str == "2w":
            return 14
        elif range_cat_str == "1m" or range_cat_str == "month":
            return 30
        elif range_cat_str == "3m" or range_cat_str == "quarter":
            return 90
        elif range_cat_str == "all":
            return 30  # Default for "all"
        else:
            # Try to convert to int directly
            return int(range_cat)
    except Exception as e:
        print(f"Error in get_horizon_for_category: {e}")
        return 30  # Default fallback

def update_progress(current_trial, total_trials, current_rmse, current_mape, cycle=1):
    """Update progress tracking file."""
    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists
    progress = {
        "current_trial": current_trial,
        "total_trials": total_trials,
        "trial_progress": current_trial / total_trials if total_trials > 0 else 0,
        "current_rmse": current_rmse,
        "current_mape": current_mape,
        "cycle": cycle
    }
    try:
        with open(PROGRESS_FILE, "w") as f:
            yaml.dump(progress, f)
    except Exception as e:
        print(f"Error writing progress: {e}")

# Add these imports near the top if not already present
import os
import yaml
import time
from datetime import datetime
import platform

# Cross-platform file locking
if platform.system() == 'Windows':
    # Windows-specific file locking
    import msvcrt
    
    def lock_file(file_handle):
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except IOError:
            return False
    
    def unlock_file(file_handle):
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            return True
        except IOError:
            return False
else:
    # Unix/Linux/Mac file locking
    import fcntl
    
    def lock_file(file_handle):
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except IOError:
            return False
    
    def unlock_file(file_handle):
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
            return True
        except IOError:
            return False

# Improved progress_callback with hybrid approach - both file and session state
def progress_callback(study, trial):
    """Update progress via both file system and session state for best performance"""
    current_trial = trial.number + 1  # Zero-indexed to one-indexed
    total_trials = study.user_attrs.get("n_trials", TUNING_TRIALS_PER_CYCLE_max)
    current_rmse = trial.user_attrs.get("rmse", None)
    current_mape = trial.user_attrs.get("mape", None)
    cycle = st.session_state.get("cycle", 1)
    
    # Create progress data once
    progress_data = {
        "current_trial": current_trial,
        "total_trials": total_trials,
        "trial_progress": current_trial / total_trials if total_trials > 0 else 0,
        "current_rmse": current_rmse,
        "current_mape": current_mape,
        "cycle": cycle,
        "timestamp": time.time()  # Add timestamp for freshness check
    }
    
    # 1. Direct session state update (fast, no file I/O)
    try:
        st.session_state['live_progress'] = progress_data
    except:
        pass  # Fail silently if not in Streamlit context
    
    # 2. Also write to file for persistence (keeps working if dashboard restarts)
    try:
        with open(PROGRESS_FILE, "w") as f:
            yaml.safe_dump(progress_data, f)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Error updating progress file: {e}")