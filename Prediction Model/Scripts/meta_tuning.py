# meta_tuning.py
"""
Implements an Optuna-based hyperparameter search with ensemble approach and walk-forward.
"""

import numpy as np
import yaml
import sys
import os
import signal
import optuna
import logging
from config import N_STARTUP_TRIALS, TUNING_TRIALS_PER_CYCLE_max, TUNING_LOOP, RMSE_THRESHOLD, MAPE_THRESHOLD, MODEL_TYPES, LOSS_FUNCTIONS, START_DATE, get_active_feature_names, get_horizon_for_category, get_hyperparameter_ranges
from progress_helper import update_progress_in_yaml
from datetime import datetime
import streamlit as st

stop_requested = False
def set_stop_requested(val: bool):
    """
    Set global stop_requested to True or False.
    """
    global stop_requested
    stop_requested = val

if sys.platform != "win32":
    def signal_handler(sig, frame):
        global stop_requested
        stop_requested = True
        print("\nManual stop requested. Exiting tuning loop.")
    signal.signal(signal.SIGINT, signal_handler)

def update_progress_in_yaml(cycle=None, current_trial=None, total_trials=None, current_rmse=None, current_mape=None):
    """
    Update a local progress.yaml file with cycle, current_trial, total_trials, current_rmse, and current_mape.
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

from data import fetch_data
from features import feature_engineering_with_params
from preprocessing import scale_data, create_sequences
from model import evaluate_predictions, build_model_by_type

def walk_forward_ensemble_eval(df, feature_cols, horizon, wf_size, submodel_params_dict, ensemble_weights):
    """
    Splits data in walk-forward increments, trains each submodel for 1 epoch
    on the respective training slice, then accumulates predictions for an ensemble.
    """
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

    import numpy as np
    while True:
        if idx + unified_lookback + horizon > total_samples:
            break
        train_slice = scaled_df.iloc[: idx + unified_lookback]
        test_slice  = scaled_df.iloc[idx + unified_lookback : idx + unified_lookback + horizon]
        X_train, y_train = create_sequences(train_slice, feature_cols, "Close", unified_lookback, horizon)
        X_test, y_test  = create_sequences(test_slice, feature_cols, "Close", unified_lookback, horizon)
        if len(X_train) == 0 or len(X_test) == 0:
            break

        ensemble_preds = np.zeros_like(y_test, dtype=float)
        for mtype, wval in ensemble_weights.items():
            if wval < 1e-9:
                continue
            if mtype in ["lstm", "rnn"]:
                arch_params = {"units_per_layer": submodel_params_dict[mtype]["units_per_layer"]}
                model = build_model_by_type(
                    model_type=mtype,
                    num_features=len(feature_cols),
                    horizon=horizon,
                    learning_rate=submodel_params_dict[mtype]["lr"],
                    dropout_rate=submodel_params_dict[mtype]["dropout"],
                    loss_function=submodel_params_dict[mtype]["loss_function"],
                    lookback=unified_lookback,
                    architecture_params=arch_params
                )
                model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
                preds = model.predict(X_test)
                ensemble_preds += wval * preds
            elif mtype == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(
                    n_estimators=submodel_params_dict[mtype]["n_est"],
                    max_depth=submodel_params_dict[mtype]["mdepth"],
                    random_state=42
                )
                X_tr_flat = X_train.reshape(X_train.shape[0], -1)
                y_tr_flat = y_train[:, 0]
                rf.fit(X_tr_flat, y_tr_flat)
                X_te_flat = X_test.reshape(X_test.shape[0], -1)
                preds_1d = rf.predict(X_te_flat)
                preds_2d = np.tile(preds_1d.reshape(-1,1), (1, horizon))
                ensemble_preds += wval * preds_2d
            elif mtype == "xgboost":
                import xgboost as xgb
                xgb_model = xgb.XGBRegressor(
                    n_estimators=submodel_params_dict[mtype]["n_est"],
                    learning_rate=submodel_params_dict[mtype]["lr"],
                    random_state=42
                )
                X_tr_flat = X_train.reshape(X_train.shape[0], -1)
                y_tr_flat = y_train[:, 0]
                xgb_model.fit(X_tr_flat, y_tr_flat)
                X_te_flat = X_test.reshape(X_test.shape[0], -1)
                preds_1d = xgb_model.predict(X_te_flat)
                preds_2d = np.tile(preds_1d.reshape(-1,1), (1, horizon))
                ensemble_preds += wval * preds_2d

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
    """
    Objective function for Optuna. Builds an ensemble of submodels 
    and performs walk-forward to evaluate performance.
    """
    global stop_requested
    if stop_requested:
        raise optuna.exceptions.TrialPruned("Stop requested globally.")

    # Feature-engineering hyperparams
    rsi_period      = trial.suggest_int("rsi_period", 5, 30)
    macd_fast       = trial.suggest_int("macd_fast", 8, 15)
    macd_slow       = trial.suggest_int("macd_slow", 20, 40)
    macd_signal     = trial.suggest_int("macd_signal", 5, 15)
    boll_window     = trial.suggest_int("boll_window", 10, 40)
    boll_nstd       = trial.suggest_float("boll_nstd", 1.0, 3.0)
    atr_period      = trial.suggest_int("atr_period", 5, 30)
    werpi_level     = trial.suggest_int("werpi_level", 2, 5)
    werpi_n_states  = trial.suggest_int("werpi_n_states", 2, 5)
    werpi_scale     = trial.suggest_float("werpi_scale", 0.5, 2.0)
    werpi_wavelet   = trial.suggest_categorical("werpi_wavelet", ["db2", "db3", "db4", "db5"])
    wf_size         = trial.suggest_int("wf_size", 5, 60)

    # Ensemble weights
    import numpy as np
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
    ensemble_weights = {m: raw_weights[m]/total_w for m in MODEL_TYPES}

    # Submodel hyperparams
    horizon = get_horizon_for_category(range_cat)
    submodel_params = {}
    for mtype in MODEL_TYPES:
        if mtype in ["lstm", "rnn"]:
            loss_fn = trial.suggest_categorical(f"{mtype}_loss_function", LOSS_FUNCTIONS)
            lookb = trial.suggest_int(f"{mtype}_lookback", 10, 90)
            lr = trial.suggest_float(f"{mtype}_lr", 1e-5, 1e-2, log=True)
            dropout = trial.suggest_float(f"{mtype}_dropout", 0.0, 0.5)
            num_layers = trial.suggest_int(f"{mtype}_num_layers", 1, 3)
            units_list = []
            for i in range(num_layers):
                layer_units = trial.suggest_categorical(f"{mtype}_units_layer_{i}", [16, 32, 64, 128])
                units_list.append(layer_units)
            submodel_params[mtype] = {
                "loss_function": loss_fn,
                "lookback": lookb,
                "lr": lr,
                "dropout": dropout,
                "units_per_layer": units_list
            }
        elif mtype == "random_forest":
            submodel_params[mtype] = {
                "n_est": trial.suggest_int("rf_n_est", 50, 300),
                "mdepth": trial.suggest_int("rf_mdepth", 3, 20)
            }
        elif mtype == "xgboost":
            submodel_params[mtype] = {
                "n_est": trial.suggest_int("xgb_n_est", 50, 300),
                "lr": trial.suggest_float("xgb_lr", 1e-4, 1e-1, log=True)
            }

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

    if rmse <= RMSE_THRESHOLD and mape_val <= MAPE_THRESHOLD:
        print(f"Thresholds met (RMSE={rmse:.4f}, MAPE={mape_val:.2f}%). Stopping tuning.")
        set_stop_requested(True)

    try:
        import streamlit as st
        trial_logs = st.session_state.get('trial_logs', [])
        trial_logs.append({
            'trial_number': trial.number,
            'params': trial.params,
            'rmse': rmse,
            'mape': mape_val,
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
    except Exception as e:
        logging.warning(f"Could not log trial to session state: {e}")

    objective_val = rmse + (mape_val / 100.0)
    return objective_val

def create_or_load_study(study_name):
    """
    Create or load an existing study by name, with in-memory storage or DB.
    """
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=None,
            direction="minimize",
            load_if_exists=True
        )
        return study
    except Exception as e:
        logging.error(f"Error creating/loading study: {e}")
        return optuna.create_study(direction="minimize")

def tune_for_combo(ticker, timeframe, horizon, window_size=None, n_trials=None):
    # 1) If n_trials is None, fallback to config or TUNING_TRIALS_PER_CYCLE_max
    if n_trials is None:
        from config import TUNING_TRIALS_PER_CYCLE_max
        n_trials = TUNING_TRIALS_PER_CYCLE_max

    study_name = f"study_{ticker}_{timeframe}_{horizon}"
    study = create_or_load_study(study_name)

    def objective(trial):
        # your existing objective code...
        pass

    # Now call study.optimize with the user-specified n_trials
    study.optimize(objective, n_trials=n_trials)

    # the rest of your logic...
    return study


def main():
    """
    Default entry point for meta_tuning: runs an example study with ensemble approach.
    """
    from config import TICKER, TIMEFRAMES, TUNING_TRIALS_PER_CYCLE_max
    ticker = TICKER
    timeframe = TIMEFRAMES[0]
    range_cat = "all"
    study_name = f"ensemble_study_{ticker}_{timeframe}"
    study = optuna.create_study(direction="minimize", study_name=study_name, load_if_exists=True)

    def objective(trial):
        objective_val = ensemble_with_walkforward_objective(trial, ticker, timeframe, range_cat)
        # Retrieve the RMSE and MAPE set as user attributes
        current_rmse = trial.user_attrs.get("rmse", None)
        current_mape = trial.user_attrs.get("mape", None)
        update_progress_in_yaml(
            cycle=1,
            current_trial=trial.number + 1,
            total_trials=TUNING_TRIALS_PER_CYCLE_max,
            current_rmse=current_rmse,
            current_mape=current_mape
        )
        return objective_val

    study.optimize(objective, n_trials=TUNING_TRIALS_PER_CYCLE_max)
    print("Best Parameters:", study.best_params)
    print("Best RMSE+MAPE combo:", study.best_value)
