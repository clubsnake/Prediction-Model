# walk_forward.py
"""
Implements a simple walk-forward validation scheme.
"""

import numpy as np
from config import (
    PREDICTION_HORIZON, WALK_FORWARD_DEFAULT, START_DATE
)
from preprocessing import create_sequences
from utils import validate_walk_forward
from concurrent.futures import ThreadPoolExecutor
from model import build_model_by_type  # Make sure this is imported
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil

def run_walk_forward(model, df, feature_cols, target_col, lookback, horizon, window_size=None):
    """
    Perform walk-forward validation/training on the given model.

    :param model: A compiled/trained model (e.g. LSTM) that supports further .fit calls.
    :param df: DataFrame with features & target.
    :param feature_cols: List of feature columns used for X input.
    :param target_col: Target column name (e.g. "Close").
    :param lookback: Timesteps to look back in the past.
    :param horizon: Timesteps to predict into the future.
    :param window_size: Interval for walk-forward steps.
    :return: (model, predictions_wf, actuals_wf)
    """
    wf_window = validate_walk_forward(window_size or WALK_FORWARD_DEFAULT)
    predictions_wf = []
    actuals_wf = []
    cycle = 0

    start_idx = 0
    if "date" in df.columns:
        mask = df["date"] >= START_DATE
        valid_indices = df.index[mask]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]

    total_samples = len(df)
    for start in range(start_idx, total_samples - lookback - horizon + 1, wf_window):
        cycle += 1
        train_slice = df.iloc[: start + lookback]
        test_slice = df.iloc[start + lookback : start + lookback + horizon]

        X_train, y_train = create_sequences(train_slice, feature_cols, target_col, lookback, horizon)
        X_test, y_test = create_sequences(test_slice, feature_cols, target_col, lookback, horizon)
        if len(X_train) == 0 or len(X_test) == 0:
            break

        # Minimal retraining
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, shuffle=False)
        preds = model.predict(X_test)
        predictions_wf.append(preds)
        actuals_wf.append(y_test)

    return model, predictions_wf, actuals_wf

def get_model_prediction(mtype, submodel_params_dict, X_train, y_train, X_test, horizon, unified_lookback, feature_cols):
    """
    Get predictions from a specific model type using threading for parallel execution.
    
    Args:
        mtype: Model type (lstm, rnn, random_forest, xgboost)
        submodel_params_dict: Dictionary of hyperparameters for each model type
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        horizon: Prediction horizon
        unified_lookback: Lookback window size
        feature_cols: Feature column names
        
    Returns:
        Model predictions
    """
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
        # Use the tuned epochs and batch size.
        epochs = submodel_params_dict[mtype].get("epochs", 1)
        batch_size = submodel_params_dict[mtype].get("batch_size", 32)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        preds = model.predict(X_test)
        return preds
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
        preds_2d = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
        return preds_2d
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
        preds_2d = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
        return preds_2d

def train_parallel_models(model_configs, X_train, y_train, X_test, feature_cols, horizon):
    """
    Train multiple models in parallel with a hybrid approach:
    - ThreadPoolExecutor for neural network models (TF/Keras)
    - ProcessPoolExecutor for tree-based models (sklearn/XGBoost)
    
    Args:
        model_configs: List of (model_type, params) tuples
        X_train, y_train: Training data
        X_test: Test data for prediction
        feature_cols: Feature column names
        horizon: Prediction horizon
        
    Returns:
        Dictionary mapping model types to predictions
    """
    # Separate models by type
    nn_models = []
    tree_models = []
    
    for mtype, params in model_configs.items():
        if mtype in ['lstm', 'rnn']:
            nn_models.append((mtype, params))
        elif mtype in ['random_forest', 'xgboost']:
            tree_models.append((mtype, params))
    
    results = {}
    num_cores = max(1, psutil.cpu_count(logical=False) - 1)  # Reserve one core
    
    # Train neural network models with threading
    if nn_models:
        with ThreadPoolExecutor(max_workers=len(nn_models)) as executor:
            futures = {}
            for mtype, params in nn_models:
                future = executor.submit(
                    _train_nn_model, 
                    mtype, params, X_train, y_train, X_test, 
                    horizon, len(feature_cols)
                )
                futures[future] = mtype
                
            for future in futures:
                mtype = futures[future]
                try:
                    results[mtype] = future.result()
                except Exception as e:
                    print(f"Error training {mtype}: {e}")
                    results[mtype] = None
    
    # Train tree models with process pool
    if tree_models:
        # Convert numpy arrays to lists for pickling
        X_train_list = X_train.tolist() if isinstance(X_train, np.ndarray) else X_train
        y_train_list = y_train.tolist() if isinstance(y_train, np.ndarray) else y_train
        X_test_list = X_test.tolist() if isinstance(X_test, np.ndarray) else X_test
        
        with ProcessPoolExecutor(max_workers=min(len(tree_models), num_cores)) as executor:
            futures = {}
            for mtype, params in tree_models:
                future = executor.submit(
                    _train_tree_model_wrapper,
                    mtype, params, X_train_list, y_train_list, X_test_list, horizon
                )
                futures[future] = mtype
                
            for future in futures:
                mtype = futures[future]
                try:
                    results[mtype] = np.array(future.result())
                except Exception as e:
                    print(f"Error training {mtype}: {e}")
                    results[mtype] = None
    
    return results

# These helper functions handle the actual training
def _train_nn_model(mtype, params, X_train, y_train, X_test, horizon, num_features):
    """Train a neural network model (runs in a thread)"""
    from model import build_model_by_type
    import tensorflow as tf
    
    # Set thread local GPU memory growth
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Extract architecture params
    arch_params = {"units_per_layer": params["units_per_layer"]}
    if "use_batch_norm" in params:
        arch_params["use_batch_norm"] = params["use_batch_norm"]
    
    # Build model
    model = build_model_by_type(
        model_type=mtype,
        num_features=num_features,
        horizon=horizon,
        learning_rate=params["lr"],
        dropout_rate=params["dropout"],
        loss_function=params["loss_function"],
        lookback=params.get("lookback", X_train.shape[1]),
        architecture_params=arch_params
    )
    
    # Get training params
    epochs = params.get("epochs", 1)
    batch_size = params.get("batch_size", 32)
    
    # Train and predict
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_test, verbose=0)
    
    # Clean up
    tf.keras.backend.clear_session()
    
    return preds

def _train_tree_model_wrapper(mtype, params, X_train_list, y_train_list, X_test_list, horizon):
    """Wrapper for tree-based models that works with ProcessPoolExecutor"""
    # Convert back to numpy arrays
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    
    # Flatten inputs
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
            random_state=42,
            n_jobs=1  # Important: use 1 job within the pool
        )
    elif mtype == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=params["n_est"],
            learning_rate=params["lr"],
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            random_state=42,
            n_jobs=1  # Important: use 1 job within the pool
        )
    
    # Fit model
    model.fit(X_tr_flat, y_tr_flat)
    
    # Predict
    preds_1d = model.predict(X_te_flat)
    preds_2d = np.tile(preds_1d.reshape(-1, 1), (1, horizon))
    
    return preds_2d.tolist()  # Convert to list for pickling
