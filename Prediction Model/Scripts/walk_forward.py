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
