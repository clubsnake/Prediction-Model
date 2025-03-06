# hyperparameter_tuning.py
"""
Implements a GridSearchCV-based hyperparameter search using scikeras KerasRegressor.
"""

import logging
import os
import sys

import joblib
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from config import (
    FULL_GRID,
    GRID_SEARCH_TYPE,
    INTERVAL,
    LOOKBACK,
    LOSS_FUNCTIONS,
    MODEL_TYPES,
    NORMAL_GRID,
    ORIGINAL_LEARNING_RATE,
    PREDICTION_HORIZON,
    START_DATE,
    THOROUGH_GRID,
    TICKER,
    get_active_feature_names,
)
from models.model import build_model_by_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def my_model_builder(
    model_type,
    num_features,
    horizon,
    learning_rate,
    dropout_rate,
    loss_function,
    lookback,
    architecture_params,
):
    return build_model_by_type(
        model_type,
        num_features,
        horizon,
        learning_rate,
        dropout_rate,
        loss_function,
        lookback,
        architecture_params,
    )


regressor = KerasRegressor(
    model=my_model_builder,
    model_type="lstm",
    num_features=len(get_active_feature_names()),
    horizon=PREDICTION_HORIZON,
    learning_rate=ORIGINAL_LEARNING_RATE,
    dropout_rate=0.2,
    loss_function="mean_squared_error",
    lookback=LOOKBACK,
    architecture_params={"units_per_layer": [64]},
    epochs=50,
    batch_size=16,
    verbose=0,
)

if GRID_SEARCH_TYPE == "normal":
    grid = {
        "regressor__epochs": NORMAL_GRID["epochs"],
        "regressor__batch_size": NORMAL_GRID["batch_size"],
        "regressor__learning_rate": NORMAL_GRID["learning_rate"],
        "regressor__lookback": NORMAL_GRID["lookback"],
        "regressor__dropout_rate": NORMAL_GRID["dropout_rate"],
        "regressor__loss_function": LOSS_FUNCTIONS,
        "regressor__model_type": MODEL_TYPES,
        "regressor__architecture_params": [
            {"units_per_layer": [64]},
            {"units_per_layer": [128, 64]},
        ],
    }
elif GRID_SEARCH_TYPE == "thorough":
    grid = {
        "regressor__epochs": THOROUGH_GRID["epochs"],
        "regressor__batch_size": THOROUGH_GRID["batch_size"],
        "regressor__learning_rate": THOROUGH_GRID["learning_rate"],
        "regressor__lookback": THOROUGH_GRID["lookback"],
        "regressor__dropout_rate": THOROUGH_GRID["dropout_rate"],
        "regressor__loss_function": LOSS_FUNCTIONS,
        "regressor__model_type": MODEL_TYPES,
        "regressor__architecture_params": [
            {"units_per_layer": [64]},
            {"units_per_layer": [128, 64]},
            {"units_per_layer": [128, 128]},
        ],
    }
elif GRID_SEARCH_TYPE == "full":
    grid = {
        "regressor__epochs": FULL_GRID["epochs"],
        "regressor__batch_size": FULL_GRID["batch_size"],
        "regressor__learning_rate": FULL_GRID["learning_rate"],
        "regressor__lookback": FULL_GRID["lookback"],
        "regressor__dropout_rate": FULL_GRID["dropout_rate"],
        "regressor__loss_function": LOSS_FUNCTIONS,
        "regressor__model_type": MODEL_TYPES,
        "regressor__architecture_params": [
            {"units_per_layer": [64]},
            {"units_per_layer": [128, 64]},
            {"units_per_layer": [128, 128]},
            {"units_per_layer": [256, 128]},
        ],
    }
else:
    raise ValueError("Invalid GRID_SEARCH_TYPE in config")

pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])


def main_training_loop(
    num_cycles=1, X_train=None, y_train=None, X_val=None, y_val=None
):
    """
    Main routine for grid search. Repeats for num_cycles if desired.
    """
    best_score = -np.inf
    best_model = None

    for cycle in range(num_cycles):
        logger.info("Cycle %d: Running GridSearchCV...", cycle + 1)
        grid_search = GridSearchCV(pipeline, grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        score = grid_search.best_score_
        logger.info("Cycle %d: Best CV Score: %.4f", cycle + 1, score)
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            joblib.dump(best_model, "best_dynamic_model.pkl")
            logger.info(
                "Cycle %d: New best model saved with score %.4f", cycle + 1, best_score
            )

    logger.info("Grid Search complete. Best overall score: %.4f", best_score)
    if best_model is not None and X_val is not None and y_val is not None:
        val_score = best_model.score(X_val, y_val)
        logger.info("Validation Score of best model: %.4f", val_score)
    return best_model


if __name__ == "__main__":
    from src.features.features import feature_engineering
    from src.data.preprocessing import create_sequences, scale_data
    from sklearn.model_selection import train_test_split

    from data.data import fetch_data

    df = fetch_data(TICKER, start=START_DATE, interval=INTERVAL)
    df = feature_engineering(df)
    target_col = "Close"
    feature_cols = [c for c in df.columns if c not in ["date", target_col]]
    scaled_df, _, _ = scale_data(df, feature_cols, target_col)
    X, y = create_sequences(
        scaled_df,
        feature_cols,
        target_col,
        lookback=LOOKBACK,
        horizon=PREDICTION_HORIZON,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    best_model = main_training_loop(
        num_cycles=1, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )
