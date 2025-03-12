# Study Manager for hyperparameter optimization
# Provides centralized management of Optuna studies for different model types

import logging
import os
import sys
from datetime import datetime

import optuna

# Configure logger before using it
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Import cross-platform locking from your existing threadsafe module
try:
    from src.utils.threadsafe import FileLock
except ImportError:
    logger.warning("Could not import FileLock from threadsafe, using fallback implementation")
    # Simple fallback implementation if import fails
    class FileLock:
        def __init__(self, filepath, timeout=10.0, retry_delay=0.1):
            self.filepath = filepath
        
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

# Fix import path
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tuning.progress_helper import TESTED_MODELS_FILE, update_trial_info_in_yaml

"""
Hyperparameter optimization study management module.

This module provides functionality to create and manage Optuna studies for
hyperparameter optimization. It includes:

1. Functions to create objective functions for different model types
2. Study persistence and management utilities
3. Result evaluation and processing tools

The study_manager connects the model definitions with the optimization process,
allowing systematic exploration of hyperparameter spaces for each model type.
It's used by the meta_tuning module and the hyperparameter tuning dashboard.
"""


def create_model_objective(
    model_type, ticker, timeframe, range_cat, metric_weights=None
):
    """
    Create an Optuna objective function for optimizing a specific model type.

    This function generates an objective function that can be passed to Optuna
    for hyperparameter optimization. The objective function:
    1. Creates model instances with trial-suggested parameters
    2. Trains and evaluates the model using walk-forward validation
    3. Returns a metric value to be minimized (typically RMSE or weighted metrics)

    Args:
        model_type: Type of model to optimize (e.g., 'lstm', 'xgboost')
        ticker: Stock/crypto symbol to use for optimization
        timeframe: Time interval for data (e.g., '1d', '1h')
        range_cat: Range category for prediction horizon
        metric_weights: Optional dictionary of weights for different metrics

    Returns:
        callable: An objective function that can be passed to Optuna's optimize method
    """
    # Default weights if none provided
    if metric_weights is None:
        metric_weights = {
            "rmse": 1.0,  # Root Mean Squared Error
            "mape": 0.5,  # Mean Absolute Percentage Error
            "da": 0.3,  # Directional Accuracy (higher is better, will be inverted)
        }

    # Normalize weights
    total_weight = sum(metric_weights.values())
    normalized_weights = {k: v / total_weight for k, v in metric_weights.items()}

    def objective(trial):
        # Force this model type
        trial.suggest_categorical("model_type", [model_type])

        # Use your existing evaluation function
        results = evaluate_model_with_walkforward(
            trial, ticker, timeframe, range_cat, model_type
        )

        # Extract individual metrics
        rmse = results.get("rmse", float("inf"))
        mape = results.get("mape", float("inf"))

        # Directional accuracy (higher is better, invert for Optuna's minimization)
        da = results.get("directional_accuracy", 0)
        inverted_da = 100 - da  # Assuming DA is in percentage (0-100)

        # Calculate weighted score
        weighted_score = (
            normalized_weights.get("rmse", 0) * rmse
            + normalized_weights.get("mape", 0) * mape
            + normalized_weights.get("da", 0) * inverted_da
        )

        # Store metrics in trial
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("mape", mape)
        trial.set_user_attr("directional_accuracy", da)
        trial.set_user_attr("combined_score", weighted_score)
        trial.set_user_attr("model_type", model_type)

        # Log progress
        logger.info(
            f"Trial {trial.number} ({model_type}): RMSE={rmse:.4f}, MAPE={mape:.2f}%, DA={da:.2f}%"
        )
        logger.info(f"Combined score: {weighted_score:.4f}")

        # Save trial info
        trial_info = {
            "number": trial.number,
            "params": trial.params,
            "model_type": model_type,
            "value": float(weighted_score),
            "metrics": {
                "rmse": float(rmse),
                "mape": float(mape),
                "directional_accuracy": float(da),
                "combined_score": float(weighted_score),
            },
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "timeframe": timeframe,
        }

        update_trial_info_in_yaml(TESTED_MODELS_FILE, trial_info)

        return weighted_score

    return objective


def evaluate_model_with_walkforward(
    trial, ticker, timeframe, range_cat, model_type=None
):
    """
    Evaluate a single model using walk-forward validation.

    Args:
        trial: Optuna trial object
        ticker: Ticker symbol
        timeframe: Timeframe
        range_cat: Range category
        model_type: Type of model to evaluate

    Returns:
        Dict with evaluation metrics
    """
    # Get active model types from config
    from config.config_loader import ACTIVE_MODEL_TYPES

    # If model_type not specified, get it from trial params
    if model_type is None:
        model_type = trial.params.get("model_type")
        if model_type is None:
            # Default to whatever model type is being suggested
            model_type = trial.suggest_categorical("model_type", ACTIVE_MODEL_TYPES)

    # Store the model type in trial user attributes
    trial.set_user_attr("model_type", model_type)

    # Get model-specific hyperparameters
    params = suggest_model_hyperparameters(trial, model_type)

    try:
        # Get data for this ticker and timeframe
        from src.data.data import fetch_data

        df = fetch_data(ticker, timeframe)

        # Get feature columns
        from config.config_loader import get_active_feature_names

        feature_cols = get_active_feature_names()

        # Import walk-forward validation function
        from src.training.walk_forward import unified_walk_forward

        # Create submodel params dict with just this model
        submodel_params_dict = {model_type: params}

        # Create ensemble weights with just this model having weight 1.0
        ensemble_weights = {
            mt: 1.0 if mt == model_type else 0.0 for mt in ACTIVE_MODEL_TYPES
        }

        # Perform walk-forward validation
        _, metrics = unified_walk_forward(
            df=df,
            feature_cols=feature_cols,
            submodel_params_dict=submodel_params_dict,
            ensemble_weights=ensemble_weights,
            trial=trial,
            update_dashboard=False,
        )

        return metrics
    except Exception as e:
        logger.error(f"Error in walk-forward validation: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"rmse": float("inf"), "mape": float("inf")}


def suggest_model_hyperparameters(trial, model_type):
    """
    Suggest hyperparameters for a specific model type.

    Args:
        trial: Optuna trial object
        model_type: Model type to suggest hyperparameters for

    Returns:
        Dict with suggested hyperparameters
    """
    # Import OptunaSuggester but handle circular imports carefully
    try:
        # First try direct import
        from src.tuning.meta_tuning import OptunaSuggester

        suggester = OptunaSuggester(trial)
        return suggester.suggest_model_params(model_type)
    except ImportError:
        # Fallback to inline implementation if circular import happens
        if model_type in ["lstm", "rnn"]:
            return {
                "lr": trial.suggest_float(f"{model_type}_lr", 1e-5, 1e-2, log=True),
                "dropout": trial.suggest_float(f"{model_type}_dropout", 0.0, 0.5),
                "lookback": trial.suggest_int(f"{model_type}_lookback", 7, 90),
                "units_per_layer": [
                    trial.suggest_categorical(
                        f"{model_type}_units", [32, 64, 128, 256, 512]
                    )
                ],
                "loss_function": trial.suggest_categorical(
                    f"{model_type}_loss",
                    ["mean_squared_error", "mean_absolute_error", "huber_loss"],
                ),
                "epochs": trial.suggest_int(f"{model_type}_epochs", 5, 50),
                "batch_size": trial.suggest_categorical(
                    f"{model_type}_batch_size", [16, 32, 64, 128]
                ),
            }
        elif model_type == "random_forest":
            return {
                "n_est": trial.suggest_int("rf_n_est", 50, 500),
                "mdepth": trial.suggest_int("rf_mdepth", 3, 25),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
            }
        elif model_type == "xgboost":
            return {
                "n_est": trial.suggest_int("xgb_n_est", 50, 500),
                "lr": trial.suggest_float("xgb_lr", 1e-4, 0.5, log=True),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            }
        else:
            # Default empty params
            return {}


class StudyManager:
    """
    Centralized manager for Optuna studies.
    Ensures consistent study creation and prevents duplicate studies.
    """

    def __init__(self):
        """Initialize the study manager."""
        self.studies = {}
        self.study_configs = {}
        self.db_dir = os.path.join(project_root, "Data", "DB")
        os.makedirs(self.db_dir, exist_ok=True)

    def create_study(
        self,
        model_type,
        ticker,
        timeframe,
        range_cat,
        cycle=1,
        n_trials=None,
        metric_weights=None,
    ):
        """
        Create (or get existing) study for a specific model.

        Args:
            model_type: Type of model ('lstm', 'rnn', etc.)
            ticker: Ticker symbol
            timeframe: Timeframe
            range_cat: Range category
            cycle: Tuning cycle number
            n_trials: Number of trials
            metric_weights: Optional dict of metric weights

        Returns:
            Optuna study
        """
        # Create a unique study name
        study_name = f"{ticker}_{timeframe}_{range_cat}_{model_type}_cycle{cycle}"

        # Check if study already exists
        if study_name in self.studies:
            logger.info(f"Using existing study: {study_name}")
            return self.studies[study_name]

        # Set up storage
        storage_name = os.path.join(self.db_dir, f"{study_name}.db")
        storage_url = f"sqlite:///{storage_name}"

        # Create study
        try:
            # Load existing study if it exists in the database
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            logger.info(f"Loaded existing study from database: {study_name}")
        except (optuna.exceptions.StorageInvalidURLError, KeyError):
            # Create a new study if it doesn't exist
            try:
                # Use the FileLock from your threadsafe module
                with FileLock(storage_name):
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=storage_url,
                        direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=42),
                        load_if_exists=True,
                    )
                    logger.info(f"Created new study: {study_name}")
            except Exception as e:
                logger.error(f"Error creating study: {str(e)}")
                # Fallback: in-memory storage
                study = optuna.create_study(direction="minimize")
                logger.warning(f"Falling back to in-memory study due to error: {e}")

        # Store study in dict
        self.studies[study_name] = study

        # Store config
        self.study_configs[study_name] = {
            "model_type": model_type,
            "ticker": ticker,
            "timeframe": timeframe,
            "range_cat": range_cat,
            "cycle": cycle,
            "n_trials": n_trials,
            "metric_weights": metric_weights,
        }

        return study

    def run_all_studies(self, studies_config):
        """
        Run all studies with proper resource allocation.

        Args:
            studies_config: List of study configurations

        Returns:
            Dict with results for all model types
        """
        from src.utils.memory_utils import adaptive_memory_clean

        results = {}

        # Run each study sequentially with proper cleanup in between
        for i, study_config in enumerate(studies_config):
            model_type = study_config["component_type"]
            n_trials = study_config["n_trials"]
            objective_func = study_config["objective_func"]
            callbacks = study_config.get("callbacks", [])

            logger.info(f"Running study for {model_type} ({i+1}/{len(studies_config)})")

            # Get study for this model type
            study_name = list(
                filter(
                    lambda x: self.study_configs[x]["model_type"] == model_type,
                    self.studies.keys(),
                )
            )

            if not study_name:
                logger.warning(f"No study found for {model_type}, skipping")
                continue

            study_name = study_name[0]
            study = self.studies[study_name]

            # Run optimization
            try:
                study.optimize(objective_func, n_trials=n_trials, callbacks=callbacks)

                # Store results
                results[model_type] = {
                    "study": study,
                    "best_params": study.best_params,
                    "best_value": study.best_value,
                    "best_trial": study.best_trial,
                }
            except Exception as e:
                logger.error(f"Error in study for {model_type}: {e}")
                results[model_type] = {"error": str(e)}

            # Clean up memory
            adaptive_memory_clean("medium")

        return results

    def get_best_model(self):
        """
        Get the best model across all studies.

        Returns:
            Tuple of (model_type, best_params, best_value)
        """
        best_value = float("inf")
        best_model_type = None
        best_params = None

        for study_name, study in self.studies.items():
            if study.trials:
                if study.best_value < best_value:
                    best_value = study.best_value
                    best_model_type = self.study_configs[study_name]["model_type"]
                    best_params = study.best_params

        return best_model_type, best_params, best_value

    def get_best_params_for_model(self, model_type):
        """
        Get the best parameters for a specific model.

        Args:
            model_type: Model type to get best parameters for

        Returns:
            Dict with best parameters or None if not found
        """
        for study_name, study in self.studies.items():
            if (
                self.study_configs[study_name]["model_type"] == model_type
                and study.trials
            ):
                return study.best_params

        return None


# Create a singleton instance
study_manager = StudyManager()
