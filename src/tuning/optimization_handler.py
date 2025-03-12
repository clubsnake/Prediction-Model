"""
Handles optimization and hyperparameter tuning for models.
"""

import os
import sys

# Add project root to path to fix import issues
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Now we can import from Scripts
from src.utils.env_setup import setup_tf_environment

# Configure environment before other imports
setup_tf_environment()

import json

# Rest of imports
import logging
from datetime import datetime

import numpy as np
import optuna
import pandas as pd

from src.training.walk_forward import run_walk_forward_ensemble_eval
from src.tuning.optuna_logger import OptunaLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("optimization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class OptimizationHandler:
    """
    Handles optimization of model hyperparameters using Optuna.
    """

    def __init__(self, df, feature_cols, target_col, config=None):
        """
        Initialize the optimization handler.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            config: Configuration dictionary
        """
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.config = config or {}

        # Default configuration
        self.n_trials = self.config.get("n_trials", 100)
        self.timeout = self.config.get("timeout", 3600)  # 1 hour default
        self.study_name = self.config.get(
            "study_name", f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Storage settings - Update to use the Models directory
        from config.config_loader import DATA_DIR

        self.db_dir = self.config.get("db_dir", os.path.join(DATA_DIR, "Models", "DB"))
        os.makedirs(self.db_dir, exist_ok=True)

        # Set up Optuna logger
        self.optuna_logger = OptunaLogger(self.study_name)

        # Report setup
        logger.info("Optimization handler initialized:")
        logger.info(f"  - Study name: {self.study_name}")
        logger.info(f"  - Trials: {self.n_trials}")
        logger.info(f"  - Timeout: {self.timeout}s")
        logger.info(f"  - Features: {len(self.feature_cols)} columns")
        logger.info(f"  - Database directory: {self.db_dir}")

    def objective(self, trial):
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            performance metric (lower is better)
        """
        # Record start time for this trial
        start_time = datetime.now()

        try:
            # Add missing start_time initialization
            start_time = datetime.now()

            # Log trial start
            logger.info(f"Starting trial {trial.number}")

            # Sample hyperparameters for LSTM
            lstm_params = {
                "units_per_layer": [trial.suggest_int("lstm_units", 32, 256)],
                "lr": trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True),
                "dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
                "loss_function": trial.suggest_categorical("lstm_loss", ["mae", "mse"]),
                "batch_size": trial.suggest_int("lstm_batch_size", 16, 128),
                "epochs": trial.suggest_int("lstm_epochs", 1, 10),
            }

            # Sample hyperparameters for RandomForest
            rf_params = {
                "n_est": trial.suggest_int("rf_n_est", 50, 500),
                "mdepth": trial.suggest_int("rf_mdepth", 5, 30),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            }

            # Sample hyperparameters for XGBoost
            xgb_params = {
                "n_est": trial.suggest_int("xgb_n_est", 50, 500),
                "lr": trial.suggest_float("xgb_lr", 0.01, 0.3),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            }

            # Sample ensemble weights
            w_lstm = trial.suggest_float("w_lstm", 0.0, 1.0)
            w_rf = trial.suggest_float("w_rf", 0.0, 1.0)
            w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)

            # Normalize weights
            sum_weights = w_lstm + w_rf + w_xgb
            if sum_weights > 0:
                w_lstm /= sum_weights
                w_rf /= sum_weights
                w_xgb /= sum_weights
            else:
                # Fallback to equal weights if all are zero
                w_lstm = w_rf = w_xgb = 1 / 3

            # Create parameter dictionary
            submodel_params_dict = {
                "lstm": lstm_params,
                "random_forest": rf_params,
                "xgboost": xgb_params,
            }

            ensemble_weights = {"lstm": w_lstm, "random_forest": w_rf, "xgboost": w_xgb}

            # Log hyperparameters
            logger.info(f"Trial {trial.number} parameters:")
            logger.info(f"  LSTM: {json.dumps(lstm_params)}")
            logger.info(f"  Random Forest: {json.dumps(rf_params)}")
            logger.info(f"  XGBoost: {json.dumps(xgb_params)}")
            logger.info(f"  Weights: {json.dumps(ensemble_weights)}")

            # Run walk-forward validation
            horizon = self.config.get("horizon", 7)
            wf_size = self.config.get("wf_size", 30)

            mse_val, mape_val = run_walk_forward_ensemble_eval(
                df=self.df,
                feature_cols=self.feature_cols,
                horizon=horizon,
                wf_size=wf_size,
                submodel_params_dict=submodel_params_dict,
                ensemble_weights=ensemble_weights,
                trial=trial,  # Pass trial for pruning
            )

            # Calculate final metric (RMSE)
            final_metric = np.sqrt(mse_val)

            # Calculate trial duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            trial.set_user_attr("duration", duration)

            # Log results
            logger.info(f"Trial {trial.number} completed:")
            logger.info(f"  RMSE: {final_metric:.6f}")
            logger.info(f"  MAPE: {mape_val:.6f}")
            logger.info(f"  Duration: {duration:.2f}s")

            return final_metric

        except optuna.exceptions.TrialPruned as e:
            logger.info(f"Trial {trial.number} pruned")
            raise e
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}", exc_info=True)
            raise e

    def run_optimization(self):
        """
        Run the optimization process.

        Returns:
            Optuna study object with results
        """
        # Create storage
        storage_path = os.path.join(self.db_dir, f"{self.study_name}.db")
        storage = f"sqlite:///{storage_path}"

        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=1
            ),
        )

        logger.info(f"Starting optimization for study '{self.study_name}'")
        logger.info(f"Storage: {storage}")

        try:
            # Run optimization
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[self.optuna_logger.log_study_callback],
                n_jobs=1,  # Run sequentially to avoid TF conflicts
            )

            # Log study summary
            self.optuna_logger.log_study_summary(study)

            return study

        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            self.optuna_logger.log_study_summary(study)
            return study
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            self.optuna_logger.log_study_summary(study)
            raise e


def main(config_file=None):
    """
    Main optimization handler.

    Args:
        config_file: Path to configuration file (optional)
    """
    logger.info("Starting optimization handler")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Project root: {project_root}")

    # Actual optimization code would go here
    logger.info("Optimization complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()

    main(args.config)

# Example usage
if __name__ == "__main__":
    # Load data
    sample_data = pd.read_csv("Data/sample_data.csv")
    feature_cols = [col for col in sample_data.columns if col.startswith("feature_")]
    target_col = "target"

    # Configuration
    config = {
        "n_trials": 50,
        "timeout": 3600,  # 1 hour
        "study_name": "ensemble_optimization",
        "horizon": 7,
        "wf_size": 30,
    }

    # Run optimization
    optimizer = OptimizationHandler(sample_data, feature_cols, target_col, config)
    study = optimizer.run_optimization()

    # Print best parameters
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
