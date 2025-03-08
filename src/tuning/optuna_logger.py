"""
Enhanced logging for Optuna hyperparameter optimization.
"""

import json
import logging
import os
from datetime import datetime

import optuna
import pandas as pd
import plotly.graph_objects as go

# Setup logging
logger = logging.getLogger(__name__)


class OptunaLogger:
    """
    Logger for Optuna trials that captures and stores detailed information
    about hyperparameter optimization runs.
    """
    
    def __init__(self, study_name, log_dir=None):
        """
        Initialize the logger.

        Args:
            study_name: Name of the Optuna study
            log_dir: Directory to store log files (default: 'logs/optuna')
        """
        from config.config_loader import DATA_DIR
        
        self.study_name = study_name
        # Update log directory to use Models/Logs path
        self.log_dir = log_dir or os.path.join(DATA_DIR, "Models", "Logs", "optuna")
        os.makedirs(self.log_dir, exist_ok=True)

        # Create log file paths
        self.log_file = os.path.join(self.log_dir, f"{study_name}_trials.csv")
        self.best_params_file = os.path.join(
            self.log_dir, f"{study_name}_best_params.json"
        )
        self.history_file = os.path.join(self.log_dir, f"{study_name}_history.csv")

        # Initialize trial data
        self.trial_history = []

    def log_trial(self, trial):
        """
        Log a completed trial.

        Args:
            trial: Optuna trial object
        """
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Get basic trial info
        trial_info = {
            "trial_number": trial.number,
            "value": trial.value,
            "datetime": datetime.now().isoformat(),
            "duration": getattr(trial, "duration", None),
        }

        # Add parameters
        for param_name, param_value in trial.params.items():
            trial_info[f"param_{param_name}"] = param_value

        # Add trial info to history
        self.trial_history.append(trial_info)

        # Log trial details
        logger.info(f"Trial {trial.number} finished with value: {trial.value}")
        logger.info(f"Parameters: {json.dumps(trial.params, indent=2)}")

        # Save updated history
        self._save_history()

    def log_study_callback(self, study, trial):
        """
        Callback function for Optuna study.
        Use this with study.optimize(..., callbacks=[logger.log_study_callback])

        Args:
            study: Optuna study object
            trial: Completed trial
        """
        # Log this trial
        self.log_trial(trial)

        # Save best parameters when they change
        if study.best_trial and study.best_trial.number == trial.number:
            self._save_best_params(study)

    def _save_history(self):
        """Save trial history to CSV file"""
        try:
            # Convert to DataFrame and save
            df = pd.DataFrame(self.trial_history)
            df.to_csv(self.history_file, index=False)
            logger.debug(f"Updated trial history at {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving trial history: {str(e)}")

    def _save_best_params(self, study):
        """Save best parameters to JSON file"""
        try:
            # Create dictionary with best parameters and metadata
            best_params = {
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "params": study.best_params,
                "datetime": datetime.now().isoformat(),
                "n_trials": len(study.trials),
            }

            # Save as JSON
            with open(self.best_params_file, "w") as f:
                json.dump(best_params, f, indent=4)

            logger.info(f"Updated best parameters at {self.best_params_file}")
            logger.info(f"Best value: {study.best_value}")
            logger.info(f"Best parameters: {json.dumps(study.best_params, indent=2)}")
        except Exception as e:
            logger.error(f"Error saving best parameters: {str(e)}")

    def create_parameter_importance_plot(self, study):
        """
        Generate parameter importance visualization.

        Args:
            study: Completed Optuna study
        """
        try:
            # Check if study has completed trials
            if not study.trials or len(study.trials) < 5:
                logger.warning("Not enough trials for parameter importance plot")
                return None

            # Get parameter importance
            importances = optuna.importance.get_param_importances(study)

            # Sort parameters by importance
            sorted_importances = sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            )
            param_names = [name for name, _ in sorted_importances]
            importance_values = [value for _, value in sorted_importances]

            # Create bar chart
            fig = go.Figure([go.Bar(x=param_names, y=importance_values)])

            fig.update_layout(
                title="Hyperparameter Importance",
                xaxis_title="Parameter",
                yaxis_title="Importance Score",
                template="plotly_white",
            )

            # Save figure
            fig_path = os.path.join(self.log_dir, f"{self.study_name}_importance.html")
            fig.write_html(fig_path)
            logger.info(f"Parameter importance plot saved to {fig_path}")

            return fig
        except Exception as e:
            logger.error(f"Error creating parameter importance plot: {str(e)}")
            return None

    def log_study_summary(self, study):
        """
        Log a summary of a completed study.

        Args:
            study: Completed Optuna study
        """
        try:
            # Basic study info
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            pruned_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ]
            failed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
            ]

            logger.info("=" * 50)
            logger.info(f"Study '{study.study_name}' summary:")
            logger.info(f"Best value: {study.best_value}")
            logger.info(f"Best parameters: {json.dumps(study.best_params, indent=2)}")
            logger.info(f"Number of completed trials: {len(completed_trials)}")
            logger.info(f"Number of pruned trials: {len(pruned_trials)}")
            logger.info(f"Number of failed trials: {len(failed_trials)}")

            # Save detailed summary
            summary = {
                "study_name": study.study_name,
                "datetime": datetime.now().isoformat(),
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "best_params": study.best_params,
                "completed_trials": len(completed_trials),
                "pruned_trials": len(pruned_trials),
                "failed_trials": len(failed_trials),
                "total_trials": len(study.trials),
            }

            # Save as JSON
            summary_file = os.path.join(self.log_dir, f"{self.study_name}_summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=4)

            logger.info(f"Study summary saved to {summary_file}")
            logger.info("=" * 50)

            # Create parameter importance plot
            self.create_parameter_importance_plot(study)

        except Exception as e:
            logger.error(f"Error logging study summary: {str(e)}")
