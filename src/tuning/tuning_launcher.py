# tuning_launcher.py
"""
Launch the appropriate hyperparameter tuning approach based on config settings.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import HYPERPARAM_SEARCH_METHOD
from src.utils.training_optimizer import get_training_optimizer

valid_methods = ["optuna", "grid", "both"]
if HYPERPARAM_SEARCH_METHOD not in valid_methods:
    print(
        f"Invalid HYPERPARAM_SEARCH_METHOD: {HYPERPARAM_SEARCH_METHOD}, defaulting to 'optuna'"
    )
    HYPERPARAM_SEARCH_METHOD = "optuna"

# Initialize training optimizer
training_optimizer = get_training_optimizer()

def main():
    """
    Dispatcher to run different tuning approaches depending on config:
    'optuna', 'grid', or 'both'.
    """
    if HYPERPARAM_SEARCH_METHOD == "optuna":
        import meta_tuning

        meta_tuning.main()
    elif HYPERPARAM_SEARCH_METHOD == "grid":
        import hyperparameter_tuning

        hyperparameter_tuning.main_training_loop()
    elif HYPERPARAM_SEARCH_METHOD == "both":
        import meta_tuning

        meta_tuning.main()
        import hyperparameter_tuning

        hyperparameter_tuning.main_training_loop()
    else:
        raise ValueError(
            f"Invalid HYPERPARAM_SEARCH_METHOD: {HYPERPARAM_SEARCH_METHOD}"
        )

def setup_optuna_study(study_name="prediction_model_study", storage=None):
    # Configure resources for parallel tuning
    parallel_jobs = training_optimizer.config.get("num_parallel_models", 1)

if __name__ == "__main__":
    main()
