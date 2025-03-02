# tuning_launcher.py
"""
Launch the appropriate hyperparameter tuning approach based on config settings.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import HYPERPARAM_SEARCH_METHOD

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
        raise ValueError("Invalid HYPERPARAM_SEARCH_METHOD in config")

if __name__ == "__main__":
    main()
