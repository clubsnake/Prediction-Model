# Scripts/detect_parameters.py

import os
import sys

# Add project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.tuning.hyperparameter_utils import suggest_parameter_registration

if __name__ == "__main__":
    print("Scanning codebase for parameters...")
    suggest_parameter_registration()
