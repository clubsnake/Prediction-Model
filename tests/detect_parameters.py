# Scripts/detect_parameters.py

from hyperparameter_utils import suggest_parameter_registration

if __name__ == "__main__":
    print("Scanning codebase for parameters...")
    suggest_parameter_registration()
