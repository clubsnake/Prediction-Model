# Scripts/hyperparameter_utils.py

import ast
import logging
import os


def scan_codebase_for_parameters():
    """Scan Python files for potential parameters to tune."""
    parameters = []

    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                try:
                    with open(os.path.join(root, file), "r") as f:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            # Look for assignments with numeric/boolean literals
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if isinstance(target, ast.Name):
                                        if isinstance(
                                            node.value, ast.Num
                                        ) or isinstance(node.value, ast.NameConstant):
                                            param_name = target.id
                                            if (
                                                param_name.isupper()
                                            ):  # Likely a constant
                                                parameters.append(
                                                    {
                                                        "file": file,
                                                        "name": param_name,
                                                        "value": ast.literal_eval(
                                                            node.value
                                                        ),
                                                    }
                                                )
                except Exception as e:
                    logging.warning(f"Error parsing {file}: {e}")

    return parameters


def suggest_parameter_registration():
    """Print suggested parameter registrations."""
    params = scan_codebase_for_parameters()

    print("# Add these to config.py\n")
    for param in params:
        value = param["value"]
        param_type = type(value).__name__

        if param_type == "int":
            range_values = [max(0, value // 2), value * 2]
        elif param_type == "float":
            range_values = [max(0.0, value / 2), value * 2]
        elif param_type == "bool":
            range_values = [True, False]
        else:
            range_values = None

        print(
            f"{param['name']} = register_param(\"{param['name'].lower()}\", {value}, \"{param_type}\", {range_values})"
        )


class HyperparameterManager:
    """Manage all hyperparameters across the system."""

    def __init__(self, registry=None):
        """Initialize with optional registry."""
        from config import HYPERPARAMETER_REGISTRY

        self.registry = registry or HYPERPARAMETER_REGISTRY
        self.current_values = {}

    def suggest_all(self, trial):
        """Suggest all registered hyperparameters."""
        for name, config in self.registry.items():
            param_type = config["type"]
            default = config["default"]
            range_values = config["range"]

            if param_type == "int":
                if range_values:
                    self.current_values[name] = trial.suggest_int(
                        name, range_values[0], range_values[1]
                    )
                else:
                    # Default range based on value
                    self.current_values[name] = trial.suggest_int(
                        name, max(1, default // 2), default * 2
                    )

            elif param_type == "float":
                if range_values:
                    self.current_values[name] = trial.suggest_float(
                        name, range_values[0], range_values[1]
                    )
                else:
                    # Default range with log scale for small values
                    if default < 0.01:
                        self.current_values[name] = trial.suggest_float(
                            name, default / 10, default * 10, log=True
                        )
                    else:
                        self.current_values[name] = trial.suggest_float(
                            name, max(0.0001, default / 2), default * 2
                        )

            elif param_type == "categorical":
                if range_values:
                    self.current_values[name] = trial.suggest_categorical(
                        name, range_values
                    )
                else:
                    # Default to the original value if no range specified
                    self.current_values[name] = default

        return self.current_values

    def get(self, name):
        """Get current value for a parameter."""
        return self.current_values.get(name, self.registry.get(name, {}).get("default"))


def set_all_random_states(seed):
    """Set random state for all libraries."""
    import random

    import numpy as np
    import tensorflow as tf

    # Python's random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    # Return the seed for tracking
    return seed
