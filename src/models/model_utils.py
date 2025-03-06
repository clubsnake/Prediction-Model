# Example: Function to load model configuration
import json


def load_model_config(config_path):
    """Loads model configuration from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
