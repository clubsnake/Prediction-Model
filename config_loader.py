# WARNING: This appears to be a duplicate of config/config_loader.py
# Consider removing this file or renaming it to avoid confusion
# Renamed to: legacy_config_loader.py

import json
import yaml
import os

def load_config(system_config_path, user_config_path):
    """
    Load configurations from system and user config files.

    Args:
        system_config_path (str): Path to the system configuration file (JSON).
        user_config_path (str): Path to the user configuration file (YAML).

    Returns:
        tuple: A tuple containing the system and user configurations as dictionaries.
    """
    try:
        with open(system_config_path, 'r') as f:
            system_config = json.load(f)
    except FileNotFoundError:
        system_config = {}
        print(f"Warning: System config file not found at {system_config_path}. Using default settings.")
    except json.JSONDecodeError as e:
        system_config = {}
        print(f"Warning: System config file contains invalid JSON at {system_config_path}. Using default settings. Error: {e}")

    try:
        with open(user_config_path, 'r') as f:
            user_config = yaml.safe_load(f)
    except FileNotFoundError:
        user_config = {}
        print(f"Warning: User config file not found at {user_config_path}. Using default settings.")
    except yaml.YAMLError as e:
        user_config = {}
        print(f"Warning: User config file contains invalid YAML at {user_config_path}. Using default settings. Error: {e}")

    return system_config, user_config

def setup_directories(system_config, user_config):
    """
    Set up the necessary directories based on the configurations.

    Args:
        system_config (dict): System configuration dictionary.
        user_config (dict): User configuration dictionary.

    Returns:
        dict: A dictionary containing the paths to the set up directories.
    """
    base_dir = user_config.get('base_dir', system_config.get('base_dir', '.'))

    DATA_DIR = os.path.join(base_dir, system_config.get('data_dir', 'data'))
    LOGS_DIR = os.path.join(base_dir, system_config.get('logs_dir', 'logs'))
    MODELS_DIR = os.path.join(base_dir, system_config.get('models_dir', 'models'))
    REPORTS_DIR = os.path.join(base_dir, system_config.get('reports_dir', 'reports'))
    
    # Ensure all directories exist
    for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(directory, exist_ok=True)

    return {
        'DATA_DIR': DATA_DIR,
        'LOGS_DIR': LOGS_DIR,
        'MODELS_DIR': MODELS_DIR,
        'REPORTS_DIR': REPORTS_DIR
    }

# Load configurations
SYSTEM_CONFIG_PATH = 'system_config.json'
USER_CONFIG_PATH = 'user_config.yaml'
system_config, user_config = load_config(SYSTEM_CONFIG_PATH, USER_CONFIG_PATH)

# Setup directories
directories = setup_directories(system_config, user_config)

# Expose directory paths
DATA_DIR = directories['DATA_DIR']
LOGS_DIR = directories['LOGS_DIR']
MODELS_DIR = directories['MODELS_DIR']
REPORTS_DIR = directories['REPORTS_DIR']
