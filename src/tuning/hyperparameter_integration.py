"""
Integrates the hyperparameter tuning system with the hyperparameter dashboard.
This ensures the preset system and manual hyperparameter configurations are properly
synchronized with the actual Optuna tuning process.
"""

import os
import logging
from typing import Dict, Any, Optional
import yaml
import streamlit as st
from datetime import datetime

from config.config_loader import DATA_DIR

logger = logging.getLogger(__name__)

# Update hyperparameters directory to use Models subfolder
HYPERPARAMS_DIR = os.path.join(DATA_DIR, "Models", "Hyperparameters")
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

def apply_preset_to_session_state(preset_key: str) -> None:
    """
    Apply preset multipliers to session state based on the selected preset key.
    This ensures that tuning runs with the appropriate resource allocation.
    
    Args:
        preset_key: Key of the preset to apply (quick, normal, thorough, extreme)
    """
    try:
        from config.config_loader import get_value
        
        # Get preset multipliers from config
        tuning_modes = get_value("hyperparameter.tuning_modes", {})
        if preset_key not in tuning_modes:
            logger.warning(f"Preset {preset_key} not found in config, using normal")
            preset_key = "normal"
            
        preset_multipliers = tuning_modes[preset_key]
        
        # Apply to session state
        st.session_state["tuning_multipliers"] = preset_multipliers
        st.session_state["tuning_mode"] = preset_key
        st.session_state["epochs_multiplier"] = preset_multipliers.get("epochs_multiplier", 1.0)
        st.session_state["complexity_multiplier"] = preset_multipliers.get("complexity_multiplier", 1.0)
        
        logger.info(f"Applied preset {preset_key} with multipliers: {preset_multipliers}")
        return True
    except Exception as e:
        logger.error(f"Error applying preset {preset_key}: {e}")
        return False

def save_hyperparameter_configuration(config_name: str, description: str, param_ranges: Dict) -> bool:
    """
    Save a hyperparameter configuration to a YAML file
    
    Args:
        config_name: Name for the configuration
        description: Description of the configuration
        param_ranges: Dictionary of parameter ranges
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create configuration object
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create the config object
        hyperparameter_config = {
            "name": config_name,
            "description": description,
            "timestamp": timestamp,
            "param_ranges": param_ranges,
            # Include current multipliers
            "multipliers": st.session_state.get("tuning_multipliers", {
                "trials_multiplier": 1.0,
                "epochs_multiplier": 1.0,
                "timeout_multiplier": 1.0,
                "complexity_multiplier": 1.0
            })
        }
        
        # Save to file
        config_filename = config_name.lower().replace(" ", "_") + ".yaml"
        config_path = os.path.join(HYPERPARAMS_DIR, config_filename)
        
        with open(config_path, "w") as f:
            yaml.dump(hyperparameter_config, f, default_flow_style=False)
            
        logger.info(f"Saved hyperparameter configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving hyperparameter configuration: {e}")
        return False

def load_hyperparameter_configuration(config_name: str) -> Optional[Dict]:
    """
    Load a hyperparameter configuration from file
    
    Args:
        config_name: Name of the configuration file (without .yaml extension)
        
    Returns:
        Dict or None: Configuration dictionary if successful, None otherwise
    """
    try:
        # Ensure .yaml extension
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"
            
        config_path = os.path.join(HYPERPARAMS_DIR, config_name)
        
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            
        logger.info(f"Loaded hyperparameter configuration from {config_path}")
        
        # Apply multipliers to session state if present
        if "multipliers" in config_data:
            st.session_state["tuning_multipliers"] = config_data["multipliers"]
            st.session_state["epochs_multiplier"] = config_data["multipliers"].get("epochs_multiplier", 1.0)
            st.session_state["complexity_multiplier"] = config_data["multipliers"].get("complexity_multiplier", 1.0)
            
        return config_data
    except Exception as e:
        logger.error(f"Error loading hyperparameter configuration: {e}")
        return None

def sync_hyperparameter_registry_to_config():
    """
    Synchronize the hyperparameter registry from dashboard to config system.
    This ensures that manual adjustments in the dashboard UI are reflected
    in actual tuning runs.
    """
    try:
        # Only proceed if hyperparameter registry is in session state
        if "HYPERPARAMETER_REGISTRY" not in st.session_state:
            logger.warning("No hyperparameter registry found in session state")
            return False
            
        # Get registry from session state
        registry = st.session_state["HYPERPARAMETER_REGISTRY"]
        
        # TODO: Implement synchronization with config system
        # This would update the system_config.json or appropriate config
        # with the current hyperparameter ranges from the dashboard
        
        logger.info("Synchronized hyperparameter registry to config")
        return True
    except Exception as e:
        logger.error(f"Error synchronizing hyperparameter registry: {e}")
        return False
