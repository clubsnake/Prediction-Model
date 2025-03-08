"""
Configures system resources and provides an interface to GPU memory management.
Acts as a thin wrapper around the more comprehensive gpu_memory_management module.
"""

import logging
import os
import json
import sys

# Add the project root to the path to ensure we can import from config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# System resources configuration
SYSTEM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'system_config.json')

def load_system_config():
    """Load system configuration from JSON file"""
    try:
        if os.path.exists(SYSTEM_CONFIG_PATH):
            with open(SYSTEM_CONFIG_PATH, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"System config file not found: {SYSTEM_CONFIG_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading system config: {e}")
        return {}

def configure_gpu_resources():
    """Configure GPU resources using the centralized gpu_memory_management module"""
    # Load settings from system config
    system_config = load_system_config()
    gpu_config = system_config.get('gpu', {})
    
    # Get mixed precision setting from both configs (system and user)
    try:
        from config.config_loader import get_value
        use_mixed_precision = get_value("hardware.use_mixed_precision", False)
        logger.info(f"Using mixed_precision setting from config: {use_mixed_precision}")
    except ImportError:
        # If we can't import config_loader, use the setting from system_config
        use_mixed_precision = gpu_config.get("use_mixed_precision", False)
        logger.info(f"Using mixed_precision setting from system config: {use_mixed_precision}")
    
    # Set environment variable based on the mixed precision setting
    if not use_mixed_precision:
        os.environ["TF_FORCE_FLOAT32"] = "1"
        logger.info("Setting TF_FORCE_FLOAT32=1 to disable mixed precision")
    
    # Import centralized GPU memory management
    try:
        from src.utils.gpu_memory_management import configure_gpu_memory, configure_mixed_precision
        
        # Configuration for GPU memory
        memory_config = {
            "allow_growth": gpu_config.get("allow_growth", True),
            "memory_limit_mb": gpu_config.get("memory_limit_mb", None),
            "visible_gpus": gpu_config.get("visible_gpus", None),
            "mixed_precision": use_mixed_precision
        }
        
        # Configure GPU memory
        result = configure_gpu_memory(memory_config)
        
        # Explicitly configure mixed precision
        configure_mixed_precision(use_mixed_precision)
        
        return memory_config
    except ImportError as e:
        logger.warning(f"GPU memory management utilities not found: {e}")
        return gpu_config

# Initialize GPU configuration at import time
GPU_CONFIG = configure_gpu_resources()

# Export configuration for other modules
GPU_MEMORY_LIMIT = GPU_CONFIG.get("memory_limit_mb")
USE_MIXED_PRECISION = GPU_CONFIG.get("mixed_precision", False)
