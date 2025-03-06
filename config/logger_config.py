"""
Provides standardized logging setup for all modules in the prediction model project.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Add project root to the path to fix imports if needed
current_file = os.path.abspath(__file__)
config_dir = os.path.dirname(current_file)
project_root = os.path.dirname(config_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Directly use paths instead of importing them to avoid circular imports
DATA_DIR = os.path.join(project_root, "Data")  # Capital D to match system_config  # lowercase 'data'
LOGS_DIR = os.path.join(DATA_DIR, "Logs")

# Make sure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up a default logger for direct imports
logger = logging.getLogger("prediction_model")
logger.setLevel(logging.INFO)

# Ensure we have at least one handler to avoid "no handlers found" warning
if not logger.handlers:
    # Add console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Try to add file handler
    try:
        log_file = os.path.join(LOGS_DIR, f"prediction_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}")


def setup_logger(
    name: str,
    level: int = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: Optional[str] = None,
    log_file_prefix: Optional[str] = None,
) -> logging.Logger:
    """
    Creates a configured logger that outputs to both console and file.

    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (default from config or INFO)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory to store log files (default from config)
        log_file_prefix: Prefix for log filenames (default: logger name)

    Returns:
        Configured logger instance
    """
    # Use INFO as default level if not specified
    if level is None:
        try:
            from config.config_loader import get_value
            level_name = get_value("logger.default_level", "INFO")
            level = getattr(logging, level_name)
        except ImportError:
            level = logging.INFO

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Get log format
    try:
        from config.config_loader import get_value
        log_format = get_value(
            "logger.default_format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    except ImportError:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        # Determine log directory
        if log_dir is None:
            log_dir = LOGS_DIR

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = log_file_prefix or name.replace(".", "_")
        log_file_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

        # Add file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Log the file location
        logger.info(f"Logging to file: {log_file_path}")

    return logger