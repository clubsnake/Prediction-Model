"""
Provides standardized logging setup for all modules in the prediction model project.
"""

import os
import logging
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: Optional[str] = None,
    log_file_prefix: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Creates a configured logger that outputs to both console and file.
    
    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (default: INFO)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory to store log files (default: 'logs')
        log_file_prefix: Prefix for log filenames (default: logger name)
        log_format: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Use default format if none specified
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            # Default to 'logs' directory in project root
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..'
            ))
            log_dir = os.path.join(project_root, "logs")
            
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = log_file_prefix or name.replace(".", "_")
        log_file_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
        
        # Add file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log the file location
        logger.info(f"Logging to file: {log_file_path}")
    
    return logger

# Example usage in other modules:
# from Scripts.logger_config import setup_logger
# logger = setup_logger(__name__)
