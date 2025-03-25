"""
Provides standardized logging setup for all modules in the prediction model project.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
import glob
import time
from pathlib import Path

# Add project root to the path to fix imports if needed
current_file = os.path.abspath(__file__)
config_dir = os.path.dirname(current_file)
project_root = os.path.dirname(config_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import paths from config_loader first to ensure consistency
try:
    from config.config_loader import DATA_DIR, LOGS_DIR
except ImportError:
    # Fallback to direct path definitions if import fails
    DATA_DIR = os.path.join(project_root, "data")  # Use lowercase for consistency
    LOGS_DIR = os.path.join(DATA_DIR, "logs")  # Use lowercase for consistency

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
        log_file = os.path.join(
            LOGS_DIR, f"prediction_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
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
            "%(asctime)s - %(name)s - %(levellevel)s - %(message)s",
        )
    except ImportError:
        log_format = "%(asctime)s - %(name)s - %(levellevel)s - %(message)s"

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
        logger.info("Logging to file: %s", log_file_path)
    
        # Add log rotation capability
        try:
            from config.config_loader import get_value
            max_log_files = get_value("logger.max_log_files", 10)
            max_log_age_days = get_value("logger.max_log_age_days", 7)
            
            # Clean up old log files
            log_pattern = os.path.join(log_dir, f"{prefix}_*.log")
            log_files = glob.glob(log_pattern)
            
            # Sort by modification time (oldest first)
            log_files.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove old logs if we have too many
            if len(log_files) > max_log_files:
                files_to_remove = log_files[:-max_log_files]
                for old_file in files_to_remove:
                    try:
                        os.remove(old_file)
                        logger.info("Cleaned up old log file: %s", old_file)
                    except Exception as e:
                        logger.warning("Could not remove old log file %s: %s", old_file, e)
            
            logger.info("Cleaned up %d old log files", len(files_to_remove) if 'files_to_remove' in locals() else 0)
        except Exception as e:
            logger.warning("Error during log rotation: %s", e)

    return logger


def cleanup_old_logs(max_files=10, max_age_days=7):
    """
    Remove old log files based on retention settings.
    
    Args:
        max_files: Maximum number of log files to keep
        max_age_days: Maximum age of log files in days
    
    Returns:
        tuple: (number of files deleted, list of deleted files)
    """
    try:
        # Find all prediction model log files
        log_pattern = os.path.join(LOGS_DIR, "prediction_model_*.log")
        log_files = glob.glob(log_pattern)
        
        if not log_files:
            return 0, []
        
        # Sort by modification time (newest first)
        log_files.sort(key=os.path.getmtime, reverse=True)
        
        # Keep track of deleted files
        deleted_files = []
        
        # Calculate cutoff time for age-based deletion
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        cutoff_time = current_time - max_age_seconds
        
        # Delete old files based on count
        if max_files > 0 and len(log_files) > max_files:
            files_to_delete_by_count = log_files[max_files:]
            for file_path in files_to_delete_by_count:
                try:
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
                except Exception as e:
                    logger.warning(f"Failed to delete old log file {file_path}: {e}")
        
        # Delete files based on age
        for file_path in log_files[:max_files]:  # Only check remaining files
            try:
                mtime = os.path.getmtime(file_path)
                if mtime < cutoff_time:
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
            except Exception as e:
                logger.warning(f"Failed to check/delete log file {file_path}: {e}")
        
        # Log summary if files were deleted
        if deleted_files:
            logger.info(f"Cleaned up {len(deleted_files)} old log files")
            
        return len(deleted_files), deleted_files
    except Exception as e:
        logger.error(f"Error cleaning up log files: {e}")
        return 0, []

# Run cleanup at module import
try:
    # Try to import the config values first
    from config.config_loader import get_value
    
    max_files = get_value("logger.retention.max_files", 10)
    max_age_days = get_value("logger.retention.max_age_days", 7)
    auto_cleanup = get_value("logger.retention.auto_cleanup", True)
    
    if auto_cleanup:
        cleaned_count, _ = cleanup_old_logs(max_files, max_age_days)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old log files on startup")
except Exception as e:
    # Fall back to defaults if config can't be loaded
    logger.warning(f"Using default log retention settings: {e}")
    cleanup_old_logs(10, 7)
