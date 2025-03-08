import logging
import os
from config_loader import LOGS_DIR

# Define the log file path
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'app.log')

# Configure logging
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    """
    Logs an info message.

    Args:
        message (str): The message to log.
    """
    logging.info(message)

def log_error(message):
    """
    Logs an error message.

    Args:
        message (str): The message to log.
    """
    logging.error(message)
