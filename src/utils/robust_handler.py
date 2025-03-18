"""
Provides robust error handling and graceful shutdown capabilities for the prediction model.
"""

import atexit
import functools
import logging
import os
import signal
import sys
import threading
import traceback
import time
from contextlib import contextmanager
from typing import Callable

# Configure logger
logger = logging.getLogger(__name__)

# Global state tracking
_shutdown_callbacks = []
_registered_threads = []
_shutdown_initiated = False
_shutdown_lock = threading.Lock()


class GracefulShutdownHandler:
    """Handles graceful shutdown of the application with resource cleanup."""

    @staticmethod
    def register_handler():
        """Register signal handlers for graceful shutdown."""
        # Only register signal handlers in main thread to avoid the ValueError
        if threading.current_thread() is threading.main_thread():
            try:
                # Handle keyboard interrupts (CTRL+C)
                signal.signal(signal.SIGINT, GracefulShutdownHandler.handle_signal)

                # Handle termination signals
                if hasattr(signal, "SIGTERM"):
                    signal.signal(signal.SIGTERM, GracefulShutdownHandler.handle_signal)

                logger.info("Graceful shutdown handler registered")
            except ValueError as e:
                logger.warning(f"Could not register signal handlers: {e}")
        else:
            logger.info("Skipping signal handler registration (not in main thread)")

        # Register with atexit to ensure cleanup on normal exit (works in any thread)
        atexit.register(GracefulShutdownHandler.cleanup)
        
        # Register a separate function specifically to reset tuning status
        # This increases the chance it will run even if other shutdown steps fail
        atexit.register(GracefulShutdownHandler.reset_tuning_status)

    @staticmethod
    def handle_signal(sig_num, frame):
        """Handle termination signals by initiating shutdown."""
        signal_name = (
            signal.Signals(sig_num).name
            if hasattr(signal, "Signals")
            else f"signal {sig_num}"
        )
        logger.info(f"Received {signal_name}, initiating graceful shutdown")

        # Reset tuning status immediately when receiving a signal
        # This is crucial for handling Ctrl+C properly
        GracefulShutdownHandler.reset_tuning_status()

        # Call the central shutdown function
        initiate_shutdown(source=f"Signal: {signal_name}")

        # Exit with a specific code for signal handling
        sys.exit(128 + sig_num)

    @staticmethod
    def reset_tuning_status():
        """
        Reset tuning status file immediately - this function is designed to run
        early during shutdown to ensure tuning status is properly reset even if
        the process is killed abruptly.
        """
        try:
            # Try multiple approaches to update the tuning status file
            
            # Approach 1: Direct file write (most reliable in emergency shutdown)
            try:
                # Find the data directory
                current_file = os.path.abspath(__file__)
                src_dir = os.path.dirname(os.path.dirname(current_file))
                project_root = os.path.dirname(src_dir)
                data_dir = os.path.join(project_root, "data")
                
                # Write directly to the status file
                status_file = os.path.join(data_dir, "tuning_status.txt")
                if os.path.exists(status_file):
                    with open(status_file, 'w') as f:
                        f.write("status: stopped\nis_running: false\nstopped_manually: true\n")
                    logger.info(f"Reset tuning status file directly: {status_file}")
            except Exception as e:
                logger.warning(f"Could not directly update tuning status file: {e}")
            
            # Approach 2: Use the progress_helper module
            try:
                from src.tuning.progress_helper import write_tuning_status
                write_tuning_status({
                    "status": "stopped",
                    "is_running": False,
                    "stopped_manually": True,
                    "stop_time": time.time()
                })
                logger.info("Reset tuning status using progress_helper")
            except Exception as e:
                logger.warning(f"Could not update tuning status via progress_helper: {e}")
                
        except Exception as e:
            # Last resort logging
            logger.error(f"Failed to reset tuning status during shutdown: {e}")

    @staticmethod
    def cleanup():
        """Perform cleanup operations during shutdown."""
        logger.info("Performing final cleanup on application exit")

        # Reset tuning status as the first step of cleanup
        GracefulShutdownHandler.reset_tuning_status()

        # Call all registered shutdown callbacks
        for callback in _shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")


def register_shutdown_callback(callback: Callable) -> None:
    """Register a callback to be executed during shutdown."""
    if callable(callback) and callback not in _shutdown_callbacks:
        _shutdown_callbacks.append(callback)
        logger.debug(f"Registered shutdown callback: {callback.__name__}")


def register_thread(thread: threading.Thread) -> None:
    """Register a thread to be monitored during shutdown."""
    _registered_threads.append(thread)


def initiate_shutdown(source: str = "unknown") -> None:
    """
    Initiate a graceful shutdown of the application.

    Args:
        source: Description of what triggered the shutdown
    """
    global _shutdown_initiated

    # Use a lock to prevent multiple simultaneous shutdown attempts
    with _shutdown_lock:
        if _shutdown_initiated:
            logger.info(f"Shutdown already in progress, ignoring request from {source}")
            return

        logger.info(f"Initiating graceful shutdown from {source}")
        _shutdown_initiated = True

    # First, reset tuning status
    GracefulShutdownHandler.reset_tuning_status()

    # Then stop any tuning processes
    try:
        # Import here to avoid circular imports
        from src.tuning.meta_tuning import stop_tuning_process

        logger.info("Stopping any running tuning processes")
        stop_tuning_process()
    except ImportError:
        logger.warning("Could not import tuning module")
    except Exception as e:
        logger.error(f"Error stopping tuning: {e}")

    # Call all registered shutdown callbacks
    for callback in _shutdown_callbacks:
        try:
            callback()
        except Exception as e:
            logger.error(f"Error in shutdown callback: {e}")

    # Wait for registered threads to finish (with timeout)
    for thread in _registered_threads:
        if thread.is_alive():
            try:
                logger.info(f"Waiting for thread {thread.name} to finish")
                thread.join(timeout=3.0)  # 3 second timeout per thread
            except Exception as e:
                logger.error(f"Error waiting for thread {thread.name}: {e}")


def is_shutdown_in_progress() -> bool:
    """Check if shutdown is in progress."""
    return _shutdown_initiated


# Initialize the handler
GracefulShutdownHandler.register_handler()


# Define error boundary decorator
def error_boundary(func):
    """
    Decorator for functions to catch and log errors without crashing.

    Args:
        func: The function to decorate

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            )
            # Return None for non-critical errors
            return None

    return wrapper


@contextmanager
def ErrorLogContext(component_name: str, context_data=None):
    """
    Context manager for logging errors with additional context.

    Args:
        component_name: Name of the component for logging
        context_data: Additional data to log with errors
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {component_name}: {str(e)}")
        if context_data:
            logger.error(f"Context: {context_data}")
        logger.error(traceback.format_exc())
        # Re-raise the exception for critical errors
        raise
