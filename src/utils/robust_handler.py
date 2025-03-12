"""
Provides robust error handling and graceful shutdown capabilities for the prediction model.
"""

import atexit
import functools
import logging
import signal
import sys
import threading
import traceback
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

    @staticmethod
    def handle_signal(sig_num, frame):
        """Handle termination signals by initiating shutdown."""
        signal_name = (
            signal.Signals(sig_num).name
            if hasattr(signal, "Signals")
            else f"signal {sig_num}"
        )
        logger.info(f"Received {signal_name}, initiating graceful shutdown")

        # Call the central shutdown function
        initiate_shutdown(source=f"Signal: {signal_name}")

        # Exit with a specific code for signal handling
        sys.exit(128 + sig_num)

    @staticmethod
    def cleanup():
        """Perform cleanup operations during shutdown."""
        logger.info("Performing final cleanup on application exit")

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

    # First, stop any tuning processes
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
