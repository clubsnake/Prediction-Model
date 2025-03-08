"""
Provides robust error handling and graceful shutdown capabilities for the prediction model.
"""

import atexit
import functools
import logging
import signal
import threading
import traceback
from contextlib import contextmanager
from typing import Any, Callable

# Set up logger
logger = logging.getLogger(__name__)
_shutdown_callbacks = []
_registered_threads = []
_shutdown_initiated = False
_shutdown_lock = threading.Lock()


class GracefulShutdownHandler:
    """
    Handles graceful shutdown of the application.
    Manages stopping threads and processes in an orderly fashion.
    """

    def __init__(self):
        """Initialize the shutdown handler."""
        self.stop_event = threading.Event()
        self.resources = {}
        self.shutdown_initiated = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Register atexit
        atexit.register(self._handle_atexit)

    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name}")
        self.initiate_shutdown(f"Signal {sig_name}")

    def _handle_atexit(self):
        """Handle atexit event."""
        if not self.shutdown_initiated:
            self.initiate_shutdown("atexit")

    def register_resource(self, name: str, resource: Any, cleanup_func: Callable):
        """
        Register a resource that needs to be cleaned up during shutdown.

        Args:
            name: Identifier for the resource
            resource: The resource object
            cleanup_func: Function to call to clean up the resource
        """
        self.resources[name] = {"resource": resource, "cleanup": cleanup_func}
        logger.debug(f"Registered resource {name} for shutdown")

    def register_thread(self, thread: threading.Thread):
        """
        Register a thread that should be joined during shutdown.

        Args:
            thread: Thread to join during shutdown
        """
        register_thread(thread)  # Use the global registration

    def initiate_shutdown(self, source: str = "unknown"):
        """
        Initiate graceful shutdown process.

        Args:
            source: Source of the shutdown request
        """
        if self.shutdown_initiated:
            logger.info(f"Shutdown already initiated. Ignoring request from {source}")
            return

        self.shutdown_initiated = True
        self.stop_event.set()

        # Call the global shutdown function
        initiate_shutdown(source)


def register_shutdown_callback(callback: Callable) -> None:
    """
    Register a function to be called during shutdown.

    Args:
        callback: Function to call during shutdown
    """
    global _shutdown_callbacks
    _shutdown_callbacks.append(callback)
    logger.debug(f"Registered shutdown callback: {callback.__name__}")


def register_thread(thread: threading.Thread) -> None:
    """
    Register a thread that should be joined during shutdown.

    Args:
        thread: Thread to join during shutdown
    """
    global _registered_threads
    _registered_threads.append(thread)
    logger.debug(f"Registered thread for shutdown: {thread.name}")


def initiate_shutdown(source: str = "unknown") -> None:
    """
    Initiate graceful shutdown process.

    Args:
        source: Source of the shutdown request
    """
    global _shutdown_initiated, _shutdown_lock

    with _shutdown_lock:
        if _shutdown_initiated:
            logger.info(f"Shutdown already initiated. Ignoring request from {source}")
            return
        _shutdown_initiated = True

    logger.info(f"Initiating graceful shutdown (source: {source})")

    # Call all registered shutdown callbacks
    for callback in _shutdown_callbacks:
        try:
            logger.info(f"Running shutdown callback: {callback.__name__}")
            callback()
        except Exception as e:
            logger.error(f"Error in shutdown callback {callback.__name__}: {str(e)}")

    # Join all registered threads
    for thread in _registered_threads:
        try:
            if thread.is_alive():
                logger.info(f"Waiting for thread to finish: {thread.name}")
                thread.join(timeout=5.0)
        except Exception as e:
            logger.error(f"Error joining thread {thread.name}: {str(e)}")

    logger.info("Graceful shutdown completed")


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
