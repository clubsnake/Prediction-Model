"""
Provides graceful shutdown functionality for the dashboard.
This helps ensure all data is saved and resources are released properly.
"""

import atexit
import logging
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

import streamlit as st

# Import the dashboard_error module for robust error handling
try:
    from src.dashboard.dashboard.dashboard_error import robust_error_boundary, section_error_boundary
except ImportError:
    try:
        from dashboard_error import robust_error_boundary, section_error_boundary
    except ImportError:
        # Define simple error boundary if import fails
        def robust_error_boundary(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {func.__name__}: {e}")
                    return None
            return wrapper
            
        @contextmanager
        def section_error_boundary(section_name):
            try:
                yield
            except Exception as e:
                print(f"Error in section {section_name}: {e}")

# Try to import from src.utils if available
try:
    from src.utils.robust_handler import (
        initiate_shutdown,
        register_shutdown_callback,
        register_thread,
    )
except ImportError:
    # Define fallback implementations if not available
    def initiate_shutdown(source="unknown"):
        print(f"Shutdown initiated from {source}")
        
    def register_shutdown_callback(callback):
        atexit.register(callback)
        
    def register_thread(thread):
        pass

# Set up logger
logger = logging.getLogger(__name__)

# Track active dashboard components
_dashboard_components = {}
_dashboard_threads = []
_is_shutting_down = False
_shutdown_event = threading.Event()


def register_dashboard_component(
    name: str, component: Any, cleanup_func: Optional[Callable] = None
) -> None:
    """
    Register a dashboard component that needs cleanup during shutdown.

    Args:
        name: Name/identifier for the component
        component: The component object
        cleanup_func: Optional function to call for cleanup
    """
    _dashboard_components[name] = {"component": component, "cleanup": cleanup_func}
    logger.info(f"Registered dashboard component: {name}")


def register_dashboard_thread(thread: threading.Thread) -> None:
    """
    Register a dashboard-specific thread for proper shutdown handling.

    Args:
        thread: Thread to register
    """
    _dashboard_threads.append(thread)
    register_thread(thread)  # Also register with global handler
    logger.info(f"Registered dashboard thread: {thread.name}")


def cleanup_dashboard_components() -> None:
    """Clean up all registered dashboard components."""
    global _is_shutting_down
    _is_shutting_down = True
    _shutdown_event.set()

    if _dashboard_components:
        logger.info("Cleaning up dashboard components...")
        for name, item in _dashboard_components.items():
            try:
                if item["cleanup"] and callable(item["cleanup"]):
                    logger.info(f"Running cleanup for {name}")
                    item["cleanup"]()
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}")

    # Join any dashboard-specific threads
    for thread in _dashboard_threads:
        try:
            if thread.is_alive():
                logger.info(f"Waiting for dashboard thread to finish: {thread.name}")
                thread.join(timeout=2.0)
        except Exception as e:
            logger.error(f"Error joining thread {thread.name}: {e}")

    logger.info("Dashboard cleanup completed")


def show_shutdown_message() -> None:
    """Display a shutdown message in the Streamlit UI."""
    try:
        st.warning(
            "⚠️ Application is shutting down. Please wait while we save your data..."
        )
        # Create a placeholder for a progress bar
        progress_placeholder = st.empty()

        # Show a progress bar that fills during shutdown
        for i in range(100):
            progress_placeholder.progress(i + 1)
            time.sleep(0.03)  # Short delay for visual effect

        st.success("✅ Shutdown complete! You can close this window.")

    except Exception as e:
        # If Streamlit is already shutting down, this might fail
        logger.debug(f"Could not show shutdown message: {e}")


def is_shutting_down() -> bool:
    """Check if the dashboard is currently shutting down."""
    return _is_shutting_down


@contextmanager
def handle_dashboard_errors(component_name: str = "dashboard"):
    """
    Context manager to handle errors in dashboard components.

    Args:
        component_name: Name of the component for logging
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {component_name}: {str(e)}", exc_info=True)
        # Don't crash completely, but show error in UI
        st.error(f"An error occurred in {component_name}. Check logs for details.")
        # In case of critical errors, consider shutdown
        if isinstance(e, (MemoryError, KeyboardInterrupt)):
            initiate_shutdown(source=f"Critical error in {component_name}")


def setup_dashboard_shutdown() -> None:
    """Initialize dashboard shutdown handlers."""
    # Register the dashboard cleanup as a shutdown callback
    register_shutdown_callback(cleanup_dashboard_components)

    # Register exit handler
    atexit.register(cleanup_dashboard_components)

    # Set handlers for CTRL+C
    def handle_keyboard_interrupt(*args):
        st.warning("Shutdown initiated by user (Ctrl+C)")
        initiate_shutdown(source="Keyboard Interrupt (Ctrl+C)")
        return True

    # Register for SIGINT if running in main thread
    try:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    except Exception:
        # May fail in non-main threads or some environments
        pass

    logger.info("Dashboard shutdown handlers initialized")


# Initialize on import
setup_dashboard_shutdown()
