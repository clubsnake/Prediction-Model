"""
Utilities package for common functions used throughout the prediction model project.

This package provides various utility functions and classes that are used across
different modules in the project. It includes:

1. Environment setup utilities for TensorFlow
2. Memory management tools like WeakRefCache and cleanup functions
3. Training optimization utilities
4. Logging utilities

The utils package is designed to be imported by other modules that need access
to these common utilities, centralizing shared functionality.
"""

# Import key utilities for easier access from other modules
# Remove problematic imports that don't exist or can't be found
# from src.utils.env_setup import setup_tf_environment
# from src.utils.memory_utils import WeakRefCache, cleanup_tf_session, log_memory_usage
# from src.utils.training_optimizer import get_training_optimizer

# Export common functions for easier imports by other modules
__all__ = [
    "setup_tf_environment",
    "WeakRefCache",
    "cleanup_tf_session",
    "log_memory_usage",
    "get_training_optimizer",
]
