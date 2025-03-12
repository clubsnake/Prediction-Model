# This import is incorrect - config_loader is in the config package
# from config_loader import DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR

# Correct import path
try:
    from config.config_loader import DATA_DIR, LOGS_DIR, MODELS_DIR

    # Note: REPORTS_DIR doesn't appear to exist in config.config_loader
    __all__ = ["DATA_DIR", "LOGS_DIR", "MODELS_DIR"]
except ImportError:
    # Define fallbacks if import fails
    import os

    project_root = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(project_root, "data")
    LOGS_DIR = os.path.join(DATA_DIR, "logs")
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    __all__ = ["DATA_DIR", "LOGS_DIR", "MODELS_DIR"]
