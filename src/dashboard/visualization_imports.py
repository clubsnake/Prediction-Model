"""
Unified imports for visualization modules.
This module centralizes all imports needed by various visualization components.
"""

import logging
import os
import sys


# Configure logger
logger = logging.getLogger(__name__)

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(dashboard_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import visualization libraries with error handling
try:
    # Matplotlib imports
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    matplotlib.use("Agg")  # Use non-interactive backend

    # Plotly imports
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Set flag indicating visualization libraries are available
    HAS_VISUALIZATION_LIBS = True
    logger.info("Successfully imported visualization libraries")
except ImportError as e:
    HAS_VISUALIZATION_LIBS = False
    logger.warning(f"Error importing visualization libraries: {e}")

# Try to import Streamlit for dashboard components
try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    logger.warning("Streamlit not available. Dashboard visualizations will be limited.")

# Try to import config values with error handling
try:
    from config.config_loader import get_value

    logger.info("Successfully loaded visualization config from config_loader")
except ImportError:
    logger.warning("Config not found, using default visualization settings")

    # Define a fallback get_value function
    def get_value(path, default=None, target=None):
        return default

    # Default settings if config import fails
    SHOW_PREDICTION_PLOTS = True
    SHOW_TRAINING_HISTORY = True
    SHOW_WEIGHT_HISTOGRAMS = True
    VIZ_SETTINGS = {
        "lookback": 30,
        "prediction_horizon": 5,  # Default to 5 days if not specified
        "chart_height": 600,
        "color_theme": "default",
        "show_volume": True,
        "indicators": {
            "show_ma": False,
            "show_bb": False,
            "show_rsi": False,
            "show_macd": False,
        },
    }

# Standard color schemes
COLOR_SCHEMES = {
    "default": {
        "price_up": "#26a69a",
        "price_down": "#ef5350",
        "forecast": "#9c27b0",
        "volume_up": "rgba(38, 166, 154, 0.5)",
        "volume_down": "rgba(239, 83, 80, 0.5)",
        "ma20": "#ff9800",
        "ma50": "#2196f3",
        "ma200": "#757575",
    },
    "dark": {
        "price_up": "#00e676",
        "price_down": "#ff5252",
        "forecast": "#bb86fc",
        "volume_up": "rgba(0, 230, 118, 0.5)",
        "volume_down": "rgba(255, 82, 82, 0.5)",
        "ma20": "#ffab40",
        "ma50": "#40c4ff",
        "ma200": "#b0bec5",
    },
    "monochrome": {
        "price_up": "#424242",
        "price_down": "#757575",
        "forecast": "#000000",
        "volume_up": "rgba(66, 66, 66, 0.5)",
        "volume_down": "rgba(117, 117, 117, 0.5)",
        "ma20": "#212121",
        "ma50": "#616161",
        "ma200": "#9e9e9e",
    },
}


def get_active_color_scheme():
    """Get the currently active color scheme based on settings"""
    theme = VIZ_SETTINGS.get("color_theme", "default")
    return COLOR_SCHEMES.get(theme, COLOR_SCHEMES["default"])
