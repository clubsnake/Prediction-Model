"""
dashboard_utils.py

Utility functions for the dashboard that don't fit in other categories.
"""

import os
import base64
from datetime import datetime

import pandas as pd
import streamlit as st

from config.logger_config import logger

# Use absolute imports for dashboard components
from src.dashboard.dashboard.dashboard_error import robust_error_boundary

# Add paths to project root if needed
try:
    current_file = os.path.abspath(__file__)
    dashboard_dir = os.path.dirname(current_file)
    dashboard_parent = os.path.dirname(dashboard_dir)
    src_dir = os.path.dirname(dashboard_parent)
    project_root = os.path.dirname(src_dir)

    # Define common paths
    DATA_DIR = os.path.join(project_root, "data")
except Exception as e:
    logger.warning(f"Error setting up paths in dashboard_utils: {e}")
    DATA_DIR = "data"  # Fallback


def safe_mkdir(directory):
    """Create directory safely with error handling"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False


def clean_memory(force_gc=False):
    """Clean up memory to avoid out-of-memory issues"""
    try:
        import gc

        if force_gc:
            gc.collect()
    except Exception as e:
        logger.error(f"Error cleaning memory: {e}")


@robust_error_boundary
def show_error_log():
    """Display error log for debugging"""
    st.header("Error Log")

    if "error_log" not in st.session_state or not st.session_state["error_log"]:
        st.info("No errors recorded in this session.")
        return

    if st.button("Clear Error Log"):
        st.session_state["error_log"] = []
        st.success("Error log cleared.")
        return

    # Convert to DataFrame for easier display
    try:
        error_data = []
        for error in st.session_state["error_log"]:
            error_data.append(
                {
                    "Time": error.get("timestamp", datetime.now()),
                    "Function": error.get("function", "Unknown"),
                    "Error": error.get("error", "Unknown error"),
                }
            )

        df_errors = pd.DataFrame(error_data)
        st.dataframe(df_errors)

        # Show detailed traceback for selected error
        if len(error_data) > 0:
            selected_index = st.selectbox(
                "Select error to view details",
                range(len(st.session_state["error_log"])),
                format_func=lambda i: f"{error_data[i]['Time']} - {error_data[i]['Function']}",
            )

            selected_error = st.session_state["error_log"][selected_index]
            if "traceback" in selected_error:
                st.code(selected_error["traceback"], language="python")
    except Exception as e:
        st.error(f"Error displaying error log: {e}")
        # Fallback to simpler display
        for i, error in enumerate(st.session_state["error_log"]):
            st.write(
                f"**Error {i+1}:** {error.get('timestamp')} - {error.get('function')} - {error.get('error')}"
            )


@robust_error_boundary
def create_download_section():
    """Create a section for downloading data and models"""
    st.subheader("Download Data and Models")

    # Download raw data
    if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
        df_raw = st.session_state["df_raw"]
        csv = df_raw.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="raw_data.csv">Download Raw Data (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No raw data available for download.")

    # Download model weights
    if (
        "current_model" in st.session_state
        and st.session_state["current_model"] is not None
    ):
        try:
            model = st.session_state["current_model"]
            model.save_weights("model_weights.h5")
            with open("model_weights.h5", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/h5;base64,{b64}" download="model_weights.h5">Download Model Weights (H5)</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error downloading model weights: {e}")
    else:
        st.info("No model available for download.")


def get_cache_function():
    """Get appropriate Streamlit cache function based on available version"""
    try:
        if hasattr(st, "cache_data"):
            return st.cache_data
        elif hasattr(st, "cache_resource"):
            return st.cache_resource
        else:
            return lambda **kwargs: (
                lambda f: f
            )  # Return identity decorator as fallback
    except Exception as e:
        logger.error(f"Error setting up cache function: {e}")
        return lambda f: f  # Return identity function as fallback


@robust_error_boundary
def detect_streamlit_version():
    """Detect Streamlit version and capabilities"""
    try:
        import streamlit as st

        version = st.__version__

        features = {
            "cache_data": hasattr(st, "cache_data"),
            "cache_resource": hasattr(st, "cache_resource"),
            "checkbox": hasattr(st, "checkbox"),
            "columns": hasattr(st, "columns"),
            "session_state": hasattr(st, "session_state"),
        }

        return version, features
    except Exception as e:
        logger.error(f"Error detecting Streamlit version: {e}")
        return "unknown", {}


@robust_error_boundary
def check_file_access(filepath):
    """Check if a file exists and is accessible"""
    try:
        exists = os.path.exists(filepath)
        readable = os.access(filepath, os.R_OK) if exists else False
        writable = os.access(filepath, os.W_OK) if exists else False

        return {
            "exists": exists,
            "readable": readable,
            "writable": writable,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error checking file access for {filepath}: {e}")
        return {"exists": False, "readable": False, "writable": False, "error": str(e)}


@robust_error_boundary
def get_dashboard_status():
    """Get current status of dashboard components"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "session_state_keys": (
            list(st.session_state.keys()) if hasattr(st, "session_state") else []
        ),
        "components": {
            "model_loaded": st.session_state.get("model_loaded", False),
            "tuning_in_progress": st.session_state.get("tuning_in_progress", False),
            "watchdog_active": "watchdog" in st.session_state
            and st.session_state["watchdog"] is not None,
            "data_loaded": "df_raw" in st.session_state
            and st.session_state["df_raw"] is not None,
        },
        "streamlit": detect_streamlit_version(),
    }

    return status
