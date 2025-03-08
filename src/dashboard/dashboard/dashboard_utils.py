"""
dashboard_utils.py

Utility functions for the dashboard that don't fit in other categories.
"""

import base64
import io
import os
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard_error import robust_error_boundary
from config.logger_config import logger


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
            return lambda **kwargs: (lambda f: f)  # Return identity decorator as fallback
    except Exception as e:
        logger.error(f"Error setting up cache function: {e}")
        return lambda f: f  # Return identity function as fallback