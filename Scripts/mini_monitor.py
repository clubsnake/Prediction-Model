# mini_monitor.py
"""
A simplified streamlit app that imports the 'create_mini_monitor' from dashboard.
"""

import streamlit as st
from dashboard import create_mini_monitor

def run_mini_monitor():
    """
    Run a minimal monitoring page, for quick checks or embedding.
    """
    st.set_page_config(
        page_title="Prediction Monitor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    create_mini_monitor()

if __name__ == "__main__":
    run_mini_monitor()
