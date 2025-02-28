# main.py
"""
Main entry point. Sets environment variables, launches Streamlit dashboard, 
and optionally performs real-time updates or auto-tuning.
"""

import os
from config import OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)

from config import (
    TICKER, REALTIME_UPDATE, START_DATE, INTERVAL,
    AUTO_RUN_TUNING
)
import subprocess
import threading
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='prediction_model.log'
)
logger = logging.getLogger()

def launch_dashboard() -> None:
    """
    Launch Streamlit dashboard in a separate subprocess.
    """
    try:
        subprocess.Popen(["streamlit", "run", "Scripts/dashboard.py"])
        logger.info("Dashboard launched successfully.")
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")

def main() -> None:
    """
    Main entry for the script. Optionally performs real-time data update
    and triggers the auto-tuning if configured.
    """
    logger.info("Starting main program...")
    # Launch Streamlit
    dashboard_thread = threading.Thread(target=launch_dashboard, daemon=True)
    dashboard_thread.start()

    # Example placeholder for real-time updates
    if REALTIME_UPDATE:
        logger.info("Real-time update logic would go here. (placeholder)")

    # Auto-run tuning if desired
    if AUTO_RUN_TUNING:
        try:
            import Scripts.meta_tuning as meta_tuning
            meta_tuning.main()
        except Exception as e:
            logger.error(f"Error in auto-run tuning: {e}")

if __name__ == "__main__":
    main()
