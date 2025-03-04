# main.py
"""
Main entry point. Sets environment variables, launches Streamlit dashboard, 
and optionally performs real-time updates or auto-tuning.
"""
import os
import sys
import subprocess
import threading
import logging
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

# Import TensorFlow safely
try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow is not installed. Install it with 'pip install tensorflow'.")

# Configure GPU first (ensure resource_config exists and is correct)
try:
    import resource_config  # This should configure GPU resources
except Exception as e:
    print(f"Error importing resource_config: {e}")

# Set environment variables from config; ensure they exist in config.py
try:
    from config import OMP_NUM_THREADS, TICKER, REALTIME_UPDATE, START_DATE, INTERVAL, AUTO_RUN_TUNING
except Exception as e:
    print(f"Error importing configuration: {e}")
    sys.exit(1)

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"  # Enable XLA devices

# Ensure the Logs directory exists
LOGS_DIR = os.path.join("Data", "Logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Update logging configuration to use the new directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(LOGS_DIR, "prediction_model.log")
)
logger = logging.getLogger()

# Import and register TFT model safely
try:
    from Scripts.temporal_fusion_transformer import add_tft_to_model_types, add_tft_to_optuna_search
    import Scripts.meta_tuning as meta_tuning

    # Register TFT with model system
    add_tft_to_model_types()

    # Add TFT to Optuna search
    add_tft_to_optuna_search(meta_tuning)

    # Update MODEL_TYPES in config if needed
    from config import MODEL_TYPES
    if 'tft' not in MODEL_TYPES:
        MODEL_TYPES.append('tft')

except Exception as e:
    logger.error(f"Error initializing TFT model: {e}")

# Fix Streamlit cache issue with backward compatibility
try:
    import streamlit as st
    # Handle Streamlit caching issue dynamically
    if hasattr(st, "cache_data"):
        cache_function = st.cache_data
    elif hasattr(st, "cache_resource"):
        cache_function = st.cache_resource
    else:
        cache_function = st.cache
        
    # For model loading specifically
    if hasattr(st, "cache_resource"):
        cache_model = st.cache_resource
    else:
        cache_model = st.cache(allow_output_mutation=True)
        
except ImportError:
    logger.error("Streamlit is not installed. Please install it using 'pip install streamlit'.")
    # Define fallback functions to avoid errors if streamlit isn't available
    def cache_function(func):
        return func
        
    def cache_model(func):
        return func

# Launch Dashboard
def launch_dashboard(mode="full") -> None:
    """
    Launch the appropriate dashboard in a separate subprocess.
    
    Args:
        mode: 'full' for main dashboard, 'enhanced' for the enhanced version
    """
    try:
        if mode == "enhanced":
            script_path = os.path.join("enhanced_dashboard.py")
            logger.info("Launching enhanced dashboard...")
        else:
            script_path = os.path.join("Scripts", "dashboard.py")
            logger.info(f"Launching standard dashboard...")
            
        command = [sys.executable, "-m", "streamlit", "run", script_path]
        subprocess.Popen(command, cwd=os.getcwd())
        logger.info(f"Dashboard ({mode}) launched successfully with command: %s", " ".join(command))
    except Exception as e:
        logger.error(f"Error launching {mode} dashboard: {e}")

def main() -> None:
    """
    Main entry for the script. Optionally performs real-time data update
    and triggers the auto-tuning if configured.
    """
    logger.info("Starting main program...")

    # Launch Streamlit dashboard in a separate thread
    try:
        # You can change this to "enhanced" to use the enhanced dashboard
        dashboard_mode = os.environ.get("DASHBOARD_MODE", "full")
        dashboard_thread = threading.Thread(target=lambda: launch_dashboard(dashboard_mode), daemon=True)
        dashboard_thread.start()
    except Exception as e:
        logger.error(f"Error starting dashboard thread: {e}")

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", choices=["full", "enhanced"], default="full", 
                        help="Select which dashboard to launch")
    args = parser.parse_args()
    
    os.environ["DASHBOARD_MODE"] = args.dashboard
    main()
