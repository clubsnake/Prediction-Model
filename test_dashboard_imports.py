"""
Tests that all required imports for the dashboard are working.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Streamlit: {e}")

try:
    from Scripts.env_setup import setup_tf_environment
    print("✅ env_setup imported successfully")
    setup_tf_environment()
    print("✅ setup_tf_environment executed successfully")
except ImportError as e:
    print(f"❌ Failed to import env_setup: {e}")
except Exception as e:
    print(f"❌ Error during setup_tf_environment: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully, version {tf.__version__}")
    print(f"   GPU available: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"❌ Failed to import TensorFlow: {e}")

try:
    from Scripts.gpu_memory_management import configure_gpu_memory
    print("✅ gpu_memory_management imported successfully")
except ImportError as e:
    print(f"❌ Failed to import gpu_memory_management: {e}")

# Try importing dashboard components
try:
    from Scripts.dashboard import create_control_panel, create_model_comparison, create_download_section
    print("✅ All dashboard components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dashboard components: {e}")

print("\nTest complete. If you see any failure messages above, they need to be fixed.")
