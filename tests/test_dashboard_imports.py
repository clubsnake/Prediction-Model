"""
Tests that all required imports for the dashboard are working.
"""

import os
import sys
import unittest
import importlib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Streamlit: {e}")

try:
    # Import directly from project structure
    from src.utils.env_setup import setup_tf_environment

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
    print("✅ gpu_memory_management imported successfully")
except ImportError as e:
    print(f"❌ Failed to import gpu_memory_management: {e}")

# Try importing dashboard components
try:
    print("✅ All dashboard components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dashboard components: {e}")

print("\nTest complete. If you see any failure messages above, they need to be fixed.")

class TestDashboardImports(unittest.TestCase):
    """Test that all dashboard modules can be imported without errors"""
    
    def test_dashboard_imports(self):
        """Test importing all dashboard modules"""
        # Add project root to path
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        
        # List of modules to test
        modules = [
            "src.dashboard.dashboard.dashboard_core",
            "src.dashboard.dashboard.dashboard_data",
            "src.dashboard.dashboard.dashboard_error",
            "src.dashboard.dashboard.dashboard_model",
            "src.dashboard.dashboard.dashboard_state",
            "src.dashboard.dashboard.dashboard_ui",
            "src.dashboard.dashboard.dashboard_utils",
            "src.dashboard.dashboard.dashboard_visualization"
        ]
        
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module, f"Failed to import {module_name}")
            except Exception as e:
                self.fail(f"Error importing {module_name}: {str(e)}")
    
    def test_config_loader(self):
        """Test importing configuration"""
        try:
            from config.config_loader import TUNING_TRIALS_PER_CYCLE, TUNING_TRIALS_PER_CYCLE_min, TUNING_TRIALS_PER_CYCLE_max
            
            self.assertIsNotNone(TUNING_TRIALS_PER_CYCLE)
            self.assertIsNotNone(TUNING_TRIALS_PER_CYCLE_min)
            self.assertIsNotNone(TUNING_TRIALS_PER_CYCLE_max)
        except Exception as e:
            self.fail(f"Error importing config constants: {str(e)}")

if __name__ == "__main__":
    unittest.main()
