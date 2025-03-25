"""
Diagnostic script to help debug issues with the tuning button in the dashboard.
This script checks the key dependencies and environment setup required for tuning.
"""

import os
import sys
import time
from datetime import datetime
import traceback

# Add project root to Python path for imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define data directory
DATA_DIR = os.path.join(project_root, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def check_imports():
    """Test importing key modules needed for tuning"""
    modules_to_check = [
        "src.dashboard.dashboard.dashboard_model",
        "src.tuning.meta_tuning",
        "src.tuning.progress_helper",
        "src.utils.threadsafe",
        "optuna",
        "yaml",
        "streamlit"
    ]
    
    print("Running import tests...\n")
    all_successful = True
    
    for module_name in modules_to_check:
        try:
            print(f"Trying to import {module_name}...")
            __import__(module_name)
            print(f"✅ Successfully imported {module_name}\n")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}\n")
            all_successful = False
        except Exception as e:
            print(f"❓ Unexpected error importing {module_name}: {e}\n")
            all_successful = False
            
    return all_successful

def check_tuning_status_file():
    """Check if tuning status file can be read and written"""
    print("\nChecking tuning status file operations...\n")
    status_file = os.path.join(DATA_DIR, "tuning_status.txt")
    
    # Check if file exists and can be read
    try:
        if os.path.exists(status_file):
            print(f"Tuning status file exists at: {status_file}")
            with open(status_file, "r") as f:
                content = f.read()
            print(f"Current content: {content[:100]}...")
        else:
            print(f"Tuning status file does not exist yet at: {status_file}")
    except Exception as e:
        print(f"❌ Error reading tuning status file: {e}")
        traceback.print_exc()
    
    # Try to write to the file
    try:
        print("\nTrying to write to tuning status file...")
        test_status = {
            "is_running": False,
            "status": "test",
            "timestamp": datetime.now().isoformat(),
            "test": True
        }
        
        # First try using progress_helper
        try:
            from src.tuning.progress_helper import write_tuning_status
            write_tuning_status(test_status)
            print("✅ Successfully wrote to tuning status file using progress_helper")
        except Exception as e:
            print(f"❌ Error using progress_helper: {e}")
            
            # Fallback to direct file writing
            try:
                import yaml
                with open(status_file, "w") as f:
                    yaml.safe_dump(test_status, f)
                print("✅ Successfully wrote to tuning status file directly")
            except Exception as e2:
                print(f"❌ Error writing directly to status file: {e2}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Unexpected error working with tuning status file: {e}")
        traceback.print_exc()

def test_start_tuning_import():
    """Test importing and running the start_tuning function"""
    print("\nTesting start_tuning function import...\n")
    
    try:
        from src.dashboard.dashboard.dashboard_model import start_tuning
        print("✅ Successfully imported start_tuning from dashboard_model")
        
        print("\nChecking function signature and docstring:")
        import inspect
        print(inspect.signature(start_tuning))
        if start_tuning.__doc__:
            print(f"Docstring: {start_tuning.__doc__[:100]}...")
        else:
            print("No docstring available")
            
        return True
    except ImportError as e:
        print(f"❌ Error importing start_tuning: {e}")
        return False
    except Exception as e:
        print(f"❓ Unexpected error: {e}")
        traceback.print_exc()
        return False

def check_path_issues():
    """Check for path issues that might affect imports"""
    print("\nChecking Python path...\n")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")
        
    # Check specific paths that should be present
    required_paths = [
        project_root,
        os.path.join(project_root, "src"),
        os.path.join(project_root, "src", "dashboard"),
        os.path.join(project_root, "src", "tuning")
    ]
    
    print("\nChecking required paths:")
    for path in required_paths:
        if path in sys.path:
            print(f"✅ Path present: {path}")
        else:
            print(f"❌ Path missing: {path}")
            print(f"   Adding path to sys.path...")
            sys.path.insert(0, path)

def check_directories():
    """Check if required directories exist and are writable"""
    print("\nChecking required directories...\n")
    
    # List of directories that should exist
    directories = [
        DATA_DIR,
        os.path.join(DATA_DIR, "models"),
        os.path.join(DATA_DIR, "models", "hyperparameters"),
        os.path.join(DATA_DIR, "model_progress"),
        os.path.join(DATA_DIR, "tested_models")
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
            # Check if writable
            test_file = os.path.join(directory, ".test_write")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                print(f"   Directory is writable")
            except Exception as e:
                print(f"❌ Directory is not writable: {e}")
        else:
            print(f"❌ Directory missing: {directory}")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   Created directory: {directory}")
            except Exception as e:
                print(f"❌ Could not create directory: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("TUNING BUTTON DIAGNOSTIC TOOL")
    print("=" * 50)
    print("\nThis script helps diagnose issues with the tuning button.")
    
    # Check Python and environment info
    print("\nEnvironment Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Run checks
    check_path_issues()
    check_directories()
    imports_ok = check_imports()
    check_tuning_status_file()
    start_tuning_ok = test_start_tuning_import()
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if imports_ok and start_tuning_ok:
        print("\n✅ All essential checks passed!")
        print("The tuning button should work properly. If it still doesn't work,")
        print("try running manual_study_launcher.py to test the tuning process directly.")
    else:
        print("\n❌ Some checks failed!")
        print("Please fix the issues highlighted above and try again.")
        print("Common issues:")
        print("1. Missing or incorrect imports")
        print("2. Path configuration problems")
        print("3. Permission issues with directories or files")
        print("4. Circular import issues")
    
    print("\nFor further diagnosis, try running:")
    print("python manual_study_launcher.py --ticker ETH-USD --timeframe 1d")
