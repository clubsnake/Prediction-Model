#!/usr/bin/env python
"""
Debugging script for meta_tuning.py
This script will help identify where the 'os' variable issue is occurring.
"""

import sys
import traceback
import importlib
import inspect
import os
from pathlib import Path

# Set unlimited traceback depth
sys.tracebacklimit = None

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure basic logging to console
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_tuning")

def check_module_for_os_shadowing(module_name):
    """Check if the module has code that shadows the os module"""
    try:
        module = importlib.import_module(module_name)
        module_source = inspect.getsource(module)
        
        # Look for patterns like "def func(os):" or "os = something"
        problematic_lines = []
        for i, line in enumerate(module_source.splitlines()):
            if "def " in line and "(os" in line:
                problematic_lines.append((i+1, line.strip()))
            elif "os =" in line and not line.strip().startswith("#") and not "import os" in line:
                problematic_lines.append((i+1, line.strip()))
        
        if problematic_lines:
            print(f"⚠️ WARNING: Possible 'os' shadowing in {module_name}:")
            for line_num, line in problematic_lines:
                print(f"  Line {line_num}: {line}")
                
        return problematic_lines
    except Exception as e:
        logger.warning(f"Could not check module {module_name}: {e}")
        return []

def run_debug_checks():
    """Run preliminary debug checks"""
    print("=" * 80)
    print("RUNNING PRELIMINARY DEBUG CHECKS")
    print("=" * 80)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check os module
    print(f"OS module: {os.__name__} from {os.__file__}")
    
    # Check for modules that might shadow 'os'
    modules_to_check = [
        "src.tuning.meta_tuning",
        "src.tuning.study_manager",
        "src.tuning.progress_helper",
        "config.config_loader"
    ]
    
    for module_name in modules_to_check:
        check_module_for_os_shadowing(module_name)
    
    # Check for circular imports
    print("\nChecking for circular import issues...")
    already_loaded = set(sys.modules.keys())
    
    try:
        print("Loading study_manager...")
        import src.tuning.study_manager
        print("✓ Successfully loaded study_manager")
    except Exception as e:
        print(f"✗ Error loading study_manager: {e}")
        traceback.print_exc()
    
    try:
        print("Loading meta_tuning...")
        import src.tuning.meta_tuning
        print("✓ Successfully loaded meta_tuning")
    except Exception as e:
        print(f"✗ Error loading meta_tuning: {e}")
        traceback.print_exc()
    
    newly_loaded = set(sys.modules.keys()) - already_loaded
    print(f"Newly loaded modules: {newly_loaded}")
    
    print("\nChecking for StudyManager and related imports...")
    try:
        from src.tuning.study_manager import StudyManager
        print("✓ Successfully imported StudyManager class")
    except Exception as e:
        print(f"✗ Error importing StudyManager: {e}")
    
    print("=" * 80)

def run_tuning_test():
    """Try to run the tuning process"""
    print("=" * 80)
    print("ATTEMPTING TO START TUNING PROCESS")
    print("=" * 80)
    
    # Define test parameters
    ticker = "ETH-USD"
    timeframe = "1d"
    
    try:
        # First try importing modules without calling anything
        print("Step 1: Importing necessary modules...")
        from src.tuning.progress_helper import read_tuning_status, set_stop_requested
        from src.utils.memory_utils import adaptive_memory_clean
        
        print("Step 2: Importing study_manager...")
        from src.tuning.study_manager import StudyManager
        
        # Load tuning module carefully
        print("Step 3: Loading meta_tuning module...")
        meta_tuning_spec = importlib.util.find_spec("src.tuning.meta_tuning")
        print(f"Found meta_tuning at: {meta_tuning_spec.origin}")
        
        # Import functions individually
        print("Step 4: Importing start_tuning_process...")
        from src.tuning.meta_tuning import start_tuning_process
        
        # Now try to run the tuning process
        print("\nStep 5: Starting tuning process...")
        print(f"Parameters: ticker={ticker}, timeframe={timeframe}")
        
        result = start_tuning_process(ticker, timeframe)
        
        print("✓ Tuning process completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print("\n\n❌ ERROR DURING TUNING PROCESS:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # After error, try to import differently
        print("\nAttempting alternate debugging approach...")
        try:
            # Import the main module directly to see where it fails
            import src.tuning.meta_tuning
            print("✓ Module imported successfully on second try")
        except Exception as e2:
            print(f"❌ Module import also failed: {e2}")

def direct_import_test():
    """Test direct importing of modules for debugging"""
    print("=" * 80)
    print("TESTING DIRECT IMPORTS")
    print("=" * 80)
    
    test_modules = [
        "config.config_loader",
        "src.utils.threadsafe",
        "src.utils.memory_utils",
        "src.tuning.progress_helper",
        "src.tuning.study_manager",
        "src.tuning.meta_tuning"
    ]
    
    for module_name in test_modules:
        try:
            print(f"Importing {module_name}...")
            module = importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name}")
            
            # Print first few defined names in the module
            module_attrs = dir(module)
            print(f"  Module defines: {module_attrs[:5]}...")
            
        except Exception as e:
            print(f"❌ Failed to import {module_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEBUG SCRIPT FOR TUNING PROCESS")
    print("=" * 80)
    
    try:
        # Run checks in sequence
        run_debug_checks()
        direct_import_test()
        run_tuning_test()
        
    except Exception as e:
        print("\n\n❌ UNEXPECTED ERROR IN DEBUG SCRIPT:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DEBUG SCRIPT COMPLETE")
    print("=" * 80)