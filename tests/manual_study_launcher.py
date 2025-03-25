"""
A utility script to manually launch tuning studies for testing purposes.
This can help diagnose issues with the tuning button in the dashboard.
"""

import os
import sys
import time
from datetime import datetime

# Add project root to Python path for imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define data directory
DATA_DIR = os.path.join(project_root, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def write_tuning_status(status_dict):
    """Write tuning status to a file"""
    try:
        # First try using progress_helper
        from src.tuning.progress_helper import write_tuning_status as wts
        wts(status_dict)
        print(f"Wrote tuning status via progress_helper: {status_dict}")
    except Exception as e:
        print(f"Error using progress_helper: {e}")
        # Fallback to direct file writing
        status_file = os.path.join(DATA_DIR, "tuning_status.txt")
        try:
            import yaml
            with open(status_file, "w") as f:
                yaml.safe_dump(status_dict, f)
            print(f"Wrote tuning status directly to file: {status_dict}")
        except Exception as e2:
            print(f"Error writing status file: {e2}")

def start_manual_tuning(ticker="ETH-USD", timeframe="1d"):
    """Start a tuning process manually to test the system"""
    print(f"Starting manual tuning for {ticker}/{timeframe}")
    
    # First update status
    status = {
        "is_running": True,
        "status": "starting",
        "ticker": ticker,
        "timeframe": timeframe,
        "timestamp": datetime.now().isoformat(),
        "start_time": time.time()
    }
    write_tuning_status(status)
    
    # Import and call the tuning function from meta_tuning
    try:
        from src.tuning.meta_tuning import start_tuning_process
        print("Successfully imported start_tuning_process")
        
        # Call with default parameters for testing
        result = start_tuning_process(
            ticker=ticker, 
            timeframe=timeframe, 
            multipliers={"trials_multiplier": 0.2, "n_startup_trials": 5},
            force_start=True
        )
        
        print(f"Tuning process completed with result: {result}")
        return result
    except Exception as e:
        print(f"Error starting tuning process: {e}")
        import traceback
        traceback.print_exc()
        
        # Update status to show error
        error_status = {
            "is_running": False,
            "status": "error",
            "ticker": ticker,
            "timeframe": timeframe,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        write_tuning_status(error_status)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual study launcher for testing")
    parser.add_argument("--ticker", type=str, default="ETH-USD", help="Ticker symbol")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe")
    
    args = parser.parse_args()
    
    print(f"Starting manual tuning for {args.ticker}/{args.timeframe}")
    result = start_manual_tuning(args.ticker, args.timeframe)
    
    print(f"Manual tuning completed with result: {result}")
