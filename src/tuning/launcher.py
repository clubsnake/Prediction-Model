"""
Standalone launcher for hyperparameter tuning.
This script provides a direct way to start tuning without going through the dashboard.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add project root to Python path for imports
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "tuning_launcher.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tuning_launcher")

def launch_tuning(ticker="ETH-USD", timeframe="1d", multipliers=None, cycle=1):
    """
    Launch the tuning process directly without going through the dashboard.
    
    Args:
        ticker: Ticker symbol
        timeframe: Timeframe
        multipliers: Dict with tuning multipliers
        cycle: Tuning cycle number
    
    Returns:
        Dict with tuning results
    """
    try:
        logger.info(f"Launching tuning for {ticker}/{timeframe}")
        
        # Set default multipliers if none provided
        if not multipliers:
            multipliers = {
                "trials_multiplier": 1.0,
                "epochs_multiplier": 1.0,
                "patience_multiplier": 1.0,
                "complexity_multiplier": 1.0,
            }
        
        # Import the tuning function
        from src.tuning.meta_tuning import start_tuning_process
        
        # Launch the tuning process
        logger.info(f"Starting tuning process for {ticker}/{timeframe} (cycle {cycle})")
        result = start_tuning_process(
            ticker=ticker,
            timeframe=timeframe,
            multipliers=multipliers,
            cycle=cycle,
            force_start=True
        )
        
        logger.info(f"Tuning process completed with result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error launching tuning: {e}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch hyperparameter tuning")
    parser.add_argument("--ticker", type=str, default="ETH-USD", help="Ticker symbol")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe")
    parser.add_argument("--trials", type=float, default=1.0, help="Trials multiplier")
    parser.add_argument("--epochs", type=float, default=1.0, help="Epochs multiplier")
    parser.add_argument("--cycle", type=int, default=1, help="Tuning cycle number")
    
    args = parser.parse_args()
    
    # Configure multipliers
    multipliers = {
        "trials_multiplier": args.trials,
        "epochs_multiplier": args.epochs,
        "patience_multiplier": 1.0,
        "complexity_multiplier": 1.0,
    }
    
    # Launch tuning
    print(f"Launching tuning for {args.ticker}/{args.timeframe} with multipliers: {multipliers}")
    result = launch_tuning(args.ticker, args.timeframe, multipliers, args.cycle)
    
    # Print result
    print(f"Tuning completed with result: {result}")
