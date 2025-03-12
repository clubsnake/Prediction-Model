"""
Simple launcher for the Streamlit dashboard
"""

import os
import subprocess
from pathlib import Path

# Set environment variables to prevent common errors
os.environ["MPLBACKEND"] = "Agg"  # Fix for matplotlib.use() error

# Get dashboard path
dashboard_path = (
    Path(__file__).parent / "src" / "dashboard" / "dashboard" / "dashboard_core.py"
)

# Launch the dashboard
print("\n🚀 Starting Streamlit dashboard...")
try:
    subprocess.run(["streamlit", "run", str(dashboard_path)])
except Exception as e:
    print(f"\n❌ Error starting dashboard: {e}")
    print("\nTrying alternative launch method...")
    os.system(f"streamlit run {dashboard_path}")
