import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dashboard.dashboard.dashboard_model import read_tuning_status, write_tuning_status

print("Testing fallback functions...")

# Test read_tuning_status
status = read_tuning_status()
print(f"Current tuning status: {status}")

# Test write_tuning_status
write_tuning_status({
    "is_running": False,
    "status": "test_status",
    "test_key": "test_value",
    "timestamp": "2025-03-31T02:07:00"
})

# Read it back to verify
status = read_tuning_status()
print(f"Updated tuning status: {status}")

print("Fallback tests completed successfully!")
