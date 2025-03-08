"""
Test script to demonstrate random seed functionality.
"""
import os
import sys
import numpy as np
import random

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import initialize_random_seed, RANDOM_SEED_MODE

def test_seed_consistency():
    """Test if random numbers are consistent with the seed."""
    seed = initialize_random_seed()
    
    print(f"Random Seed Mode: {RANDOM_SEED_MODE}")
    print(f"Seed used: {seed}")
    
    # Generate some random numbers
    print("\nRandom numbers from different libraries:")
    print("Python random:", [random.random() for _ in range(3)])
    print("NumPy random:", [np.random.random() for _ in range(3)])
    
    try:
        import tensorflow as tf
        print("TensorFlow random:", [tf.random.uniform([]).numpy() for _ in range(3)])
    except ImportError:
        print("TensorFlow not available")
    
    try:
        import torch
        print("PyTorch random:", [torch.rand(1).item() for _ in range(3)])
    except ImportError:
        print("PyTorch not available")

if __name__ == "__main__":
    test_seed_consistency()
