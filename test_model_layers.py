"""
Unit tests for custom TensorFlow layers and model components.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path to import from Scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Disable TensorFlow logging for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class TestModelLayers(unittest.TestCase):
    """Test cases for custom model layers."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        # Create a small test dataset
        self.X_test = np.random.normal(0, 1, (10, 5, 3))
        self.y_test = np.random.normal(0, 1, (10, 7))
    
    def test_lstm_layer_shapes(self) -> None:
        """Test that LSTM layer produces expected output shapes."""
        # Create a simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(5, 3), return_sequences=False),
            tf.keras.layers.Dense(7)
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Check output shape
        output = model.predict(self.X_test)
        self.assertEqual(output.shape, (10, 7))
    
    def test_model_training_stability(self) -> None:
        """Test that model training is stable with different initializations."""
        losses = []
        
        # Train multiple models with different seeds
        for seed in range(3):
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(16, input_shape=(5, 3)),
                tf.keras.layers.Dense(7)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            history = model.fit(
                self.X_test, self.y_test, 
                epochs=5, verbose=0
            )
            
            losses.append(history.history['loss'][-1])
        
        # Check that losses aren't too different (stability check)
        max_diff = max(losses) - min(losses)
        self.assertLess(max_diff, 1.0, "Model training is unstable across seeds")
    
    def test_numerical_stability(self) -> None:
        """Test numerical stability with extreme values."""
        # Create data with extreme values
        X_extreme = np.concatenate([
            np.random.normal(0, 1e5, (5, 5, 3)),  # Very large values
            np.random.normal(0, 1e-5, (5, 5, 3))   # Very small values
        ])
        y_extreme = np.concatenate([
            np.random.normal(0, 1e5, (5, 7)),
            np.random.normal(0, 1e-5, (5, 7))
        ])
        
        # Build model with batch normalization for stability
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, input_shape=(5, 3), return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(7)
        ])
        
        # Use a robust optimizer and loss
        optimizer = tf.keras.optimizers.RMSprop(clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='huber')
        
        # This should run without numerical errors
        try:
            model.fit(X_extreme, y_extreme, epochs=3, verbose=0)
            prediction = model.predict(X_extreme)
            self.assertEqual(prediction.shape, (10, 7))
            self.assertTrue(np.all(np.isfinite(prediction)))
        except tf.errors.InvalidArgumentError:
            self.fail("Numerical instability detected")
    
    def test_output_range(self) -> None:
        """Test that outputs are within reasonable ranges."""
        # Create a model with tanh activation for bounded output
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, input_shape=(5, 3)),
            tf.keras.layers.Dense(7, activation='tanh')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(self.X_test, self.y_test, epochs=5, verbose=0)
        
        # Check predictions are in the expected range of tanh (-1 to 1)
        predictions = model.predict(self.X_test)
        self.assertTrue(np.all(predictions >= -1.0))
        self.assertTrue(np.all(predictions <= 1.0))

if __name__ == "__main__":
    unittest.main()
