"""
Utilities for hyperparameter management and experiment reproducibility.
"""
import random
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class HyperparameterManager:
    """
    Central registry for hyperparameter definitions to maintain consistency
    across different optimization and training modules.
    """
    
    def __init__(self):
        self.parameter_registry = {}
        self._initialize_default_registry()
    
    def _initialize_default_registry(self):
        """Initialize the registry with default parameter definitions."""
        # LSTM parameters
        self.register_parameter('lstm_lr', {
            'type': 'float',
            'range': [1e-5, 1e-2],
            'default': 0.001,
            'log_scale': True,
            'description': 'Learning rate for LSTM models'
        })
        
        self.register_parameter('lstm_dropout', {
            'type': 'float',
            'range': [0.0, 0.5],
            'default': 0.2,
            'description': 'Dropout rate for LSTM models'
        })
        
        self.register_parameter('lstm_units', {
            'type': 'int',
            'range': [16, 256],
            'default': 64,
            'log_scale': True,
            'description': 'Number of units in LSTM layers'
        })
        
        # CNN parameters
        self.register_parameter('cnn_num_conv_layers', {
            'type': 'int',
            'range': [1, 5],
            'default': 3,
            'description': 'Number of convolutional layers in CNN'
        })
        
        self.register_parameter('cnn_num_filters', {
            'type': 'int',
            'range': [16, 256],
            'default': 64,
            'log_scale': True,
            'description': 'Number of filters in CNN layers'
        })
        
        self.register_parameter('cnn_kernel_size', {
            'type': 'int',
            'range': [2, 7],
            'default': 3,
            'description': 'Kernel size for CNN layers'
        })
        
        # Random Forest parameters
        self.register_parameter('rf_n_est', {
            'type': 'int',
            'range': [50, 500],
            'default': 100,
            'log_scale': True,
            'description': 'Number of estimators in Random Forest'
        })
        
        self.register_parameter('rf_mdepth', {
            'type': 'int',
            'range': [3, 25],
            'default': 10,
            'description': 'Maximum depth of trees in Random Forest'
        })
        
        # XGBoost parameters
        self.register_parameter('xgb_lr', {
            'type': 'float',
            'range': [1e-4, 0.5],
            'default': 0.1,
            'log_scale': True,
            'description': 'Learning rate for XGBoost'
        })
        
        # Generic parameters
        self.register_parameter('lookback', {
            'type': 'int',
            'range': [7, 90],
            'default': 30,
            'description': 'Number of past time steps to consider'
        })
        
        # Add more parameters as needed...
    
    def register_parameter(self, name, config):
        """
        Register a hyperparameter with its configuration.
        
        Args:
            name: Parameter name
            config: Dictionary with parameter configuration
                - type: Parameter type ('int', 'float', 'categorical', 'bool')
                - range: List with [min, max] or categorical options
                - default: Default value
                - log_scale: Whether to use log scale for numerical parameters
                - description: Parameter description
        """
        self.parameter_registry[name] = config
        logger.debug(f"Registered parameter: {name}")
    
    def get_parameter_config(self, name):
        """Get the configuration for a parameter."""
        if name not in self.parameter_registry:
            logger.warning(f"Parameter '{name}' not found in registry")
            return None
        return self.parameter_registry[name]
    
    def suggest_parameter(self, trial, name, override_range=None):
        """
        Suggest a parameter value using Optuna trial.
        
        Args:
            trial: Optuna trial object
            name: Parameter name
            override_range: Optional override for parameter range
            
        Returns:
            Suggested parameter value
        """
        if name not in self.parameter_registry:
            logger.warning(f"Parameter '{name}' not found in registry")
            return None
        
        config = self.parameter_registry[name]
        param_type = config['type']
        param_range = override_range or config.get('range')
        use_log = config.get('log_scale', False)
        
        if param_type == 'int':
            if param_range:
                return trial.suggest_int(name, param_range[0], param_range[1], log=use_log)
            else:
                return config['default']
        elif param_type == 'float':
            if param_range:
                return trial.suggest_float(name, param_range[0], param_range[1], log=use_log)
            else:
                return config['default']
        elif param_type == 'categorical':
            if param_range:
                return trial.suggest_categorical(name, param_range)
            else:
                return config['default']
        elif param_type == 'bool':
            return trial.suggest_categorical(name, [True, False])
        
        return config['default']

def set_all_random_states(seed):
    """
    Set random seeds for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: Random seed to use
        
    Returns:
        The seed used
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set TensorFlow's random seed
    try:
        tf.random.set_seed(seed)
    except:
        # Fall back to older TensorFlow versions
        try:
            tf.set_random_seed(seed)
        except:
            logger.warning("Could not set TensorFlow random seed")
    
    # Try to set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    logger.info(f"Set all random states to seed: {seed}")
    return seed
