"""
Advanced N-BEATS (Neural Basis Expansion Analysis for Time Series) implementation
optimized for price prediction with full Optuna hyperparameter tuning integration.

Based on the paper: N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
by Oreshkin et al. (https://arxiv.org/abs/1905.10437)
"""

import logging
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras import layers, models, optimizers, regularizers  # type: ignore

logger = logging.getLogger(__name__)


class StackType(str, Enum):
    """Stack types for N-BEATS architecture"""
    GENERIC = "generic"
    TREND = "trend"
    SEASONALITY = "seasonality"
    INTERPRETABLE = "interpretable"
    PRICE_SPECIFIC = "price_specific"  # Special stack for financial time series


class NBeatsBlock(layers.Layer):
    """
    N-BEATS Block implementation with support for different basis expansions.
    Each block consists of fully connected stack with forecast and backcast branches.
    """
    
    def __init__(
        self,
        units: int,
        stack_type: StackType,
        thetas_dim: int,
        share_thetas: bool = False,
        num_layers: int = 4,
        layer_width: int = 256,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        basis_function_type: str = "default",
        expansion_coefficient_dim: int = 5,
        backcast_length: int = 30,
        forecast_length: int = 7,
        **kwargs
    ):
        """
        Initialize N-BEATS block.
        
        Args:
            units: Output dimension for the block
            stack_type: Type of stack this block belongs to
            thetas_dim: Dimension of thetas for basis expansion
            share_thetas: Whether to share thetas between backcast and forecast
            num_layers: Number of fully connected layers
            layer_width: Width of each fully connected layer
            activation: Activation function for dense layers
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            kernel_regularizer: Regularization for kernel weights
            basis_function_type: Type of basis function to use
            expansion_coefficient_dim: Dimension for basis expansion coefficients
            backcast_length: Length of input sequence
            forecast_length: Length of forecast sequence
        """
        super(NBeatsBlock, self).__init__(**kwargs)
        self.units = units
        self.stack_type = stack_type
        self.thetas_dim = thetas_dim
        self.share_thetas = share_thetas
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.basis_function_type = basis_function_type
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.expansion_coefficient_dim = expansion_coefficient_dim
        
        # FC stack
        self.fc_layers = []
        for i in range(num_layers):
            self.fc_layers.append(
                layers.Dense(
                    layer_width,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                    name=f'fc_{i}'
                )
            )
            if dropout_rate > 0:
                self.fc_layers.append(layers.Dropout(dropout_rate))
            if use_batch_norm:
                self.fc_layers.append(layers.BatchNormalization())
                
        # Theta layer for backcast
        self.theta_b_layer = layers.Dense(
            thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_b'
        )
        
        # Theta layer for forecast (optional sharing)
        if share_thetas:
            self.theta_f_layer = self.theta_b_layer
        else:
            self.theta_f_layer = layers.Dense(
                thetas_dim,
                activation='linear',
                use_bias=False,
                name='theta_f'
            )

    def build(self, input_shape):
        super(NBeatsBlock, self).build(input_shape)
        
        # Initialize basis expansion matrices based on stack type
        if self.stack_type == StackType.TREND:
            self._build_trend_basis()
        elif self.stack_type == StackType.SEASONALITY:
            self._build_seasonality_basis()
        elif self.stack_type == StackType.PRICE_SPECIFIC:
            self._build_price_specific_basis()
        # For GENERIC and INTERPRETABLE, basis is built dynamically

    def _build_trend_basis(self):
        """Build polynomial basis for trend stack"""
        # Create polynomial trend basis
        t_backcast = np.arange(self.backcast_length) / self.backcast_length
        t_forecast = np.arange(self.forecast_length) / self.forecast_length
        
        # Use polynomial basis of order = expansion_coefficient_dim
        backcast_basis = np.zeros((self.backcast_length, self.expansion_coefficient_dim))
        forecast_basis = np.zeros((self.forecast_length, self.expansion_coefficient_dim))
        
        for i in range(self.expansion_coefficient_dim):
            backcast_basis[:, i] = t_backcast ** i
            forecast_basis[:, i] = t_forecast ** i
            
        self.backcast_basis = tf.constant(backcast_basis, dtype=tf.float32)
        self.forecast_basis = tf.constant(forecast_basis, dtype=tf.float32)

    def _build_seasonality_basis(self):
        """Build Fourier basis for seasonality stack"""
        # Create Fourier basis functions
        backcast_frequencies = np.arange(self.expansion_coefficient_dim)
        forecast_frequencies = np.arange(self.expansion_coefficient_dim)
        
        backcast_time = np.arange(self.backcast_length)
        forecast_time = np.arange(self.forecast_length)
        
        backcast_basis = np.zeros((self.backcast_length, 2 * self.expansion_coefficient_dim))
        forecast_basis = np.zeros((self.forecast_length, 2 * self.expansion_coefficient_dim))
        
        for i, freq in enumerate(backcast_frequencies):
            # Sine component
            backcast_basis[:, 2*i] = np.sin(2 * np.pi * freq * backcast_time / self.backcast_length)
            # Cosine component
            backcast_basis[:, 2*i+1] = np.cos(2 * np.pi * freq * backcast_time / self.backcast_length)
            
        for i, freq in enumerate(forecast_frequencies):
            # Sine component
            forecast_basis[:, 2*i] = np.sin(2 * np.pi * freq * forecast_time / self.forecast_length)
            # Cosine component
            forecast_basis[:, 2*i+1] = np.cos(2 * np.pi * freq * forecast_time / self.forecast_length)
        
        self.backcast_basis = tf.constant(backcast_basis, dtype=tf.float32)
        self.forecast_basis = tf.constant(forecast_basis, dtype=tf.float32)

    def _build_price_specific_basis(self):
        """Build specialized basis for price time series that combines multiple patterns"""
        # Combine polynomial trend, Fourier seasonality, and exponential components
        # specialized for financial time series
        
        # Time vectors
        t_backcast = np.arange(self.backcast_length) / self.backcast_length
        t_forecast = np.arange(self.forecast_length) / self.forecast_length
        
        # Number of basis functions for each component
        trend_dim = self.expansion_coefficient_dim // 3
        seasonality_dim = self.expansion_coefficient_dim // 3
        volatility_dim = self.expansion_coefficient_dim - trend_dim - seasonality_dim
        
        # Initialize basis arrays
        backcast_basis = np.zeros((self.backcast_length, self.expansion_coefficient_dim))
        forecast_basis = np.zeros((self.forecast_length, self.expansion_coefficient_dim))
        
        # 1. Trend components (polynomial)
        for i in range(trend_dim):
            backcast_basis[:, i] = t_backcast ** (i + 1)
            forecast_basis[:, i] = t_forecast ** (i + 1)
            
        # 2. Seasonality components (Fourier)
        for i in range(seasonality_dim):
            freq = i + 1
            idx = trend_dim + i
            # Use only cosine for simplicity (market often has weekly/monthly patterns)
            backcast_basis[:, idx] = np.cos(2 * np.pi * freq * t_backcast)
            forecast_basis[:, idx] = np.cos(2 * np.pi * freq * t_forecast)
            
        # 3. Volatility/momentum components (exponential decay)
        for i in range(volatility_dim):
            rate = 1.0 + i * 2.0  # Different decay rates
            idx = trend_dim + seasonality_dim + i
            # Exponential decay from recent to older values (momentum)
            backcast_basis[:, idx] = np.exp(-rate * (1.0 - t_backcast))
            forecast_basis[:, idx] = np.exp(-rate * (1.0 - t_forecast))
            
        self.backcast_basis = tf.constant(backcast_basis, dtype=tf.float32)
        self.forecast_basis = tf.constant(forecast_basis, dtype=tf.float32)

    def call(self, inputs, training=None):
        """Forward pass for the N-BEATS block"""
        x = inputs
        # Apply fully connected layers
        for layer in self.fc_layers:
            x = layer(x, training=training)
            
        # Compute thetas for backcast and forecast
        theta_b = self.theta_b_layer(x)
        theta_f = self.theta_f_layer(x)
        
        # Compute backcast and forecast outputs based on stack type
        if self.stack_type in [StackType.TREND, StackType.SEASONALITY, StackType.PRICE_SPECIFIC]:
            backcast = tf.matmul(self.backcast_basis, theta_b, transpose_b=True)
            forecast = tf.matmul(self.forecast_basis, theta_f, transpose_b=True)
        elif self.stack_type == StackType.GENERIC:
            backcast = theta_b
            forecast = theta_f
        else:  # StackType.INTERPRETABLE - dynamic basis creation
            # This is a simplified version - a real implementation would have more basis types
            backcast = self._dynamic_basis_expansion(theta_b, self.backcast_length)
            forecast = self._dynamic_basis_expansion(theta_f, self.forecast_length)
        
        return backcast, forecast
    
    def _dynamic_basis_expansion(self, thetas, target_length):
        """Create a dynamic basis expansion based on learned weights"""
        # Reshape thetas for interpretability
        thetas_reshaped = tf.reshape(thetas, (-1, self.expansion_coefficient_dim))
        
        # Create dynamic basis matrix based on theta parameters
        basis_vectors = []
        for i in range(self.expansion_coefficient_dim):
            if i % 3 == 0:  # Trend
                t = tf.range(target_length, dtype=tf.float32) / target_length
                basis = t ** (i // 3 + 1)
            elif i % 3 == 1:  # Seasonality
                t = tf.range(target_length, dtype=tf.float32)
                basis = tf.cos(2 * np.pi * (i // 3 + 1) * t / target_length)
            else:  # Volatility/momentum
                t = tf.range(target_length, dtype=tf.float32) / target_length
                basis = tf.exp(-2.0 * (i // 3 + 1) * (1.0 - t))
                
            basis_vectors.append(basis)
        
        # Stack basis vectors
        stacked_basis = tf.stack(basis_vectors, axis=1)
        
        # Apply thetas to basis
        return tf.matmul(stacked_basis, thetas_reshaped, transpose_b=True)
    
    def get_config(self):
        config = super(NBeatsBlock, self).get_config()
        config.update({
            'units': self.units,
            'stack_type': self.stack_type,
            'thetas_dim': self.thetas_dim,
            'share_thetas': self.share_thetas,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'basis_function_type': self.basis_function_type,
            'expansion_coefficient_dim': self.expansion_coefficient_dim,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
        })
        return config


class NBeatsStack(layers.Layer):
    """
    Stack of N-BEATS blocks with the same type and architecture.
    Acts as a sub-network specialized for a particular pattern (trend, seasonality, etc.)
    """
    
    def __init__(
        self,
        stack_type: StackType,
        num_blocks: int = 3,
        thetas_dim: int = 8,
        share_thetas: bool = False,
        num_layers: int = 4,
        layer_width: int = 256,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        basis_function_type: str = "default",
        expansion_coefficient_dim: int = 5,
        backcast_length: int = 30,
        forecast_length: int = 7,
        **kwargs
    ):
        """
        Initialize stack of N-BEATS blocks.
        
        Args:
            stack_type: Type of stack (trend, seasonality, generic)
            num_blocks: Number of blocks in this stack
            thetas_dim: Dimension of thetas for basis expansion
            share_thetas: Whether to share thetas between backcast and forecast
            num_layers: Number of fully connected layers per block
            layer_width: Width of each fully connected layer
            activation: Activation function for dense layers
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            kernel_regularizer: Regularization for kernel weights
            basis_function_type: Type of basis function to use
            expansion_coefficient_dim: Dimension for basis expansion coefficients
            backcast_length: Length of input sequence
            forecast_length: Length of forecast sequence
        """
        super(NBeatsStack, self).__init__(**kwargs)
        self.stack_type = stack_type
        self.num_blocks = num_blocks
        self.thetas_dim = thetas_dim
        self.share_thetas = share_thetas
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.basis_function_type = basis_function_type
        self.expansion_coefficient_dim = expansion_coefficient_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        
        # Create regularizer if specified
        if isinstance(kernel_regularizer, dict):
            if kernel_regularizer.get('type') == 'l1':
                self.kernel_regularizer = regularizers.l1(kernel_regularizer.get('value', 0.01))
            elif kernel_regularizer.get('type') == 'l2':
                self.kernel_regularizer = regularizers.l2(kernel_regularizer.get('value', 0.01))
            elif kernel_regularizer.get('type') == 'l1_l2':
                self.kernel_regularizer = regularizers.l1_l2(
                    l1=kernel_regularizer.get('l1', 0.01),
                    l2=kernel_regularizer.get('l2', 0.01)
                )
            else:
                self.kernel_regularizer = None
        else:
            self.kernel_regularizer = kernel_regularizer
        
        # Create blocks
        self.blocks = []
        for i in range(num_blocks):
            block = NBeatsBlock(
                units=layer_width,
                stack_type=stack_type,
                thetas_dim=thetas_dim,
                share_thetas=share_thetas,
                num_layers=num_layers,
                layer_width=layer_width,
                activation=activation,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                kernel_regularizer=self.kernel_regularizer,
                basis_function_type=basis_function_type,
                expansion_coefficient_dim=expansion_coefficient_dim,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                name=f'block_{i}'
            )
            self.blocks.append(block)

    def call(self, inputs, training=None):
        """Forward pass through the stack with double residual architecture"""
        backcast = inputs
        forecast = None
        
        # Process blocks sequentially with residual connections
        for block in self.blocks:
            # Extract backcast and forecast from block
            block_backcast, block_forecast = block(backcast, training=training)
            
            # Apply residual connection for backcast
            backcast = layers.subtract([backcast, block_backcast])
            
            # Initialize or accumulate forecast
            if forecast is None:
                forecast = block_forecast
            else:
                forecast = layers.add([forecast, block_forecast])
                
        return backcast, forecast
    
    def get_config(self):
        config = super(NBeatsStack, self).get_config()
        config.update({
            'stack_type': self.stack_type,
            'num_blocks': self.num_blocks,
            'thetas_dim': self.thetas_dim,
            'share_thetas': self.share_thetas,
            'num_layers': self.num_layers,
            'layer_width': self.layer_width,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'basis_function_type': self.basis_function_type,
            'expansion_coefficient_dim': self.expansion_coefficient_dim,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
        })
        return config


def build_nbeats_model(
    num_features: int,
    horizon: int,
    # Model architecture parameters
    lookback: int = 30,
    stack_types: List[StackType] = None,
    num_stacks: int = 2,
    num_blocks: Union[int, List[int]] = 3,
    num_layers: Union[int, List[int]] = 4,
    layer_width: Union[int, List[int]] = 256,
    thetas_dim: Union[int, List[int]] = 8,
    share_thetas: Union[bool, List[bool]] = False,
    expansion_coefficient_dim: Union[int, List[int]] = 5,
    # Training parameters
    learning_rate: float = 0.001,
    loss_function: str = "mse",
    use_gradient_clipping: bool = True,
    gradient_clip_norm: float = 1.0,
    use_weighted_loss: bool = False,
    use_learning_rate_schedule: bool = False,
    # Regularization parameters
    dropout_rate: Union[float, List[float]] = 0.0,
    use_batch_norm: Union[bool, List[bool]] = False,
    kernel_regularizer_type: str = None,
    kernel_regularizer_value: float = 0.01,
    activity_regularizer_value: float = 0.0,
    # Advanced features
    use_attention: bool = False,
    add_exogenous_variables: bool = True,
    use_residual_connections: bool = True,
    use_advanced_activations: bool = False,
    use_cyclic_learning_rate: bool = False,
    use_ensemble_outputs: bool = False,
    ensemble_size: int = 3,
    # Specific to price forecasting
    include_price_specific_stack: bool = True,
    use_volume_attention: bool = False,
    normalize_input_windows: bool = True,
    apply_price_scaling: bool = True,
):
    """
    Build a comprehensive N-BEATS model for time-series forecasting, optimized for price prediction.
    
    Args:
        num_features: Number of input features.
        horizon: Forecast horizon (number of timesteps to predict).
        lookback: Number of past timesteps to use for prediction.
        
        # Model architecture parameters
        stack_types: List of stack types to use; if None, automatically configured
        num_stacks: Number of stacks in the model
        num_blocks: Number of blocks per stack (can be a list for per-stack configuration)
        num_layers: Number of layers per block (can be a list for per-stack configuration)
        layer_width: Width of hidden layers (can be a list for per-stack configuration)
        thetas_dim: Dimension of thetas for basis expansion (can be a list)
        share_thetas: Whether to share thetas between backcast and forecast (can be a list)
        expansion_coefficient_dim: Dimension for basis expansion coefficients (can be a list)
        
        # Training parameters
        learning_rate: Learning rate for the optimizer
        loss_function: Loss function to use (mse, mae, huber, etc.)
        use_gradient_clipping: Whether to use gradient clipping
        gradient_clip_norm: Maximum gradient norm for clipping
        use_weighted_loss: Whether to apply temporal weighting to the loss
        use_learning_rate_schedule: Whether to use learning rate scheduling
        
        # Regularization parameters
        dropout_rate: Dropout rate for regularization (can be a list for per-stack)
        use_batch_norm: Whether to use batch normalization (can be a list for per-stack)
        kernel_regularizer_type: Type of kernel regularizer (l1, l2, l1_l2, None)
        kernel_regularizer_value: Value for kernel regularization
        activity_regularizer_value: Value for activity regularization
        
        # Advanced features
        use_attention: Whether to add attention mechanism to the model
        add_exogenous_variables: Whether to use exogenous variables in addition to target
        use_residual_connections: Whether to use residual connections in blocks
        use_advanced_activations: Whether to use advanced activations (PReLU, etc.)
        use_cyclic_learning_rate: Whether to use cyclic learning rate
        use_ensemble_outputs: Whether to use ensemble averaging for outputs
        ensemble_size: Number of ensemble outputs to average
        
        # Specific to price forecasting
        include_price_specific_stack: Whether to include a price-specific stack
        use_volume_attention: Whether to add volume-based attention for price forecasting
        normalize_input_windows: Whether to normalize each input window separately
        apply_price_scaling: Whether to apply logarithmic or percentage scaling to prices
        
    Returns:
        A compiled TensorFlow/Keras model
    """
    logger.info(f"Building NBEATS model with lookback={lookback}, horizon={horizon}, features={num_features}")
    
    # Determine stack types
    if stack_types is None:
        # Default stack configuration for price prediction
        if include_price_specific_stack:
            stack_types = [StackType.TREND, StackType.SEASONALITY, StackType.PRICE_SPECIFIC, StackType.GENERIC]
        else:
            stack_types = [StackType.TREND, StackType.SEASONALITY, StackType.GENERIC]
        # Limit to requested number of stacks
        stack_types = stack_types[:num_stacks]
    else:
        # Use provided stack types
        stack_types = stack_types[:num_stacks]
    
    # Convert single values to lists for per-stack configuration
    if isinstance(num_blocks, int):
        num_blocks = [num_blocks] * len(stack_types)
    if isinstance(num_layers, int):
        num_layers = [num_layers] * len(stack_types)
    if isinstance(layer_width, int):
        layer_width = [layer_width] * len(stack_types)
    if isinstance(thetas_dim, int):
        thetas_dim = [thetas_dim] * len(stack_types)
    if isinstance(share_thetas, bool):
        share_thetas = [share_thetas] * len(stack_types)
    if isinstance(expansion_coefficient_dim, int):
        expansion_coefficient_dim = [expansion_coefficient_dim] * len(stack_types)
    if isinstance(dropout_rate, float):
        dropout_rate = [dropout_rate] * len(stack_types)
    if isinstance(use_batch_norm, bool):
        use_batch_norm = [use_batch_norm] * len(stack_types)
    
    # Create kernel regularizer based on configuration
    kernel_regularizer = None
    if kernel_regularizer_type:
        if kernel_regularizer_type == 'l1':
            kernel_regularizer = regularizers.l1(kernel_regularizer_value)
        elif kernel_regularizer_type == 'l2':
            kernel_regularizer = regularizers.l2(kernel_regularizer_value)
        elif kernel_regularizer_type == 'l1_l2':
            kernel_regularizer = regularizers.l1_l2(
                l1=kernel_regularizer_value / 2,
                l2=kernel_regularizer_value / 2
            )
    
    # Activity regularizer
    activity_regularizer = None
    if activity_regularizer_value > 0:
        activity_regularizer = regularizers.l2(activity_regularizer_value)
    
    # Determine activations
    activation = 'prelu' if use_advanced_activations else 'relu'
    
    # --- MODEL CONSTRUCTION ---
    
    # Input layer
    inputs = layers.Input(shape=(lookback, num_features), name="input_sequence")
    
    # Optional input normalization
    if normalize_input_windows:
        # Apply per-window normalization to stabilize training on price data
        # Mean-normalize each window separately (important for price levels)
        norm_layer = layers.Lambda(
            lambda x: (x - tf.reduce_mean(x, axis=1, keepdims=True)) / 
                    (tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-10),
            name="window_normalization"
        )
        x = norm_layer(inputs)
    else:
        x = inputs
    
    # Optional price scaling for financial time series
    if apply_price_scaling:
        # Apply log1p transformation to handle different price scales
        price_index = 0  # Assuming first feature is the price
        
        def logarithmic_price_transform(tensor):
            price_channel = tensor[:, :, price_index:price_index+1]
            other_channels = tensor[:, :, price_index+1:]
            
            # Extract first price from each sequence for relative scaling
            first_price = tf.expand_dims(price_channel[:, 0, :], axis=1)
            
            # Apply relative log transformation
            transformed_price = tf.math.log(price_channel / (first_price + 1e-10) + 1.0)
            
            # Concatenate back
            return tf.concat([transformed_price, other_channels], axis=-1)
        
        x = layers.Lambda(logarithmic_price_transform, name="price_scaling")(x)
    
    # Optional attention mechanism for volume weighting
    if use_volume_attention and use_attention:
        # Assuming volume is the second feature (index 1)
        volume_index = 1
        
        def volume_attention_mechanism(tensor):
            # Extract volume channel and normalize it
            volume = tensor[:, :, volume_index:volume_index+1]
            normalized_volume = volume / (tf.reduce_max(volume, axis=1, keepdims=True) + 1e-10)
            
            # Use volume as attention weights
            weighted_input = tensor * normalized_volume
            return weighted_input
        
        x = layers.Lambda(volume_attention_mechanism, name="volume_attention")(x)
    
    # General self-attention mechanism if requested
    if use_attention and not use_volume_attention:
        # Multi-head self-attention layer
        x = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=num_features,
            dropout=dropout_rate[0]
        )(x, x)
    
    # Reshape for initial 1D processing
    # Each NBEATS block will internally handle reshaping
    backcast = layers.Flatten()(x)
    
    # Create stacks with different specializations (trend, seasonality, etc.)
    forecast_outputs = []
    
    for i, stack_type in enumerate(stack_types):
        # Calculate dynamic expansion coefficient dimensions based on stack type
        if stack_type == StackType.TREND:
            exp_coef_dim = min(expansion_coefficient_dim[i], 5)  # Trend usually needs fewer dimensions
        elif stack_type == StackType.SEASONALITY:
            exp_coef_dim = min(expansion_coefficient_dim[i], 10)  # Seasonality may need more harmonics
        elif stack_type == StackType.PRICE_SPECIFIC:
            # Price-specific needs more components for market patterns
            exp_coef_dim = min(expansion_coefficient_dim[i], 15)
        else:
            exp_coef_dim = expansion_coefficient_dim[i]
        
        # Create stack
        stack = NBeatsStack(
            stack_type=stack_type,
            num_blocks=num_blocks[i],
            thetas_dim=thetas_dim[i],
            share_thetas=share_thetas[i],
            num_layers=num_layers[i],
            layer_width=layer_width[i],
            activation=activation,
            dropout_rate=dropout_rate[i],
            use_batch_norm=use_batch_norm[i],
            kernel_regularizer=kernel_regularizer,
            basis_function_type="default",
            expansion_coefficient_dim=exp_coef_dim,
            backcast_length=lookback * num_features,
            forecast_length=horizon,
            name=f"{stack_type}_stack_{i}"
        )
        
        # Apply stack
        residual_backcast, stack_forecast = stack(backcast)
        
        # Apply residual connections for the stack output
        if use_residual_connections:
            backcast = residual_backcast
        else:
            # Without residual connections, stacks operate independently
            # This simplifies to a more traditional ensemble approach
            pass
        
        # Collect forecasts for later combination
        forecast_outputs.append(stack_forecast)
    
    # Combine forecasts from all stacks
    if use_ensemble_outputs:
        # Create ensemble of forecasts by averaging multiple output heads
        ensemble_forecasts = []
        
        for i in range(ensemble_size):
            ensemble_head = layers.Dense(
                horizon,
                activation="linear",
                name=f"ensemble_head_{i}"
            )(layers.concatenate(forecast_outputs, axis=-1))
            ensemble_forecasts.append(ensemble_head)
        
        # Average ensemble outputs
        final_forecast = layers.Average(name="ensemble_average")(ensemble_forecasts)
    else:
        # Simple averaging of stack outputs
        if len(forecast_outputs) > 1:
            final_forecast = layers.Average(name="forecast_average")(forecast_outputs)
        else:
            final_forecast = forecast_outputs[0]
    
    # Compile model
    model = models.Model(inputs=inputs, outputs=final_forecast)
    
    # Configure optimizer
    if use_learning_rate_schedule:
        # Learning rate schedule for better convergence
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
    elif use_cyclic_learning_rate:
        # Cyclic learning rate helps escaping local minima
        def cyclic_lr(step):
            # Triangle wave function
            cycle = step // (2 * 1000)
            x = step - (2 * 1000) * cycle
            x = x / 1000
            x = (x - 1.0) if x > 1.0 else x
            return learning_rate * (1 + x) if x < 0.5 else learning_rate * (2.0 - x)
            
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(cyclic_lr)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Apply gradient clipping if requested
    if use_gradient_clipping:
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=gradient_clip_norm
        )
    
    # Configure loss function with optional temporal weighting
    if use_weighted_loss:
        # Create weights that emphasize recent observations more
        def temporal_weighted_loss(y_true, y_pred):
            # Create weights that increase for later timesteps
            # This gives more importance to predictions that are closer in time
            weights = tf.range(1.0, tf.cast(tf.shape(y_true)[1] + 1, tf.float32))
            weights = weights / tf.reduce_sum(weights)
            
            # Compute squared error
            squared_error = tf.square(y_true - y_pred)
            
            # Apply weights along time dimension
            weighted_error = squared_error * tf.reshape(weights, [1, -1])
            
            # Return mean over all dimensions
            return tf.reduce_mean(weighted_error)
        
        loss = temporal_weighted_loss
    else:
        loss = loss_function
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)
    
    logger.info(f"NBEATS model built with {len(stack_types)} stacks: {stack_types}")
    return model


def get_nbeats_hyperparameter_ranges():
    """
    Define hyperparameter ranges for Optuna optimization.
    These cover most aspects of the N-BEATS model architecture with wider ranges.
    """
    return {
        # Core architecture parameters - expanded ranges
        "lookback": {"type": "int", "low": 5, "high": 120, "step": 1},
        "num_stacks": {"type": "int", "low": 1, "high": 6, "step": 1},
        "num_blocks": {"type": "int", "low": 1, "high": 8, "step": 1},
        "num_layers": {"type": "int", "low": 1, "high": 8, "step": 1},
        "layer_width": {"type": "int", "low": 16, "high": 1024, "log": True},
        "thetas_dim": {"type": "int", "low": 3, "high": 24, "step": 1},
        "expansion_coefficient_dim": {"type": "int", "low": 2, "high": 20, "step": 1},
        
        # Training parameters - wider ranges
        "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "use_gradient_clipping": {"type": "categorical", "choices": [True, False]},
        "gradient_clip_norm": {"type": "float", "low": 0.1, "high": 20.0, "log": True},
        "use_weighted_loss": {"type": "categorical", "choices": [True, False]},
        "use_learning_rate_schedule": {"type": "categorical", "choices": [True, False]},
        
        # Regularization - expanded options
        "dropout_rate": {"type": "float", "low": 0.0, "high": 0.7, "step": 0.05},
        "use_batch_norm": {"type": "categorical", "choices": [True, False]},
        "kernel_regularizer_type": {
            "type": "categorical", 
            "choices": [None, "l1", "l2", "l1_l2"]
        },
        "kernel_regularizer_value": {"type": "float", "low": 1e-8, "high": 1e-1, "log": True},
        
        # Advanced features - more choices
        "use_attention": {"type": "categorical", "choices": [True, False]},
        "add_exogenous_variables": {"type": "categorical", "choices": [True, False]},
        "use_residual_connections": {"type": "categorical", "choices": [True, False]},
        "use_advanced_activations": {"type": "categorical", "choices": [True, False]},
        "use_ensemble_outputs": {"type": "categorical", "choices": [True, False]},
        "ensemble_size": {"type": "int", "low": 2, "high": 7, "step": 1},
        
        # Price prediction specific - additional options
        "include_price_specific_stack": {"type": "categorical", "choices": [True, False]},
        "use_volume_attention": {"type": "categorical", "choices": [True, False]},
        "normalize_input_windows": {"type": "categorical", "choices": [True, False]},
        "apply_price_scaling": {"type": "categorical", "choices": [True, False]},
        
        # New advanced parameters
        "use_cyclic_learning_rate": {"type": "categorical", "choices": [True, False]},
        "share_thetas_across_blocks": {"type": "categorical", "choices": [True, False]},
        "activation_type": {"type": "categorical", "choices": ["relu", "elu", "selu", "gelu", "swish"]},
        "seasonality_fourier_terms": {"type": "int", "low": 1, "high": 12, "step": 1},
    }


def suggest_nbeats_hyperparameters(trial):
    """
    Suggest hyperparameters for N-BEATS model during Optuna optimization.
    This function can be used directly in hyperparameter optimization routines.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameter values
    """
    param_ranges = get_nbeats_hyperparameter_ranges()
    params = {}
    
    for param_name, param_config in param_ranges.items():
        param_type = param_config["type"]
        
        if param_type == "int":
            low = param_config["low"]
            high = param_config["high"]
            step = param_config.get("step", 1)
            log = param_config.get("log", False)
            params[param_name] = trial.suggest_int(param_name, low, high, step=step, log=log)
            
        elif param_type == "float":
            low = param_config["low"]
            high = param_config["high"]
            log = param_config.get("log", False)
            step = param_config.get("step", None)
            if step:
                params[param_name] = trial.suggest_float(param_name, low, high, step=step, log=log)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)
                
        elif param_type == "categorical":
            choices = param_config["choices"]
            params[param_name] = trial.suggest_categorical(param_name, choices)
            
    return params