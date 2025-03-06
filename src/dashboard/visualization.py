# unified_visualization.py
"""
Unified visualization module for AI price prediction models.
Combines functionality from visualization.py and model_visualization.py.
Supports both Matplotlib and Plotly visualizations.
"""

import io
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore

# Set up logging
logger = logging.getLogger(__name__)

# Check configuration - use sensible defaults if config module not found
try:
    from config.config_loader import SHOW_PREDICTION_PLOTS, SHOW_TRAINING_HISTORY, SHOW_WEIGHT_HISTOGRAMS
except ImportError:
    logger.warning("Config not found, using default visualization settings")
    SHOW_PREDICTION_PLOTS = True
    SHOW_TRAINING_HISTORY = True
    SHOW_WEIGHT_HISTOGRAMS = True


# ===== MATPLOTLIB VISUALIZATIONS =====

def visualize_training_history_mpl(history, figsize=(12, 5)):
    """
    Create training/validation loss plot using Matplotlib.

    Args:
        history: Keras history object or dictionary with keys 'loss', 'val_loss'
        figsize: Figure size as (width, height) tuple
    
    Returns:
        matplotlib figure that can be displayed with plt.show() or st.pyplot()
    """
    if not SHOW_TRAINING_HISTORY:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(history, dict):
        ax.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss')
    else:
        # Assume it's a Keras History object
        ax.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='Validation Loss')

    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig


def visualize_weight_histograms_mpl(model):
    """
    Create histograms of model weights if SHOW_WEIGHT_HISTOGRAMS is enabled.

    Args:
        model: Keras model object

    Returns:
        list of matplotlib figures that can be displayed with plt.show() or st.pyplot()
    """
    if not SHOW_WEIGHT_HISTOGRAMS:
        return []

    figures = []
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if not weights:
            continue
            
        # Create a histogram for each weight matrix in the layer
        for j, w in enumerate(weights):
            if w.size > 0:  # Only plot if weights exist
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(w.flatten(), bins=50)
                ax.set_title(f"Layer {i} ({layer.name}) - Weights {j} Histogram")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Count")
                ax.grid(True, linestyle='--', alpha=0.6)
                figures.append(fig)

    return figures


def visualize_predictions_mpl(y_true, y_pred, sample_idx=0, figsize=(12, 5)):
    """
    Create plot of predicted vs. actual series for a single sample using Matplotlib.

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        sample_idx: Index of the sample to visualize
        figsize: Figure size as (width, height) tuple
    
    Returns:
        matplotlib figure that can be displayed with plt.show() or st.pyplot()
    """
    if not SHOW_PREDICTION_PLOTS:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different shapes of input data
    if isinstance(y_true, (list, np.ndarray)) and len(np.shape(y_true)) > 1:
        # Get the specific sample
        sample_true = y_true[sample_idx] if sample_idx < len(y_true) else y_true[0]
    else:
        # Single series
        sample_true = y_true
        
    if isinstance(y_pred, (list, np.ndarray)) and len(np.shape(y_pred)) > 1:
        # Get the specific sample
        sample_pred = y_pred[sample_idx] if sample_idx < len(y_pred) else y_pred[0]
    else:
        # Single series
        sample_pred = y_pred
    
    ax.plot(sample_true, label='True', marker='o')
    ax.plot(sample_pred, label='Predicted', marker='x')
    
    ax.set_title(f"Predictions vs True Values (Sample {sample_idx})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig


def plot_combined_validation_and_forecast_mpl(
    true_prices,
    predicted_prices,
    forecast_prices,
    dates,
    forecast_dates,
    validation_rmse,
    validation_mape,
    figsize=(12, 6)
):
    """
    Combine validation and forecast data into a single line plot using Matplotlib.

    Args:
        true_prices: Array of true prices
        predicted_prices: Array of predicted prices
        forecast_prices: Array of forecasted prices
        dates: Array of dates for validation period
        forecast_dates: Array of dates for forecast period
        validation_rmse: RMSE value for validation period
        validation_mape: MAPE value for validation period
        figsize: Figure size as (width, height) tuple
    
    Returns:
        matplotlib figure that can be displayed with plt.show() or st.pyplot()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot validation data
    ax.plot(dates, true_prices, marker='o', linestyle='-', label='True (Validation)')
    ax.plot(dates, predicted_prices, marker='x', linestyle='--', label='Predicted (Validation)')
    
    # Plot forecast data
    ax.plot(forecast_dates, forecast_prices, linestyle='dashed', label='Forecast')
    
    # Styling
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Combined Validation and Future Forecast')
    ax.legend()
    
    # Add error metrics
    error_text = f"Validation RMSE: {validation_rmse:.4f}\nValidation MAPE: {validation_mape:.2f}%"
    ax.text(
        dates[0], 
        max(true_prices) * 0.95,
        error_text,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    ax.grid(True)
    fig.tight_layout()

    return fig


def visualize_neural_network_mpl(model, feature_names, max_neurons_per_layer=10, figsize=(12, 8)):
    """
    Visualize the architecture of Keras-based neural networks using Matplotlib.

    Args:
        model: A compiled Keras model
        feature_names: Names of the input features
        max_neurons_per_layer: Maximum number of neurons to display per layer
        figsize: Figure size as (width, height) tuple
        
    Returns:
        A matplotlib figure object with the network diagram
    """
    layers_info = []
    for i, layer in enumerate(model.layers):
        # Get layer information
        layer_type = layer.__class__.__name__
        
        # Get number of neurons (units or filters)
        num_neurons = 0
        if hasattr(layer, 'units'):
            num_neurons = layer.units
        elif hasattr(layer, 'filters'):
            num_neurons = layer.filters
        elif hasattr(layer, 'output_shape'):
            if isinstance(layer.output_shape, tuple):
                num_neurons = layer.output_shape[-1]
            else:
                # Handle multiple outputs
                if hasattr(layer.output_shape, '__iter__'):
                    num_neurons = layer.output_shape[0][-1]
        
        # Limit number of neurons to display
        display_neurons = min(num_neurons, max_neurons_per_layer)
        
        layers_info.append({
            'name': layer.name,
            'type': layer_type,
            'neurons': display_neurons,
            'total_neurons': num_neurons
        })

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    n_layers = len(layers_info)
    layer_spacing = 1.0 / (n_layers + 1)

    for i, layer_info in enumerate(layers_info):
        layer_x = (i + 1) * layer_spacing
        
        # Draw neurons
        num_neurons = layer_info['neurons']
        if num_neurons > 0:
            neuron_spacing = 0.8 / (num_neurons + 1)
            
            for j in range(num_neurons):
                neuron_y = 0.1 + (j + 1) * neuron_spacing
                circle = plt.Circle((layer_x, neuron_y), 0.02, color='skyblue', fill=True)
                ax.add_patch(circle)
                
                # Draw connections to previous layer
                if i > 0:
                    prev_neurons = layers_info[i-1]['neurons']
                    prev_x = layer_x - layer_spacing
                    
                    prev_neuron_spacing = 0.8 / (prev_neurons + 1)
                    for k in range(prev_neurons):
                        prev_y = 0.1 + (k + 1) * prev_neuron_spacing
                        ax.plot([prev_x, layer_x], [prev_y, neuron_y], 'gray', alpha=0.3)
            
            # Show layer name and type
            if layer_info['total_neurons'] > layer_info['neurons']:
                layer_name = f"{layer_info['name']} ({layer_info['type']})\n{layer_info['total_neurons']} neurons"
            else:
                layer_name = f"{layer_info['name']} ({layer_info['type']})"
                
            ax.text(layer_x, 0.02, layer_name, 
                   horizontalalignment='center', verticalalignment='center')

    # Show feature names for input layer
    if feature_names and layers_info:
        # We only display up to max_neurons_per_layer, so limit feature names as well
        display_features = feature_names[:layers_info[0]['neurons']]
        neuron_spacing = 0.8 / (layers_info[0]['neurons'] + 1)
        
        for j, feature in enumerate(display_features):
            if j < layers_info[0]['neurons']:
                neuron_y = 0.1 + (j + 1) * neuron_spacing
                ax.text(layer_spacing - 0.05, neuron_y, feature, 
                       horizontalalignment='right', verticalalignment='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title(f"Neural Network Architecture: {model.__class__.__name__}")

    return fig


# ===== PLOTLY VISUALIZATIONS =====

def plot_training_history_plotly(history):
    """
    Plot model training history using Plotly.

    Args:
        history: Keras history object or dictionary with keys 'loss', 'val_loss'
        
    Returns:
        A plotly.graph_objs.Figure object
    """
    if not SHOW_TRAINING_HISTORY:
        return None

    fig = go.Figure()
    
    # Extract history data
    if isinstance(history, dict):
        history_dict = history
    else:
        # Assume it's a Keras History object
        history_dict = history.history
    
    # Add training loss
    epochs = range(1, len(history_dict['loss']) + 1)
    fig.add_trace(go.Scatter(
        x=epochs, y=history_dict['loss'],
        mode='lines+markers',
        name='Training Loss'
    ))
    
    # Add validation loss if available
    if 'val_loss' in history_dict:
        fig.add_trace(go.Scatter(
            x=epochs, y=history_dict['val_loss'],
            mode='lines+markers',
            name='Validation Loss'
        ))
    
    # Layout
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def visualize_weight_histograms_plotly(model):
    """
    Create histograms of model weights using Plotly.
    
    Args:
        model: Keras model object
        
    Returns:
        List of plotly.graph_objs.Figure objects
    """
    if not SHOW_WEIGHT_HISTOGRAMS:
        return []
    
    figures = []
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if not weights:
            continue
        
        # Create a histogram for each weight matrix in the layer
        for j, w in enumerate(weights):
            if w.size > 0:  # Only plot if weights exist
                fig = px.histogram(
                    x=w.flatten(), 
                    nbins=50,
                    title=f"Layer {i} ({layer.name}) - Weights {j} Histogram"
                )
                
                fig.update_layout(
                    xaxis_title="Weight Value",
                    yaxis_title="Count",
                    template='plotly_white'
                )
                
                figures.append(fig)
    
    return figures


def visualize_predictions_plotly(y_true, y_pred, sample_idx=0):
    """
    Create plot of predicted vs. actual series for a single sample using Plotly.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        sample_idx: Index of the sample to visualize
        
    Returns:
        A plotly.graph_objs.Figure object
    """
    if not SHOW_PREDICTION_PLOTS:
        return None
    
    # Handle different shapes of input data
    if isinstance(y_true, (list, np.ndarray)) and len(np.shape(y_true)) > 1:
        # Get the specific sample
        sample_true = y_true[sample_idx] if sample_idx < len(y_true) else y_true[0]
    else:
        # Single series
        sample_true = y_true
        
    if isinstance(y_pred, (list, np.ndarray)) and len(np.shape(y_pred)) > 1:
        # Get the specific sample
        sample_pred = y_pred[sample_idx] if sample_idx < len(y_pred) else y_pred[0]
    else:
        # Single series
        sample_pred = y_pred
    
    # Create time steps
    time_steps = list(range(len(sample_true)))
    
    # Create figure
    fig = go.Figure()
    
    # Add true values
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=sample_true,
        mode='lines+markers',
        name='True',
        marker=dict(symbol='circle')
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=sample_pred,
        mode='lines+markers',
        name='Predicted',
        marker=dict(symbol='cross')
    ))
    
    # Layout
    fig.update_layout(
        title=f"Predictions vs True Values (Sample {sample_idx})",
        xaxis_title="Time Step",
        yaxis_title="Value",
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_combined_validation_and_forecast_plotly(
    true_prices,
    predicted_prices,
    forecast_prices,
    dates,
    forecast_dates,
    validation_rmse,
    validation_mape
):
    """
    Combine validation and forecast data into a single line plot using Plotly.
    
    Args:
        true_prices: Array of true prices
        predicted_prices: Array of predicted prices
        forecast_prices: Array of forecasted prices
        dates: Array of dates for validation period
        forecast_dates: Array of dates for forecast period
        validation_rmse: RMSE value for validation period
        validation_mape: MAPE value for validation period
        
    Returns:
        A plotly.graph_objs.Figure object
    """
    fig = go.Figure()
    
    # Add true prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=true_prices,
        mode='lines+markers',
        name='True (Validation)',
        marker=dict(symbol='circle')
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted_prices,
        mode='lines+markers',
        name='Predicted (Validation)',
        line=dict(dash='dash'),
        marker=dict(symbol='cross')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode='lines',
        name='Forecast',
        line=dict(dash='dash', color='green')
    ))
    
    # Add vertical line separating validation and forecast
    if len(dates) > 0 and len(forecast_dates) > 0:
        last_validation_date = dates[-1]
        fig.add_vline(
            x=last_validation_date, 
            line_width=1, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top right"
        )
    
    # Add error metrics as annotations
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Validation RMSE: {validation_rmse:.4f}<br>Validation MAPE: {validation_mape:.2f}%",
        showarrow=False,
        font=dict(size=14),
        bordercolor='gray',
        borderwidth=1,
        borderpad=4,
        bgcolor='white',
        opacity=0.8
    )
    
    # Layout
    fig.update_layout(
        title='Combined Validation and Future Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def visualize_neural_network_plotly(model, feature_names=None, max_neurons_per_layer=10):
    """
    Visualize neural network architecture using Plotly.
    
    Args:
        model: A compiled Keras model
        feature_names: Names of the input features
        max_neurons_per_layer: Maximum number of neurons to display per layer
        
    Returns:
        A plotly.graph_objs.Figure object
    """
    # Extract layer information
    layers_info = []
    for i, layer in enumerate(model.layers):
        # Get layer information
        layer_type = layer.__class__.__name__
        
        # Get number of neurons (units or filters)
        num_neurons = 0
        if hasattr(layer, 'units'):
            num_neurons = layer.units
        elif hasattr(layer, 'filters'):
            num_neurons = layer.filters
        elif hasattr(layer, 'output_shape'):
            if isinstance(layer.output_shape, tuple):
                num_neurons = layer.output_shape[-1]
            else:
                # Handle multiple outputs
                if hasattr(layer.output_shape, '__iter__'):
                    num_neurons = layer.output_shape[0][-1]
        
        # Limit number of neurons to display
        display_neurons = min(num_neurons, max_neurons_per_layer)
        
        layers_info.append({
            'name': layer.name,
            'type': layer_type,
            'neurons': display_neurons,
            'total_neurons': num_neurons,
            'layer_idx': i
        })
    
    # Create nodes and edges for network visualization
    nodes = []
    edges = []
    
    # Set visualization parameters
    n_layers = len(layers_info)
    max_neurons = max([layer['neurons'] for layer in layers_info]) if layers_info else 1
    
    # Keep track of node indices
    node_count = 0
    layer_start_indices = []
    
    # Create nodes for each layer
    for i, layer_info in enumerate(layers_info):
        layer_start_indices.append(node_count)
        
        # Create nodes for neurons in this layer
        for j in range(layer_info['neurons']):
            node_name = f"{layer_info['name']}_{j}"
            if i == 0 and feature_names and j < len(feature_names):
                node_label = feature_names[j]
            else:
                node_label = ""
                
            nodes.append({
                'id': node_count,
                'label': node_label,
                'layer': i,
                'position': j,
                'title': f"{layer_info['name']} (Neuron {j})",
                'color': 'lightblue',
                'size': 10  # Size can be adjusted for visualization
            })
            node_count += 1
        
        # Add display for additional neurons that aren't shown
        if layer_info['total_neurons'] > layer_info['neurons']:
            nodes.append({
                'id': node_count,
                'label': f"+{layer_info['total_neurons'] - layer_info['neurons']}",
                'layer': i,
                'position': layer_info['neurons'] + 0.5,
                'title': f"{layer_info['total_neurons'] - layer_info['neurons']} more neurons",
                'color': 'lightgray',
                'size': 10
            })
            node_count += 1
    
    # Create edges between layers
    for i in range(1, len(layers_info)):
        prev_layer_start = layer_start_indices[i-1]
        prev_layer_neurons = layers_info[i-1]['neurons']
        
        curr_layer_start = layer_start_indices[i]
        curr_layer_neurons = layers_info[i]['neurons']
        
        # Connect each neuron in current layer to each neuron in previous layer
        for j in range(curr_layer_neurons):
            curr_node_idx = curr_layer_start + j
            
            for k in range(prev_layer_neurons):
                prev_node_idx = prev_layer_start + k
                
                edges.append({
                    'from': prev_node_idx,
                    'to': curr_node_idx,
                    'width': 1,
                    'color': 'rgba(200,200,200,0.2)'
                })
    
    # Calculate node positions
    x_positions = []
    y_positions = []
    node_texts = []
    node_colors = []
    node_sizes = []
    edge_x = []
    edge_y = []
    
    # Calculate positions for layer visualization
    layer_spacing = 1.0 / (n_layers + 1)
    
    for node in nodes:
        layer = node['layer']
        position = node['position']
        
        x = (layer + 1) * layer_spacing
        
        # Calculate y position of neuron
        neurons_in_layer = layers_info[layer]['neurons']
        if neurons_in_layer > 0:
            neuron_spacing = 0.8 / (neurons_in_layer + 1)
            y = 0.1 + (position + 1) * neuron_spacing
        else:
            y = 0.5  # Default position
            
        x_positions.append(x)
        y_positions.append(y)
        node_texts.append(node['label'])
        node_colors.append(node['color'])
        node_sizes.append(node['size'])
    
    # Create edges (connections between neurons)
    for edge in edges:
        from_idx = edge['from']
        to_idx = edge['to']
        
        # Line from source to target
        edge_x.extend([x_positions[from_idx], x_positions[to_idx], None])
        edge_y.extend([y_positions[from_idx], y_positions[to_idx], None])
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='rgba(180,180,180,0.3)', width=1),
        hoverinfo='none'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='markers+text',
        text=node_texts,
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='rgba(50,50,50,0.8)')
        ),
        hovertemplate="%{customdata}",
        customdata=[node['title'] for node in nodes]
    ))
    
    # Add layer labels
    for i, layer_info in enumerate(layers_info):
        fig.add_annotation(
            x=(i + 1) * layer_spacing,
            y=0.05,
            text=f"{layer_info['name']}<br>({layer_info['type']})",
            showarrow=False,
            xshift=0,
            yshift=0,
            align='center',
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title=f"Neural Network Architecture: {model.__class__.__name__}",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=20, l=5, r=5, t=40),
        height=600,
        template='plotly_white'
    )
    
    return fig


# ===== UTILITY FUNCTIONS =====

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to base64 string for embedding in HTML.
    
    Args:
        fig: matplotlib figure object
        
    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def get_model_summary(model):
    """
    Generate an improved string representation of a Keras model.
    
    Args:
        model: Keras model
        
    Returns:
        String with formatted model summary
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_info = "\n".join(stringlist)
    
    return model_info


def normalize_series(series):
    """
    Min-max normalize a series to [0,1].
    
    Args:
        series: NumPy array or list
        
    Returns:
        Normalized array
    """
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val - min_val > 0:
        return (series - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(series)


def calculate_errors(y_true, y_pred):
    """
    Calculate common error metrics between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of error metrics
    """
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate MAPE with handling for zero values
    mask = y_true != 0
    if np.any(mask):
        mape = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        mape = np.inf
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def create_sequential_visualization(model, input_shape, output_shape):
    """
    Create a visualization of model data flow from input to output.
    
    Args:
        model: Keras model
        input_shape: Shape of input data
        output_shape: Shape of output data
        
    Returns:
        Dict with model visualization information
    """
    layers = []
    
    # Add input shape
    layers.append({
        'name': 'Input',
        'type': 'Input',
        'shape': str(input_shape)
    })
    
    # Add each layer
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__
        }
        
        # Get output shape
        if hasattr(layer, 'output_shape'):
            if isinstance(layer.output_shape, tuple):
                layer_info['shape'] = str(layer.output_shape)
            else:
                layer_info['shape'] = str(layer.output_shape[0])
        
        # Get number of parameters
        layer_info['params'] = "{:,}".format(layer.count_params())
        
        # Get activation if available
        if hasattr(layer, 'activation') and layer.activation is not None:
            if hasattr(layer.activation, '__name__'):
                layer_info['activation'] = layer.activation.__name__
            else:
                layer_info['activation'] = str(layer.activation)
        
        layers.append(layer_info)
    
    # Add output shape
    layers.append({
        'name': 'Output',
        'type': 'Output',
        'shape': str(output_shape)
    })
    
    return {
        'layers': layers,
        'total_params': "{:,}".format(model.count_params())
    }
