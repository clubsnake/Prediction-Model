# model_visualization.py
"""
Provides visualization routines for neural networks, training history, 
feature importance, and ensemble contributions.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf

def visualize_neural_network(model: tf.keras.Model, feature_names: list, max_neurons_per_layer: int = 10):
    """
    Visualize the architecture of Keras-based neural networks (e.g. LSTM, RNN).

    :param model: A compiled Keras model.
    :param feature_names: Names of the input features.
    :param max_neurons_per_layer: Maximum number of neurons to display per layer.
    :return: A matplotlib figure object with the network diagram.
    """
    layers_info = []
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        if hasattr(layer, 'units'):
            units = layer.units
        elif hasattr(layer, 'output_shape') and layer.output_shape is not None:
            # Fallback for layers that don't have 'units' but have output_shape
            shape = layer.output_shape
            if isinstance(shape, list) and len(shape) > 0:
                shape = shape[0]
            units = shape[-1] if shape and len(shape) > 1 else 0
        else:
            units = 0
        
        layers_info.append({
            'layer_index': i,
            'layer_type': layer_type,
            'units': min(units, max_neurons_per_layer)
        })
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    n_layers = len(layers_info)
    layer_spacing = 1.0 / (n_layers + 1)
    
    for i, layer_info in enumerate(layers_info):
        layer_x = (i + 1) * layer_spacing
        units = layer_info['units']
        
        if units > 0:
            neuron_spacing = 1.0 / (units + 1)
            for j in range(units):
                neuron_y = (j + 1) * neuron_spacing
                circle = plt.Circle((layer_x, neuron_y), 0.02, color='blue', fill=True)
                ax.add_patch(circle)
                
                # Connect to previous layer
                if i > 0 and layers_info[i-1]['units'] > 0:
                    prev_units = layers_info[i-1]['units']
                    prev_neuron_spacing = 1.0 / (prev_units + 1)
                    if units <= 5 and prev_units <= 5:
                        for k in range(prev_units):
                            prev_y = (k + 1) * prev_neuron_spacing
                            plt.plot([layer_x - layer_spacing, layer_x], 
                                     [prev_y, neuron_y], 
                                     'k-', alpha=0.1)
        
        plt.text(layer_x, 0.02, f"{layer_info['layer_type']} ({layer_info['units']})", 
                 ha='center', va='bottom', rotation=90, fontsize=8)
    
    # Show feature names for input layer
    if feature_names and layers_info:
        if layers_info[0]['units'] > 0:
            units = min(len(feature_names), layers_info[0]['units'])
            neuron_spacing = 1.0 / (units + 1)
            for j in range(units):
                neuron_y = (j + 1) * neuron_spacing
                feature_name = feature_names[j] if j < len(feature_names) else ""
                plt.text(layer_spacing - 0.05, neuron_y, feature_name, 
                         ha='right', va='center', fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title(f"Neural Network Architecture: {model.__class__.__name__}")
    
    return fig

def plot_training_history(history: dict):
    """
    Plot model training history using Plotly.

    :param history: A dict-like object with keys like 'loss', 'val_loss'.
    :return: A plotly.graph_objs.Figure object.
    """
    if not history:
        return None
    
    fig = go.Figure()
    if 'loss' in history:
        fig.add_trace(go.Scatter(y=history['loss'], name='Training Loss'))
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
    
    fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
    return fig

def plot_feature_importance(importance_scores, feature_names, model_type='unknown'):
    """
    Plot feature importance scores as a bar chart.

    :param importance_scores: Iterable of feature importances.
    :param feature_names: Corresponding list of feature names.
    :param model_type: String indicating the model type (for display).
    :return: A plotly.graph_objs.Figure object.
    """
    if importance_scores is None or len(importance_scores) == 0:
        return None
        
    fig = go.Figure([go.Bar(
        x=feature_names,
        y=importance_scores,
        text=np.round(importance_scores, 3),
        textposition='auto',
    )])
    
    fig.update_layout(
        title=f'Feature Importance ({model_type})',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        showlegend=False
    )
    return fig

def plot_ensemble_contribution(ensemble_weights: dict, submodel_metrics: dict):
    """
    Visualize the contribution of each model in the ensemble.

    :param ensemble_weights: Dict of {model_name: weight}.
    :param submodel_metrics: Dict of {model_name: metric}, typically RMSE or similar.
    :return: A plotly.graph_objs.Figure object.
    """
    models = list(ensemble_weights.keys())
    weights = list(ensemble_weights.values())
    
    # Calculate normalized contribution (weight × inverse of metric)
    if submodel_metrics:
        performances = [submodel_metrics.get(m, 0) for m in models]
        inv_perf = [1/p if p > 0 else 0 for p in performances]
        total = sum(inv_perf)
        norm_perf = [ip/total if total > 0 else 0 for ip in inv_perf]
        contribution = [w * np for w, np in zip(weights, norm_perf)]
    else:
        contribution = weights
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Model Weights', 'Effective Contribution'),
                        specs=[[{"type": "pie"}, {"type": "pie"}]])
    
    fig.add_trace(go.Pie(
        labels=models,
        values=weights,
        name="Weights",
        hole=.3
    ), row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=models,
        values=contribution,
        name="Contribution",
        hole=.3
    ), row=1, col=2)
    
    fig.update_layout(height=400, width=800, title_text="Ensemble Model Analysis")
    
    return fig

def plot_prediction_errors(y_true, y_pred, forecast_dates=None):
    """
    Create error visualization for predictions.

    :param y_true: True values array.
    :param y_pred: Predicted values array.
    :param forecast_dates: (Optional) array of dates corresponding to predictions.
    :return: A plotly.graph_objs.Figure object with error timeseries & histogram.
    """
    errors = y_pred - y_true
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    if forecast_dates is not None and len(forecast_dates) == len(errors):
        df = pd.DataFrame({'Date': forecast_dates, 'Error': errors, 'AbsError': np.abs(errors)})
        x_values = df['Date']
    else:
        df = pd.DataFrame({'Index': range(len(errors)), 'Error': errors, 'AbsError': np.abs(errors)})
        x_values = df['Index']
    
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=(f'Prediction Errors (RMSE: {rmse:.4f})', 'Error Distribution'),
                       vertical_spacing=0.15)
    
    # Error time series
    fig.add_trace(go.Scatter(x=x_values, y=df['Error'], mode='lines+markers', name='Error'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x_values, y=[0]*len(df), mode='lines',
                             line=dict(color='red', dash='dash'), name='Zero Line'),
                  row=1, col=1)
    
    # Error histogram
    fig.add_trace(go.Histogram(x=df['Error'], nbinsx=20, name='Error Distribution'),
                  row=2, col=1)
    
    fig.update_layout(height=600, width=900, showlegend=False)
    return fig

def visualize_attention(model: tf.keras.Model, X_sample: np.ndarray, layer_name="attention"):
    """
    Attempt to visualize attention weights for a model with an attention layer.

    :param model: The Keras model that includes an attention layer.
    :param X_sample: A sample batch of data to pass through the model.
    :param layer_name: Name or partial name for the attention layer.
    :return: A matplotlib figure with a heatmap of attention weights, or None if not found.
    """
    attention_layer = None
    for layer in model.layers:
        if layer_name.lower() in layer.name.lower():
            attention_layer = layer
            break
    
    if attention_layer is None:
        return None
    
    attention_model = Model(inputs=model.input, outputs=attention_layer.output)
    attention_weights = attention_model.predict(X_sample)
    
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights[0], cmap='viridis')
    plt.title("Attention Weights Visualization")
    plt.xlabel("Timesteps")
    plt.ylabel("Attention Weights")
    return fig

def create_model_visualization_tab():
    """
    Create a Streamlit tab for model visualization. 
    This can be integrated into the main dashboard.
    """
    st.header("Model Visualization")
    viz_tabs = st.tabs(["Architecture", "Training", "Feature Importance", "Predictions", "Ensemble"])
    
    with viz_tabs[0]:
        st.subheader("Neural Network Architecture")
        # Implementation example
        st.info("To see architecture, store your model in session_state['current_models'].")
    
    with viz_tabs[1]:
        st.subheader("Training Progress")
        st.info("Display training history if session_state has 'training_history'.")
    
    with viz_tabs[2]:
        st.subheader("Feature Importance")
        st.info("Plot feature importance if you have a tree-based model in session_state.")
    
    with viz_tabs[3]:
        st.subheader("Prediction Analysis")
        st.info("Plot prediction errors if 'actual_values' and 'predicted_values' exist in session_state.")
    
    with viz_tabs[4]:
        st.subheader("Ensemble Model Analysis")
        st.info("Plot ensemble contribution if 'ensemble_weights' and 'submodel_metrics' exist in session_state.")
