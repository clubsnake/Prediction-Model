import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

class ModelVisualizationDashboard:
    """Comprehensive dashboard for visualizing ensemble model performance and learning"""
    
    def __init__(self, ensemble_weighter, model_directory="model_weights"):
        """
        Initialize visualization dashboard
        
        Args:
            ensemble_weighter: Instance of AdvancedEnsembleWeighter
            model_directory: Directory where model weights are stored
        """
        self.weighter = ensemble_weighter
        self.model_directory = model_directory
        
        # Create directory if it doesn't exist
        os.makedirs(model_directory, exist_ok=True)
        
        # Model type colors for consistent visualization
        self.model_colors = {
            'lstm': '#1f77b4',  # blue
            'rnn': '#ff7f0e',   # orange
            'xgboost': '#2ca02c',  # green
            'random_forest': '#d62728',  # red
            'cnn': '#9467bd',  # purple
            'nbeats': '#8c564b',  # brown
            'ltc': '#e377c2',  # pink
            'tft': '#7f7f7f',  # gray
        }
        
        # Collect error metrics if available
        self.error_metrics = {}
        if hasattr(self.weighter, 'error_history'):
            self.error_metrics = self.weighter.error_history
    
    def render_dashboard(self):
        """Render the complete model visualization dashboard"""
        st.title("Ensemble Model Visualization Dashboard")
        
        # Create tabs for different visualization groups
        tabs = st.tabs([
            "Ensemble Weights", 
            "Model Performance", 
            "Regime Analysis",
            "Model Architecture",
            "Learning Insights"
        ])
        
        # Tab 1: Ensemble Weights
        with tabs[0]:
            self.render_ensemble_weights_tab()
        
        # Tab 2: Model Performance
        with tabs[1]:
            self.render_model_performance_tab()
        
        # Tab 3: Regime Analysis
        with tabs[2]:
            self.render_regime_analysis_tab()
        
        # Tab 4: Model Architecture
        with tabs[3]:
            self.render_model_architecture_tab()
        
        # Tab 5: Learning Insights
        with tabs[4]:
            self.render_learning_insights_tab()
    
    def render_ensemble_weights_tab(self):
        """Render visualizations related to ensemble weights"""
        st.header("Ensemble Weight Evolution")
        
        # Weight history plot
        if hasattr(self.weighter, 'historical_weights') and self.weighter.historical_weights:
            # Create dataframe from weight history
            weights_data = []
            for i, weights in enumerate(self.weighter.historical_weights):
                for model, weight in weights.items():
                    weights_data.append({
                        'timestep': i,
                        'model': model,
                        'weight': weight
                    })
            
            if weights_data:
                weights_df = pd.DataFrame(weights_data)
                
                # Create the line chart
                fig = px.line(
                    weights_df, 
                    x='timestep', 
                    y='weight', 
                    color='model',
                    color_discrete_map=self.model_colors,
                    title="Ensemble Weight Evolution Over Time",
                    labels={'timestep': 'Time Steps', 'weight': 'Model Weight'},
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Create weight distribution pie chart for current weights
                current_weights = self.weighter.current_weights
                
                fig = px.pie(
                    values=list(current_weights.values()),
                    names=list(current_weights.keys()),
                    color=list(current_weights.keys()),
                    color_discrete_map=self.model_colors,
                    title="Current Model Weight Distribution"
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show regime transitions if available
                if hasattr(self.weighter, 'weight_change_reasons') and self.weighter.weight_change_reasons:
                    st.subheader("Market Regime Transitions")
                    
                    # Create a dataframe for regime transitions
                    regime_data = []
                    for change in self.weighter.weight_change_reasons:
                        try:
                            regime_data.append({
                                'timestamp': pd.to_datetime(change.get('timestamp', '2023-01-01')),
                                'regime': change.get('regime', 'unknown'),
                                'reason': change.get('reason', 'No reason provided')
                            })
                        except:
                            # Skip any problematic entries
                            pass
                    
                    if regime_data:
                        regime_df = pd.DataFrame(regime_data)
                        
                        # Show as plotly timeline
                        fig = px.timeline(
                            regime_df, 
                            x_start='timestamp', 
                            x_end=[ts + timedelta(hours=12) for ts in regime_df['timestamp']],
                            y='regime',
                            color='regime',
                            hover_data=['reason'],
                            title="Market Regime Changes"
                        )
                        
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show as table for details
                        st.dataframe(regime_df)
            else:
                st.warning("No weight history data available yet")
        else:
            st.warning("No weight history available in the ensemble weighter")
            
        # Show correlations between models if available
        if hasattr(self.weighter, 'model_correlation_matrix') and self.weighter.model_correlation_matrix:
            st.subheader("Model Correlation Heatmap")
            
            # Extract correlation data
            models = list(self.weighter.base_weights.keys())
            corr_matrix = np.zeros((len(models), len(models)))
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    key = (model1, model2)
                    if key in self.weighter.model_correlation_matrix:
                        corr_matrix[i, j] = self.weighter.model_correlation_matrix[key]
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                x=models,
                y=models,
                color_continuous_scale='RdBu_r',
                title="Model Performance Correlation Matrix",
                labels=dict(x="Model", y="Model", color="Correlation"),
                zmin=-1, zmax=1
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance_tab(self):
        """Render visualizations related to model performance"""
        st.header("Model Performance Analysis")
        
        # Create error over time plot
        if self.error_metrics:
            # Create dataframe from error history
            error_data = []
            for model, errors in self.error_metrics.items():
                for i, error in enumerate(errors):
                    error_data.append({
                        'timestep': i,
                        'model': model,
                        'error': error
                    })
            
            if error_data:
                error_df = pd.DataFrame(error_data)
                
                # Allow log scale option for errors
                use_log_scale = st.checkbox("Use Log Scale for Errors", value=True)
                
                # Create the line chart
                fig = px.line(
                    error_df, 
                    x='timestep', 
                    y='error', 
                    color='model',
                    color_discrete_map=self.model_colors,
                    title="Model Error Over Time",
                    labels={'timestep': 'Time Steps', 'error': 'Error Metric'},
                    log_y=use_log_scale
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate average errors and create bar chart
                avg_errors = error_df.groupby('model')['error'].mean().reset_index()
                
                fig = px.bar(
                    avg_errors,
                    x='model',
                    y='error',
                    color='model',
                    color_discrete_map=self.model_colors,
                    title="Average Error by Model Type",
                    labels={'model': 'Model Type', 'error': 'Avg Error'},
                    log_y=use_log_scale
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create error distribution (by model)
                st.subheader("Error Distribution by Model")
                
                fig = px.box(
                    error_df,
                    x='model',
                    y='error',
                    color='model',
                    color_discrete_map=self.model_colors,
                    title="Error Distribution by Model Type",
                    log_y=use_log_scale
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show direction accuracy if available
                if hasattr(self.weighter, 'direction_history') and self.weighter.direction_history:
                    st.subheader("Direction Prediction Accuracy")
                    
                    # Create dataframe from direction accuracy history
                    direction_data = []
                    for model, accuracies in self.weighter.direction_history.items():
                        for i, accuracy in enumerate(accuracies):
                            direction_data.append({
                                'timestep': i,
                                'model': model,
                                'accuracy': accuracy * 100  # Convert to percentage
                            })
                    
                    if direction_data:
                        direction_df = pd.DataFrame(direction_data)
                        
                        # Create the line chart
                        fig = px.line(
                            direction_df, 
                            x='timestep', 
                            y='accuracy', 
                            color='model',
                            color_discrete_map=self.model_colors,
                            title="Direction Prediction Accuracy Over Time",
                            labels={'timestep': 'Time Steps', 'accuracy': 'Accuracy (%)'},
                        )
                        
                        # Add 50% line (random guessing)
                        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                                     annotation_text="Random Guessing")
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate average accuracy
                        avg_acc = direction_df.groupby('model')['accuracy'].mean().reset_index()
                        
                        fig = px.bar(
                            avg_acc,
                            x='model',
                            y='accuracy',
                            color='model',
                            color_discrete_map=self.model_colors,
                            title="Average Direction Accuracy by Model Type",
                            labels={'model': 'Model Type', 'accuracy': 'Avg Accuracy (%)'},
                        )
                        
                        # Add 50% line (random guessing)
                        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                                     annotation_text="Random Guessing")
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No error history data available yet")
        else:
            st.warning("No error metrics available in the ensemble weighter")
    
    def render_regime_analysis_tab(self):
        """Render visualizations related to market regime analysis"""
        st.header("Market Regime Analysis")
        
        # Get current regime if available
        current_regime = "Unknown"
        if hasattr(self.weighter, 'current_regime'):
            current_regime = self.weighter.current_regime
        
        # Display current regime in a big metric
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Current Market Regime", current_regime)
        
        # Regime performance if available
        if hasattr(self.weighter, 'optuna_feedback') and 'regime_performance' in self.weighter.optuna_feedback:
            regime_perf = self.weighter.optuna_feedback['regime_performance']
            
            if regime_perf:
                st.subheader("Model Performance by Market Regime")
                
                # Create tabs for each regime
                regime_tabs = st.tabs(list(regime_perf.keys()))
                
                for i, regime in enumerate(regime_perf.keys()):
                    with regime_tabs[i]:
                        performances = regime_perf[regime]
                        
                        if performances:
                            # Extract data
                            regime_data = []
                            for perf in performances:
                                weighted_error = perf.get('weighted_error', 0)
                                weights = perf.get('weights', {})
                                
                                for model, weight in weights.items():
                                    regime_data.append({
                                        'model': model,
                                        'weight': weight,
                                        'weighted_error': weighted_error
                                    })
                            
                            if regime_data:
                                regime_df = pd.DataFrame(regime_data)
                                
                                # Calculate average weight by model
                                avg_weights = regime_df.groupby('model')['weight'].mean().reset_index()
                                
                                fig = px.bar(
                                    avg_weights,
                                    x='model',
                                    y='weight',
                                    color='model',
                                    color_discrete_map=self.model_colors,
                                    title=f"Average Model Weights in {regime.capitalize()} Regime",
                                    labels={'model': 'Model Type', 'weight': 'Avg Weight'},
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show best weights for this regime
                                best_weights = None
                                best_error = float('inf')
                                
                                for perf in performances:
                                    if perf.get('weighted_error', float('inf')) < best_error:
                                        best_error = perf.get('weighted_error')
                                        best_weights = perf.get('weights')
                                
                                if best_weights:
                                    st.subheader(f"Best Weight Configuration for {regime.capitalize()} Regime")
                                    st.write(f"Weighted Error: {best_error:.6f}")
                                    
                                    # Create pie chart for best weights
                                    fig = px.pie(
                                        values=list(best_weights.values()),
                                        names=list(best_weights.keys()),
                                        color=list(best_weights.keys()),
                                        color_discrete_map=self.model_colors,
                                        title=f"Best Weights for {regime.capitalize()} Regime"
                                    )
                                    
                                    fig.update_traces(textposition='inside', textinfo='percent+label')
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No performance data for {regime} regime")
                        else:
                            st.warning(f"No performance data for {regime} regime")
        
        # Model-specific regime performance
        if hasattr(self.weighter, 'optuna_feedback') and 'model_performance' in self.weighter.optuna_feedback:
            model_perf = self.weighter.optuna_feedback['model_performance']
            
            if model_perf:
                st.subheader("Individual Model Performance by Regime")
                
                # Extract and reshape data
                regime_model_data = []
                
                for model, performances in model_perf.items():
                    for perf in performances:
                        if 'error' in perf and 'regime' in perf:
                            regime_model_data.append({
                                'model': model,
                                'regime': perf['regime'],
                                'error': perf['error'],
                                'weight': perf.get('weight', 0)
                            })
                
                if regime_model_data:
                    df = pd.DataFrame(regime_model_data)
                    
                    # Calculate average error by model and regime
                    pivot_df = df.pivot_table(
                        index='model', 
                        columns='regime', 
                        values='error',
                        aggfunc='mean'
                    ).reset_index()
                    
                    # Convert to long format for plotting
                    plot_df = pd.melt(
                        pivot_df, 
                        id_vars=['model'], 
                        var_name='regime', 
                        value_name='avg_error'
                    )
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        plot_df,
                        x='model',
                        y='avg_error',
                        color='regime',
                        title="Average Error by Model and Regime",
                        labels={'model': 'Model Type', 'avg_error': 'Avg Error', 'regime': 'Market Regime'},
                        barmode='group'
                    )
                    
                    use_log_scale = st.checkbox("Use Log Scale for Regime Analysis", value=True)
                    if use_log_scale:
                        fig.update_layout(yaxis_type="log")
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_model_architecture_tab(self):
        """Render visualizations related to model architecture"""
        st.header("Model Architecture Visualization")
        
        # Get model types and filter out 'transformer' if it exists since it's the same as 'tft'
        model_types = [model for model in self.weighter.base_weights.keys() if model != 'transformer']
        
        # Select model to visualize
        selected_model = st.selectbox("Select Model to Visualize", model_types)
        
        # Architecture visualization based on model type
        if selected_model:
            st.subheader(f"{selected_model.upper()} Architecture")
            
            if selected_model == 'lstm':
                self._visualize_lstm_architecture()
            elif selected_model == 'rnn':
                self._visualize_rnn_architecture()
            elif selected_model == 'xgboost':
                self._visualize_tree_architecture('XGBoost')
            elif selected_model == 'random_forest':
                self._visualize_tree_architecture('Random Forest')
            elif selected_model == 'cnn':
                self._visualize_cnn_architecture()
            elif selected_model == 'nbeats':
                self._visualize_nbeats_architecture()
            elif selected_model == 'ltc':
                self._visualize_ltc_architecture()
            elif selected_model == 'tft':
                self._visualize_tft_architecture()
            else:
                st.warning(f"Visualization not available for {selected_model} architecture")
    
    def render_learning_insights_tab(self):
        """Render visualizations related to model learning process"""
        st.header("Model Learning Insights")
        
        # Optuna feedback and suggestions
        if hasattr(self.weighter, 'optuna_feedback') and 'suggested_adjustments' in self.weighter.optuna_feedback:
            suggestions = self.weighter.optuna_feedback['suggested_adjustments']
            
            if suggestions:
                st.subheader("Optuna Feedback & Suggestions")
                
                # Create dataframe from suggestions
                suggestion_data = []
                for sugg in suggestions:
                    suggestion_data.append({
                        'model': sugg.get('model', 'unknown'),
                        'current_weight': sugg.get('current_base_weight', 0),
                        'suggested_weight': sugg.get('suggested_base_weight', 0),
                        'regime': sugg.get('regime', 'all'),
                        'reason': sugg.get('reason', 'No reason provided')
                    })
                
                if suggestion_data:
                    sugg_df = pd.DataFrame(suggestion_data)
                    
                    # Calculate weight change
                    sugg_df['weight_change'] = sugg_df['suggested_weight'] - sugg_df['current_weight']
                    sugg_df['percent_change'] = (sugg_df['weight_change'] / 
                                              sugg_df['current_weight'].replace(0, 0.0001) * 100)
                    
                    # Create visual comparison
                    fig = go.Figure()
                    
                    for i, row in sugg_df.iterrows():
                        model = row['model']
                        current = row['current_weight']
                        suggested = row['suggested_weight']
                        color = self.model_colors.get(model, '#1f77b4')
                        
                        # Add current weight
                        fig.add_trace(go.Bar(
                            name=f'{model} (Current)',
                            x=[f"{model} ({row['regime']})"],
                            y=[current],
                            marker_color=color,
                            opacity=0.6,
                            width=0.3,
                            offset=-0.2,
                            text=[f"{current:.3f}"],
                            textposition='outside'
                        ))
                        
                        # Add suggested weight
                        fig.add_trace(go.Bar(
                            name=f'{model} (Suggested)',
                            x=[f"{model} ({row['regime']})"],
                            y=[suggested],
                            marker_color=color,
                            width=0.3,
                            offset=0.2,
                            text=[f"{suggested:.3f}"],
                            textposition='outside'
                        ))
                    
                    fig.update_layout(
                        title="Weight Adjustment Suggestions from Optuna",
                        barmode='group',
                        height=400 + len(sugg_df) * 30
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table with reasoning
                    st.dataframe(sugg_df[['model', 'regime', 'current_weight', 
                                       'suggested_weight', 'percent_change', 'reason']])
                    
                    # Visualize weight changes as arrow chart
                    st.subheader("Weight Adjustment Direction")
                    
                    fig = go.Figure()
                    
                    for i, row in sugg_df.iterrows():
                        model = row['model']
                        current = row['current_weight']
                        change = row['weight_change']
                        color = 'green' if change > 0 else 'red'
                        
                        fig.add_trace(go.Scatter(
                            x=[0, change],
                            y=[i, i],
                            mode='lines+markers',
                            name=f"{model} ({row['regime']})",
                            line=dict(color=color, width=3),
                            marker=dict(size=10),
                        ))
                        
                        # Add model name as annotation
                        fig.add_annotation(
                            x=0,
                            y=i,
                            text=f"{model} ({row['regime']})",
                            showarrow=False,
                            xanchor='right',
                            xshift=-10
                        )
                    
                    fig.update_layout(
                        title="Weight Adjustment Direction and Magnitude",
                        xaxis_title="Weight Change",
                        showlegend=False,
                        height=400 + len(sugg_df) * 30,
                        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
                    )
                    
                    # Hide y-axis
                    fig.update_yaxes(showticklabels=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Learning trajectories visualization
        st.subheader("Model Learning Trajectories")
        
        # Create a line chart showing how models improve over time
        if self.error_metrics:
            # Calculate rolling average errors
            rolling_errors = {}
            window_size = 10  # Rolling window size
            
            for model, errors in self.error_metrics.items():
                if len(errors) > window_size:
                    # Convert to numpy for easier manipulation
                    err_array = np.array(list(errors))
                    rolling = np.array([
                        np.mean(err_array[max(0, i-window_size):i+1]) 
                        for i in range(len(err_array))
                    ])
                    rolling_errors[model] = rolling
            
            if rolling_errors:
                # Create plot
                fig = go.Figure()
                
                for model, errors in rolling_errors.items():
                    color = self.model_colors.get(model, '#1f77b4')
                    
                    # Calculate learning rate
                    if len(errors) > 1:
                        # Simple linear regression for trend
                        x = np.arange(len(errors))
                        z = np.polyfit(x, np.log(errors + 1e-10), 1)
                        slope = z[0]
                        learning_rate = -slope  # Negative slope = positive learning rate
                        
                        # Add trace with learning rate in name
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=errors,
                            mode='lines',
                            name=f"{model} (LR: {learning_rate:.4f})",
                            line=dict(color=color)
                        ))
                
                fig.update_layout(
                    title="Model Learning Trajectories (Rolling Average Error)",
                    xaxis_title="Training Steps",
                    yaxis_title="Error (Rolling Avg)",
                    legend_title="Model Types",
                    height=500
                )
                
                # Allow log scale option
                use_log_scale = st.checkbox("Use Log Scale for Learning Curves", value=True)
                if use_log_scale:
                    fig.update_layout(yaxis_type="log")
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _visualize_lstm_architecture(self):
        """Visualize LSTM architecture"""
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center;">Long Short-Term Memory (LSTM) Architecture</h3>
            <p style="text-align: center;">LSTM uses gates to control information flow over time, making it suitable for time series prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        # LSTM cell diagram
        lstm_diagram = """
        digraph G {
            rankdir=TB;
            
            subgraph cluster_lstm {
                label="LSTM Cell";
                color=blue;
                
                input [label="Input", shape=oval];
                forget [label="Forget Gate", shape=box, style=filled, fillcolor=lightyellow];
                input_gate [label="Input Gate", shape=box, style=filled, fillcolor=lightblue];
                cell_state [label="Cell State", shape=box, style=filled, fillcolor=lightgreen];
                output_gate [label="Output Gate", shape=box, style=filled, fillcolor=lightpink];
                hidden [label="Hidden State", shape=oval];
                
                input -> forget;
                input -> input_gate;
                input -> cell_state;
                input -> output_gate;
                
                forget -> cell_state;
                input_gate -> cell_state;
                cell_state -> output_gate;
                cell_state -> hidden;
                output_gate -> hidden;
            }
            
            x_t [label="X_t (Input)"];
            h_t [label="h_t (Output)"];
            h_t_prev [label="h_(t-1) (Previous Hidden)"];
            c_t_prev [label="c_(t-1) (Previous Cell)"];
            c_t [label="c_t (New Cell)"];
            
            x_t -> input;
            h_t_prev -> input;
            c_t_prev -> cell_state;
            hidden -> h_t;
            cell_state -> c_t;
        }
        """
        
        try:
            from graphviz import Source
            st.graphviz_chart(lstm_diagram)
        except:
            st.code(lstm_diagram, language="dot")
        
        st.markdown("""
        ### LSTM Key Components:
        
        1. **Forget Gate**: Decides what information to discard from cell state
        2. **Input Gate**: Updates the cell state with new information
        3. **Cell State**: Long-term memory component 
        4. **Output Gate**: Controls what information to output
        
        ### Advantages for Time Series:
        
        - Handles long-term dependencies in data
        - Avoids vanishing gradient problem
        - Can remember patterns over many time steps
        - Effective for cryptocurrency price prediction
        
        ### Typical Architecture:
        
        - Input layer → LSTM layer(s) → Dense layer(s) → Output
        - Often includes dropout for regularization
        - Can be stacked for more complex patterns
        """)
    
    def _visualize_rnn_architecture(self):
        """Visualize RNN architecture"""
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center;">Recurrent Neural Network (RNN) Architecture</h3>
            <p style="text-align: center;">Simple but powerful architecture for processing sequential data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # RNN diagram
        rnn_diagram = """
        digraph G {
            rankdir=LR;
            
            subgraph cluster_rnn {
                label="RNN Cell";
                color=blue;
                
                x_t [label="X_t (Input)"];
                h_t_prev [label="h_(t-1)"];
                cell [label="tanh", shape=box, style=filled, fillcolor=lightblue];
                h_t [label="h_t (Output)"];
                
                x_t -> cell;
                h_t_prev -> cell;
                cell -> h_t;
            }
            
            h_t -> h_t_next [style=dashed, label="To next timestep"];
            h_t_prev_src [label="From previous timestep", style=invisible];
            h_t_prev_src -> h_t_prev [style=dashed];
        }
        """
        
        try:
            from graphviz import Source
            st.graphviz_chart(rnn_diagram)
        except:
            st.code(rnn_diagram, language="dot")
        
        # Unrolled RNN
        st.markdown("### Unrolled Through Time:")
        
        unrolled_rnn = """
        digraph G {
            rankdir=LR;
            
            x_1 [label="X_1"];
            x_2 [label="X_2"];
            x_3 [label="X_3"];
            
            h_0 [label="h_0"];
            h_1 [label="h_1"];
            h_2 [label="h_2"];
            h_3 [label="h_3"];
            
            rnn_1 [label="RNN", shape=box, style=filled, fillcolor=lightblue];
            rnn_2 [label="RNN", shape=box, style=filled, fillcolor=lightblue];
            rnn_3 [label="RNN", shape=box, style=filled, fillcolor=lightblue];
            
            y_1 [label="y_1"];
            y_2 [label="y_2"];
            y_3 [label="y_3"];
            
            x_1 -> rnn_1;
            h_0 -> rnn_1;
            rnn_1 -> h_1;
            h_1 -> y_1;
            
            x_2 -> rnn_2;
            h_1 -> rnn_2;
            rnn_2 -> h_2;
            h_2 -> y_2;
            
            x_3 -> rnn_3;
            h_2 -> rnn_3;
            rnn_3 -> h_3;
            h_3 -> y_3;
        }
        """
        
        try:
            from graphviz import Source
            st.graphviz_chart(unrolled_rnn)
        except:
            st.code(unrolled_rnn, language="dot")
        
        st.markdown("""
        ### RNN Characteristics:
        
        1. **Recurrent Connection**: Hidden state is fed back into the network
        2. **Shared Parameters**: Same weights used at each time step
        3. **Memory**: Limited short-term memory capability
        
        ### Limitations:
        
        - Struggles with long-term dependencies due to vanishing gradient
        - Less powerful than LSTM for complex patterns
        - Works best for shorter time windows
        
        ### Common Uses:
        
        - Baseline for time series prediction
        - Simpler patterns in market data
        - When computational efficiency is important
        """)
    
    def _visualize_tree_architecture(self, model_type):
        """Visualize tree-based model architecture (XGBoost/Random Forest)"""
        is_xgboost = model_type == 'XGBoost'
        
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center;">{model_type} Architecture</h3>
            <p style="text-align: center;">Tree-based ensemble method that excels at capturing non-linear patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create sample tree visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"### Sample {model_type} Tree")
            
            # Sample tree diagram
            tree_diagram = """
            digraph G {
                node [shape=box, style="filled", color="black", fillcolor="lightblue"];
                
                root [label="Feature 3 <= 0.5?", fillcolor="lightgreen"];
                node1 [label="Feature 1 <= 0.3?"];
                node2 [label="Feature 2 <= 0.7?"];
                node3 [label="Value: 1.2"];
                node4 [label="Value: 2.5"];
                node5 [label="Feature 5 <= 0.2?"];
                node6 [label="Value: 3.8"];
                node7 [label="Value: 0.7"];
                node8 [label="Value: 1.9"];
                
                root -> node1 [label="Yes"];
                root -> node2 [label="No"];
                node1 -> node3 [label="Yes"];
                node1 -> node4 [label="No"];
                node2 -> node5 [label="Yes"];
                node2 -> node6 [label="No"];
                node5 -> node7 [label="Yes"];
                node5 -> node8 [label="No"];
            }
            """
            
            try:
                from graphviz import Source
                st.graphviz_chart(tree_diagram)
            except:
                st.code(tree_diagram, language="dot")
        
        with col2:
            st.markdown(f"### {model_type} Ensemble")
            
            # Ensemble diagram
            if is_xgboost:
                ensemble_diagram = """
                digraph G {
                    rankdir=TB;
                    
                    input [label="Input Features", shape=oval];
                    
                    subgraph cluster_trees {
                        label="Sequential Trees";
                        tree1 [label="Tree 1", shape=triangle, style=filled, fillcolor=lightblue];
                        tree2 [label="Tree 2", shape=triangle, style=filled, fillcolor=lightblue];
                        tree3 [label="Tree 3", shape=triangle, style=filled, fillcolor=lightblue];
                        dots [label="...", shape=plaintext];
                        treeN [label="Tree N", shape=triangle, style=filled, fillcolor=lightblue];
                        
                        tree1 -> tree2 [label="Residuals"];
                        tree2 -> tree3 [label="Residuals"];
                        tree3 -> dots [label="Residuals"];
                        dots -> treeN [label="Residuals"];
                    }
                    
                    sum [label="+", shape=circle, style=filled, fillcolor=yellow];
                    output [label="Prediction", shape=oval];
                    
                    input -> tree1;
                    tree1 -> sum;
                    tree2 -> sum;
                    tree3 -> sum;
                    treeN -> sum;
                    sum -> output;
                }
                """
            else:  # Random Forest
                ensemble_diagram = """
                digraph G {
                    rankdir=TB;
                    
                    input [label="Input Features", shape=oval];
                    
                    subgraph cluster_trees {
                        label="Parallel Trees";
                        tree1 [label="Tree 1\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
                        tree2 [label="Tree 2\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
                        tree3 [label="Tree 3\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
                        dots [label="...", shape=plaintext];
                        treeN [label="Tree N\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
                    }
                    
                    avg [label="Average", shape=circle, style=filled, fillcolor=yellow];
                    output [label="Prediction", shape=oval];
                    
                    input -> tree1;
                    input -> tree2;
                    input -> tree3;
                    input -> treeN;
                    tree1 -> avg;
                    tree2 -> avg;
                    tree3 -> avg;
                    treeN -> avg;
                    avg -> output;
                }
                """
            
            try:
                from graphviz import Source
                st.graphviz_chart(ensemble_diagram)
            except:
                st.code(ensemble_diagram, language="dot")
        
        st.markdown(f"""
        ### {model_type} Key Characteristics:
        
        1. **{'Boosting' if is_xgboost else 'Bagging'}**: 
           {'Each tree corrects errors of previous trees' if is_xgboost else 'Each tree is built independently on different bootstrap samples'}
        
        2. **Tree Structure**: Decision rules based on feature thresholds
        
        3. **{'Gradient Descent' if is_xgboost else 'Random Feature Selection'}**: 
           {'Minimizes loss function using gradient information' if is_xgboost else 'Each tree considers a random subset of features'}
        
        ### Advantages for Crypto Prediction:
        
        - Captures non-linear relationships in price data
        - Robust to outliers and market anomalies
        - Can identify important features automatically
        - {'Excellent performance for trend detection' if is_xgboost else 'Less prone to overfitting volatile markets'}
        
        ### Feature Importance:
        
        {model_type} can identify the most important features for prediction, such as:
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Volume metrics
        - Market sentiment scores
        - Historical price patterns
        """)
        
        # Add sample feature importance plot
        importance_data = {
            'RSI': 0.23,
            'Volume': 0.18,
            'MACD': 0.15,
            'Bollinger_Width': 0.12,
            'Price_Change': 0.10,
            'Volatility': 0.08,
            'MA_Crossover': 0.07,
            'Sentiment': 0.07
        }
        
        importance_df = pd.DataFrame({
            'Feature': list(importance_data.keys()),
            'Importance': list(importance_data.values())
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Sample {model_type} Feature Importance",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _visualize_cnn_architecture(self):
        """Visualize CNN architecture for time series"""
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center;">Convolutional Neural Network (CNN) for Time Series</h3>
            <p style="text-align: center;">Using 1D convolutions to detect patterns in sequential financial data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # CNN visualization
        st.markdown("### CNN Architecture for Time Series")
        
        # Create a simple visual representation
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            st.markdown("#### Input Layer")
            st.markdown("""
            ```
            [t-n]
            [t-n+1]
            [t-n+2]
            ...
            [t-2]
            [t-1]
            [t]
            ```
            """)
            st.markdown("Time series data")
        
        with col2:
            st.markdown("#### Conv1D Layer")
            st.markdown("""
            ```
            |-----|
            |-----|
            |-----|
            ```
            """)
            st.markdown("Extract local patterns")
        
        with col3:
            st.markdown("#### Pooling Layer")
            st.markdown("""
            ```
            |---|
            |---|
            ```
            """)
            st.markdown("Reduce & extract features")
        
        with col4:
            st.markdown("#### Dense Layers")
            st.markdown("""
            ```
            [     ]
            [     ]
            ```
            """)
            st.markdown("Non-linear combinations")
        
        with col5:
            st.markdown("#### Output")
            st.markdown("""
            ```
            [t+1]
            [t+2]
            ...
            [t+n]
            ```
            """)
            st.markdown("Future predictions")
        
        st.markdown("""
        ### CNN for Time Series - Key Concepts:
        
        1. **1D Convolutions**: Slides window over time dimension instead of 2D space (unlike image CNNs)
        
        2. **Kernel/Filter**: Detect specific patterns in time series (e.g., trend reversals, price spikes)
        
        3. **Pooling**: Reduces dimensionality while retaining important features
        
        4. **Multiple Filter Types**: Different filters can detect different price patterns
        
        ### Advantages for Crypto Price Prediction:
        
        - Automatically detects local patterns in price movements
        - Robust to time shifts (same pattern at different times)
        - Efficiently handles multivariate inputs (price, volume, indicators)
        - Can capture both short-term and medium-term dependencies
        
        ### Example CNN-LSTM Hybrid Architecture:
        
        Many cryptocurrency prediction models combine CNN and LSTM layers:
        - CNN layers first extract local patterns
        - LSTM layers then model the temporal dependencies
        - This combines spatial and temporal feature extraction
        """)
        
        # Show sample CNN filters visualization
        st.subheader("CNN Filter Visualization")
        
        # Create sample filter activations
        np.random.seed(42)
        filter_data = []
        activation_data = []
        
        # Sample time series
        time_steps = 100
        x = np.linspace(0, 10, time_steps)
        y = np.sin(x) + 0.1 * np.random.randn(time_steps)
        
        # Three sample filter activations
        act1 = np.convolve(y, [0.2, 0.5, 0.3], mode='valid')
        act2 = np.convolve(y, [-0.3, 0.1, 0.7], mode='valid')
        act3 = np.convolve(y, [0, -0.5, 0.5], mode='valid')
        
        df = pd.DataFrame({
            'time': x,
            'price': y,
            'filter1': np.pad(act1, (1, 1), 'constant'),
            'filter2': np.pad(act2, (1, 1), 'constant'),
            'filter3': np.pad(act3, (1, 1), 'constant')
        })
        
        fig = px.line(
            df,
            x='time',
            y=['price', 'filter1', 'filter2', 'filter3'],
            title="CNN Filters Detecting Different Patterns",
            labels={'value': 'Activation', 'variable': 'Signal Type'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def _visualize_nbeats_architecture(self):
        """Visualize N-BEATS architecture"""
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center;">N-BEATS Architecture</h3>
            <p style="text-align: center;">Neural Basis Expansion Analysis for interpretable Time Series forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        # N-BEATS diagram
        st.markdown("### N-BEATS Block Structure")
        
        nbeats_diagram = """
        digraph G {
            rankdir=TB;
            
            input [label="Input Time Series"];
            
            subgraph cluster_block {
                label="N-BEATS Block";
                style=filled;
                color=lightgrey;
                
                fc1 [label="Fully Connected Layers", style=filled, fillcolor=lightblue];
                split [label="Split", shape=diamond];
                
                backcast [label="Backcast", style=filled, fillcolor=lightgreen];
                forecast [label="Forecast", style=filled, fillcolor=lightpink];
                
                fc1 -> split;
                split -> backcast;
                split -> forecast;
            }
            
            subgraph cluster_stack {
                label="Stack of Blocks";
                node [style=filled, fillcolor=lightgrey];
                
                block1 [label="Block 1"];
                block2 [label="Block 2"];
                block3 [label="..."];
                blockN [label="Block N"];
                
                block1 -> block2;
                block2 -> block3;
                block3 -> blockN;
            }
            
            input -> block1;
            blockN -> output [label="Final Forecast"];
            output [label="Output Prediction"];
        }
        """
        
        try:
            from graphviz import Source
            st.graphviz_chart(nbeats_diagram)
        except:
            st.code(nbeats_diagram, language="dot")
        
        st.markdown("""
        ### N-BEATS Key Components:
        
        1. **Block Structure**: Each block produces both backcast (reconstruction) and forecast outputs
        
        2. **Double Residual Stacking**: Each block processes residuals from previous blocks
        
        3. **Basis Functions**: 
           - Trend block: polynomial basis functions
           - Seasonality block: Fourier basis functions
           - Generic block: learned basis functions
        
        ### Advantages for Crypto Prediction:
        
        - Interpretable components (can separate trend and seasonality)
        - Handles multiple seasonalities (daily, weekly patterns)
        - Pure deep learning approach (no feature engineering required)
        - Strong performance for multi-step forecasting
        
        ### Stack Types:
        
        - **Generic Stack**: Learns patterns directly from data
        - **Trend Stack**: Specialized for trend components
        - **Seasonality Stack**: Specialized for cyclical patterns
        
        N-BEATS often outperforms traditional models by combining interpretability with deep learning power.
        """)
        
        # Sample basis visualization
        st.subheader("N-BEATS Basis Function Visualization")
        
        # Create sample data for basis functions
        x = np.linspace(0, 14, 100)
        
        # Trend bases
        trend1 = 0.5 * x
        trend2 = 0.1 * x**2
        trend3 = 0.01 * x**3
        
        # Seasonal bases
        seasonal1 = np.sin(x)
        seasonal2 = np.sin(2*x)
        seasonal3 = np.sin(4*x)
        
        # Create dataframe
        basis_df = pd.DataFrame({
            'x': x,
            'Linear Trend': trend1,
            'Quadratic Trend': trend2,
            'Cubic Trend': trend3,
            'Daily Seasonality': seasonal1,
            'Half-day Seasonality': seasonal2,
            '6-hour Seasonality': seasonal3
        })
        
        # Plot trend bases
        fig1 = px.line(
            basis_df,
            x='x',
            y=['Linear Trend', 'Quadratic Trend', 'Cubic Trend'],
            title="Trend Basis Functions",
            labels={'value': 'Amplitude', 'variable': 'Basis Type'}
        )
        
        # Plot seasonal bases
        fig2 = px.line(
            basis_df,
            x='x',
            y=['Daily Seasonality', 'Half-day Seasonality', '6-hour Seasonality'],
            title="Seasonality Basis Functions",
            labels={'value': 'Amplitude', 'variable': 'Basis Type'}
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
    
    def _visualize_ltc_architecture(self):
        """Visualize Liquid Time-Constant Networks architecture"""
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center;">Liquid Time-Constant (LTC) Networks</h3>
            <p style="text-align: center;">Biologically-inspired recurrent neural networks for continuous-time dynamics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # LTC diagram
        ltc_diagram = """
        digraph G {
            rankdir=LR;
            
            node [shape=circle, style=filled, color=black, fillcolor=lightblue];
            
            input [label="Input", shape=plaintext];
            
            subgraph cluster_ltc {
                label="LTC Layer";
                style=filled;
                color=lightgrey;
                
                neuron1 [label="τ₁"];
                neuron2 [label="τ₂"];
                neuron3 [label="τ₃"];
                neuron4 [label="..."];
                neuron5 [label="τₙ"];
                
                neuron1 -> neuron2 [dir=both, style=dashed];
                neuron1 -> neuron3 [dir=both, style=dashed];
                neuron2 -> neuron3 [dir=both, style=dashed];
                neuron2 -> neuron4 [dir=both, style=dashed];
                neuron3 -> neuron4 [dir=both, style=dashed];
                neuron3 -> neuron5 [dir=both, style=dashed];
                neuron4 -> neuron5 [dir=both, style=dashed];
            }
            
            output [label="Output", shape=plaintext];
            
            input -> neuron1;
            input -> neuron2;
            input -> neuron3;
            
            neuron3 -> output;
            neuron4 -> output;
            neuron5 -> output;
            
            // Self-loops
            neuron1 -> neuron1 [label="τ₁"];
            neuron3 -> neuron3 [label="τ₃"];
            neuron5 -> neuron5 [label="τₙ"];
        }
        """
        
        try:
            from graphviz import Source
            st.graphviz_chart(ltc_diagram)
        except:
            st.code(ltc_diagram, language="dot")
        
        st.markdown("""
        ### LTC Network Key Concepts:
        
        1. **Time Constants (τ)**: Each neuron has its own learnable time constant
           - Controls how fast/slow the neuron responds to inputs
           - Different time scales capture different market dynamics
        
        2. **Continuous-Time Dynamics**: Uses differential equations instead of discrete updates
           - More natural for modeling market time series with variable sample rates
        
        3. **Liquid State Machines**: Inspired by biological neural networks
           - Better at handling complex non-linear dynamics
        
        ### Advantages for Crypto Prediction:
        
        - Can model multiple timescales simultaneously (minutes, hours, days)
        - Handles irregular sampling and missing data gracefully
        - Captures both fast (volatility) and slow (trend) market dynamics
        - More biologically plausible learning dynamics
        
        ### LTC vs Traditional RNNs:
        
        - LSTM/GRU: Uses gates to control information flow
        - LTC: Uses continuous-time dynamics with learnable time constants
        - Better theoretical properties for capturing complex dynamical systems
        - Often more parameter-efficient than traditional RNNs
        """)
        
        # Time constant visualization
        st.subheader("LTC Time Constants Visualization")
        
        # Sample data for time constants
        x = np.linspace(0, 10, 1000)
        
        # Different time constants
        tau1 = 0.5  # Fast
        tau2 = 2.0  # Medium
        tau3 = 5.0  # Slow
        
        # Input signal - step function at t=2
        input_signal = np.zeros_like(x)
        input_signal[x > 2] = 1.0
        
        # Response with different time constants
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y3 = np.zeros_like(x)
        
        # Simple first-order dynamics: dy/dt = (u - y) / tau
        for i in range(1, len(x)):
            dt = x[i] - x[i-1]
            y1[i] = y1[i-1] + dt * (input_signal[i-1] - y1[i-1]) / tau1
            y2[i] = y2[i-1] + dt * (input_signal[i-1] - y2[i-1]) / tau2
            y3[i] = y3[i-1] + dt * (input_signal[i-1] - y3[i-1]) / tau3
        
        # Plot responses
        tau_df = pd.DataFrame({
            'Time': x,
            'Input': input_signal,
            f'τ = {tau1} (Fast)': y1,
            f'τ = {tau2} (Medium)': y2,
            f'τ = {tau3} (Slow)': y3
        })
        
        fig = px.line(
            tau_df,
            x='Time',
            y=['Input', f'τ = {tau1} (Fast)', f'τ = {tau2} (Medium)', f'τ = {tau3} (Slow)'],
            title="Effect of Different Time Constants on Neuron Response",
            labels={'value': 'Response', 'variable': 'Signal Type'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        The chart above demonstrates how different time constants affect neuron responses:
        
        - **Fast neurons** (small τ): Respond quickly to market changes but are sensitive to noise
        - **Medium neurons**: Balance between responsiveness and stability
        - **Slow neurons** (large τ): Filter out noise but may miss sudden market movements
        
        LTC Networks learn the optimal time constants for each neuron automatically during training.
        """)
    
    def _visualize_tft_architecture(self):
        """Visualize Temporal Fusion Transformer architecture"""
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center;">Temporal Fusion Transformer (TFT)</h3>
            <p style="text-align: center;">State-of-the-art architecture combining attention mechanisms with specialized components for time series</p>
        </div>
        """, unsafe_allow_html=True)
        
        # TFT architecture diagram
        st.markdown("### TFT Architecture Overview")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            tft_diagram = """
            digraph G {
                rankdir=TB;
                
                // Inputs
                past [label="Past Inputs", shape=box, style=filled, fillcolor=lightblue];
                known [label="Known Future Inputs", shape=box, style=filled, fillcolor=lightgreen];
                
                // Processing layers
                gating [label="Variable Selection Networks", shape=box, style=filled, fillcolor=lightyellow];
                static [label="Static Covariate Encoders", shape=box, style=filled, fillcolor=lightyellow];
                lstm_encoder [label="LSTM Encoder", shape=box, style=filled, fillcolor=pink];
                lstm_decoder [label="LSTM Decoder", shape=box, style=filled, fillcolor=pink];
                
                // Attention mechanism
                attention [label="Multi-head Attention", shape=box, style=filled, fillcolor=lightcoral];
                
                // Output
                temporal_fusion [label="Temporal Fusion Layer", shape=box, style=filled, fillcolor=lightgrey];
                position_wise [label="Position-wise Feed-forward", shape=box, style=filled, fillcolor=lightgrey];
                quantile [label="Quantile Outputs", shape=box, style=filled, fillcolor=lightblue];
                
                // Connections
                past -> gating;
                known -> gating;
                gating -> lstm_encoder;
                gating -> lstm_decoder;
                static -> lstm_encoder;
                static -> lstm_decoder;
                static -> attention;
                lstm_encoder -> attention;
                lstm_decoder -> attention;
                attention -> temporal_fusion;
                temporal_fusion -> position_wise;
                position_wise -> quantile;
            }
            """
            
            try:
                from graphviz import Source
                st.graphviz_chart(tft_diagram)
            except:
                st.code(tft_diagram, language="dot")
        
        with col2:
            st.markdown("""
            **Key Components:**
            
            1. Variable Selection Networks
            2. LSTM Encoding/Decoding
            3. Multi-head Attention
            4. Temporal Fusion
            5. Quantile Forecasts
            """)
        
        st.markdown("""
        ### TFT Advanced Components:
        
        1. **Variable Selection Networks**:
           - Dynamically selects most relevant features at each timestep
           - Learns importance of different indicators
        
        2. **Multi-head Attention**:
           - Identifies dependencies between different time points
           - Can model relationships between distant time points
        
        3. **Gating Mechanisms**:
           - Controls information flow between components
           - Helps manage multiple input types
        
        4. **Quantile Forecasts**:
           - Provides prediction intervals, not just point estimates
           - Critical for risk management in crypto trading
        
        ### Advantages for Crypto Prediction:
        
        - State-of-the-art performance on complex time series
        - Built-in uncertainty estimation through quantiles
        - Handles static (e.g., coin fundamentals) and temporal features
        - Excellent interpretation capabilities
        - Robust to noisy data common in crypto markets
        
        ### Interpretability Features:
        
        - Variable importance scores at each timestep
        - Attention weights show which past periods influence predictions
        - Uncertainty quantification through prediction intervals
        """)
        
        # Sample interpretability plots
        st.subheader("TFT Interpretability Visualizations")
        
        # Create sample attention weights
        num_days = 30
        dates = pd.date_range(end=pd.Timestamp.now(), periods=num_days).date
        
        # Create synthetic attention weights
        np.random.seed(42)
        attention_weights = np.zeros((10, num_days))
        for i in range(10):
            weights = np.abs(np.random.randn(num_days))
            attention_weights[i] = weights / weights.sum()
        
        # Create average attention weights
        avg_attention = attention_weights.mean(axis=0)
        
        # Create attention heatmap
        fig1 = px.imshow(
            attention_weights,
            x=dates,
            labels=dict(x="Date", y="Attention Head", color="Weight"),
            title="Multi-head Attention Weights",
            color_continuous_scale="Viridis"
        )
        
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Feature importance
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Width', 'Sentiment', 'Market_Cap', 'Exchange_Inflow']
        importance = np.array([0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05])
        
        # Create sample time-varying importance
        time_importance = np.zeros((len(features), min(10, num_days)))
        for i, base_imp in enumerate(importance):
            time_importance[i] = base_imp + 0.05 * np.random.randn(min(10, num_days))
            time_importance[i] = np.maximum(0, time_importance[i])
        
        # Normalize columns
        for j in range(time_importance.shape[1]):
            time_importance[:, j] = time_importance[:, j] / time_importance[:, j].sum()
        
        # Feature importance over time
        recent_dates = dates[-min(10, num_days):]
        fig2 = px.imshow(
            time_importance,
            x=recent_dates,
            y=features,
            labels=dict(x="Date", y="Feature", color="Importance"),
            title="Feature Importance Over Time",
            color_continuous_scale="Viridis"
        )
        
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Quantile predictions
        x = np.arange(num_days)
        y_median = 100 + 0.1 * x + 5 * np.sin(x/5) + np.random.randn(num_days)
        y_low = y_median - 5 - 0.2 * x
        y_high = y_median + 5 + 0.2 * x
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': dates,
            'P10': y_low,
            'P50': y_median,
            'P90': y_high
        })
        
        # Plot quantile forecasts
        fig3 = go.Figure()
        
        # Add quantile range
        fig3.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['P90'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='P90'
        ))
        
        fig3.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['P10'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='P10'
        ))
        
        # Add median prediction
        fig3.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['P50'],
            mode='lines',
            line=dict(color='rgb(0,100,80)', width=2),
            name='P50 (Median)'
        ))
        
        fig3.update_layout(
            title="TFT Quantile Forecasts with Uncertainty",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        
        st.plotly_chart(fig3, use_container_width=True)
