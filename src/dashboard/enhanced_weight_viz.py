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

class WeightVisualizationDashboard:
    """Advanced dashboard for visualizing ensemble weight evolution and providing weight adjustment insights"""
    
    def __init__(self, ensemble_weighter, model_directory="model_weights"):
        """
        Initialize visualization dashboard
        
        Args:
            ensemble_weighter: Instance of EnsembleWeighter
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
    
    def render_dashboard(self):
        """Render the enhanced weight visualization dashboard"""
        st.title("Enhanced Weight Visualization Dashboard")
        
        # Create tabs for different visualization groups
        tabs = st.tabs([
            "Weight Evolution", 
            "Weight Flow", 
            "Interactive Adjustment",
            "Weight Network",
            "Strategy Weights"
        ])
        
        # Tab 1: Weight Evolution 
        with tabs[0]:
            self._render_3d_weight_evolution()
        
        # Tab 2: Weight Flow
        with tabs[1]:
            self._render_weight_flow_diagram()
        
        # Tab 3: Interactive Adjustment
        with tabs[2]:
            self._render_interactive_weight_tool()
        
        # Tab 4: Weight Network
        with tabs[3]:
            self._render_weight_network_graph()
        
        # Tab 5: Strategy Weights
        with tabs[4]:
            self._render_strategy_weights()
    
    def _render_3d_weight_evolution(self):
        """Render 3D visualization of weight evolution through time"""
        st.subheader("3D Weight Evolution Through Time")
        
        # Check if we have enough historical weights
        if not hasattr(self.weighter, 'historical_weights') or len(self.weighter.historical_weights) < 5:
            st.warning("Not enough weight history data for 3D visualization")
            return
        
        # Get historical weight data
        weights_data = []
        for i, weights in enumerate(self.weighter.historical_weights):
            for model, weight in weights.items():
                weights_data.append({
                    'timestep': i,
                    'model': model,
                    'weight': weight
                })
        
        if not weights_data:
            st.warning("No weight data available")
            return
        
        # Create 3D plot with plotly
        df = pd.DataFrame(weights_data)
        
        # Create a continuous color scale based on model
        models = df['model'].unique()
        model_to_idx = {model: i for i, model in enumerate(models)}
        df['model_idx'] = df['model'].map(model_to_idx)
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=df[df['model'] == model]['timestep'],
                y=df[df['model'] == model]['model_idx'],
                z=df[df['model'] == model]['weight'],
                mode='lines+markers',
                name=model,
                marker=dict(
                    size=5,
                    color=self.model_colors.get(model, '#1f77b4'),
                    opacity=0.8
                ),
                line=dict(
                    color=self.model_colors.get(model, '#1f77b4'),
                    width=2
                )
            ) for model in models
        ])
        
        fig.update_layout(
            title="3D Model Weight Evolution",
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Model",
                zaxis_title="Weight",
                yaxis=dict(
                    ticktext=list(models),
                    tickvals=list(range(len(models)))
                )
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        This 3D visualization shows how model weights have evolved over time. Each line represents
        a different model, and the height shows the weight assigned to that model at each timestep.
        
        - **Spikes**: Rapid weight adjustments due to performance changes
        - **Plateaus**: Periods of stable weights
        - **Convergence/Divergence**: Models becoming more or less important
        """)

    def _render_weight_flow_diagram(self):
        """Render Sankey diagram showing weight flow between time periods"""
        st.subheader("Weight Flow Diagram")
        
        # Check if we have enough historical weights
        if not hasattr(self.weighter, 'historical_weights') or len(self.weighter.historical_weights) < 2:
            st.warning("Not enough weight history data for flow visualization")
            return
        
        # Select time points for comparison (initial, middle, current)
        historical_weights = self.weighter.historical_weights
        n_periods = len(historical_weights)
        
        if n_periods >= 3:
            periods = {
                "Initial": 0,
                "Middle": n_periods // 2,
                "Current": -1
            }
        else:
            periods = {
                "Initial": 0,
                "Current": -1
            }
        
        # Create Sankey diagram data
        source = []
        target = []
        value = []
        label = []
        
        # Create nodes
        models = list(self.weighter.base_weights.keys())
        period_names = list(periods.keys())
        
        # Create node labels
        for period in period_names:
            for model in models:
                label.append(f"{period} {model}")
        
        # Create links
        for p_idx in range(len(period_names) - 1):
            current_period = period_names[p_idx]
            next_period = period_names[p_idx + 1]
            
            current_weights = historical_weights[periods[current_period]]
            next_weights = historical_weights[periods[next_period]]
            
            for m_idx, model in enumerate(models):
                source_idx = p_idx * len(models) + m_idx
                target_idx = (p_idx + 1) * len(models) + m_idx
                
                # Use the current model weight as the value
                current_value = current_weights.get(model, 0)
                next_value = next_weights.get(model, 0)
                
                # Add the link
                source.append(source_idx)
                target.append(target_idx)
                
                # Use the average of current and next weight for link value
                link_value = (current_value + next_value) / 2
                value.append(link_value)
        
        # Create color scale
        colors = []
        for model in models:
            for _ in range(len(period_names)):
                colors.append(self.model_colors.get(model, '#1f77b4'))
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label,
                color=colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=[f"rgba{tuple(int(c[1:][i:i+2], 16) for i in (0, 2, 4)) + (0.5,)}" 
                       for c in colors[:len(source)]]
            )
        )])
        
        fig.update_layout(
            title="Weight Flow Between Time Periods",
            height=600,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        This Sankey diagram shows how weights have flowed between different time periods.
        
        - **Wider flows**: Models with higher weights
        - **Changing widths**: Weight increases or decreases over time
        - **Color**: Different models in the ensemble
        """)

    def _render_interactive_weight_tool(self):
        """Render interactive tool to adjust model weights and see their effect"""
        st.subheader("Interactive Weight Adjustment Tool")
        
        # Get current weights
        current_weights = self.weighter.current_weights
        
        # Create sliders for each model weight
        st.markdown("### Adjust Weights Manually")
        
        # Create a column for each model
        cols = st.columns(min(4, len(current_weights)))
        
        # Initialize adjusted weights
        adjusted_weights = {}
        
        # Add sliders for each model
        for i, (model, weight) in enumerate(current_weights.items()):
            col_idx = i % len(cols)
            with cols[col_idx]:
                color_hex = self.model_colors.get(model, '#1f77b4')
                # Convert to RGB format for st.markdown
                r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                
                st.markdown(f"""
                <div style='background-color: rgb({r},{g},{b}); 
                            padding: 5px; 
                            border-radius: 5px; 
                            color: white; 
                            text-align: center;'>
                    <strong>{model}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Add slider for adjustment
                adjusted_weights[model] = st.slider(
                    f"Adjust {model}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(weight),
                    step=0.01,
                    key=f"adjust_{model}",
                    label_visibility="collapsed"
                )
        
        # Normalize adjusted weights to sum to 1
        total = sum(adjusted_weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in adjusted_weights.items()}
        else:
            normalized_weights = {k: 1.0/len(adjusted_weights) for k in adjusted_weights}
        
        # Display the normalized weights
        st.markdown("### Normalized Weights")
        
        # Create visualization for adjusted weights
        fig = go.Figure()
        
        # Add current weights
        fig.add_trace(go.Bar(
            x=list(current_weights.keys()),
            y=list(current_weights.values()),
            name='Current',
            marker_color='blue',
            opacity=0.6
        ))
        
        # Add adjusted weights
        fig.add_trace(go.Bar(
            x=list(normalized_weights.keys()),
            y=list(normalized_weights.values()),
            name='Adjusted',
            marker_color='green',
            opacity=0.6
        ))
        
        fig.update_layout(
            title="Current vs Adjusted Weights",
            xaxis_title="Model",
            yaxis_title="Weight",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show weight changes
        st.markdown("### Weight Changes")
        
        changes = []
        for model in current_weights:
            current = current_weights.get(model, 0)
            adjusted = normalized_weights.get(model, 0)
            change = adjusted - current
            pct_change = change / current * 100 if current > 0 else float('inf')
            
            changes.append({
                'Model': model,
                'Current': current,
                'Adjusted': adjusted,
                'Change': change,
                'Pct Change': pct_change
            })
        
        changes_df = pd.DataFrame(changes)
        
        # Format the dataframe for display
        formatted_df = changes_df.copy()
        formatted_df['Current'] = formatted_df['Current'].map('{:.4f}'.format)
        formatted_df['Adjusted'] = formatted_df['Adjusted'].map('{:.4f}'.format)
        formatted_df['Change'] = formatted_df['Change'].map('{:+.4f}'.format)
        formatted_df['Pct Change'] = formatted_df['Pct Change'].map('{:+.2f}%'.format)
        
        # Use dataframe styling
        st.dataframe(
            formatted_df,
            column_config={
                "Pct Change": st.column_config.ProgressColumn(
                    "Percent Change",
                    format="%+.2f%%",
                    min_value=-100,
                    max_value=100
                )
            }
        )
    
    def _render_weight_network_graph(self):
        """Render network graph showing model correlations"""
        st.subheader("Model Correlation Network")
        
        # Check if we have correlation data
        if not hasattr(self.weighter, 'model_correlation_matrix') or self.weighter.model_correlation_matrix is None:
            st.warning("No model correlation data available")
            return
        
        # Get correlation matrix
        corr_matrix = self.weighter.model_correlation_matrix
        
        # Create nodes and edges for the graph
        nodes = []
        edges = []
        
        # Add nodes for each model
        for model, weight in self.weighter.current_weights.items():
            nodes.append({
                'name': model,
                'size': weight * 50,  # Scale node size by weight
                'color': self.model_colors.get(model, '#1f77b4')
            })
        
        # Add edges for each correlation
        for (model1, model2), correlation in corr_matrix.items():
            # Only add edges once (undirected graph)
            if model1 > model2:
                continue
            
            # Scale correlation to edge width
            edge_width = abs(correlation)
            
            # Determine edge color based on correlation
            if correlation > 0:
                edge_color = 'green'
            else:
                edge_color = 'red'
            
            # Find node indices
            source_idx = next((i for i, node in enumerate(nodes) if node['name'] == model1), None)
            target_idx = next((i for i, node in enumerate(nodes) if node['name'] == model2), None)
            
            # Add edge
            if source_idx is not None and target_idx is not None:
                edges.append({
                    'source': source_idx,
                    'target': target_idx,
                    'value': edge_width,
                    'color': edge_color
                })
        
        # Create networkx graph for visualization
        try:
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes
            for i, node in enumerate(nodes):
                G.add_node(i, name=node['name'], size=node['size'], color=node['color'])
            
            # Add edges
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], 
                          weight=edge['value'], 
                          color=edge['color'])
            
            # Position nodes using force-directed layout
            pos = nx.spring_layout(G, seed=42)
            
            # Create plotly figure
            edge_trace = []
            
            # Add edges
            for edge in edges:
                source = edge['source']
                target = edge['target']
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=edge['value'] * 5,  # Scale width by correlation
                        color=edge['color'],
                        dash='solid' if edge['value'] > 0.5 else 'dash'
                    ),
                    hoverinfo='none'
                ))
            
            # Add nodes
            node_trace = go.Scatter(
                x=[pos[i][0] for i in range(len(nodes))],
                y=[pos[i][1] for i in range(len(nodes))],
                mode='markers+text',
                text=[G.nodes[i]['name'] for i in range(len(nodes))],
                textposition="top center",
                marker=dict(
                    size=[G.nodes[i]['size'] for i in range(len(nodes))],
                    color=[G.nodes[i]['color'] for i in range(len(nodes))],
                    line=dict(width=2, color='white')
                ),
                hoverinfo='text'
            )
            
            # Create figure
            fig = go.Figure(data=edge_trace + [node_trace])
            
            fig.update_layout(
                title="Model Correlation Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            This network graph shows the relationships between models:
            
            - **Node Size**: Model weight in ensemble
            - **Edge Color**: Red = negative correlation, Green = positive correlation
            - **Edge Thickness**: Strength of correlation
            - **Node Color**: Model type
            
            Models that are negatively correlated (red connections) tend to complement each other well
            in an ensemble, while positively correlated models (green) often make similar predictions.
            """)
            
        except ImportError:
            st.error("NetworkX library is required for network visualization")
            st.info("Install with: pip install networkx")

    def _render_strategy_weights(self):
        """Render visualizations for strategy-level weights"""
        st.subheader("Strategy-Level Weight Analysis")
        
        # Check if we have strategy weights (look for attr specific to strategy weights)
        has_strategy_weights = (
            hasattr(self.weighter, 'strategy_weights') or 
            hasattr(self.weighter, 'meta_weights') or
            hasattr(self.weighter, 'feature_importance')
        )
        
        if not has_strategy_weights:
            # Create mock strategy weights for demonstration
            st.info("Creating sample strategy weight visualization (replace with actual strategy weights)")
            
            # Create sample data
            strategy_categories = {
                "Loss Functions": {
                    "MSE": 0.30,
                    "MAE": 0.25,
                    "Huber": 0.25,
                    "Quantile": 0.20
                },
                "Time Horizons": {
                    "Short-term (1-3 days)": 0.25,
                    "Medium-term (4-14 days)": 0.45,
                    "Long-term (15-30 days)": 0.30
                },
                "Feature Groups": {
                    "Price": 0.35,
                    "Volume": 0.20,
                    "Technical": 0.30,
                    "Sentiment": 0.15
                },
                "Regime-Specific Strategies": {
                    "Trend-following": 0.40,
                    "Mean-reversion": 0.30,
                    "Breakout": 0.15,
                    "Volatility-based": 0.15
                }
            }
            
            # Create tabs for each strategy category
            category_tabs = st.tabs(list(strategy_categories.keys()))
            
            for i, (category, weights) in enumerate(strategy_categories.items()):
                with category_tabs[i]:
                    # Create pie chart
                    fig = px.pie(
                        values=list(weights.values()),
                        names=list(weights.keys()),
                        title=f"{category} Weight Distribution",
                        hole=0.4
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create bar chart with threshold for current vs optimal
                    fig2 = go.Figure()
                    
                    # Add current weights
                    fig2.add_trace(go.Bar(
                        x=list(weights.keys()),
                        y=list(weights.values()),
                        name='Current',
                        marker_color='blue'
                    ))
                    
                    # Add "optimal" weights (slightly different)
                    import numpy as np
                    optimal_weights = {k: v * (1 + 0.2 * np.random.randn()) for k, v in weights.items()}
                    # Normalize
                    total = sum(optimal_weights.values())
                    optimal_weights = {k: v/total for k, v in optimal_weights.items()}
                    
                    fig2.add_trace(go.Bar(
                        x=list(weights.keys()),
                        y=list(optimal_weights.values()),
                        name='Suggested Optimal',
                        marker_color='green'
                    ))
                    
                    fig2.update_layout(
                        title=f"Current vs Suggested {category} Weights",
                        xaxis_title="Strategy Component",
                        yaxis_title="Weight",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Show adjustment rationale
                    st.subheader("Adjustment Rationale")
                    
                    # Create a dataframe with adjustment explanations
                    adjustments = []
                    for k, v in weights.items():
                        optimal = optimal_weights[k]
                        change = optimal - v
                        direction = "increase" if change > 0 else "decrease"
                        
                        # Create random explanation
                        explanations = [
                            f"Historical performance suggests {direction}",
                            f"Recent market regime favors {direction}",
                            f"Correlation analysis indicates potential {direction}",
                            f"Optimization tests show improved results with {direction}"
                        ]
                        import random
                        explanation = random.choice(explanations)
                        
                        adjustments.append({
                            'Component': k,
                            'Current': v,
                            'Suggested': optimal,
                            'Change': change,
                            'Rationale': explanation
                        })
                    
                    adj_df = pd.DataFrame(adjustments)
                    
                    # Format for display
                    formatted_df = adj_df.copy()
                    formatted_df['Current'] = formatted_df['Current'].map('{:.4f}'.format)
                    formatted_df['Suggested'] = formatted_df['Suggested'].map('{:.4f}'.format)
                    formatted_df['Change'] = formatted_df['Change'].map('{:+.4f}'.format)
                    
                    st.dataframe(
                        formatted_df,
                        column_config={
                            "Change": st.column_config.ProgressColumn(
                                "Change",
                                format="%+.4f",
                                min_value=-0.5,
                                max_value=0.5
                            ),
                        },
                        hide_index=True
                    )
                    
                    # Add apply button
                    st.button(f"Apply Suggested {category} Weights", key=f"apply_{category}", type="primary")
        
        else:
            # Use actual strategy weights from the weighter
            st.info("Displaying actual strategy weights from the model")
            
            if hasattr(self.weighter, 'strategy_weights'):
                strategy_weights = self.weighter.strategy_weights
                
                # Convert strategy weights to dataframe
                strategy_df = pd.DataFrame({
                    'component': list(strategy_weights.keys()),
                    'weight': list(strategy_weights.values())
                })
                
                # Create visualizations
                fig = px.pie(
                    strategy_df,
                    values='weight',
                    names='component',
                    title=f"Strategy Weights Distribution"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create bar chart comparison
                if hasattr(self.weighter, 'optimal_strategy_weights'):
                    optimal_weights = self.weighter.optimal_strategy_weights
                    
                    comparison_data = []
                    for component in strategy_weights:
                        comparison_data.append({
                            'component': component,
                            'current': strategy_weights.get(component, 0),
                            'optimal': optimal_weights.get(component, 0)
                        })
                        
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig2 = px.bar(
                        comparison_df,
                        x='component',
                        y=['current', 'optimal'],
                        barmode='group',
                        title="Current vs Optimal Strategy Weights"
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
            elif hasattr(self.weighter, 'feature_importance'):
                # Handle feature importance visualization
                feature_importance = self.weighter.feature_importance
                
                # Convert to dataframe
                feature_df = pd.DataFrame({
                    'feature': list(feature_importance.keys()),
                    'importance': list(feature_importance.values())
                })
                
                # Sort by importance
                feature_df = feature_df.sort_values('importance', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    feature_df,
                    x='feature',
                    y='importance',
                    title='Feature Importance',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)