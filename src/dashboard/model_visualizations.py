"""
Comprehensive model visualization dashboard for ensemble time series models.

This module provides visualization tools to better understand model architectures,
performance metrics, and ensemble dynamics for various types of time series models
including traditional machine learning, deep learning, and specialized architectures.
"""

import os
from datetime import timedelta
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


class ModelVisualizationDashboard:
    """
    Comprehensive dashboard for visualizing ensemble model performance and architecture.
    
    Provides interactive visualizations for model weights, performance metrics,
    regime analysis, architectural details, and learning insights.
    """

    def __init__(self, ensemble_weighter, model_directory: str = "model_weights"):
        """
        Initialize visualization dashboard.

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
            "lstm": "#1f77b4",       # blue
            "rnn": "#ff7f0e",        # orange
            "xgboost": "#2ca02c",    # green
            "random_forest": "#d62728",  # red
            "cnn": "#9467bd",        # purple
            "nbeats": "#8c564b",     # brown
            "ltc": "#e377c2",        # pink
            "tabnet": "#bcbd22",     # olive
            "tft": "#7f7f7f",        # gray
        }

        # Collect error metrics if available
        self.error_metrics = {}
        if hasattr(self.weighter, "error_history"):
            self.error_metrics = self.weighter.error_history

    def render_dashboard(self):
        """
        Render the complete model visualization dashboard with all tabs.
        
        Creates and renders tabs for ensemble weights, model performance,
        regime analysis, model architecture, and learning insights.
        """
        st.title("Ensemble Model Visualization Dashboard")

        # Create tabs for different visualization groups
        tabs = st.tabs(
            [
                "Ensemble Weights",
                "Model Performance",
                "Regime Analysis",
                "Model Architecture",
                "Learning Insights",
            ]
        )

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
        """
        Render visualizations related to ensemble weights.
        
        Shows weight evolution over time, current weight distribution,
        and regime transitions when available.
        """
        st.header("Ensemble Weight Evolution")

        # Weight history plot
        if (
            hasattr(self.weighter, "historical_weights")
            and self.weighter.historical_weights
        ):
            # Create dataframe from weight history
            weights_data = []
            for i, weights in enumerate(self.weighter.historical_weights):
                for model, weight in weights.items():
                    weights_data.append(
                        {"timestep": i, "model": model, "weight": weight}
                    )

            if weights_data:
                weights_df = pd.DataFrame(weights_data)

                # Create the line chart
                fig = px.line(
                    weights_df,
                    x="timestep",
                    y="weight",
                    color="model",
                    color_discrete_map=self.model_colors,
                    title="Ensemble Weight Evolution Over Time",
                    labels={"timestep": "Time Steps", "weight": "Model Weight"},
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
                    title="Current Model Weight Distribution",
                )

                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

                # Show regime transitions if available
                if (
                    hasattr(self.weighter, "weight_change_reasons")
                    and self.weighter.weight_change_reasons
                ):
                    st.subheader("Market Regime Transitions")

                    # Create a dataframe for regime transitions
                    regime_data = []
                    for change in self.weighter.weight_change_reasons:
                        try:
                            regime_data.append(
                                {
                                    "timestamp": pd.to_datetime(
                                        change.get("timestamp", "2023-01-01")
                                    ),
                                    "regime": change.get("regime", "unknown"),
                                    "reason": change.get(
                                        "reason", "No reason provided"
                                    ),
                                }
                            )
                        except:
                            # Skip any problematic entries
                            pass

                    if regime_data:
                        regime_df = pd.DataFrame(regime_data)

                        # Show as plotly timeline
                        fig = px.timeline(
                            regime_df,
                            x_start="timestamp",
                            x_end=[
                                ts + timedelta(hours=12)
                                for ts in regime_df["timestamp"]
                            ],
                            y="regime",
                            color="regime",
                            hover_data=["reason"],
                            title="Market Regime Changes",
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
        if (
            hasattr(self.weighter, "model_correlation_matrix")
            and self.weighter.model_correlation_matrix
        ):
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
                color_continuous_scale="RdBu_r",
                title="Model Performance Correlation Matrix",
                labels=dict(x="Model", y="Model", color="Correlation"),
                zmin=-1,
                zmax=1,
            )

            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    def render_model_performance_tab(self):
        """
        Render visualizations related to model performance.
        
        Shows error over time, average error by model type, error distribution,
        and direction accuracy if available.
        """
        st.header("Model Performance Analysis")

        # Create error over time plot
        if self.error_metrics:
            # Create dataframe from error history
            error_data = []
            for model, errors in self.error_metrics.items():
                for i, error in enumerate(errors):
                    error_data.append({"timestep": i, "model": model, "error": error})

            if error_data:
                error_df = pd.DataFrame(error_data)

                # Allow log scale option for errors
                use_log_scale = st.checkbox("Use Log Scale for Errors", value=True)

                # Create the line chart
                fig = px.line(
                    error_df,
                    x="timestep",
                    y="error",
                    color="model",
                    color_discrete_map=self.model_colors,
                    title="Model Error Over Time",
                    labels={"timestep": "Time Steps", "error": "Error Metric"},
                    log_y=use_log_scale,
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Calculate average errors and create bar chart
                avg_errors = error_df.groupby("model")["error"].mean().reset_index()

                fig = px.bar(
                    avg_errors,
                    x="model",
                    y="error",
                    color="model",
                    color_discrete_map=self.model_colors,
                    title="Average Error by Model Type",
                    labels={"model": "Model Type", "error": "Avg Error"},
                    log_y=use_log_scale,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Create error distribution (by model)
                st.subheader("Error Distribution by Model")

                fig = px.box(
                    error_df,
                    x="model",
                    y="error",
                    color="model",
                    color_discrete_map=self.model_colors,
                    title="Error Distribution by Model Type",
                    log_y=use_log_scale,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show direction accuracy if available
                if (
                    hasattr(self.weighter, "direction_history")
                    and self.weighter.direction_history
                ):
                    st.subheader("Direction Prediction Accuracy")

                    # Create dataframe from direction accuracy history
                    direction_data = []
                    for model, accuracies in self.weighter.direction_history.items():
                        for i, accuracy in enumerate(accuracies):
                            direction_data.append(
                                {
                                    "timestep": i,
                                    "model": model,
                                    "accuracy": accuracy * 100,  # Convert to percentage
                                }
                            )

                    if direction_data:
                        direction_df = pd.DataFrame(direction_data)

                        # Create the line chart
                        fig = px.line(
                            direction_df,
                            x="timestep",
                            y="accuracy",
                            color="model",
                            color_discrete_map=self.model_colors,
                            title="Direction Prediction Accuracy Over Time",
                            labels={
                                "timestep": "Time Steps",
                                "accuracy": "Accuracy (%)",
                            },
                        )

                        # Add 50% line (random guessing)
                        fig.add_hline(
                            y=50,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Random Guessing",
                        )

                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate average accuracy
                        avg_acc = (
                            direction_df.groupby("model")["accuracy"]
                            .mean()
                            .reset_index()
                        )

                        fig = px.bar(
                            avg_acc,
                            x="model",
                            y="accuracy",
                            color="model",
                            color_discrete_map=self.model_colors,
                            title="Average Direction Accuracy by Model Type",
                            labels={
                                "model": "Model Type",
                                "accuracy": "Avg Accuracy (%)",
                            },
                        )

                        # Add 50% line (random guessing)
                        fig.add_hline(
                            y=50,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Random Guessing",
                        )

                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No error history data available yet")
        else:
            st.warning("No error metrics available in the ensemble weighter")

    def render_regime_analysis_tab(self):
        """
        Render visualizations related to market regime analysis.
        
        Shows current regime, regime performance if available, and 
        model-specific regime performance.
        """
        st.header("Market Regime Analysis")

        # Get current regime if available
        current_regime = "Unknown"
        if hasattr(self.weighter, "current_regime"):
            current_regime = self.weighter.current_regime

        # Display current regime in a big metric
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Current Market Regime", current_regime)

        # Regime performance if available
        if (
            hasattr(self.weighter, "optuna_feedback")
            and "regime_performance" in self.weighter.optuna_feedback
        ):
            regime_perf = self.weighter.optuna_feedback["regime_performance"]

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
                                weighted_error = perf.get("weighted_error", 0)
                                weights = perf.get("weights", {})

                                for model, weight in weights.items():
                                    regime_data.append(
                                        {
                                            "model": model,
                                            "weight": weight,
                                            "weighted_error": weighted_error,
                                        }
                                    )

                            if regime_data:
                                regime_df = pd.DataFrame(regime_data)

                                # Calculate average weight by model
                                avg_weights = (
                                    regime_df.groupby("model")["weight"]
                                    .mean()
                                    .reset_index()
                                )

                                fig = px.bar(
                                    avg_weights,
                                    x="model",
                                    y="weight",
                                    color="model",
                                    color_discrete_map=self.model_colors,
                                    title=f"Average Model Weights in {regime.capitalize()} Regime",
                                    labels={
                                        "model": "Model Type",
                                        "weight": "Avg Weight",
                                    },
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Show best weights for this regime
                                best_weights = None
                                best_error = float("inf")

                                for perf in performances:
                                    if (
                                        perf.get("weighted_error", float("inf"))
                                        < best_error
                                    ):
                                        best_error = perf.get("weighted_error")
                                        best_weights = perf.get("weights")

                                if best_weights:
                                    st.subheader(
                                        f"Best Weight Configuration for {regime.capitalize()} Regime"
                                    )
                                    st.write(f"Weighted Error: {best_error:.6f}")

                                    # Create pie chart for best weights
                                    fig = px.pie(
                                        values=list(best_weights.values()),
                                        names=list(best_weights.keys()),
                                        color=list(best_weights.keys()),
                                        color_discrete_map=self.model_colors,
                                        title=f"Best Weights for {regime.capitalize()} Regime",
                                    )

                                    fig.update_traces(
                                        textposition="inside", textinfo="percent+label"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"No performance data for {regime} regime")
                        else:
                            st.warning(f"No performance data for {regime} regime")

        # Model-specific regime performance
        if (
            hasattr(self.weighter, "optuna_feedback")
            and "model_performance" in self.weighter.optuna_feedback
        ):
            model_perf = self.weighter.optuna_feedback["model_performance"]

            if model_perf:
                st.subheader("Individual Model Performance by Regime")

                # Extract and reshape data
                regime_model_data = []

                for model, performances in model_perf.items():
                    for perf in performances:
                        if "error" in perf and "regime" in perf:
                            regime_model_data.append(
                                {
                                    "model": model,
                                    "regime": perf["regime"],
                                    "error": perf["error"],
                                    "weight": perf.get("weight", 0),
                                }
                            )

                if regime_model_data:
                    df = pd.DataFrame(regime_model_data)

                    # Calculate average error by model and regime
                    pivot_df = df.pivot_table(
                        index="model", columns="regime", values="error", aggfunc="mean"
                    ).reset_index()

                    # Convert to long format for plotting
                    plot_df = pd.melt(
                        pivot_df,
                        id_vars=["model"],
                        var_name="regime",
                        value_name="avg_error",
                    )

                    # Create grouped bar chart
                    fig = px.bar(
                        plot_df,
                        x="model",
                        y="avg_error",
                        color="regime",
                        title="Average Error by Model and Regime",
                        labels={
                            "model": "Model Type",
                            "avg_error": "Avg Error",
                            "regime": "Market Regime",
                        },
                        barmode="group",
                    )

                    use_log_scale = st.checkbox(
                        "Use Log Scale for Regime Analysis", value=True
                    )
                    if use_log_scale:
                        fig.update_layout(yaxis_type="log")

                    st.plotly_chart(fig, use_container_width=True)

    def render_model_architecture_tab(self):
        """
        Render visualizations related to model architecture.
        
        Allows users to select and visualize different model architectures
        including LSTM, RNN, XGBoost, Random Forest, CNN, NBEATS, LTC, 
        TabNet, and TFT.
        """
        st.header("Model Architecture Visualization")

        # Get model types and filter out 'transformer' if it exists since it's the same as 'tft'
        try:
            model_types = [
                model
                for model in self.weighter.base_weights.keys()
                if model != "transformer"
            ]
        except (AttributeError, TypeError):
            # In case weighter doesn't have base_weights
            model_types = []
        
        # ALWAYS include these model types even if they're not in base_weights
        required_models = ["lstm", "rnn", "xgboost", "random_forest", "cnn", "nbeats", "ltc", "tabnet", "tft"]
        
        # Add any missing models to ensure they always appear in the selection
        for model in required_models:
            if model not in model_types:
                model_types.append(model)

        # Select model to visualize
        selected_model = st.selectbox("Select Model to Visualize", sorted(model_types))

        # Architecture visualization based on model type
        if selected_model:
            st.subheader(f"{selected_model.upper()} Architecture")

            if selected_model == "lstm":
                self._visualize_lstm_architecture()
            elif selected_model == "rnn":
                self._visualize_rnn_architecture()
            elif selected_model == "xgboost":
                self._visualize_tree_architecture("XGBoost")
            elif selected_model == "random_forest":
                self._visualize_tree_architecture("Random Forest")
            elif selected_model == "cnn":
                self._visualize_cnn_architecture()
            elif selected_model == "nbeats":
                self._visualize_nbeats_architecture()
            elif selected_model == "ltc":
                self._visualize_ltc_architecture()
            elif selected_model == "tabnet":
                self._visualize_tabnet_architecture()
            elif selected_model == "tft":
                self._visualize_tft_architecture()
            else:
                st.warning(
                    f"Visualization not available for {selected_model} architecture"
                )

    def render_learning_insights_tab(self):
        """
        Render visualizations related to model learning process.
        
        Shows Optuna feedback, suggested adjustments, and learning trajectories.
        """
        st.header("Model Learning Insights")

        # Optuna feedback and suggestions
        if (
            hasattr(self.weighter, "optuna_feedback")
            and "suggested_adjustments" in self.weighter.optuna_feedback
        ):
            suggestions = self.weighter.optuna_feedback["suggested_adjustments"]

            if suggestions:
                st.subheader("Optuna Feedback & Suggestions")

                # Create dataframe from suggestions
                suggestion_data = []
                for sugg in suggestions:
                    suggestion_data.append(
                        {
                            "model": sugg.get("model", "unknown"),
                            "current_weight": sugg.get("current_base_weight", 0),
                            "suggested_weight": sugg.get("suggested_base_weight", 0),
                            "regime": sugg.get("regime", "all"),
                            "reason": sugg.get("reason", "No reason provided"),
                        }
                    )

                if suggestion_data:
                    sugg_df = pd.DataFrame(suggestion_data)

                    # Calculate weight change
                    sugg_df["weight_change"] = (
                        sugg_df["suggested_weight"] - sugg_df["current_weight"]
                    )
                    sugg_df["percent_change"] = (
                        sugg_df["weight_change"]
                        / sugg_df["current_weight"].replace(0, 0.0001)
                        * 100
                    )

                    # Create visual comparison
                    fig = go.Figure()

                    for i, row in sugg_df.iterrows():
                        model = row["model"]
                        current = row["current_weight"]
                        suggested = row["suggested_weight"]
                        color = self.model_colors.get(model, "#1f77b4")

                        # Add current weight
                        fig.add_trace(
                            go.Bar(
                                name=f"{model} (Current)",
                                x=[f"{model} ({row['regime']})"],
                                y=[current],
                                marker_color=color,
                                opacity=0.6,
                                width=0.3,
                                offset=-0.2,
                                text=[f"{current:.3f}"],
                                textposition="outside",
                            )
                        )

                        # Add suggested weight
                        fig.add_trace(
                            go.Bar(
                                name=f"{model} (Suggested)",
                                x=[f"{model} ({row['regime']})"],
                                y=[suggested],
                                marker_color=color,
                                width=0.3,
                                offset=0.2,
                                text=[f"{suggested:.3f}"],
                                textposition="outside",
                            )
                        )

                    fig.update_layout(
                        title="Weight Adjustment Suggestions from Optuna",
                        barmode="group",
                        height=400 + len(sugg_df) * 30,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show table with reasoning
                    st.dataframe(
                        sugg_df[
                            [
                                "model",
                                "regime",
                                "current_weight",
                                "suggested_weight",
                                "percent_change",
                                "reason",
                            ]
                        ]
                    )

                    # Visualize weight changes as arrow chart
                    st.subheader("Weight Adjustment Direction")

                    fig = go.Figure()

                    for i, row in sugg_df.iterrows():
                        model = row["model"]
                        current = row["current_weight"]
                        change = row["weight_change"]
                        color = "green" if change > 0 else "red"

                        fig.add_trace(
                            go.Scatter(
                                x=[0, change],
                                y=[i, i],
                                mode="lines+markers",
                                name=f"{model} ({row['regime']})",
                                line=dict(color=color, width=3),
                                marker=dict(size=10),
                            )
                        )

                        # Add model name as annotation
                        fig.add_annotation(
                            x=0,
                            y=i,
                            text=f"{model} ({row['regime']})",
                            showarrow=False,
                            xanchor="right",
                            xshift=-10,
                        )

                    fig.update_layout(
                        title="Weight Adjustment Direction and Magnitude",
                        xaxis_title="Weight Change",
                        showlegend=False,
                        height=400 + len(sugg_df) * 30,
                        xaxis=dict(
                            zeroline=True, zerolinewidth=2, zerolinecolor="black"
                        ),
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
                    rolling = np.array(
                        [
                            np.mean(err_array[max(0, i - window_size) : i + 1])
                            for i in range(len(err_array))
                        ]
                    )
                    rolling_errors[model] = rolling

            if rolling_errors:
                # Create plot
                fig = go.Figure()

                for model, errors in rolling_errors.items():
                    color = self.model_colors.get(model, "#1f77b4")

                    # Calculate learning rate
                    if len(errors) > 1:
                        # Simple linear regression for trend
                        x = np.arange(len(errors))
                        z = np.polyfit(x, np.log(errors + 1e-10), 1)
                        slope = z[0]
                        learning_rate = (
                            -slope
                        )  # Negative slope = positive learning rate

                        # Add trace with learning rate in name
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=errors,
                                mode="lines",
                                name=f"{model} (LR: {learning_rate:.4f})",
                                line=dict(color=color),
                            )
                        )

                fig.update_layout(
                    title="Model Learning Trajectories (Rolling Average Error)",
                    xaxis_title="Training Steps",
                    yaxis_title="Error (Rolling Avg)",
                    legend_title="Model Types",
                    height=500,
                )

                # Allow log scale option
                use_log_scale = st.checkbox(
                    "Use Log Scale for Learning Curves", value=True
                )
                if use_log_scale:
                    fig.update_layout(yaxis_type="log")

                st.plotly_chart(fig, use_container_width=True)

    def _visualize_lstm_architecture(self):
        """
        Visualize LSTM architecture with gates and information flow.
        
        Shows the LSTM cell structure, key components, and advantages
        for time series prediction.
        """
        st.markdown(
            """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">Long Short-Term Memory (LSTM) Architecture</h3>
            <p style="text-align: center; color: #34495E;">LSTM uses gates to control information flow over time, making it suitable for time series prediction</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

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
            st.graphviz_chart(lstm_diagram)
        except Exception as e:
            st.warning(f"Unable to render graph visualization: {e}")
            st.code(lstm_diagram, language="dot")

        st.markdown(
            """
        ### <span style="color:#2C3E50; font-weight:bold;">LSTM Key Components:</span>
        
        1. <span style="color:#154360; font-weight:bold;">Forget Gate:</span> Decides what information to discard from cell state
        2. <span style="color:#154360; font-weight:bold;">Input Gate:</span> Updates the cell state with new information
        3. <span style="color:#154360; font-weight:bold;">Cell State:</span> Long-term memory component 
        4. <span style="color:#154360; font-weight:bold;">Output Gate:</span> Controls what information to output
        
        ### <span style="color:#2C3E50; font-weight:bold;">Advantages for Time Series:</span>
        
        - <span style="color:#1E8449;">Handles long-term dependencies</span> in data
        - <span style="color:#1E8449;">Avoids vanishing gradient problem</span>
        - <span style="color:#1E8449;">Can remember patterns</span> over many time steps
        - <span style="color:#1E8449;">Effective for price prediction</span>
        
        ### <span style="color:#2C3E50; font-weight:bold;">Typical Architecture:</span>
        
        - Input layer → LSTM layer(s) → Dense layer(s) → Output
        - Often includes dropout for regularization
        - Can be stacked for more complex patterns
        """,
            unsafe_allow_html=True
        )

    def _visualize_rnn_architecture(self):
        """
        Visualize RNN architecture with recurrent connections.
        
        Shows the RNN cell structure, unrolled network, characteristics,
        and limitations for time series prediction.
        """
        st.markdown(
            """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">Recurrent Neural Network (RNN) Architecture</h3>
            <p style="text-align: center; color: #34495E;">Simple but powerful architecture for processing sequential data</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

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
            st.graphviz_chart(rnn_diagram)
        except Exception as e:
            st.warning(f"Unable to render graph visualization: {e}")
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
            st.graphviz_chart(unrolled_rnn)
        except Exception as e:
            st.warning(f"Unable to render graph visualization: {e}")
            st.code(unrolled_rnn, language="dot")

        st.markdown(
            """
        ### <span style="color:#2C3E50; font-weight:bold;">RNN Characteristics:</span>
        
        1. <span style="color:#154360; font-weight:bold;">Recurrent Connection:</span> Hidden state is fed back into the network
        2. <span style="color:#154360; font-weight:bold;">Shared Parameters:</span> Same weights used at each time step
        3. <span style="color:#154360; font-weight:bold;">Memory:</span> Limited short-term memory capability
        
        ### <span style="color:#2C3E50; font-weight:bold;">Limitations:</span>
        
        - <span style="color:#922B21;">Struggles with long-term dependencies</span> due to vanishing gradient
        - <span style="color:#922B21;">Less powerful than LSTM</span> for complex patterns
        - <span style="color:#922B21;">Works best for shorter time windows</span>
        
        ### <span style="color:#2C3E50; font-weight:bold;">Common Uses:</span>
        
        - Baseline for time series prediction
        - Simpler patterns in market data
        - When computational efficiency is important
        """,
            unsafe_allow_html=True
        )

    def _visualize_tree_architecture(self, model_type):
        """
        Visualize tree-based model architecture.
        
        Args:
            model_type: Either "XGBoost" or "Random Forest"
            
        Shows the tree structure, ensemble approach, and feature importance
        for the specified tree-based model.
        """
        is_xgboost = model_type == "XGBoost"

        st.markdown(
            f"""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">{model_type} Architecture</h3>
            <p style="text-align: center; color: #34495E;">Tree-based ensemble method that excels at capturing non-linear patterns</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

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
                st.graphviz_chart(tree_diagram)
            except Exception as e:
                st.warning(f"Unable to render graph visualization: {e}")
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
                        tree1 [label="Tree 1\\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
                        tree2 [label="Tree 2\\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
                        tree3 [label="Tree 3\\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
                        dots [label="...", shape=plaintext];
                        treeN [label="Tree N\\nBootstrap Sample", shape=triangle, style=filled, fillcolor=lightblue];
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
                st.graphviz_chart(ensemble_diagram)
            except Exception as e:
                st.warning(f"Unable to render graph visualization: {e}")
                st.code(ensemble_diagram, language="dot")

        st.markdown(
            f"""
        ### <span style="color:#2C3E50; font-weight:bold;">{model_type} Key Characteristics:</span>
        
        1. <span style="color:#154360; font-weight:bold;">{'Boosting' if is_xgboost else 'Bagging'}:</span> 
           {'Each tree corrects errors of previous trees' if is_xgboost else 'Each tree is built independently on different bootstrap samples'}
        
        2. <span style="color:#154360; font-weight:bold;">Tree Structure:</span> Decision rules based on feature thresholds
        
        3. <span style="color:#154360; font-weight:bold;">{'Gradient Descent' if is_xgboost else 'Random Feature Selection'}:</span> 
           {'Minimizes loss function using gradient information' if is_xgboost else 'Each tree considers a random subset of features'}
        
        ### <span style="color:#2C3E50; font-weight:bold;">Advantages for Price Prediction:</span>
        
        - <span style="color:#1E8449;">Captures non-linear relationships</span> in price data
        - <span style="color:#1E8449;">Robust to outliers</span> and market anomalies
        - <span style="color:#1E8449;">Identifies important features</span> automatically
        - <span style="color:#1E8449;">{'Excellent for trend detection' if is_xgboost else 'Less prone to overfitting'}</span>
        
        ### <span style="color:#2C3E50; font-weight:bold;">Feature Importance:</span>
        
        {model_type} can identify the most important features for prediction, such as:
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Volume metrics
        - Market sentiment scores
        - Historical price patterns
        """,
            unsafe_allow_html=True
        )

        # Add sample feature importance plot
        importance_data = {
            "RSI": 0.23,
            "Volume": 0.18,
            "MACD": 0.15,
            "Bollinger_Width": 0.12,
            "Price_Change": 0.10,
            "Volatility": 0.08,
            "MA_Crossover": 0.07,
            "Sentiment": 0.07,
        }

        importance_df = pd.DataFrame(
            {
                "Feature": list(importance_data.keys()),
                "Importance": list(importance_data.values()),
            }
        ).sort_values("Importance", ascending=False)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Sample {model_type} Feature Importance",
            color="Importance",
            color_continuous_scale="Viridis",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _visualize_cnn_architecture(self):
        """
        Visualize CNN architecture for time series.
        
        Shows the CNN structure for time series data, key components,
        advantages, and sample filter activations.
        """
        st.markdown(
            """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">Convolutional Neural Network (CNN) for Time Series</h3>
            <p style="text-align: center; color: #34495E;">Using 1D convolutions to detect patterns in sequential financial data</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        
        # Add a structured diagram for CNN
        cnn_diagram = """
        digraph G {
            rankdir=LR;
            
            // Input
            subgraph cluster_input {
                label="Input";
                style=filled;
                color=lightgrey;
                node [style=filled, fillcolor="#AED6F1"];
                
                input [label="Time Series\\nInput\\n[batch, seq_len, features]"];
            }
            
            // Conv1D layers
            subgraph cluster_conv {
                label="Convolutional Layers";
                style=filled;
                color=lightgrey;
                node [style=filled, fillcolor="#A2D9CE"];
                
                conv1 [label="Conv1D\\nKernel Size = 3\\nStride = 1"];
                conv2 [label="Conv1D\\nKernel Size = 3\\nStride = 1"];
            }
            
            // Pooling
            subgraph cluster_pooling {
                label="Pooling";
                style=filled;
                color=lightgrey;
                node [style=filled, fillcolor="#F9E79F"];
                
                pool [label="MaxPooling1D\\nPool Size = 2"];
            }
            
            // Flattening
            subgraph cluster_flatten {
                label="Reshape";
                style=filled;
                color=lightgrey;
                node [style=filled, fillcolor="#F5B7B1"];
                
                flat [label="Flatten"];
            }
            
            // Dense layers
            subgraph cluster_dense {
                label="Dense Layers";
                style=filled;
                color=lightgrey;
                node [style=filled, fillcolor="#D7BDE2"];
                
                dense1 [label="Dense\\nReLU"];
                dense2 [label="Dense\\nReLU"];
            }
            
            // Output
            subgraph cluster_output {
                label="Output";
                style=filled;
                color=lightgrey;
                node [style=filled, fillcolor="#85C1E9"];
                
                output [label="Predictions"];
            }
            
            // Connections
            input -> conv1;
            conv1 -> conv2;
            conv2 -> pool;
            pool -> flat;
            flat -> dense1;
            dense1 -> dense2;
            dense2 -> output;
        }
        """

        try:
            st.graphviz_chart(cnn_diagram)
        except Exception as e:
            st.warning(f"Unable to render graph visualization: {e}")
            st.code(cnn_diagram, language="dot")

        st.markdown(
            """
        ### <span style="color:#2C3E50; font-weight:bold;">CNN Key Components:</span>
        
        1. <span style="color:#154360; font-weight:bold;">Convolutional Layers:</span> Slide filters over time steps to detect patterns
           - Each filter learns to recognize specific temporal patterns
           - Shared weights across time reduce parameters
        
        2. <span style="color:#154360; font-weight:bold;">Pooling Layers:</span> Downsample outputs to extract the most important features
           - Reduces dimensionality
           - Provides some translation invariance
        
        3. <span style="color:#154360; font-weight:bold;">Dense Layers:</span> Combine extracted features for final prediction
           - Non-linear combinations of detected patterns
           - Final decision making
        
        ### <span style="color:#2C3E50; font-weight:bold;">Advantages for Price Prediction:</span>
        
        - <span style="color:#1E8449;">Automatically detects local patterns</span> in price movements
        - <span style="color:#1E8449;">Robust to time shifts</span> (same pattern at different times)
        - <span style="color:#1E8449;">Efficiently handles multivariate inputs</span> (price, volume, indicators)
        - <span style="color:#1E8449;">Can capture both short-term and medium-term dependencies</span>
        """,
            unsafe_allow_html=True,
        )

        # Show sample CNN filters visualization
        st.subheader("CNN Filter Visualization")

        # Create sample filter activations
        import numpy as np
        import pandas as pd
        import plotly.express as px
        
        np.random.seed(42)
        
        # Sample time series
        time_steps = 100
        x = np.linspace(0, 10, time_steps)
        y = np.sin(x) + 0.1 * np.random.randn(time_steps)

        # Three sample filter activations
        act1 = np.convolve(y, [0.2, 0.5, 0.3], mode="valid")
        act2 = np.convolve(y, [-0.3, 0.1, 0.7], mode="valid")
        act3 = np.convolve(y, [0, -0.5, 0.5], mode="valid")

        df = pd.DataFrame(
            {
                "time": x,
                "price": y,
                "filter1": np.pad(act1, (1, 1), "constant"),
                "filter2": np.pad(act2, (1, 1), "constant"),
                "filter3": np.pad(act3, (1, 1), "constant"),
            }
        )

        # Use more distinct colors for better visibility
        custom_colors = {
            'price': '#2C3E50',
            'filter1': '#E74C3C',
            'filter2': '#2ECC71',
            'filter3': '#3498DB'
        }
        
        fig = px.line(
            df,
            x="time",
            y=["price", "filter1", "filter2", "filter3"],
            title="CNN Filters Detecting Different Patterns",
            labels={"value": "Activation", "variable": "Signal Type"},
            color_discrete_map=custom_colors
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of filters with better formatting
        st.markdown(
            """
            <div style="background-color: #EBF5FB; padding: 15px; border-radius: 5px; border-left: 5px solid #3498DB;">
            <span style="font-weight: bold; color: #2C3E50;">The chart above demonstrates how different CNN filters activate:</span><br><br>
            
            - <span style="color: #2C3E50; font-weight: bold;">Original price series</span> (dark blue): The input time series data
            - <span style="color: #E74C3C; font-weight: bold;">Filter 1</span> (red): Detects upward trends ([0.2, 0.5, 0.3])
            - <span style="color: #2ECC71; font-weight: bold;">Filter 2</span> (green): Detects local extrema ([-0.3, 0.1, 0.7])
            - <span style="color: #3498DB; font-weight: bold;">Filter 3</span> (blue): Detects sudden changes ([0, -0.5, 0.5])
            
            Each filter responds to different patterns in the price data, allowing the CNN to recognize multiple types of market conditions.
            </div>
            """,
            unsafe_allow_html=True
        )

    def _visualize_nbeats_architecture(self):
        """
        Visualize N-BEATS architecture.
        
        Shows the N-BEATS block structure, key components, basis functions,
        and advantages for time series forecasting.
        """
        st.markdown(
            """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">N-BEATS Architecture</h3>
            <p style="text-align: center; color: #34495E;">Neural Basis Expansion Analysis for interpretable Time Series forecasting</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

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
            st.graphviz_chart(nbeats_diagram)
        except Exception as e:
            st.warning(f"Unable to render graph visualization: {e}")
            st.code(nbeats_diagram, language="dot")

        st.markdown(
            """
        ### <span style="color:#2C3E50; font-weight:bold;">N-BEATS Key Components:</span>
        
        1. <span style="color:#154360; font-weight:bold;">Block Structure:</span> Each block produces both backcast (reconstruction) and forecast outputs
        
        2. <span style="color:#154360; font-weight:bold;">Double Residual Stacking:</span> Each block processes residuals from previous blocks
        
        3. <span style="color:#154360; font-weight:bold;">Basis Functions:</span> 
           - Trend block: polynomial basis functions
           - Seasonality block: Fourier basis functions
           - Generic block: learned basis functions
        
        ### <span style="color:#2C3E50; font-weight:bold;">Advantages for Price Prediction:</span>
        
        - <span style="color:#1E8449;">Interpretable components</span> (can separate trend and seasonality)
        - <span style="color:#1E8449;">Handles multiple seasonalities</span> (daily, weekly patterns)
        - <span style="color:#1E8449;">Pure deep learning approach</span> (no feature engineering required)
        - <span style="color:#1E8449;">Strong performance for multi-step forecasting</span>
        
        ### <span style="color:#2C3E50; font-weight:bold;">Stack Types:</span>
        
        - <span style="color:#8E44AD;">Generic Stack:</span> Learns patterns directly from data
        - <span style="color:#8E44AD;">Trend Stack:</span> Specialized for trend components
        - <span style="color:#8E44AD;">Seasonality Stack:</span> Specialized for cyclical patterns
        
        N-BEATS often outperforms traditional models by combining interpretability with deep learning power.
        """,
            unsafe_allow_html=True
        )

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
        seasonal2 = np.sin(2 * x)
        seasonal3 = np.sin(4 * x)

        # Create dataframe
        basis_df = pd.DataFrame(
            {
                "x": x,
                "Linear Trend": trend1,
                "Quadratic Trend": trend2,
                "Cubic Trend": trend3,
                "Daily Seasonality": seasonal1,
                "Half-day Seasonality": seasonal2,
                "6-hour Seasonality": seasonal3,
            }
        )

        # Plot trend bases with better colors
        trend_colors = {
            "Linear Trend": "#E74C3C",
            "Quadratic Trend": "#3498DB",
            "Cubic Trend": "#2ECC71"
        }
        
        fig1 = px.line(
            basis_df,
            x="x",
            y=["Linear Trend", "Quadratic Trend", "Cubic Trend"],
            title="Trend Basis Functions",
            labels={"value": "Amplitude", "variable": "Basis Type"},
            color_discrete_map=trend_colors
        )

        # Plot seasonal bases with better colors
        seasonal_colors = {
            "Daily Seasonality": "#9B59B6",
            "Half-day Seasonality": "#F39C12",
            "6-hour Seasonality": "#1ABC9C"
        }
        
        fig2 = px.line(
            basis_df,
            x="x",
            y=["Daily Seasonality", "Half-day Seasonality", "6-hour Seasonality"],
            title="Seasonality Basis Functions",
            labels={"value": "Amplitude", "variable": "Basis Type"},
            color_discrete_map=seasonal_colors
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)

    def _visualize_ltc_architecture(self):
        """
        Visualize Liquid Time-Constant (LTC) Networks architecture.
        
        Shows the LTC network structure, time constants, and dynamics
        for financial time series prediction.
        """
        st.markdown(
            """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">Liquid Time-Constant (LTC) Networks</h3>
            <p style="text-align: center; color: #34495E;">Advanced specialized neural network with learnable timescales, distinct from standard RNNs</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # LTC diagram
        ltc_diagram = """
        digraph G {
            rankdir=LR;
            
            node [shape=circle, style=filled, color=black, fillcolor="#AED6F1"];
            
            input [label="Input", shape=plaintext];
            
            subgraph cluster_ltc {
                label="LTC Layer";
                style=filled;
                color=lightgrey;
                
                neuron1 [label="τ₁", fontcolor=black];
                neuron2 [label="τ₂", fontcolor=black];
                neuron3 [label="τ₃", fontcolor=black];
                neuron4 [label="...", fontcolor=black];
                neuron5 [label="τₙ", fontcolor=black];
                
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
            neuron1 -> neuron1 [label="τ₁", fontcolor=black];
            neuron3 -> neuron3 [label="τ₃", fontcolor=black];
            neuron5 -> neuron5 [label="τₙ", fontcolor=black];
        }
        """

        try:
            st.graphviz_chart(ltc_diagram)
        except Exception as e:
            st.warning(f"Unable to render graph visualization: {e}")
            st.code(ltc_diagram, language="dot")

        st.markdown(
            """
        ### <span style="color:#2C3E50; font-weight:bold;">LTC Network Key Concepts:</span>
        
        1. <span style="color:#154360; font-weight:bold;">Time Constants (τ):</span> Each neuron has its own learnable time constant
           - Controls how fast/slow the neuron responds to inputs
           - Different time scales capture different market dynamics
        
        2. <span style="color:#154360; font-weight:bold;">Continuous-Time Dynamics:</span> Uses differential equations instead of discrete updates
           - More natural for modeling market time series with variable sample rates
        
        3. <span style="color:#154360; font-weight:bold;">Liquid State Machines:</span> Inspired by biological neural networks
           - Better at handling complex non-linear dynamics
        
        ### <span style="color:#2C3E50; font-weight:bold;">Advantages for Financial Prediction:</span>
        
        - <span style="color:#1E8449;">Multiple timescales</span> simultaneously (minutes, hours, days)
        - <span style="color:#1E8449;">Handles irregular sampling</span> and missing data gracefully
        - <span style="color:#1E8449;">Captures both fast</span> (volatility) and <span style="color:#1E8449;">slow</span> (trend) market dynamics
        - <span style="color:#1E8449;">More biologically plausible</span> learning dynamics
        """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
        <div style="background-color: #FADBD8; padding: 15px; border-radius: 5px; border-left: 5px solid #E74C3C;">
        <span style="font-weight: bold; color: #2C3E50;">Important clarification:</span> Although LTC networks belong to the recurrent neural network family, they are a distinct architecture with significant differences:
        <ul style="margin-top: 10px; color: #2C3E50;">
          <li><span style="font-weight: bold;">Traditional RNNs/LSTM/GRU:</span> Use discrete update steps and gates to control information flow</li>
          <li><span style="font-weight: bold;">LTC Networks:</span> Use continuous-time dynamics with learnable time constants for each neuron</li>
          <li>LTC models have better theoretical properties for capturing complex dynamical systems</li>
          <li>Often more parameter-efficient than traditional RNNs</li>
          <li>Specifically designed for multi-scale temporal patterns in financial data</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True
        )

        # Time constant visualization
        st.subheader("LTC Time Constants Visualization")

        # Sample data for time constants
        import numpy as np
        import pandas as pd
        import plotly.express as px
        
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
            dt = x[i] - x[i - 1]
            y1[i] = y1[i - 1] + dt * (input_signal[i - 1] - y1[i - 1]) / tau1
            y2[i] = y2[i - 1] + dt * (input_signal[i - 1] - y2[i - 1]) / tau2
            y3[i] = y3[i - 1] + dt * (input_signal[i - 1] - y3[i - 1]) / tau3

        # Plot responses
        tau_df = pd.DataFrame(
            {
                "Time": x,
                "Input": input_signal,
                f"τ = {tau1} (Fast)": y1,
                f"τ = {tau2} (Medium)": y2,
                f"τ = {tau3} (Slow)": y3,
            }
        )

        # Custom colors for better visibility
        colors = {
            "Input": "#000000",
            f"τ = {tau1} (Fast)": "#E74C3C",
            f"τ = {tau2} (Medium)": "#3498DB",
            f"τ = {tau3} (Slow)": "#2ECC71",
        }

        fig = px.line(
            tau_df,
            x="Time",
            y=[
                "Input",
                f"τ = {tau1} (Fast)",
                f"τ = {tau2} (Medium)",
                f"τ = {tau3} (Slow)",
            ],
            title="Effect of Different Time Constants on Neuron Response",
            labels={"value": "Response", "variable": "Signal Type"},
            color_discrete_map=colors,
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
        <div style="background-color: #EBF5FB; padding: 15px; border-radius: 5px; border-left: 5px solid #3498DB;">
        <span style="font-weight: bold; color: #2C3E50;">The chart above demonstrates how different time constants affect neuron responses:</span><br><br>
        
        - <span style="color: #E74C3C; font-weight: bold;">Fast neurons</span> (small τ): Respond quickly to market changes but are sensitive to noise
        - <span style="color: #3498DB; font-weight: bold;">Medium neurons</span>: Balance between responsiveness and stability
        - <span style="color: #2ECC71; font-weight: bold;">Slow neurons</span> (large τ): Filter out noise but may miss sudden market movements
        
        LTC Networks learn the optimal time constants for each neuron automatically during training.
        </div>
        """,
            unsafe_allow_html=True
        )
        
        # Add timescale distribution visualization
        st.subheader("Learned Timescale Distribution")
        
        # Generate sample timescale distribution
        np.random.seed(42)
        
        # Create different distributions for different neuron groups
        fast_neurons = np.random.lognormal(mean=-1.5, sigma=0.4, size=30)
        medium_neurons = np.random.lognormal(mean=0, sigma=0.3, size=40)
        slow_neurons = np.random.lognormal(mean=1.0, sigma=0.5, size=20)
        
        # Combine distributions
        all_timescales = np.concatenate([fast_neurons, medium_neurons, slow_neurons])
        
        # Create histogram with more discernible colors
        timescale_df = pd.DataFrame({"Timescale": all_timescales})
        
        fig = px.histogram(
            timescale_df,
            x="Timescale",
            nbins=30,
            title="Distribution of Learned Time Constants (τ)",
            color_discrete_sequence=["#1ABC9C"],
        )
        
        fig.update_layout(
            xaxis_title="Time Constant (τ)",
            yaxis_title="Count",
            height=400,
        )
        
        # Add vertical lines for typical timescales with better colors
        fig.add_vline(x=0.5, line_dash="dash", line_color="#E74C3C", 
                     annotation_text="Fast", annotation_position="top right")
        fig.add_vline(x=2.0, line_dash="dash", line_color="#3498DB", 
                     annotation_text="Medium", annotation_position="top right")
        fig.add_vline(x=5.0, line_dash="dash", line_color="#2ECC71", 
                     annotation_text="Slow", annotation_position="top right")
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _visualize_tabnet_architecture(self):
        """
        Visualize TabNet architecture for tabular data.
        
        Shows the TabNet structure, feature selection mechanism, decision steps,
        and advantages for price prediction with tabular data.
        """
        st.markdown(
            """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">TabNet Architecture</h3>
            <p style="text-align: center; color: #34495E;">Deep learning for tabular data with feature selection and interpretability</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # TabNet diagram
        tabnet_diagram = """
        digraph G {
            rankdir=LR;
            
            subgraph cluster_input {
                label="Input";
                node [style=filled, fillcolor="#AED6F1"];
                features [label="Tabular Features", fontcolor=black];
            }
            
            subgraph cluster_feature_transformer {
                label="Feature Transformer";
                node [style=filled, fillcolor="#A2D9CE"];
                ft [label="Feature Transformer Network", fontcolor=black];
            }
            
            subgraph cluster_attentive_transformer {
                label="Attentive Transformer";
                node [style=filled, fillcolor="#F9E79F"];
                at [label="Attentive Transformer", fontcolor=black];
                masks [label="Feature Masks", fontcolor=black];
            }
            
            subgraph cluster_decision_steps {
                label="Decision Steps";
                node [style=filled, fillcolor="#F5B7B1"];
                step1 [label="Step 1", fontcolor=black];
                step2 [label="Step 2", fontcolor=black];
                stepN [label="Step N", fontcolor=black];
                decision_agg [label="Aggregated Decisions", fontcolor=black];
            }
            
            subgraph cluster_output {
                label="Output";
                node [style=filled, fillcolor="#D7BDE2"];
                output [label="Predictions", fontcolor=black];
            }
            
            // Connections
            features -> ft;
            ft -> at;
            at -> masks;
            masks -> ft [label="Masked Features"];
            
            masks -> step1;
            step1 -> step2;
            step2 -> stepN [style=dotted];
            stepN -> decision_agg;
            decision_agg -> output;
        }
        """

        # Display diagram using graphviz
        try:
            st.graphviz_chart(tabnet_diagram)
        except Exception as e:
            st.warning(f"Unable to render graph visualization: {e}")
            st.code(tabnet_diagram, language="dot")

        # Explanation
        st.markdown(
            """
        ### <span style="color:#2C3E50; font-weight:bold;">TabNet Key Components:</span>
        
        1. <span style="color:#154360; font-weight:bold;">Feature Transformer:</span> Processes input features using shared blocks and specific blocks.
        
        2. <span style="color:#154360; font-weight:bold;">Attentive Transformer:</span> Creates sparse feature selection masks at each decision step, 
           controlling which features are used and how much attention they receive.
        
        3. <span style="color:#154360; font-weight:bold;">Decision Steps:</span> Sequential processing steps that build upon previous steps' results,
           with each step focusing on different aspects of the data.
        
        4. <span style="color:#154360; font-weight:bold;">Feature Selection:</span> Learns which features are important for each decision step, providing
           interpretability and reducing noise.
        
        ### <span style="color:#2C3E50; font-weight:bold;">Advantages for Price Prediction:</span>
        
        - <span style="color:#1E8449;">Interpretability:</span> Shows which features are important at each decision step
        - <span style="color:#1E8449;">Learning Efficiency:</span> Uses fewer parameters than fully-connected networks
        - <span style="color:#1E8449;">Feature Selection:</span> Automatically focuses on relevant features
        - <span style="color:#1E8449;">Regularization:</span> Sparse feature selection provides implicit regularization
        - <span style="color:#1E8449;">Local Feature Interactions:</span> Captures complex interactions within tabular data
        
        ### <span style="color:#2C3E50; font-weight:bold;">Unique Properties:</span>
        
        - <span style="color:#8E44AD;">Instance-wise Feature Selection:</span> Different samples can use different features
        - <span style="color:#8E44AD;">Self-supervised Pretraining:</span> Can be pretrained on unlabeled data
        - <span style="color:#8E44AD;">Dual Optimization:</span> Simultaneously optimizes prediction quality and sparsity
        """,
            unsafe_allow_html=True
        )

        # Create feature importance visualization
        st.subheader("TabNet Feature Selection Visualization")

        # Sample data for feature importance over steps
        import numpy as np
        import pandas as pd
        import plotly.express as px
        
        # Generate sample feature importances for 10 features across 4 decision steps
        np.random.seed(42)
        feature_names = [f"Feature {i+1}" for i in range(10)]
        steps = ["Step 1", "Step 2", "Step 3", "Step 4"]
        
        # Create sample importance data with some features being important in different steps
        importances = np.zeros((len(feature_names), len(steps)))
        for i in range(len(steps)):
            # Make some features more important than others in each step
            step_importances = np.random.dirichlet(np.ones(len(feature_names)) * 0.5)
            importances[:, i] = step_importances
        
        # Create DataFrame for visualization
        importance_data = []
        for i, feature in enumerate(feature_names):
            for j, step in enumerate(steps):
                importance_data.append({
                    "Feature": feature,
                    "Decision Step": step,
                    "Importance": importances[i, j]
                })
        
        df = pd.DataFrame(importance_data)
        
        # Create heatmap
        fig = px.imshow(
            importances,
            x=steps,
            y=feature_names,
            color_continuous_scale="Blues",
            title="Feature Importance by Decision Step",
            labels=dict(x="Decision Step", y="Feature", color="Importance")
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Decision flow visualization
        st.subheader("TabNet Decision Flow")
        
        # Create sample decision flow data
        sample_data = pd.DataFrame({
            "Decision Step": ["Initial"] + steps,
            "Cumulative Accuracy": [0.5, 0.7, 0.8, 0.85, 0.89],
            "Step Contribution": [0.5, 0.2, 0.1, 0.05, 0.04]
        })
        
        # Create bar chart
        fig = px.bar(
            sample_data,
            x="Decision Step",
            y="Step Contribution",
            title="Contribution of Each Decision Step to Overall Accuracy",
            text_auto='.2f',
            color="Step Contribution",
            color_continuous_scale="Teal",
        )
        
        # Add line for cumulative accuracy
        fig.add_scatter(
            x=sample_data["Decision Step"],
            y=sample_data["Cumulative Accuracy"],
            mode="lines+markers",
            name="Cumulative Accuracy",
            line=dict(color="red", width=2),
            yaxis="y2"
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            yaxis=dict(title="Step Contribution"),
            yaxis2=dict(
                title="Cumulative Accuracy",
                overlaying="y",
                side="right",
                range=[0, 1]
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _visualize_tft_architecture(self):
        """
        Visualize Temporal Fusion Transformer architecture.
        
        Shows the TFT structure, attention mechanism, gating, and
        interpretability features for time series prediction.
        """
        st.markdown(
            """
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
            <h3 style="text-align: center; color: #2C3E50;">Temporal Fusion Transformer (TFT)</h3>
            <p style="text-align: center; color: #34495E;">State-of-the-art architecture combining attention mechanisms with specialized components for time series</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # TFT architecture diagram
        st.markdown("### TFT Architecture Overview")

        col1, col2 = st.columns([3, 1])

        with col1:
            tft_diagram = """
            digraph G {
                rankdir=TB;
                
                // Inputs
                past [label="Past Inputs", shape=box, style=filled, fillcolor=lightblue, fontcolor=black];
                known [label="Known Future Inputs", shape=box, style=filled, fillcolor=lightgreen, fontcolor=black];
                
                // Processing layers
                gating [label="Variable Selection Networks", shape=box, style=filled, fillcolor=lightyellow, fontcolor=black];
                static [label="Static Covariate Encoders", shape=box, style=filled, fillcolor=lightyellow, fontcolor=black];
                lstm_encoder [label="LSTM Encoder", shape=box, style=filled, fillcolor=pink, fontcolor=black];
                lstm_decoder [label="LSTM Decoder", shape=box, style=filled, fillcolor=pink, fontcolor=black];
                
                // Attention mechanism
                attention [label="Multi-head Attention", shape=box, style=filled, fillcolor=lightcoral, fontcolor=black];
                
                // Output
                temporal_fusion [label="Temporal Fusion Layer", shape=box, style=filled, fillcolor=lightgrey, fontcolor=black];
                position_wise [label="Position-wise Feed-forward", shape=box, style=filled, fillcolor=lightgrey, fontcolor=black];
                quantile [label="Quantile Outputs", shape=box, style=filled, fillcolor=lightblue, fontcolor=black];
                
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
                st.graphviz_chart(tft_diagram)
            except Exception as e:
                st.warning(f"Unable to render graph visualization: {e}")
                st.code(tft_diagram, language="dot")

        with col2:
            st.markdown(
                """
            ### <span style="color:#2C3E50; font-weight:bold;">Key Components:</span>
            
            1. <span style="color:#154360; font-weight:bold;">Variable Selection Networks</span>
            
            2. <span style="color:#154360; font-weight:bold;">LSTM Encoding/Decoding</span>
            
            3. <span style="color:#154360; font-weight:bold;">Multi-head Attention</span>
            
            4. <span style="color:#154360; font-weight:bold;">Temporal Fusion</span>
            
            5. <span style="color:#154360; font-weight:bold;">Quantile Forecasts</span>
            """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
        ### <span style="color:#2C3E50; font-weight:bold;">TFT Advanced Components:</span>
        
        1. <span style="color:#154360; font-weight:bold;">Variable Selection Networks:</span>
           - Dynamically selects most relevant features at each timestep
           - Learns importance of different indicators
        
        2. <span style="color:#154360; font-weight:bold;">Multi-head Attention:</span>
           - Identifies dependencies between different time points
           - Can model relationships between distant time points
        
        3. <span style="color:#154360; font-weight:bold;">Gating Mechanisms:</span>
           - Controls information flow between components
           - Helps manage multiple input types
        
        4. <span style="color:#154360; font-weight:bold;">Quantile Forecasts:</span>
           - Provides prediction intervals, not just point estimates
           - Critical for risk management in trading
        
        ### <span style="color:#2C3E50; font-weight:bold;">Advantages for Price Prediction:</span>
        
        - <span style="color:#1E8449;">State-of-the-art performance</span> on complex time series
        - <span style="color:#1E8449;">Built-in uncertainty estimation</span> through quantiles
        - <span style="color:#1E8449;">Handles static and temporal features</span>
        - <span style="color:#1E8449;">Excellent interpretation capabilities</span>
        - <span style="color:#1E8449;">Robust to noisy data</span> common in markets
        
        ### <span style="color:#2C3E50; font-weight:bold;">Interpretability Features:</span>
        
        - Variable importance scores at each timestep
        - Attention weights show which past periods influence predictions
        - Uncertainty quantification through prediction intervals
        """,
            unsafe_allow_html=True
        )

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
            color_continuous_scale="Viridis",
        )

        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)

        # Feature importance
        features = [
            "Close",
            "Volume",
            "RSI",
            "MACD",
            "BB_Width",
            "Sentiment",
            "Market_Cap",
            "Exchange_Inflow",
        ]
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
        recent_dates = dates[-min(10, num_days) :]
        fig2 = px.imshow(
            time_importance,
            x=recent_dates,
            y=features,
            labels=dict(x="Date", y="Feature", color="Importance"),
            title="Feature Importance Over Time",
            color_continuous_scale="Viridis",
        )

        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

        # Quantile predictions with distinct colors
        x = np.arange(num_days)
        y_median = 100 + 0.1 * x + 5 * np.sin(x / 5) + np.random.randn(num_days)
        y_low = y_median - 5 - 0.2 * x
        y_high = y_median + 5 + 0.2 * x

        # Create forecast dataframe
        forecast_df = pd.DataFrame(
            {"Date": dates, "P10": y_low, "P50": y_median, "P90": y_high}
        )

        # Plot quantile forecasts with better colors
        fig3 = go.Figure()

        # Add quantile range
        fig3.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["P90"],
                fill=None,
                mode="lines",
                line_color="rgba(41, 128, 185, 0.3)",
                name="P90",
            )
        )

        fig3.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["P10"],
                fill="tonexty",
                mode="lines",
                line_color="rgba(41, 128, 185, 0.3)",
                name="P10",
            )
        )

        # Add median prediction
        fig3.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["P50"],
                mode="lines",
                line=dict(color="#2980B9", width=2),
                name="P50 (Median)",
            )
        )

        fig3.update_layout(
            title="TFT Quantile Forecasts with Uncertainty",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
        )

        st.plotly_chart(fig3, use_container_width=True)


def plot_feature_importance(model, feature_names, max_features=20, model_type=None):
    """
    Plot feature importance for the given model.
    
    Args:
        model: The trained model (sklearn, xgboost, etc.)
        feature_names: List of feature names
        max_features: Maximum number of features to display
        model_type: Optional model type to help determine how to extract importances
        
    Returns:
        Matplotlib figure or None if importances cannot be determined
    """
    try:
        importances = None
        
        # Extract feature importance based on model type
        if model_type is None:
            # Try to determine model type automatically
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (RandomForest, XGBoost, etc.)
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)[0] if model.coef_.ndim > 1 else np.abs(model.coef_)
        elif model_type == 'keras':
            # For Keras/TensorFlow models, use permutation importance
            # This is a placeholder - in practice you'd need validation data
            st.warning("Feature importance for neural networks requires validation data")
            return None
        
        if importances is None:
            st.warning("Could not determine feature importance for this model type")
            return None
        
        # Create DataFrame for plotting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        })
        
        # Sort by importance and limit to max_features
        feature_importance_df = feature_importance_df.sort_values(
            'Importance', ascending=False
        ).reset_index(drop=True).head(max_features)
# Create plot with improved styling
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_importance_df) * 0.3)))
        
        color_palette = sns.color_palette("viridis", n_colors=len(feature_importance_df))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance_df,
            ax=ax,
            palette=color_palette
        )
        
        ax.set_title('Feature Importance', fontsize=14, color='#2C3E50')
        ax.set_xlabel('Importance', fontsize=12, color='#2C3E50')
        ax.set_ylabel('Feature', fontsize=12, color='#2C3E50')
        ax.tick_params(axis='y', labelsize=10, colors='#2C3E50')
        ax.tick_params(axis='x', colors='#2C3E50')
        
        # Add grid for easier reading
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error plotting feature importance: {e}")
        return None