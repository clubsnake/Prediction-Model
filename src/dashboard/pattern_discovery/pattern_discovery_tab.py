import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.config_loader import DATA_DIR

from .pattern_management import PatternManager


class PatternDiscoveryTab:
    """
    Dashboard tab for discovering and visualizing market patterns and correlations.
    This tab mines data for predictive patterns and presents them in a user-friendly way.
    """

    def __init__(
        self,
        df,
        ensemble_weighter=None,
        lookback_days=180,
        min_occurrences=5,
        signal_threshold=1.5,
    ):
        """
        Initialize the pattern discovery tab

        Args:
            df: DataFrame with market data and indicators
            ensemble_weighter: Optional ensemble weighter instance
            lookback_days: Days of data to analyze for patterns
            min_occurrences: Minimum number of occurrences for a pattern to be considered valid
            signal_threshold: Minimum signal strength (z-score) for correlation to be significant
        """
        self.df = df
        self.weighter = ensemble_weighter
        self.lookback_days = lookback_days
        self.min_occurrences = min_occurrences
        self.signal_threshold = signal_threshold

        # Create directory for saving patterns - update to use Models folder
        self.patterns_dir = os.path.join(DATA_DIR, "Models", "discovered_patterns")
        os.makedirs(self.patterns_dir, exist_ok=True)

        # Create pattern manager
        self.pattern_manager = PatternManager(self.patterns_dir)

        # Color scheme for pattern types
        self.pattern_colors = {
            "bullish": "#4CAF50",  # Green
            "bearish": "#F44336",  # Red
            "volatility": "#FF9800",  # Orange
            "volume": "#2196F3",  # Blue
            "divergence": "#9C27B0",  # Purple
            "consolidation": "#607D8B",  # Blue-grey
        }

    def render_tab(self):
        """Render the pattern discovery dashboard tab"""
        st.title("📊 Pattern Discovery")
        st.write("Discover and track predictive patterns in market data")

        # Add pattern mining controls
        with st.expander("Pattern Discovery Settings", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                lookback = st.number_input(
                    "Lookback Days",
                    min_value=30,
                    max_value=365,
                    value=self.lookback_days,
                )

                self.lookback_days = lookback

            with col2:
                min_occurrences = st.number_input(
                    "Min Occurrences",
                    min_value=3,
                    max_value=50,
                    value=self.min_occurrences,
                )

                self.min_occurrences = min_occurrences

            with col3:
                signal_threshold = st.number_input(
                    "Signal Threshold (Z-score)",
                    min_value=1.0,
                    max_value=3.0,
                    value=self.signal_threshold,
                    step=0.1,
                )

                self.signal_threshold = signal_threshold

            # Add button to discover patterns
            if st.button("Discover New Patterns", use_container_width=True):
                with st.spinner("Mining for patterns..."):
                    # Use the pattern manager to discover patterns
                    discovered_patterns = self.pattern_manager.discover_patterns(
                        self.df,
                        min_occurrences=self.min_occurrences,
                        significance_threshold=self.signal_threshold,
                    )

                    # Save discovered patterns
                    for pattern in discovered_patterns:
                        self.pattern_manager.save_pattern(pattern)

                    st.success(f"Discovered {len(discovered_patterns)} new patterns!")

        # Create tabs for different pattern views
        pattern_tabs = st.tabs(
            [
                "Active Patterns",
                "Pattern Library",
                "Real-time Signals",
                "Pattern Analytics",
            ]
        )

        # Tab 1: Active Patterns (currently triggering)
        with pattern_tabs[0]:
            self._render_active_patterns_tab()

        # Tab 2: Pattern Library (all discovered patterns)
        with pattern_tabs[1]:
            self._render_pattern_library_tab()

        # Tab 3: Real-time Signals (patterns triggering now)
        with pattern_tabs[2]:
            self._render_realtime_signals_tab()

        # Tab 4: Pattern Analytics (performance & statistics)
        with pattern_tabs[3]:
            self._render_pattern_analytics_tab()

    def _render_active_patterns_tab(self):
        """Render the active patterns that are currently triggering"""
        st.header("Currently Active Patterns")

        # Get active patterns using the pattern manager
        active_patterns = self.pattern_manager.get_active_patterns(self.df)

        if not active_patterns:
            st.info(
                "No patterns are currently active. Run pattern discovery or wait for patterns to trigger."
            )

            # Add sample pattern for demonstration
            st.subheader("Example Pattern (Demo)")
            self._render_sample_pattern()
            return

        # Display active patterns
        st.write(f"Found {len(active_patterns)} active patterns")

        # Sort patterns by expected impact (strongest first)
        active_patterns.sort(
            key=lambda x: abs(x.get("expected_return", 0)), reverse=True
        )

        # Display each active pattern
        for i, pattern in enumerate(active_patterns):
            self._render_pattern_card(pattern, is_active=True)

            # Add separator between patterns
            if i < len(active_patterns) - 1:
                st.markdown("---")

    def _render_pattern_library_tab(self):
        """Render the full library of discovered patterns"""
        st.header("Pattern Library")

        # Load all discovered patterns
        all_patterns = self.pattern_manager.list_patterns(include_archived=False)

        if not all_patterns:
            st.info(
                "No patterns have been discovered yet. Run pattern discovery to find patterns."
            )
            return

        # Add filtering options
        col1, col2, col3 = st.columns(3)

        with col1:
            pattern_categories = list(
                set([p.get("category", "Other") for p in all_patterns])
            )
            selected_categories = st.multiselect(
                "Pattern Categories", pattern_categories, default=pattern_categories
            )

        with col2:
            min_reliability = st.slider("Min Reliability Score", 0.0, 1.0, 0.5)
            min_return = st.slider("Min Expected Return", -10.0, 10.0, -5.0)
            max_return = st.slider("Max Expected Return", min_return, 20.0, 10.0)

        with col3:
            sort_options = [
                "Reliability",
                "Expected Return",
                "Discovery Date",
                "Occurrences",
            ]
            sort_by = st.selectbox("Sort By", sort_options, index=1)
            ascending = st.checkbox("Ascending Order", value=False)

        # Filter patterns based on selection
        filtered_patterns = [
            p
            for p in all_patterns
            if p.get("category", "Other") in selected_categories
            and p.get("reliability", 0) >= min_reliability
            and min_return <= p.get("expected_return", 0) <= max_return
        ]

        # Sort patterns
        if sort_by == "Reliability":
            filtered_patterns.sort(
                key=lambda x: x.get("reliability", 0), reverse=not ascending
            )
        elif sort_by == "Expected Return":
            filtered_patterns.sort(
                key=lambda x: x.get("expected_return", 0), reverse=not ascending
            )
        elif sort_by == "Discovery Date":
            filtered_patterns.sort(
                key=lambda x: x.get("discovery_date", "2023-01-01"),
                reverse=not ascending,
            )
        else:  # Occurrences
            filtered_patterns.sort(
                key=lambda x: x.get("occurrences", 0), reverse=not ascending
            )

        # Display patterns count
        st.write(f"Displaying {len(filtered_patterns)} of {len(all_patterns)} patterns")

        # Create an expandable section for each pattern
        for i, pattern in enumerate(filtered_patterns):
            with st.expander(
                f"{pattern.get('name', f'Pattern #{i+1}')} ({pattern.get('category', 'Other')})"
            ):
                self._render_pattern_card(
                    pattern, is_active=self._is_pattern_active(pattern)
                )

    def _render_realtime_signals_tab(self):
        """Render real-time signals from patterns"""
        st.header("Real-time Pattern Signals")

        # Get active patterns
        active_patterns = self.pattern_manager.get_active_patterns(self.df)

        if not active_patterns:
            st.info("No pattern signals are currently active.")

            # Add sample signal timeline for demonstration
            self._render_sample_signal_timeline()
            return

        # Signal timeline chart
        st.subheader("Signal Timeline")

        # Get signal data for timeline
        signal_data = []
        for pattern in active_patterns:
            signal_data.append(
                {
                    "pattern": pattern.get("name", "Unknown Pattern"),
                    "activation_time": pattern.get(
                        "last_activation",
                        datetime.now() - timedelta(hours=random.randint(1, 24)),
                    ),
                    "expected_return": pattern.get("expected_return", 0),
                    "timeframe": pattern.get("timeframe", "1d"),
                    "category": pattern.get("category", "Other"),
                }
            )

        # Create timeline of signals
        if signal_data:
            signal_df = pd.DataFrame(signal_data)

            # Sort by activation time
            signal_df.sort_values("activation_time", inplace=True)

            # Create visualization
            fig = px.timeline(
                signal_df,
                x_start="activation_time",
                x_end=[a + timedelta(days=1) for a in signal_df["activation_time"]],
                y="pattern",
                color="expected_return",
                color_continuous_scale="RdYlGn",
                hover_data=["category", "timeframe", "expected_return"],
                title="Signal Activation Timeline",
            )

            # Update layout
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)

        # Current active signals
        st.subheader("Active Signals Summary")

        # Create summary table
        if active_patterns:
            summary_data = []
            for pattern in active_patterns:
                summary_data.append(
                    {
                        "Pattern": pattern.get("name", "Unknown"),
                        "Category": pattern.get("category", "Other"),
                        "Expected Return": f"{pattern.get('expected_return', 0):.2f}%",
                        "Reliability": f"{pattern.get('reliability', 0)*100:.1f}%",
                        "Timeframe": pattern.get("timeframe", "1d"),
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True)

        # Add signal convergence analysis
        st.subheader("Signal Convergence Analysis")

        # Check for signal clusters (signals pointing in same direction)
        if active_patterns:
            bullish_signals = [
                p for p in active_patterns if p.get("expected_return", 0) > 0
            ]
            bearish_signals = [
                p for p in active_patterns if p.get("expected_return", 0) < 0
            ]

            # Calculate aggregate signal strength
            bullish_strength = sum(
                [
                    p.get("expected_return", 0) * p.get("reliability", 0)
                    for p in bullish_signals
                ]
            )
            bearish_strength = abs(
                sum(
                    [
                        p.get("expected_return", 0) * p.get("reliability", 0)
                        for p in bearish_signals
                    ]
                )
            )

            # Create columns for bullish/bearish signals
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Bullish Signal Strength",
                    f"{bullish_strength:.2f}",
                    f"{len(bullish_signals)} signals",
                )

            with col2:
                st.metric(
                    "Bearish Signal Strength",
                    f"{bearish_strength:.2f}",
                    f"{len(bearish_signals)} signals",
                )

            # Signal balance meter
            total_strength = bullish_strength + bearish_strength
            if total_strength > 0:
                bull_pct = bullish_strength / total_strength * 100

                # Create gauge chart
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=bull_pct,
                        title={"text": "Signal Balance (Bullish %)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "green"},
                            "steps": [
                                {"range": [0, 40], "color": "lightcoral"},
                                {"range": [40, 60], "color": "lightyellow"},
                                {"range": [60, 100], "color": "lightgreen"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 3},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    def _render_pattern_analytics_tab(self):
        """Render analytics for pattern performance"""
        st.header("Pattern Performance Analytics")

        # Load all patterns
        all_patterns = self.pattern_manager.list_patterns(include_archived=True)

        if not all_patterns:
            st.info("No patterns available for analysis. Run pattern discovery first.")
            return

        # Create stats for patterns
        pattern_stats = self._calculate_pattern_stats(all_patterns)

        # Create tabs for different analytics views
        analytics_tabs = st.tabs(
            [
                "Pattern Performance",
                "Pattern Categories",
                "Correlation Network",
                "Prediction Impact",
            ]
        )

        # Tab 1: Pattern Performance
        with analytics_tabs[0]:
            st.subheader("Pattern Performance Analysis")

            # Create performance chart
            performance_data = []
            for pattern in all_patterns:
                # Ensure expected_return and reliability are properly typed
                expected_return = pattern.get("expected_return", 0)
                reliability = pattern.get("reliability", 0)
                
                # Convert numpy scalar types to Python native types to avoid reindexing errors
                if isinstance(expected_return, np.number):
                    expected_return = float(expected_return)
                if isinstance(reliability, np.number):
                    reliability = float(reliability)
                    
                performance_data.append(
                    {
                        "Pattern": pattern.get("name", "Unknown"),
                        "Expected Return": expected_return,
                        "Reliability": reliability * 100,  # Convert to percentage
                        "Occurrences": pattern.get("occurrences", 0),
                    }
                )

            if performance_data:
                perf_df = pd.DataFrame(performance_data)

                # Create scatter plot of return vs reliability
                fig = px.scatter(
                    perf_df,
                    x="Expected Return",
                    y="Reliability",
                    size="Occurrences",
                    hover_name="Pattern",
                    color="Expected Return",
                    color_continuous_scale="RdYlGn",
                    title="Pattern Performance: Expected Return vs. Reliability",
                    labels={
                        "Expected Return": "Expected Return (%)",
                        "Reliability": "Reliability Score (%)",
                    },
                )

                # Add reference lines
                fig.add_hline(
                    y=50, line_dash="dash", line_color="gray", annotation_text="Random"
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")

                # Add quadrant labels
                fig.add_annotation(
                    x=5,
                    y=75,
                    text="High Quality Bullish",
                    showarrow=False,
                    font_size=14,
                )
                fig.add_annotation(
                    x=-5,
                    y=75,
                    text="High Quality Bearish",
                    showarrow=False,
                    font_size=14,
                )
                fig.add_annotation(
                    x=5, y=25, text="Low Quality Bullish", showarrow=False, font_size=14
                )
                fig.add_annotation(
                    x=-5,
                    y=25,
                    text="Low Quality Bearish", showarrow=False, font_size=14
                )

                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Pattern Categories
        with analytics_tabs[1]:
            st.subheader("Pattern Categories Analysis")

            # Summarize patterns by category
            category_data = []
            for pattern in all_patterns:
                category = pattern.get("category", "Other")
                expected_return = pattern.get("expected_return", 0)
                reliability = pattern.get("reliability", 0)

                category_data.append(
                    {
                        "Category": category,
                        "Expected Return": expected_return,
                        "Reliability": reliability,
                        "Type": "Bullish" if expected_return > 0 else "Bearish",
                    }
                )

            if category_data:
                cat_df = pd.DataFrame(category_data)

                # Group by category
                cat_summary = (
                    cat_df.groupby("Category")
                    .agg(
                        {
                            "Expected Return": ["mean", "count"],
                            "Reliability": "mean",
                            "Type": lambda x: (x == "Bullish").mean()
                            * 100,  # Percentage bullish
                        }
                    )
                    .reset_index()
                )

                # Flatten multi-level columns
                cat_summary.columns = [
                    "Category",
                    "Avg Return",
                    "Count",
                    "Avg Reliability",
                    "% Bullish",
                ]

                # Create bar chart of pattern counts by category
                fig = px.bar(
                    cat_summary,
                    x="Category",
                    y="Count",
                    color="Avg Return",
                    color_continuous_scale="RdYlGn",
                    title="Pattern Count by Category",
                    labels={
                        "Count": "Number of Patterns",
                        "Category": "Pattern Category",
                    },
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show summary table
                st.dataframe(
                    cat_summary.style.format(
                        {
                            "Avg Return": "{:.2f}%",
                            "Avg Reliability": "{:.1%}",
                            "% Bullish": "{:.1f}%",
                        }
                    )
                )

        # Tab 3: Correlation Network
        with analytics_tabs[2]:
            st.subheader("Pattern Correlation Network")

            # This requires pattern activation history to calculate correlations
            # For now, we'll just show a placeholder visualization

            st.info(
                "Pattern correlation network requires activation history data. This is a placeholder visualization."
            )

            # Create a sample network
            try:
                import networkx as nx

                # Create a sample network
                G = nx.Graph()

                # Add some nodes (patterns)
                for i, pattern in enumerate(all_patterns[:15]):  # Limit to 15 patterns
                    name = pattern.get("name", f"Pattern {i}")
                    category = pattern.get("category", "Other")
                    return_val = pattern.get("expected_return", 0)

                    # Add node with attributes
                    G.add_node(
                        name,
                        category=category,
                        return_val=return_val,
                        size=abs(return_val) * 2 + 10,  # Size based on return
                    )

                # Add some edges (correlations)
                for i in range(len(list(G.nodes))):
                    for j in range(i + 1, len(list(G.nodes))):
                        # Random correlation
                        if random.random() < 0.3:  # 30% chance of connection
                            node_i = list(G.nodes)[i]
                            node_j = list(G.nodes)[j]

                            # Random correlation -1 to 1
                            correlation = (random.random() * 2) - 1

                            G.add_edge(node_i, node_j, weight=correlation)

                # Convert to plotly
                pos = nx.spring_layout(G, seed=42)

                edge_traces = []

                # Add edges
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    weight = G.edges[edge]["weight"]

                    color = "green" if weight > 0 else "red"
                    width = abs(weight) * 3  # Width based on correlation strength

                    edge_trace = go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=width, color=color),
                        hoverinfo="none",
                        mode="lines",
                    )
                    edge_traces.append(edge_trace)

                # Add nodes
                node_trace = go.Scatter(
                    x=[pos[node][0] for node in G.nodes()],
                    y=[pos[node][1] for node in G.nodes()],
                    mode="markers+text",
                    text=[node for node in G.nodes()],
                    textposition="top center",
                    marker=dict(
                        size=[G.nodes[node]["size"] for node in G.nodes()],
                        color=[
                            "green" if G.nodes[node]["return_val"] > 0 else "red"
                            for node in G.nodes()
                        ],
                        line=dict(width=2, color="black"),
                    ),
                    hoverinfo="text",
                )

                # Create figure
                fig = go.Figure(data=edge_traces + [node_trace])
                fig.update_layout(
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    title="Pattern Correlation Network (Sample)",
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    """
                **Network Legend:**
                - **Node Size**: Magnitude of expected return
                - **Node Color**: Green = bullish, Red = bearish
                - **Edge Width**: Strength of correlation
                - **Edge Color**: Green = positive correlation, Red = negative correlation
                """
                )

            except ImportError:
                st.error("NetworkX library required for network visualization")
                st.info("Install it with: pip install networkx")

        # Tab 4: Prediction Impact
        with analytics_tabs[3]:
            st.subheader("Pattern Impact on Predictions")

            # This requires integration with prediction model
            if self.weighter is None:
                st.info(
                    "Connect an ensemble weighter to see pattern impact on predictions"
                )

                # Create sample visualization
                dates = pd.date_range(end=pd.Timestamp.now(), periods=30).date
                base_pred = 100 + np.cumsum(np.random.normal(0, 1, 30))

                # Create some sample patterns
                pattern_impacts = [
                    {"name": "RSI Oversold", "impact": 2.5, "start": 5, "duration": 3},
                    {
                        "name": "MACD Crossover",
                        "impact": -1.8,
                        "start": 12,
                        "duration": 4,
                    },
                    {"name": "Volume Spike", "impact": 3.2, "start": 20, "duration": 5},
                ]

                # Apply pattern impacts
                adjusted_pred = base_pred.copy()
                for pattern in pattern_impacts:
                    start = pattern["start"]
                    end = start + pattern["duration"]
                    adjusted_pred[start:end] += pattern["impact"]

                # Create DataFrame
                pred_df = pd.DataFrame(
                    {
                        "Date": dates,
                        "Base Prediction": base_pred,
                        "Pattern-Adjusted": adjusted_pred,
                    }
                )

                # Create visualization
                fig = go.Figure()

                # Add base prediction
                fig.add_trace(
                    go.Scatter(
                        x=pred_df["Date"],
                        y=pred_df["Base Prediction"],
                        name="Base Prediction",
                        line=dict(color="blue", width=2),
                    )
                )

                # Add adjusted prediction
                fig.add_trace(
                    go.Scatter(
                        x=pred_df["Date"],
                        y=pred_df["Pattern-Adjusted"],
                        name="Pattern-Adjusted",
                        line=dict(color="green", width=2),
                    )
                )

                # Add pattern regions
                for pattern in pattern_impacts:
                    start_idx = pattern["start"]
                    end_idx = start_idx + pattern["duration"]

                    # Add rectangle for pattern
                    fig.add_shape(
                        type="rect",
                        x0=dates[start_idx],
                        x1=dates[end_idx - 1],
                        y0=min(
                            pred_df["Base Prediction"].min(),
                            pred_df["Pattern-Adjusted"].min(),
                        ),
                        y1=max(
                            pred_df["Base Prediction"].max(),
                            pred_df["Pattern-Adjusted"].max(),
                        ),
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        line=dict(width=0),
                        layer="below",
                    )

                    # Add annotation
                    fig.add_annotation(
                        x=dates[start_idx + pattern["duration"] // 2],
                        y=max(
                            pred_df["Base Prediction"].max(),
                            pred_df["Pattern-Adjusted"].max(),
                        ),
                        text=pattern["name"],
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        arrowsize=1,
                        arrowwidth=2,
                    )

                fig.update_layout(
                    title="Impact of Patterns on Price Predictions (Sample)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    """
                This sample visualization shows how detected patterns could modify the base prediction.
                When properly integrated with the prediction model, patterns would adjust the forecast
                based on their expected returns and reliability.
                """
                )
            else:
                # If weighter is provided, calculate actual pattern impact
                # This would require actual implementation based on your model
                st.info(
                    "Pattern impact analysis with the connected model would appear here"
                )

    def _render_pattern_card(self, pattern, is_active=False):
        """Render a card with pattern details"""
        # Extract pattern information
        name = pattern.get("name", "Unnamed Pattern")
        description = pattern.get("description", "No description available")
        conditions = pattern.get("conditions", [])
        expected_return = pattern.get("expected_return", 0)
        reliability = pattern.get("reliability", 0)
        timeframe = pattern.get("timeframe", "1d")
        occurrences = pattern.get("occurrences", 0)
        discovery_date = pattern.get("discovery_date", "Unknown")
        category = pattern.get("category", "Other")
        historical_examples = pattern.get("historical_examples", [])

        # Determine color scheme based on expected return
        color = "green" if expected_return > 0 else "red"
        bg_color = (
            "#e6ffe6" if expected_return > 0 else "#ffe6e6"
        )  # Light green or light red

        # Set active badge if active
        active_badge = "🔴 ACTIVE NOW" if is_active else ""

        # Create header with styling
        st.markdown(
            f"""
        <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; border-left: 5px solid {color};">
            <div style="display: flex; justify-content: space-between;">
                <h3 style="margin: 0; padding: 0; color: {color};">{name}</h3>
                <span style="color: {color}; font-weight: bold;">{active_badge}</span>
            </div>
            <p style="margin-top: 5px;"><strong>Category:</strong> {category} | <strong>Timeframe:</strong> {timeframe}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Pattern description
        st.markdown(f"**Description**: {description}")

        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Expected Return", f"{expected_return:.2f}%")

        with col2:
            st.metric("Reliability", f"{reliability*100:.1f}%")

        with col3:
            st.metric("Occurrences", occurrences)

        with col4:
            st.metric("Discovered", discovery_date)

        # Pattern conditions
        st.markdown("**Conditions:**")
        for i, condition in enumerate(conditions):
            indicator = condition.get("indicator", "Unknown")
            operator = condition.get("operator", ">")
            value = condition.get("value", 0)

            # Create styled condition
            st.markdown(f"- {indicator} {operator} {value}")

        # Pattern visualization
        if "visualization_data" in pattern:
            vis_data = pattern["visualization_data"]

            # Create visualization based on data type
            if isinstance(vis_data, dict) and "type" in vis_data:
                if vis_data["type"] == "price":
                    self._render_price_pattern_visualization(vis_data)
                elif vis_data["type"] == "indicator":
                    self._render_indicator_pattern_visualization(vis_data)
            else:
                self._render_generic_pattern_visualization(pattern)
        else:
            # Create a generic visualization
            self._render_generic_pattern_visualization(pattern)

        # Historical examples if available
        if historical_examples:
            st.markdown("**Historical Examples:**")

            # Create a dataframe with examples
            examples_df = pd.DataFrame(historical_examples)

            if "date" in examples_df.columns and "return" in examples_df.columns:
                # Create visualization
                fig = px.bar(
                    examples_df,
                    x="date",
                    y="return",
                    title="Historical Pattern Returns",
                    labels={"date": "Date", "return": "Return (%)"},
                    color="return",
                    color_continuous_scale="RdYlGn",
                )

                st.plotly_chart(fig, use_container_width=True)

    def _render_sample_pattern(self):
        """Render a sample pattern for demonstration"""
        import random
        from datetime import datetime, timedelta

        # Create a sample pattern
        sample_pattern = {
            "name": "WERPI State 3 with Bollinger Squeeze",
            "description": "When WERPI enters State 3 and Bollinger Bands are contracting (width < 2% of price), price typically moves up significantly within 3 days.",
            "conditions": [
                {"indicator": "WERPI State", "operator": "==", "value": 3},
                {"indicator": "BB Width", "operator": "<", "value": "2% of price"},
                {"indicator": "RSI(14)", "operator": "<", "value": 45},
            ],
            "expected_return": 3.7,
            "reliability": 0.82,
            "timeframe": "1d",
            "occurrences": 14,
            "discovery_date": (datetime.now() - timedelta(days=45)).strftime(
                "%Y-%m-%d"
            ),
            "category": "Technical Pattern",
            "historical_examples": [
                {
                    "date": (
                        datetime.now() - timedelta(days=random.randint(10, 200))
                    ).strftime("%Y-%m-%d"),
                    "return": 4.2,
                },
                {
                    "date": (
                        datetime.now() - timedelta(days=random.randint(10, 200))
                    ).strftime("%Y-%m-%d"),
                    "return": 3.8,
                },
                {
                    "date": (
                        datetime.now() - timedelta(days=random.randint(10, 200))
                    ).strftime("%Y-%m-%d"),
                    "return": 5.1,
                },
                {
                    "date": (
                        datetime.now() - timedelta(days=random.randint(10, 200))
                    ).strftime("%Y-%m-%d"),
                    "return": -0.4,
                },
                {
                    "date": (
                        datetime.now() - timedelta(days=random.randint(10, 200))
                    ).strftime("%Y-%m-%d"),
                    "return": 3.9,
                },
                {
                    "date": (
                        datetime.now() - timedelta(days=random.randint(10, 200))
                    ).strftime("%Y-%m-%d"),
                    "return": 4.6,
                },
                {
                    "date": (
                        datetime.now() - timedelta(days=random.randint(10, 200))
                    ).strftime("%Y-%m-%d"),
                    "return": 2.8,
                },
            ],
        }

        # Render the sample pattern
        self._render_pattern_card(sample_pattern, is_active=True)

    def _render_sample_signal_timeline(self):
        """Render a sample signal timeline for demonstration"""
        import random
        from datetime import datetime, timedelta

        # Create sample signal data
        sample_signals = [
            {
                "pattern": "WERPI State 3 with BB Squeeze",
                "activation_time": datetime.now() - timedelta(hours=3),
                "expected_return": 3.7,
                "timeframe": "1d",
                "category": "Technical Pattern",
            },
            {
                "pattern": "RSI Divergence with Volume Spike",
                "activation_time": datetime.now() - timedelta(hours=12),
                "expected_return": 2.8,
                "timeframe": "1d",
                "category": "Divergence Pattern",
            },
            {
                "pattern": "Double Bottom with Support Test",
                "activation_time": datetime.now() - timedelta(hours=14),
                "expected_return": 5.2,
                "timeframe": "1d",
                "category": "Chart Pattern",
            },
            {
                "pattern": "MACD Bearish Cross",
                "activation_time": datetime.now() - timedelta(hours=8),
                "expected_return": -2.1,
                "timeframe": "1d",
                "category": "Momentum Pattern",
            },
            {
                "pattern": "Volume Dry-Up at Support",
                "activation_time": datetime.now() - timedelta(hours=6),
                "expected_return": 1.9,
                "timeframe": "1d",
                "category": "Volume Pattern",
            },
        ]

        # Create timeline visualization
        signal_df = pd.DataFrame(sample_signals)

        # Sort by activation time
        signal_df.sort_values("activation_time", inplace=True)

        # Create visualization
        fig = px.timeline(
            signal_df,
            x_start="activation_time",
            x_end=[a + timedelta(days=1) for a in signal_df["activation_time"]],
            y="pattern",
            color="expected_return",
            color_continuous_scale="RdYlGn",
            hover_data=["category", "timeframe", "expected_return"],
            title="Sample Signal Activation Timeline",
        )

        # Update layout
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

        # Add sample signal summary
        st.subheader("Sample Signals Summary")

        # Create summary table
        summary_data = []
        for signal in sample_signals:
            summary_data.append(
                {
                    "Pattern Name": signal["pattern"],
                    "Category": signal["category"],
                    "Signal Strength": round(random.uniform(1.0, 3.0), 1),
                    "Expected Return": f"{signal['expected_return']:.2f}%",
                    "Timeframe": signal["timeframe"],
                    "Reliability": f"{random.uniform(0.6, 0.9)*100:.1f}%",
                    "Status": "Active",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True)

        # Signal convergence example
        st.subheader("Sample Signal Convergence Analysis")

        # Split bullish/bearish signals
        bullish_signals = [s for s in sample_signals if s["expected_return"] > 0]
        bearish_signals = [s for s in sample_signals if s["expected_return"] < 0]

        # Calculate strength
        bull_strength = sum(
            [s["expected_return"] * random.uniform(0.6, 0.9) for s in bullish_signals]
        )
        bear_strength = abs(
            sum(
                [
                    s["expected_return"] * random.uniform(0.6, 0.9)
                    for s in bearish_signals
                ]
            )
        )

        # Create columns
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Bullish Signal Strength",
                f"{bull_strength:.2f}",
                f"{len(bullish_signals)} signals",
            )

        with col2:
            st.metric(
                "Bearish Signal Strength",
                f"{bear_strength:.2f}",
                f"{len(bearish_signals)} signals",
            )

        # Create gauge chart
        total_strength = bull_strength + bear_strength
        if total_strength > 0:
            bull_pct = bull_strength / total_strength * 100

            # Create gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=bull_pct,
                    title={"text": "Signal Balance (Bullish %)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "green"},
                        "steps": [
                            {"range": [0, 40], "color": "lightred"},
                            {"range": [40, 60], "color": "lightyellow"},
                            {"range": [60, 100], "color": "lightgreen"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    def _render_price_pattern_visualization(self, vis_data):
        """Render a price pattern visualization"""
        if "price_data" not in vis_data:
            st.warning("Visualization data is missing price information")
            return

        price_data = vis_data["price_data"]

        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(price_data, list):
            price_df = pd.DataFrame(price_data)
        else:
            price_df = price_data

        # Check for required columns
        required_cols = ["date", "open", "high", "low", "close"]
        if not all(
            col.lower() in [c.lower() for c in price_df.columns]
            for col in required_cols
        ):
            st.warning(f"Price data missing required columns: {required_cols}")
            return

        # Create candlestick chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=price_df["date"],
                    open=price_df["open"],
                    high=price_df["high"],
                    low=price_df["low"],
                    close=price_df["close"],
                    increasing_line_color="green",
                    decreasing_line_color="red",
                )
            ]
        )

        # Add pattern markers if available
        if "pattern_zones" in vis_data:
            for zone in vis_data["pattern_zones"]:
                start = zone.get("start", None)
                end = zone.get("end", None)
                label = zone.get("label", "Pattern Zone")
                color = zone.get(
                    "color", "rgba(255, 165, 0, 0.3)"
                )  # Default: semi-transparent orange

                if start and end:
                    fig.add_vrect(
                        x0=start,
                        x1=end,
                        fillcolor=color,
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        annotation_text=label,
                        annotation_position="top left",
                    )

        # Add indicators if available
        if "indicators" in vis_data:
            for indicator in vis_data["indicators"]:
                name = indicator.get("name", "Indicator")
                data = indicator.get("data", [])
                color = indicator.get("color", "blue")

                if data and len(data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=price_df["date"],
                            y=data,
                            mode="lines",
                            name=name,
                            line=dict(color=color),
                        )
                    )

        # Update layout
        fig.update_layout(
            title="Pattern Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            xaxis_rangeslider_visible=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_indicator_pattern_visualization(self, vis_data):
        """Render an indicator pattern visualization"""
        if "price_data" not in vis_data or "indicator_data" not in vis_data:
            st.warning("Visualization data is missing price or indicator information")
            return

        price_data = vis_data["price_data"]
        indicator_data = vis_data["indicator_data"]

        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(price_data, list):
            price_df = pd.DataFrame(price_data)
        else:
            price_df = price_data

        if isinstance(indicator_data, list):
            ind_df = pd.DataFrame(indicator_data)
        else:
            ind_df = indicator_data

        # Create subplots: price and indicator
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Price Chart", vis_data.get("indicator_name", "Indicator")),
        )

        # Add price (candlestick or line)
        if all(col in price_df.columns for col in ["open", "high", "low", "close"]):
            fig.add_trace(
                go.Candlestick(
                    x=price_df["date"],
                    open=price_df["open"],
                    high=price_df["high"],
                    low=price_df["low"],
                    close=price_df["close"],
                    increasing_line_color="green",
                    decreasing_line_color="red",
                ),
                row=1,
                col=1,
            )
        elif "close" in price_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_df["date"],
                    y=price_df["close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

        # Add indicator
        indicator_col = vis_data.get(
            "indicator_column",
            ind_df.columns[1] if len(ind_df.columns) > 1 else ind_df.columns[0],
        )

        if indicator_col in ind_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=ind_df["date"] if "date" in ind_df.columns else ind_df.index,
                    y=ind_df[indicator_col],
                    mode="lines",
                    name=vis_data.get("indicator_name", indicator_col),
                    line=dict(color="purple"),
                ),
                row=2,
                col=1,
            )

            # Add threshold lines if available
            if "thresholds" in vis_data:
                for threshold in vis_data["thresholds"]:
                    value = threshold.get("value", 0)
                    label = threshold.get("label", f"Threshold: {value}")
                    color = threshold.get("color", "red")

                    fig.add_hline(
                        y=value,
                        line_dash="dash",
                        line_color=color,
                        annotation_text=label,
                        annotation_position="bottom right",
                        row=2,
                        col=1,
                    )

        # Add pattern zones if available
        if "pattern_zones" in vis_data:
            for zone in vis_data["pattern_zones"]:
                start = zone.get("start", None)
                end = zone.get("end", None)
                label = zone.get("label", "Pattern Zone")
                color = zone.get(
                    "color", "rgba(255, 165, 0, 0.3)"
                )  # Default: semi-transparent orange

                if start and end:
                    # Add to both subplots
                    fig.add_vrect(
                        x0=start,
                        x1=end,
                        fillcolor=color,
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        annotation_text=label,
                        annotation_position="top left",
                        row=1,
                        col=1,
                    )

                    fig.add_vrect(
                        x0=start,
                        x1=end,
                        fillcolor=color,
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=2,
                        col=1,
                    )

        # Update layout
        fig.update_layout(
            title=vis_data.get("title", "Indicator Pattern Visualization"),
            height=700,
            xaxis_rangeslider_visible=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_generic_pattern_visualization(self, pattern):
        """Render a generic pattern visualization based on pattern type"""
        category = pattern.get("category", "Other")
        pattern_type = pattern.get("type", "price")

        # Generate sample data based on pattern type
        np.random.seed(42)  # For consistent randomization

        # X-axis - dates
        dates = pd.date_range(end=pd.Timestamp.now(), periods=90).date

        # Generate price data based on pattern type
        if pattern_type == "price" or category in ["Chart Pattern", "Price Action"]:
            # Create a sample price chart
            base = 100
            trend = np.linspace(0, 20, 90) * (
                1 if pattern.get("expected_return", 0) > 0 else -1
            )
            noise = np.random.randn(90) * 5
            price = base + trend + noise

            # Create oscillations for pattern
            oscillation = np.sin(np.linspace(0, 4 * np.pi, 90)) * 10
            price = price + oscillation

            # Create a pattern zone in the middle
            pattern_zone = slice(60, 75)

            # Create dataframe
            df = pd.DataFrame(
                {
                    "date": dates,
                    "price": price,
                    "pattern_zone": [i in pattern_zone for i in range(90)],
                }
            )

            # Create visualization
            fig = px.line(
                df,
                x="date",
                y="price",
                title="Sample Price Pattern",
                labels={"date": "Date", "price": "Price"},
            )

            # Add pattern zone
            fig.add_vrect(
                x0=dates[pattern_zone.start],
                x1=dates[pattern_zone.stop - 1],
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Pattern Zone",
                annotation_position="top left",
            )

            # Add vertical line for current date
            fig.add_vline(
                x=dates[-1],
                line_dash="dash",
                line_color="red",
                annotation_text="Current",
                annotation_position="top right",
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif pattern_type == "indicator" or category in [
            "Technical Indicator",
            "Oscillator",
        ]:
            # Create a price and indicator chart
            base = 100
            trend = np.linspace(0, 20, 90) * (
                1 if pattern.get("expected_return", 0) > 0 else -1
            )
            noise = np.random.randn(90) * 5
            price = base + trend + noise

            # Create indicator (e.g., RSI-like oscillator)
            indicator = (
                50
                + np.sin(np.linspace(0, 6 * np.pi, 90)) * 25
                + np.random.randn(90) * 5
            )

            # Create pattern zone
            pattern_zone = slice(60, 75)

            # Create dataframe
            df = pd.DataFrame(
                {
                    "date": dates,
                    "price": price,
                    "indicator": indicator,
                    "pattern_zone": [i in pattern_zone for i in range(90)],
                }
            )

            # Create a figure with subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Price", "Indicator"),
            )

            # Add price trace
            fig.add_trace(
                go.Scatter(x=dates, y=price, name="Price", line=dict(color="blue")),
                row=1,
                col=1,
            )

            # Add indicator trace
            fig.add_trace(
                go.Scatter(
                    x=dates, y=indicator, name="Indicator", line=dict(color="purple")
                ),
                row=2,
                col=1,
            )

            # Add pattern zone
            fig.add_vrect(
                x0=dates[pattern_zone.start],
                x1=dates[pattern_zone.stop - 1],
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Pattern Zone",
                annotation_position="top left",
                row=1,
                col=1,
            )

            fig.add_vrect(
                x0=dates[pattern_zone.start],
                x1=dates[pattern_zone.stop - 1],
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=2,
                col=1,
            )

            # Add threshold lines for indicator
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="red",
                annotation_text="Overbought",
                annotation_position="left",
                row=2,
                col=1,
            )

            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                annotation_text="Oversold",
                annotation_position="left",
                row=2,
                col=1,
            )

            # Add vertical line for current date
            fig.add_vline(
                x=dates[-1],
                line_dash="dash",
                line_color="red",
                annotation_text="Current",
                annotation_position="top right",
                row=1,
                col=1,
            )

            fig.update_layout(title="Sample Indicator Pattern", height=600)

            st.plotly_chart(fig, use_container_width=True)

        elif pattern_type == "divergence" or category == "Divergence Pattern":
            # Create a divergence pattern
            base = 100
            # Price makes lower lows while indicator makes higher lows (bullish divergence)
            price_trend = (
                -np.linspace(0, 15, 90)
                if pattern.get("expected_return", 0) > 0
                else np.linspace(0, 15, 90)
            )
            price = (
                base
                + price_trend
                + np.sin(np.linspace(0, 3 * np.pi, 90)) * 10
                + np.random.randn(90) * 3
            )

            # Indicator diverges from price
            indicator_trend = (
                np.linspace(0, 15, 90)
                if pattern.get("expected_return", 0) > 0
                else -np.linspace(0, 15, 90)
            )
            indicator = (
                50
                + indicator_trend
                + np.sin(np.linspace(0, 3 * np.pi, 90)) * 8
                + np.random.randn(90) * 2
            )

            # Create pattern zone
            pattern_zone = slice(60, 75)

            # Identify local minima/maxima for divergence lines
            from scipy.signal import find_peaks

            if pattern.get("expected_return", 0) > 0:
                # Bullish divergence - price lows with indicator highs
                price_neg = -price
                price_peaks, _ = find_peaks(price_neg, distance=15)
                ind_peaks, _ = find_peaks(indicator, distance=15)
            else:
                # Bearish divergence - price highs with indicator lows
                price_peaks, _ = find_peaks(price, distance=15)
                ind_neg = -indicator
                ind_peaks, _ = find_peaks(ind_neg, distance=15)

            # Keep only peaks in our target zone
            price_peaks = [p for p in price_peaks if 50 <= p < 80]
            ind_peaks = [p for p in ind_peaks if 50 <= p < 80]

            # Pick two points for divergence if available
            price_points = (
                sorted(price_peaks[-2:]) if len(price_peaks) >= 2 else [60, 75]
            )
            ind_points = sorted(ind_peaks[-2:]) if len(ind_peaks) >= 2 else [60, 75]

            # Create dataframe
            df = pd.DataFrame(
                {
                    "date": dates,
                    "price": price,
                    "indicator": indicator,
                }
            )

            # Create a figure with subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Price", "Indicator"),
            )

            # Add price trace
            fig.add_trace(
                go.Scatter(x=dates, y=price, name="Price", line=dict(color="blue")),
                row=1,
                col=1,
            )

            # Add indicator trace
            fig.add_trace(
                go.Scatter(
                    x=dates, y=indicator, name="Indicator", line=dict(color="purple")
                ),
                row=2,
                col=1,
            )

            # Add divergence lines
            if len(price_points) >= 2 and len(ind_points) >= 2:
                # Price divergence line
                fig.add_trace(
                    go.Scatter(
                        x=[dates[price_points[0]], dates[price_points[1]]],
                        y=[price[price_points[0]], price[price_points[1]]],
                        mode="lines+markers",
                        name="Price Trend",
                        line=dict(color="red", width=2, dash="dash"),
                        marker=dict(size=8),
                    ),
                    row=1,
                    col=1,
                )

                # Indicator divergence line
                fig.add_trace(
                    go.Scatter(
                        x=[dates[ind_points[0]], dates[ind_points[1]]],
                        y=[indicator[ind_points[0]], indicator[ind_points[1]]],
                        mode="lines+markers",
                        name="Indicator Trend",
                        line=dict(color="red", width=2, dash="dash"),
                        marker=dict(size=8),
                    ),
                    row=2,
                    col=1,
                )

            # Add pattern zone
            fig.add_vrect(
                x0=dates[pattern_zone.start],
                x1=dates[pattern_zone.stop - 1],
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Divergence Zone",
                annotation_position="top left",
                row=1,
                col=1,
            )

            fig.add_vrect(
                x0=dates[pattern_zone.start],
                x1=dates[pattern_zone.stop - 1],
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=2,
                col=1,
            )

            # Add vertical line for current date
            fig.add_vline(
                x=dates[-1],
                line_dash="dash",
                line_color="red",
                annotation_text="Current",
                annotation_position="top right",
                row=1,
                col=1,
            )

            divergence_type = (
                "Bullish" if pattern.get("expected_return", 0) > 0 else "Bearish"
            )
            fig.update_layout(
                title=f"Sample {divergence_type} Divergence Pattern", height=600
            )

            st.plotly_chart(fig, use_container_width=True)

        elif pattern_type == "volume" or category == "Volume Pattern":
            # Create a volume pattern
            base = 100
            trend = np.linspace(0, 20, 90) * (
                1 if pattern.get("expected_return", 0) > 0 else -1
            )
            noise = np.random.randn(90) * 5
            price = base + trend + noise

            # Create volume with a spike
            base_volume = np.random.randint(100000, 300000, 90)
            volume_spike = np.zeros(90)
            spike_location = 65
            volume_spike[spike_location - 2 : spike_location + 3] = np.array(
                [200000, 500000, 1000000, 400000, 200000]
            )
            volume = base_volume + volume_spike

            # Create pattern zone
            pattern_zone = slice(60, 75)

            # Create dataframe
            df = pd.DataFrame(
                {
                    "date": dates,
                    "price": price,
                    "volume": volume,
                }
            )

            # Create a figure with subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Price", "Volume"),
            )

            # Add price trace
            fig.add_trace(
                go.Scatter(x=dates, y=price, name="Price", line=dict(color="blue")),
                row=1,
                col=1,
            )

            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=volume,
                    name="Volume",
                    marker_color="darkblue",
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )

            # Add pattern zone
            fig.add_vrect(
                x0=dates[pattern_zone.start],
                x1=dates[pattern_zone.stop - 1],
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Volume Pattern Zone",
                annotation_position="top left",
                row=1,
                col=1,
            )

            fig.add_vrect(
                x0=dates[pattern_zone.start],
                x1=dates[pattern_zone.stop - 1],
                fillcolor="orange",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=2,
                col=1,
            )

            # Add vertical line for current date
            fig.add_vline(
                x=dates[-1],
                line_dash="dash",
                line_color="red",
                annotation_text="Current",
                annotation_position="top right",
                row=1,
                col=1,
            )

            # Add average volume line
            avg_volume = np.mean(base_volume)
            fig.add_hline(
                y=avg_volume,
                line_dash="dash",
                line_color="red",
                annotation_text="Avg Volume",
                annotation_position="left",
                row=2,
                col=1,
            )

            fig.update_layout(title="Sample Volume Pattern", height=600)

            st.plotly_chart(fig, use_container_width=True)

    def _get_active_patterns(self):
        """Get currently active patterns"""
        # In a real implementation, this would scan current market data
        # against pattern definitions to find active patterns
        # For now, we'll return a simulated list

        import random
        from datetime import datetime, timedelta

        # Create some sample active patterns
        if not hasattr(self, "_mock_active_patterns"):
            self._mock_active_patterns = []

            # Add sample patterns
            pattern_types = ["price", "indicator", "divergence", "volume"]
            categories = [
                "Price Action",
                "Technical Indicator",
                "Divergence Pattern",
                "Volume Pattern",
            ]

            for i in range(random.randint(0, 5)):
                p_type = random.choice(pattern_types)
                cat = random.choice(categories)

                expected_return = random.uniform(-5, 8)
                if expected_return > 0:
                    name_options = [
                        "Bullish Engulfing with Volume",
                        "Golden Cross with Confirmation",
                        "RSI Bullish Divergence",
                        "WERPI State 3 Transition",
                        "Bollinger Band Squeeze Breakout",
                        "Double Bottom Confirmation",
                    ]
                else:
                    name_options = [
                        "Bearish Engulfing Pattern",
                        "Death Cross Formation",
                        "RSI Bearish Divergence",
                        "Distribution Volume Pattern",
                        "Head and Shoulders Top",
                        "Rising Wedge Breakdown",
                    ]

                self._mock_active_patterns.append(
                    {
                        "name": random.choice(name_options),
                        "description": f'Sample {cat} pattern that signals {"bullish" if expected_return > 0 else "bearish"} price movement',
                        "conditions": [
                            {
                                "indicator": f"Sample Indicator {i+1}",
                                "operator": random.choice([">", "<", "==", "<=", ">="]),
                                "value": round(random.uniform(0, 100), 1),
                            }
                            for i in range(random.randint(1, 3))
                        ],
                        "expected_return": expected_return,
                        "reliability": random.uniform(0.5, 0.95),
                        "timeframe": random.choice(["1h", "4h", "1d"]),
                        "occurrences": random.randint(5, 30),
                        "discovery_date": (
                            datetime.now() - timedelta(days=random.randint(10, 100))
                        ).strftime("%Y-%m-%d"),
                        "category": cat,
                        "type": p_type,
                        "current_signal_strength": random.uniform(1.0, 3.0),
                        "last_activation": datetime.now()
                        - timedelta(hours=random.randint(1, 24)),
                    }
                )

        return self._mock_active_patterns

    def _load_all_patterns(self):
        """Load all discovered patterns from storage"""
        # In a real implementation, this would load patterns from files
        # For now, we'll return a simulated list that includes the active patterns
        # plus some additional historical patterns

        import random
        from datetime import datetime, timedelta

        # Get active patterns first
        patterns = self._get_active_patterns()

        # Add some inactive patterns
        if not hasattr(self, "_mock_inactive_patterns"):
            self._mock_inactive_patterns = []

            pattern_types = ["price", "indicator", "divergence", "volume"]
            categories = [
                "Price Action",
                "Technical Indicator",
                "Divergence Pattern",
                "Volume Pattern",
            ]

            for i in range(random.randint(10, 20)):
                p_type = random.choice(pattern_types)
                cat = random.choice(categories)

                expected_return = random.uniform(-8, 10)

                if expected_return > 0:
                    name_options = [
                        "Bullish Flag Breakout",
                        "Triple Bottom Reversal",
                        "Upward Channel Confirmation",
                        "Volume Price Confirmation",
                        "Oversold Bounce",
                        "Support Bounce with Volume",
                    ]
                else:
                    name_options = [
                        "Bearish Flag Breakdown",
                        "Triple Top Formation",
                        "Downward Channel Continuation",
                        "Lower Highs with Volume",
                        "Overbought Reversal",
                        "Resistance Rejection Pattern",
                    ]

                self._mock_inactive_patterns.append(
                    {
                        "name": random.choice(name_options),
                        "description": f'Historical {cat} pattern that signals {"bullish" if expected_return > 0 else "bearish"} price movement',
                        "conditions": [
                            {
                                "indicator": f"Sample Indicator {i+1}",
                                "operator": random.choice([">", "<", "==", "<=", ">="]),
                                "value": round(random.uniform(0, 100), 1),
                            }
                            for i in range(random.randint(1, 3))
                        ],
                        "expected_return": expected_return,
                        "reliability": random.uniform(0.3, 0.95),
                        "timeframe": random.choice(["1h", "4h", "1d"]),
                        "occurrences": random.randint(5, 50),
                        "discovery_date": (
                            datetime.now() - timedelta(days=random.randint(10, 200))
                        ).strftime("%Y-%m-%d"),
                        "category": cat,
                        "type": p_type,
                        "historical_examples": [
                            {
                                "date": (
                                    datetime.now()
                                    - timedelta(days=random.randint(10, 200))
                                ).strftime("%Y-%m-%d"),
                                "return": expected_return
                                * (1 + random.uniform(-0.5, 0.5)),
                            }
                            for _ in range(random.randint(5, 10))
                        ],
                    }
                )

        return patterns + self._mock_inactive_patterns

    def _is_pattern_active(self, pattern):
        """Check if a pattern is currently active"""
        active_patterns = self._get_active_patterns()
        return any(p.get("name") == pattern.get("name") for p in active_patterns)

    def _calculate_pattern_stats(self, patterns):
        """Calculate statistics about patterns"""
        stats = {}

        # Count patterns by category
        categories = {}
        for p in patterns:
            cat = p.get("category", "Other")
            if cat in categories:
                categories[cat] += 1
            else:
                categories[cat] = 1

        stats["category_counts"] = categories

        # Count by direction (bullish/bearish)
        bullish = sum(1 for p in patterns if p.get("expected_return", 0) > 0)
        bearish = sum(1 for p in patterns if p.get("expected_return", 0) < 0)

        stats["direction_counts"] = {"bullish": bullish, "bearish": bearish}

        # Calculate average metrics
        avg_return = (
            sum(p.get("expected_return", 0) for p in patterns) / len(patterns)
            if patterns
            else 0
        )
        avg_reliability = (
            sum(p.get("reliability", 0) for p in patterns) / len(patterns)
            if patterns
            else 0
        )

        stats["averages"] = {"return": avg_return, "reliability": avg_reliability}

        # Count by timeframe
        timeframes = {}
        for p in patterns:
            tf = p.get("timeframe", "unknown")
            if tf in timeframes:
                timeframes[tf] += 1
            else:
                timeframes[tf] = 1

        stats["timeframe_counts"] = timeframes

        return stats

    def analyze_seasonality(self, df, patterns=None):
        """
        Analyze seasonality in pattern performance with robust error handling
        
        Args:
            df: DataFrame with market data
            patterns: List of patterns to analyze
            
        Returns:
            dict: Seasonality analysis results
        """
        try:
            if df is None or df.empty:
                return {"error": "No data available for seasonality analysis"}
                
            if not patterns:
                patterns = self.pattern_manager.list_patterns()
                if not patterns:
                    return {"error": "No patterns available for seasonality analysis"}
            
            # Helper function to safely convert any data to pandas Series
            def safe_convert_to_series(data, index=None):
                """Safely convert any data type to a pandas Series"""
                if isinstance(data, pd.Series):
                    return data
                elif isinstance(data, pd.DataFrame):
                    if len(data.columns) > 0:
                        return data.iloc[:, 0]
                    else:
                        return pd.Series([], index=index if index is not None else [])
                elif isinstance(data, (float, int, np.number, np.float64, np.int64)):
                    if index is not None:
                        return pd.Series([float(data)], index=[index[0]] if len(index) > 0 else [0])
                    else:
                        return pd.Series([float(data)])
                elif isinstance(data, list) or isinstance(data, np.ndarray):
                    return pd.Series(data, index=index)
                else:
                    # Return empty series as fallback
                    return pd.Series([], index=index if index is not None else [])
            
            results = {}
            
            # Ensure we're working with pandas Series/DataFrames and not numpy scalars
            for pattern in patterns:
                try:
                    pattern_id = pattern.get('id', 'unknown')
                    results[pattern_id] = {}
                    
                    # Get historical examples for pattern
                    examples = pattern.get('historical_examples', [])
                    if not examples:
                        continue
                        
                    # Create a pandas Series for examples
                    dates = []
                    returns = []
                    
                    for example in examples:
                        try:
                            date = pd.to_datetime(example.get('date'))
                            ret = float(example.get('return', 0))
                            dates.append(date)
                            returns.append(ret)
                        except (ValueError, TypeError):
                            continue
                    
                    if not dates:
                        continue
                        
                    # Create a pandas Series
                    series = pd.Series(returns, index=dates)
                    
                    # Now we can safely do pandas operations
                    # Monthly analysis
                    if len(series) >= 12:
                        monthly = series.groupby(series.index.month).mean()
                        # Safely convert to dict
                        results[pattern_id]['monthly'] = monthly.to_dict()
                    
                    # Day of week analysis  
                    if len(series) >= 7:
                        daily = series.groupby(series.index.dayofweek).mean()
                        # Safely convert to dict
                        results[pattern_id]['day_of_week'] = daily.to_dict()
                    
                except Exception as e:
                    results[pattern_id] = {"error": f"Error analyzing pattern: {str(e)}"}
            
            return results
            
        except Exception as e:
            return {"error": f"Error in seasonality analysis: {str(e)}"}

    def analyze_seasonality(df):
        """Analyze daily, weekly, and monthly seasonality patterns."""
        st.header("Seasonality Analysis")
        
        # Make sure we have the necessary columns
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df.index.dayofweek
        if "month" not in df.columns:
            df["month"] = df.index.month
        
        # Day of week analysis
        st.subheader("Day of Week Effect")
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Get the day of week returns - THIS IS WHERE THE ERROR OCCURS
        day_returns = df.groupby("day_of_week")["Close"].pct_change().mean() * 100
        
        # FIX: Handle the case where day_returns is a scalar instead of a Series
        if isinstance(day_returns, (float, int, np.number, np.float64)):
            # It's a scalar, create a Series manually
            st.info(f"Converting scalar day_returns ({day_returns}) to Series")
            
            # Create a Series with the single value we have
            day_returns = pd.Series([day_returns], index=[0])
            
            # Expand to cover all days of week with NaN
            full_series = pd.Series(index=range(7), dtype=float)
            full_series.iloc[0] = day_returns[0]  # Put our value in Monday
            day_returns = full_series
        
        # Create the chart with reindexed data
        try:
            # Only try to reindex if it's a Series or DataFrame
            if hasattr(day_returns, 'reindex'):
                day_chart_data = day_returns.reindex(range(7))
                day_chart = pd.DataFrame({
                    "Day": day_names,
                    "Return %": day_chart_data.values
                })
                st.bar_chart(day_chart.set_index("Day"))
            else:
                # Still a scalar somehow, create a simple chart
                day_chart = pd.DataFrame({
                    "Day": day_names[0],
                    "Return %": [float(day_returns)]
                })
                st.bar_chart(day_chart.set_index("Day"))
        except Exception as e:
            st.error(f"Could not create day of week chart: {e}")
        
        # ALSO FIX THE MONTHLY ANALYSIS SECTION:
        st.subheader("Monthly Effect")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # Get monthly returns
        month_returns = df.groupby("month")["Close"].pct_change().mean() * 100
        
        # FIX: Handle scalar month_returns
        if isinstance(month_returns, (float, int, np.number, np.float64)):
            st.info(f"Converting scalar month_returns ({month_returns}) to Series")
            
            # Create a Series with the single value
            month_returns = pd.Series([month_returns], index=[1])  # January
            
            # Expand to cover all months with NaN
            full_series = pd.Series(index=range(1, 13), dtype=float)
            full_series.iloc[0] = month_returns[1]  # Put our value in January
            month_returns = full_series
        
        # Create the chart
        try:
            if hasattr(month_returns, 'reindex'):
                month_chart_data = month_returns.reindex(range(1, 13))
                month_chart = pd.DataFrame({
                    "Month": month_names,
                    "Return %": month_chart_data.values
                })
                st.bar_chart(month_chart.set_index("Month"))
            else:
                # Still a scalar somehow
                month_chart = pd.DataFrame({
                    "Month": month_names[0],
                    "Return %": [float(month_returns)]
                })
                st.bar_chart(month_chart.set_index("Month"))
        except Exception as e:
            st.error(f"Could not create monthly chart: {e}")

# Function to add the Pattern Discovery tab to your dashboard
def add_pattern_discovery_tab(df, ensemble_weighter=None):
    """
    Add the Pattern Discovery tab to your dashboard

    Args:
        df: DataFrame with market data
        ensemble_weighter: Optional ensemble weighter instance
    """
    st.title("Pattern Discovery")

    # Initialize and render the tab
    pattern_tab = PatternDiscoveryTab(df, ensemble_weighter)
    pattern_tab.render_tab()
