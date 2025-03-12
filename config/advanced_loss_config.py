"""
Advanced UI components for configuring loss functions with dynamic weights in the Streamlit dashboard.
"""

import altair as alt
import pandas as pd
import streamlit as st
import os
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use proper import path
from config.config_loader import get_value, set_value


def display_advanced_loss_settings():
    """Display advanced loss function configuration with dynamic weight adjustment."""
    st.subheader("Advanced Loss Function Configuration")

    # Get current loss function settings
    loss_functions = get_value("loss_functions.available")
    default_loss_fn = get_value("loss_functions.default")
    quantile_values = get_value("loss_functions.optimization.quantile_values")
    opt_settings = get_value("loss_functions.optimization")

    with st.expander("Loss Functions", expanded=False):
        # Display currently available loss functions
        st.write("Available Loss Functions:")

        # Create checkboxes for each potential loss function
        all_possible_loss_fns = {
            "mean_squared_error": "Mean Squared Error (MSE)",
            "mean_absolute_error": "Mean Absolute Error (MAE)",
            "mean_absolute_percentage_error": "Mean Absolute Percentage Error (MAPE)",
            "huber_loss": "Huber Loss",
            "log_cosh": "Log-Cosh Loss",
            "mean_squared_logarithmic_error": "Mean Squared Log Error (MSLE)",
            "quantile_loss": "Quantile Loss",
            "binary_crossentropy": "Binary Cross-Entropy (BCE)",
            "categorical_crossentropy": "Categorical Cross-Entropy",
        }

        # Keep track of selected loss functions
        selected_loss_fns = []

        for loss_fn, display_name in all_possible_loss_fns.items():
            is_active = loss_fn in loss_functions
            is_selected = st.checkbox(
                f"{display_name}", value=is_active, key=f"adv_loss_fn_{loss_fn}"
            )
            if is_selected:
                selected_loss_fns.append(loss_fn)

                # For quantile loss, show additional options
                if loss_fn == "quantile_loss":
                    st.write("Quantile Values:")
                    quantile_vals = [
                        float(q)
                        for q in st.text_input(
                            "Enter comma-separated quantile values (0-1)",
                            value=",".join([str(q) for q in quantile_values]),
                            key="quantile_values",
                        ).split(",")
                        if q.strip()
                    ]

        # Default loss function selection
        default_loss = st.selectbox(
            "Default Loss Function",
            options=selected_loss_fns,
            index=(
                selected_loss_fns.index(default_loss_fn)
                if default_loss_fn in selected_loss_fns
                else 0
            ),
            key="adv_default_loss_fn",
        )

    with st.expander("Dynamic Weight Adjustment", expanded=True):
        # Dynamic weight adjustment settings
        enable_dynamic_weights = st.checkbox(
            "Enable Dynamic Weight Adjustment",
            value=opt_settings.get("dynamic_weights", True),
            key="enable_dynamic_weights",
            help="Automatically adjust metric weights during optimization based on performance",
        )

        if enable_dynamic_weights:
            # Weight adjustment method
            adjustment_method = st.selectbox(
                "Weight Adjustment Method",
                options=["performance_based", "adaptive", "thompson_sampling"],
                index=["performance_based", "adaptive", "thompson_sampling"].index(
                    opt_settings.get("weight_adjustment", {}).get(
                        "method", "performance_based"
                    )
                ),
                key="weight_adjustment_method",
                help="""
                    Methods for dynamic weight adjustment:
                    - performance_based: Assigns higher weights to metrics that show more improvement
                    - adaptive: Assigns higher weights to metrics with high variance
                    - thompson_sampling: Uses multi-armed bandit approach for exploration/exploitation
                """,
            )

            # Exploration factor
            exploration_factor = st.slider(
                "Exploration Factor",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    opt_settings.get("weight_adjustment", {}).get(
                        "exploration_factor", 0.2
                    )
                ),
                step=0.05,
                key="exploration_factor",
                help="Higher values favor exploration of different metrics. Range: 0.0-1.0",
            )

            # History window size
            history_window = st.slider(
                "History Window Size",
                min_value=1,
                max_value=20,
                value=int(
                    opt_settings.get("weight_adjustment", {}).get("history_window", 5)
                ),
                step=1,
                key="history_window",
                help="Number of recent trials to consider for weight updates",
            )

            # Show current metric weights
            st.subheader("Initial Metric Weights")
            st.write(
                "These weights will be adjusted automatically during optimization."
            )

            weighted_metrics = opt_settings.get("weighted_metrics", {})

            col1, col2 = st.columns(2)

            with col1:
                rmse_weight = st.number_input(
                    "RMSE Weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(weighted_metrics.get("rmse", 1.0)),
                    step=0.1,
                    key="rmse_weight_dyn",
                )

                mape_weight = st.number_input(
                    "MAPE Weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(weighted_metrics.get("mape", 1.0)),
                    step=0.1,
                    key="mape_weight_dyn",
                )

            with col2:
                huber_weight = st.number_input(
                    "Huber Loss Weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(weighted_metrics.get("huber", 0.8)),
                    step=0.1,
                    key="huber_weight_dyn",
                )

                quantile_weight = st.number_input(
                    "Quantile Loss Weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(weighted_metrics.get("quantile", 1.0)),
                    step=0.1,
                    key="quantile_weight_dyn",
                )

            # Display weight history if available
            if (
                "tuning_progress" in st.session_state
                and "weight_history" in st.session_state["tuning_progress"]
            ):
                st.subheader("Weight Adjustment History")

                weight_history = st.session_state["tuning_progress"]["weight_history"]
                if weight_history:
                    # Convert history to a dataframe for visualization
                    history_data = []
                    for entry in weight_history:
                        trial_data = {"trial": entry["trial"]}
                        trial_data.update(entry["weights"])
                        history_data.append(trial_data)

                    history_df = pd.DataFrame(history_data)

                    # Plot weight history
                    chart = (
                        alt.Chart(
                            history_df.melt(
                                "trial", var_name="metric", value_name="weight"
                            )
                        )
                        .mark_line()
                        .encode(
                            x="trial:Q",
                            y="weight:Q",
                            color="metric:N",
                            tooltip=["trial", "metric", "weight"],
                        )
                        .properties(
                            title="Metric Weight Adaptation Over Time",
                            width=600,
                            height=300,
                        )
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No weight adjustment history available yet.")

    with st.expander("Quantile Loss Settings", expanded=False):
        st.write("Configure quantile values for Quantile Loss function")

        # Get current quantile values
        current_quantiles = opt_settings.get("quantile_values", [0.1, 0.5, 0.9])

        # Display as a text input for easy editing
        quantile_input = st.text_input(
            "Quantile Values (comma-separated, between 0 and 1)",
            value=", ".join([str(q) for q in current_quantiles]),
            key="quantile_values_input",
            help="Quantile values determine prediction intervals for Quantile Loss",
        )

        # Parse and validate input
        try:
            new_quantiles = [
                float(q.strip()) for q in quantile_input.split(",") if q.strip()
            ]
            valid_quantiles = [q for q in new_quantiles if 0 <= q <= 1]

            if len(valid_quantiles) != len(new_quantiles):
                st.warning(
                    "Some quantile values were outside the valid range [0,1] and were ignored."
                )

            if valid_quantiles:
                quantile_values = valid_quantiles
            else:
                st.error("No valid quantile values provided. Using default values.")
                quantile_values = [0.1, 0.5, 0.9]
        except Exception as e:
            st.error(f"Error parsing quantile values: {e}")
            quantile_values = [0.1, 0.5, 0.9]

        # Show example of quantile loss interpretation
        st.subheader("Quantile Loss Interpretation")
        st.write(
            """
        Quantile Loss helps with prediction intervals:
        - q=0.1: Prediction has 10% chance of being below this value
        - q=0.5: Median prediction (50% above, 50% below)
        - q=0.9: Prediction has 90% chance of being below this value
        """
        )

    with st.expander("Hyperparameter Tuning Integration", expanded=False):
        st.write("Configure how loss functions integrate with hyperparameter tuning")

        # Loss function importance in hyperparameter tuning
        loss_fn_importance = st.slider(
            "Loss Function Importance",
            min_value=0.1,
            max_value=1.0,
            value=float(
                opt_settings.get("hyperparameter_weights", {}).get(
                    "loss_fn_importance", 0.7
                )
            ),
            step=0.05,
            key="loss_fn_importance",
            help="How much to prioritize loss function selection vs other hyperparameters",
        )

        # Option to optimize loss function per model
        optimize_per_model = st.checkbox(
            "Optimize Loss Function Per Model Type",
            value=opt_settings.get("optimize_per_model", True),
            key="optimize_per_model",
            help="Allow each model type to use a different optimal loss function",
        )

    # Save button
    if st.button("Save Advanced Loss Settings"):
        try:
            # Prepare data to save

            # 1. Loss functions
            set_value("loss_functions.available", selected_loss_fns)
            set_value("loss_functions.default", default_loss)

            # 2. Dynamic weight adjustment settings
            set_value(
                "loss_functions.optimization.dynamic_weights", enable_dynamic_weights
            )

            if enable_dynamic_weights:
                set_value(
                    "loss_functions.optimization.weight_adjustment.method",
                    adjustment_method,
                )
                set_value(
                    "loss_functions.optimization.weight_adjustment.exploration_factor",
                    exploration_factor,
                )
                set_value(
                    "loss_functions.optimization.weight_adjustment.history_window",
                    history_window,
                )

            # 3. Weighted metrics
            weighted_metrics = {
                "rmse": rmse_weight,
                "mape": mape_weight,
                "huber": huber_weight,
                "quantile": quantile_weight,
            }
            set_value("loss_functions.optimization.weighted_metrics", weighted_metrics)

            # 4. Quantile values
            set_value("loss_functions.optimization.quantile_values", quantile_values)

            # 5. Hyperparameter tuning integration
            set_value(
                "loss_functions.hyperparameter_weights.loss_fn_importance",
                loss_fn_importance,
            )
            set_value(
                "loss_functions.optimization.optimize_per_model", optimize_per_model
            )

            st.success("Settings saved successfully!")

        except Exception as e:
            st.error(f"Error saving settings: {e}")


def display_weight_history_visualization():
    """Display a visualization of how weights evolved during optimization."""
    st.subheader("Metric Weight Evolution")

    if (
        "tuning_progress" in st.session_state
        and "weight_history" in st.session_state["tuning_progress"]
    ):
        weight_history = st.session_state["tuning_progress"]["weight_history"]

        if weight_history:
            # Convert to dataframe
            history_data = []
            for entry in weight_history:
                trial_data = {"trial": entry["trial"]}
                if "weights" in entry:
                    trial_data.update(entry["weights"])
                history_data.append(trial_data)

            df = pd.DataFrame(history_data)

            # Create line chart
            chart = (
                alt.Chart(df.melt("trial", var_name="metric", value_name="weight"))
                .mark_line()
                .encode(
                    x="trial:Q",
                    y="weight:Q",
                    color="metric:N",
                    tooltip=["trial", "metric", "weight"],
                )
                .properties(
                    title="Weight Evolution During Optimization", width=700, height=400
                )
            )

            st.altair_chart(chart, use_container_width=True)

            # Show table with weight history
            if st.checkbox("Show Weight History Table"):
                st.dataframe(df)
        else:
            st.info("No weight adjustment history available yet.")
    else:
        st.info("Start optimization to see weight adjustment history.")


if __name__ == "__main__":
    # This allows testing this UI component independently
    st.set_page_config(page_title="Advanced Loss Function Configuration", layout="wide")
    display_advanced_loss_settings()
