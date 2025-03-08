"""
UI components for configuring loss functions in the Streamlit dashboard.
"""

import streamlit as st

from config.config_loader import set_value
from src.training.walk_forward import (
    get_available_loss_functions,
    get_loss_optimization_settings,
)


def display_loss_function_settings():
    """Display and manage loss function configuration in the dashboard."""
    st.subheader("Loss Function Configuration")

    # Get current loss function settings
    loss_functions, default_loss_fn = get_available_loss_functions()
    opt_settings = get_loss_optimization_settings()

    with st.expander("Loss Function Settings", expanded=False):
        # Display currently available loss functions
        st.write("Available Loss Functions:")

        # Create checkboxes for each potential loss function
        all_possible_loss_fns = {
            "mean_squared_error": "Mean Squared Error (MSE)",
            "mean_absolute_error": "Mean Absolute Error (MAE)",
            "huber_loss": "Huber Loss",
            "log_cosh": "Log-Cosh Loss",
            "mean_squared_logarithmic_error": "Mean Squared Log Error (MSLE)",
            "binary_crossentropy": "Binary Cross-Entropy (BCE)",
            "categorical_crossentropy": "Categorical Cross-Entropy",
        }

        # Keep track of selected loss functions
        selected_loss_fns = []

        for loss_fn, display_name in all_possible_loss_fns.items():
            is_active = loss_fn in loss_functions
            is_selected = st.checkbox(
                f"{display_name}", value=is_active, key=f"loss_fn_{loss_fn}"
            )
            if is_selected:
                selected_loss_fns.append(loss_fn)

        # Default loss function selection
        default_loss = st.selectbox(
            "Default Loss Function",
            options=selected_loss_fns,
            index=(
                selected_loss_fns.index(default_loss_fn)
                if default_loss_fn in selected_loss_fns
                else 0
            ),
            key="default_loss_fn",
        )

        # Optimization settings
        st.subheader("Optimization Settings")

        select_optimal = st.checkbox(
            "Allow Optuna to select optimal loss function",
            value=opt_settings["select_optimal"],
            key="select_optimal_loss",
        )

        optimize_per_model = st.checkbox(
            "Optimize loss function for each model type separately",
            value=opt_settings["optimize_per_model"],
            key="optimize_per_model_loss",
        )

        # Metric weights
        st.subheader("Metric Weights for Optimization")

        col1, col2, col3 = st.columns(3)
        with col1:
            rmse_weight = st.number_input(
                "RMSE Weight",
                min_value=0.0,
                max_value=10.0,
                value=float(opt_settings["weighted_metrics"].get("rmse", 1.0)),
                step=0.1,
                key="rmse_weight",
            )

        with col2:
            mape_weight = st.number_input(
                "MAPE Weight",
                min_value=0.0,
                max_value=10.0,
                value=float(opt_settings["weighted_metrics"].get("mape", 1.0)),
                step=0.1,
                key="mape_weight",
            )

        with col3:
            huber_weight = st.number_input(
                "Huber Weight",
                min_value=0.0,
                max_value=10.0,
                value=float(opt_settings["weighted_metrics"].get("huber", 0.8)),
                step=0.1,
                key="huber_weight",
            )

        # Loss function importance in hyperparameter tuning
        loss_fn_importance = st.slider(
            "Loss Function Importance in Hyperparameter Tuning",
            min_value=0.1,
            max_value=1.0,
            value=float(opt_settings["loss_fn_importance"]),
            step=0.05,
            key="loss_fn_importance",
        )

        # Save button
        if st.button("Save Loss Function Settings"):
            try:
                # Update configuration
                set_value("loss_functions.available", selected_loss_fns)
                set_value("loss_functions.default", default_loss)
                set_value("loss_functions.optimization.select_optimal", select_optimal)
                set_value(
                    "loss_functions.optimization.optimize_per_model", optimize_per_model
                )

                weighted_metrics = {
                    "rmse": rmse_weight,
                    "mape": mape_weight,
                    "huber": huber_weight,
                }
                set_value(
                    "loss_functions.optimization.weighted_metrics", weighted_metrics
                )
                set_value(
                    "loss_functions.hyperparameter_weights.loss_fn_importance",
                    loss_fn_importance,
                )

                st.success("Loss function settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving loss function settings: {e}")


if __name__ == "__main__":
    # This allows testing this UI component independently
    st.set_page_config(page_title="Loss Function Configuration", layout="wide")
    display_loss_function_settings()
