import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns
import streamlit as st
import yaml
import altair as alt
from config import MAPE_THRESHOLD, MODEL_TYPES, RMSE_THRESHOLD
from config.config_loader import DB_DIR, HYPERPARAMS_DIR

# Define the directory for database files - update to use Models folder
# DB_DIR = os.path.join(DATA_DIR, "Models", "DB")
# os.makedirs(DB_DIR, exist_ok=True)

# Define hyperparameter storage directory - update to use Models folder
# HYPERPARAMS_DIR = os.path.join(DATA_DIR, "Models", "Hyperparameters")
# os.makedirs(HYPERPARAMS_DIR, exist_ok=True)


# Create a global registry for hyperparameters if it doesn't exist
if "HYPERPARAMETER_REGISTRY" not in globals():
    HYPERPARAMETER_REGISTRY = {}

    # Initialize with some default parameters for different model types
    for model_type in MODEL_TYPES:
        if model_type == "lstm":
            HYPERPARAMETER_REGISTRY.update(
                {
                    "lstm_units": {
                        "type": "int",
                        "default": 64,
                        "range": [32, 256],
                        "group": "lstm",
                    },
                    "lstm_layers": {
                        "type": "int",
                        "default": 2,
                        "range": [1, 5],
                        "group": "lstm",
                    },
                    "lstm_dropout": {
                        "type": "float",
                        "default": 0.2,
                        "range": [0.0, 0.5],
                        "group": "lstm",
                    },
                }
            )
        elif model_type == "nbeats":
            HYPERPARAMETER_REGISTRY.update(
                {
                    "nbeats_blocks": {
                        "type": "int",
                        "default": 3,
                        "range": [1, 8],
                        "group": "nbeats",
                    },
                    "nbeats_layers": {
                        "type": "int",
                        "default": 4,
                        "range": [2, 8],
                        "group": "nbeats",
                    },
                    "nbeats_width": {
                        "type": "int",
                        "default": 256,
                        "range": [64, 512],
                        "group": "nbeats",
                    },
                }
            )

# Add preset configurations
PRESET_CONFIGS = {
    "quick": {
        "name": "Quick Exploration",
        "description": "Fast tuning with narrower ranges, fewer trials",
        "n_trials": 50,
        "range_multiplier": 0.5,  # Narrows ranges by half
        "pruning_patience": 5,
    },
    "balanced": {
        "name": "Balanced",
        "description": "Good balance between speed and exploration",
        "n_trials": 200,
        "range_multiplier": 1.0,  # Default ranges
        "pruning_patience": 10,
    },
    "thorough": {
        "name": "Thorough Exploration",
        "description": "Extensive search with wider ranges",
        "n_trials": 500,
        "range_multiplier": 1.5,  # Expands ranges by 50%
        "pruning_patience": 20,
    },
    "extreme": {
        "name": "Extreme Exploration",
        "description": "Very wide parameter search with maximum trials",
        "n_trials": 1000,
        "range_multiplier": 2.0,  # Doubles the ranges
        "pruning_patience": 30,
    },
}


# Add this function to estimate trial time
def estimate_trial_time(study_name):
    """Estimate average trial time based on completed trials"""
    try:
        storage = f"sqlite:///{os.path.join(DB_DIR, f'{study_name}.db')}"
        study = optuna.load_study(study_name=study_name, storage=storage)

        durations = []
        for trial in study.trials:
            if hasattr(trial, "duration") and trial.duration is not None:
                durations.append(trial.duration.total_seconds())

        if durations:
            avg_duration = sum(durations) / len(durations)
            return avg_duration

    except Exception as e:
        st.warning(f"Error estimating trial time: {e}")

    return None


def create_save_load_interface():
    """Create interface for saving and loading hyperparameter configurations"""
    st.header("Save/Load Hyperparameter Configurations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Save Current Configuration")

        # Get a name for the configuration
        config_name = st.text_input(
            "Configuration Name", placeholder="my_hyperparam_config"
        )

        # Add optional description
        config_desc = st.text_area(
            "Description (optional)", placeholder="Description of this configuration"
        )

        if st.button("Save Configuration"):
            if not config_name:
                st.error("Please enter a configuration name")
            else:
                # Create configuration object
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Extract current parameter ranges
                param_ranges = {}
                for param_name, config in HYPERPARAMETER_REGISTRY.items():
                    if config["range"]:
                        param_ranges[param_name] = config["range"]

                # Create the config object
                hyperparameter_config = {
                    "name": config_name,
                    "description": config_desc,
                    "timestamp": timestamp,
                    "param_ranges": param_ranges,
                }

                # Save to file
                config_filename = config_name.lower().replace(" ", "_") + ".yaml"
                config_path = os.path.join(HYPERPARAMS_DIR, config_filename)

                try:
                    with open(config_path, "w") as f:
                        yaml.dump(hyperparameter_config, f)
                    st.success(f"Configuration saved to {config_path}")
                except Exception as e:
                    st.error(f"Error saving configuration: {e}")

    with col2:
        st.subheader("Load Configuration")

        # Get list of saved configurations
        saved_configs = []
        try:
            for file in os.listdir(HYPERPARAMS_DIR):
                if file.endswith(".yaml") and not file.startswith("tuning_config_"):
                    saved_configs.append(file)
        except Exception as e:
            st.error(f"Error listing configurations: {e}")

        if not saved_configs:
            st.info("No saved configurations found.")
            return

        # Select configuration to load
        selected_config = st.selectbox(
            "Select Configuration",
            saved_configs,
            format_func=lambda x: x.replace(".yaml", "").replace("_", " ").title(),
        )

        # Load and display configuration details
        if selected_config:
            config_path = os.path.join(HYPERPARAMS_DIR, selected_config)

            try:
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)

                st.write(
                    f"**Description**: {config_data.get('description', 'No description')}"
                )
                st.write(f"**Created**: {config_data.get('timestamp', 'Unknown')}")
                st.write(f"**Parameters**: {len(config_data.get('param_ranges', {}))}")

                # Add button to load this configuration
                if st.button("Load This Configuration"):
                    # Apply the parameter ranges
                    param_ranges = config_data.get("param_ranges", {})

                    for param_name, range_value in param_ranges.items():
                        if param_name in HYPERPARAMETER_REGISTRY:
                            HYPERPARAMETER_REGISTRY[param_name]["range"] = range_value

                    st.success(f"Loaded configuration '{config_data.get('name')}'")
                    st.experimental_rerun()  # Refresh the page to show updated values
            except Exception as e:
                st.error(f"Error loading configuration: {e}")


# Add this to create a new "Presets" tab
def create_presets_tab():
    st.header("Hyperparameter Presets")

    col1, col2 = st.columns([3, 1])

    with col1:
        preset_option = st.selectbox(
            "Select a preset configuration",
            list(PRESET_CONFIGS.keys()),
            format_func=lambda x: PRESET_CONFIGS[x]["name"],
        )

        preset = PRESET_CONFIGS[preset_option]
        st.write(f"**Description:** {preset['description']}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Number of Trials", preset["n_trials"])
        with col_b:
            st.metric("Range Multiplier", preset["range_multiplier"])
        with col_c:
            st.metric("Pruning Patience", preset["pruning_patience"])

    with col2:
        if st.button("Apply Preset", key="apply_preset"):
            st.session_state["active_preset"] = preset_option

            # Adjust hyperparameter ranges based on preset
            apply_preset_to_ranges(preset_option)

            st.success(f"Applied '{preset['name']}' preset")

            # Request page refresh to show updated ranges
            st.experimental_rerun()

    st.info(
        "⚠️ To start tuning with these settings, use the Start Tuning button in the Control Panel sidebar."
    )


def apply_preset_to_ranges(preset_key):
    """Apply a preset's range multiplier to all hyperparameters"""
    preset = PRESET_CONFIGS[preset_key]
    multiplier = preset["range_multiplier"]

    # Store original ranges if not already stored
    if "original_ranges" not in st.session_state:
        st.session_state["original_ranges"] = {}
        for param, config in HYPERPARAMETER_REGISTRY.items():
            if config["type"] in ["int", "float"] and config["range"]:
                st.session_state["original_ranges"][param] = config["range"].copy()

    # Apply multiplier to numerical ranges
    for param, config in HYPERPARAMETER_REGISTRY.items():
        if (
            config["type"] in ["int", "float"]
            and param in st.session_state["original_ranges"]
        ):
            orig_min, orig_max = st.session_state["original_ranges"][param]

            # Calculate center
            center = (orig_min + orig_max) / 2

            # Calculate half range and apply multiplier
            half_range = (orig_max - orig_min) / 2
            new_half_range = half_range * multiplier

            # Calculate new min/max
            new_min = (
                max(0, center - new_half_range)
                if orig_min >= 0
                else center - new_half_range
            )
            new_max = center + new_half_range

            # Update in registry
            HYPERPARAMETER_REGISTRY[param]["range"] = [new_min, new_max]


# Add a visual range editor function
def visual_range_editor():
    """Create a visual editor for parameter ranges"""
    st.header("Visual Range Editor")

    # Group parameters by category
    params_by_group = {}
    for name, config in HYPERPARAMETER_REGISTRY.items():
        group = config.get("group", "misc")
        if group not in params_by_group:
            params_by_group[group] = []
        params_by_group[group].append((name, config))

    # Create tabs for each group
    group_tabs = st.tabs(list(params_by_group.keys()))

    for i, (group, params) in enumerate(params_by_group.items()):
        with group_tabs[i]:
            st.write(f"### {group.capitalize()} Parameters")

            # Filter numerical parameters
            numerical_params = [
                (name, config)
                for name, config in params
                if config["type"] in ["int", "float"]
            ]

            if numerical_params:
                for name, config in numerical_params:
                    if config["range"]:
                        min_val, max_val = config["range"]

                        st.write(f"**{name}** (current: {config['default']})")

                        # Create range slider
                        if config["type"] == "int":
                            # For integers, ensure range is appropriate
                            display_min = int(min_val)
                            display_max = int(max_val * 1.5)  # Allow extra room
                            step = max(1, (display_max - display_min) // 100)

                            new_range = st.slider(
                                f"Range for {name}",
                                min_value=display_min,
                                max_value=display_max,
                                value=(int(min_val), int(max_val)),
                                step=step,
                                key=f"range_{name}",
                            )
                        else:
                            # For floats
                            display_min = float(min_val)
                            display_max = float(max_val * 1.5)  # Allow extra room
                            step = (display_max - display_min) / 100

                            new_range = st.slider(
                                f"Range for {name}",
                                min_value=display_min,
                                max_value=display_max,
                                value=(float(min_val), float(max_val)),
                                step=step,
                                key=f"range_{name}",
                            )

                        # Update the range if changed
                        if new_range != (min_val, max_val):
                            HYPERPARAMETER_REGISTRY[name]["range"] = list(new_range)
                            st.success(f"Updated range for {name}")

            # Handle categorical parameters
            categorical_params = [
                (name, config)
                for name, config in params
                if config["type"] == "categorical"
            ]

            if categorical_params:
                st.write("### Categorical Parameters")

                for name, config in categorical_params:
                    if config["range"]:
                        st.write(f"**{name}** (current: {config['default']})")

                        # Create multiselect for options
                        options = config["range"]
                        selected = st.multiselect(
                            f"Options for {name}",
                            options=options,
                            default=options,
                            key=f"cat_{name}",
                        )

                        # Update if changed
                        if selected and set(selected) != set(options):
                            HYPERPARAMETER_REGISTRY[name]["range"] = selected
                            st.success(f"Updated options for {name}")


# Add a function to visualize hyperparameter trends
def visualize_hyperparameter_trends(study_name):
    """Visualize trends in hyperparameter values over trials"""
    try:
        # Load study
        storage = f"sqlite:///{os.path.join(DB_DIR, f'{study_name}.db')}"
        study = optuna.load_study(study_name=study_name, storage=storage)

        # Get completed trials
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_trials) < 5:
            st.info("Not enough completed trials for trend analysis.")
            return

        # Get parameter names
        param_names = list(completed_trials[0].params.keys())

        # Create selection for parameters to visualize
        selected_params = st.multiselect(
            "Select parameters to visualize",
            options=param_names,
            default=param_names[:5] if len(param_names) > 5 else param_names,
        )

        if not selected_params:
            return

        # Create dataframe of parameter values
        df = pd.DataFrame(
            {
                "trial": [t.number for t in completed_trials],
                "value": [t.value for t in completed_trials],
            }
        )

        for param in selected_params:
            df[param] = [t.params.get(param, None) for t in completed_trials]

        # Create plots
        fig, ax = plt.subplots(
            len(selected_params), 1, figsize=(10, 3 * len(selected_params))
        )

        # Handle single parameter case
        if len(selected_params) == 1:
            ax = [ax]

        for i, param in enumerate(selected_params):
            # Get parameter values
            df[param]

            # Create scatter plot
            sns.scatterplot(
                x="trial", y=param, data=df, hue="value", palette="viridis", ax=ax[i]
            )

            # Add trend line
            try:
                sns.regplot(
                    x="trial",
                    y=param,
                    data=df,
                    scatter=False,
                    line_kws={"color": "red"},
                    ax=ax[i],
                )
            except:
                pass  # Skip trend line if it fails

            # Set title and labels
            ax[i].set_title(f"{param} over trials")
            ax[i].set_xlabel("Trial number")
            ax[i].set_ylabel(param)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error visualizing parameter trends: {e}")


# Add a test duration estimator function
def create_duration_estimator():
    """Create an interface to estimate test duration based on parameters"""
    st.header("Test Duration Estimator")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Get available studies for time estimates
        studies = []
        for file in os.listdir(DB_DIR):
            if file.endswith(".db"):
                study_name = file.replace(".db", "")
                studies.append(study_name)

        if not studies:
            st.info("No completed studies found for time estimation.")
            return

        selected_study = st.selectbox(
            "Select a reference study for time estimation", studies
        )

        # Get time per trial from this study
        avg_time = estimate_trial_time(selected_study)

        if avg_time:
            st.write(f"Average time per trial: **{avg_time:.2f}** seconds")

            num_trials = st.number_input(
                "Number of trials to run", min_value=10, value=100, step=10
            )

            # Calculate total time
            total_seconds = avg_time * num_trials
            estimated_time = timedelta(seconds=total_seconds)

            # Format as days, hours, minutes, seconds
            days = estimated_time.days
            hours, remainder = divmod(estimated_time.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Display estimate
            if days > 0:
                time_str = f"{days}d {hours}h {minutes}m {seconds}s"
            elif hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            else:
                time_str = f"{minutes}m {seconds}s"

            st.info(f"Estimated total time: **{time_str}**")

            # Recommendations based on time
            if total_seconds < 3600:  # Less than 1 hour
                st.success("✅ Quick run - good for iterative testing")
            elif total_seconds < 86400:  # Less than 1 day
                st.success(
                    "✅ Moderate run - good balance between speed and thoroughness"
                )
            else:  # More than 1 day
                st.warning(
                    "⚠️ Long run - consider running overnight or using a preset with fewer trials"
                )
        else:
            st.warning("Could not estimate time from selected study.")

    with col2:
        st.subheader("Duration Factors")
        st.write("Factors that affect test duration:")

        st.markdown(
            """
        - **Number of trials**: More trials = longer duration
        - **Parameter ranges**: Wider ranges = more exploration
        - **Model complexity**: More complex models take longer to train
        - **Early stopping**: Can reduce time for poor trials
        - **Hardware**: GPU acceleration greatly reduces time
        """
        )

        st.write(
            "**Tip**: Start with quick presets for initial exploration, then use thorough presets for final optimization."
        )


# Add this to the main function to include the new tabs
def main():
    # ... existing code ...

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to",
        [
            "Current Configuration",
            "Visual Range Editor",
            "Presets",
            "Pruning Settings",
            "Study Visualization",
            "Parameter Trends",
            "Test Duration Estimator",
            "Optuna Parameter Search",
            "Run Optimization",
        ],
    )

    if section == "Current Configuration":
        show_current_config()
    elif section == "Visual Range Editor":
        visual_range_editor()
    elif section == "Presets":
        create_presets_tab()
    elif section == "Pruning Settings":
        create_pruning_settings()
    elif section == "Save/Load Configurations":
        create_save_load_interface()
    elif section == "Study Visualization":
        visualize_studies()
    elif section == "Parameter Trends":
        # Get available studies
        studies = []
        for file in os.listdir(DB_DIR):
            if file.endswith(".db"):
                study_name = file.replace(".db", "")
                studies.append(study_name)

        if studies:
            selected_study = st.selectbox(
                "Select study for parameter trend analysis", studies
            )
            visualize_hyperparameter_trends(selected_study)
        else:
            st.info("No studies available for trend analysis.")
    elif section == "Test Duration Estimator":
        create_duration_estimator()
    # Add new sections
    elif section == "Optuna Parameter Search":
        create_optuna_parameter_ui()
    elif section == "Run Optimization":
        create_optuna_run_ui()


def show_current_config():
    """Show the current hyperparameter configuration."""
    st.header("Current Hyperparameter Configuration")

    # Get active models from session state or use all models
    active_models = st.session_state.get("active_model_types", MODEL_TYPES)

    # First, allow model selection
    with st.expander("Select Models to Include", expanded=True):
        selected_model_types = st.multiselect(
            "Select Model Types for the Ensemble:",
            options=MODEL_TYPES,
            default=active_models,
            help="Choose which model types to include in hyperparameter tuning.",
        )

        # Save the selected models to session state
        st.session_state["active_model_types"] = selected_model_types

        # Display active models
        st.info(f"Active models: {', '.join(selected_model_types)}")

    # Display current hyperparameters
    # Show hyperparameters only for selected models
    for group in sorted(
        set(param["group"] for param in HYPERPARAMETER_REGISTRY.values())
    ):
        with st.expander(f"{group.upper()} Parameters"):
            for name, param in HYPERPARAMETER_REGISTRY.items():
                # Skip parameters for models that are not selected
                if any(model_type in name for model_type in MODEL_TYPES) and not any(
                    model_type in name for model_type in selected_model_types
                ):
                    continue

                if param["group"] == group:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"**{name}**")
                    with col2:
                        if param["type"] == "bool":
                            st.checkbox(
                                "",
                                value=param["default"],
                                key=f"param_{name}",
                                disabled=True,
                            )
                        else:
                            st.write(param["default"])

                    # Show range if available
                    if param["range"]:
                        st.write(f"Range: {param['range']}")
                    st.divider()


# Update your optimization function to filter based on selected models
def create_optuna_study():
    """Create and run an Optuna study for hyperparameter optimization."""
    # Get active models
    active_models = st.session_state.get("active_model_types", MODEL_TYPES)

    # Create trial function
    @st.cache_data(ttl=60)
    def objective(trial):
        # Filter for only active models
        model_params = {}

        if "lstm" in active_models:
            # Add LSTM parameters
            model_params["lstm"] = {
                "units": trial.suggest_int("lstm_units", 32, 256),
                # ...other lstm params
            }

        if "nbeats" in active_models:
            # Add N-BEATS parameters
            model_params["nbeats"] = {
                "num_blocks": trial.suggest_int("nbeats_blocks", 1, 8),
                "num_layers": trial.suggest_int("nbeats_layers", 2, 8),
                "layer_width": trial.suggest_int("nbeats_width", 64, 512, log=True),
                # ...other nbeats params
            }

        # Only include parameters for active models
        # ...rest of the objective function...

    # ...existing code to run the study...


# Add the missing visualize_studies function
def visualize_studies():
    """Visualize Optuna studies from the database."""
    st.header("Study Visualization")

    # List available studies
    studies = []
    if os.path.exists(DB_DIR):
        for file in os.listdir(DB_DIR):
            if file.endswith(".db"):
                studies.append(file.replace(".db", ""))

    if not studies:
        st.info("No studies found in the database directory.")
        return

    # Select study to visualize
    selected_study = st.selectbox("Select study to visualize", studies)

    if not selected_study:
        return

    try:
        # Load the study
        storage = f"sqlite:///{os.path.join(DB_DIR, f'{selected_study}.db')}"
        study = optuna.load_study(study_name=selected_study, storage=storage)

        # Display study information
        st.subheader(f"Study: {selected_study}")
        st.write(f"Number of trials: {len(study.trials)}")

        # Get best trial
        if study.best_trial:
            st.write(f"Best value: {study.best_value:.6f}")
            st.write("Best parameters:")
            for param, value in study.best_params.items():
                st.write(f"- {param}: {value}")

        # Plot optimization history
        st.subheader("Optimization History")
        fig = optuna.visualization.plot_optimization_history(study)
        st.plotly_chart(fig, use_container_width=True)

        # Plot parameter importances if we have enough completed trials
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed_trials) >= 5:
            try:
                st.subheader("Parameter Importances")
                fig = optuna.visualization.plot_param_importances(study)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Could not generate parameter importance plot: {e}")

        # Plot contour if we have parameters
        if study.best_params and len(study.best_params) >= 2:
            try:
                st.subheader("Contour Plot")
                param_names = list(study.best_params.keys())
                if len(param_names) >= 2:
                    fig = optuna.visualization.plot_contour(study, param_names[:2])
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Could not generate contour plot: {e}")

    except Exception as e:
        st.error(f"Error visualizing study: {e}")


# Add this function to create pruning settings
def create_pruning_settings():
    """Create interface for adjusting pruning settings"""
    st.header("Pruning Settings")

    with st.expander("What is Pruning?", expanded=False):
        st.write(
            """
        **Pruning** stops unpromising trials early to save computation time. There are two types:
        
        1. **Median-based pruning**: Stops trials that perform worse than the median of completed trials
        2. **Absolute threshold pruning**: Stops trials that exceed absolute performance thresholds
        """
        )

    # Toggle pruning on/off
    pruning_enabled = st.checkbox(
        "Enable Pruning",
        value=HYPERPARAMETER_REGISTRY["pruning_enabled"]["default"],
        help="When enabled, poorly performing trials will be stopped early",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Median-based Pruning")

        # Minimum trials before median pruning
        min_trials = st.number_input(
            "Minimum Completed Trials",
            value=HYPERPARAMETER_REGISTRY["pruning_min_trials"]["default"],
            min_value=5,
            max_value=100,
            help="Wait for this many completed trials before using median pruning",
        )

        # Median factor
        median_factor = st.slider(
            "Median Factor",
            value=HYPERPARAMETER_REGISTRY["pruning_median_factor"]["default"],
            min_value=1.1,
            max_value=5.0,
            step=0.1,
            help="Prune trials with RMSE worse than (median × factor)",
        )

        # Show example
        st.info(
            f"Example: If median RMSE = 100, prune trials with RMSE > {100 * median_factor:.1f}"
        )

    with col2:
        st.subheader("Absolute Threshold Pruning")

        # RMSE factor
        rmse_factor = st.slider(
            "RMSE Threshold Factor",
            value=HYPERPARAMETER_REGISTRY["pruning_absolute_rmse_factor"]["default"],
            min_value=1.1,
            max_value=5.0,
            step=0.1,
            help="Prune trials with RMSE > (threshold × factor)",
        )

        # MAPE factor
        mape_factor = st.slider(
            "MAPE Threshold Factor",
            value=HYPERPARAMETER_REGISTRY["pruning_absolute_mape_factor"]["default"],
            min_value=1.1,
            max_value=10.0,
            step=0.1,
            help="Prune trials with MAPE > (threshold × factor)",
        )

        st.info(
            f"""
        Examples:
        - Prune if RMSE > {RMSE_THRESHOLD * rmse_factor:.1f}
        - Prune if MAPE > {MAPE_THRESHOLD * mape_factor:.1f}%
        """
        )

    # Save button
    if st.button("Save Pruning Settings"):
        # Update registry
        HYPERPARAMETER_REGISTRY["pruning_enabled"]["default"] = pruning_enabled
        HYPERPARAMETER_REGISTRY["pruning_min_trials"]["default"] = min_trials
        HYPERPARAMETER_REGISTRY["pruning_median_factor"]["default"] = median_factor
        HYPERPARAMETER_REGISTRY["pruning_absolute_rmse_factor"]["default"] = rmse_factor
        HYPERPARAMETER_REGISTRY["pruning_absolute_mape_factor"]["default"] = mape_factor

        # Update global variables
        global PRUNING_ENABLED, PRUNING_MIN_TRIALS, PRUNING_MEDIAN_FACTOR
        global PRUNING_ABSOLUTE_RMSE_FACTOR, PRUNING_ABSOLUTE_MAPE_FACTOR

        PRUNING_ENABLED = pruning_enabled
        PRUNING_MIN_TRIALS = min_trials
        PRUNING_MEDIAN_FACTOR = median_factor
        PRUNING_ABSOLUTE_RMSE_FACTOR = rmse_factor
        PRUNING_ABSOLUTE_MAPE_FACTOR = mape_factor

        st.success("Pruning settings saved!")

        # You might also want to save these to a config file for persistence


# Add this function to create UI for defining parameter search spaces for Optuna
def create_optuna_parameter_ui():
    """Create UI for defining parameter search spaces for Optuna"""
    st.header("Optuna Parameter Search Spaces")

    # Get parameter registry organized by groups
    param_groups = {}
    for name, config in HYPERPARAMETER_REGISTRY.items():
        group = config.get("group", "misc")
        if group not in param_groups:
            param_groups[group] = []
        param_groups[group].append((name, config))

    # Create tabs for different parameter groups
    tabs = st.tabs(list(param_groups.keys()))

    for i, (group, params) in enumerate(param_groups.items()):
        with tabs[i]:
            st.subheader(f"{group.capitalize()} Parameters")

            for name, config in params:
                # UI for parameter search space
                st.write(f"#### {name}")

                param_type = config["type"]

                # Different UI based on parameter type
                if param_type == "int":
                    col1, col2 = st.columns(2)
                    with col1:
                        min_val = st.number_input(
                            f"Min value for {name}",
                            value=config["range"][0] if config["range"] else 1,
                        )
                    with col2:
                        max_val = st.number_input(
                            f"Max value for {name}",
                            value=config["range"][1] if config["range"] else 100,
                        )

                    # Update the search space in the registry
                    if st.button(f"Update range for {name}"):
                        HYPERPARAMETER_REGISTRY[name]["range"] = [min_val, max_val]
                        st.success(f"Updated search space for {name}")

                elif param_type == "float":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        min_val = st.number_input(
                            f"Min value for {name}",
                            value=config["range"][0] if config["range"] else 0.0,
                            format="%.6f",
                        )
                    with col2:
                        max_val = st.number_input(
                            f"Max value for {name}",
                            value=config["range"][1] if config["range"] else 1.0,
                            format="%.6f",
                        )
                    with col3:
                        log_scale = st.checkbox(
                            f"Log scale for {name}",
                            value=config.get("log_scale", False),
                        )

                    # Update the search space
                    if st.button(f"Update range for {name}"):
                        HYPERPARAMETER_REGISTRY[name]["range"] = [min_val, max_val]
                        HYPERPARAMETER_REGISTRY[name]["log_scale"] = log_scale
                        st.success(f"Updated search space for {name}")

                elif param_type == "categorical":
                    current_options = config["range"] if config["range"] else []
                    options_str = st.text_input(
                        f"Options for {name} (comma-separated)",
                        value=",".join(map(str, current_options)),
                    )

                    # Parse options based on the parameter's default value type
                    if st.button(f"Update options for {name}"):
                        try:
                            if isinstance(config["default"], bool):
                                # For boolean parameters
                                options = [
                                    opt.strip().lower() == "true"
                                    for opt in options_str.split(",")
                                ]
                            elif isinstance(config["default"], int):
                                # For integer parameters
                                options = [
                                    int(opt.strip()) for opt in options_str.split(",")
                                ]
                            elif isinstance(config["default"], float):
                                # For float parameters
                                options = [
                                    float(opt.strip()) for opt in options_str.split(",")
                                ]
                            else:
                                # For string parameters
                                options = [
                                    opt.strip() for opt in options_str.split(",")
                                ]

                            HYPERPARAMETER_REGISTRY[name]["range"] = options
                            st.success(f"Updated options for {name}")
                        except Exception as e:
                            st.error(f"Error parsing options: {e}")

                st.divider()


# Add this function to create UI for running Optuna trials and viewing results
def create_optuna_run_ui():
    """Create UI for running Optuna trials and viewing results"""
    st.header("Run Optuna Optimization")

    # Get active models
    st.session_state.get("active_model_types", MODEL_TYPES)

    # Optimization settings
    col1, col2 = st.columns(2)

    with col1:
        n_trials = st.number_input("Number of trials", min_value=10, value=100, step=10)
        study_name = st.text_input(
            "Study name", value=f"{datetime.now().strftime('%Y%m%d%H%M')}_ensemble"
        )

    with col2:
        # Option to use pre-defined preset
        preset = st.selectbox(
            "Use preset configuration",
            ["custom"] + list(PRESET_CONFIGS.keys()),
            format_func=lambda x: (
                "Custom" if x == "custom" else PRESET_CONFIGS[x]["name"]
            ),
        )

        if preset != "custom":
            n_trials = PRESET_CONFIGS[preset]["n_trials"]
            st.info(f"Using {n_trials} trials based on preset")

            # Apply preset configurations to parameter ranges
            if st.button("Apply preset ranges"):
                apply_preset_to_ranges(preset)
                st.success(f"Applied {PRESET_CONFIGS[preset]['name']} preset ranges")

    # Ticker and timeframe selection
    ticker = st.selectbox(
        "Ticker", options=st.session_state.get("available_tickers", ["ETH-USD"])
    )
    timeframe = st.selectbox(
        "Timeframe", options=st.session_state.get("available_timeframes", ["1d"])
    )

    # Replace the start button with a message directing to the control panel
    st.info(
        """
    ### ⚠️ Important
    To start tuning, please use the **Start Tuning** button in the Control Panel sidebar.
    This ensures proper cycle tracking and consistent user experience.
    """
    )

    # Display current config that will be used
    st.subheader("Current Configuration Summary")
    st.write(f"**Ticker:** {ticker}")
    st.write(f"**Timeframe:** {timeframe}")
    st.write(f"**Trials:** {n_trials}")
    st.write(f"**Preset:** {preset if preset != 'custom' else 'Custom configuration'}")


# Fix potential unused variables
def render_parameter_importance_chart(study, top_n=10):
    """
    Render a chart showing parameter importance.
    
    Args:
        study: Optuna study
        top_n: Number of parameters to display
    """
    try:
        # Only proceed if we have completed trials
        if not study.trials:
            st.info("No trials available to analyze parameter importance")
            return
            
        # Calculate parameter importance
        importances = optuna.importance.get_param_importances(study)
        
        # Convert to DataFrame for plotting
        importance_df = pd.DataFrame({
            'Parameter': list(importances.keys())[:top_n],
            'Importance': list(importances.values())[:top_n]
        })
        
        if importance_df.empty:
            st.info("No parameter importance data available")
            return
            
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create chart
        chart = alt.Chart(importance_df).mark_bar().encode(
            x=alt.X('Importance:Q'),
            y=alt.Y('Parameter:N', sort='-x'),
            tooltip=['Parameter', 'Importance']
        ).properties(
            title='Parameter Importance',
            height=min(400, len(importance_df) * 30)  # Adjust height based on number of parameters
        )
        
        st.altair_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering parameter importance: {e}")
