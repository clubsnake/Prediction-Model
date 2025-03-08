"""
Enhanced UI components for the Streamlit dashboard.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dashboard.dashboard.dashboard_error import robust_error_boundary, load_latest_progress
from config.config_loader import TICKERS, TIMEFRAMES, TICKER, N_STARTUP_TRIALS
from config.logger_config import logger

# Import dashboard components
from src.dashboard.dashboard.dashboard_visualization import (
    plot_price_history_with_predictions,
    plot_feature_importance,
    plot_model_performance,
    generate_correlation_heatmap,
    create_interactive_price_chart # Make sure this is imported
)
from src.dashboard.dashboard.dashboard_data import calculate_indicators
from src.dashboard.prediction_service import generate_predictions

# Update N_STARTUP_TRIALS to 10000
N_STARTUP_TRIALS = 10000

@robust_error_boundary
def create_header():
    """Create a visually appealing header section with app branding"""
    # Use columns for better layout
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <h1 style="margin: 0; padding: 0;">
                <span style="color: #1E88E5;">📈</span> AI Price Prediction Dashboard
            </h1>
        </div>
        <p style="font-size: 1.1em; color: #455a64;">
            Advanced machine learning for financial market prediction and analysis
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        # Status indicator with dynamic styling - adjust vertical position
        if st.session_state.get("tuning_in_progress", False):
            st.markdown("""
            <div style="background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50; 
                        padding: 10px; border-radius: 4px; margin-top: 20px;">
                <span style="font-weight: bold; color: #4CAF50;">🔄 Tuning Active</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: rgba(30, 136, 229, 0.1); border-left: 4px solid #1E88E5; 
                        padding: 10px; border-radius: 4px; margin-top: 20px;">
                <span style="font-weight: bold; color: #1E88E5;">⏸️ Tuning Inactive</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Last updated timestamp with refresh button
        st.markdown(f"""
        <div style="text-align: right; color: #78909c; font-size: 0.9em;">
            Last updated:<br/>{datetime.now().strftime('%H:%M:%S')}
            <br><br>
            <button onclick="Streamlit.rerunScript()" style="background-color: #4CAF50; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer;">
                🔄 Refresh
            </button>
        </div>
        """, unsafe_allow_html=True)
        
    
    # Add a horizontal line for visual separation
    st.markdown("<hr style='margin: 0.5rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)


@robust_error_boundary
def create_control_panel():
    """Create an enhanced control panel for user inputs with better organization"""
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <h2 style="color: #1E88E5;">Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Add tuning controls at the top
    if not st.session_state.get("tuning_in_progress", False):
        # If not currently tuning, show a Start button
        if st.sidebar.button("🚀 Start Hyperparameter Tuning", key="btn_start_tuning", use_container_width=True):
            from dashboard_model import start_tuning
            start_tuning(st.session_state.get("selected_ticker", "BTC-USD"), 
                        st.session_state.get("selected_timeframe", "1d"))
            # Reset best_metrics since a new tuning session starts
            st.session_state["best_metrics"] = {}
    else:
        # If tuning is ongoing, give option to stop
        if st.sidebar.button("⏹️ Stop Tuning", key="btn_stop_tuning", use_container_width=True):
            from dashboard_model import stop_tuning
            stop_tuning()
            
    # Add some space
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Create sections in sidebar for better organization
    st.sidebar.markdown("### 📊 Data Selection")
    
    # Add custom ticker input option
    use_custom_ticker = st.sidebar.checkbox("Use custom ticker", 
                                           key="cb_use_custom_ticker",
                                           value=st.session_state.get("use_custom_ticker", False))
    
    if use_custom_ticker:
        # Text input for custom ticker
        ticker = st.sidebar.text_input(
            "Enter ticker symbol:",
            key="input_custom_ticker",
            value=st.session_state.get("custom_ticker", ""),
            help="Example: AAPL, MSFT, BTC-USD, ETH-USD"
        )
        if not ticker:  # If empty, use default
            ticker = TICKER
    else:
        # Standard dropdown selection
        ticker = st.sidebar.selectbox(
            "Select ticker:",
            key="select_ticker",
            options=TICKERS,
            index=TICKERS.index(st.session_state.get("selected_ticker", TICKER))
            if st.session_state.get("selected_ticker", TICKER) in TICKERS
            else 0,
        )
    
    # Store the custom ticker in session state
    if use_custom_ticker:
        st.session_state["custom_ticker"] = ticker
    st.session_state["use_custom_ticker"] = use_custom_ticker
    st.session_state["selected_ticker"] = ticker

    # Select timeframe with default selection handling
    selected_timeframe_index = 0
    if "selected_timeframe" in st.session_state and st.session_state["selected_timeframe"] in TIMEFRAMES:
        selected_timeframe_index = TIMEFRAMES.index(st.session_state["selected_timeframe"])
    
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        TIMEFRAMES,
        key="select_timeframe",
        index=selected_timeframe_index,
        help="Choose data frequency/interval"
    )
    st.session_state["selected_timeframe"] = timeframe

    # Date range section with better organization
    st.sidebar.markdown("### 📅 Date Range")
    
    # Ask for start_date and end_date without automatic recalculation
    default_start = datetime.now().date() - timedelta(days=30)
    start_date = st.sidebar.date_input(
        "Start Date",
        key="input_start_date",
        value=st.session_state.get("start_date_user", default_start),
        help="Starting date for visualization"
    )
    # Only store in session state if user made a change
    if st.session_state.get("start_date_user") != start_date:
        st.session_state["start_date_user"] = start_date
    
    # Make end_date completely independent
    default_end = datetime.now().date() + timedelta(days=30)
    end_date = st.sidebar.date_input(
        "Forecast End Date",
        key="input_end_date",
        value=st.session_state.get("end_date_user", default_end),
        help="End date for forecast visualization (future date for predictions)"
    )
    # Only store in session state if user made a change
    if st.session_state.get("end_date_user") != end_date:
        st.session_state["end_date_user"] = end_date
    
    # Training settings section
    st.sidebar.markdown("### 🧠 Model Training Settings")
    
    default_training_start = datetime.now().date() - timedelta(days=365*5)
    training_start_date = st.sidebar.date_input(
        "Training Start Date",
        key="input_training_start_date",
        value=st.session_state.get("training_start_date_user", default_training_start),
        help="Starting date for training data (earlier means more data)"
    )
    # Only store in session state if user made a change
    if st.session_state.get("training_start_date_user") != training_start_date:
        st.session_state["training_start_date_user"] = training_start_date
    
    # Advanced settings in an expander
    with st.sidebar.expander("Advanced Settings", expanded=False):
        # Calculate windows but make them independent of each other
        current_date = datetime.now().date()
        
        historical_window = (current_date - start_date).days
        forecast_window = (end_date - current_date).days

    # Add indicator selection at the top of chart settings
    st.sidebar.markdown("### 📊 Chart Settings")
    
    # Custom indicators first
    st.sidebar.write("**Custom Indicators:**")
    show_werpi = st.sidebar.checkbox("WERPI", key="cb_show_werpi", value=False, 
                                   help="Wavelet-based Encoded Relative Price Indicator")
    show_vmli = st.sidebar.checkbox("VMLI", key="cb_show_vmli", value=False, 
                                  help="Volatility-Momentum-Liquidity Indicator")
    
    # Standard indicators - all off by default
    st.sidebar.write("**Technical Indicators:**")
    show_ma = st.sidebar.checkbox("Moving Averages", key="cb_show_ma", value=False)
    show_bb = st.sidebar.checkbox("Bollinger Bands", key="cb_show_bb", value=False)
    show_rsi = st.sidebar.checkbox("RSI", key="cb_show_rsi", value=False)
    show_macd = st.sidebar.checkbox("MACD", key="cb_show_macd", value=False)
    
    # Forecast options
    st.sidebar.write("**Forecast Options:**")
    show_forecast = st.sidebar.checkbox("Show Forecast", key="cb_show_forecast", value=True)
    
    # Auto-refresh settings moved here
    st.sidebar.write("**Auto-Refresh Settings:**")
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        key="cb_auto_refresh",
        value=st.session_state.get("auto_refresh", True),
        help="Automatically refresh the dashboard"
    )
    st.session_state["auto_refresh"] = auto_refresh

    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            key="slider_refresh_interval",
            min_value=10,
            max_value=300,
            value=st.session_state.get("refresh_interval", 30),
            help="How often to refresh the dashboard"
        )
        st.session_state["refresh_interval"] = refresh_interval
        
        # Show a countdown timer if auto-refresh is enabled
        if "last_refresh" in st.session_state:
            time_since_refresh = int(datetime.now().timestamp() - st.session_state["last_refresh"])
            time_to_next_refresh = max(0, refresh_interval - time_since_refresh)
            
            # Show progress bar for refresh timer
            refresh_progress = 1 - (time_to_next_refresh / refresh_interval)
            st.sidebar.progress(min(1.0, max(0.0, refresh_progress)))
            st.sidebar.text(f"Next refresh in {time_to_next_refresh} seconds")
    
    # Store indicator preferences in session state
    indicators = {
        "show_ma": show_ma,
        "show_bb": show_bb,
        "show_rsi": show_rsi,
        "show_macd": show_macd,
        "show_werpi": show_werpi,
        "show_vmli": show_vmli,
        "show_forecast": show_forecast
    }
    st.session_state["indicators"] = indicators
    
    # Add some helpful information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.85em; color: #78909c;">
        <strong>Tips:</strong>
        <ul>
            <li>Choose a longer historical window for more context</li>
            <li>Auto-refresh keeps predictions and metrics updated</li>
            <li>Training with more data gives better results but takes longer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Removed "Active Model Types" section from sidebar
    
    # Add hyperparameter tuning link at the bottom of the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <a href="#hyperparameter-tuning" style="background-color: #1E88E5; color: white; 
           padding: 8px 16px; border-radius: 4px; text-decoration: none; display: block;">
           ⚙️ Hyperparameter Tuning Settings
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Return selected parameters with auto-calculated values and indicators
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "training_start_date": training_start_date,
        "historical_window": historical_window,
        "forecast_window": forecast_window,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "indicators": indicators
    }


@robust_error_boundary
def create_hyperparameter_tuning_panel():
    """Create a panel for controlling hyperparameter tuning"""
    
    st.markdown("""
    <div style="text-align: center;">
        <h2 style="color: #1E88E5;">Hyperparameter Tuning</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different tuning sections
    tab1, tab2, tab3, tab4 = st.tabs(["Tuning Control", "Model Parameters", "Advanced Settings", "Validation Metrics"])
    
    with tab1:
        # Load tuning mode settings from config
        from config.config_loader import get_value
        tuning_modes = get_value("hyperparameter.tuning_modes", {
            "quick": {"trials_multiplier": 0.25, "epochs_multiplier": 0.3, "timeout_multiplier": 0.2},
            "normal": {"trials_multiplier": 1.0, "epochs_multiplier": 1.0, "timeout_multiplier": 1.0},
            "thorough": {"trials_multiplier": 3.0, "epochs_multiplier": 2.0, "timeout_multiplier": 4.0},
            "extreme": {"trials_multiplier": 10.0, "epochs_multiplier": 3.0, "timeout_multiplier": 15.0}
        })
        
        default_mode = get_value("hyperparameter.default_mode", "normal")
        
        # Tuning mode selection
        st.subheader("Tuning Mode")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            mode = st.selectbox(
                "Select Tuning Mode",
                options=list(tuning_modes.keys()),
                index=list(tuning_modes.keys()).index(default_mode) if default_mode in tuning_modes else 1,
                format_func=lambda x: x.capitalize(),
                help="Controls tuning intensity (trials, epochs, timeout)"
            )
        
        with col2:
            # Store the selected mode in session state
            if st.button("Apply Mode", use_container_width=True):
                st.session_state["tuning_mode"] = mode
                st.session_state["tuning_multipliers"] = tuning_modes[mode]
                st.success(f"Set to {mode.capitalize()} mode")
        
        # Show current multipliers
        current_mode = st.session_state.get("tuning_mode", default_mode)
        current_multipliers = tuning_modes.get(current_mode, tuning_modes[default_mode])
        
        st.markdown(f"**Current Mode:** {current_mode.capitalize()}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trials", f"{current_multipliers['trials_multiplier']}x")
        col2.metric("Epochs", f"{current_multipliers['epochs_multiplier']}x")
        col3.metric("Timeout", f"{current_multipliers['timeout_multiplier']}x")
        col4.metric("Complexity", f"{current_multipliers.get('complexity_multiplier', 1.0)}x")
        
        # Tuning thresholds
        st.subheader("Performance Thresholds")
        col1, col2 = st.columns(2)
        
        rmse_threshold = get_value("hyperparameter.thresholds.rmse", 5.0)
        mape_threshold = get_value("hyperparameter.thresholds.mape", 5.0)
        
        with col1:
            rmse_target = st.number_input("Target RMSE", 
                               min_value=0.1, 
                               max_value=100.0, 
                               value=float(rmse_threshold),
                               step=0.1,
                               help="Target RMSE threshold for stopping tuning")
        
        with col2:
            mape_target = st.number_input("Target MAPE (%)", 
                               min_value=0.1, 
                               max_value=50.0, 
                               value=float(mape_threshold),
                               step=0.1,
                               help="Target MAPE threshold for stopping tuning")
        
        # Save thresholds button
        if st.button("Save Thresholds", use_container_width=True):
            from config.config_loader import set_value
            set_value("hyperparameter.thresholds.rmse", float(rmse_target))
            set_value("hyperparameter.thresholds.mape", float(mape_target))
            st.success("Thresholds updated!")
        
        # Adaptive Parameters Section (NEW)
        st.subheader("Adaptive Parameters")
        
        use_adaptive_window = st.checkbox("Use Adaptive Window Size", 
                                         value=get_value("hyperparameter.adaptive.window_size", False),
                                         help="Automatically adjust window size based on market volatility")
        
        use_adaptive_threshold = st.checkbox("Use Adaptive Retraining Threshold", 
                                           value=get_value("hyperparameter.adaptive.threshold", False),
                                           help="Automatically adjust retraining threshold based on performance and volatility")
        
        # Save adaptive settings
        if st.button("Save Adaptive Settings", use_container_width=True):
            from config.config_loader import set_value
            set_value("hyperparameter.adaptive.window_size", bool(use_adaptive_window))
            set_value("hyperparameter.adaptive.threshold", bool(use_adaptive_threshold))
            st.success("Adaptive settings updated!")
        
        # Tuning Status
        st.subheader("Tuning Status")
        
        if st.session_state.get("tuning_in_progress", False):
            status_color = "#4CAF50"  # Green
            status_text = "Active"
        else:
            status_color = "#F44336"  # Red
            status_text = "Inactive"
            
        st.markdown(f"""
        <div style="background-color: rgba({status_color.lstrip('#')}, 0.1); 
                    border-left: 4px solid {status_color}; 
                    padding: 10px; border-radius: 4px;">
            <span style="font-weight: bold; color: {status_color};">
                Tuning Status: {status_text}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.get("tuning_in_progress", False):
                if st.button("Start Tuning", use_container_width=True):
                    from dashboard_ui import start_tuning
                    # Pass the mode multipliers to the tuning function
                    start_tuning(
                        st.session_state.get("selected_ticker", "BTC-USD"), 
                        st.session_state.get("selected_timeframe", "1d"),
                        multipliers=current_multipliers
                    )
        
        with col2:
            if st.session_state.get("tuning_in_progress", False):
                if st.button("Stop Tuning", use_container_width=True):
                    from dashboard_model import stop_tuning
                    stop_tuning()
    
    with tab2:
        # Model parameters panel
        st.subheader("Model Type Selection")
        
        # Get model types from config
        from config.config_loader import MODEL_TYPES, ACTIVE_MODEL_TYPES
        
        # Add CNN to model types
        MODEL_TYPES.append("cnn")
        
        # Create checkboxes for each model type
        model_type_states = {}
        
        # Organize model types in columns
        cols = st.columns(2)
        for i, model_type in enumerate(MODEL_TYPES):
            col_idx = i % 2
            with cols[col_idx]:
                is_active = model_type in ACTIVE_MODEL_TYPES
                model_type_states[model_type] = st.checkbox(
                    f"{model_type.upper()}", 
                    value=is_active,
                    key=f"hp_model_{model_type}"
                )
        
        # Apply button for model types
        if st.button("Apply Model Selection", use_container_width=True):
            # Update session state for active model types
            active_models = [model for model, is_active in model_type_states.items() if is_active]
            
            if not active_models:
                st.error("At least one model type must be selected!")
            else:
                st.session_state["ACTIVE_MODEL_TYPES"] = active_models
                st.success(f"Updated active models: {', '.join(active_models)}")
        
        # Show model parameter details
        st.subheader("Model Parameters")
        
        # Create tabs for each model type
        model_tabs = st.tabs([m.upper() for m in MODEL_TYPES])
        
        for i, model_type in enumerate(MODEL_TYPES):
            with model_tabs[i]:
                if model_type == "lstm":
                    st.markdown("""
                    **LSTM Parameters:**
                    - Learning rate: 1e-5 to 1e-2 (log scale) - *The rate at which the model learns.*
                    - Units: 16 to 512 per layer - *Number of neurons in each LSTM layer.*
                    - Layers: 1 to 3 - *Number of LSTM layers.*
                    - Dropout: 0.0 to 0.5 - *Dropout rate for regularization.*
                    - Attention: Enabled/Disabled - *Whether to use attention mechanism.*
                    - Attention type: dot, multiplicative, additive - *Type of attention mechanism.*
                    - Batch normalization: Enabled/Disabled - *Whether to use batch normalization.*
                    """)
                elif model_type == "rnn":
                    st.markdown("""
                    **RNN Parameters:**
                    - Learning rate: 1e-5 to 1e-2 (log scale) - *The rate at which the model learns.*
                    - Units: 16 to 256 per layer - *Number of neurons in each RNN layer.*
                    - Layers: 1 to 2 - *Number of RNN layers.*
                    - Dropout: 0.0 to 0.5 - *Dropout rate for regularization.*
                    - Batch normalization: Enabled/Disabled - *Whether to use batch normalization.*
                    """)
                elif model_type == "random_forest":
                    st.markdown("""
                    **Random Forest Parameters:**
                    - n_estimators: 50 to 1000 - *Number of trees in the forest.*
                    - max_depth: 5 to 50 - *Maximum depth of the trees.*
                    - min_samples_split: 2 to 20 - *Minimum samples required to split an internal node.*
                    - min_samples_leaf: 1 to 20 - *Minimum samples required to be at a leaf node.*
                    - max_features: sqrt, log2, None - *Number of features to consider when looking for the best split.*
                    - bootstrap: True/False - *Whether bootstrap samples are used when building trees.*
                    - criterion: squared_error, absolute_error, poisson - *The function to measure the quality of a split.*
                    """)
                elif model_type == "xgboost":
                    st.markdown("""
                    **XGBoost Parameters:**
                    - n_estimators: 50 to 1000 - *Number of boosting rounds.*
                    - learning_rate: 0.001 to 0.5 (log scale) - *Boosting learning rate.*
                    - max_depth: 3 to 15 - *Maximum depth of a tree.*
                    - subsample: 0.5 to 1.0 - *Subsample ratio of the training instance.*
                    - colsample_bytree: 0.5 to 1.0 - *Subsample ratio of columns when constructing each tree.*
                    - gamma: 0 to 5 - *Minimum loss reduction required to make a further partition on a leaf node.*
                    - min_child_weight: 1 to 10 - *Minimum sum of instance weight (hessian) needed in a child.*
                    - objective: reg:squarederror, reg:absoluteerror - *The learning objective.*
                    """)
                elif model_type == "tft":
                    st.markdown("""
                    **Temporal Fusion Transformer Parameters:**
                    - Learning rate: 1e-5 to 1e-2 (log scale) - *The rate at which the model learns.*
                    - Hidden size: 32 to 512 - *Size of hidden layers.*
                    - LSTM units: 32 to 512 - *Number of LSTM units.*
                    - Number of heads: 1 to 8 - *Number of attention heads.*
                    - Dropout: 0.0 to 0.5 - *Dropout rate for regularization.*
                    """)
                elif model_type == "ltc":
                    st.markdown("""
                    **Liquid Time Constant Parameters:**
                    - Learning rate: 1e-5 to 1e-2 (log scale) - *The rate at which the model learns.*
                    - Units: 32 to 512 - *Number of LTC units.*
                    - Timescales: min/max optimization - *Optimization range for timescales.*
                    - Hidden size: 32 to 512 - *Size of hidden layers.*
                    """)
                elif model_type == "tabnet":
                    st.markdown("""
                    **TabNet Parameters:**
                    - n_d (width dimension): 8 to 256 - *Width of the attention embedding for each mask.*
                    - n_a (attention dimension): 8 to 256 - *Width of the attention embedding for each step.*
                    - n_steps (steps in feature selection): 1 to 15 - *Number of steps in the feature selection process.*
                    - gamma (feature selection regularization): 0.5 to 3.0 - *Coefficient for feature selection regularization.*
                    - lambda_sparse (sparsity regularization): 1e-7 to 1e-1 - *Sparsity regularization coefficient.*
                    - Learning rate: 1e-5 to 0.5 - *The rate at which the model learns.*
                    - Batch size: 128 to 4096 - *Number of samples processed in each batch.*
                    - Virtual batch size: 16 to 1024 - *Batch size used for Ghost Batch Normalization.*
                    - Momentum: 0.005 to 0.5 - *Momentum for batch normalization.*
                    - Max epochs: 50 to 500 - *Maximum number of training epochs.*
                    - Patience: 5 to 50 - *Number of epochs with no improvement after which training will stop.*
                    """)
                elif model_type == "cnn":
                    st.markdown("""
                    **CNN Parameters:**
                    - Learning rate: 1e-5 to 1e-2 (log scale) - *The rate at which the model learns.*
                    - Filters: 16 to 512 per layer - *Number of convolutional filters.*
                    - Kernel size: 2 to 5 - *Size of the convolutional kernel.*
                    - Layers: 1 to 5 - *Number of convolutional layers.*
                    - Dropout: 0.0 to 0.5 - *Dropout rate for regularization.*
                    - Batch normalization: Enabled/Disabled - *Whether to use batch normalization.*
                    """)
    
    with tab3:
        # Advanced tuning settings
        st.subheader("Advanced Tuning Settings")
        
        # Pruning settings
        st.write("**Pruning Settings**")
        
        from config.config_loader import PRUNING_ENABLED, PRUNING_MEDIAN_FACTOR, PRUNING_MIN_TRIALS
        
        pruning_enabled = st.checkbox("Enable Pruning", value=PRUNING_ENABLED)
        pruning_median_factor = st.slider("Pruning Median Factor", 1.0, 5.0, float(PRUNING_MEDIAN_FACTOR), 0.1)
        pruning_min_trials = st.slider("Minimum Trials Before Pruning", 3, 50, int(PRUNING_MIN_TRIALS))
        
        # Optuna specific settings
        st.write("**Optuna Settings**")
        
        from config.config_loader import N_STARTUP_TRIALS
        
        n_startup_trials = st.slider("Number of Startup Trials", 5, 10000, int(N_STARTUP_TRIALS))
        
        # Save advanced settings
        if st.button("Save Advanced Settings", use_container_width=True):
            from config.config_loader import set_value
            
            set_value("hyperparameter.pruning.enabled", bool(pruning_enabled), target="system")
            set_value("hyperparameter.pruning.median_factor", float(pruning_median_factor), target="system")
            set_value("hyperparameter.pruning.min_trials", int(pruning_min_trials), target="system")
    
    with tab4:
        # Validation Metrics Panel (NEW)
        st.subheader("Validation Metrics")
        
        # Get recent trial metrics if available
        if "tuning_history" in st.session_state and st.session_state["tuning_history"]:
            history = st.session_state["tuning_history"]
            
            # Extract metrics from history
            trials = [h.get("trial_number", i+1) for i, h in enumerate(history)]
            rmse_values = [h.get("metrics", {}).get("rmse", None) for h in history]
            mape_values = [h.get("metrics", {}).get("mape", None) for h in history]
            mae_values = [h.get("metrics", {}).get("mae", None) for h in history]
            
            # Filter out None values
            valid_data = [(t, r, m, a) for t, r, m, a in zip(trials, rmse_values, mape_values, mae_values) 
                         if r is not None and m is not None and a is not None]
            
            if valid_data:
                # Unpack valid data
                trials_filtered, rmse_filtered, mape_filtered, mae_filtered = zip(*valid_data)
                
                # Create DataFrame for metrics
                import pandas as pd
                metrics_df = pd.DataFrame({
                    "Trial": trials_filtered,
                    "RMSE": rmse_filtered,
                    "MAPE (%)": mape_filtered,
                    "MAE": mae_filtered
                })
                
                # Display metrics table
                st.dataframe(metrics_df, use_container_width=True)
                
                # Display metrics chart
                st.subheader("RMSE Trend")
                
                import plotly.express as px
                fig = px.line(metrics_df, x="Trial", y="RMSE", title="RMSE across Trials")
                st.plotly_chart(fig, use_container_width=True)
                
                # Best model metrics
                best_idx = rmse_filtered.index(min(rmse_filtered))
                
                st.subheader("Best Model Metrics")
                cols = st.columns(3)
                cols[0].metric("Best RMSE", f"{rmse_filtered[best_idx]:.4f}")
                cols[1].metric("Best MAPE", f"{mape_filtered[best_idx]:.2f}%")
                cols[2].metric("Best MAE", f"{mae_filtered[best_idx]:.4f}")
            else:
                st.info("No validation metrics available yet.")
        else:
            st.info("No validation metrics available yet. Start tuning to collect metrics.")

@robust_error_boundary
def create_metrics_cards():
    """Create a row of key metrics cards with improved styling and visual indicators"""
    # Get the ticker and timeframe from session state
    current_ticker = st.session_state.get("selected_ticker", "Unknown")
    current_timeframe = st.session_state.get("selected_timeframe", "Unknown")
    
    # Get the latest progress for the specific ticker and timeframe
    try:
        # Try to get ticker and timeframe specific progress
        progress = load_latest_progress(ticker=current_ticker, timeframe=current_timeframe)
        if not progress:  # If no specific progress found
            progress = load_latest_progress()  # Fall back to general progress
    except:
        progress = load_latest_progress()  # Fall back if any error
        
    current_trial = progress.get("current_trial", 0)
    total_trials = progress.get("total_trials", 1)  # Use 1 as default to avoid division by zero
    current_rmse = progress.get("current_rmse", None)
    current_mape = progress.get("current_mape", None)
    cycle = progress.get("cycle", 1)
    
    # Display ticker and timeframe info
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 10px;">
        <span style="background-color: rgba(33, 150, 243, 0.1); padding: 5px 10px; border-radius: 4px; color: #2196F3; font-weight: bold;">
            {current_ticker} / {current_timeframe}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Retrieve best metrics from session (or infinities if not set)
    best_rmse = st.session_state.get("best_metrics", {}).get("rmse", float("inf"))
    best_mape = st.session_state.get("best_metrics", {}).get("mape", float("inf"))

    # Ensure best metrics are numeric
    import numpy as np
    if not isinstance(best_rmse, (int, float)) or np.isnan(best_rmse):
        best_rmse = float("inf")
    if not isinstance(best_mape, (int, float)) or np.isnan(best_mape):
        best_mape = float("inf")

    # Compute improvement deltas if current metrics exist
    rmse_delta = None if current_rmse is None else best_rmse - current_rmse
    mape_delta = None if current_mape is None else best_mape - current_mape

    # Handle any NaN or inf in deltas
    if rmse_delta is not None and (np.isnan(rmse_delta) or np.isinf(rmse_delta)):
        rmse_delta = None
    if mape_delta is not None and (np.isnan(mape_delta) or np.isinf(mape_delta)):
        mape_delta = None

    # Create a 5-column layout for metrics with improved styling
    st.markdown('<div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">', unsafe_allow_html=True)
    
    # Column 1: Cycle indicator - Fix string formatting
    cycle_html = f'''
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">CURRENT CYCLE</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{cycle}</h2>
        <div style="height: 4px; background: #f0f0f0; width: 100%; border-radius: 2px;">
            <div style="height: 100%; width: {min(100, cycle*10)}%; background: #4CAF50; border-radius: 2px;"></div>
        </div>
    </div>
    '''
    st.markdown(cycle_html, unsafe_allow_html=True)
    
    # Column 2: Trial Progress - Fix string formatting
    progress_pct = int((current_trial / total_trials) * 100) if total_trials else 0
    progress_html = f'''
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">TRIAL PROGRESS</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{current_trial}/{total_trials}</h2>
        <div style="height: 4px; background: #f0f0f0; width: 100%; border-radius: 2px;">
            <div style="height: 100%; width: {progress_pct}%; background: #2196F3; border-radius: 2px;"></div>
        </div>
    </div>
    '''
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Column 3: Current RMSE - Fix string formatting
    rmse_color = "#4CAF50" if rmse_delta and rmse_delta > 0 else "#F44336"
    rmse_arrow = "↓" if rmse_delta and rmse_delta > 0 else "↑"
    rmse_delta_display = f"{rmse_arrow} {abs(rmse_delta):.2f}" if rmse_delta else ""
    
    # Fix null formatting issue with conditional display
    rmse_display = f"{current_rmse:.2f}" if current_rmse is not None else "N/A"
    
    rmse_html = f'''
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">CURRENT RMSE</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{rmse_display}</h2>
        <p style="font-size: 0.9em; margin: 0; color: {rmse_color};">{rmse_delta_display}</p>
    </div>
    '''
    st.markdown(rmse_html, unsafe_allow_html=True)
    
    # Column 4: Current MAPE - Fix string formatting
    mape_color = "#4CAF50" if mape_delta and mape_delta > 0 else "#F44336"
    mape_arrow = "↓" if mape_delta and mape_delta > 0 else "↑"
    mape_delta_display = f"{mape_arrow} {abs(mape_delta):.2f}%" if mape_delta else ""
    
    # Fix null formatting issue with conditional display
    mape_display = f"{current_mape:.2f}%" if current_mape is not None else "N/A"
    
    mape_html = f'''
    <div class="metric-container" style="flex: 1; min-width: 150px;">
        <p style="font-size: 0.8em; margin: 0; color: #78909c;">CURRENT MAPE</p>
        <h2 style="font-size: 1.8em; margin: 5px 0;">{mape_display}</h2>
        <p style="font-size: 0.9em; margin: 0; color: {mape_color};">{mape_delta_display}</p>
    </div>
    '''
    st.markdown(mape_html, unsafe_allow_html=True)

    # Column 5: Direction Accuracy - Fix string formatting
    if "ensemble_predictions_log" in st.session_state and st.session_state["ensemble_predictions_log"]:
        predictions = st.session_state["ensemble_predictions_log"]
        success_rate = 0
        
        if len(predictions) > 1:
            try:
                # Calculate direction accuracy
                correct_direction = 0
                for i in range(1, len(predictions)):
                    actual_direction = predictions[i]["actual"] > predictions[i - 1]["actual"]
                    pred_direction = predictions[i]["predicted"] > predictions[i - 1]["predicted"]
                    if actual_direction == pred_direction:
                        correct_direction += 1

                success_rate = (correct_direction / (len(predictions) - 1)) * 100
            except (KeyError, TypeError) as e:
                logger.error(f"Error calculating direction accuracy: {e}")
                success_rate = 0
        
        # Color based on accuracy
        accuracy_color = "#4CAF50" if success_rate >= 60 else "#FFC107" if success_rate >= 50 else "#F44336"
        
        accuracy_html = f'''
        <div class="metric-container" style="flex: 1; min-width: 150px;">
            <p style="font-size: 0.8em; margin: 0; color: #78909c;">DIRECTION ACCURACY</p>
            <h2 style="font-size: 1.8em; margin: 5px 0; color: {accuracy_color};">{success_rate:.1f}%</h2>
            <p style="font-size: 0.9em; margin: 0; color: #78909c;">{len(predictions)} predictions</p>
        </div>
        '''
        st.markdown(accuracy_html, unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="metric-container" style="flex: 1; min-width: 150px;">
            <p style="font-size: 0.8em; margin: 0; color: #78909c;">DIRECTION ACCURACY</p>
            <h2 style="font-size: 1.8em; margin: 5px 0;">N/A</h2>
            <p style="font-size: 0.9em; margin: 0; color: #78909c;">No predictions yet</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

"""
Main UI component for the dashboard.
Provides the layout and UI elements for the prediction model dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import logging

# Add project directory to path for imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import dashboard components
from src.dashboard.dashboard.dashboard_visualization import (
    plot_price_history_with_predictions,
    plot_feature_importance,
    plot_model_performance,
    generate_correlation_heatmap
)

# Import training resource optimizer dashboard
from src.dashboard.training_resource_optimizer_dashboard import render_training_optimizer_tab

# Import other pages/tabs
from src.dashboard.pattern_discovery.pattern_discovery_tab import add_pattern_discovery_tab
from src.dashboard.explainable_ai_tab import render_explainable_ai_tab  # Updated import
from src.dashboard.monitoring import PredictionMonitor

# Initialize logger
logger = logging.getLogger(__name__)

# Add imports for monitoring and prediction service
try:
    from src.dashboard.monitoring import PredictionMonitor
    from src.dashboard.prediction_service import PredictionService, update_dashboard_forecast
    HAS_PREDICTION_SERVICE = True
except ImportError:
    logger.warning("PredictionService not available")
    HAS_PREDICTION_SERVICE = False

# Import centralized state management
from src.dashboard.dashboard.dashboard_state import (
    get_state, set_state,
    get_prediction_monitor, get_prediction_service,
    get_ensemble_weights, update_ensemble_weights,
    get_current_ticker, get_current_timeframe,
    get_lookback_period, get_forecast_window
)

def render_dashboard_header():
    """Render the dashboard header with title and description."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Stock Price Prediction Dashboard")
        st.markdown("""
        This dashboard provides real-time insights into stock price predictions using 
        an ensemble of machine learning models including LSTM, RNN, TFT, Random Forest, XGBoost, and TabNet.
        """)
    with col2:
        # Add logo or other visual element if available
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.write(f"Last updated: {current_time}")
        
        # Add refresh button
        if st.button("🔄 Refresh Data", key="refresh_dashboard"):
            st.session_state["refresh_requested"] = True
            st.experimental_rerun()

def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.title("Navigation")
    
    # Main page selection
    page_options = [
        "Dashboard", 
        "Pattern Discovery",
        "Explainable AI",
        "Model Performance",
        "Training Optimizer",
        "Settings"
    ]
    
    selected_page = st.sidebar.radio("Select Page", page_options)
    
    # Ticker selection
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
    selected_ticker = st.sidebar.selectbox(
        "Select Stock Ticker",
        tickers,
        key="selected_ticker"
    )
    
    # Timeframe selection
    timeframes = ["1d", "1h", "15min", "5min"]
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        timeframes,
        key="selected_timeframe"
    )
    
    # Date range for analysis
    st.sidebar.subheader("Analysis Period")
    
    # Default to last 365 days
    default_start_date = datetime.now() - timedelta(days=365)
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        key="training_start_date"
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=datetime.now(),
        key="training_end_date"
    )
    
    # Forecast window
    forecast_days = st.sidebar.slider(
        "Forecast Window (Days)", 
        min_value=1, 
        max_value=90, 
        value=30,
        key="forecast_window"
    )
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    
    # Available model types
    model_types = ["lstm", "rnn", "tft", "random_forest", "xgboost", "tabnet", "cnn", "ltc"]
    
    # Create a container to display model weights
    with st.sidebar.expander("Ensemble Model Weights", expanded=False):
        # Get current weights from session state or initialize
        if "ensemble_weights" not in st.session_state:
            st.session_state["ensemble_weights"] = {
                model: 1.0/len(model_types) for model in model_types
            }
        
        # Allow user to adjust model weights
        new_weights = {}
        for model in model_types:
            current_weight = st.session_state["ensemble_weights"].get(model, 0.0)
            new_weights[model] = st.slider(
                f"{model.upper()} Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(current_weight),
                format="%.2f",
                key=f"weight_{model}"
            )
        
        # Normalize weights to sum to 1
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            normalized_weights = {k: v/weight_sum for k, v in new_weights.items()}
            st.session_state["ensemble_weights"] = normalized_weights
        else:
            st.warning("Total weight must be greater than 0")
    
    # Training button
    if st.sidebar.button("Train Models"):
        st.session_state["training_requested"] = True
        
    # Return the selected options
    return {
        "page": selected_page,
        "ticker": selected_ticker,
        "timeframe": selected_timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "forecast_window": forecast_days,
    }

def render_main_dashboard(options):
    """Render the main dashboard page with current predictions and charts."""
    
    # Check if data is available
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("No data available. Please fetch data first.")
        return
    
    df = st.session_state["df"]
    
    # Display tabs for different dashboard views
    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Technical Indicators", "Correlations", "Raw Data"])
    
    with tab1:
        st.subheader(f"{options['ticker']} Price Forecast")
        
        # Get past predictions from session state
        past_predictions = st.session_state.get("past_predictions", {})
        
        # Display price chart with predictions
        # USE THE CORRECT FUNCTION HERE
        # fig = plot_price_history_with_predictions(
        #     df, 
        #     future_predictions=st.session_state.get("future_forecast", []),
        #     ticker=options["ticker"],
        #     past_predictions=past_predictions
        # )
        
        # Calculate indicators and plot the chart
        df_indicators = calculate_indicators(df)
        
        # Initialize future_forecast as None before attempting to generate it
        future_forecast = None
        
        # Only attempt forecast if model exists and user wants to see it
        model = st.session_state.get('model')
        if model and indicators.get("show_forecast", True):
            with st.spinner("Generating forecast..."):
                # Determine feature columns (exclude date and target 'Close')
                feature_cols = [col for col in df.columns if col not in ["date", "Date", "Close"]]
                future_forecast = generate_forecast(model, df, feature_cols)
        
        # Get indicator preferences from options
        indicators = options.get("indicators", {})
        
        # Pass indicator options to visualization function
        create_interactive_price_chart(df_indicators, options, 
                                     future_forecast=future_forecast, 
                                     indicators=indicators,
                                     height=700)  # Increase height for better visualization
        
        # Display forecast metrics
        if "metrics" in st.session_state:
            metrics = st.session_state["metrics"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            with col2:
                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
            with col3:
                last_price = df["Close"].iloc[-1] if len(df) > 0 else 0
                last_pred = st.session_state.get("future_forecast", [0])[0]
                change = ((last_pred - last_price) / last_price) * 100 if last_price > 0 else 0
                st.metric(
                    "Next Day Forecast", 
                    f"${last_pred:.2f}", 
                    f"{change:.2f}%",
                    delta_color="normal" if change >= 0 else "inverse"
                )
        
        # Add a section to show past prediction accuracy
        if past_predictions:
            st.subheader("Past Prediction Accuracy")
            
            # Create a dataframe with past predictions
            past_pred_data = []
            for date_str, pred_info in past_predictions.items():
                if pred_info['actual'] is not None:
                    past_pred_data.append({
                        'Date': date_str,
                        'Predicted': f"${pred_info['predicted']:.2f}",
                        'Actual': f"${pred_info['actual']:.2f}",
                        'Error': f"${pred_info['error']:.2f}",
                        'Error (%)': f"{pred_info['pct_error']:.2f}%"
                    })
            
            if past_pred_data:
                past_df = pd.DataFrame(past_pred_data)
                st.dataframe(past_df.sort_values('Date', ascending=False).head(10), use_container_width=True)
                
                # Show accuracy metrics
                if len(past_pred_data) >= 3:  # Need at least a few data points
                    # Calculate accuracy metrics
                    pct_errors = [abs(p['pct_error']) for d, p in past_predictions.items() 
                                if p.get('pct_error') is not None]
                    
                    mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0
                    
                    # For direction accuracy, we need at least 2 consecutive predictions
                    direction_correct = sum(1 for d, p in past_predictions.items() 
                                         if p.get('actual') is not None and p.get('predicted') is not None and 
                                         len(df) > 2 and 
                                         ((p['actual'] > df['Close'].iloc[-2] and p['predicted'] > df['Close'].iloc[-2]) or
                                          (p['actual'] < df['Close'].iloc[-2] and p['predicted'] < df['Close'].iloc[-2])))
                    
                    direction_accuracy = direction_correct / len([p for d, p in past_predictions.items() 
                                                               if p.get('actual') is not None]) * 100 if direction_correct > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Historical MAPE", f"{mape:.2f}%")
                    with col2:
                        st.metric("Direction Accuracy", f"{direction_accuracy:.1f}%")

def render_settings_page():
    """Render the settings page for configuring the dashboard."""
    st.header("Dashboard Settings")
    
    # Model training settings
    st.subheader("Model Training Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lookback window for sequence models
        lookback = st.slider(
            "Lookback Window (Days)",
            min_value=5,
            max_value=120,
            value=st.session_state.get("lookback", 30),
            key="lookback"
        )
        
        # Batch size for neural models
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128, 256, 512],
            value=st.session_state.get("batch_size", 32),
            key="batch_size"
        )
    
    with col2:
        # Walk-forward window size
        wf_size = st.slider(
            "Walk Forward Window",
            min_value=1,
            max_value=30,
            value=st.session_state.get("wf_size", 5),
            key="wf_size"
        )
        
        # Learning rate for neural models
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            value=st.session_state.get("learning_rate", 0.001),
            format="%.4f",
            key="learning_rate"
        )
    
    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Epochs for training
            epochs = st.slider(
                "Training Epochs",
                min_value=1,
                max_value=50,
                value=st.session_state.get("epochs", 10),
                key="epochs"
            )
            
            # Dropout rate
            dropout = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.8,
                value=st.session_state.get("dropout", 0.2),
                step=0.05,
                format="%.2f",
                key="dropout"
            )
            
        with col2:
            # Early stopping patience
            patience = st.slider(
                "Early Stopping Patience",
                min_value=1,
                max_value=20,
                value=st.session_state.get("patience", 5),
                key="patience"
            )
            
            # Loss function selection
            loss_function = st.selectbox(
                "Loss Function",
                ["mean_squared_error", "mean_absolute_error", "huber_loss", "log_cosh"],
                index=0,
                key="loss_function"
            )
    
    # Data preprocessing settings
    st.subheader("Data Preprocessing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scaling method
        scaling_method = st.selectbox(
            "Scaling Method",
            ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
            index=0,
            key="scaling_method"
        )
        
        # Handle missing data
        handle_missing = st.selectbox(
            "Handle Missing Data",
            ["Forward Fill", "Backward Fill", "Mean", "Median", "Drop"],
            index=0,
            key="handle_missing"
        )
        
    with col2:
        # Outlier treatment
        outlier_treatment = st.selectbox(
            "Outlier Treatment",
            ["None", "Winsorize", "Remove", "Cap"],
            index=0,
            key="outlier_treatment"
        )
        
        # Feature engineering
        feature_engineering = st.multiselect(
            "Feature Engineering",
            ["Technical Indicators", "Date Features", "Lagged Features", "Returns", "Volatility"],
            default=["Technical Indicators", "Date Features", "Returns"],
            key="feature_engineering"
        )
    
    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

def plot_rsi(df):
    """Plot Relative Strength Index chart."""
    if "RSI" not in df.columns:
        st.warning("RSI indicator not available in data")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_macd(df):
    """Plot MACD chart."""
    required_cols = ["MACD", "MACD_signal", "MACD_hist"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing MACD columns: {missing_cols}")
        return
    
    fig = go.Figure()
    
    # Plot MACD line
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue")))
    
    # Plot Signal line
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="red")))
    
    # Plot Histogram
    colors = ["green" if val > 0 else "red" for val in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram", marker_color=colors))
    
    fig.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_bollinger_bands(df):
    """Plot Bollinger Bands chart."""
    required_cols = ["boll_upper", "boll_middle", "boll_lower"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing Bollinger Bands columns: {missing_cols}")
        return
    
    fig = go.Figure()
    
    # Plot Price
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="black")))
    
    # Plot Upper Band
    fig.add_trace(go.Scatter(x=df.index, y=df["boll_upper"], name="Upper Band", 
                            line=dict(color="red", dash="dash")))
    
    # Plot Middle Band (SMA)
    fig.add_trace(go.Scatter(x=df.index, y=df["boll_middle"], name="SMA", 
                            line=dict(color="blue")))
    
    # Plot Lower Band
    fig.add_trace(go.Scatter(x=df.index, y=df["boll_lower"], name="Lower Band", 
                            line=dict(color="green", dash="dash")))
    
    fig.update_layout(
        title="Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_moving_averages(df):
    """Plot Moving Averages chart."""
    fig = go.Figure()
    
    # Plot Price
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="black")))
    
    # Calculate and plot SMAs if not available
    ma_periods = [5, 20, 50, 200]
    ma_colors = ["blue", "green", "orange", "red"]
    
    for period, color in zip(ma_periods, ma_colors):
        col_name = f"SMA_{period}"
        if col_name not in df.columns:
            # Calculate SMA if not in dataframe
            df[col_name] = df["Close"].rolling(period).mean()
        
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[col_name], 
            name=f"SMA {period}", 
            line=dict(color=color)
        ))
    
    fig.update_layout(
        title="Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_atr(df):
    """Plot Average True Range chart."""
    if "ATR" not in df.columns:
        st.warning("ATR indicator not available in data")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], name="ATR", line=dict(color="purple")))
    
    fig.update_layout(
        title="Average True Range (ATR)",
        xaxis_title="Date",
        yaxis_title="ATR Value",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_obv(df):
    """Plot On-Balance Volume chart."""
    if "OBV" not in df.columns:
        # Calculate OBV if not available
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="blue")))
    
    fig.update_layout(
        title="On-Balance Volume (OBV)",
        xaxis_title="Date",
        yaxis_title="OBV Value",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_model_performance_page():
    """Render the model performance page with metrics and comparisons."""
    st.header("Model Performance")
    
    # Get model metrics if available
    if "model_metrics" not in st.session_state:
        st.warning("No model metrics available yet. Train models first.")
        return
    
    metrics = st.session_state["model_metrics"]
    
    # Display overall metrics
    st.subheader("Overall Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    with col2:
        st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    with col3:
        st.metric("R²", f"{metrics.get('r2', 0):.4f}")
    with col4:
        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
    
    # Plot performance metrics
    performance_fig = plot_model_performance(metrics)
    st.plotly_chart(performance_fig, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Comparison")
    
    # Get individual model metrics if available
    if "submodel_metrics" in st.session_state:
        submodel_metrics = st.session_state["submodel_metrics"]
        
        # Create dataframe for comparison
        comparison_data = []
        for model_type, model_metrics in submodel_metrics.items():
            comparison_data.append({
                "Model": model_type,
                "RMSE": model_metrics.get("rmse", 0),
                "MAPE": model_metrics.get("mape", 0),
                "R²": model_metrics.get("r2", 0),
                "MAE": model_metrics.get("mae", 0),
                "Training Time": f"{model_metrics.get('training_time', 0):.2f}s"
            })
        
        # Create and display comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Plot comparison chart
        st.subheader("RMSE by Model Type")
        
        fig = px.bar(
            comparison_df,
            x="Model",
            y="RMSE",
            color="Model",
            title="RMSE Comparison by Model Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Individual model metrics not available")
    
    # Prediction Monitor
    st.subheader("Prediction Monitoring")
    
    # Get monitor from state management
    monitor = get_prediction_monitor()
    
    if monitor:
        # Get accuracy metrics for current ticker/timeframe
        ticker = get_current_ticker()
        timeframe = get_current_timeframe()
        
        metrics_24h = monitor.get_accuracy_metrics("24h", ticker, timeframe)
        metrics_7d = monitor.get_accuracy_metrics("7d", ticker, timeframe)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("24h MAPE", f"{metrics_24h.get('mean_error', 0):.2f}%")
            
        with col2:
            st.metric("7d MAPE", f"{metrics_7d.get('mean_error', 0):.2f}%")
            
        with col3:
            st.metric("Direction Accuracy", f"{metrics_24h.get('correct_direction', 0)*100:.1f}%")
        
        # Show recent predictions
        st.subheader("Recent Predictions")
        recent_predictions = monitor.get_recent_predictions(10, ticker, timeframe)
        if not recent_predictions.empty:
            st.dataframe(recent_predictions)
        else:
            st.info("No prediction history available yet")
    else:
        st.info("Prediction monitoring is not initialized")

def render_dashboard():
    """Main function to render the dashboard UI."""
    # Set page config
    st.set_page_config(
        page_title="Stock Price Prediction Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize dashboard state
    from src.dashboard.dashboard.dashboard_state import initialize_state
    initialize_state()
    
    # Handle auto-refresh if enabled
    if get_state("auto_refresh", False):
        current_time = time.time()
        last_refresh = get_state("last_refresh", 0)
        refresh_interval = get_state("refresh_interval", 30)
        
        if current_time - last_refresh > refresh_interval:
            set_state("last_refresh", current_time)
            set_state("refresh_requested", True)
    
    # REMOVED: Redundant initialization of prediction_monitor
    # The monitor is now managed by dashboard_state.py
    
    # Get the prediction service for this session
    prediction_service = get_prediction_service()
    
    # Set the correct ticker and timeframe on the service
    if prediction_service:
        prediction_service.current_ticker = get_current_ticker()
        prediction_service.current_timeframe = get_current_timeframe()
        
        # Update monitor reference if needed
        if prediction_service.monitor is None:
            prediction_service.monitor = get_prediction_monitor()
    
    # Render sidebar and get options
    options = render_sidebar()
    
    # Display appropriate page based on selection
    if options["page"] == "Dashboard":
        render_dashboard_header()
        render_main_dashboard(options)
    elif options["page"] == "Pattern Discovery":
        if "df" in st.session_state:
            add_pattern_discovery_tab(st.session_state["df"])
        else:
            st.warning("Please load data first to discover patterns.")
    elif options["page"] == "Explainable AI":
        render_explainable_ai_tab()
    elif options["page"] == "Model Performance":
        render_model_performance_page()
    elif options["page"] == "Training Optimizer":
        render_training_optimizer_tab()
    elif options["page"] == "Settings":
        render_settings_page()
    
    # Handle refresh request
    if st.session_state["refresh_requested"]:
        # Reset the flag
        st.session_state["refresh_requested"] = False
        
        # Save current predictions before fetching new data
        if "df" in st.session_state and st.session_state["df"] is not None:
            if "future_forecast" in st.session_state and st.session_state["future_forecast"]:
                from src.dashboard.dashboard.dashboard_visualization import save_best_prediction
                save_best_prediction(st.session_state["df"], st.session_state["future_forecast"])
        
        # Fetch new data here
        with st.spinner("Fetching latest data..."):
            try:
                from src.data.data import fetch_data
                
                df = fetch_data(
                    ticker=options["ticker"],
                    start=options["start_date"].strftime("%Y-%m-%d"),
                    end=options["end_date"].strftime("%Y-%m-%d"),
                    interval=options["timeframe"]
                )
                
                if df is not None and not df.empty:
                    st.session_state["df"] = df
                    st.success("Data refreshed successfully!")
                else:
                    st.error("Failed to fetch data.")
            except Exception as e:
                st.error(f"Error refreshing data: {e}")
    
    # Handle training request
    if st.session_state["training_requested"]:
        # Reset the flag
        st.session_state["training_requested"] = False
        
        # Check if data is available
        if "df" not in st.session_state or st.session_state["df"] is None:
            st.error("No data available. Please fetch data first.")
            return
        
        # Start model training
        with st.spinner("Training models..."):
            try:
                from src.training.walk_forward import unified_walk_forward_optimized
                from src.features.features import get_feature_list

                # Get feature columns
                feature_cols = get_feature_list(st.session_state["df"])
                
                # Get ensemble weights
                ensemble_weights = st.session_state.get("ensemble_weights", None)
                
                # Train models using walk-forward validation
                ensemble_model, metrics = unified_walk_forward_optimized(
                    df=st.session_state["df"],
                    feature_cols=feature_cols,
                    ensemble_weights=ensemble_weights,
                    window_size=st.session_state.get("wf_size", 5),
                    update_dashboard=True
                )
                
                # Store model and metrics
                st.session_state["current_model"] = ensemble_model
                st.session_state["metrics"] = metrics
                
                st.success(f"Models trained successfully! RMSE: {metrics.get('rmse', 0):.4f}, MAPE: {metrics.get('mape', 0):.2f}%")
            except Exception as e:
                logger.exception("Error during model training")
                st.error(f"Error training models: {e}")

if __name__ == "__main__":
    render_dashboard()