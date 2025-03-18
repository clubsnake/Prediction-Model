"""
Core functionality for the unified dashboard. This file serves as the main entry point
for the dashboard application and orchestrates all the dashboard components.
"""

import os
import sys
import base64
import traceback
import time
from datetime import datetime, timedelta
import logging

# Add project root to Python path for reliable imports
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standard library imports (always available)
import pandas as pd
import numpy as np
import streamlit as st

# Setup basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dashboard_core')

# Import core dashboard components with proper error handling
from src.dashboard.dashboard.dashboard_state import init_session_state
from src.dashboard.dashboard.dashboard_error import robust_error_boundary

# Import utility functions for GPU optimization and threadsafe operation
try:
    from src.utils.gpu_memory_management import (
        configure_gpu_memory,
        clean_gpu_memory
    )
    from src.utils.threadsafe import cleanup_stale_locks
    logger.info("Successfully imported GPU and threadsafe utilities")
except ImportError as e:
    logger.warning(f"Error importing GPU or threadsafe utilities: {e}")

# Initialize session state
init_session_state()

# Import configuration with fallbacks
try:
    from config.config_loader import get_config
    config = get_config()
    DATA_DIR = config.get("DATA_DIR", os.path.join(project_root, "data"))
    TICKER = config.get("TICKER", "ETH-USD")
    TICKERS = config.get("TICKERS", ["ETH-USD", "BTC-USD"])
    TIMEFRAMES = config.get("TIMEFRAMES", ["1d", "1h"])
    logger.info("Successfully imported config")
except ImportError as e:
    logger.warning(f"Error importing config: {e}")
    # Fallback values
    DATA_DIR = os.path.join(project_root, "data")
    TICKER = "ETH-USD"
    TICKERS = ["ETH-USD", "BTC-USD"]
    TIMEFRAMES = ["1d", "1h"]


@robust_error_boundary
def configure_gpu_for_inference(enable_mixed_precision=False, log_config=True):
    """
    Configure GPU for optimal inference performance.

    Args:
        enable_mixed_precision: Whether to enable mixed precision 
        log_config: Whether to log configuration

    Returns:
        Dict with configuration results
    """
    # Use centralized GPU configuration
    config = {
        "allow_growth": True,  # Allow memory growth to avoid OOM errors
        "mixed_precision": enable_mixed_precision,  # Usually False for inference, True for training
        "use_xla": True,  # Enable XLA for better performance
        "directml_enabled": True  # Support DirectML on Windows
    }
    
    # Apply configuration
    try:
        result = configure_gpu_memory(config)
        
        if log_config:
            logger.info(f"Configured GPU for inference: {result}")
        
        return result
    except Exception as e:
        logger.warning(f"Could not configure GPU: {e}")
        return {"error": str(e), "success": False}


@robust_error_boundary
def load_data_with_caching(
    ticker, 
    start_date, 
    end_date=None, 
    interval="1d", 
    force_refresh=False, 
    use_cached=True,
    cache_timeout_minutes=120
):
    """
    Enhanced load_data function with improved caching to prevent excessive API calls.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date for data
        end_date: End date for data (defaults to now)
        interval: Data interval
        force_refresh: Force refresh from API
        use_cached: Use cached data if available and not expired
        cache_timeout_minutes: Cache expiration in minutes
        
    Returns:
        DataFrame with market data
    """
    # Generate a cache key based on parameters
    cache_key = f"data_{ticker}_{interval}_{start_date}_{end_date}"
    
    # Check if we have cached data and it's not expired
    if (use_cached and 
        "data_cache" in st.session_state and 
        cache_key in st.session_state["data_cache"] and
        not force_refresh):
        
        cached_data = st.session_state["data_cache"][cache_key]
        cache_timestamp = cached_data.get("timestamp", 0)
        cache_age_minutes = (time.time() - cache_timestamp) / 60
        
        # Use cached data if it's not expired
        if cache_age_minutes < cache_timeout_minutes:
            logger.info(f"Using cached data for {ticker} ({interval}), age: {cache_age_minutes:.1f} minutes")
            return cached_data.get("data")
        else:
            logger.info(f"Cached data expired for {ticker} ({interval}), age: {cache_age_minutes:.1f} minutes")
    
    # Create a spinner while loading data
    with st.spinner(f"Loading market data for {ticker}..."):
        try:
            # Import the actual load_data function - this avoids circular imports
            from src.dashboard.dashboard.dashboard_data import load_data as original_load_data
            
            # Call the original load_data function
            df = original_load_data(ticker, start_date, end_date, interval)
            
            # Cache the data if successful
            if df is not None and not df.empty:
                if "data_cache" not in st.session_state:
                    st.session_state["data_cache"] = {}
                
                st.session_state["data_cache"][cache_key] = {
                    "data": df,
                    "timestamp": time.time()
                }
                
                # Log cache update
                logger.info(f"Updated cache for {ticker} ({interval}), cached {len(df)} rows")
                
                return df
            else:
                logger.warning(f"Failed to load data for {ticker} ({interval})")
                return None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            
            # Return cached data even if expired in case of error
            if use_cached and "data_cache" in st.session_state and cache_key in st.session_state["data_cache"]:
                logger.info(f"Returning expired cached data due to load error")
                return st.session_state["data_cache"][cache_key].get("data")
            
            return None


@robust_error_boundary
def setup_parallel_model_execution():
    """
    Configure for optimal parallel model execution.
    
    Returns:
        Dict with configuration settings
    """
    try:
        import tensorflow as tf
        import multiprocessing
        
        # Get CPU count for optimal threading
        cpu_count = multiprocessing.cpu_count()
        
        # Configure thread parallelism for better performance
        # Use try/except to handle the "already initialized" case
        try:
            if hasattr(tf.config.threading, "set_inter_op_parallelism_threads"):
                tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
        except RuntimeError as e:
            # Ignore "Inter op parallelism cannot be modified after initialization" error
            logger.warning(f"Could not set inter_op_parallelism_threads: {e}")
        
        try:
            if hasattr(tf.config.threading, "set_intra_op_parallelism_threads"):
                tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
        except RuntimeError as e:
            # Ignore "Intra op parallelism cannot be modified after initialization" error
            logger.warning(f"Could not set intra_op_parallelism_threads: {e}")
        
        # Check if we have GPUs
        gpus = tf.config.list_physical_devices('GPU')
        has_gpu = len(gpus) > 0
        
        if has_gpu:
            # Configure memory growth
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # This typically happens when the runtime is already initialized
                    logger.warning(f"Could not set memory growth for GPU: {e}")
            
            # Enable XLA compilation for better performance
            if hasattr(tf.config.optimizer, "set_jit"):
                try:
                    tf.config.optimizer.set_jit(True)
                except RuntimeError as e:
                    logger.warning(f"Could not set JIT optimization: {e}")
        
        # Return configuration
        return {
            "cpu_count": cpu_count,
            "gpu_count": len(gpus),
            "memory_growth": True,
            "xla_enabled": hasattr(tf.config.optimizer, "set_jit"),
            "threading_configured": True
        }
    except Exception as e:
        logger.error(f"Error setting up parallel execution: {e}")
        return {
            "error": str(e),
            "cpu_count": multiprocessing.cpu_count() if 'multiprocessing' in locals() else "unknown",
            "configured": False
        }


@robust_error_boundary
def improved_refresh_mechanism():
    """
    Improved refresh mechanism that shows a countdown and handles refresh more gracefully.
    """
    # Check if auto-refresh is enabled
    if not st.session_state.get("auto_refresh", False):
        return
    
    # Get refresh interval
    refresh_interval = st.session_state.get("refresh_interval", 30)
    
    # Calculate time until next refresh
    current_time = time.time()
    if "last_refresh" in st.session_state:
        time_since_last = current_time - st.session_state["last_refresh"]
        time_to_next = max(0, refresh_interval - time_since_last)
        
        # Display progress bar in sidebar for refresh countdown
        if "sidebar_displayed" in st.session_state and st.session_state["sidebar_displayed"]:
            progress = 1 - (time_to_next / refresh_interval)
            
            # Only rerun if we've reached the refresh interval
            if time_to_next <= 0:
                # Update last refresh time
                st.session_state["last_refresh"] = current_time
                
                # Delay slightly to avoid immediate refresh
                time.sleep(0.1)
                st.experimental_rerun()
    else:
        # Initialize last_refresh
        st.session_state["last_refresh"] = current_time


@robust_error_boundary
def check_and_update_session_state(state_key, default_value):
    """
    Safely check and update session state with a default value if key doesn't exist.
    
    Args:
        state_key: Key to check in session state
        default_value: Default value to set if key doesn't exist
        
    Returns:
        Current value of the state key
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value
    return st.session_state[state_key]


@robust_error_boundary
def check_cache_status():
    """
    Check the status of data caches and return statistics.
    
    Returns:
        Dict with cache statistics
    """
    cache_stats = {
        "data_cache_enabled": "data_cache" in st.session_state,
        "model_cache_enabled": "model_cache" in st.session_state,
        "prediction_cache_enabled": "prediction_cache" in st.session_state,
        "data_cache_entries": 0,
        "model_cache_entries": 0,
        "prediction_cache_entries": 0,
        "total_cache_entries": 0
    }
    
    # Count data cache entries
    if "data_cache" in st.session_state:
        cache_stats["data_cache_entries"] = len(st.session_state["data_cache"])
        cache_stats["total_cache_entries"] += len(st.session_state["data_cache"])
    
    # Count model cache entries
    if "model_cache" in st.session_state:
        cache_stats["model_cache_entries"] = len(st.session_state["model_cache"])
        cache_stats["total_cache_entries"] += len(st.session_state["model_cache"])
    
    # Count prediction cache entries
    if "prediction_cache" in st.session_state:
        cache_stats["prediction_cache_entries"] = len(st.session_state["prediction_cache"])
        cache_stats["total_cache_entries"] += len(st.session_state["prediction_cache"])
    
    return cache_stats


@robust_error_boundary
def init_dashboard_state():
    """
    Initialize all required dashboard state variables.
    """
    # Basic dashboard state
    check_and_update_session_state("error_log", [])
    check_and_update_session_state("metrics_container", None)
    check_and_update_session_state("tuning_in_progress", False)
    
    # Cache initialization
    check_and_update_session_state("data_cache", {})
    check_and_update_session_state("model_cache", {})
    check_and_update_session_state("prediction_cache", {})
    
    # Feature visibility
    check_and_update_session_state("show_werpi", False)
    check_and_update_session_state("show_vmli", False)
    check_and_update_session_state("show_ma", False)
    check_and_update_session_state("show_bb", False)
    check_and_update_session_state("show_rsi", False)
    check_and_update_session_state("show_macd", False)
    check_and_update_session_state("show_forecast", True)
    check_and_update_session_state("show_confidence", True)
    
    # Aggregate indicators into a single dict
    check_and_update_session_state("indicators", {
        "show_ma": st.session_state.get("show_ma", False),
        "show_bb": st.session_state.get("show_bb", False),
        "show_rsi": st.session_state.get("show_rsi", False),
        "show_macd": st.session_state.get("show_macd", False),
        "show_werpi": st.session_state.get("show_werpi", False),
        "show_vmli": st.session_state.get("show_vmli", False),
        "show_forecast": st.session_state.get("show_forecast", True),
        "show_confidence": st.session_state.get("show_confidence", True),
    })
    
    # Auto-refresh settings
    check_and_update_session_state("auto_refresh", True)
    check_and_update_session_state("refresh_interval", 30)
    check_and_update_session_state("last_refresh", time.time())
    
    # Walk-forward settings
    check_and_update_session_state("update_during_walk_forward", True)
    check_and_update_session_state("update_during_walk_forward_interval", 5)
    
    # UI state tracking
    check_and_update_session_state("sidebar_displayed", False)
    
    # Parallel execution setup
    if "parallel_config" not in st.session_state:
        st.session_state["parallel_config"] = setup_parallel_model_execution()
    
    # Configure GPU for inference
    if "gpu_configured" not in st.session_state:
        st.session_state["gpu_configured"] = configure_gpu_for_inference(
            enable_mixed_precision=False
        )
    
    # Clean stale locks on startup
    if "locks_cleaned" not in st.session_state:
        try:
            cleaned_count = cleanup_stale_locks(max_age=60)
            st.session_state["locks_cleaned"] = cleaned_count
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} stale lock files on dashboard startup")
        except Exception as e:
            logger.warning(f"Error cleaning stale locks: {e}")
            st.session_state["locks_cleaned"] = 0


@robust_error_boundary
def load_model_status():
    """
    Load the current status of all models being trained.
    
    Returns:
        Dict with model status information
    """
    try:
        from config.config_loader import TESTED_MODELS_FILE
        from src.utils.threadsafe import safe_read_yaml
        
        # Read tested models file
        tested_models = safe_read_yaml(TESTED_MODELS_FILE, default=[])
        
        # Group by model type
        model_stats = {}
        
        if tested_models and isinstance(tested_models, list):
            for model in tested_models:
                model_type = model.get("model_type", "unknown")
                
                if model_type not in model_stats:
                    model_stats[model_type] = {
                        "count": 0,
                        "best_rmse": float('inf'),
                        "best_mape": float('inf'),
                        "last_updated": None
                    }
                
                # Update stats
                model_stats[model_type]["count"] += 1
                
                # Update best metrics
                rmse = model.get("rmse")
                if rmse is not None and rmse < model_stats[model_type]["best_rmse"]:
                    model_stats[model_type]["best_rmse"] = rmse
                
                mape = model.get("mape")
                if mape is not None and mape < model_stats[model_type]["best_mape"]:
                    model_stats[model_type]["best_mape"] = mape
                
                # Update timestamp
                timestamp = model.get("timestamp")
                if timestamp:
                    if model_stats[model_type]["last_updated"] is None or timestamp > model_stats[model_type]["last_updated"]:
                        model_stats[model_type]["last_updated"] = timestamp
        
        return {
            "models": model_stats,
            "total_models": len(tested_models),
            "model_types": len(model_stats)
        }
    except Exception as e:
        logger.error(f"Error loading model status: {e}")
        return {
            "error": str(e),
            "models": {},
            "total_models": 0,
            "model_types": 0
        }


@robust_error_boundary
def import_dashboard_modules():
    """Import dashboard modules with error handling to avoid circular imports"""
    modules = {}
    
    # Try to import UI components
    try:
        from src.dashboard.dashboard.dashboard_ui import (
            create_header,
            create_control_panel,
            create_metrics_cards,
            create_tuning_panel,
        )
        modules.update({
            "create_header": create_header,
            "create_control_panel": create_control_panel,
            "create_metrics_cards": create_metrics_cards,
            "create_tuning_panel": create_tuning_panel,
        })
    except ImportError as e:
        logger.error(f"Error importing UI components: {e}")
        # Define minimal fallbacks
        def create_header_fallback():
            st.title("AI Price Prediction Dashboard")
        
        def create_control_panel_fallback():
            ticker = st.sidebar.selectbox("Ticker", TICKERS)
            timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES)
            return {"ticker": ticker, "timeframe": timeframe}
        
        def create_metrics_cards_fallback():
            st.write("Metrics not available")
        
        def create_tuning_panel_fallback():
            st.info("Tuning panel not available")
        
        modules.update({
            "create_header": create_header_fallback,
            "create_control_panel": create_control_panel_fallback,
            "create_metrics_cards": create_metrics_cards_fallback,
            "create_tuning_panel": create_tuning_panel_fallback,
        })
    
    # Try to import data handling functions
    try:
        from src.dashboard.dashboard.dashboard_data import (
            calculate_indicators,
            ensure_date_column,
            generate_dashboard_forecast,
            load_data,
        )
        modules.update({
            "load_data": load_data,
            "calculate_indicators": calculate_indicators,
            "generate_dashboard_forecast": generate_dashboard_forecast,
            "ensure_date_column": ensure_date_column,
        })
    except ImportError as e:
        logger.error(f"Error importing data functions: {e}")
        
        # Define fallback functions
        def load_data_fallback(ticker, start_date, end_date=None, interval="1d", training_mode=False):
            logger.warning(f"Using fallback load_data for {ticker}")
            return None

        def calculate_indicators_fallback(df):
            logger.warning("Using fallback calculate_indicators")
            return df

        def generate_forecast_fallback(model, df, feature_cols):
            logger.warning("Using fallback generate_forecast")
            return []

        def ensure_date_column_fallback(df, default_name="date"):
            logger.warning("Using fallback ensure_date_column")
            if df is None or df.empty:
                return df, default_name

            # Try to find a date column
            date_col = None
            for col in ["date", "Date", "timestamp", "Timestamp"]:
                if col in df.columns:
                    date_col = col
                    break

            # If no date column found, create one
            if date_col is None:
                df = df.copy()
                df[default_name] = pd.date_range(
                    start=datetime.now() - timedelta(days=len(df)), periods=len(df)
                )
                date_col = default_name

            return df, date_col

        modules.update({
            "load_data": load_data_fallback,
            "calculate_indicators": calculate_indicators_fallback,
            "generate_dashboard_forecast": generate_forecast_fallback,
            "ensure_date_column": ensure_date_column_fallback,
        })
    
    # Add utility functions
    def standardize_column_names(df, ticker=None):
        """Standardize column names by removing ticker-specific parts for OHLCV."""
        if df is None or df.empty:
            return df
        df_copy = df.copy()

        # First check if standard columns already exist
        has_standard = all(
            col in df_copy.columns for col in ["Open", "High", "Low", "Close"]
        )
        if has_standard:
            return df_copy

        # Handle ticker-specific column names
        if ticker:
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                # Check for pattern like 'Close_ETH-USD'
                if f"{col}_{ticker}" in df_copy.columns:
                    df_copy[col] = df_copy[f"{col}_{ticker}"]
                # Check for pattern like 'ETH-USD_Close'
                elif f"{ticker}_{col}" in df_copy.columns:
                    df_copy[col] = df_copy[f"{ticker}_{col}"]
                # Check for pattern with underscores removed
                ticker_nohyphen = ticker.replace("-", "")
                if f"{col}_{ticker_nohyphen}" in df_copy.columns:
                    df_copy[col] = df_copy[f"{col}_{ticker_nohyphen}"]

        # Also handle lowercase variants
        for old_col, std_col in [
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
            ("volume", "Volume"),
        ]:
            if old_col in df_copy.columns and std_col not in df_copy.columns:
                df_copy[std_col] = df_copy[old_col]

        # Log missing columns that we couldn't standardize
        missing = [
            c for c in ["Open", "High", "Low", "Close"] if c not in df_copy.columns
        ]
        if missing:
            logger.warning(
                f"Missing required columns after standardization: {missing}"
            )
            logger.info(f"Available columns: {df_copy.columns.tolist()}")

        return df_copy
    
    modules["standardize_column_names"] = standardize_column_names
    
    # Try to import visualization functions
    try:
        from src.dashboard.dashboard.dashboard_visualization import (
            prepare_dataframe_for_display,
            create_interactive_price_chart,
            create_technical_indicators_chart,
            show_market_regime_info,
            show_confidence_breakdown,
            show_advanced_dashboard_tabs,
        )
        modules.update({
            "prepare_dataframe_for_display": prepare_dataframe_for_display,
            "create_interactive_price_chart": create_interactive_price_chart,
            "create_technical_indicators_chart": create_technical_indicators_chart,
            "show_market_regime_info": show_market_regime_info,
            "show_confidence_breakdown": show_confidence_breakdown,
            "show_advanced_dashboard_tabs": show_advanced_dashboard_tabs,
        })
    except ImportError as e:
        logger.error(f"Error importing visualization functions: {e}")
        
        # Define fallback visualization functions
        def prepare_dataframe_for_display_fallback(df):
            """Fallback function for dataframe display preparation"""
            logger.warning("Using fallback prepare_dataframe_for_display")
            if df is None or df.empty:
                return pd.DataFrame()
            
            return df.copy()
        
        def create_interactive_price_chart_fallback(df, options, **kwargs):
            """Fallback function for price chart"""
            logger.warning("Using fallback create_interactive_price_chart")
            if df is None or df.empty:
                st.warning("No data available to display chart.")
                return
                
            if "Close" in df.columns:
                st.line_chart(df["Close"])
            else:
                st.warning("No price data found in expected columns.")
        
        def create_technical_indicators_chart_fallback(df, options=None):
            """Fallback function for technical indicators chart"""
            logger.warning("Using fallback create_technical_indicators_chart")
            if df is None or df.empty:
                st.warning("No data available to display technical indicators.")
                return
                
            if "Close" in df.columns:
                st.line_chart(df[["Close"]])
        
        def show_market_regime_info_fallback(regime_stats):
            """Fallback function for market regime info"""
            logger.warning("Using fallback show_market_regime_info")
            st.info("Market regime information not available.")
        
        def show_confidence_breakdown_fallback(confidence_components, future_dates):
            """Fallback function for confidence breakdown"""
            logger.warning("Using fallback show_confidence_breakdown")
            st.info("Confidence breakdown not available.")
        
        def show_advanced_dashboard_tabs_fallback(df):
            """Fallback function for advanced dashboard tabs"""
            logger.warning("Using fallback show_advanced_dashboard_tabs")
            st.info("Advanced dashboard features not available.")
            
        modules.update({
            "prepare_dataframe_for_display": prepare_dataframe_for_display_fallback,
            "create_interactive_price_chart": create_interactive_price_chart_fallback,
            "create_technical_indicators_chart": create_technical_indicators_chart_fallback,
            "show_market_regime_info": show_market_regime_info_fallback,
            "show_confidence_breakdown": show_confidence_breakdown_fallback,
            "show_advanced_dashboard_tabs": show_advanced_dashboard_tabs_fallback,
        })
    
    # Try to import model visualization modules
    try:
        from src.dashboard.model_visualizations import ModelVisualizationDashboard
        modules["ModelVisualizationDashboard"] = ModelVisualizationDashboard
    except ImportError:
        logger.warning("ModelVisualizationDashboard not available")
        
    # Try to import pattern discovery module
    try:
        from src.dashboard.pattern_discovery import (
            identify_patterns, 
            visualize_patterns,
            find_similar_patterns,
        )
        modules.update({
            "identify_patterns": identify_patterns,
            "visualize_patterns": visualize_patterns,
            "find_similar_patterns": find_similar_patterns,
        })
    except ImportError:
        logger.warning("Pattern discovery module not available")
    
    # Try to import drift monitoring module
    try:
        from src.dashboard.drift_dashboard import show_drift_visualization
        modules["show_drift_visualization"] = show_drift_visualization
    except ImportError:
        logger.warning("Drift visualization module not available")
        
        def show_drift_visualization_fallback():
            """Fallback function for drift visualization"""
            st.info("Drift monitoring visualization not available.")
            
        modules["show_drift_visualization"] = show_drift_visualization_fallback
    
    # Try to import XAI module
    try:
        from src.dashboard.xai_integration import create_xai_explorer
        modules["create_xai_explorer"] = create_xai_explorer
    except ImportError:
        logger.warning("XAI integration module not available")
        
        def create_xai_explorer_fallback(*args, **kwargs):
            """Fallback function for XAI explorer"""
            st.info("XAI explorer not available.")
            
        modules["create_xai_explorer"] = create_xai_explorer_fallback
    
    return modules


# Import dashboard modules using the function
dashboard_modules = import_dashboard_modules()

# Safety check to ensure we always have a dictionary
if dashboard_modules is None:
    logger.error("dashboard_modules is None - creating fallback dictionary")
    dashboard_modules = {}
    
    # Define minimal fallbacks for essential functions
    dashboard_modules["create_header"] = lambda: st.title("AI Price Prediction Dashboard")
    dashboard_modules["create_control_panel"] = lambda: {"ticker": "ETH-USD", "timeframe": "1d"}
    dashboard_modules["create_metrics_cards"] = lambda: st.write("Metrics not available")
    dashboard_modules["create_tuning_panel"] = lambda: st.info("Tuning panel not available")
    
    # Other essential fallbacks
    dashboard_modules["calculate_indicators"] = lambda df: df
    dashboard_modules["ensure_date_column"] = lambda df, name="date": (df, name)
    dashboard_modules["standardize_column_names"] = lambda df, ticker=None: df
    dashboard_modules["prepare_dataframe_for_display"] = lambda df: df
    dashboard_modules["create_interactive_price_chart"] = lambda df, params, **kwargs: st.line_chart(df["Close"] if "Close" in df.columns else df)


@robust_error_boundary
def set_page_config():
    """Configure the Streamlit page settings with modern styling"""
    try:
        st.set_page_config(
            page_title="AI Price Prediction Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        # Custom CSS for dark mode look
        st.markdown(
            """
        <style>
        .main {
            padding-top: 1rem;
            background-color: #121212;
            color: #e0e0e0;
        }
        h1, h2, h3 {
            font-family: 'Roboto', sans-serif;
            color: #2196F3;
        }
        .sidebar .sidebar-content {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .metric-container {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            border-left: 4px solid #2196F3;
            color: #e0e0e0;
        }
        .stButton>button {
            background-color: #2196F3;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
        }
        .chart-container {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
            border-top: 4px solid #2196F3;
            color: #e0e0e0;
        }
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #4CAF50, #8BC34A);
        }
        /* Dark theme tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e1e1e;
            border-radius: 8px 8px 0px 0px;
            gap: 1px;
            padding-top: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #2a2a2a;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 500;
            color: #e0e0e0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2196F3 !important;
            color: white !important;
        }
        /* Header bar styling */
        header[data-testid="stHeader"] {
            background-color: #121212;
            color: white;
        }
        /* Improve contrast for text */
        p, span, div {
            color: #e0e0e0;
        }
        .metric-value {
            color: #2196F3;
            font-size: 1.5em;
            font-weight: bold;
        }
        /* Make inputs and selects readable in dark mode */
        .stSelectbox>div>div, .stDateInput>div>div {
            background-color: #2a2a2a;
            color: #e0e0e0;
        }
        .stDataFrame {
            background-color: #1e1e1e;
        }
        /* Make tab sizing better */
        .stTabs [data-baseweb="tab"] {
            min-width: 120px;
            padding: 12px 24px;
            font-size: 1.05em;
            text-align: center;
        }
        /* Increase clickable area for tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            padding: 0px 0px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
    except Exception:
        # This might fail if called twice
        logger.warning("Error setting page config - may already be set")


@robust_error_boundary
def initialize_session_state():
    """Initialize session state variables with defensive checks"""
    # Use the enhanced initialization function
    init_dashboard_state()
    
    # These lines kept for backwards compatibility
    if "error_log" not in st.session_state:
        st.session_state["error_log"] = []
    if "metrics_container" not in st.session_state:
        st.session_state["metrics_container"] = None
    if "tuning_in_progress" not in st.session_state:
        st.session_state["tuning_in_progress"] = False
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = TICKER
    if "selected_timeframe" not in st.session_state:
        st.session_state["selected_timeframe"] = TIMEFRAMES[0]
    
    # Flag to mark sidebar as displayed
    st.session_state["sidebar_displayed"] = True


@robust_error_boundary
def check_tuning_status():
    """Check tuning status from file and update session state accordingly."""
    try:
        # First try to import from progress_helper for more reliable status
        try:
            from src.tuning.progress_helper import read_tuning_status as read_from_helper
            status = read_from_helper()
        except ImportError:
            # Fall back to dashboard_error version - import inside function to avoid circular imports
            try:
                from src.dashboard.dashboard.dashboard_error import read_tuning_status
                status = read_tuning_status()
            except ImportError:
                # Use fallback function if neither import works
                status = {"status": "unknown", "is_running": False}

        logger.info(f"Checking tuning status: {status}")

        # Make sure is_running is a boolean
        is_running = status.get("is_running", False)
        if isinstance(is_running, str):
            # Convert string "True"/"False" to boolean
            is_running = is_running.lower() == "true"

        # Log any mismatch
        current_status = st.session_state.get("tuning_in_progress", False)
        if is_running != current_status:
            logger.info(
                f"Tuning status mismatch - File: {is_running}, Session: {current_status}"
            )
            # Update session state to match file
            st.session_state["tuning_in_progress"] = is_running

        # Store additional info from status
        if "ticker" in status:
            st.session_state["tuning_ticker"] = status["ticker"]
        if "timeframe" in status:
            st.session_state["tuning_timeframe"] = status["timeframe"]
        if "start_time" in status:
            st.session_state["tuning_start_time"] = status["start_time"]

        return status
    except Exception as e:
        logger.error(f"Error checking tuning status: {e}")
        return {"is_running": False, "error": str(e)}


@robust_error_boundary
def clean_stale_locks():
    """Remove any stale lock files that might be preventing proper operation."""
    try:
        # Get path to DATA_DIR - we need this to handle potential lock files
        lock_dir = DATA_DIR
        
        # Use threadsafe's built-in cleanup function if available
        try:
            from src.utils.threadsafe import cleanup_stale_locks
            logger.info("Running stale lock cleanup...")
            cleaned_count = cleanup_stale_locks(max_age=3600)  # 1 hour
            logger.info(f"Cleaned {cleaned_count} stale lock files")
        except ImportError:
            logger.warning("threadsafe module not available, using basic lock cleanup")
            # Basic lock file cleanup - find any .lock files and check if they're stale
            for filename in os.listdir(lock_dir):
                if filename.endswith('.lock'):
                    lock_path = os.path.join(lock_dir, filename)
                    try:
                        # Check if lock file is older than 1 hour (stale)
                        file_age = datetime.now().timestamp() - os.path.getmtime(lock_path)
                        if file_age > 3600:  # 1 hour in seconds
                            os.remove(lock_path)
                            logger.info(f"Removed stale lock file: {lock_path}")
                    except Exception as e:
                        logger.warning(f"Error checking/removing lock file {lock_path}: {e}")

        # If tuning status shows running but UI doesn't think so, reset it
        try:
            # Check tuning status - import inside function to avoid circular imports
            from src.dashboard.dashboard.dashboard_error import read_tuning_status
            status = read_tuning_status()

            if status.get("is_running", False) and not st.session_state.get(
                "tuning_in_progress", False
            ):
                logger.info("Tuning status mismatch - resetting tuning status file")
                try:
                    # Get current values
                    ticker = status.get(
                        "ticker", st.session_state.get("selected_ticker", "unknown")
                    )
                    timeframe = status.get(
                        "timeframe", st.session_state.get("selected_timeframe", "1d")
                    )

                    # Write with is_running: False - import inside function
                    from src.dashboard.dashboard.dashboard_error import write_tuning_status
                    write_tuning_status(ticker, timeframe, is_running=False)
                    logger.info("Reset stale tuning status to not running")
                except Exception as e:
                    logger.error(f"Error resetting tuning status: {e}")
        except Exception as e:
            logger.error(f"Error checking tuning status: {e}")

    except Exception as e:
        logger.error(f"Error in clean_stale_locks: {e}")


@robust_error_boundary
def main_dashboard():
    """Main dashboard entry point with robust error handling"""
    try:
        # Configure GPU for inference if not already done
        if "gpu_configured" not in st.session_state:
            st.session_state["gpu_configured"] = configure_gpu_for_inference()
        
        # Clean up stale locks first thing
        clean_stale_locks()

        # Initialize session state
        initialize_session_state()

        # Setup page config
        set_page_config()

        # Check if shutdown was requested
        try:
            # Import conditionally to avoid circular imports
            from src.dashboard.dashboard.dashboard_shutdown import is_shutting_down, show_shutdown_message
            if is_shutting_down():
                show_shutdown_message()
                st.stop()
        except ImportError:
            # Continue without shutdown check if module not available
            pass

        # Check tuning status and update session state
        tuning_status = check_tuning_status()
        logger.info(f"Current tuning status: {tuning_status}")

        # Build UI components
        create_header_func = dashboard_modules.get("create_header", lambda: st.title("AI Price Prediction Dashboard"))
        create_header_func()

        # Create sidebar with controls
        create_control_panel_func = dashboard_modules.get(
            "create_control_panel",
            lambda: {"ticker": st.sidebar.selectbox("Ticker", TICKERS), 
                    "timeframe": st.sidebar.selectbox("Timeframe", TIMEFRAMES)}
        )
        params = create_control_panel_func()

        # Check if params is None (indicating an error)
        if params is None:
            st.error("Error in control panel. Using default parameters instead.")
            # Provide default values as fallback
            params = {
                "ticker": st.session_state.get("selected_ticker", TICKER),
                "timeframe": st.session_state.get("selected_timeframe", TIMEFRAMES[0]),
                "start_date": datetime.now() - timedelta(days=90),
                "end_date": datetime.now() + timedelta(days=30),
                "training_start_date": datetime.now() - timedelta(days=365 * 5),
                "historical_window": st.session_state.get("historical_window", 90),
                "forecast_window": st.session_state.get("forecast_window", 30),
                "auto_refresh": True,
                "refresh_interval": 30,
            }

        # Get parameters from the control panel
        ticker = params["ticker"]
        timeframe = params["timeframe"]
        start_date = params.get("start_date")  # For chart visualization
        end_date = params.get("end_date")
        training_start = params.get("training_start_date")  # For data loading and model training
        historical_window = params.get("historical_window")
        forecast_window = params.get("forecast_window")

        # Store parameters in session state for persistence
        st.session_state["selected_ticker"] = ticker
        st.session_state["selected_timeframe"] = timeframe
        st.session_state["start_date"] = start_date  # Chart visualization start
        st.session_state["end_date"] = end_date
        st.session_state["training_start_date"] = training_start  # Model training data start
        st.session_state["historical_window"] = historical_window
        st.session_state["forecast_window"] = forecast_window
        st.session_state["lookback"] = historical_window  # Add lookback as an alias

        # Show a loading spinner while fetching data
        with st.spinner("Loading market data..."):
            # Ensure we only fetch historical data up to today
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Convert dates to string format if needed
            start_date_str = params.get("start_date", datetime.now() - timedelta(days=30))
            if not isinstance(start_date_str, str):
                start_date_str = start_date_str.strftime("%Y-%m-%d")
                
            # Handle training_start_date separately 
            training_start_date_str = params.get("training_start_date", datetime.now() - timedelta(days=365*5))
            if not isinstance(training_start_date_str, str):
                training_start_date_str = training_start_date_str.strftime("%Y-%m-%d")
            
            # Use enhanced caching function for data loading
            # Important: Pass training_start_date for actual data retrieval
            df_vis = load_data_with_caching(
                ticker,
                training_start_date_str,  # Use training_start_date for data fetching
                today,  # Use current date instead of future end_date
                timeframe,
                force_refresh=params.get("force_refresh", False),
                cache_timeout_minutes=params.get("cache_timeout", 30)
            )
            
            if df_vis is not None and not df_vis.empty:
                # Standardize column names
                standardize_func = dashboard_modules.get("standardize_column_names", lambda df, ticker: df)
                df_vis = standardize_func(df_vis, ticker)
                
                # Fix: Use the centralized date handling function
                ensure_date_func = dashboard_modules.get("ensure_date_column", lambda df, name="date": (df, name))
                df_vis, date_col = ensure_date_func(df_vis)
                
                # Cache data in session state
                st.session_state["df_raw"] = df_vis
                
                # Store the actual chart visualization start date separately
                st.session_state["chart_start_date"] = start_date_str
            else:
                # If no new data, use cached data if available
                if st.session_state.get("df_raw") is not None:
                    df_vis = st.session_state["df_raw"]
                    st.warning("Using previously loaded data as new data could not be fetched.")
                else:
                    st.error(f"No data available for {ticker} from {start_date_str} to {today}.")
                    st.stop()  # abort if we have absolutely no data

        # UNIFIED CHART AT THE TOP 
        # Create a container for the chart with styling
        chart_container = st.container()
        with chart_container:
            # Initialize forecast data
            future_forecast = st.session_state.get("future_forecast", None)
            confidence_scores = st.session_state.get("forecast_confidence", None)
            confidence_components = st.session_state.get("confidence_components", {})
            
            # Get indicator preferences from params
            indicators = params.get("indicators", {})
            model = st.session_state.get("model")
            
            if model and indicators.get("show_forecast", True) and not future_forecast:
                with st.spinner("Generating forecast..."):
                    # Determine feature columns (exclude date and target 'Close')
                    feature_cols = [
                        col for col in df_vis.columns if col not in ["date", "Date", "Close"]
                    ]
                    # Use the consolidated function from dashboard_data
                    generate_forecast_func = dashboard_modules.get("generate_dashboard_forecast", lambda model, df, features: None)
                    future_forecast = generate_forecast_func(model, df_vis, feature_cols)
                    
                    # Save the prediction for historical comparison
                    if future_forecast and len(future_forecast) > 0:
                        try:
                            # Import visualization function
                            visualize_func = dashboard_modules.get("save_best_prediction", lambda df, forecast: None)
                            visualize_func(df_vis, future_forecast)
                        except Exception as e:
                            logger.error(f"Error saving prediction: {e}")

            # Calculate indicators and plot the chart with increased height
            calculate_indicators_func = dashboard_modules.get("calculate_indicators", lambda df: df)
            df_vis_indicators = calculate_indicators_func(df_vis)

            # Pass indicator options to visualization function with confidence
            try:
                # Try to use the imported function directly
                create_chart_func = dashboard_modules.get("create_interactive_price_chart", lambda df, params, **kwargs: st.line_chart(df[["Close"]]))
                create_chart_func(
                    df_vis_indicators,
                    params,
                    future_forecast=future_forecast,
                    confidence_scores=confidence_scores,
                    indicators=indicators,
                    height=850,
                )
                
                # Add confidence breakdown if available
                if confidence_scores is not None and confidence_components:
                    # Create future dates for display
                    last_date = df_vis["date"].iloc[-1] if "date" in df_vis.columns else pd.Timestamp.now()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1), 
                        periods=len(confidence_scores)
                    )
                    
                    # Show the confidence breakdown
                    with st.expander("Prediction Confidence Analysis", expanded=False):
                        show_confidence_func = dashboard_modules.get("show_confidence_breakdown", lambda components, dates: None)
                        show_confidence_func(confidence_components, future_dates)
                        
                        # Add an explanation
                        st.markdown("""
                        ### Confidence Score Interpretation:
                        
                        - **80-100%**: High confidence, models strongly agree and market conditions are stable
                        - **60-80%**: Good confidence, models generally agree with some minor variations
                        - **40-60%**: Moderate confidence, consider additional factors before trading
                        - **20-40%**: Low confidence, models disagree or market conditions are volatile
                        - **0-20%**: Very low confidence, forecast should be considered highly speculative
                        """)
            except Exception as e:
                logger.error(f"Error creating price chart: {e}")
                # Simple fallback if the chart function fails
                st.warning("Error displaying interactive chart. Showing basic chart instead.")
                if "Close" in df_vis.columns:
                    st.line_chart(df_vis["Close"])

        # Get model metrics and progress information
        best_metrics = st.session_state.get("best_metrics", {})
        best_rmse = best_metrics.get("rmse")
        best_mape = best_metrics.get("mape")
        direction_accuracy = best_metrics.get("direction_accuracy")

        # Get progress data
        try:
            from src.dashboard.dashboard.dashboard_error import load_latest_progress
            progress = load_latest_progress(ticker=ticker, timeframe=timeframe)
        except ImportError:
            progress = {
                "current_trial": 0,
                "total_trials": 1,
                "current_rmse": None,
                "current_mape": None,
                "cycle": 1,
            }
            
        cycle = progress.get("cycle", 1)
        current_trial = progress.get("current_trial", 0)
        total_trials = progress.get("total_trials", 1)

        # Show metrics below chart in a row
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
        
        # Display metrics in the row
        with metrics_col1:
            if best_rmse is not None:
                st.metric("Model RMSE", f"{best_rmse:.2f}")
            else:
                st.metric("Model RMSE", "N/A")
        with metrics_col2:
            if best_mape is not None:
                st.metric("Model MAPE", f"{best_mape:.2f}%")
            else:
                st.metric("Model MAPE", "N/A")
        with metrics_col3:
            if direction_accuracy is not None:
                st.metric("Direction Accuracy", f"{direction_accuracy:.1f}%")
            else:
                st.metric("Direction Accuracy", "N/A")
        with metrics_col4:
            st.markdown(
                """
                <div style="font-size: 1.15em;">
                    <p style="margin-bottom: 5px;"><strong>Cycle Status:</strong></p>
                    <p>Cycle: <strong>{}</strong> | Trial: <strong>{}/{}</strong></p>
                </div>
                """.format(
                    cycle, current_trial, total_trials
                ),
                unsafe_allow_html=True,
            )
        # Progress bar in the last column
        with metrics_col5:
            st.markdown(
                "<p style='font-size: 1.15em; margin-bottom: 5px;'><strong>Cycle Progress:</strong></p>",
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, current_trial / max(1, total_trials)))

        # Create main content tabs with all 9 tabs
        main_tabs = st.tabs([
            "🧠 Model Insights", 
            "⚙️ Model Tuning", 
            "📈 Technical Indicators", 
            "📊 Price Data",
            "🔄 Ensemble Weights",
            "🧮 Neural Architecture",
            "📉 Learning Insights",
            "🔍 Pattern Discovery",
            "🔄 Drift Monitor"
        ])

        # Tab 1: Model Insights
        with main_tabs[0]:
            st.subheader("Model Performance & Insights")
            
            # Use columns instead of nested tabs for better compatibility
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance Metrics
                st.markdown("### Performance Metrics")
                if "best_metrics" in st.session_state and st.session_state["best_metrics"]:
                    metrics = st.session_state["best_metrics"]
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Best RMSE", metrics.get("rmse", "N/A"))
                    with metrics_cols[1]:
                        mape_val = metrics.get("mape", "N/A")
                        mape_display = f"{mape_val:.2f}%" if isinstance(mape_val, (int, float)) else "N/A"
                        st.metric("Best MAPE", mape_display)
                    with metrics_cols[2]:
                        dir_acc = metrics.get("direction_accuracy", "N/A")
                        dir_acc_display = f"{dir_acc:.1f}%" if isinstance(dir_acc, (int, float)) else "N/A"
                        st.metric("Direction Accuracy", dir_acc_display)
                    with metrics_cols[3]:
                        st.metric("Model Type", metrics.get("model_type", "N/A"))
                else:
                    st.info(
                        "No model metrics available yet. Train a model to see performance insights."
                    )
                    
                # Show performance history if available
                if "model_history" in st.session_state and st.session_state["model_history"]:
                    st.subheader("Training History")
                    history_df = pd.DataFrame(st.session_state["model_history"])
                    st.line_chart(history_df)
            
            with col2:
                # Feature Importance 
                st.markdown("### Feature Importance")
                if "feature_importance" in st.session_state and st.session_state["feature_importance"]:
                    feature_df = pd.DataFrame(
                        {
                            "Feature": st.session_state["feature_importance"].keys(),
                            "Importance": st.session_state["feature_importance"].values(),
                        }
                    ).sort_values(by="Importance", ascending=False)
                    
                    # Use the safe prepare_dataframe_for_display function
                    prepare_df_func = dashboard_modules.get("prepare_dataframe_for_display", lambda df: df)
                    st.dataframe(prepare_df_func(feature_df))
                else:
                    st.info("No feature importance data available yet.")
            
                # Add XAI Explorer if available
                st.markdown("### Explainable AI")
                
                # Check if we have a model and sample data for XAI
                model = st.session_state.get("model")
                if model and "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
                    try:
                        # Try to get feature names
                        df = st.session_state["df_raw"]
                        feature_names = [col for col in df.columns if col not in ["date", "Date", "Close"]]
                        
                        # Get a sample for XAI analysis
                        X_sample = df[feature_names].values[-10:]
                        
                        # Create the XAI explorer
                        create_xai_func = dashboard_modules.get("create_xai_explorer")
                        if create_xai_func:
                            create_xai_func(st, model, X_sample, feature_names)
                        else:
                            st.info("XAI Explorer not available")
                    except Exception as e:
                        logger.error(f"Error creating XAI explorer: {e}")
                        st.error(f"Error creating XAI explorer: {str(e)}")
                else:
                    st.info("Model or data not available for XAI analysis")
            
            # Prediction Analysis section
            with st.expander("Prediction Analysis", expanded=False):
                if "prediction_history" in st.session_state and st.session_state["prediction_history"]:
                    st.subheader("Prediction History")
                    pred_df = pd.DataFrame(st.session_state["prediction_history"])
                    
                    # Use the safe prepare_dataframe_for_display function
                    prepare_df_func = dashboard_modules.get("prepare_dataframe_for_display", lambda df: df)
                    st.dataframe(prepare_df_func(pred_df))
                    
                    # Calculate direction accuracy
                    if "actual_direction" in pred_df.columns and "predicted_direction" in pred_df.columns:
                        pred_df["correct_direction"] = (
                            pred_df["actual_direction"] == pred_df["predicted_direction"]
                        )
                        correct_dir = (pred_df["correct_direction"]).mean() * 100
                        st.metric("Direction Accuracy", "{:.1f}%".format(correct_dir))
                else:
                    st.info("No prediction history available yet.")

        # Tab 2: Model Tuning - fix the nested tabs issue
        with main_tabs[1]:
            st.subheader("Hyperparameter Tuning")
            
            # Create two columns rather than tabs
            tuning_cols = st.columns(2)
            
            with tuning_cols[0]:
                st.markdown("### Tuning Status")
                if st.session_state.get("tuning_in_progress", False):
                    st.success("Tuning is currently in progress.")
                else:
                    st.info("No tuning in progress.")

                if "best_params" in st.session_state and st.session_state["best_params"]:
                    st.subheader("Best Parameters")
                    best_params = st.session_state["best_params"]
                    for param, value in best_params.items():
                        st.write(f"**{param}:** {value}")
                        
            with tuning_cols[1]:
                with st.expander("View All Tested Models", expanded=False):
                    # Try to display tested models
                    try:
                        # First try direct import
                        try:
                            from src.dashboard.dashboard.dashboard_model import display_tested_models
                            display_tested_models()
                        except ImportError:
                            # Then try from dashboard_modules
                            display_tested_models = dashboard_modules.get(
                                "display_tested_models",
                                lambda: st.info("Tested models display not available"),
                            )
                            display_tested_models()
                    except Exception as e:
                        logger.error(f"Error displaying tested models: {e}")
                        st.info("Tested models display not available")
            
            # Use the unified tuning panel instead of separate functions
            # This avoids nested tabs
            create_tuning_panel_func = dashboard_modules.get("create_tuning_panel")
            if create_tuning_panel_func:
                create_tuning_panel_func()
            else:
                st.error("Tuning panel not available. Check that dashboard_ui.py has been updated with the refactored code.")

        # Tab 3: Technical Analysis
        with main_tabs[2]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Technical Indicators")

            with st.spinner("Loading technical indicators..."):
                # Only render if we have data
                if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
                    # Calculate indicators on the dataset
                    df_indicators = calculate_indicators_func(st.session_state["df_raw"])

                    # Apply custom indicators if needed
                    if "indicators" in st.session_state:
                        apply_custom_indicators_func = dashboard_modules.get(
                            "apply_custom_indicators", lambda df, timeframe, indicators: df
                        )
                        df_indicators = apply_custom_indicators_func(
                            df_indicators, timeframe, st.session_state["indicators"]
                        )

                    # Display technical indicators chart
                    try:
                        if "Close" in df_indicators.columns:
                            st.line_chart(df_indicators[["Close"]])
                        else:
                            # Display what columns are available and use the first numeric column
                            numeric_cols = df_indicators.select_dtypes(include=['float64', 'int64']).columns
                            if not numeric_cols.empty:
                                st.line_chart(df_indicators[[numeric_cols[0]]])
                            else:
                                st.warning("No suitable numeric columns found for charting.")
                    except Exception as e:
                        st.error(f"Error displaying technical chart: {str(e)}")
                        st.write("Available columns:", list(df_indicators.columns))
                else:
                    st.warning("No data available for technical indicators")
            st.markdown("</div>", unsafe_allow_html=True)

            # Market Regime Analysis Section
            try:
                # Get regime information
                regime_stats = None
                if 'metadata' in st.session_state and 'regime_stats' in st.session_state['metadata']:
                    regime_stats = st.session_state['metadata']['regime_stats']

                # Show market regime information
                with st.expander("Market Regime Analysis", expanded=False):
                    show_regime_func = dashboard_modules.get(
                        "show_market_regime_info", 
                        lambda regime_stats: st.info("Market regime visualization not available")
                    )
                    show_regime_func(regime_stats)
            except Exception as e:
                logger.error(f"Error displaying market regime information: {e}")
                st.error(f"Error displaying market regime information: {e}")

            # Advanced analysis dashboard
            if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
                with st.spinner("Loading advanced analysis..."):
                    try:
                        show_advanced_dashboard_func = dashboard_modules.get(
                            "show_advanced_dashboard_tabs", 
                            lambda df: st.info("Advanced dashboard features not available")
                        )
                        show_advanced_dashboard_func(st.session_state["df_raw"])
                    except Exception as e:
                        logger.error(f"Error creating advanced analysis: {e}")
                        st.error(f"Error creating advanced analysis: {e}")
            else:
                st.warning("No data available for advanced analysis")

        # Tab 4: Price Data
        with main_tabs[3]:
            with st.spinner("Loading price data..."):
                if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
                    df_vis = st.session_state["df_raw"]

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.subheader("Data Summary")
                        st.write(f"**Ticker:** {ticker}")
                        st.write(f"**Timeframe:** {timeframe}")
                        st.write(f"**Period:** {start_date} to {end_date}")
                        st.write(f"**Data points:** {len(df_vis)}")

                        # Show key statistics
                        if len(df_vis) > 0:
                            try:
                                if "Close" in df_vis.columns:
                                    st.metric("Current Price", f"${df_vis['Close'].iloc[-1]:.2f}")
                                    if len(df_vis) > 1:
                                        change = float(df_vis["Close"].iloc[-1]) - float(df_vis["Close"].iloc[-2])
                                        pct_change = (change / float(df_vis["Close"].iloc[-2])) * 100
                                        st.metric("Last Change", f"${change:.2f}", f"{pct_change:.2f}%")
                            except Exception as e:
                                logger.error(f"Error displaying metrics: {e}")
                                st.error(f"Error displaying metrics: {e}")

                        # Download button for CSV
                        if not df_vis.empty:
                            try:
                                csv_data = df_vis.to_csv(index=False)
                                b64 = base64.b64encode(csv_data.encode()).decode()
                                download_link = '<a href="data:file/csv;base64,{}" download="{}_{}_data.csv" class="download-button">📥 Download Data</a>'.format(
                                    b64, ticker, timeframe
                                )
                                st.markdown(download_link, unsafe_allow_html=True)
                            except Exception as e:
                                logger.error(f"Error creating download link: {e}")
                                st.error(f"Error creating download link: {e}")

                    with col2:
                        st.subheader("Market Data")

                        # Prepare dataframe for display
                        try:
                            prepare_df_func = dashboard_modules.get("prepare_dataframe_for_display", lambda df: df)
                            display_df = prepare_df_func(df_vis)

                            # Check if display_df is not None and not empty before styling
                            if display_df is not None and not display_df.empty:
                                try:
                                    # Since styling can fail, wrap it in try/except
                                    # Show only a preview (first 100 rows) for performance
                                    preview_df = display_df.head(100)
                                    st.dataframe(preview_df, height=400)

                                    # Show total number of rows
                                    if len(display_df) > 100:
                                        st.info(f"Showing 100 of {len(display_df)} rows. Download the CSV for full data.")
                                except Exception as e:
                                    # Fallback if styling fails
                                    logger.error(f"Error styling dataframe: {e}")
                                    st.dataframe(display_df.head(100), height=400)
                            else:
                                st.warning("No data available to display.")
                        except Exception as e:
                            logger.error(f"Error preparing dataframe: {e}")
                            st.error(f"Error preparing dataframe: {e}")
                            # Absolute fallback - just show raw data
                            st.dataframe(df_vis.head(100))

                        # Summary Statistics
                        with st.expander("Summary Statistics"):
                            if df_vis is not None and not df_vis.empty:
                                try:
                                    numeric_cols = df_vis.select_dtypes(include=["float64", "int64"]).columns
                                    stats_df = df_vis[numeric_cols].describe().transpose()
                                    st.dataframe(stats_df)
                                except Exception as e:
                                    logger.error(f"Error displaying summary statistics: {e}")
                                    st.error(f"Error displaying summary statistics: {e}")
                            else:
                                st.info("No data available for statistics.")
                else:
                    st.warning("No data available to display.")

        # Tab 5: Ensemble Weights
        with main_tabs[4]:
            st.header("Ensemble Weights Analysis")
            
            try:
                # Try to use ModelVisualizationDashboard if available
                ModelVisualizationDashboard = dashboard_modules.get("ModelVisualizationDashboard")
                
                if ModelVisualizationDashboard:
                    # Check if we have an ensemble weighter in session state
                    if "ensemble_weighter" in st.session_state and st.session_state["ensemble_weighter"] is not None:
                        # Create visualization dashboard
                        viz_dashboard = ModelVisualizationDashboard(st.session_state["ensemble_weighter"])
                        # Render only the ensemble weights tab
                        viz_dashboard.render_ensemble_weights_tab()
                    else:
                        # Create a sample visualization with dummy data
                        st.info("No ensemble weighter data available. Showing sample visualization.")
                        # Create a placeholder for weights evolution
                        st.subheader("Ensemble Weight Evolution")
                        
                        # Generate sample data
                        sample_models = ["lstm", "rnn", "xgboost", "random_forest", "cnn"]
                        sample_weights = {model: np.random.rand() for model in sample_models}
                        normalized_weights = {k: v/sum(sample_weights.values()) for k, v in sample_weights.items()}
                        
                        # Show a pie chart of weights
                        weights_df = pd.DataFrame({
                            "Model": list(normalized_weights.keys()),
                            "Weight": list(normalized_weights.values())
                        })
                        
                        st.bar_chart(weights_df.set_index("Model"))
                else:
                    st.warning("ModelVisualizationDashboard is not available.")
            except Exception as e:
                logger.error(f"Error rendering ensemble weights tab: {e}")
                st.error(f"Error visualizing ensemble weights: {str(e)}")

        # Tab 6: Neural Architecture
        with main_tabs[5]:
            st.header("Neural Network Architecture")
            
            try:
                # Option to select model type
                model_types = ["lstm", "rnn", "xgboost", "random_forest", "cnn", "nbeats", "tft"]
                selected_model = st.selectbox("Select model architecture to view:", model_types)
                
                # Try to use ModelVisualizationDashboard if available
                ModelVisualizationDashboard = dashboard_modules.get("ModelVisualizationDashboard")
                if ModelVisualizationDashboard:
                    # Create a minimal dashboard for visualization
                    viz_dashboard = ModelVisualizationDashboard(None)
                    
                    # Call the appropriate visualization method based on selected model
                    if selected_model == "lstm":
                        viz_dashboard._visualize_lstm_architecture()
                    elif selected_model == "rnn":
                        viz_dashboard._visualize_rnn_architecture()
                    elif selected_model in ["xgboost", "random_forest"]:
                        viz_dashboard._visualize_tree_architecture(selected_model.replace("_", " ").title())
                    elif selected_model == "cnn":
                        viz_dashboard._visualize_cnn_architecture()
                    elif selected_model == "nbeats":
                        viz_dashboard._visualize_nbeats_architecture()
                    elif selected_model == "tft":
                        viz_dashboard._visualize_tft_architecture()
                    else:
                        st.info(f"Visualization not available for {selected_model}")
                else:
                    st.warning("ModelVisualizationDashboard is not available.")
            except Exception as e:
                logger.error(f"Error rendering neural architecture: {e}")
                st.error(f"Error displaying neural architecture: {str(e)}")

        # Tab 7: Learning Insights
        with main_tabs[6]:
            st.header("Learning Insights")
            
            try:
                # Try to use ModelVisualizationDashboard if available
                ModelVisualizationDashboard = dashboard_modules.get("ModelVisualizationDashboard")
                if ModelVisualizationDashboard:
                    # Create visualization dashboard with available data
                    viz_dashboard = ModelVisualizationDashboard(
                        st.session_state.get("ensemble_weighter", None)
                    )
                    
                    # Render learning insights tab
                    viz_dashboard.render_learning_insights_tab()
                else:
                    st.warning("ModelVisualizationDashboard is not available.")
            except Exception as e:
                logger.error(f"Error rendering learning insights: {e}")
                st.error(f"Error displaying learning insights: {str(e)}")
                
                # Display sample learning curves
                st.subheader("Sample Learning Curves")
                
                # Generate sample data
                timesteps = 100
                models = ["lstm", "rnn", "xgboost", "random_forest"]
                data = {
                    model: [10 * np.exp(-0.05 * x) + 0.5 + np.random.rand() * 0.5 for x in range(timesteps)]
                    for model in models
                }
                
                # Add timesteps
                data["timestep"] = list(range(timesteps))
                
                # Convert to DataFrame
                sample_df = pd.DataFrame(data)
                
                # Plot
                st.line_chart(sample_df.set_index("timestep"))

        # Tab 8: Pattern Discovery
        with main_tabs[7]:
            st.header("Pattern Discovery")
            
            if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
                df = st.session_state["df_raw"]
                
                # Add pattern discovery options
                pattern_window = st.slider("Pattern Window Size", 5, 50, 20)
                
                # Create sections for different pattern analyses
                pattern_tabs = st.tabs(["Identified Patterns", "Pattern Matching", "Seasonality"])
                
                with pattern_tabs[0]:
                    st.subheader("Identified Market Patterns")
                    
                    try:
                        # Try to use imported pattern discovery functions
                        identify_patterns_func = dashboard_modules.get("identify_patterns")
                        visualize_patterns_func = dashboard_modules.get("visualize_patterns")
                        
                        if identify_patterns_func and visualize_patterns_func:
                            # Identify patterns
                            patterns = identify_patterns_func(df, window_size=pattern_window)
                            
                            # Visualize patterns
                            visualize_patterns_func(patterns, df)
                        else:
                            st.info("Pattern discovery functions not available.")
                    except Exception as e:
                        logger.error(f"Error in pattern discovery: {e}")
                        st.error(f"Error in pattern identification: {str(e)}")
                        
                        # Sample pattern visualization
                        st.write("Sample pattern detection - proper implementation required")
                        
                        # Create a sample pattern chart
                        if "Close" in df.columns:
                            st.line_chart(df["Close"].iloc[-100:])
                
                with pattern_tabs[1]:
                    st.subheader("Pattern Matching")
                    st.info("Pattern matching shows current market conditions compared to historical patterns")
                    
                    try:
                        # Try to use imported pattern matching functions
                        find_similar_patterns_func = dashboard_modules.get("find_similar_patterns")
                        
                        if find_similar_patterns_func:
                            if "Close" in df.columns:
                                # Get current pattern
                                current_pattern = df["Close"].iloc[-pattern_window:].values
                                
                                # Find similar patterns
                                similar_patterns = find_similar_patterns_func(
                                    df, current_pattern, max_patterns=5
                                )
                                
                                # Display results
                                if similar_patterns and "patterns" in similar_patterns:
                                    st.write(f"Found {len(similar_patterns['patterns'])} similar patterns")
                                    
                                    for i, pattern in enumerate(similar_patterns["patterns"]):
                                        st.write(f"**Pattern {i+1}** - Similarity: {pattern['similarity']:.2f}")
                                        st.line_chart(pd.DataFrame({
                                            "Current": current_pattern / current_pattern[0],
                                            "Historical": pattern["values"] / pattern["values"][0]
                                        }))
                                else:
                                    st.info("No similar patterns found")
                        else:
                            st.info("Pattern matching function not available.")
                    except Exception as e:
                        logger.error(f"Error in pattern matching: {e}")
                        st.error(f"Error in pattern matching: {str(e)}")
                        
                        # Sample visualization
                        if "Close" in df.columns:
                            # Generate sample data
                            current_pattern = df["Close"].iloc[-30:].values
                            historical_pattern = current_pattern * (1 + np.random.normal(0, 0.02, len(current_pattern)))
                            
                            # Create DataFrame for display
                            pattern_df = pd.DataFrame({
                                "Current": current_pattern,
                                "Historical Similar": historical_pattern
                            })
                            
                            # Display
                            st.line_chart(pattern_df)
                
                with pattern_tabs[2]:
                    st.subheader("Seasonality Analysis")
                    st.info("Seasonality analysis identifies recurring patterns in price data")
                    
                    try:
                        # Simple seasonality analysis
                        if "Close" in df.columns and "date" in df.columns:
                            # Convert date to datetime if needed
                            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                                df["date"] = pd.to_datetime(df["date"])
                            
                            # Create components
                            df["day_of_week"] = df["date"].dt.dayofweek
                            df["month"] = df["date"].dt.month
                            
                            # Day of week analysis
                            st.subheader("Day of Week Effect")
                            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                            day_returns = df.groupby("day_of_week")["Close"].pct_change().mean() * 100
                            day_returns = day_returns.reindex(range(7))
                            
                            # Create DataFrame for plot
                            day_returns_df = pd.DataFrame({
                                "Day": [day_names[i] for i in range(7)],
                                "Avg Return %": day_returns.values
                            })
                            
                            # Plot
                            st.bar_chart(day_returns_df.set_index("Day"))
                            
                            # Monthly analysis
                            st.subheader("Monthly Effect")
                            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                            month_returns = df.groupby("month")["Close"].pct_change().mean() * 100
                            month_returns = month_returns.reindex(range(1, 13))
                            
                            # Create DataFrame for plot
                            month_returns_df = pd.DataFrame({
                                "Month": [month_names[i] for i in range(12)],
                                "Avg Return %": month_returns.values
                            })
                            
                            # Plot
                            st.bar_chart(month_returns_df.set_index("Month"))
                        else:
                            st.warning("Price data with proper date column required for seasonality analysis")
                    except Exception as e:
                        logger.error(f"Error in seasonality analysis: {e}")
                        st.error(f"Error in seasonality analysis: {str(e)}")
            else:
                st.warning("No data available for pattern discovery")

        # Tab 9: Drift Monitor
        with main_tabs[8]:
            st.header("Drift Monitoring")
            
            try:
                # Try to use imported drift visualization function
                show_drift_func = dashboard_modules.get("show_drift_visualization")
                if show_drift_func:
                    show_drift_func()
                else:
                    st.info("Drift visualization module not available")
            except Exception as e:
                logger.error(f"Error displaying drift monitor: {e}")
                st.error(f"Error displaying drift monitor: {str(e)}")
                
                # Basic fallback visualization
                st.subheader("Drift Detection")
                st.info("Drift detection monitors changes in data distribution and model performance over time")
                
                # Sample visualization
                if "df_raw" in st.session_state and st.session_state["df_raw"] is not None:
                    df = st.session_state["df_raw"]
                    if "Close" in df.columns:
                        # Generate sample drift data
                        sample_size = min(100, len(df))
                        sample_drift = np.random.uniform(0, 1, sample_size)
                        sample_dates = df["date"].iloc[-sample_size:] if "date" in df.columns else pd.date_range(end=pd.Timestamp.now(), periods=sample_size)
                        
                        # Create DataFrame for plot
                        drift_df = pd.DataFrame({
                            "Date": sample_dates,
                            "Drift Score": sample_drift
                        })
                        
                        # Plot
                        st.line_chart(drift_df.set_index("Date"))
                        
                        # Sample thresholds explanation
                        st.write("""
                        ### Drift Score Interpretation
                        
                        - **Score > 0.8**: Significant drift detected, model retraining needed
                        - **Score 0.5-0.8**: Moderate drift, monitor closely
                        - **Score < 0.5**: No significant drift detected
                        """)

        # Apply improved refresh mechanism at the end of the function
        improved_refresh_mechanism()

        # Auto-refresh logic at the end of the function - keep this for compatibility
        if params.get("auto_refresh", False):
            current_time = datetime.now().timestamp()
            if "last_refresh" in st.session_state:
                time_since_last_refresh = current_time - st.session_state["last_refresh"]
                refresh_interval = params.get("refresh_interval", 30)

                if time_since_last_refresh >= refresh_interval:
                    # Update last refresh time
                    st.session_state["last_refresh"] = current_time
                    st.experimental_rerun() 
            else:
                # Initialize last_refresh if not set
                st.session_state["last_refresh"] = current_time

        # Pass update_during_walk_forward to the session state
        if "indicators" in st.session_state:
            st.session_state["indicators"]["update_during_walk_forward"] = st.session_state.get("update_during_walk_forward", True)

        # Ensure walk-forward settings are correctly stored
        if "update_during_walk_forward" in st.session_state:
            # Add to parameters dictionary to be passed to walk_forward functions
            params["update_during_walk_forward"] = st.session_state["update_during_walk_forward"]
            
            # Also store in metadata if available
            if "metadata" in st.session_state:
                st.session_state["metadata"]["update_during_walk_forward"] = st.session_state["update_during_walk_forward"]

    except Exception as e:
        st.error(f"Critical error in main dashboard: {e}")
        logger.error(f"Critical error in main dashboard: {e}", exc_info=True)
        st.code(traceback.format_exc())


# Check if user requested the full training optimizer dashboard
if st.session_state.get("show_training_optimizer", False):
    st.title("Full Training Resource Optimizer Dashboard")
    
    try:
        from src.dashboard.training_resource_optimizer_dashboard import render_training_optimizer_tab
        render_training_optimizer_tab()
        
        # Add button to return to main dashboard
        if st.button("Return to Main Dashboard"):
            st.session_state["show_training_optimizer"] = False
            st.experimental_rerun()
    except ImportError as e:
        st.error(f"Could not load training optimizer dashboard: {e}")
        # Reset the flag
        st.session_state["show_training_optimizer"] = False


# Only run the app if this script is executed directly
if __name__ == "__main__":
    try:
        # Configure GPU on startup
        configure_gpu_for_inference()
        main_dashboard()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        st.code(traceback.format_exc())
        # Clean GPU memory on error
        try:
            clean_gpu_memory()
        except:
            pass