"""
Visualization functions for the dashboard.
Includes functions for plotting price history, feature importance, and model performance.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logger
logger = logging.getLogger("dashboard_visualization")

# Define color dictionary for consistent styling across visualizations
COLORS = {
    "candle_up": "#26a69a",  # Teal for rising candles
    "candle_down": "#ef5350",  # Red for falling candles
    "forecast": "#9c27b0",  # Purple for forecast line
    "past_pred": "#9c27b0",  # Purple for past predictions
    "actual": "#4caf50",  # Green for actual prices
    "historical": "#3f51b5",  # Purple/blue for historical price
    "volume_up": "rgba(38, 166, 154, 0.5)",  # Semi-transparent green for volume
    "volume_down": "rgba(239, 83, 80, 0.5)",  # Semi-transparent red for volume
    "ma20": "#ff9800",  # Orange for 20-day MA
    "ma50": "#2196f3",  # Blue for 50-day MA
    "ma200": "#757575",  # Gray for 200-day MA
    "bb_band": "rgba(173, 216, 230, 0.2)",  # Light blue for Bollinger Bands area
    "upper_band": "rgba(250, 120, 120, 0.7)",  # Pink-ish for upper band
    "lower_band": "rgba(250, 120, 120, 0.7)",  # Same for lower band
    "middle_band": "#ff5722",  # Deep orange for middle band
    "rsi": "#9c27b0",  # Purple for RSI
    "macd_line": "#2196f3",  # Blue for MACD line
    "macd_signal": "#f44336",  # Red for signal line
    "macd_hist_up": "#4caf50",  # Green for positive histogram
    "macd_hist_down": "#f44336",  # Red for negative histogram
    "werpi": "#9c27b0",  # Purple for WERPI indicator
    "vmli": "#00bcd4",  # Cyan for VMLI indicator
}


def robust_error_boundary(func):
    """
    Decorator to create a robust error boundary around any function.
    Catches any exceptions and handles them gracefully.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function that handles errors gracefully
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            st.error(f"Error in visualization: {str(e)}")
            return None
    return wrapper


def ensure_date_column(df: pd.DataFrame, default_name: str = "date") -> Tuple[pd.DataFrame, str]:
    """
    Ensure the DataFrame has a proper datetime date column.
    
    Args:
        df: DataFrame to process
        default_name: Default column name to use for the date column
        
    Returns:
        Tuple of (DataFrame with guaranteed date column, name of date column)
    """
    if df is None or df.empty:
        return df, default_name
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Check for date columns with various names
    date_col = None
    possible_date_cols = ["date", "Date", "timestamp", "Timestamp", "time", "Time"]
    
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break
    
    # If no date column but index is datetime, convert index to column
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        df[default_name] = df.index
        date_col = default_name
    
    # If still no date column, create a synthetic one
    if date_col is None:
        df[default_name] = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=len(df)), 
            periods=len(df)
        )
        date_col = default_name
    
    # Ensure date column is datetime type
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.warning(f"Error converting {date_col} to datetime: {e}")
        # If conversion fails, create a new date column
        df[default_name] = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=len(df)), 
            periods=len(df)
        )
        date_col = default_name
    
    return df, date_col


@robust_error_boundary
def save_best_prediction(df: pd.DataFrame, current_predictions: List[float]) -> Dict:
    """
    Save the current prediction as the 'best prediction' for each date once actual data is available.
    
    Args:
        df: DataFrame with actual price data
        current_predictions: List of predicted prices
        
    Returns:
        Updated past_predictions dictionary
    """
    # Initialize past_predictions if not already in session state
    if "past_predictions" not in st.session_state:
        st.session_state["past_predictions"] = {}
    
    # Return if df is invalid
    if df is None or df.empty:
        return st.session_state["past_predictions"]
    
    # Ensure date column is present
    df, date_col = ensure_date_column(df)
    
    # Get last date in the dataframe (represents most recent actual data point)
    last_date = pd.to_datetime(df[date_col].iloc[-1])
    
    # Get ticker and timeframe for filename
    ticker = st.session_state.get("selected_ticker", "unknown")
    timeframe = st.session_state.get("selected_timeframe", "1d")
    
    # If we have predictions and the latest datapoint is recent
    if current_predictions and len(current_predictions) > 0:
        # Get the latest prediction date (first prediction is for day after last actual data)
        pred_date = last_date + pd.Timedelta(days=1)
        
        # Store prediction for this date (first element of future_predictions)
        st.session_state["past_predictions"][pred_date.strftime("%Y-%m-%d")] = {
            "predicted": float(current_predictions[0]),
            "actual": None,  # Will be filled when actual data becomes available
        }
    
    # Update past predictions with actual values when available
    for date_str, pred_info in list(st.session_state["past_predictions"].items()):
        # Skip if actual value is already recorded
        if pred_info["actual"] is not None:
            continue
        
        # Convert string date to datetime
        pred_date = pd.to_datetime(date_str)
        
        # Check if we now have actual data for this prediction date
        actual_row = df[pd.to_datetime(df[date_col]) == pred_date]
        if not actual_row.empty:
            # We have actual data, so record it
            try:
                actual_close = float(actual_row["Close"].iloc[0])
                st.session_state["past_predictions"][date_str]["actual"] = actual_close
                
                # Calculate accuracy metrics
                predicted = st.session_state["past_predictions"][date_str]["predicted"]
                error = actual_close - predicted
                pct_error = error / actual_close * 100
                
                # Store error metrics
                st.session_state["past_predictions"][date_str]["error"] = error
                st.session_state["past_predictions"][date_str]["pct_error"] = pct_error
            except Exception as e:
                logger.error(f"Error updating prediction with actual data: {e}")
    
    # ADDED: Save predictions to disk for long-term storage
    try:
        import json
        
        # Create directory if it doesn't exist
        predictions_dir = os.path.join(project_root, "data", "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save to a ticker and timeframe specific file
        filename = f"{ticker}_{timeframe}_predictions.json"
        filepath = os.path.join(predictions_dir, filename)
        
        # Convert predictions to serializable format
        serializable_predictions = {}
        for date_str, pred_info in st.session_state["past_predictions"].items():
            serializable_predictions[date_str] = {
                k: float(v) if v is not None else None for k, v in pred_info.items()
            }
        
        # Write to file
        with open(filepath, "w") as f:
            json.dump(serializable_predictions, f, indent=2)
        
        logger.info(f"Saved {len(serializable_predictions)} predictions to {filepath}")
    except Exception as e:
        logger.error(f"Error saving predictions to disk: {e}")
    
    # Return the updated predictions
    return st.session_state["past_predictions"]


@robust_error_boundary
def load_past_predictions_from_disk() -> Dict:
    """
    Load past predictions from disk based on current ticker and timeframe.
    
    Returns:
        Dictionary of past predictions
    """
    try:
        import json
        
        ticker = st.session_state.get("selected_ticker", "unknown")
        timeframe = st.session_state.get("selected_timeframe", "1d")
        
        # Path to predictions file
        predictions_dir = os.path.join(project_root, "data", "predictions")
        filename = f"{ticker}_{timeframe}_predictions.json"
        filepath = os.path.join(predictions_dir, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.info(f"No saved predictions found for {ticker} ({timeframe})")
            return {}
        
        # Load predictions from file
        with open(filepath, "r") as f:
            predictions = json.load(f)
        
        logger.info(f"Loaded {len(predictions)} predictions from {filepath}")
        return predictions
    except Exception as e:
        logger.error(f"Error loading predictions from disk: {e}")
        return {}


@robust_error_boundary
def create_interactive_price_chart(
    df: pd.DataFrame, 
    options: Dict, 
    future_forecast: Optional[List[float]] = None, 
    indicators: Optional[Dict[str, bool]] = None, 
    height: int = 600
) -> None:
    """
    Create an interactive plotly chart with price history and forecast.
    
    Args:
        df: DataFrame with price data
        options: Options for chart display
        future_forecast: List of future price predictions
        indicators: Dictionary of indicator display flags
        height: Height of the chart in pixels
    
    Returns:
        None - Displays the chart in Streamlit
    """
    if df is None or df.empty:
        st.warning("No data available to display chart.")
        return
    
    try:
        # Import plotly here to avoid issues if it's not available
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("Plotly is required for interactive charts. Please install it with: pip install plotly")
        return
    
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # Set default indicators if none provided
    if indicators is None:
        indicators = {
            "show_ma": False,
            "show_bb": False,
            "show_rsi": False,
            "show_macd": False,
            "show_werpi": False,
            "show_vmli": False,
            "show_forecast": True,
        }
    
    # Create subplot layout with main chart and volume/indicators
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    
    # Find the date column
    df, date_col = ensure_date_column(df)
    x_values = df[date_col]
    
    # Ensure we have OHLC data
    ohlc_columns = ["Open", "High", "Low", "Close"]
    all_ohlc_present = all(col in df.columns for col in ohlc_columns)
    
    # Plot main price data
    if all_ohlc_present:
        # Use candlestick if we have full OHLC data
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    else:
        # Fall back to line chart if full OHLC data not available
        if "Close" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df["Close"],
                    mode="lines",
                    name="Price",
                    line=dict(width=2, color="blue"),
                ),
                row=1,
                col=1,
            )
        else:
            # If no price data at all, show a warning
            st.warning("Price data not available in the correct format.")
            # Try to plot whatever columns we have
            for col in df.select_dtypes(include=['float64', 'int64']).columns[:1]:  # Just use first numeric column
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=df[col],
                        mode="lines",
                        name=col,
                        line=dict(width=2, color="blue"),
                    ),
                    row=1,
                    col=1,
                )
    
    # Add volume if available
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df["Volume"],
                name="Volume",
                marker_color="rgba(100, 100, 255, 0.3)",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
    
    # Add moving averages if requested
    if indicators.get("show_ma", False):
        for period, color in zip([20, 50, 200], ["blue", "orange", "red"]):
            ma_col = f"MA{period}"
            
            # Calculate MA if not already present
            if ma_col not in df.columns:
                if "Close" in df.columns:
                    df[ma_col] = df["Close"].rolling(window=period).mean()
            
            # Only plot if we have the data
            if ma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=df[ma_col],
                        mode="lines",
                        name=f"MA{period}",
                        line=dict(width=1, color=color),
                    ),
                    row=1,
                    col=1,
                )
    
    # Add Bollinger Bands if requested
    if indicators.get("show_bb", False):
        # Check if we have all required columns
        bb_cols = ["BB_upper", "BB_middle", "BB_lower"]
        all_bb_present = all(col in df.columns for col in bb_cols)
        
        # Calculate them if not present
        if not all_bb_present and "Close" in df.columns:
            window = 20
            std_dev = 2
            df["BB_middle"] = df["Close"].rolling(window=window).mean()
            rolling_std = df["Close"].rolling(window=window).std()
            df["BB_upper"] = df["BB_middle"] + (rolling_std * std_dev)
            df["BB_lower"] = df["BB_middle"] - (rolling_std * std_dev)
            all_bb_present = True
        
        if all_bb_present:
            # Add upper band
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df["BB_upper"],
                    mode="lines",
                    name="BB Upper",
                    line=dict(width=1, color="rgba(255, 0, 0, 0.5)", dash="dash"),
                ),
                row=1,
                col=1,
            )
            
            # Add lower band with fill
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df["BB_lower"],
                    mode="lines",
                    name="BB Lower",
                    line=dict(width=1, color="rgba(0, 255, 0, 0.5)", dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(200, 200, 200, 0.2)",
                ),
                row=1,
                col=1,
            )
    
    # Add RSI if requested
    if indicators.get("show_rsi", False):
        if "RSI" not in df.columns and "Close" in df.columns:
            # Calculate RSI if not present
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
            df["RSI"] = 100 - (100 / (1 + rs))
        
        if "RSI" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df["RSI"],
                    mode="lines",
                    name="RSI",
                    line=dict(color="purple"),
                ),
                row=2,
                col=1,
            )
            
            # Add overbought/oversold lines
            fig.add_hline(
                y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1
            )
            fig.add_hline(
                y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1
            )
    
    # Add MACD if requested
    if indicators.get("show_macd", False):
        # Check if we have MACD components
        macd_cols = ["MACD", "MACD_signal"]
        all_macd_present = all(col in df.columns for col in macd_cols)
        
        # Calculate them if not present
        if not all_macd_present and "Close" in df.columns:
            # Calculate MACD components
            df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
            df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = df["EMA12"] - df["EMA26"]
            df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
            all_macd_present = True
        
        if all_macd_present:
            # Add MACD line
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df["MACD"],
                    mode="lines",
                    name="MACD",
                    line=dict(color="blue"),
                ),
                row=2,
                col=1,
            )
            
            # Add signal line
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df["MACD_signal"],
                    mode="lines",
                    name="Signal",
                    line=dict(color="red"),
                ),
                row=2,
                col=1,
            )
            
            # Add histogram if available
            if "MACD_hist" in df.columns:
                colors = ["green" if val > 0 else "red" for val in df["MACD_hist"]]
                fig.add_trace(
                    go.Bar(
                        x=x_values, 
                        y=df["MACD_hist"], 
                        name="Histogram", 
                        marker_color=colors
                    ),
                    row=2,
                    col=1,
                )
    
    # Add WERPI if requested
    if indicators.get("show_werpi", False) and "WERPI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["WERPI"],
                mode="lines",
                name="WERPI",
                line=dict(color="darkblue", width=1),
            ),
            row=2,
            col=1,
        )
    
    # Add VMLI if requested
    if indicators.get("show_vmli", False) and "VMLI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["VMLI"],
                mode="lines",
                name="VMLI",
                line=dict(color="darkgreen", width=1),
            ),
            row=2,
            col=1,
        )
    
    # Add forecast if available and requested
    if future_forecast is not None and len(future_forecast) > 0 and indicators.get("show_forecast", True):
        last_date = df[date_col].iloc[-1]
        last_price = df["Close"].iloc[-1] if "Close" in df.columns else None
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=len(future_forecast)
        )
        
        # Connect the last actual price with the forecast
        if last_price is not None:
            # Include the last actual price in the forecast line for continuity
            fig.add_trace(
                go.Scatter(
                    x=[last_date] + list(future_dates),
                    y=[last_price] + future_forecast,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="rgb(31, 119, 180)", width=3),
                ),
                row=1,
                col=1,
            )
        else:
            # If we don't have the last price, just plot the forecast
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=future_forecast,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="rgb(31, 119, 180)", width=3),
                ),
                row=1,
                col=1,
            )
    
    # Get chart title from options
    ticker = options.get("ticker", "")
    timeframe = options.get("timeframe", "")
    chart_title = f"{ticker} - {timeframe} Chart" if ticker and timeframe else "Price Chart"
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Determine appropriate title for the indicator panel
    indicator_title = ""
    if indicators.get("show_rsi", False):
        indicator_title = "RSI"
    elif indicators.get("show_macd", False):
        indicator_title = "MACD"
    elif indicators.get("show_werpi", False):
        indicator_title = "WERPI"
    elif indicators.get("show_vmli", False):
        indicator_title = "VMLI"
    elif "Volume" in df.columns:
        indicator_title = "Volume"
    
    # Update y-axis title for indicator panel
    fig.update_yaxes(title_text=indicator_title, row=2, col=1)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


@robust_error_boundary
def prepare_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a DataFrame for display in the UI with proper formatting.
    
    Args:
        df: The DataFrame to prepare
        
    Returns:
        The prepared DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    display_df = df.copy()
    
    # Find date column (if any)
    date_cols = [
        col for col in display_df.columns
        if isinstance(col, str) and "date" in col.lower()
    ]
    
    # Set date as index if present
    if date_cols:
        date_col = date_cols[0]
        # Ensure date column is datetime type
        display_df[date_col] = pd.to_datetime(display_df[date_col], errors='coerce')
        # Set as index
        display_df.set_index(date_col, inplace=True)
    
    return display_df


@robust_error_boundary
def plot_feature_importance(importance_data: Dict[str, float]) -> Optional[go.Figure]:
    """
    Plot feature importance using Plotly.
    
    Args:
        importance_data: Dictionary mapping feature names to importance values
        
    Returns:
        Plotly figure or None if error
    """
    if not importance_data:
        logger.warning("No feature importance data available to plot")
        return None
    
    try:
        import plotly.express as px
    except ImportError:
        st.error("Plotly Express is required for feature importance plots. Please install it with: pip install plotly")
        return None
    
    # Create DataFrame from importance data
    importance_df = pd.DataFrame(
        {
            "Feature": list(importance_data.keys()),
            "Importance": list(importance_data.values()),
        }
    ).sort_values("Importance", ascending=False)
    
    # Create bar chart
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale="Viridis",
    )
    
    return fig


@robust_error_boundary
def update_forecast_in_dashboard(
    ensemble_model, 
    df: pd.DataFrame, 
    feature_cols: List[str], 
    ensemble_weights: Optional[Dict[str, float]] = None
) -> Optional[List[float]]:
    """
    Update forecast in dashboard with predictions from the model.
    
    Args:
        ensemble_model: The trained ensemble model
        df: DataFrame with features
        feature_cols: List of feature column names
        ensemble_weights: Dictionary of model weights
        
    Returns:
        List of forecast values or None on error
    """
    try:
        # Get dashboard settings
        lookback = st.session_state.get("lookback", 30)
        forecast_window = st.session_state.get("forecast_window", 30)
        
        # Get the last 'lookback' days of data for input
        if len(df) < lookback:
            logger.error(f"Not enough data for forecast: have {len(df)}, need {lookback}")
            return None
            
        last_data = df.iloc[-lookback:].copy()
        
        # Generate the forecast
        # Import specialized functions if available
        try:
            # Try to import from training module
            from src.training.walk_forward import generate_future_forecast as gff
            future_forecast = gff(ensemble_model, df, feature_cols, lookback, forecast_window)
        except ImportError:
            logger.warning("Could not import generate_future_forecast from walk_forward, using fallback method")
            
            # Fallback implementation if walk_forward module not available
            future_forecast = _generate_forecast_fallback(ensemble_model, df, feature_cols, lookback, forecast_window)
        
        if future_forecast and len(future_forecast) > 0:
            # Save the forecast in session state
            st.session_state["future_forecast"] = future_forecast
            st.session_state["last_forecast_update"] = pd.Timestamp.now()
            
            # Store the ensemble weights if provided
            if ensemble_weights:
                st.session_state["ensemble_weights"] = ensemble_weights
            
            # Save the prediction to the history
            save_best_prediction(df, future_forecast)
            
            logger.info(f"Updated forecast with {len(future_forecast)} days")
            return future_forecast
        else:
            logger.warning("Generated forecast was empty")
            return None
            
    except Exception as e:
        logger.error(f"Error updating forecast: {e}", exc_info=True)
        return None


def _generate_forecast_fallback(
    model, 
    df: pd.DataFrame, 
    feature_cols: List[str], 
    lookback: int = 30, 
    horizon: int = 30
) -> List[float]:
    """
    Fallback method to generate forecasts when specialized modules are not available.
    
    Args:
        model: The trained model (ensemble or single model)
        df: DataFrame with historical data
        feature_cols: Feature columns to use
        lookback: Number of past days to use for input
        horizon: Number of days to forecast
        
    Returns:
        List of forecast values
    """
    from sklearn.preprocessing import StandardScaler
    
    # Get the last 'lookback' days of data for input
    last_data = df.iloc[-lookback:].copy()

    # Create a scaler for feature normalization
    scaler = StandardScaler()
    scaler.fit(last_data[feature_cols])

    # Initialize array to store predictions
    future_prices = []
    current_data = last_data.copy()

    # Try to get create_sequences from preprocessing
    try:
        from src.data.preprocessing import create_sequences
    except ImportError:
        # Define a minimal version if not available
        def create_sequences(data, features, target, window, horizon):
            import numpy as np
            
            X = []
            for i in range(len(data) - window):
                X.append(data[features].values[i:i+window])
            return np.array(X), None

    # Create initial input sequence
    X_input, _ = create_sequences(current_data, feature_cols, "Close", lookback, 1)

    # Generate forecasts one step at a time
    for i in range(horizon):
        try:
            preds = model.predict(X_input, verbose=0)
            # Handle different prediction shapes
            if hasattr(preds, 'shape') and len(preds.shape) > 1:
                next_price = float(preds[0][0])
            else:
                next_price = float(preds[0])
            
            future_prices.append(next_price)

            # Update input data with prediction
            next_row = current_data.iloc[-1:].copy()
            if isinstance(next_row.index[0], pd.Timestamp):
                next_row.index = [next_row.index[0] + pd.Timedelta(days=1)]
            else:
                next_row.index = [next_row.index[0] + 1]

            next_row["Close"] = next_price
            current_data = pd.concat([current_data.iloc[1:], next_row])

            # Create new sequence for next prediction
            X_input, _ = create_sequences(
                current_data, feature_cols, "Close", lookback, 1
            )
        except Exception as e:
            logger.error(f"Error in forecast iteration {i}: {e}")
            break

    return future_prices


@robust_error_boundary
def show_advanced_dashboard_tabs(df: pd.DataFrame) -> None:
    """
    Show advanced dashboard tabs with technical analysis and statistics.
    
    Args:
        df: DataFrame with price data and indicators
        
    Returns:
        None (displays in Streamlit)
    """
    if df is None or df.empty:
        st.warning("No data available for advanced analysis")
        return
        
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("Plotly is required for advanced dashboard tabs. Please install it with: pip install plotly")
        return

    # Create tabs for different analyses
    tab_names = ["Correlation Analysis", "Statistics", "Performance"]
    tabs = st.tabs(tab_names)

    # Tab 1: Correlation Analysis
    with tabs[0]:
        st.subheader("Feature Correlation Analysis")

        # Filter numeric columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

        # Create correlation matrix for selected numeric columns
        corr_matrix = df[numeric_cols].corr()

        # Plot correlation heatmap
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
        )

        # Update layout
        fig.update_layout(
            height=600, 
            template="plotly_white", 
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Provide interpretation
        st.markdown(
            """
        ### Correlation Interpretation:
        
        - **Positive correlation (blue)**: When one variable increases, the other tends to increase as well.
        - **Negative correlation (red)**: When one variable increases, the other tends to decrease.
        - **Near-zero correlation (white)**: The two variables have little to no linear relationship.
        
        Technical indicators often show correlations with price or each other. Understanding these relationships
        helps identify redundant indicators and build more robust trading strategies.
        """
        )

    # Tab 2: Statistics
    with tabs[1]:
        st.subheader("Statistical Analysis")

        # Get price and return statistics
        if "Close" in df.columns:
            returns = df["Close"].pct_change().dropna()

            # Create statistics
            stats = {
                "Mean": returns.mean(),
                "Median": returns.median(),
                "Std Dev": returns.std(),
                "Skewness": returns.skew(),
                "Kurtosis": returns.kurt(),
                "Max": returns.max(),
                "Min": returns.min(),
            }

            # Create dataframe from statistics
            stats_df = pd.DataFrame(list(stats.items()), columns=["Statistic", "Value"])

            # Display statistics
            col1, col2 = st.columns([2, 3])

            with col1:
                # Convert Value column to numeric before formatting
                stats_df["Value"] = pd.to_numeric(stats_df["Value"], errors="coerce")
                st.dataframe(stats_df.style.format({"Value": "{:.6f}"}), height=300)

            with col2:
                # Create returns histogram
                fig = px.histogram(
                    returns,
                    nbins=50,
                    title="Return Distribution",
                    labels={"value": "Daily Return", "count": "Frequency"},
                    marginal="box",
                    color_discrete_sequence=["rgba(0, 121, 255, 0.7)"],
                )

                fig.update_layout(
                    height=300,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=50, b=10),
                )

                st.plotly_chart(fig, use_container_width=True)

            # Add QQ plot to check for normality
            try:
                from scipy import stats as scipy_stats

                qq_data = scipy_stats.probplot(returns, dist="norm")

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[0][1],
                        mode="markers",
                        name="Returns",
                        marker=dict(color="rgba(0, 121, 255, 0.7)"),
                    )
                )

                # Add the line representing perfect normality
                fig.add_trace(
                    go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                        mode="lines",
                        name="Normal",
                        line=dict(color="rgba(255, 0, 0, 0.7)"),
                    )
                )

                fig.update_layout(
                    title="Normal Q-Q Plot",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    height=400,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=50, b=10),
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create Q-Q plot: {e}")

    # Tab 3: Performance
    with tabs[2]:
        st.subheader("Performance Metrics")

        if "Close" in df.columns:
            # Calculate daily returns
            df_perf = df.copy()
            df_perf["Daily_Return"] = df_perf["Close"].pct_change()

            # Calculate cumulative returns
            df_perf["Cumulative_Return"] = (1 + df_perf["Daily_Return"]).cumprod()

            # Calculate drawdowns
            df_perf["Peak"] = df_perf["Cumulative_Return"].cummax()
            df_perf["Drawdown"] = (df_perf["Cumulative_Return"] / df_perf["Peak"]) - 1

            # Calculate rolling metrics
            window = 20  # 20-day window
            df_perf["Rolling_Volatility"] = df_perf["Daily_Return"].rolling(window).std() * (252**0.5)  # Annualized
            df_perf["Rolling_Return"] = df_perf["Daily_Return"].rolling(window).mean() * 252  # Annualized

            # Create metrics cards
            col1, col2, col3, col4 = st.columns(4)

            # Filter out NaN values for metrics
            returns_no_nan = df_perf["Daily_Return"].dropna()

            with col1:
                total_return = df_perf["Cumulative_Return"].iloc[-1] - 1 if len(df_perf) > 0 else 0
                st.metric("Total Return", f"{total_return:.2%}")

            with col2:
                annual_return = returns_no_nan.mean() * 252
                st.metric("Annual Return", f"{annual_return:.2%}")

            with col3:
                annual_vol = returns_no_nan.std() * (252**0.5)
                st.metric("Annual Volatility", f"{annual_vol:.2%}")

            with col4:
                max_drawdown = df_perf["Drawdown"].min()
                st.metric("Max Drawdown", f"{max_drawdown:.2%}", delta_color="inverse")

            # Plot cumulative returns
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_perf.index if isinstance(df_perf.index, pd.DatetimeIndex) else df_perf["date"],
                    y=df_perf["Cumulative_Return"],
                    mode="lines",
                    name="Cumulative Return",
                    line=dict(color="rgba(0, 121, 255, 0.7)", width=2),
                )
            )

            fig.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Plot drawdown
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_perf.index if isinstance(df_perf.index, pd.DatetimeIndex) else df_perf["date"],
                    y=df_perf["Drawdown"],
                    mode="lines",
                    name="Drawdown",
                    line=dict(color="rgba(255, 0, 0, 0.7)", width=2),
                    fill="tozeroy",
                )
            )

            fig.update_layout(
                title="Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                height=300,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(tickformat=".1%"),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add rolling metrics chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add rolling return
            fig.add_trace(
                go.Scatter(
                    x=df_perf.index if isinstance(df_perf.index, pd.DatetimeIndex) else df_perf["date"],
                    y=df_perf["Rolling_Return"],
                    name="Rolling Return (Ann.)",
                    line=dict(color="green", width=1.5),
                ),
                secondary_y=False,
            )

            # Add rolling volatility
            fig.add_trace(
                go.Scatter(
                    x=df_perf.index if isinstance(df_perf.index, pd.DatetimeIndex) else df_perf["date"],
                    y=df_perf["Rolling_Volatility"],
                    name="Rolling Volatility (Ann.)",
                    line=dict(color="red", width=1.5),
                ),
                secondary_y=True,
            )

            # Add figure labels and formatting
            fig.update_layout(
                title="Rolling Metrics (20-Day Window)",
                template="plotly_white",
                height=300,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                hovermode="x unified",
            )

            # Set y-axes titles
            fig.update_yaxes(title_text="Return", secondary_y=False, tickformat=".1%")
            fig.update_yaxes(title_text="Volatility", secondary_y=True, tickformat=".1%")

            st.plotly_chart(fig, use_container_width=True)


@robust_error_boundary
def create_technical_indicators_chart(df: pd.DataFrame, options: Optional[Dict] = None) -> None:
    """
    Create an interactive chart with technical indicators.
    
    Args:
        df: DataFrame with price and indicator data
        options: Dictionary of chart options
        
    Returns:
        None - Displays the chart in Streamlit
    """
    if df is None or df.empty:
        st.warning("No data available to display technical indicators.")
        return
    
    try:
        # Import plotly here to avoid issues if it's not available
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("Plotly is required for interactive charts. Please install it with: pip install plotly")
        return
    
    # Extract ticker from options
    ticker = options.get("ticker", "") if options else ""
    
    # Ensure we have a date column
    df, date_col = ensure_date_column(df)
    x_values = df[date_col]
    
    # Handle different column naming patterns
    # First check for standard OHLCV columns
    standard_cols = ["Open", "High", "Low", "Close", "Volume"]
    # Then check for ticker-suffixed columns
    ticker_cols = [f"{col}_{ticker}" for col in standard_cols]
    
    # Determine which column naming pattern to use
    col_pattern = "standard"
    if all(col in df.columns for col in ticker_cols):
        col_pattern = "ticker_suffix"
    
    # Set column names based on pattern
    if col_pattern == "ticker_suffix":
        open_col = f"Open_{ticker}"
        high_col = f"High_{ticker}"
        low_col = f"Low_{ticker}"
        close_col = f"Close_{ticker}"
        volume_col = f"Volume_{ticker}"
    else:
        open_col = "Open"
        high_col = "High"
        low_col = "Low"
        close_col = "Close"
        volume_col = "Volume"
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ensure required columns exist
    ohlc_cols = [open_col, high_col, low_col, close_col]
    all_ohlc_present = all(col in df.columns for col in ohlc_cols)
    
    # Add candlestick or line chart
    if all_ohlc_present:
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name="Price",
            )
        )
    elif close_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df[close_col],
                mode="lines",
                name="Price",
                line=dict(width=2, color="blue"),
            )
        )
    else:
        st.warning("Price data not found in expected columns.")
        return
    
    # Add volume if available
    if volume_col in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df[volume_col],
                name="Volume",
                marker_color="rgba(100, 100, 255, 0.3)",
            ),
            secondary_y=True,
        )
    
    # Add moving averages
    for ma in [20, 50, 200]:
        ma_col = f"MA{ma}"
        if ma_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df[ma_col],
                    mode="lines",
                    name=f"{ma}-day MA",
                    line=dict(width=1.5),
                )
            )
    
    # Add Bollinger Bands
    bb_cols = ["BB_upper", "BB_middle", "BB_lower"]
    if all(col in df.columns for col in bb_cols):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["BB_upper"],
                mode="lines",
                name="Upper BB",
                line=dict(color="rgba(255, 0, 0, 0.5)", dash="dash"),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["BB_lower"],
                mode="lines",
                name="Lower BB",
                fill="tonexty",
                fillcolor="rgba(200, 200, 200, 0.2)",
                line=dict(color="rgba(0, 255, 0, 0.5)", dash="dash"),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["BB_middle"],
                mode="lines",
                name="Middle BB",
                line=dict(color="rgba(255, 87, 34, 0.7)"),
            )
        )
    
    # Add RSI
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple"),
            )
        )
        
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green")
    
    # Add MACD
    macd_cols = ["MACD", "MACD_signal", "MACD_hist"]
    if all(col in df.columns for col in macd_cols):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="blue"),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["MACD_signal"],
                mode="lines",
                name="MACD Signal",
                line=dict(color="red"),
            )
        )
        
        # Add histogram with color based on value
        colors = ["green" if val > 0 else "red" for val in df["MACD_hist"]]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df["MACD_hist"],
                name="MACD Histogram",
                marker_color=colors,
            )
        )
    
    # Add WERPI
    if "WERPI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["WERPI"],
                mode="lines",
                name="WERPI",
                line=dict(color="darkblue", width=1.5),
            )
        )
    
    # Add VMLI
    if "VMLI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df["VMLI"],
                mode="lines",
                name="VMLI",
                line=dict(color="darkgreen", width=1.5),
            )
        )