"""
Visualization functions for the dashboard.
Includes functions for plotting price history, feature importance, and model performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import os
import sys
from plotly.subplots import make_subplots

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utilities and configurations
try:
    from src.dashboard.dashboard.dashboard_error import robust_error_boundary
    from config.logger_config import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dashboard")
    
    def robust_error_boundary(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Error in {func.__name__}: {str(e)}")
                return None
        return wrapper

@robust_error_boundary
def save_best_prediction(df, current_predictions):
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
    if df is None or df.empty or "date" not in df.columns:
        return st.session_state["past_predictions"]
    
    # Get last date in the dataframe (represents most recent actual data point)
    last_date = pd.to_datetime(df["date"].iloc[-1])
    
    # Get ticker and timeframe for filename
    ticker = st.session_state.get("selected_ticker", "unknown")
    timeframe = st.session_state.get("selected_timeframe", "1d")
    
    # If we have predictions and the latest datapoint is recent (today or yesterday)
    if current_predictions and len(current_predictions) > 0:
        # Get the latest prediction date (first prediction is for day after last actual data)
        pred_date = last_date + pd.Timedelta(days=1)
        
        # Store prediction for this date (first element of future_predictions)
        st.session_state.past_predictions[pred_date.strftime('%Y-%m-%d')] = {
            'predicted': float(current_predictions[0]),
            'actual': None  # Will be filled when actual data becomes available
        }
    
    # Update past predictions with actual values when available
    for date_str, pred_info in list(st.session_state.past_predictions.items()):
        # Skip if actual value is already recorded
        if pred_info['actual'] is not None:
            continue
            
        # Convert string date to datetime
        pred_date = pd.to_datetime(date_str)
        
        # Check if we now have actual data for this prediction date
        actual_row = df[pd.to_datetime(df['date']) == pred_date]
        if not actual_row.empty:
            # We have actual data, so record it
            actual_close = float(actual_row['Close'].iloc[0])
            st.session_state.past_predictions[date_str]['actual'] = actual_close
            
            # Calculate accuracy metrics
            predicted = st.session_state.past_predictions[date_str]['predicted']
            error = actual_close - predicted
            pct_error = error / actual_close * 100
            
            # Store error metrics
            st.session_state.past_predictions[date_str]['error'] = error
            st.session_state.past_predictions[date_str]['pct_error'] = pct_error
    
    # ADDED: Save predictions to disk for long-term storage
    try:
        import os
        import json
        
        # Create directory if it doesn't exist
        predictions_dir = os.path.join(project_root, "data", "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save to a ticker and timeframe specific file
        filename = f"{ticker}_{timeframe}_predictions.json"
        filepath = os.path.join(predictions_dir, filename)
        
        # Convert predictions to serializable format
        serializable_predictions = {}
        for date_str, pred_info in st.session_state.past_predictions.items():
            serializable_predictions[date_str] = {
                k: float(v) if v is not None else None 
                for k, v in pred_info.items()
            }
            
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(serializable_predictions, f, indent=2)
            
        logger.info(f"Saved {len(serializable_predictions)} predictions to {filepath}")
    except Exception as e:
        logger.error(f"Error saving predictions to disk: {e}")
    
    # Return the updated predictions
    return st.session_state.past_predictions

# ADDED: New function to load predictions from disk
@robust_error_boundary
def load_past_predictions_from_disk():
    """
    Load past predictions from disk based on current ticker and timeframe.
    
    Returns:
        Dictionary of past predictions
    """
    try:
        import os
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
        with open(filepath, 'r') as f:
            predictions = json.load(f)
            
        logger.info(f"Loaded {len(predictions)} predictions from {filepath}")
        return predictions
    except Exception as e:
        logger.error(f"Error loading predictions from disk: {e}")
        return {}

@robust_error_boundary
def plot_price_history_with_predictions(df, future_predictions=None, ticker="Unknown", past_predictions=None):
    """
    Create an interactive plot of price history with future predictions.
    
    Args:
        df: DataFrame with historical price data
        future_predictions: List of predicted future prices
        ticker: Ticker symbol for the security
        past_predictions: Dictionary of past predictions with actual values
        
    Returns:
        Plotly figure
    """
    if df is None or df.empty:
        # Create empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            xaxis_title="Date",
            yaxis_title="Price",
            annotations=[dict(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="No price data available",
                showarrow=False
            )]
        )
        return fig
    
    # Convert date column to datetime if string
    if isinstance(df['date'].iloc[0], str):
        df['date'] = pd.to_datetime(df['date'])
    
    # Create the main price chart
    fig = go.Figure()
    
    # Add historical price data
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['Close'],
        name='Historical Price',
        line=dict(color='blue', width=2)
    ))
    
    # Get the last date and price from historical data
    last_date = df['date'].iloc[-1]
    last_price = df['Close'].iloc[-1]
    
    # Add past predictions if available
    if past_predictions is None and "past_predictions" in st.session_state:
        past_predictions = st.session_state.past_predictions
        
    if past_predictions:
        # Convert past predictions to dataframe for plotting
        past_pred_data = []
        for date_str, pred_info in past_predictions.items():
            if pred_info['actual'] is not None:  # Only include predictions with actual data
                past_pred_data.append({
                    'date': pd.to_datetime(date_str),
                    'predicted': pred_info['predicted'],
                    'actual': pred_info['actual']
                })
        
        if past_pred_data:
            past_df = pd.DataFrame(past_pred_data).sort_values('date')
            
            # Add past predictions line
            fig.add_trace(go.Scatter(
                x=past_df['date'],
                y=past_df['predicted'],
                name='Past Predictions',
                line=dict(color='green', width=2, dash='dash')
            ))
    
    # Add future predictions if available
    if future_predictions is not None and len(future_predictions) > 0:
        # Generate future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predictions)).tolist()
        
        # Create predictions trace
        fig.add_trace(go.Scatter(
            x=[last_date] + future_dates,  # Start from last actual price point
            y=[last_price] + future_predictions,  # Connect to last actual price
            name='Forecast',
            line=dict(color='red', width=2.5)
        ))
        
        # Add confidence intervals around predictions if available
        if "prediction_intervals" in st.session_state:
            intervals = st.session_state.prediction_intervals
            if intervals and 'lower' in intervals and 'upper' in intervals:
                lower_bound = intervals['lower']
                upper_bound = intervals['upper']
                
                # Only add if dimensions match
                if len(lower_bound) == len(future_predictions) and len(upper_bound) == len(future_predictions):
                    # Add upper bound
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=upper_bound,
                        name='Upper Bound',
                        line=dict(color='rgba(255,0,0,0.2)', width=0),
                        showlegend=False
                    ))
                    
                    # Add lower bound
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=lower_bound,
                        name='Lower Bound',
                        line=dict(color='rgba(255,0,0,0.2)', width=0),
                        fill='tonexty',  # Fill area between the traces
                        showlegend=False
                    ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price History and Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

@robust_error_boundary
def plot_feature_importance(importance_data):
    """Plot feature importance using Plotly"""
    if not importance_data:
        logger.warning("No feature importance data available to plot")
        return None
    
    # Create DataFrame from importance data
    importance_df = pd.DataFrame({
        'Feature': list(importance_data.keys()),
        'Importance': list(importance_data.values())
    }).sort_values('Importance', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    return fig

@robust_error_boundary
def plot_model_performance(metrics):
    """Plot model performance metrics"""
    if not metrics:
        logger.warning("No model metrics available to plot")
        return None
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    # Create bar chart
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Value',
        title="Model Performance Metrics",
        color='Value',
        color_continuous_scale='Plasma'
    )
    
    return fig

@robust_error_boundary
def generate_correlation_heatmap(df):
    """Generate correlation heatmap using Plotly"""
    if df is None or df.empty:
        logger.warning("No data available to generate correlation heatmap")
        return None
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig

def ensure_date_column(df):
    """
    Ensure the DataFrame has a proper date column.
    
    Args:
        df: DataFrame with historical price data
        
    Returns:
        Tuple of DataFrame and date column name
    """
    date_col = 'date'
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column")
    
    df[date_col] = pd.to_datetime(df[date_col])
    return df, date_col

@robust_error_boundary
def create_interactive_price_chart(df, options, future_forecast=None, indicators=None, height=600):
    """
    Create an interactive plotly chart with price history and forecast.
    
    Args:
        df: DataFrame with price data
        options: Options for chart display
        future_forecast: List of future price predictions
        indicators: Dictionary of indicator display flags
        height: Height of the chart in pixels
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Ensure we have valid data
    if df is None or df.empty:
        st.warning("No data available to display chart.")
        return
    
    # Create a copy to avoid modifying the original
    df = df.copy()
        
    # Set default indicators if None provided
    if indicators is None:
        indicators = {
            "show_ma": False,
            "show_bb": False,
            "show_rsi": False,
            "show_macd": False,
            "show_werpi": False,
            "show_vmli": False,
            "show_forecast": True
        }
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Find the date column
    date_col = None
    for col in ['date', 'Date']:
        if col in df.columns:
            date_col = col
            break
    
    # If date column exists, make sure it's datetime
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        x_values = df[date_col]
    # If no date column but index is DatetimeIndex, use index
    elif isinstance(df.index, pd.DatetimeIndex):
        x_values = df.index
    # Otherwise reset index to create a simple sequence
    else:
        # Create a sequence of dates if no date column exists
        df['_date'] = pd.date_range(start='2000-01-01', periods=len(df))
        date_col = '_date'
        x_values = df[date_col]
    
    # Candlestick chart - use x_values instead of string 'date'
    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price",
        showlegend=True
    ), row=1, col=1)
    
    # Add volume as a bar chart on the secondary y-axis
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=x_values,
            y=df['Volume'],
            name="Volume",
            marker_color='rgba(100, 100, 255, 0.3)',
            showlegend=True
        ), row=2, col=1)
    
    # Add indicators based on flag settings
    
    # Moving Averages
    if indicators.get("show_ma", False):
        for period, color in zip([20, 50, 200], ['blue', 'orange', 'red']):
            ma_col = f"MA{period}" if f"MA{period}" in df.columns else None
            
            # Calculate MA if not in dataframe
            if ma_col is None:
                df[f"MA{period}"] = df["Close"].rolling(window=period).mean()
                ma_col = f"MA{period}"
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=df[ma_col],
                mode='lines',
                name=f"MA{period}",
                line=dict(width=1, color=color)
            ), row=1, col=1)
    
    # Bollinger Bands
    if indicators.get("show_bb", False) and all(col in df.columns for col in ["BB_upper", "BB_middle", "BB_lower"]):
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["BB_upper"],
            mode='lines',
            name="BB Upper",
            line=dict(width=1, color='rgba(255, 0, 0, 0.5)', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["BB_lower"],
            mode='lines',
            name="BB Lower",
            line=dict(width=1, color='rgba(0, 255, 0, 0.5)', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.2)'
        ), row=1, col=1)
    
    # RSI in separate subplot
    if indicators.get("show_rsi", False) and "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["RSI"],
            mode='lines',
            name="RSI",
            line=dict(color='purple')
        ), row=2, col=1)
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
        
    # MACD
    if indicators.get("show_macd", False):
        if all(col in df.columns for col in ["MACD", "MACD_signal"]):
            fig.add_trace(go.Scatter(
                x=x_values,
                y=df["MACD"],
                mode='lines',
                name="MACD",
                line=dict(color='blue')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=df["MACD_signal"],
                mode='lines',
                name="Signal",
                line=dict(color='red')
            ), row=2, col=1)
        
        if "MACD_hist" in df.columns:
            colors = ["green" if val > 0 else "red" for val in df["MACD_hist"]]
            fig.add_trace(go.Bar(
                x=x_values,
                y=df["MACD_hist"],
                name="Histogram",
                marker_color=colors
            ), row=2, col=1)
    
    # Custom indicators
    if indicators.get("show_werpi", False) and "WERPI" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["WERPI"],
            mode='lines',
            name="WERPI",
            line=dict(color='darkblue', width=1)
        ), row=2, col=1)
        
    if indicators.get("show_vmli", False) and "VMLI" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["VMLI"],
            mode='lines',
            name="VMLI",
            line=dict(color='darkgreen', width=1)
        ), row=2, col=1)
    
    # Add future forecast if available
    if future_forecast is not None and indicators.get("show_forecast", True):
        # Get the last date in our dataframe
        if date_col:
            last_date = df[date_col].iloc[-1]
        else:
            last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp('today')
        
        # Generate dates for forecast
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=len(future_forecast)
        )
        
        # Create forecast trace with confidence intervals
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='rgb(31, 119, 180)', width=3)
        ), row=1, col=1)
    
    # Update layout for better appearance
    fig.update_layout(
        title=f"{options['ticker']} - {options['timeframe']} Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=height,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Set axis titles for second row
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
        
    fig.update_yaxes(title_text=indicator_title, row=2, col=1)
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

@robust_error_boundary
def create_forecast_comparison_chart(historical_data, forecast_data, past_predictions=None, height=400):
    """
    Create a dedicated chart comparing forecast vs actual prices for better visualization.
    
    Args:
        historical_data: DataFrame with historical price data
        forecast_data: DataFrame or list with forecast values
        past_predictions: Optional dictionary of past predictions
        height: Height of the chart
        
    Returns:
        None (displays chart in Streamlit)
    """
    if historical_data is None or historical_data.empty:
        return
    
    # Ensure date columns are datetime
    historical_data, hist_date_col = ensure_date_column(historical_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price data as a line
    fig.add_trace(go.Scatter(
        x=historical_data[hist_date_col],
        y=historical_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color=COLORS["actual"], width=2)
    ))
    
    # Add past predictions if available
    if past_predictions is not None and past_predictions:
        past_pred_data = []
        for date_str, pred_info in past_predictions.items():
            if pred_info['actual'] is not None:
                past_pred_data.append({
                    'date': pd.to_datetime(date_str),
                    'predicted': pred_info['predicted'],
                    'actual': pred_info['actual']
                })
        
        if past_pred_data:
            past_df = pd.DataFrame(past_pred_data).sort_values('date')
            
            fig.add_trace(go.Scatter(
                x=past_df['date'],
                y=past_df['predicted'],
                mode='lines+markers',
                name='Past Predictions',
                line=dict(color=COLORS["past_pred"], width=2, dash='dash'),
                marker=dict(size=8)
            ))
    
    # Add forecast data
    if forecast_data is not None:
        # Handle forecast data as DataFrame or list
        if isinstance(forecast_data, pd.DataFrame):
            forecast_data, forecast_date_col = ensure_date_column(forecast_data)
            forecast_values = forecast_data['Forecast'] if 'Forecast' in forecast_data.columns else forecast_data.iloc[:, 1]
            forecast_dates = forecast_data[forecast_date_col]
        else:
            # Handle as list
            last_date = historical_data[hist_date_col].iloc[-1]
            last_price = historical_data['Close'].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_data)).tolist()
            forecast_values = forecast_data
            
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=[historical_data[hist_date_col].iloc[-1]] + list(forecast_dates),
            y=[historical_data['Close'].iloc[-1]] + list(forecast_values),
            mode='lines+markers',
            name='Forecast',
            line=dict(color=COLORS["forecast"], width=3),
            marker=dict(size=8, symbol='circle')
        ))
    
    # Update layout
    fig.update_layout(
        title="Price Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        height=height,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=False))
    )
    
    st.plotly_chart(fig, use_container_width=True)


@robust_error_boundary
def create_model_performance_card(model_metrics=None):
    """
    Create a card displaying model performance metrics.
    
    Args:
        model_metrics: Dictionary with model performance metrics
        
    Returns:
        None (displays metrics in Streamlit)
    """
    if model_metrics is None:
        model_metrics = {
            "rmse": st.session_state.get("model_rmse", None),
            "mape": st.session_state.get("model_mape", None),
            "r2": st.session_state.get("model_r2", None)
        }
    
    st.subheader("Model Performance")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rmse = model_metrics.get("rmse")
        st.metric(
            "RMSE",
            f"{rmse:.4f}" if rmse else "N/A",
            help="Root Mean Squared Error - lower is better"
        )
    
    with col2:
        mape = model_metrics.get("mape")
        st.metric(
            "MAPE",
            f"{mape:.2f}%" if mape else "N/A",
            help="Mean Absolute Percentage Error - lower is better"
        )
    
    with col3:
        r2 = model_metrics.get("r2")
        st.metric(
            "R²",
            f"{r2:.4f}" if r2 else "N/A",
            help="Coefficient of Determination - higher is better"
        )


@robust_error_boundary
def create_prediction_summary_widget(historical_data, forecast_data):
    """
    Create a widget with prediction summary statistics.
    
    Args:
        historical_data: DataFrame with historical price data
        forecast_data: DataFrame or list with forecast values
        
    Returns:
        None (displays stats in Streamlit)
    """
    if historical_data is None or historical_data.empty or forecast_data is None:
        return
    
    # Get the last price from historical data
    last_price = historical_data['Close'].iloc[-1]
    
    # Get the forecast prices (handle different formats)
    if isinstance(forecast_data, pd.DataFrame):
        forecast_prices = forecast_data['Forecast'].values if 'Forecast' in forecast_data.columns else forecast_data.iloc[:, 1].values
    else:
        forecast_prices = forecast_data
    
    # Calculate prediction statistics
    max_price = max(forecast_prices)
    min_price = min(forecast_prices)
    avg_price = sum(forecast_prices) / len(forecast_prices)
    end_price = forecast_prices[-1]
    
    # Calculate price changes
    max_change_pct = ((max_price / last_price) - 1) * 100
    min_change_pct = ((min_price / last_price) - 1) * 100
    avg_change_pct = ((avg_price / last_price) - 1) * 100
    end_change_pct = ((end_price / last_price) - 1) * 100
    
    # Create card
    st.subheader("Forecast Summary")
    
    # Display last known price
    st.metric(
        "Current Price",
        f"${last_price:.2f}",
    )
    
    # Display forecast metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Max Forecast",
            f"${max_price:.2f}",
            f"{max_change_pct:+.2f}%",
            delta_color="normal" 
        )
    
    with col2:
        st.metric(
            "Min Forecast",
            f"${min_price:.2f}",
            f"{min_change_pct:+.2f}%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Avg Forecast",
            f"${avg_price:.2f}",
            f"{avg_change_pct:+.2f}%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Final Forecast",
            f"${end_price:.2f}",
            f"{end_change_pct:+.2f}%",
            delta_color="normal"
        )


@robust_error_boundary
def render_html_table(data, title=None, max_rows=10):
    """Render a nice HTML table for displaying data in Streamlit."""
    html_table = """
    <style>
        .styled-table {
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            width: 100%;
        }
        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: thin solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #009879;
        }
    </style>
    """
    
    if title:
        html_table += f"<h3>{title}</h3>"
    
    html_table += "<table class='styled-table'><thead><tr>"
    
    # Add headers
    for col in data.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr></thead><tbody>"
    
    # Add rows (limited to max_rows)
    for i, row in data.head(max_rows).iterrows():
        html_table += "<tr>"
        for col in data.columns:
            value = row[col]
            if isinstance(value, float):
                html_table += f"<td>{value:.4f}</td>"
            else:
                html_table += f"<td>{value}</td>"
        html_table += "</tr>"
    
    html_table += "</tbody></table>"
    
    if len(data) > max_rows:
        html_table += f"<p><em>Showing {max_rows} of {len(data)} rows</em></p>"
    
    st.markdown(html_table, unsafe_allow_html=True)

@robust_error_boundary
def update_forecast_in_dashboard(ensemble_model, df, feature_cols, ensemble_weights=None):
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
        import streamlit as st
        from datetime import datetime
        
        # Get lookback and forecast window from session state or use defaults
        lookback = st.session_state.get("lookback", 30)
        forecast_window = st.session_state.get("forecast_window", 30)
        
        # Try to import generate_future_forecast from walk_forward
        try:
            from src.training.walk_forward import generate_future_forecast
            
            # Generate forecast with the ensemble model
            future_forecast = generate_future_forecast(
                ensemble_model, df, feature_cols, lookback, forecast_window
            )
            
            # Update session state
            st.session_state["future_forecast"] = future_forecast
            st.session_state["last_forecast_update"] = datetime.now()
            
            # Also store the ensemble weights if provided
            if ensemble_weights:
                st.session_state["ensemble_weights"] = ensemble_weights
            
            # Save the prediction to the history
            save_best_prediction(df, future_forecast)
            
            logger.info(f"Updated dashboard forecast with {len(future_forecast)} days")
            return future_forecast
            
        except ImportError:
            # Fall back to using prediction service if available
            try:
                from src.dashboard.prediction_service import PredictionService
                
                # Create prediction service with the model
                prediction_service = PredictionService(model_instance=ensemble_model)
                
                # Generate forecast
                future_forecast = prediction_service.generate_forecast(
                    df, feature_cols, lookback, forecast_window
                )
                
                # Update session state
                st.session_state["future_forecast"] = future_forecast
                st.session_state["last_forecast_update"] = datetime.now()
                
                # Store ensemble weights
                if ensemble_weights:
                    st.session_state["ensemble_weights"] = ensemble_weights
                
                # Save the prediction to the history
                save_best_prediction(df, future_forecast)
                
                logger.info(f"Updated dashboard forecast with {len(future_forecast)} days using prediction service")
                return future_forecast
                
            except ImportError:
                logger.error("Could not import either walk_forward or prediction_service")
                return None
            
    except Exception as e:
        logger.error(f"Error updating forecast in dashboard: {e}", exc_info=True)
        return None

# Add missing Color dictionary
COLORS = {
    "candle_up": "#26a69a",      # Teal for rising candles
    "candle_down": "#ef5350",    # Red for falling candles
    "forecast": "#9c27b0",       # Purple for forecast line
    "past_pred": "#9c27b0",      # Purple for past predictions
    "actual": "#4caf50",         # Green for actual prices
    "historical": "#3f51b5",     # Purple/blue for historical price
    "volume_up": "rgba(38, 166, 154, 0.5)",  # Semi-transparent green for volume
    "volume_down": "rgba(239, 83, 80, 0.5)", # Semi-transparent red for volume
    "ma20": "#ff9800",           # Orange for 20-day MA
    "ma50": "#2196f3",           # Blue for 50-day MA
    "ma200": "#757575",          # Gray for 200-day MA
    "bb_band": "rgba(173, 216, 230, 0.2)",  # Light blue for Bollinger Bands area
    "upper_band": "rgba(250, 120, 120, 0.7)",  # Pink-ish for upper band
    "lower_band": "rgba(250, 120, 120, 0.7)",  # Same for lower band
    "middle_band": "#ff5722",    # Deep orange for middle band
    "rsi": "#9c27b0",            # Purple for RSI
    "macd_line": "#2196f3",      # Blue for MACD line
    "macd_signal": "#f44336",    # Red for signal line
    "macd_hist_up": "#4caf50",   # Green for positive histogram
    "macd_hist_down": "#f44336", # Red for negative histogram
    "werpi": "#9c27b0",          # Purple for WERPI indicator
    "vmli": "#00bcd4"            # Cyan for VMLI indicator
}

@robust_error_boundary
def create_technical_indicators_chart(df, params):
    """
    Create technical indicators chart with interactive visualization.
    
    Args:
        df: DataFrame with price and indicator data
        params: Chart parameters
        
    Returns:
        None (displays chart in Streamlit)
    """
    if df is None or df.empty:
        st.warning("No data available to display technical indicators")
        return
    
    # Ensure date column is properly formatted
    df, date_col = ensure_date_column(df)
    
    # Set up tabs for different indicator groups
    tab_names = ["Moving Averages", "Oscillators", "Volume", "Custom"]
    tabs = st.tabs(tab_names)
    
    # Tab 1: Moving Averages
    with tabs[0]:
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df[date_col],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing=dict(line=dict(color=COLORS["candle_up"]), fillcolor=COLORS["candle_up"]),
            decreasing=dict(line=dict(color=COLORS["candle_down"]), fillcolor=COLORS["candle_down"])
        ))
        
        # Add Moving Averages
        if 'MA20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['MA20'], 
                name='20-day MA',
                line=dict(color=COLORS["ma20"], width=1.5)
            ))
        
        if 'MA50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['MA50'], 
                name='50-day MA',
                line=dict(color=COLORS["ma50"], width=1.5)
            ))
        
        if 'MA200' in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['MA200'], 
                name='200-day MA',
                line=dict(color=COLORS["ma200"], width=1.5)
            ))
        
        # Add Bollinger Bands
        if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['BB_upper'], 
                name='Upper BB',
                line=dict(color=COLORS["upper_band"], width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['BB_lower'], 
                name='Lower BB',
                fill='tonexty',  # Fill area between upper and lower bands
                fillcolor=COLORS["bb_band"],
                line=dict(color=COLORS["lower_band"], width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['BB_middle'], 
                name='Middle BB',
                line=dict(color=COLORS["middle_band"], width=1)
            ))
        
        # Update layout
        fig.update_layout(
            title="Moving Averages & Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Oscillators
    with tabs[1]:
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=["Price", "Oscillators"]
        )
        
        # Add candlestick chart to first row
        fig.add_trace(go.Candlestick(
            x=df[date_col],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing=dict(line=dict(color=COLORS["candle_up"]), fillcolor=COLORS["candle_up"]),
            decreasing=dict(line=dict(color=COLORS["candle_down"]), fillcolor=COLORS["candle_down"])
        ), row=1, col=1)
        
        # Add RSI to second row
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['RSI'], 
                name='RSI',
                line=dict(color=COLORS["rsi"], width=1.5)
            ), row=2, col=1)
            
            # Add overbought/oversold lines
            fig.add_shape(
                type='line',
                x0=df[date_col].iloc[0],
                x1=df[date_col].iloc[-1],
                y0=70, y1=70,
                line=dict(color='red', width=1, dash='dash'),
                row=2, col=1
            )
            
            fig.add_shape(
                type='line',
                x0=df[date_col].iloc[0],
                x1=df[date_col].iloc[-1],
                y0=30, y1=30,
                line=dict(color='green', width=1, dash='dash'),
                row=2, col=1
            )
            
            # Add annotations for overbought/oversold
            fig.add_annotation(
                text="Overbought",
                x=df[date_col].iloc[0],
                y=70,
                xanchor='left',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color="red"),
                row=2, col=1
            )
            
            fig.add_annotation(
                text="Oversold",
                x=df[date_col].iloc[0],
                y=30,
                xanchor='left',
                yanchor='top',
                showarrow=False,
                font=dict(size=10, color="green"),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Oscillators - RSI",
            xaxis_title="Date",
            yaxis2_title="RSI",
            yaxis_title="Price",
            height=700,
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set y-axis range for RSI
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Volume Analysis
    with tabs[2]:
        if 'Volume' in df.columns:
            fig = make_subplots(
                rows=2, 
                cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=["Price", "Volume"]
            )
            
            # Add candlestick chart to first row
            fig.add_trace(go.Candlestick(
                x=df[date_col],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing=dict(line=dict(color=COLORS["candle_up"]), fillcolor=COLORS["candle_up"]),
                decreasing=dict(line=dict(color=COLORS["candle_down"]), fillcolor=COLORS["candle_down"])
            ), row=1, col=1)
            
            # Add volume bars to second row with colors based on price movement
            colors = [COLORS["volume_up"] if df['Close'].iloc[i] > df['Close'].iloc[i-1] 
                     else COLORS["volume_down"] for i in range(1, len(df))]
            colors.insert(0, "gray")  # Add color for first bar
            
            fig.add_trace(go.Bar(
                x=df[date_col],
                y=df['Volume'],
                name='Volume',
                marker=dict(color=colors)
            ), row=2, col=1)
            
            # Add volume moving average if available
            if 'Volume_MA' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df['Volume_MA'],
                    name='Volume MA (20)',
                    line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
                ), row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title="Volume Analysis",
                xaxis_title="Date",
                yaxis_title="Price",
                yaxis2_title="Volume",
                height=700,
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Volume data is not available")
    
    # Tab 4: Custom Indicators
    with tabs[3]:
        # Create multi-plot figure for custom indicators
        custom_indicators = []
        
        if 'WERPI' in df.columns:
            custom_indicators.append(('WERPI', 'werpi'))
        
        if 'VMLI' in df.columns:
            custom_indicators.append(('VMLI', 'vmli'))
        
        if custom_indicators:
            fig = make_subplots(
                rows=len(custom_indicators) + 1, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6] + [0.4/len(custom_indicators)] * len(custom_indicators),
                subplot_titles=["Price"] + [ind[0] for ind in custom_indicators]
            )
            
            # Add candlestick chart to first row
            fig.add_trace(go.Candlestick(
                x=df[date_col],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing=dict(line=dict(color=COLORS["candle_up"]), fillcolor=COLORS["candle_up"]),
                decreasing=dict(line=dict(color=COLORS["candle_down"]), fillcolor=COLORS["candle_down"])
            ), row=1, col=1)
            
            # Add each custom indicator on its own row
            for i, (indicator_name, color_key) in enumerate(custom_indicators):
                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[indicator_name],
                    name=indicator_name,
                    line=dict(color=COLORS[color_key], width=1.5)
                ), row=i+2, col=1)
            
            # Update layout
            fig.update_layout(
                title="Custom Technical Indicators",
                xaxis_title="Date",
                yaxis_title="Price",
                height=800,
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No custom indicators available")


@robust_error_boundary
def prepare_dataframe_for_display(df):
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
    
    # Set date as index if present
    date_cols = [col for col in display_df.columns if 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        display_df.set_index(date_col, inplace=True)
    
    return display_df


@robust_error_boundary
def show_advanced_dashboard_tabs(df):
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
    
    # Create tabs for different analyses
    tab_names = ["Correlation Analysis", "Statistics", "Performance"]
    tabs = st.tabs(tab_names)
    
    # Tab 1: Correlation Analysis
    with tabs[0]:
        st.subheader("Feature Correlation Analysis")
        
        # Filter numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Create correlation matrix for selected numeric columns
        corr_matrix = df[numeric_cols].corr()
        
        # Plot correlation heatmap
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Heatmap"
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Provide interpretation
        st.markdown("""
        ### Correlation Interpretation:
        
        - **Positive correlation (blue)**: When one variable increases, the other tends to increase as well.
        - **Negative correlation (red)**: When one variable increases, the other tends to decrease.
        - **Near-zero correlation (white)**: The two variables have little to no linear relationship.
        
        Technical indicators often show correlations with price or each other. Understanding these relationships
        helps identify redundant indicators and build more robust trading strategies.
        """)
    
    # Tab 2: Statistics
    with tabs[1]:
        st.subheader("Statistical Analysis")
        
        # Get price and return statistics
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().dropna()
            
            # Create statistics
            stats = {
                "Mean": returns.mean(),
                "Median": returns.median(),
                "Std Dev": returns.std(),
                "Skewness": returns.skew(),
                "Kurtosis": returns.kurt(),
                "Max": returns.max(),
                "Min": returns.min()
            }
            
            # Create dataframe from statistics
            stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
            
            # Display statistics
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.dataframe(stats_df.style.format({"Value": "{:.6f}"}), height=300)
            
            with col2:
                # Create returns histogram
                fig = px.histogram(
                    returns, 
                    nbins=50,
                    title="Return Distribution",
                    labels={"value": "Daily Return", "count": "Frequency"},
                    marginal="box",
                    color_discrete_sequence=['rgba(0, 121, 255, 0.7)']
                )
                
                fig.update_layout(
                    height=300,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add QQ plot to check for normality
            from scipy import stats as scipy_stats
            
            qq_data = scipy_stats.probplot(returns, dist="norm")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Returns',
                marker=dict(color='rgba(0, 121, 255, 0.7)')
            ))
            
            # Add the line representing perfect normality
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                mode='lines',
                name='Normal',
                line=dict(color='rgba(255, 0, 0, 0.7)')
            ))
            
            fig.update_layout(
                title="Normal Q-Q Plot",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=400,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Performance
    with tabs[2]:
        st.subheader("Performance Metrics")
        
        if 'Close' in df.columns:
            # Calculate daily returns
            df_perf = df.copy()
            df_perf['Daily_Return'] = df_perf['Close'].pct_change()
            
            # Calculate cumulative returns
            df_perf['Cumulative_Return'] = (1 + df_perf['Daily_Return']).cumprod()
            
            # Calculate drawdowns
            df_perf['Peak'] = df_perf['Cumulative_Return'].cummax()
            df_perf['Drawdown'] = (df_perf['Cumulative_Return'] / df_perf['Peak']) - 1
            
            # Calculate rolling metrics
            window = 20  # 20-day window
            df_perf['Rolling_Volatility'] = df_perf['Daily_Return'].rolling(window).std() * (252 ** 0.5)  # Annualized
            df_perf['Rolling_Return'] = df_perf['Daily_Return'].rolling(window).mean() * 252  # Annualized
            
            # Create metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            # Filter out NaN values for metrics
            returns_no_nan = df_perf['Daily_Return'].dropna()
            
            with col1:
                total_return = df_perf['Cumulative_Return'].iloc[-1] - 1 if len(df_perf) > 0 else 0
                st.metric("Total Return", f"{total_return:.2%}")
            
            with col2:
                annual_return = returns_no_nan.mean() * 252
                st.metric("Annual Return", f"{annual_return:.2%}")
            
            with col3:
                annual_vol = returns_no_nan.std() * (252 ** 0.5)
                st.metric("Annual Volatility", f"{annual_vol:.2%}")
            
            with col4:
                max_drawdown = df_perf['Drawdown'].min()
                st.metric("Max Drawdown", f"{max_drawdown:.2%}", delta_color="inverse")
            
            # Plot cumulative returns
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_perf.index,
                y=df_perf['Cumulative_Return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color='rgba(0, 121, 255, 0.7)', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot drawdown
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_perf.index,
                y=df_perf['Drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='rgba(255, 0, 0, 0.7)', width=2),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title="Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                height=300,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(tickformat=".1%")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add rolling metrics
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add rolling return
            fig.add_trace(
                go.Scatter(
                    x=df_perf.index,
                    y=df_perf['Rolling_Return'],
                    name="Rolling Return (Ann.)",
                    line=dict(color='green', width=1.5)
                ),
                secondary_y=False,
            )
            
            # Add rolling volatility
            fig.add_trace(
                go.Scatter(
                    x=df_perf.index,
                    y=df_perf['Rolling_Volatility'],
                    name="Rolling Volatility (Ann.)",
                    line=dict(color='red', width=1.5)
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
                hovermode="x unified"
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Return", secondary_y=False, tickformat=".1%")
            fig.update_yaxes(title_text="Volatility", secondary_y=True, tickformat=".1%")
            
            st.plotly_chart(fig, use_container_width=True)

