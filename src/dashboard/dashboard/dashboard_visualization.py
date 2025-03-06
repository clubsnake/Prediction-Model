"""
dashboard_visualization.py

Functions for creating various visualizations in the dashboard.
"""

import os
import sys
import io
from datetime import datetime, timedelta

# Add project root to Python path
current_file = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file)
dashboard_parent = os.path.dirname(dashboard_dir)
src_dir = os.path.dirname(dashboard_parent)
project_root = os.path.dirname(src_dir)

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from dashboard_error import robust_error_boundary
    from dashboard_data import calculate_indicators
    from config.logger_config import logger
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dashboard")
    logger.warning(f"Error importing modules: {e}")
    
    # Define error boundary if not available
    def robust_error_boundary(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Error in {func.__name__}: {str(e)}")
                return None
        return wrapper


@robust_error_boundary
def prepare_dataframe_for_display(display_df):
    """
    Prepare a dataframe for display by ensuring datetime columns are properly formatted.
    
    Args:
        display_df: DataFrame to prepare for display
    """
    if display_df is None or display_df.empty:
        return display_df
        
    # Make a copy to avoid modifying the original
    display_df = display_df.copy()
    
    # Handle date column if it exists
    if 'date' in display_df.columns:
        try:
            # Convert to datetime in case it's mixed types
            display_df['date'] = pd.to_datetime(display_df['date'])
            # Format as string in a format that's safe for display
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            # If conversion fails, just convert to string
            display_df['date'] = display_df['date'].astype(str)
    
    # Do the same for Date column (capitalized version)
    if 'Date' in display_df.columns:
        try:
            display_df['Date'] = pd.to_datetime(display_df['Date'])
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            display_df['Date'] = display_df['Date'].astype(str)
    
    # Check for any other object columns that might cause issues
    for col in display_df.select_dtypes(include=['object']).columns:
        # Convert object columns to string
        display_df[col] = display_df[col].astype(str)
    
    return display_df


@robust_error_boundary
def create_interactive_price_chart(df, params, future_forecast=None, indicators=None, height=600):
    """
    Create an interactive price chart with technical indicators and future forecast.
    
    Args:
        df: DataFrame with price data and indicators
        params: Dictionary of parameters from control panel
        predictions_log: Log of previous predictions (optional)
        future_forecast: List of future price predictions (optional)
        indicators: Dictionary of indicator display flags
        height: Height of the chart in pixels
    """
    try:
        # Use DataFrame's index name or default to 'Date'
        date_col = df.index.name if df.index.name else 'Date'
        
        # Check if date is in columns instead of being the index
        if 'date' in df.columns:
            date_col = 'date'
        elif 'Date' in df.columns:
            date_col = 'Date'
            
        # Ensure we have the date column or index before proceeding
        if date_col in df.columns:
            df = df.dropna(subset=[date_col])
        else:
            # If date is the index, just drop any NaN indices
            df = df.dropna()
        
        # Validate inputs
        if df is None or df.empty:
            st.warning("No data available for chart visualization.")
            return
        
        # Use default indicators if none provided
        if indicators is None:
            indicators = {
                "show_ma": True,
                "show_bb": False,
                "show_rsi": True,
                "show_macd": True,
                "show_werpi": False,
                "show_vmli": False,
                "show_forecast": True
            }
            
        # Make sure future_forecast is a list or None
        if future_forecast is not None and not isinstance(future_forecast, list):
            logger.warning(f"Invalid future_forecast type: {type(future_forecast)}, converting to list")
            try:
                future_forecast = list(future_forecast)
            except:
                future_forecast = []
                
        # Initialize the chart
        fig = go.Figure()
        
        # Handle date column - ensure it's properly formatted
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Ensure df has a date column we can work with
        if 'date' not in df.columns and 'Date' not in df.columns:
            # Try to use index as date
            if isinstance(df.index, pd.DatetimeIndex):
                df['date'] = df.index
            else:
                # Create a synthetic date column
                df['date'] = pd.date_range(start=datetime.now() - timedelta(days=len(df)), periods=len(df))
                
        # Use either 'date' or 'Date' column
        date_col = 'date' if 'date' in df.columns else 'Date'
        
        # Ensure date column is datetime type
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            logger.warning(f"Error converting {date_col} to datetime: {e}")
            # Create a new date column as fallback
            df['chart_date'] = pd.date_range(start=datetime.now() - timedelta(days=len(df)), periods=len(df))
            date_col = 'chart_date'
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df[date_col],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Add moving averages if enabled
        if indicators.get("show_ma", True):
            if 'MA20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_col], 
                    y=df['MA20'],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='MA20'
                ))
                
            if 'MA50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_col], 
                    y=df['MA50'],
                    mode='lines',
                    line=dict(color='orange', width=1),
                    name='MA50'
                ))
            
        # Add Bollinger Bands if enabled
        if indicators.get("show_bb", False) and all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['BB_upper'],
                mode='lines',
                line=dict(color='rgba(0,128,255,0.3)', width=1),
                name='BB Upper'
            ))
            
            fig.add_trace(go.Scatter(
                x=df[date_col], 
                y=df['BB_lower'],
                mode='lines',
                line=dict(color='rgba(0,128,255,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0,128,255,0.1)',
                name='BB Lower'
            ))
            
        # Add WERPI if enabled
        if indicators.get("show_werpi", False):
            try:
                # Calculate WERPI
                werpi_values = (df["Close"] - df["Close"].rolling(window=20).mean()) / df["Close"].rolling(window=20).std()
                werpi_values = werpi_values.rolling(window=5).mean()
                
                fig.add_trace(go.Scatter(
                    x=df[date_col], 
                    y=werpi_values,
                    mode='lines',
                    line=dict(color='purple', width=1.5),
                    name='WERPI'
                ))
            except Exception as e:
                logger.warning(f"Error calculating WERPI: {e}")
                
        # Add VMLI if enabled
        if indicators.get("show_vmli", False):
            try:
                # Calculate VMLI components
                volatility = df["Close"].pct_change().rolling(window=14).std() * 100
                momentum = df["Close"].pct_change(periods=14) * 100
                vmli_values = (momentum / volatility).fillna(0)
                vmli_values = 50 + (vmli_values * 10)
                
                fig.add_trace(go.Scatter(
                    x=df[date_col], 
                    y=vmli_values,
                    mode='lines',
                    line=dict(color='green', width=1.5),
                    name='VMLI'
                ))
            except Exception as e:
                logger.warning(f"Error calculating VMLI: {e}")
        
        # Add future forecast if available and enabled
        if future_forecast and len(future_forecast) > 0 and indicators.get("show_forecast", True):
            try:
                # Create forecast dates
                last_date = df[date_col].iloc[-1]
                
                # Handle different date types
                if isinstance(last_date, str):
                    last_date = pd.to_datetime(last_date)
                
                # Generate future dates based on the interval/timeframe
                interval = params.get('timeframe', '1d')
                if interval.endswith('d'):
                    days_increment = int(interval[:-1])
                    forecast_dates = [last_date + timedelta(days=days_increment * (i+1)) for i in range(len(future_forecast))]
                elif interval.endswith('h'):
                    hours_increment = int(interval[:-1])
                    forecast_dates = [last_date + timedelta(hours=hours_increment * (i+1)) for i in range(len(future_forecast))]
                elif interval.endswith('m') and not interval.endswith('mo'):
                    minutes_increment = int(interval[:-1])
                    forecast_dates = [last_date + timedelta(minutes=minutes_increment * (i+1)) for i in range(len(future_forecast))]
                elif interval == '1wk':
                    forecast_dates = [last_date + timedelta(weeks=(i+1)) for i in range(len(future_forecast))]
                elif interval == '1mo':
                    # Approximate a month as 30 days
                    forecast_dates = [last_date + timedelta(days=30 * (i+1)) for i in range(len(future_forecast))]
                else:
                    # Default to daily increments
                    forecast_dates = [last_date + timedelta(days=(i+1)) for i in range(len(future_forecast))]
                
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=future_forecast,
                    mode='lines+markers',
                    line=dict(color='green', width=2, dash='dash'),
                    marker=dict(size=6, color='green'),
                    name='Forecast'
                ))
                
                # Add a vertical line separating historical data from forecast
                fig.add_vline(
                    x=last_date, 
                    line_width=1, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="Forecast Start",
                    annotation_position="top right"
                )
                
                # Update session state with forecast
                st.session_state["future_forecast"] = future_forecast
                
            except Exception as e:
                logger.error(f"Error adding forecast to chart: {e}")
                st.error(f"Could not display forecast: {e}")
        
        # Configure layout
        fig.update_layout(
            title=f"{params.get('ticker', 'Price')} - {params.get('timeframe', 'Daily')} Chart",
            xaxis_title='Date',
            yaxis_title='Price',
            height=height,  # Use the passed height parameter
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating price chart: {e}", exc_info=True)
        st.error(f"Error creating chart: {str(e)}")

@robust_error_boundary
def create_technical_indicators_chart(df, params):
    """
    Create a chart with technical indicators.
    
    Args:
        df: DataFrame with price data and indicators
        params: Dictionary of parameters from control panel
    """
    try:
        # Validate inputs
        if df is None or df.empty:
            st.warning("No data available for technical indicator visualization.")
            return
            
        # Create subplots - 3 rows
        fig = go.Figure()
        
        # Ensure df has a date column
        if 'date' not in df.columns and 'Date' not in df.columns:
            # Try to use index as date
            if isinstance(df.index, pd.DatetimeIndex):
                df['date'] = df.index
            else:
                # Create a synthetic date column
                df['date'] = pd.date_range(start=datetime.now() - timedelta(days=len(df)), periods=len(df))
                
        # Use either 'date' or 'Date' column
        date_col = 'date' if 'date' in df.columns else 'Date'
        
        # Check for common indicators
        has_rsi = 'RSI' in df.columns
        has_macd = all(col in df.columns for col in ['MACD', 'MACD_signal'])
        has_volume = 'Volume' in df.columns
        
        # Create subplots based on available indicators
        subplot_titles = []
        if has_rsi:
            subplot_titles.append("RSI")
        if has_macd:
            subplot_titles.append("MACD")
        if has_volume:
            subplot_titles.append("Volume")
            
        # Create figure with subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=len(subplot_titles),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )
        
        # Add indicators to the subplots
        row = 1
        
        # Add RSI
        if has_rsi:
            fig.add_trace(
                go.Scatter(
                    x=df[date_col], 
                    y=df['RSI'],
                    line=dict(color='purple', width=1.5),
                    name='RSI'
                ),
                row=row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=df[date_col].iloc[0],
                y0=70,
                x1=df[date_col].iloc[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                row=row, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=df[date_col].iloc[0],
                y0=30,
                x1=df[date_col].iloc[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                row=row, col=1
            )
            
            # Set Y-axis range for RSI
            fig.update_yaxes(range=[0, 100], row=row, col=1)
            row += 1
        
        # Add MACD
        if has_macd:
            fig.add_trace(
                go.Scatter(
                    x=df[date_col], 
                    y=df['MACD'],
                    line=dict(color='blue', width=1.5),
                    name='MACD'
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df[date_col], 
                    y=df['MACD_signal'],
                    line=dict(color='red', width=1.5),
                    name='Signal'
                ),
                row=row, col=1
            )
            
            # Add MACD histogram if available
            if 'MACD_hist' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df[date_col], 
                        y=df['MACD_hist'],
                        marker_color=np.where(df['MACD_hist'] >= 0, 'green', 'red'),
                        name='MACD Histogram'
                    ),
                    row=row, col=1
                )
            
            row += 1
        
        # Add Volume
        if has_volume:
            fig.add_trace(
                go.Bar(
                    x=df[date_col], 
                    y=df['Volume'],
                    marker_color='blue',
                    opacity=0.7,
                    name='Volume'
                ),
                row=row, col=1
            )
            
            # Add volume moving average if available
            if 'Volume_MA' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col], 
                        y=df['Volume_MA'],
                        line=dict(color='orange', width=1.5),
                        name='Volume MA'
                    ),
                    row=row, col=1
                )
        
        # Configure layout
        fig.update_layout(
            height=600,
            title_text=f"Technical Indicators: {params.get('ticker', '')}",
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating technical indicators chart: {e}", exc_info=True)
        st.error(f"Error creating technical indicators: {str(e)}")

@robust_error_boundary
def show_advanced_dashboard_tabs(df):
    """
    Show advanced analysis tabs with additional visualizations.
    """
    try:
        # Create tabs for advanced analysis - reorganized as requested
        advanced_tabs = st.tabs([
            "Custom Indicators",
            "Standard Indicators", 
            "Correlation Matrix", 
            "Volume Analysis", 
            "Pattern Recognition"
        ])
        
        # Tab 1: Custom Indicators - kept the same
        with advanced_tabs[0]:
            # Create nested tabs for different custom indicators
            custom_tabs = st.tabs(["WERPI Indicator", "VMLI Analysis"])
            
            # WERPI Tab
            with custom_tabs[0]:
                st.subheader("Wavelet-based Encoded Relative Price Indicator (WERPI)")
                st.info(
                    "WERPI is a custom indicator that uses wavelet transforms to identify potential price movements."
                )

                # Create placeholder WERPI data using simpler calculations for reliability
                try:
                    # Simple placeholder calculation
                    werpi_values = (
                        df["Close"] - df["Close"].rolling(window=20).mean()
                    ) / df["Close"].rolling(window=20).std()
                    werpi_values = werpi_values.rolling(window=5).mean()  # Smooth the values

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df["date"],
                            y=df["Close"],
                            name="Price",
                            line=dict(color="blue"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df["date"], y=werpi_values, name="WERPI", line=dict(color="red")
                        )
                    )
                    fig.update_layout(
                        title="Price vs WERPI Indicator",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified",
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating WERPI: {e}")

            # VMLI Tab
            with custom_tabs[1]:
                st.subheader("Volatility-Momentum-Liquidity Indicator (VMLI)")
                st.info(
                    "VMLI combines volatility, momentum and liquidity metrics into a single indicator."
                )

                # Create simplified VMLI calculation for reliability
                try:
                    # Calculate simple components
                    volatility = df["Close"].pct_change().rolling(window=14).std() * 100
                    momentum = df["Close"].pct_change(periods=14) * 100

                    # Calculate simple VMLI
                    vmli_values = (momentum / volatility).fillna(0)
                    vmli_values = 50 + (vmli_values * 10)  # Scale to a more readable range

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df["date"],
                            y=df["Close"],
                            name="Price",
                            line=dict(color="blue"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df["date"],
                            y=vmli_values,
                            name="VMLI",
                            line=dict(color="purple"),
                        )
                    )
                    fig.update_layout(title="Price vs VMLI Indicator", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating VMLI: {e}")
        
        # Tab 2: Standard Technical Indicators - already showing detailed indicators
        with advanced_tabs[1]:
            # Simply show the technical indicators directly without redundant chart
            create_technical_indicators_chart(df, {})
        
        # Correlation Matrix Tab (now the third tab)
        with advanced_tabs[2]:
            st.subheader("Correlation Matrix")
            
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            
            # Remove any constant columns
            numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
            
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                # Calculate correlation matrix
                corr_matrix = numeric_df.corr()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='Viridis',
                    zmin=-1, zmax=1
                ))
                
                fig.update_layout(
                    title='Correlation Matrix',
                    height=700,
                    width=700,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig)
                
                # Display strongest correlations
                st.subheader("Strongest Correlations with Close Price")
                
                if 'Close' in corr_matrix.columns:
                    close_corr = corr_matrix['Close'].drop('Close')
                    if not close_corr.empty:
                        close_corr = close_corr.sort_values(ascending=False)
                    
                    # Convert to DataFrame for better display
                    close_corr_df = pd.DataFrame({
                        'Feature': close_corr.index,
                        'Correlation': close_corr.values
                    })
                    
                    # Prepare dataframe for display
                    close_corr_df = prepare_dataframe_for_display(close_corr_df)
                    
                    # Add color formatting
                    def color_corr(val):
                        color = 'green' if val > 0 else 'red'
                        return f'color: {color}'
                    
                    st.dataframe(
                        close_corr_df.style.format({'Correlation': '{:.4f}'})
                                        .applymap(color_corr, subset=['Correlation'])
                    )
                else:
                    st.info("Close price not available in correlation matrix.")
            else:
                st.info("Not enough numeric data for correlation analysis.")
        
        # Tab 4: Volume Analysis (moved down)
        with advanced_tabs[3]:
            if 'Volume' in df.columns:
                st.subheader("Volume Analysis")
                
                # Create a volume profile
                fig = go.Figure()
                
                # Add candlestick chart
                date_col = 'date' if 'date' in df.columns else 'Date'
                fig.add_trace(go.Candlestick(
                    x=df[date_col],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price',
                    xaxis='x',
                    yaxis='y'
                ))
                
                # Add volume bars
                fig.add_trace(go.Bar(
                    x=df[date_col],
                    y=df['Volume'],
                    name='Volume',
                    marker_color='blue',
                    opacity=0.5,
                    xaxis='x',
                    yaxis='y2'
                ))
                
                # Configure layout with secondary y-axis
                fig.update_layout(
                    title='Price and Volume',
                    height=600,
                    template='plotly_dark',
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(
                        title='Price',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Volume',
                        side='right',
                        overlaying='y',
                        showgrid=False
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No volume data available for analysis.")
        
        # Tab 5: Pattern Recognition (moved down)
        with advanced_tabs[4]:
            st.subheader("Pattern Recognition")
            
            # Check for required data
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                try:
                    # Import TA-Lib if available
                    import talib
                    
                    # List of patterns to check
                    patterns = {
                        'Engulfing': talib.CDLENGULFING,
                        'Hammer': talib.CDLHAMMER,
                        'Morning Star': talib.CDLMORNINGSTAR,
                        'Evening Star': talib.CDLEVENINGSTAR,
                        'Doji': talib.CDLDOJI,
                        'Three White Soldiers': talib.CDL3WHITESOLDIERS,
                        'Three Black Crows': talib.CDL3BLACKCROWS
                    }
                    
                    # Create a DataFrame to store pattern signals
                    pattern_df = pd.DataFrame(index=df.index)
                    
                    # Add date column
                    date_col = 'date' if 'date' in df.columns else 'Date'
                    if date_col in df.columns:
                        pattern_df[date_col] = df[date_col]
                    
                    # Calculate patterns
                    for pattern_name, pattern_func in patterns.items():
                        pattern_df[pattern_name] = pattern_func(
                            df['Open'].values, 
                            df['High'].values, 
                            df['Low'].values, 
                            df['Close'].values
                        )
                    
                    # Filter to show only where patterns were detected
                    pattern_df = pattern_df[pattern_df.iloc[:, 1:].any(axis=1)]
                    
                    if not pattern_df.empty:
                        st.write("Detected Patterns:")
                        
                        # Format pattern signals
                        def format_pattern(val):
                            if val > 0:
                                return "Bullish"
                            elif val < 0:
                                return "Bearish"
                            else:
                                return ""
                        
                        # Apply formatting
                        for col in pattern_df.columns:
                            if col != date_col:
                                pattern_df[col] = pattern_df[col].apply(format_pattern)
                        
                        # Prepare dataframe for display
                        pattern_df = prepare_dataframe_for_display(pattern_df)
                        
                        # Display pattern dataframe
                        st.dataframe(pattern_df)
                    else:
                        st.info("No patterns detected in the current data.")
                
                except ImportError:
                    st.warning("TA-Lib is not installed. Pattern recognition requires TA-Lib.")
                except Exception as e:
                    st.error(f"Error in pattern recognition: {e}")
            else:
                st.info("Required price data (OHLC) not available for pattern recognition.")
        
    except Exception as e:
        logger.error(f"Error showing advanced dashboard tabs: {e}", exc_info=True)
        st.error(f"Error in advanced analysis: {str(e)}")