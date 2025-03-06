import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Import necessary components for visualization tabs
from src.dashboard.model_visualizations import ModelVisualizationDashboard
from src.dashboard.enhanced_weight_viz import WeightVisualizationDashboard
from src.dashboard.pattern_discovery import add_pattern_discovery_tab

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Error handling decorator for dashboard components
def robust_error_boundary(func):
    """Decorator to catch errors in dashboard components"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            logger.exception(f"Error in {func.__name__}")
            # Display a more user-friendly message
            st.info("This component encountered an error. Please check the logs for details.")
            return None
    return wrapper

def main():
    """Main function to run the dashboard"""
    st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide")
    
    st.title("Cryptocurrency Prediction Dashboard")
    
    # Load data (placeholder for your actual data loading)
    df = load_data()
    
    # Initialize ensemble weighter (placeholder for your actual initialization)
    ensemble_weighter = initialize_ensemble_weighter()
    
    # Create tabs for different dashboard sections
    tabs = st.tabs([
        "Main Dashboard", 
        "Technical Analysis", 
        "Model Insights", 
        "Model Visualization", 
        "Enhanced Weight Visualization", 
        "Pattern Discovery"
    ])
    
    # Main Dashboard
    with tabs[0]:
        display_main_dashboard(df, ensemble_weighter)
    
    # Technical Analysis
    with tabs[1]:
        display_technical_analysis(df)
    
    # Model Insights
    with tabs[2]:
        display_model_insights(ensemble_weighter)
    
    # Model Visualization Tab
    with tabs[3]:
        if ensemble_weighter is not None:
            model_viz = ModelVisualizationDashboard(ensemble_weighter)
            model_viz.render_dashboard()
        else:
            st.warning("Ensemble weighter not initialized. Cannot display model visualization.")
    
    # Enhanced Weight Visualization Tab
    with tabs[4]:
        if ensemble_weighter is not None:
            weight_viz = WeightVisualizationDashboard(ensemble_weighter)
            weight_viz.render_dashboard()
        else:
            st.warning("Ensemble weighter not initialized. Cannot display weight visualization.")
    
    # Pattern Discovery Tab
    with tabs[5]:
        add_pattern_discovery_tab(df, ensemble_weighter)

@robust_error_boundary
def load_data():
    """Load and process data for the dashboard"""
    try:
        # This is a placeholder for your actual data loading logic
        # You would replace this with your actual implementation
        
        # For testing, create a sample dataframe if no data is available
        if 'df' not in st.session_state:
            date_range = pd.date_range(end=datetime.now(), periods=365)
            price = 10000 + np.cumsum(np.random.randn(365) * 100)
            volume = np.random.randint(1000, 10000, 365)
            
            df = pd.DataFrame({
                'Date': date_range,
                'Open': price * 0.99,
                'High': price * 1.02,
                'Low': price * 0.98,
                'Close': price,
                'Volume': volume,
                'RSI': np.random.randint(0, 100, 365),
                'MACD': np.random.randn(365) * 10,
                'Signal': np.random.randn(365) * 8,
                'BB_upper': price * 1.05,
                'BB_middle': price,
                'BB_lower': price * 0.95
            })
            st.session_state.df = df
            
        return st.session_state.df
            
    except Exception as e:
        logger.exception("Error loading data")
        # Create minimal emergency dataframe to prevent cascading failures
        date_range = pd.date_range(end=datetime.now(), periods=30)
        emergency_df = pd.DataFrame({
            'Date': date_range,
            'Close': np.linspace(100, 120, 30)
        })
        return emergency_df

@robust_error_boundary
def initialize_ensemble_weighter():
    """Initialize the ensemble weighter"""
    try:
        # This is a placeholder for your actual ensemble weighter initialization
        # For testing purposes, we'll create a simple mock object
        
        class MockEnsembleWeighter:
            def __init__(self):
                # Create some example weights
                self.base_weights = {
                    'lstm': 0.3,
                    'xgboost': 0.25,
                    'random_forest': 0.2,
                    'cnn': 0.15,
                    'nbeats': 0.1,
                }
                
                self.current_weights = self.base_weights.copy()
                
                # Create historical weights
                self.historical_weights = []
                for i in range(30):
                    # Add some random variation to weights
                    weights = {k: max(0.01, min(0.99, v + np.random.normal(0, 0.05))) for k, v in self.base_weights.items()}
                    # Normalize
                    total = sum(weights.values())
                    weights = {k: v/total for k, v in weights.items()}
                    self.historical_weights.append(weights)
                
                # Model correlation matrix
                self.model_correlation_matrix = {
                    ('lstm', 'xgboost'): 0.3,
                    ('lstm', 'random_forest'): 0.2,
                    ('lstm', 'cnn'): 0.6,
                    ('lstm', 'nbeats'): 0.1,
                    ('xgboost', 'random_forest'): 0.7,
                    ('xgboost', 'cnn'): 0.2,
                    ('xgboost', 'nbeats'): 0.3,
                    ('random_forest', 'cnn'): 0.1,
                    ('random_forest', 'nbeats'): 0.4,
                    ('cnn', 'nbeats'): 0.3,
                }
                
                # Other attributes that may be needed
                self.performance_metrics = {
                    'lstm': {'rmse': 0.05, 'mae': 0.03, 'mape': 5.2},
                    'xgboost': {'rmse': 0.06, 'mae': 0.04, 'mape': 6.1},
                    'random_forest': {'rmse': 0.07, 'mae': 0.05, 'mape': 6.8},
                    'cnn': {'rmse': 0.055, 'mae': 0.035, 'mape': 5.5},
                    'nbeats': {'rmse': 0.065, 'mae': 0.045, 'mape': 6.3},
                }
            
            def get_weights(self):
                return self.current_weights
            
            def update_weights(self, new_weights):
                self.current_weights = new_weights
                self.historical_weights.append(new_weights.copy())
            
            def get_performance(self):
                return self.performance_metrics
                
        return MockEnsembleWeighter()
        
    except Exception as e:
        logger.exception("Error initializing ensemble weighter")
        return None

@robust_error_boundary
def display_main_dashboard(df, ensemble_weighter):
    """Display the main dashboard with overview metrics and predictions"""
    st.header("Price Overview")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-2]
        price_change = current_price - previous_price
        price_change_pct = price_change / previous_price * 100
        
        st.metric(
            "Current Price", 
            f"${current_price:.2f}",
            f"{price_change_pct:.2f}%"
        )
    
    with col2:
        vol_24h = df['Volume'].iloc[-1]
        prev_vol = df['Volume'].iloc[-2]
        vol_change = (vol_24h - prev_vol) / prev_vol * 100
        
        st.metric(
            "24h Volume", 
            f"{vol_24h:,.0f}",
            f"{vol_change:.2f}%"
        )
    
    with col3:
        # 7-day volatility (standard deviation of returns)
        returns = df['Close'].pct_change().iloc[-7:]
        volatility = returns.std() * 100
        
        st.metric(
            "7-Day Volatility", 
            f"{volatility:.2f}%"
        )
    
    with col4:
        # RSI if available, otherwise a placeholder
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            
            # Color code RSI
            if rsi < 30:
                delta = "Oversold"
            elif rsi > 70:
                delta = "Overbought"
            else:
                delta = "Neutral"
                
            st.metric(
                "RSI", 
                f"{rsi:.1f}",
                delta
            )
        else:
            st.metric("Market Sentiment", "Neutral")
    
    # Create price chart
    fig = go.Figure()
    
    # Add candlestick chart if we have OHLC data
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))
    else:
        # Otherwise a line chart
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price"
        ))
    
    # Add Bollinger Bands if available
    if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['BB_upper'],
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 100, 100, 0.5)'),
            name="Upper BB"
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['BB_lower'],
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 100, 100, 0.5)'),
            fill='tonexty',
            fillcolor='rgba(100, 100, 100, 0.1)',
            name="Lower BB"
        ))
    
    # Update layout
    fig.update_layout(
        title="Price History",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display predictions if available
    st.subheader("Price Predictions")
    
    # This is a placeholder - you would replace with actual prediction logic
    days_ahead = 7
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead+1)]
    
    # Create a simple placeholder prediction model (this would be your actual model)
    last_price = df['Close'].iloc[-1]
    predicted_values = [last_price]
    for i in range(days_ahead):
        # Make a random prediction for demo purposes
        next_return = np.random.normal(0.001, 0.02)  # Mean 0.1%, stdev 2%
        next_price = predicted_values[-1] * (1 + next_return)
        predicted_values.append(next_price)
    
    predicted_values = predicted_values[1:]  # Remove the first element which is just the last actual price
    
    # Create a dataframe with predictions
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predicted_values
    })
    
    # Create confidence intervals (placeholder)
    prediction_df['Lower Bound'] = prediction_df['Predicted Price'] * 0.95
    prediction_df['Upper Bound'] = prediction_df['Predicted Price'] * 1.05
    
    # Plot the predictions
    fig = go.Figure()
    
    # Add actual price
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name="Actual Price"
    ))
    
    # Add predicted price
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted Price'],
        mode='lines',
        name="Predicted Price",
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Upper Bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Lower Bound'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        name="95% Confidence"
    ))
    
    # Update layout
    fig.update_layout(
        title="Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

@robust_error_boundary
def display_technical_analysis(df):
    """Display technical analysis indicators and charts"""
    st.header("Technical Analysis")
    
    # Create tabs for different indicators
    tabs = st.tabs(["RSI", "MACD", "Bollinger Bands", "Moving Averages"])
    
    # RSI tab
    with tabs[0]:
        st.subheader("Relative Strength Index (RSI)")
        
        if 'RSI' in df.columns:
            # Create RSI plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['RSI'],
                mode='lines',
                name="RSI"
            ))
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig.update_layout(
                title="RSI (14) Chart",
                xaxis_title="Date",
                yaxis_title="RSI",
                height=400,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current RSI
            current_rsi = df['RSI'].iloc[-1]
            
            # RSI interpretation
            if current_rsi > 70:
                st.warning(f"Current RSI: {current_rsi:.2f} - **Overbought** condition, potential reversal or consolidation ahead")
            elif current_rsi < 30:
                st.success(f"Current RSI: {current_rsi:.2f} - **Oversold** condition, potential buying opportunity")
            else:
                st.info(f"Current RSI: {current_rsi:.2f} - Within **neutral** range")
        else:
            st.warning("RSI data not available")
    
    # MACD tab
    with tabs[1]:
        st.subheader("Moving Average Convergence Divergence (MACD)")
        
        if all(col in df.columns for col in ['MACD', 'Signal']):
            # Create MACD plot
            fig = go.Figure()
            
            # Add MACD line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['MACD'],
                mode='lines',
                name="MACD"
            ))
            
            # Add Signal line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Signal'],
                mode='lines',
                name="Signal Line"
            ))
            
            # Add MACD histogram
            df['Histogram'] = df['MACD'] - df['Signal']
            
            # Color the histogram based on value and change
            colors = ['green' if h > 0 else 'red' for h in df['Histogram']]
            
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['Histogram'],
                name="Histogram",
                marker_color=colors
            ))
            
            fig.update_layout(
                title="MACD Chart",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD interpretation
            macd = df['MACD'].iloc[-1]
            signal = df['Signal'].iloc[-1]
            hist = df['Histogram'].iloc[-1]
            
            if macd > signal and hist > 0:
                st.success(f"MACD ({macd:.4f}) is above Signal Line ({signal:.4f}) - **Bullish** momentum")
            elif macd < signal and hist < 0:
                st.warning(f"MACD ({macd:.4f}) is below Signal Line ({signal:.4f}) - **Bearish** momentum")
            else:
                # Check if recent crossover
                if df['Histogram'].iloc[-2] * hist < 0:  # Sign change
                    if hist > 0:
                        st.success(f"MACD ({macd:.4f}) just crossed above Signal Line ({signal:.4f}) - Potential **bullish** crossover")
                    else:
                        st.warning(f"MACD ({macd:.4f}) just crossed below Signal Line ({signal:.4f}) - Potential **bearish** crossover")
                else:
                    st.info(f"MACD ({macd:.4f}) and Signal Line ({signal:.4f}) - No clear trend direction")
                    
        else:
            st.warning("MACD data not available")
    
    # Bollinger Bands tab
    with tabs[2]:
        st.subheader("Bollinger Bands")
        
        if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            # Create Bollinger Bands plot
            fig = go.Figure()
            
            # Add price
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name="Price"
            ))
            
            # Add upper band
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['BB_upper'],
                mode='lines',
                name="Upper Band",
                line=dict(width=0.5, color='rgba(173, 204, 255, 0.7)')
            ))
            
            # Add middle band (SMA)
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['BB_middle'],
                mode='lines',
                name="Middle Band (SMA)",
                line=dict(width=1, color='rgba(127, 127, 127, 0.7)')
            ))
            
            # Add lower band
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['BB_lower'],
                mode='lines',
                name="Lower Band",
                line=dict(width=0.5, color='rgba(173, 204, 255, 0.7)'),
                fill='tonexty',  # fill between lower and upper
                fillcolor='rgba(173, 204, 255, 0.15)'
            ))
            
            fig.update_layout(
                title="Bollinger Bands (20, 2)",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bollinger Band interpretation
            current_price = df['Close'].iloc[-1]
            upper_band = df['BB_upper'].iloc[-1]
            middle_band = df['BB_middle'].iloc[-1]
            lower_band = df['BB_lower'].iloc[-1]
            
            # Calculate bandwidth
            bandwidth = (upper_band - lower_band) / middle_band
            
            # Calculate %B
            percent_b = (current_price - lower_band) / (upper_band - lower_band) if (upper_band != lower_band) else 0.5
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Bandwidth", f"{bandwidth:.4f}")
                
                if bandwidth < 0.1:
                    st.info("Tight bands - potential breakout approaching")
                elif bandwidth > 0.3:
                    st.info("Wide bands - high volatility")
            
            with col2:
                st.metric("%B", f"{percent_b:.4f}")
                
                if percent_b > 1:
                    st.warning("Price above upper band - overbought, potential reversal")
                elif percent_b < 0:
                    st.success("Price below lower band - oversold, potential buying opportunity")
                elif percent_b > 0.8:
                    st.info("Price approaching upper band - strong uptrend")
                elif percent_b < 0.2:
                    st.info("Price approaching lower band - strong downtrend")
        else:
            st.warning("Bollinger Bands data not available")
    
    # Moving Averages tab
    with tabs[3]:
        st.subheader("Moving Averages")
        
        # For demonstration, let's calculate some MAs if they don't exist
        if 'MA_50' not in df.columns:
            df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        if 'MA_200' not in df.columns:
            df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Create Moving Averages plot
        fig = go.Figure()
        
        # Add price
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price"
        ))
        
        # Add 50-day MA
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_50'],
            mode='lines',
            name="50-day MA",
            line=dict(width=1.5, color='rgba(255, 165, 0, 0.7)')
        ))
        
        # Add 200-day MA
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_200'],
            mode='lines',
            name="200-day MA",
            line=dict(width=2, color='rgba(0, 0, 255, 0.7)')
        ))
        
        fig.update_layout(
            title="Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # MA interpretation
        ma_50 = df['MA_50'].iloc[-1]
        ma_200 = df['MA_200'].iloc[-1]
        price = df['Close'].iloc[-1]
        
        # Check for golden cross or death cross
        if df['MA_50'].iloc[-2] < df['MA_200'].iloc[-2] and ma_50 > ma_200:
            st.success("**Golden Cross**: 50-day MA just crossed above 200-day MA - Bullish signal")
        elif df['MA_50'].iloc[-2] > df['MA_200'].iloc[-2] and ma_50 < ma_200:
            st.warning("**Death Cross**: 50-day MA just crossed below 200-day MA - Bearish signal")
            
        # Current price relative to MAs
        if price > ma_50 and price > ma_200:
            st.success(f"Price is above both the 50-day and 200-day MAs - Bullish trend")
        elif price < ma_50 and price < ma_200:
            st.warning(f"Price is below both the 50-day and 200-day MAs - Bearish trend")
        elif price > ma_50 and price < ma_200:
            st.info(f"Price is above 50-day MA but below 200-day MA - Mixed signals, potential reversal")
        else:  # price < ma_50 and price > ma_200
            st.info(f"Price is below 50-day MA but above 200-day MA - Mixed signals, watch closely")

@robust_error_boundary
def display_model_insights(ensemble_weighter):
    """Display insights from the prediction model"""
    st.header("Model Insights")
    
    if ensemble_weighter is None:
        st.warning("Ensemble weighter not initialized. Cannot display model insights.")
        return
    
    # Display current model weights
    st.subheader("Current Model Weights")
    
    weights = ensemble_weighter.get_weights()
    
    # Create a pie chart of weights
    fig = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        title="Model Contribution to Ensemble"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display performance metrics
    st.subheader("Model Performance Metrics")
    
    metrics = ensemble_weighter.get_performance()
    
    # Convert to dataframe for display
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'Model'
    metrics_df = metrics_df.reset_index()
    
    # Format metrics
    formatted_df = metrics_df.copy()
    for col in ['rmse', 'mae']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map('{:.4f}'.format)
    
    if 'mape' in formatted_df.columns:
        formatted_df['mape'] = formatted_df['mape'].map('{:.2f}%'.format)
    
    # Display as table
    st.table(formatted_df)
    
    # Create performance comparison chart
    st.subheader("Model Performance Comparison")
    
    # Create bar chart of RMSE
    if 'rmse' in metrics_df.columns:
        fig = px.bar(
            metrics_df,
            x='Model',
            y='rmse',
            color='Model',
            title="RMSE by Model (Lower is Better)"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

@robust_error_boundary
def display_tested_models():
    """Display models from the tested_models.yaml file"""
    from config.config_loader import DATA_DIR
    
    # Update the tested models file path to use Models directory
    tested_models_file = os.path.join(DATA_DIR, "Models", "tested_models.yaml")
    
    if not os.path.exists(tested_models_file):
        st.info("No tested models file found.")
        return
    
    import yaml
    try:
        with open(tested_models_file, 'r') as f:
            tested_models = yaml.safe_load(f)
        
        if not tested_models:
            st.info("No tested models found in the file.")
            return
        
        # Display models in a table
        models_list = []
        for model_name, model_data in tested_models.items():
            model_info = {
                'Name': model_name,
                'Type': model_data.get('type', 'Unknown'),
                'RMSE': model_data.get('metrics', {}).get('rmse', 'N/A'),
                'MAE': model_data.get('metrics', {}).get('mae', 'N/A'),
                'MAPE': model_data.get('metrics', {}).get('mape', 'N/A'),
                'Training Date': model_data.get('training_date', 'Unknown')
            }
            models_list.append(model_info)
        
        models_df = pd.DataFrame(models_list)
        st.dataframe(models_df)
        
    except Exception as e:
        st.error(f"Error loading tested models: {e}")

# Run the dashboard
if __name__ == "__main__":
    main()
