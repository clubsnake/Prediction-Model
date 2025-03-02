"""
advanced_dashboard.py

This module provides advanced visualization functions that can be used as addons
to your main dashboard. It includes:

- Data loading and indicator calculation,
- A WERPI indicator class and chart,
- Secondary technical charts (RSI, MACD, Volume),
- A sentiment analysis visualization,
- Real learning progress visualization (with animation),
- Ensemble learning evolution animation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import logging
import pywt
from hmmlearn import hmm

def load_data(ticker, period="1y", interval="1d", start_date=None):
    """Load financial data for a given ticker using yfinance."""
    try:
        if start_date:
            data = yf.download(ticker, start=start_date, interval=interval)
        else:
            data = yf.download(ticker, period=period, interval=interval)
        data = data.reset_index()
        # Ensure there's a Date column
        if 'Date' not in data.columns:
            if 'date' in data.columns:
                data.rename(columns={'date': 'Date'}, inplace=True)
            else:
                data.rename(columns={'index': 'Date'}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(data):
    """Calculate moving averages, Bollinger Bands, RSI, and MACD."""
    if data.empty:
        return data
    try:
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        # RSI calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        # MACD calculation
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        return data
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return data

class WERPIIndicator:
    """
    Implements a simple WERPI indicator using a RandomForestRegressor.
    It extracts price, volatility, volume, and technical indicator features.
    """
    def __init__(self, ticker, interval):
        self.ticker = ticker
        self.interval = interval
        self.model = None
        self.optimized = False

    def load_or_create(self):
        """Load or create the indicator model."""
        try:
            self.model = RandomForestRegressor(n_estimators=100)
            return True
        except Exception as e:
            st.error(f"Error in WERPI load_or_create: {str(e)}")
            return False

    def train(self, data, optimize=False):
        """Train the WERPI model on the provided data."""
        try:
            features = self._extract_features(data)
            if features.empty:
                return False
            X = features.iloc[:-1]
            y = data['Close'].pct_change().iloc[1:] * 100  # Percent change as target
            self.model = RandomForestRegressor(n_estimators=100)
            self.model.fit(X, y)
            self.optimized = optimize
            # Optionally, you could store the trained model as best if its performance is superior.
            st.session_state["best_werpi_model"] = self.model
            return True
        except Exception as e:
            st.error(f"Error training WERPI: {str(e)}")
            return False

    def predict(self, data):
        """Generate the WERPI indicator values."""
        if self.model is None:
            st.error("WERPI model is not initialized.")
            return None
        try:
            features = self._extract_features(data)
            if features.empty:
                return None
            raw_preds = self.model.predict(features)
            min_val = np.min(raw_preds)
            max_val = np.max(raw_preds)
            range_val = max_val - min_val
            if range_val == 0:
                return np.zeros(len(raw_preds))
            oscillator = 100 * (raw_preds - min_val) / range_val
            return oscillator
        except Exception as e:
            st.error(f"Error in WERPI predict: {str(e)}")
            return None

    def _extract_features(self, data):
        """Extract relevant features from the data for the WERPI model."""
        try:
            df = data.copy()
            df['return_1d'] = df['Close'].pct_change()
            df['return_5d'] = df['Close'].pct_change(5)
            df['return_20d'] = df['Close'].pct_change(20)
            df['volatility_5d'] = df['return_1d'].rolling(5).std()
            df['volatility_20d'] = df['return_1d'].rolling(20).std()
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df = df.dropna()
            feature_cols = [
                'return_1d', 'return_5d', 'return_20d',
                'volatility_5d', 'volatility_20d',
                'volume_change', 'volume_ma_ratio',
                'RSI', 'MACD', 'MACD_Hist'
            ]
            return df[feature_cols]
        except Exception as e:
            st.error(f"Error extracting features for WERPI: {str(e)}")
            return pd.DataFrame()

def create_werpi_chart(data, werpi_values):
    """Create a Plotly chart overlaying the price and WERPI indicator."""
    if data.empty or werpi_values is None or len(werpi_values) == 0:
        return None
    try:
        fig = go.Figure()
        dates = data['Date'].iloc[-len(werpi_values):]
        fig.add_trace(go.Scatter(
            x=dates,
            y=data['Close'].iloc[-len(werpi_values):],
            name="Price",
            line=dict(color='black', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=werpi_values,
            name="WERPI",
            line=dict(color='purple', width=2)
        ))
        # Add threshold lines (optional)
        fig.add_shape(type="line",
                      x0=dates.iloc[0], x1=dates.iloc[-1],
                      y0=70, y1=70,
                      line=dict(color="red", dash="dash"))
        fig.add_shape(type="line",
                      x0=dates.iloc[0], x1=dates.iloc[-1],
                      y0=30, y1=30,
                      line=dict(color="green", dash="dash"))
        fig.update_layout(title="WERPI Indicator with Price", xaxis_title="Date", yaxis_title="Value", height=500)
        return fig
    except Exception as e:
        st.error(f"Error creating WERPI chart: {str(e)}")
        return None

def create_secondary_charts(data):
    """Create a multi-panel Plotly chart for Volume, RSI, and MACD."""
    if data.empty:
        return None
    try:
        fig = make_subplots(rows=3, cols=1,
                            row_heights=[0.4, 0.3, 0.3],
                            subplot_titles=("Volume", "RSI", "MACD"),
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="Volume", marker_color='rgba(0,0,255,0.5)'), row=1, col=1)
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name="RSI", line=dict(color='purple', width=1)), row=2, col=1)
            fig.add_shape(type="line", x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1],
                          y0=70, y1=70, line=dict(color="red", dash="dash"), row=2, col=1)
            fig.add_shape(type="line", x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1],
                          y0=30, y1=30, line=dict(color="green", dash="dash"), row=2, col=1)
        if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name="MACD", line=dict(color='blue', width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD_Signal'], name="Signal", line=dict(color='red', width=1)), row=3, col=1)
            colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
            fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Hist'], name="MACD Hist", marker_color=colors), row=3, col=1)
        fig.update_layout(height=600, showlegend=True, margin=dict(l=50, r=50, t=50, b=50))
        fig.update_yaxes(title_text="Volume", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0,100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        return fig
    except Exception as e:
        st.error(f"Error creating secondary charts: {str(e)}")
        return None

def get_sentiment_data(ticker):
    """Generate sentiment data for a ticker (replace with real API if available)."""
    try:
        days = 90
        dates = pd.date_range(end=datetime.now(), periods=days)
        sentiments = np.random.normal(0, 0.3, size=days)
        sentiments = np.clip(sentiments, -1, 1)
        trend = np.linspace(-0.2, 0.2, days)
        sentiments = sentiments + trend
        sentiments = np.clip(sentiments, -1, 1)
        sentiment_df = pd.DataFrame({'Date': dates, 'Sentiment': sentiments})
        return sentiment_df
    except Exception as e:
        st.error(f"Error getting sentiment data: {str(e)}")
        return pd.DataFrame()

def visualize_sentiment(sentiment_data):
    """Visualize sentiment data as a Plotly line chart."""
    if sentiment_data.empty:
        return None
    try:
        fig = go.Figure()
        sentiment_colors = ['red' if s < 0 else 'green' for s in sentiment_data['Sentiment']]
        fig.add_trace(go.Scatter(
            x=sentiment_data['Date'],
            y=sentiment_data['Sentiment'],
            mode='lines+markers',
            name="Sentiment",
            marker=dict(color=sentiment_colors),
            line=dict(color='blue')
        ))
        fig.add_shape(type="line",
                      x0=sentiment_data['Date'].iloc[0],
                      x1=sentiment_data['Date'].iloc[-1],
                      y0=0, y1=0,
                      line=dict(color="black", dash="dash"))
        fig.update_layout(title="Market Sentiment Analysis",
                          xaxis_title="Date",
                          yaxis_title="Sentiment Score (-1 to 1)",
                          yaxis=dict(range=[-1.1, 1.1]),
                          height=400,
                          margin=dict(l=50, r=50, t=50, b=50))
        return fig
    except Exception as e:
        st.error(f"Error visualizing sentiment: {str(e)}")
        return None

def visualize_learning_progress():
    """
    Visualize training progress based on weight history stored in session state.
    Expects st.session_state['model_weights_history'] as a list of dicts with keys:
    'epoch', 'train_loss', 'test_loss'.
    """
    if not st.session_state.get('model_weights_history'):
        return None
    try:
        history = st.session_state['model_weights_history']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[h['epoch'] for h in history],
            y=[h['train_loss'] for h in history],
            mode='lines+markers',
            name="Training Loss"
        ))
        fig.add_trace(go.Scatter(
            x=[h['epoch'] for h in history],
            y=[h['test_loss'] for h in history],
            mode='lines+markers',
            name="Test Loss"
        ))
        fig.update_layout(title="Learning Progress",
                          xaxis_title="Epoch",
                          yaxis_title="Loss",
                          height=400,
                          margin=dict(l=50, r=50, t=50, b=50))
        return fig
    except Exception as e:
        st.error(f"Error visualizing learning progress: {str(e)}")
        return None

def create_learning_animation(weight_history):
    """
    Create a Plotly animation of training progress.
    Expects weight_history as a list of dicts with keys 'epoch', 'train_loss', 'test_loss'.
    """
    import plotly.express as px
    if not weight_history:
        return None
    df = pd.DataFrame(weight_history)
    # Create an animation frame by epoch (using the epoch as a frame identifier)
    fig = px.line(df, x="epoch", y=["train_loss", "test_loss"],
                  labels={"value": "Loss", "variable": "Loss Type", "epoch": "Epoch"},
                  title="Training Progress Animation",
                  animation_frame="epoch",
                  range_y=[df[["train_loss", "test_loss"]].min().min()*0.95, df[["train_loss", "test_loss"]].max().max()*1.05])
    return fig

def animate_ensemble_learning():
    """
    Create an animation showing how ensemble predictions evolve over time.
    Assumes st.session_state['ensemble_predictions_log'] is updated by the tuning loop.
    Each log entry should contain: 'timestamp', 'date', 'actual', 'predicted'.
    """
    import plotly.express as px
    if not st.session_state.get('ensemble_predictions_log'):
         return None
    df = pd.DataFrame(st.session_state['ensemble_predictions_log'])
    if df.empty:
         return None
    df = df.sort_values('timestamp')
    # Use timestamp (or an index) as the animation frame
    df['frame'] = df.index.astype(str)
    fig = px.scatter(df, x="date", y="predicted", animation_frame="frame",
                     range_y=[df["actual"].min()*0.95, df["actual"].max()*1.05],
                     title="Ensemble Predictions Evolution",
                     labels={"predicted": "Predicted Price", "date": "Date"})
    return fig
