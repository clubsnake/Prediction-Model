"""
Advanced market regime detection and model optimization system.
Implements multi-factor regime classification, adaptive learning, and meta-learning
for optimal model selection across different market conditions.
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class MarketRegimeSystem:
    """
    Advanced market regime system with multi-dimensional learning and optimization.
    Detects complex market patterns and optimizes model weights based on historical performance.
    """
    
    def __init__(self, 
                 memory_length: int = 500,
                 regime_memory_file: Optional[str] = None,
                 learning_rate: float = 0.3,
                 meta_learning_rate: float = 0.05,
                 use_clustering: bool = True,
                 num_clusters: int = 7,
                 feature_history_length: int = 100):
        """
        Initialize the market regime system.
        
        Args:
            memory_length: Max entries to keep in memory
            regime_memory_file: File to save/load regime memory
            learning_rate: Base rate for model performance updates
            meta_learning_rate: Rate for updating adaptation parameters
            use_clustering: Whether to use unsupervised clustering for regime detection
            num_clusters: Number of clusters for unsupervised regime detection
            feature_history_length: Length of feature history to keep for clustering
        """
        # Standard regime categories
        self.standard_regimes = [
            'trending_up_strong',
            'trending_up_weak',
            'trending_down_strong',
            'trending_down_weak',
            'ranging_tight',
            'ranging_wide',
            'high_volatility',
            'low_volatility',
            'breakout',
            'reversal'
        ]
        
        # Initialize regime performance tracking
        self.regime_performance = {regime: {} for regime in self.standard_regimes}
        
        # Add 'unknown' and 'custom' regimes
        self.regime_performance['unknown'] = {}
        self.regime_performance['custom'] = {}
        
        # Regime clusters - will be populated by clustering algorithm
        self.regime_clusters = {}
        
        # Adaptive learning rates for each model
        self.model_learning_rates = {}
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        
        # Memory system configuration
        self.memory_length = memory_length
        self.feature_history_length = feature_history_length
        self.regime_memory_file = regime_memory_file
        
        # Initialize memory structures
        self.regime_history = []
        self.feature_history = []
        self.performance_history = {}
        
        # Market state trackers
        self.current_regime = "unknown"
        self.previous_regime = "unknown"
        self.regime_duration = 0
        self.regime_transitions = {}  # Track probability of transitions
        
        # Configure clustering
        self.use_clustering = use_clustering
        self.num_clusters = num_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)  # Reduce to 3 dimensions for easier clustering
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.cluster_initialized = False
        
        # Last update timestamp
        self.last_update = datetime.now()
        
        # Adaptation effectiveness tracking (meta-learning)
        self.adaptation_performance = {}
        
        # Load memory if file specified
        if self.regime_memory_file and os.path.exists(self.regime_memory_file):
            self.load_memory()
    
    def extract_market_features(self, df: pd.DataFrame, column: str = "Close") -> Dict[str, float]:
        """
        Extract comprehensive set of market features for regime detection.
        
        Args:
            df: DataFrame with price data
            column: Price column name
            
        Returns:
            Dictionary of market features
        """
        try:
            if df is None or df.empty or column not in df.columns:
                return {}
                
            # Ensure we have enough data
            if len(df) < 200:
                logger.warning(f"Not enough data for feature extraction: {len(df)} rows")
                return {}
            
            # Calculate returns
            df['returns'] = df[column].pct_change()
            
            # === Volatility Features ===
            # Short-term volatility (10-day)
            volatility_10d = df['returns'].rolling(window=10).std().iloc[-1] * np.sqrt(252)
            # Medium-term volatility (30-day)
            volatility_30d = df['returns'].rolling(window=30).std().iloc[-1] * np.sqrt(252)
            # Long-term volatility (90-day)
            volatility_90d = df['returns'].rolling(window=90).std().iloc[-1] * np.sqrt(252)
            # Volatility ratio (short/long)
            volatility_ratio = volatility_10d / volatility_90d if volatility_90d > 0 else 1.0
            
            # === Trend Features ===
            # Calculate moving averages
            df['MA10'] = df[column].rolling(window=10).mean()
            df['MA50'] = df[column].rolling(window=50).mean()
            df['MA200'] = df[column].rolling(window=200).mean()
            
            # Moving average relationships
            ma10_50_ratio = df['MA10'].iloc[-1] / df['MA50'].iloc[-1] if df['MA50'].iloc[-1] > 0 else 1.0
            ma50_200_ratio = df['MA50'].iloc[-1] / df['MA200'].iloc[-1] if df['MA200'].iloc[-1] > 0 else 1.0
            
            # Price vs MA relationships
            price_ma50_ratio = df[column].iloc[-1] / df['MA50'].iloc[-1] if df['MA50'].iloc[-1] > 0 else 1.0
            price_ma200_ratio = df[column].iloc[-1] / df['MA200'].iloc[-1] if df['MA200'].iloc[-1] > 0 else 1.0
            
            # Slope of moving averages
            ma50_slope = (df['MA50'].iloc[-1] / df['MA50'].iloc[-20] - 1) * 100 if df['MA50'].iloc[-20] > 0 else 0
            ma200_slope = (df['MA200'].iloc[-1] / df['MA200'].iloc[-20] - 1) * 100 if df['MA200'].iloc[-20] > 0 else 0
            
            # === Momentum Features ===
            # RSI (14-day)
            delta = df['returns']
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean().iloc[-1]
            avg_loss = loss.rolling(window=14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            df['EMA12'] = df[column].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df[column].ewm(span=26, adjust=False).mean()
            macd = df['EMA12'].iloc[-1] - df['EMA26'].iloc[-1]
            signal = df[column].ewm(span=9, adjust=False).mean().iloc[-1]
            macd_histogram = macd - signal
            
            # Mean reversion features
            bol_upper = df['MA20'] + 2 * df[column].rolling(window=20).std() if 'MA20' in df.columns else df['MA50'] + 2 * df[column].rolling(window=50).std() 
            bol_lower = df['MA20'] - 2 * df[column].rolling(window=20).std() if 'MA20' in df.columns else df['MA50'] - 2 * df[column].rolling(window=50).std()
            price_upper_ratio = df[column].iloc[-1] / bol_upper.iloc[-1] if bol_upper.iloc[-1] > 0 else 1.0
            price_lower_ratio = df[column].iloc[-1] / bol_lower.iloc[-1] if bol_lower.iloc[-1] > 0 else 1.0
            
            # Volume features if available
            volume_features = {}
            if 'Volume' in df.columns:
                recent_volume = df['Volume'].iloc[-10:].mean()
                avg_volume = df['Volume'].iloc[-50:].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                volume_features = {
                    'volume_ratio': volume_ratio,
                    'volume_trend': (df['Volume'].iloc[-1] / df['Volume'].iloc[-10] - 1) * 100 if df['Volume'].iloc[-10] > 0 else 0,
                }
            
            # Combine all features
            features = {
                # Volatility features
                'volatility_10d': volatility_10d,
                'volatility_30d': volatility_30d,
                'volatility_90d': volatility_90d,
                'volatility_ratio': volatility_ratio,
                
                # Trend features
                'ma10_50_ratio': ma10_50_ratio,
                'ma50_200_ratio': ma50_200_ratio,
                'price_ma50_ratio': price_ma50_ratio,
                'price_ma200_ratio': price_ma200_ratio,
                'ma50_slope': ma50_slope,
                'ma200_slope': ma200_slope,
                
                # Momentum features
                'rsi': rsi,
                'macd': macd,
                'macd_histogram': macd_histogram,
                'price_upper_ratio': price_upper_ratio,
                'price_lower_ratio': price_lower_ratio,
                
                # Recent returns
                'return_1d': df['returns'].iloc[-1] * 100,
                'return_5d': df['returns'].iloc[-5:].sum() * 100,
                'return_20d': df['returns'].iloc[-20:].sum() * 100,
            }
            
            # Add volume features if available
            features.update(volume_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            return {}
            
    def detect_regime_rule_based(self, features: Dict[str, float]) -> str:
        """
        Detect market regime using rule-based approach with the extracted features.
        
        Args:
            features: Dictionary of market features
            
        Returns:
            Detected market regime
        """
        try:
            if not features:
                return "unknown"
                
            # Extract key features
            vol_10d = features.get('volatility_10d', 0)
            vol_ratio = features.get('volatility_ratio', 1.0)
            ma50_200 = features.get('ma50_200_ratio', 1.0)
            ma50_slope = features.get('ma50_slope', 0)
            rsi = features.get('rsi', 50)
            macd_hist = features.get('macd_histogram', 0)
            price_ma200 = features.get('price_ma200_ratio', 1.0)
            
            # Detect high volatility regime
            if vol_10d > 0.4 or vol_ratio > 2.0:
                return "high_volatility"
                
            # Detect strong trend up
            if ma50_200 > 1.05 and ma50_slope > 1.0 and rsi > 60 and macd_hist > 0:
                return "trending_up_strong"
                
            # Detect weak trend up
            if ma50_200 > 1.02 and ma50_slope > 0.5 and rsi > 50:
                return "trending_up_weak"
                
            # Detect strong trend down
            if ma50_200 < 0.95 and ma50_slope < -1.0 and rsi < 40 and macd_hist < 0:
                return "trending_down_strong"
                
            # Detect weak trend down
            if ma50_200 < 0.98 and ma50_slope < -0.5 and rsi < 50:
                return "trending_down_weak"
                
            # Detect tight range
            if 0.98 < ma50_200 < 1.02 and abs(ma50_slope) < 0.5 and vol_10d < 0.2:
                return "ranging_tight"
                
            # Detect wide range
            if 0.97 < ma50_200 < 1.03 and abs(ma50_slope) < 0.8 and vol_10d >= 0.2:
                return "ranging_wide"
                
            # Detect low volatility
            if vol_10d < 0.15:
                return "low_volatility"
                
            # Detect breakout
            if (vol_ratio > 1.5 and abs(ma50_slope) > 1.0 and 
                ((ma50_slope > 0 and rsi > 70) or (ma50_slope < 0 and rsi < 30))):
                return "breakout"
                
            # Detect reversal
            if ((ma50_slope > 1.0 and rsi < 30) or (ma50_slope < -1.0 and rsi > 70)):
                return "reversal"
                
            # Default to ranging if no specific pattern detected
            return "ranging_wide"
            
        except Exception as e:
            logger.error(f"Error in rule-based regime detection: {e}")
            return "unknown"
            
    def update_clustering_model(self):
        """Update the clustering model with accumulated feature history"""
        if len(self.feature_history) < 20:
            return False
            
        try:
            # Convert feature history to numpy array
            feature_df = pd.DataFrame(self.feature_history)
            
            # Select numerical features
            numerical_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_array = feature_df[numerical_features].values
            
            # Fit scaler and transform data
            self.scaler.fit(feature_array)
            scaled_features = self.scaler.transform(feature_array)
            
            # Apply PCA if we have enough dimensions
            if scaled_features.shape[1] >= 3:
                self.pca.fit(scaled_features)
                reduced_features = self.pca.transform(scaled_features)
            else:
                reduced_features = scaled_features
            
            # Fit KMeans
            self.kmeans.fit(reduced_features)
            
            # Update cluster centroids
            for i in range(self.num_clusters):
                self.regime_clusters[f'cluster_{i}'] = {}
                
            # Mark as initialized
            self.cluster_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error updating clustering model: {e}")
            return False
            
    def detect_regime_clustering(self, features: Dict[str, float]) -> str:
        """
        Detect market regime using unsupervised clustering.
        
        Args:
            features: Dictionary of market features
            
        Returns:
            Detected regime cluster
        """
        if not self.cluster_initialized or not features:
            return "unknown"
            
        try:
            # Convert features to dataframe row
            feature_df = pd.DataFrame([features])
            
            # Get numerical features that match our model
            numerical_features = [f for f in feature_df.columns if f in self.scaler.feature_names_in_]
            
            if not numerical_features:
                return "unknown"
                
            # Extract and scale features
            feature_array = feature_df[numerical_features].values
            scaled_features = self.scaler.transform(feature_array)
            
            # Apply PCA
            reduced_features = self.pca.transform(scaled_features)
            
            # Predict cluster
            cluster = self.kmeans.predict(reduced_features)[0]
            
            return f"cluster_{cluster}"
            
        except Exception as e:
            logger.error(f"Error in clustering-based regime detection: {e}")
            return "unknown"
            
    def detect_regime(self, df: pd.DataFrame, column: str = "Close") -> str:
        """
        Detect current market regime using combined approach.
        
        Args:
            df: DataFrame with price data
            column: Price column name
            
        Returns:
            Detected market regime
        """
        try:
            # Extract market features
            features = self.extract_market_features(df, column)
            
            if not features:
                return "unknown"
                
            # Store in feature history
            self.feature_history.append(features)
            if len(self.feature_history) > self.feature_history_length:
                self.feature_history = self.feature_history[-self.feature_history_length:]
                
            # Get rule-based regime
            rule_regime = self.detect_regime_rule_based(features)
            
            # Update clustering model if needed
            if self.use_clustering and not self.cluster_initialized and len(self.feature_history) >= 20:
                self.update_clustering_model()
                
            # Get cluster-based regime if available
            cluster_regime = "unknown"
            if self.use_clustering and self.cluster_initialized:
                cluster_regime = self.detect_regime_clustering(features)
                
                # If we have cluster performance data, use it
                if cluster_regime in self.regime_performance:
                    # Check if we have enough data to trust this cluster
                    if len(self.regime_performance[cluster_regime]) >= 3:
                        rule_regime = cluster_regime
            
            # Update regime history
            previous_regime = self.current_regime
            self.current_regime = rule_regime
            
            # Record transition
            if previous_regime != self.current_regime:
                transition_key = f"{previous_regime}_to_{self.current_regime}"
                self.regime_transitions[transition_key] = self.regime_transitions.get(transition_key, 0) + 1
                self.regime_duration = 0
            else:
                self.regime_duration += 1
                
            # Record history entry
            history_entry = {
                'timestamp': datetime.now(),
                'regime': self.current_regime,
                'cluster': cluster_regime if self.use_clustering else "n/a",
                'features': features,
                'duration': self.regime_duration
            }
            
            self.regime_history.append(history_entry)
            if len(self.regime_history) > self.memory_length:
                self.regime_history = self.regime_history[-self.memory_length:]
                
            return self.current_regime
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"
            
    def update_model_performance(self, 
                               regime: str, 
                               model_type: str, 
                               metric_value: float,
                               adaptation_impact: Optional[float] = None):
        """
        Update performance tracking for a model in the given regime with adaptive learning rate.
        
        Args:
            regime: Market regime
            model_type: Type of model
            metric_value: Performance metric (lower is better)
            adaptation_impact: How much the previous adaptation affected performance
        """
        try:
            # Initialize regime dict if this is a new regime
            if regime not in self.regime_performance:
                self.regime_performance[regime] = {}
                
            # Get model-specific learning rate or use default
            learning_rate = self.model_learning_rates.get(model_type, self.learning_rate)
                
            # Update performance using exponential moving average
            if model_type in self.regime_performance[regime]:
                old_value = self.regime_performance[regime][model_type]
                new_value = (1 - learning_rate) * old_value + learning_rate * metric_value
                self.regime_performance[regime][model_type] = new_value
                
                # Record in performance history
                if model_type not in self.performance_history:
                    self.performance_history[model_type] = []
                    
                self.performance_history[model_type].append({
                    'timestamp': datetime.now(),
                    'regime': regime,
                    'old_value': old_value,
                    'new_value': new_value,
                    'raw_metric': metric_value
                })
                
                # Limit history size
                if len(self.performance_history[model_type]) > self.memory_length:
                    self.performance_history[model_type] = self.performance_history[model_type][-self.memory_length:]
                    
                # Meta-learning: update learning rate if adaptation impact is provided
                if adaptation_impact is not None:
                    # If adaptation had positive impact (lower metric), increase learning rate
                    # If adaptation had negative impact (higher metric), decrease learning rate
                    direction = -1 if adaptation_impact < 0 else 1  # Negative impact means decrease rate
                    adjustment = self.meta_learning_rate * abs(adaptation_impact) * direction
                    
                    # Update learning rate with bounds
                    current_rate = self.model_learning_rates.get(model_type, self.learning_rate)
                    new_rate = max(0.01, min(0.9, current_rate + adjustment))
                    self.model_learning_rates[model_type] = new_rate
                    
                    logger.debug(f"Updated learning rate for {model_type}: {current_rate:.4f} → {new_rate:.4f}")
                    
            else:
                # First entry for this model
                self.regime_performance[regime][model_type] = metric_value
                
                # Initialize in performance history
                if model_type not in self.performance_history:
                    self.performance_history[model_type] = []
                    
                self.performance_history[model_type].append({
                    'timestamp': datetime.now(),
                    'regime': regime,
                    'old_value': metric_value,
                    'new_value': metric_value,
                    'raw_metric': metric_value
                })
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            
    def get_optimal_weights(self, 
                          regime: str, 
                          base_weights: Dict[str, float],
                          context_similarity: float = 0.5) -> Dict[str, float]:
        """
        Calculate optimal model weights for the given regime with context-based adjustment.
        
        Args:
            regime: Current market regime
            base_weights: Base model weights
            context_similarity: How similar current context is to historical (0-1)
            
        Returns:
            Optimized model weights
        """
        try:
            # If regime not found or no performance data, return base weights
            if regime not in self.regime_performance or not self.regime_performance[regime]:
                return base_weights
                
            # Get performance metrics for this regime
            performances = self.regime_performance[regime]
            
            # Check for valid performance data
            valid_models = [m for m in performances.keys() if m in base_weights]
            if not valid_models:
                return base_weights
                
            # Convert metrics to weights (lower is better)
            # We use inverse weighting: 1/metric
            weights = {}
            total_inverse = 0
            
            for model in valid_models:
                # Add small epsilon to avoid division by zero
                inverse_metric = 1.0 / (performances[model] + 1e-6)
                weights[model] = inverse_metric
                total_inverse += inverse_metric
                
            # Normalize weights
            if total_inverse > 0:
                for model in weights:
                    weights[model] /= total_inverse
                    
            # Blend with base weights using context similarity as the blending factor
            # Higher similarity = more weight on historical performance
            # Lower similarity = more weight on base weights
            blended_weights = {}
            
            # Get all model types from both dictionaries
            all_models = set(weights.keys()) | set(base_weights.keys())
            
            for model in all_models:
                perf_weight = weights.get(model, 0.0)
                base_weight = base_weights.get(model, 0.0)
                # Contextual blending - more weight to historical performance if context is similar
                blended_weights[model] = context_similarity * perf_weight + (1 - context_similarity) * base_weight
                
            # Renormalize blended weights
            total = sum(blended_weights.values())
            if total > 0:
                normalized_weights = {model: w/total for model, w in blended_weights.items()}
                
                # Record adaptations for meta-learning
                for model in normalized_weights:
                    original = base_weights.get(model, 0.0)
                    adapted = normalized_weights[model]
                    # Record the absolute change amount for each model
                    adaptation = adapted - original
                    if model not in self.adaptation_performance:
                        self.adaptation_performance[model] = []
                    self.adaptation_performance[model].append({
                        'timestamp': datetime.now(),
                        'regime': regime,
                        'original_weight': original,
                        'adapted_weight': adapted,
                        'adaptation': adaptation,
                        'impact': None  # Will be updated when we measure performance
                    })
                    
                return normalized_weights
            
            return base_weights
                
        except Exception as e:
            logger.error(f"Error calculating optimal weights: {e}")
            return base_weights
            
    def measure_adaptation_impact(self, model_type: str, new_performance: float):
        """
        Measure how well our adaptation worked and use for meta-learning.
        
        Args:
            model_type: Type of model
            new_performance: New performance metric (lower is better)
        """
        if model_type not in self.adaptation_performance or not self.adaptation_performance[model_type]:
            return
            
        try:
            # Get the most recent adaptation
            adaptation = self.adaptation_performance[model_type][-1]
            
            # Get the previous performance in this regime
            regime = adaptation['regime']
            prev_performance = None
            
            if model_type in self.performance_history:
                for entry in reversed(self.performance_history[model_type]):
                    if entry['regime'] == regime:
                        prev_performance = entry['raw_metric']
                        break
                        
            if prev_performance is not None:
                # Calculate impact (negative is good because lower metrics are better)
                impact = new_performance - prev_performance
                
                # Update the impact in our record
                adaptation['impact'] = impact
                
                # Use this to update the learning rate
                self.update_model_performance(
                    regime=regime,
                    model_type=model_type,
                    metric_value=new_performance,
                    adaptation_impact=impact
                )
                
        except Exception as e:
            logger.error(f"Error measuring adaptation impact: {e}")
            
    def calculate_context_similarity(self, current_features: Dict[str, float]) -> float:
        """
        Calculate how similar current market context is to historical patterns.
        
        Args:
            current_features: Current market features
            
        Returns:
            Similarity score (0-1)
        """
        if not current_features or not self.feature_history:
            return 0.5  # Default middle value when insufficient data
            
        try:
            # Convert current features to vector
            feature_keys = sorted(current_features.keys())
            current_vector = np.array([current_features.get(k, 0) for k in feature_keys])
            
            # Get historical features for current regime
            regime_features = []
            for entry in self.regime_history:
                if entry['regime'] == self.current_regime:
                    features = entry['features']
                    feature_vector = np.array([features.get(k, 0) for k in feature_keys])
                    regime_features.append(feature_vector)
                    
            if not regime_features:
                return 0.5  # No historical data for this regime
                
            # Calculate similarity using cosine similarity
            similarities = []
            for hist_vector in regime_features:
                # Ensure vectors are the same length
                min_len = min(len(current_vector), len(hist_vector))
                if min_len == 0:
                    continue
                    
                hist_vector = hist_vector[:min_len]
                curr_vector = current_vector[:min_len]
                
                # Calculate cosine similarity
                dot_product = np.dot(curr_vector, hist_vector)
                norm_product = np.linalg.norm(curr_vector) * np.linalg.norm(hist_vector)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append(max(0, similarity))  # Only positive similarities
                    
            if similarities:
                # Use exponential weighting to emphasize recent similarities
                weights = np.exp(np.linspace(-1, 0, len(similarities)))
                weights /= weights.sum()
                
                weighted_similarity = np.sum(weights * np.array(similarities))
                # Normalize to 0-1 range
                return (weighted_similarity + 1) / 2
                
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating context similarity: {e}")
            return 0.5
            
    def save_memory(self, filepath: Optional[str] = None):
        """
        Save the market regime memory to file.
        
        Args:
            filepath: Path to save the file (uses self.regime_memory_file if None)
        """
        filepath = filepath or self.regime_memory_file
        if not filepath:
            logger.warning("No filepath specified for saving market regime memory")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare data for saving
            memory_data = {
                'regime_performance': self.regime_performance,
                'regime_history': self.regime_history[-100:],  # Save last 100 entries
                'model_learning_rates': self.model_learning_rates,
                'regime_transitions': self.regime_transitions,
                'current_regime': self.current_regime,
                'regime_duration': self.regime_duration,
                'adaptation_performance': {k: v[-20:] for k, v in self.adaptation_performance.items()},  # Last 20 entries
                'version': '1.0',
                'last_update': datetime.now().isoformat(),
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)  # Use str for non-serializable objects
                
            logger.info(f"Saved market regime memory to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving market regime memory: {e}")
            return False
            
    def load_memory(self, filepath: Optional[str] = None):
        """
        Load the market regime memory from file.
        
        Args:
            filepath: Path to load the file from (uses self.regime_memory_file if None)
        """
        filepath = filepath or self.regime_memory_file
        if not filepath or not os.path.exists(filepath):
            logger.warning(f"Market regime memory file not found: {filepath}")
            return False
            
        try:
            # Load from file
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
                
            # Restore data
            if 'regime_performance' in memory_data:
                self.regime_performance = memory_data['regime_performance']
                
            if 'regime_history' in memory_data:
                self.regime_history = memory_data['regime_history']
                
            if 'model_learning_rates' in memory_data:
                self.model_learning_rates = memory_data['model_learning_rates']
                
            if 'regime_transitions' in memory_data:
                self.regime_transitions = memory_data['regime_transitions']
                
            if 'current_regime' in memory_data:
                self.current_regime = memory_data['current_regime']
                
            if 'regime_duration' in memory_data:
                self.regime_duration = memory_data['regime_duration']
                
            if 'adaptation_performance' in memory_data:
                self.adaptation_performance = memory_data['adaptation_performance']
                
            logger.info(f"Loaded market regime memory from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading market regime memory: {e}")
            return False
            
    def get_regime_stats(self) -> Dict:
        """
        Get statistics about market regimes.
        
        Returns:
            Dictionary with regime statistics
        """
        stats = {
            'current_regime': self.current_regime,
            'previous_regime': self.previous_regime,
            'regime_duration': self.regime_duration,
            'regime_counts': {},
            'regime_performance': {},
            'model_learning_rates': self.model_learning_rates,
            'last_update': self.last_update.isoformat(),
        }
        
        # Count occurrences of each regime
        for entry in self.regime_history:
            regime = entry['regime']
            stats['regime_counts'][regime] = stats['regime_counts'].get(regime, 0) + 1
            
        # Get performance data for each regime
        for regime, performances in self.regime_performance.items():
            if performances:  # Only include non-empty regimes
                stats['regime_performance'][regime] = performances
                
        return stats

    def get_model_weight_history(self, model_type: str) -> List[Dict]:
        """
        Get weight adaptation history for a specific model.
        
        Args:
            model_type: Type of model
            
        Returns:
            List of weight adaptation records
        """
        if model_type not in self.adaptation_performance:
            return []
            
        return self.adaptation_performance[model_type]