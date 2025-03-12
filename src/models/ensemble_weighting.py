import json
import os
from collections import deque
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Add this import


class EnsembleWeighter:
    """Base class for ensemble weight management"""

    def __init__(self, base_weights, learning_rate=0.01):
        """
        Initialize the ensemble weighter.

        Args:
            base_weights: Dictionary mapping model names to their base weights
            learning_rate: Rate at which weights are updated
        """
        self.base_weights = base_weights
        self.current_weights = base_weights.copy()
        self.learning_rate = learning_rate
        self.performance_history = {model: [] for model in base_weights.keys()}

    def update(self, model_predictions, actual_values):
        """
        Update weights based on model performance.

        Args:
            model_predictions: Dictionary mapping model names to their predictions
            actual_values: Actual observed values
        """
        # Calculate error for each model
        errors = {}
        for model, preds in model_predictions.items():
            if len(preds) != len(actual_values):
                continue

            error = mean_squared_error(actual_values, preds)
            errors[model] = error
            self.performance_history[model].append(error)

        # Skip if no valid errors
        if not errors:
            return

        # Convert errors to scores (lower error = higher score)
        scores = {}
        for model, error in errors.items():
            scores[model] = 1.0 / (
                error + 1e-8
            )  # Add small epsilon to avoid division by zero

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            norm_scores = {
                model: score / total_score for model, score in scores.items()
            }

            # Update weights using simple gradient step
            for model in self.current_weights:
                if model in norm_scores:
                    self.current_weights[model] += self.learning_rate * (
                        norm_scores[model] - self.current_weights[model]
                    )

            # Normalize weights
            self._normalize_weights()

    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for model in self.current_weights:
                self.current_weights[model] /= total_weight

    def get_weights(self):
        """Return the current weights"""
        return self.current_weights.copy()


class AdvancedEnsembleWeighter(EnsembleWeighter):
    def __init__(
        self,
        base_weights,
        adaptation_rate=0.05,
        short_window=10,
        medium_window=50,
        long_window=200,
        volatility_window=20,
        regime_sensitivity=0.3,
        model_id=None,
        weights_path=None,
        learning_rate=0.01,
    ):
        """
        Initialize the advanced ensemble weighter with adaptive capabilities.

        Args:
            base_weights: Dictionary mapping model names to their base weights
            adaptation_rate: Rate at which weights adapt to new performance
            short_window: Window size for short-term performance tracking
            medium_window: Window size for medium-term performance tracking
            long_window: Window size for long-term performance tracking
            volatility_window: Window size for volatility calculation
            regime_sensitivity: How sensitive the ensemble is to regime changes
            model_id: Unique identifier for this ensemble
            weights_path: Path to save/load weights
            learning_rate: Rate at which weights are updated (passed to parent)
        """
        # Call parent's init
        super().__init__(base_weights, learning_rate)

        # Additional attributes for advanced weighter
        self.adaptation_rate = adaptation_rate
        self.model_id = model_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Time windows for different measures
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.volatility_window = volatility_window
        self.regime_sensitivity = regime_sensitivity

        # Performance tracking (extends parent's tracking)
        self.error_history = {
            model: deque(maxlen=long_window) for model in base_weights.keys()
        }
        self.direction_history = {
            model: deque(maxlen=long_window) for model in base_weights.keys()
        }
        self.historical_weights = []
        self.weight_change_reasons = []

        # Market regime tracking
        self.current_regime = "unknown"
        self.regime_history = deque(maxlen=long_window)
        self.regime_performance = {}

        # Model correlation tracking
        self.model_correlation_matrix = {}

        # Exponential moving averages for different time horizons
        self.short_term_ema = {model: None for model in base_weights.keys()}
        self.medium_term_ema = {model: None for model in base_weights.keys()}
        self.long_term_ema = {model: None for model in base_weights.keys()}

        # Optuna feedback
        self.optuna_feedback = {
            "suggested_adjustments": [],
            "regime_performance": {},
            "model_performance": {},
        }

        # Save initial weights
        self.historical_weights.append(self.current_weights.copy())

        # Load weights if path provided
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
        elif weights_path:
            self._save_weights(weights_path)

        self.weights_path = weights_path

    # Override parent's update method
    def update(self, model_errors, model_direction_accuracy=None, price_data=None):
        """
        Update ensemble weights based on model performance.

        Args:
            model_errors: Dictionary mapping model names to their errors
            model_direction_accuracy: Dictionary mapping model names to directional accuracy
            price_data: Recent price data for regime detection
        """
        if not model_errors:
            return

        # Update error history
        for model, error in model_errors.items():
            if model in self.error_history:
                self.error_history[model].append(error)

                # Also update the parent class's performance history
                self.performance_history[model].append(error)

                # Update EMAs
                self._update_ema("short", model, error)
                self._update_ema("medium", model, error)
                self._update_ema("long", model, error)

        # Update direction accuracy if provided
        if model_direction_accuracy:
            for model, accuracy in model_direction_accuracy.items():
                if model in self.direction_history:
                    self.direction_history[model].append(accuracy)

        # Detect market regime if price data provided
        if price_data is not None:
            current_regime = self._detect_market_regime(price_data)
            self.regime_history.append(current_regime)
            self.current_regime = current_regime

        # Update model correlations
        self._update_model_correlations()

        # Calculate new weights
        self._recalculate_weights()

        # Normalize weights (use parent's method)
        super()._normalize_weights()

        # Save history
        self.historical_weights.append(self.current_weights.copy())

        # Save weights to file if path exists
        if self.weights_path:
            self._save_weights(self.weights_path)

    def _update_ema(self, timeframe, model, value, alpha=None):
        """
        Update exponential moving average for a model.

        Args:
            timeframe: Which timeframe to update ('short', 'medium', 'long')
            model: Model name
            value: New value (typically error)
            alpha: Smoothing factor (defaults to standard value for the timeframe)
        """
        # Set default alpha based on timeframe
        if alpha is None:
            if timeframe == "short":
                alpha = 2.0 / (self.short_window + 1)
            elif timeframe == "medium":
                alpha = 2.0 / (self.medium_window + 1)
            elif timeframe == "long":
                alpha = 2.0 / (self.long_window + 1)
            else:
                alpha = 0.2  # Default

        # Select the appropriate EMA dictionary
        if timeframe == "short":
            ema_dict = self.short_term_ema
        elif timeframe == "medium":
            ema_dict = self.medium_term_ema
        elif timeframe == "long":
            ema_dict = self.long_term_ema
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # Initialize EMA if it doesn't exist
        if ema_dict[model] is None:
            ema_dict[model] = value
        # Update EMA with new value
        else:
            ema_dict[model] = alpha * value + (1 - alpha) * ema_dict[model]

    def _detect_market_regime(self, price_data):
        """
        Detect the current market regime based on price data.

        Args:
            price_data: DataFrame with price information

        Returns:
            String identifying the current regime
        """
        # Implement a simple regime detection algorithm
        if len(price_data) < self.volatility_window:
            return "unknown"

        # Calculate returns
        returns = price_data["close"].pct_change().dropna()

        # Calculate volatility (standard deviation of returns)
        volatility = returns[-self.volatility_window :].std()

        # Calculate recent trend
        recent_trend = (
            price_data["close"].iloc[-1]
            / price_data["close"].iloc[-min(20, len(price_data))]
            - 1
        )

        # Determine regime based on volatility and trend
        if volatility > 0.03:  # High volatility threshold
            if recent_trend > 0.05:
                return "volatile_bullish"
            elif recent_trend < -0.05:
                return "volatile_bearish"
            else:
                return "volatile_sideways"
        else:
            if recent_trend > 0.03:
                return "trend_bullish"
            elif recent_trend < -0.03:
                return "trend_bearish"
            else:
                return "range_bound"

    def _update_model_correlations(self):
        """Update correlation matrix between model performances"""
        # Need enough history to calculate correlations
        min_history = min(len(history) for history in self.error_history.values())
        if min_history < 5:  # Need at least 5 points for meaningful correlation
            return

        # Create DataFrame of errors
        error_df = pd.DataFrame(
            {
                model: list(errors)[-min_history:]
                for model, errors in self.error_history.items()
            }
        )

        # Calculate correlation matrix
        corr_matrix = error_df.corr()

        # Convert to dictionary format
        for model1 in self.base_weights.keys():
            for model2 in self.base_weights.keys():
                if model1 != model2:
                    self.model_correlation_matrix[(model1, model2)] = corr_matrix.loc[
                        model1, model2
                    ]

    def _recalculate_weights(self):
        """Recalculate weights based on performance metrics"""
        # Skip if not enough history
        if not all(
            len(errors) >= self.short_window for errors in self.error_history.values()
        ):
            return

        new_weights = self.current_weights.copy()

        # Calculate performance scores (lower is better since these are errors)
        performance_scores = {}

        for model in self.base_weights.keys():
            if model not in self.error_history or not self.error_history[model]:
                continue

            # Use short-term EMA of errors for recent performance
            if self.short_term_ema[model] is not None:
                short_term_score = 1.0 / (self.short_term_ema[model] + 1e-6)
                performance_scores[model] = short_term_score

        # Skip if no performance scores
        if not performance_scores:
            return

        # Normalize performance scores
        total_score = sum(performance_scores.values())
        if total_score > 0:
            norm_scores = {
                model: score / total_score
                for model, score in performance_scores.items()
            }

            # Adjust weights based on scores and adaptation rate
            for model in new_weights.keys():
                if model in norm_scores:
                    target_weight = norm_scores[model]
                    new_weights[model] = (
                        new_weights[model] * (1 - self.adaptation_rate)
                        + target_weight * self.adaptation_rate
                    )

        # Adjust based on regime performance if we have regime history
        if len(self.regime_history) > 10 and self.current_regime != "unknown":
            self._adjust_for_regime(new_weights)

        # Record reason for significant weight changes
        self._track_weight_changes(new_weights)

        # Update current weights
        self.current_weights = new_weights

    def _adjust_for_regime(self, weights):
        """Adjust weights based on historical performance in the current regime"""
        if self.current_regime not in self.regime_performance:
            return

        regime_perf = self.regime_performance[self.current_regime]

        # Skip if we don't have performance data for all models
        if not all(model in regime_perf for model in weights.keys()):
            return

        # Get relative performance in this regime (higher is better)
        total_perf = sum(1.0 / (perf + 1e-6) for perf in regime_perf.values())
        if total_perf == 0:
            return

        norm_perf = {
            model: (1.0 / (perf + 1e-6)) / total_perf
            for model, perf in regime_perf.items()
        }

        # Adjust weights based on regime performance
        for model in weights:
            if model in norm_perf:
                weights[model] = (
                    weights[model] * (1 - self.regime_sensitivity)
                    + norm_perf[model] * self.regime_sensitivity
                )

    def _track_weight_changes(self, new_weights):
        """Track reasons for significant weight changes"""
        if not self.current_weights:
            return

        significant_change = False
        change_reasons = []

        for model, new_weight in new_weights.items():
            old_weight = self.current_weights.get(model, 0)
            if abs(new_weight - old_weight) > 0.05:  # 5% threshold for "significant"
                significant_change = True
                direction = "increased" if new_weight > old_weight else "decreased"
                reason = f"{model} weight {direction} due to "

                # Determine reason based on performance metrics
                if (
                    self.short_term_ema.get(model) is not None
                    and self.medium_term_ema.get(model) is not None
                ):
                    if self.short_term_ema[model] < self.medium_term_ema[model]:
                        reason += "improving short-term performance"
                    else:
                        reason += "declining short-term performance"
                else:
                    reason += "performance adjustment"

                change_reasons.append(reason)

        # Add regime change as a reason if applicable
        if (
            len(self.regime_history) >= 2
            and self.regime_history[-1] != self.regime_history[-2]
        ):
            significant_change = True
            change_reasons.append(f"Market regime changed to {self.regime_history[-1]}")

        # Record significant changes
        if significant_change:
            self.weight_change_reasons.append(
                {
                    "timestamp": datetime.now(),
                    "weights_before": self.current_weights.copy(),
                    "weights_after": new_weights.copy(),
                    "regime": self.current_regime,
                    "reasons": change_reasons,
                }
            )

    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for model in self.current_weights:
                self.current_weights[model] /= total_weight

    def _save_weights(self, path):
        """Save weights and history to file"""
        data = {
            "model_id": self.model_id,
            "current_weights": self.current_weights,
            "base_weights": self.base_weights,
            "adaptation_rate": self.adaptation_rate,
            "current_regime": self.current_regime,
            "last_updated": datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_weights(self, path):
        """Load weights from file"""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.current_weights = data.get("current_weights", self.base_weights.copy())
            self.model_id = data.get("model_id", self.model_id)
            self.current_regime = data.get("current_regime", "unknown")

            # Ensure all models have weights
            for model in self.base_weights:
                if model not in self.current_weights:
                    self.current_weights[model] = self.base_weights[model]

            self._normalize_weights()

        except Exception as e:
            print(f"Error loading weights from {path}: {e}")


"""
Advanced ensemble weighting strategies with performance tracking and adaptive mechanisms.
"""


class AdaptiveEnsembleWeighter:
    """
    Dynamically adjust ensemble weights based on recent performance.
    Uses a combination of performance history, exploration, and stability measures.
    """

    def __init__(
        self,
        model_types: List[str],
        memory_factor: float = 0.7,
        min_weight: float = 0.05,
        exploration_factor: float = 0.1,
        performance_window: int = 5,
    ):
        """
        Initialize the ensemble weighter.

        Args:
            model_types: List of model types to include
            memory_factor: Weight of previous weights vs new performance (0-1)
            min_weight: Minimum weight for any model
            exploration_factor: Weight added for exploration
            performance_window: Number of recent performances to consider
        """
        self.model_types = model_types
        self.memory_factor = memory_factor
        self.min_weight = min_weight
        self.exploration_factor = exploration_factor
        self.performance_window = performance_window

        # Initialize weights equally
        self.weights = {mtype: 1.0 / len(model_types) for mtype in model_types}

        # Initialize performance history
        self.performance_history = {mtype: [] for mtype in model_types}

    def update_performance(self, model_type: str, metrics: Dict[str, float]) -> None:
        """
        Update performance history for a model.

        Args:
            model_type: Type of model
            metrics: Dictionary of performance metrics
        """
        if model_type not in self.model_types:
            return

        # Extract the main metric (e.g., RMSE)
        metric_value = metrics.get("rmse", None)
        if metric_value is None:
            return

        # Add to history
        self.performance_history[model_type].append(metric_value)

        # Trim history to window size
        if len(self.performance_history[model_type]) > self.performance_window:
            self.performance_history[model_type] = self.performance_history[model_type][
                -self.performance_window :
            ]

    def update_weights(self) -> Dict[str, float]:
        """
        Update weights based on recent performance.

        Returns:
            Dictionary of updated weights
        """
        new_weights = {}
        total_inverse_metric = 0.0

        # Calculate performance-based weights
        for mtype in self.model_types:
            history = self.performance_history[mtype]

            if not history:
                # No performance data yet, use previous weight
                new_weights[mtype] = self.weights.get(
                    mtype, 1.0 / len(self.model_types)
                )
                continue

            # Use average of recent performance (lower is better)
            avg_metric = np.mean(history)

            # Avoid division by zero and handle negative metrics
            if avg_metric <= 0:
                inverse_metric = 0.0
            else:
                inverse_metric = 1.0 / avg_metric

            new_weights[mtype] = inverse_metric
            total_inverse_metric += inverse_metric

        # Normalize performance-based weights
        if total_inverse_metric > 0:
            for mtype in self.model_types:
                new_weights[mtype] = new_weights[mtype] / total_inverse_metric

        # Apply memory factor - blend with previous weights
        if self.memory_factor > 0:
            for mtype in self.model_types:
                new_weights[mtype] = (
                    self.memory_factor
                    * self.weights.get(mtype, 1.0 / len(self.model_types))
                    + (1.0 - self.memory_factor) * new_weights[mtype]
                )

        # Apply minimum weight
        for mtype in self.model_types:
            new_weights[mtype] = max(self.min_weight, new_weights[mtype])

        # Add exploration factor
        if self.exploration_factor > 0:
            for mtype in self.model_types:
                new_weights[mtype] += self.exploration_factor

        # Normalize final weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for mtype in self.model_types:
                new_weights[mtype] /= total_weight

        # Update and return weights
        self.weights = new_weights
        return self.weights

    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights.

        Returns:
            Dictionary of current weights
        """
        return self.weights


class BayesianEnsembleWeighter:
    """
    Bayesian approach to ensemble weighting.
    Uses Thompson sampling to balance exploration vs. exploitation.
    """

    def __init__(
        self, model_types: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0
    ):
        """
        Initialize the Bayesian ensemble weighter.

        Args:
            model_types: List of model types to include
            prior_alpha: Prior alpha parameter for Beta distribution
            prior_beta: Prior beta parameter for Beta distribution
        """
        self.model_types = model_types

        # Initialize prior parameters
        self.alphas = {mtype: prior_alpha for mtype in model_types}
        self.betas = {mtype: prior_beta for mtype in model_types}

        # Initialize weights
        self.weights = {mtype: 1.0 / len(model_types) for mtype in model_types}

    def update_performance(self, model_type: str, metrics: Dict[str, float]) -> None:
        """
        Update posterior distribution for a model based on new performance.

        Args:
            model_type: Type of model
            metrics: Dictionary of performance metrics
        """
        if model_type not in self.model_types:
            return

        # Extract RMSE or other metrics
        rmse = metrics.get("rmse", None)
        mape = metrics.get("mape", None)

        if rmse is None:
            return

        # Convert RMSE to success/failure metric (smaller is better)
        # We use a sigmoid transformation to map RMSE to (0,1)
        baseline_rmse = 0.01  # Adjust based on your domain
        success_rate = 1 / (1 + rmse / baseline_rmse)

        # Update posterior
        self.alphas[model_type] += success_rate
        self.betas[model_type] += 1 - success_rate

    def sample_weights(self) -> Dict[str, float]:
        """
        Sample weights from posterior distributions.

        Returns:
            Dictionary of sampled weights
        """
        from numpy.random import beta

        # Draw samples from Beta distributions
        samples = {
            mtype: beta(self.alphas[mtype], self.betas[mtype])
            for mtype in self.model_types
        }

        # Normalize to get weights
        total = sum(samples.values())
        if total > 0:
            weights = {mtype: value / total for mtype, value in samples.items()}
        else:
            # Fallback to equal weights
            weights = {mtype: 1.0 / len(self.model_types) for mtype in self.model_types}

        self.weights = weights
        return weights

    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights.

        Returns:
            Dictionary of current weights
        """
        return self.weights

    def get_expected_weights(self) -> Dict[str, float]:
        """
        Get expected weights based on posterior means.

        Returns:
            Dictionary of expected weights
        """
        # Calculate posterior means
        means = {
            mtype: self.alphas[mtype] / (self.alphas[mtype] + self.betas[mtype])
            for mtype in self.model_types
        }

        # Normalize
        total = sum(means.values())
        if total > 0:
            return {mtype: value / total for mtype, value in means.items()}
        else:
            # Fallback to equal weights
            return {mtype: 1.0 / len(self.model_types) for mtype in self.model_types}


def get_ensemble_weighter_factory(method: str = "adaptive"):
    """
    Factory function to get the appropriate ensemble weighter.

    Args:
        method: Weighting method ("adaptive", "bayesian", or "simple")

    Returns:
        Function to create an ensemble weighter
    """
    if method == "adaptive":
        return lambda model_types, **kwargs: AdaptiveEnsembleWeighter(
            model_types, **kwargs
        )
    elif method == "bayesian":
        return lambda model_types, **kwargs: BayesianEnsembleWeighter(
            model_types, **kwargs
        )
    else:  # "simple" or fallback
        from src.models.ensemble_utils import get_ensemble_weighter

        return lambda model_types, **kwargs: {
            "get_weights": lambda: get_ensemble_weighter("equal")(model_types, None),
            "update_performance": lambda *args, **kwargs: None,
            "update_weights": lambda: get_ensemble_weighter("equal")(model_types, None),
        }
