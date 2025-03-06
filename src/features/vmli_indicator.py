from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


class VMILIndicator:
    """
    Volatility-Adjusted Momentum Liquidity Index (VMLI) Indicator

    A technical indicator that combines price momentum, volatility adjustment, and volume/liquidity
    factors to create a more robust trading signal for both stock and cryptocurrency markets.

    The VMLI is calculated in several steps:
    1. Momentum Component: Measures recent price change
    2. Volatility Adjustment: Normalizes momentum by recent volatility
    3. Liquidity/Volume Component: Factors in trading activity
    4. Composite Index: Combines these factors and smooths the result
    """

    def __init__(
        self,
        window_mom: int = 14,
        window_vol: int = 14,
        smooth_period: int = 3,
        winsorize_pct: float = 0.01,
        use_ema: bool = True,
    ):
        """
        Initialize the VMLI indicator with configurable parameters.

        Args:
            window_mom: Lookback window for momentum calculation (default: 14 periods)
            window_vol: Lookback window for volatility calculation (default: 14 periods)
            smooth_period: Period for final smoothing of the indicator (default: 3)
            winsorize_pct: Percentile for winsorizing extreme values (default: 0.01 or 1%)
            use_ema: Whether to use EMA (True) or SMA (False) for smoothing (default: True)
        """
        self.window_mom = window_mom
        self.window_vol = window_vol
        self.smooth_period = smooth_period
        self.winsorize_pct = winsorize_pct
        self.use_ema = use_ema

    def compute(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
        volume_col: Optional[str] = "volume",
        order_book: Optional[pd.DataFrame] = None,
        include_components: bool = False,
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Compute the VMLI indicator for the given price and volume data.

        Args:
            data: DataFrame with price and volume data (must contain at least price_col)
            price_col: Column name for price data (default: 'close')
            volume_col: Column name for volume data (default: 'volume')
            order_book: Optional DataFrame with order book data (bid/ask volumes)
            include_components: Whether to return individual components (default: False)

        Returns:
            If include_components is False: pd.Series with VMLI values
            If include_components is True: Dict of pd.Series with VMLI and its components
        """
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")

        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()
        price_data = df[price_col]

        # 1. Calculate momentum component
        # Use percentage change over the momentum window
        momentum = self._calculate_momentum(price_data)

        # 2. Calculate volatility adjustment
        volatility = self._calculate_volatility(price_data)

        # 3. Calculate adjusted momentum (momentum / volatility)
        # Handle division by zero or very small values
        volatility = volatility.replace(0, np.nan)  # Replace zeros with NaN
        adj_momentum = momentum / volatility

        # 4. Calculate liquidity/volume component
        if volume_col in df.columns:
            liquidity = self._calculate_liquidity(df[volume_col], order_book)
        else:
            liquidity = pd.Series(
                1.0, index=df.index
            )  # Default to 1.0 if no volume data

        # 5. Calculate raw VMLI
        vmli_raw = adj_momentum * liquidity

        # 6. Winsorize extreme values for robustness
        if self.winsorize_pct > 0:
            vmli_raw = self._winsorize(vmli_raw, self.winsorize_pct)

        # 7. Smooth the final indicator
        vmli = self._smooth(vmli_raw)

        if include_components:
            return {
                "momentum": momentum,
                "volatility": volatility,
                "adj_momentum": adj_momentum,
                "liquidity": liquidity,
                "vmli_raw": vmli_raw,
                "vmli": vmli,
            }
        else:
            return vmli

    def _calculate_momentum(self, price_data: pd.Series) -> pd.Series:
        """Calculate the momentum component using an EMA of returns."""
        # Calculate daily returns first
        daily_returns = price_data.pct_change()

        # Calculate momentum as an n-day EMA of returns
        if self.use_ema:
            momentum = daily_returns.ewm(span=self.window_mom).mean()
        else:
            momentum = daily_returns.rolling(window=self.window_mom).mean()

        return momentum

    def _calculate_volatility(self, price_data: pd.Series) -> pd.Series:
        """Calculate the volatility component (standard deviation of returns)."""
        daily_returns = price_data.pct_change()

        # Use rolling standard deviation of returns
        volatility = daily_returns.rolling(window=self.window_vol).std()

        # Ensure we don't have zero volatility (which would cause division by zero)
        min_vol = (
            volatility[volatility > 0].min()
            if len(volatility[volatility > 0]) > 0
            else 0.0001
        )
        volatility = volatility.clip(lower=min_vol)

        return volatility

    def _calculate_liquidity(
        self, volume_data: pd.Series, order_book: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate the liquidity component based on volume and optional order book data.

        Args:
            volume_data: Series of volume data
            order_book: Optional DataFrame with columns ['timestamp', 'bid_volume', 'ask_volume']
                        for order book imbalance calculation
        """
        # Calculate volume factor (current volume relative to recent median)
        vol_median = volume_data.rolling(window=self.window_mom).median()
        vol_factor = volume_data / vol_median.replace(
            0, np.nan
        )  # Avoid division by zero

        # Apply a reasonable cap to the volume factor to prevent extreme values
        vol_factor = vol_factor.clip(upper=5.0)

        # Fill NaN values with 1.0 (neutral)
        vol_factor = vol_factor.fillna(1.0)

        # Incorporate order book data if available
        if (
            order_book is not None
            and "bid_volume" in order_book.columns
            and "ask_volume" in order_book.columns
        ):
            # Calculate order book imbalance
            obi = (order_book["bid_volume"] - order_book["ask_volume"]) / (
                order_book["bid_volume"] + order_book["ask_volume"]
            )

            # Convert OBI to same index as volume data
            obi = obi.reindex(volume_data.index, method="ffill")

            # Combine volume factor with OBI (scale OBI to be centered around 1)
            liquidity = vol_factor * (1 + obi)
        else:
            # If no order book data, just use volume factor
            liquidity = vol_factor

        return liquidity

    def _winsorize(self, series: pd.Series, percentile: float) -> pd.Series:
        """
        Winsorize a series to limit extreme values.

        Args:
            series: The data series to winsorize
            percentile: The percentile threshold (e.g., 0.01 for 1%)

        Returns:
            Winsorized series
        """
        lower_bound = series.quantile(percentile)
        upper_bound = series.quantile(1 - percentile)
        return series.clip(lower=lower_bound, upper=upper_bound)

    def _smooth(self, series: pd.Series) -> pd.Series:
        """Apply final smoothing to the indicator."""
        if self.use_ema:
            return series.ewm(span=self.smooth_period).mean()
        else:
            return series.rolling(window=self.smooth_period).mean()

    def generate_signals(
        self,
        vmli: pd.Series,
        buy_threshold: float = 1.0,
        sell_threshold: float = -1.0,
        use_crossover: bool = False,
    ) -> pd.Series:
        """
        Generate trading signals based on VMLI values.

        Args:
            vmli: Series of VMLI values
            buy_threshold: Threshold for buy signals (default: 1.0)
            sell_threshold: Threshold for sell signals (default: -1.0)
            use_crossover: Whether to use threshold crossovers instead of absolute values

        Returns:
            Series with values: 1 (buy), -1 (sell), 0 (hold)
        """
        signals = pd.Series(0, index=vmli.index)  # Initialize with holds (0)

        if use_crossover:
            # Generate signals based on threshold crossovers
            buy_signal = (vmli > buy_threshold) & (vmli.shift(1) <= buy_threshold)
            sell_signal = (vmli < sell_threshold) & (vmli.shift(1) >= sell_threshold)
        else:
            # Generate signals based on absolute values
            buy_signal = vmli > buy_threshold
            sell_signal = vmli < sell_threshold

        signals[buy_signal] = 1  # Buy
        signals[sell_signal] = -1  # Sell

        return signals

    def backtest(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
        initial_capital: float = 10000.0,
        position_size: float = 1.0,
        commission: float = 0.001,
        use_crossover: bool = True,
        buy_threshold: float = 1.0,
        sell_threshold: float = -1.0,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Run a simple backtest using the VMLI indicator.

        Args:
            data: DataFrame with price data
            price_col: Column name for price data
            initial_capital: Starting capital for the backtest
            position_size: Fraction of capital to use per trade (1.0 = 100%)
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
            use_crossover: Whether to use threshold crossovers for signals
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals

        Returns:
            Tuple of (results_df, performance_metrics)
        """
        # Calculate VMLI
        vmli = self.compute(data, price_col)

        # Generate signals
        signals = self.generate_signals(
            vmli,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            use_crossover=use_crossover,
        )

        # Initialize results DataFrame
        results = pd.DataFrame(index=data.index)
        results["price"] = data[price_col]
        results["vmli"] = vmli
        results["signal"] = signals

        # Initialize portfolio metrics
        results["position"] = 0  # Current position: 0=none, 1=long, -1=short
        results["cash"] = initial_capital
        results["holdings"] = 0.0
        results["equity"] = initial_capital

        # Variable to track entry price for calculating P&L
        entry_price = 0.0

        # Simulate trading
        for i in range(1, len(results)):
            # Get current signal and previous position
            current_signal = results.iloc[i]["signal"]
            prev_position = results.iloc[i - 1]["position"]
            current_price = results.iloc[i]["price"]
            prev_cash = results.iloc[i - 1]["cash"]
            prev_holdings = results.iloc[i - 1]["holdings"]

            # Initialize with previous values
            position = prev_position
            cash = prev_cash
            holdings = prev_holdings

            # Update position based on signals
            if current_signal == 1 and prev_position <= 0:  # Buy signal
                # Calculate shares to buy
                if prev_position < 0:  # Cover short first
                    # Calculate P&L from short position
                    profit = (entry_price - current_price) * abs(prev_holdings)
                    cash = prev_cash + profit

                # Now go long
                trade_value = cash * position_size
                shares = trade_value / current_price
                commission_cost = trade_value * commission

                # Update portfolio
                cash -= trade_value + commission_cost
                holdings = shares
                position = 1
                entry_price = current_price

            elif current_signal == -1 and prev_position >= 0:  # Sell signal
                if prev_position > 0:  # Close long position
                    # Calculate P&L from long position
                    trade_value = prev_holdings * current_price
                    profit = trade_value - (prev_holdings * entry_price)
                    commission_cost = trade_value * commission

                    # Update portfolio
                    cash = prev_cash + trade_value - commission_cost
                    holdings = 0

                # Now go short if allowed
                trade_value = cash * position_size
                shares = trade_value / current_price
                commission_cost = trade_value * commission

                # Update portfolio
                cash -= commission_cost
                holdings = -shares  # Negative for short position
                position = -1
                entry_price = current_price

            # Update results
            results.loc[results.index[i], "position"] = position
            results.loc[results.index[i], "cash"] = cash
            results.loc[results.index[i], "holdings"] = holdings

            # Calculate equity
            if position == 0:
                equity = cash
            elif position == 1:
                equity = cash + (holdings * current_price)
            else:  # position == -1
                equity = cash - (holdings * current_price)

            results.loc[results.index[i], "equity"] = equity

        # Calculate performance metrics
        initial_equity = results.iloc[0]["equity"]
        final_equity = results.iloc[-1]["equity"]
        total_return = (final_equity / initial_equity) - 1

        # Calculate daily returns
        results["daily_return"] = results["equity"].pct_change()

        # Calculate drawdown
        results["peak"] = results["equity"].cummax()
        results["drawdown"] = (results["equity"] - results["peak"]) / results["peak"]
        max_drawdown = results["drawdown"].min()

        # Calculate Sharpe and Sortino ratios
        risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1

        excess_returns = results["daily_return"] - daily_risk_free
        volatility = results["daily_return"].std() * np.sqrt(252)  # Annualized

        sharpe_ratio = (
            (excess_returns.mean() * 252) / volatility if volatility > 0 else 0
        )

        # For Sortino: only consider downside volatility
        downside_returns = results["daily_return"][results["daily_return"] < 0]
        downside_volatility = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        )
        sortino_ratio = (
            (excess_returns.mean() * 252) / downside_volatility
            if downside_volatility > 0
            else 0
        )

        # Calculate win rate
        trades = results["signal"].diff().abs() > 0
        num_trades = trades.sum()

        # Calculate profitable trades
        results["trade_profit"] = np.nan
        results.loc[trades, "trade_profit"] = results.loc[trades, "equity"].diff()

        win_rate = (
            (results["trade_profit"] > 0).sum() / num_trades if num_trades > 0 else 0
        )

        # Compile performance metrics
        performance = {
            "total_return": total_return,
            "annual_return": total_return * (252 / len(results)),  # Approximate
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "num_trades": num_trades,
        }

        return results, performance


def load_example_data() -> pd.DataFrame:
    """
    Load example data for demonstration purposes.
    In a real application, this would load from CSV or API.
    """
    # Create a simple simulated price series
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Generate random walk with drift for price
    returns = np.random.normal(0.0005, 0.015, n)
    price = 100 * (1 + returns).cumprod()

    # Generate volume data (higher on trending days)
    volume = np.random.lognormal(10, 1, n) * (1 + abs(returns) * 10)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "open": price * (1 + np.random.normal(0, 0.005, n)),
            "high": price * (1 + abs(np.random.normal(0, 0.012, n))),
            "low": price * (1 - abs(np.random.normal(0, 0.012, n))),
            "close": price,
            "volume": volume,
        }
    )

    df = df.set_index("date")
    return df
