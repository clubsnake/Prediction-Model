"""
Automatic parameter tuning for technical indicators using Optuna.

This module provides functions to optimize parameters for:
1. WERPI (Wavelet-Encoded Relative Price Indicator)
2. VMLI (Volatility-Adjusted Momentum Liquidity Index)

The optimization is based on historical performance using custom metrics.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
import pywt
from hmmlearn import hmm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings from hmmlearn and optuna
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# WERPI Optimization
# ---------------------------------------------------------------------------


def calculate_werpi_with_params(
    df: pd.DataFrame, wavelet_name: str, level: int, n_states: int, scale_factor: float
) -> np.ndarray:
    """
    Calculate WERPI indicator with specified parameters.

    Args:
        df: DataFrame with price data
        wavelet_name: Wavelet type to use
        level: Decomposition level
        n_states: Number of states for HMM
        scale_factor: Scaling factor for the indicator

    Returns:
        WERPI values as numpy array
    """
    try:
        # Calculate returns if not already present
        if "returns" not in df.columns:
            df.loc[:, "returns"] = df["Close"].pct_change()

        valid_returns = df["returns"].dropna()
        if len(valid_returns) < 2**level:  # Ensure enough data points
            raise ValueError(f"Not enough data points for decomposition level {level}")

        returns_array = valid_returns.values.reshape(-1, 1)

        # Initialize and fit HMM
        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42
        )
        model.fit(returns_array)

        # Normalize start probabilities
        if hasattr(model, "startprob_"):
            total = model.startprob_.sum()
            if total <= 0 or np.isnan(total):
                raise ValueError("Invalid start probabilities in HMM.")
            model.startprob_ = model.startprob_ / total

        # Compute state probabilities
        state_probs = model.predict_proba(returns_array)
        max_probs = np.max(state_probs, axis=1)

        # Apply wavelet transform
        coeffs = pywt.wavedec(max_probs, wavelet_name, level=level)
        approx = coeffs[0] * scale_factor

        # Interpolate to original length
        x_old = np.linspace(0, len(max_probs) - 1, num=len(approx))
        x_new = np.arange(len(max_probs))
        werpi_values = np.interp(x_new, x_old, approx)

        # Scale to [0, 100] range for easier interpretation
        min_val = np.min(werpi_values)
        max_val = np.max(werpi_values)
        range_val = max_val - min_val

        if range_val == 0:
            return np.zeros(len(werpi_values))

        oscillator = 100 * (werpi_values - min_val) / range_val
        return oscillator

    except Exception as e:
        logger.error(f"Error in WERPI calculation: {str(e)}")
        return np.array([])


def evaluate_werpi_performance(
    df: pd.DataFrame,
    werpi_values: np.ndarray,
    upper_threshold: float = 70,
    lower_threshold: float = 30,
    eval_method: str = "returns",
) -> float:
    """
    Evaluate WERPI performance based on trading signals or correlation.

    Args:
        df: DataFrame with price data
        werpi_values: WERPI indicator values
        upper_threshold: Upper threshold for overbought condition
        lower_threshold: Lower threshold for oversold condition
        eval_method: Evaluation method ('returns', 'correlation', or 'signal_quality')

    Returns:
        Performance metric (higher is better)
    """
    if len(werpi_values) == 0 or len(werpi_values) > len(df):
        return -np.inf

    # Create a DataFrame with WERPI and price data
    evaluation_df = pd.DataFrame(
        {"close": df["Close"].iloc[-len(werpi_values) :].values, "werpi": werpi_values}
    )

    # Calculate returns (next day's return)
    evaluation_df["return"] = evaluation_df["close"].pct_change(1).shift(-1)

    # Simple buy/sell signals based on thresholds
    evaluation_df["signal"] = 0
    evaluation_df.loc[evaluation_df["werpi"] < lower_threshold, "signal"] = (
        1  # Buy signal
    )
    evaluation_df.loc[evaluation_df["werpi"] > upper_threshold, "signal"] = (
        -1
    )  # Sell signal

    # Fill forward signals (hold until next signal)
    evaluation_df["signal"] = evaluation_df["signal"].replace(
        to_replace=0, method="ffill"
    )

    # Remove NaN values
    evaluation_df = evaluation_df.dropna()

    if len(evaluation_df) < 10:  # Need enough data for evaluation
        return -np.inf

    if eval_method == "returns":
        # Calculate strategy returns (signal * next day's return)
        evaluation_df["strategy_return"] = (
            evaluation_df["signal"] * evaluation_df["return"]
        )

        # Calculate cumulative returns
        cumulative_return = evaluation_df["strategy_return"].sum()
        return cumulative_return

    elif eval_method == "correlation":
        # For correlation, we want WERPI to be negatively correlated with future returns
        # (high WERPI -> lower future returns, low WERPI -> higher future returns)
        correlation = -evaluation_df["werpi"].corr(evaluation_df["return"])
        return correlation

    elif eval_method == "signal_quality":
        # Calculate directional accuracy
        correct_signals = (
            (evaluation_df["signal"] == 1) & (evaluation_df["return"] > 0)
        ) | ((evaluation_df["signal"] == -1) & (evaluation_df["return"] < 0))

        accuracy = correct_signals.mean()
        return accuracy

    else:
        logger.warning(f"Unknown evaluation method: {eval_method}")
        return -np.inf


def objective_werpi(
    trial, df, upper_threshold=70, lower_threshold=30, eval_method="returns"
):
    """
    Optuna objective function for WERPI parameter tuning.

    Args:
        trial: Optuna trial object
        df: DataFrame with price data
        upper_threshold: Upper threshold for overbought
        lower_threshold: Lower threshold for oversold
        eval_method: Evaluation method to use

    Returns:
        Performance metric to be maximized
    """
    # Sample parameters
    wavelet_options = [
        "haar",
        "db2",
        "db4",
        "db6",
        "sym2",
        "sym4",
        "sym6",
        "sym8",
        "coif2",
        "coif3",
        "dmey",
        "bior1.3",
        "rbio1.3",
        "gaus1",
        "mexh",
        "morl",
        "cgau1",
        "shan",
        "fbsp",
        "cmor",
        "demy",
    ]
    wavelet_name = trial.suggest_categorical("wavelet_name", wavelet_options)

    level = trial.suggest_int("level", 0.1, 5)
    n_states = trial.suggest_int("n_states", 2, 3)
    scale_factor = trial.suggest_float("scale_factor", 0.01, 10.0, log=True)

    # Calculate WERPI with these parameters
    werpi_values = calculate_werpi_with_params(
        df, wavelet_name, level, n_states, scale_factor
    )

    # Evaluate performance
    performance = evaluate_werpi_performance(
        df, werpi_values, upper_threshold, lower_threshold, eval_method
    )

    return performance


def optimize_werpi_parameters(
    df: pd.DataFrame,
    n_trials: int = 50,
    upper_threshold: float = 70,
    lower_threshold: float = 30,
    eval_method: str = "returns",
    timeout: int = 60,
) -> Dict:
    """
    Optimize WERPI parameters using Optuna.

    Args:
        df: DataFrame with price data
        n_trials: Number of optimization trials
        upper_threshold: Upper threshold for overbought
        lower_threshold: Lower threshold for oversold
        eval_method: Evaluation method ('returns', 'correlation', or 'signal_quality')
        timeout: Maximum optimization time in seconds

    Returns:
        Dictionary of optimal parameters
    """
    logger.info("Starting WERPI parameter optimization...")

    study = optuna.create_study(direction="maximize")

    # Create partial function with fixed df and thresholds
    objective = lambda trial: objective_werpi(
        trial, df, upper_threshold, lower_threshold, eval_method
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"WERPI optimization complete. Best value: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    # Calculate performance with best parameters
    best_params = study.best_params
    werpi_values = calculate_werpi_with_params(
        df,
        best_params["wavelet_name"],
        best_params["level"],
        best_params["n_states"],
        best_params["scale_factor"],
    )

    performance = evaluate_werpi_performance(
        df, werpi_values, upper_threshold, lower_threshold, eval_method
    )

    # Return both parameters and performance
    return {"params": best_params, "performance": performance, "values": werpi_values}


def add_werpi_indicator(
    df: pd.DataFrame,
    wavelet_name: str = "db4",
    level: int = 3,
    n_states: int = 2,
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    df = df.copy()
    if "returns" not in df.columns:
        df.loc[:, "returns"] = df["Close"].pct_change()

    valid_returns = df["returns"].dropna()
    if len(valid_returns) < 2**level:
        df["WERPI"] = np.nan
        return df

    returns_array = valid_returns.values.reshape(-1, 1)

    try:
        model_hmm = hmm.GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42
        )
        model_hmm.fit(returns_array)
        if hasattr(model_hmm, "startprob_"):
            total = model_hmm.startprob_.sum()
            if total <= 0 or np.isnan(total):
                raise ValueError("Invalid start probabilities in HMM.")
            model_hmm.startprob_ = model_hmm.startprob_ / total
        else:
            raise ValueError("HMM model does not have start probabilities.")

        state_probs = model_hmm.predict_proba(returns_array)
        max_probs = np.max(state_probs, axis=1)
        coeffs = pywt.wavedec(max_probs, wavelet_name, level=level)
        approx = coeffs[0] * scale_factor

        x_old = np.linspace(0, len(max_probs) - 1, num=len(approx))
        x_new = np.arange(len(max_probs))
        werpi_values = np.interp(x_new, x_old, approx)

        # Scale to [0, 100]
        min_val = np.min(werpi_values)
        max_val = np.max(werpi_values)
        range_val = max_val - min_val
        if range_val != 0:
            werpi_values = 100 * (werpi_values - min_val) / range_val

        werpi_full = np.full((len(df), 1), np.nan)
        werpi_full[valid_returns.index, 0] = werpi_values.flatten()
        df.loc[:, "WERPI"] = werpi_full

        df["WERPI"].fillna(method="ffill", inplace=True)
        df["WERPI"].fillna(method="bfill", inplace=True)
    except Exception as e:
        print(f"Error in WERPI calculation: {str(e)}")
        df.loc[:, "WERPI"] = np.nan

    return df


# ---------------------------------------------------------------------------
# VMLI Optimization
# ---------------------------------------------------------------------------


def calculate_vmli_with_params(
    df: pd.DataFrame,
    window_mom: int,
    window_vol: int,
    smooth_period: int,
    winsorize_pct: float,
    use_ema: bool,
) -> pd.Series:
    """
    Calculate VMLI with specified parameters.

    Args:
        df: DataFrame with price data
        window_mom: Momentum window
        window_vol: Volatility window
        smooth_period: Smoothing period
        winsorize_pct: Winsorization percentage
        use_ema: Whether to use EMA (True) or SMA (False)

    Returns:
        VMLI values as pandas Series
    """
    try:
        # We'll reimplement the VMILIndicator logic here for direct parameter testing
        # This avoids having to create a new class instance for each parameter set

        price_data = df["Close"]
        volume_data = df["Volume"] if "Volume" in df.columns else None

        # 1. Calculate momentum component
        daily_returns = price_data.pct_change()
        if use_ema:
            momentum = daily_returns.ewm(span=window_mom).mean()
        else:
            momentum = daily_returns.rolling(window=window_mom).mean()

        # 2. Calculate volatility component
        volatility = daily_returns.rolling(window=window_vol).std()
        pos_vol = volatility[volatility > 0]
        min_vol = pos_vol.min() if not pos_vol.empty else 0.0001
        volatility = volatility.clip(lower=min_vol)

        # 3. Calculate adjusted momentum
        adj_momentum = momentum / volatility

        # 4. Calculate liquidity/volume component
        if volume_data is not None:
            vol_median = volume_data.rolling(window=window_mom).median()
            vol_factor = volume_data / vol_median.replace(0, np.nan)
            vol_factor = vol_factor.clip(upper=5.0).fillna(1.0)
            liquidity = vol_factor
        else:
            liquidity = pd.Series(1.0, index=df.index)

        # 5. Calculate raw VMLI
        vmli_raw = adj_momentum * liquidity

        # 6. Winsorize to remove outliers
        if winsorize_pct > 0:
            lower_bound = vmli_raw.quantile(winsorize_pct)
            upper_bound = vmli_raw.quantile(1 - winsorize_pct)
            vmli_raw = vmli_raw.clip(lower=lower_bound, upper=upper_bound)

        # 7. Smooth the final indicator
        if use_ema:
            vmli = vmli_raw.ewm(span=smooth_period).mean()
        else:
            vmli = vmli_raw.rolling(window=smooth_period).mean()

        return vmli

    except Exception as e:
        logger.error(f"Error in VMLI calculation: {str(e)}")
        return pd.Series(dtype=float)


def evaluate_vmli_performance(
    df: pd.DataFrame,
    vmli_values: pd.Series,
    buy_threshold: float = 0.8,
    sell_threshold: float = -0.8,
    eval_method: str = "returns",
) -> float:
    """
    Evaluate VMLI performance based on trading signals or correlation.

    Args:
        df: DataFrame with price data
        vmli_values: VMLI indicator values
        buy_threshold: Threshold for buy signal
        sell_threshold: Threshold for sell signal
        eval_method: Evaluation method ('returns', 'correlation', 'signal_quality')

    Returns:
        Performance metric (higher is better)
    """
    if vmli_values.empty or len(vmli_values) > len(df):
        return -np.inf

    # Create evaluation DataFrame
    evaluation_df = pd.DataFrame(
        {
            "close": df["Close"].iloc[-len(vmli_values) :].values,
            "vmli": vmli_values.values,
        }
    )

    # Calculate returns (next day's return)
    evaluation_df["return"] = evaluation_df["close"].pct_change(1).shift(-1)

    # Generate signals based on thresholds
    evaluation_df["signal"] = 0
    evaluation_df.loc[evaluation_df["vmli"] > buy_threshold, "signal"] = 1  # Buy signal
    evaluation_df.loc[evaluation_df["vmli"] < sell_threshold, "signal"] = (
        -1
    )  # Sell signal

    # Fill forward signals (hold until next signal)
    evaluation_df["signal"] = evaluation_df["signal"].replace(
        to_replace=0, method="ffill"
    )

    # Remove NaN values
    evaluation_df = evaluation_df.dropna()

    if len(evaluation_df) < 10:  # Need enough data for evaluation
        return -np.inf

    if eval_method == "returns":
        # Calculate strategy returns (signal * next day's return)
        evaluation_df["strategy_return"] = (
            evaluation_df["signal"] * evaluation_df["return"]
        )

        # Calculate cumulative returns
        cumulative_return = evaluation_df["strategy_return"].sum()
        return cumulative_return

    elif eval_method == "correlation":
        # For correlation, higher VMLI should predict higher returns
        correlation = evaluation_df["vmli"].corr(evaluation_df["return"])
        return correlation

    elif eval_method == "signal_quality":
        # Calculate directional accuracy
        correct_signals = (
            (evaluation_df["signal"] == 1) & (evaluation_df["return"] > 0)
        ) | ((evaluation_df["signal"] == -1) & (evaluation_df["return"] < 0))

        accuracy = correct_signals.mean()
        return accuracy

    else:
        logger.warning(f"Unknown evaluation method: {eval_method}")
        return -np.inf


def objective_vmli(
    trial, df, buy_threshold=0.8, sell_threshold=-0.8, eval_method="returns"
):
    """
    Optuna objective function for VMLI parameter tuning.

    Args:
        trial: Optuna trial object
        df: DataFrame with price data
        buy_threshold: Threshold for buy signals
        sell_threshold: Threshold for sell signals
        eval_method: Evaluation method to use

    Returns:
        Performance metric to be maximized
    """
    # Sample parameters
    window_mom = trial.suggest_int("window_mom", 1, 120)
    window_vol = trial.suggest_int("window_vol", 1, 120)
    smooth_period = trial.suggest_int("smooth_period", 1, 20)
    winsorize_pct = trial.suggest_float("winsorize_pct", 0.001, 0.1)
    use_ema = trial.suggest_categorical("use_ema", [True, False])

    # Calculate VMLI with these parameters
    vmli_values = calculate_vmli_with_params(
        df, window_mom, window_vol, smooth_period, winsorize_pct, use_ema
    )

    # Evaluate performance
    performance = evaluate_vmli_performance(
        df, vmli_values, buy_threshold, sell_threshold, eval_method
    )

    return performance


def optimize_vmli_parameters(
    df: pd.DataFrame,
    n_trials: int = 50,
    buy_threshold: float = 0.8,
    sell_threshold: float = -0.8,
    eval_method: str = "returns",
    timeout: int = 60,
) -> Dict:
    """
    Optimize VMLI parameters using Optuna.

    Args:
        df: DataFrame with price data
        n_trials: Number of optimization trials
        buy_threshold: Threshold for buy signals
        sell_threshold: Threshold for sell signals
        eval_method: Evaluation method ('returns', 'correlation', or 'signal_quality')
        timeout: Maximum optimization time in seconds

    Returns:
        Dictionary of optimal parameters
    """
    logger.info("Starting VMLI parameter optimization...")

    study = optuna.create_study(direction="maximize")

    # Create partial function with fixed df and thresholds
    objective = lambda trial: objective_vmli(
        trial, df, buy_threshold, sell_threshold, eval_method
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"VMLI optimization complete. Best value: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    # Calculate performance with best parameters
    best_params = study.best_params
    vmli_values = calculate_vmli_with_params(
        df,
        best_params["window_mom"],
        best_params["window_vol"],
        best_params["smooth_period"],
        best_params["winsorize_pct"],
        best_params["use_ema"],
    )

    performance = evaluate_vmli_performance(
        df, vmli_values, buy_threshold, sell_threshold, eval_method
    )

    # Return both parameters and performance
    return {"params": best_params, "performance": performance, "values": vmli_values}


# ---------------------------------------------------------------------------
# Unified Indicator Optimization
# ---------------------------------------------------------------------------


def optimize_indicators(
    df: pd.DataFrame,
    indicators: List[str] = ["werpi", "vmli"],
    n_trials: int = 50,
    eval_method: str = "returns",
    timeout: int = 120,
) -> Dict:
    """
    Optimize multiple technical indicators in one function.

    Args:
        df: DataFrame with price data
        indicators: List of indicators to optimize ('werpi', 'vmli')
        n_trials: Number of trials per indicator
        eval_method: Evaluation method ('returns', 'correlation', 'signal_quality')
        timeout: Maximum optimization time per indicator in seconds

    Returns:
        Dictionary with optimal parameters for each indicator
    """
    results = {}

    for indicator in indicators:
        if indicator.lower() == "werpi":
            results["werpi"] = optimize_werpi_parameters(
                df, n_trials=n_trials, eval_method=eval_method, timeout=timeout
            )
        elif indicator.lower() == "vmli":
            results["vmli"] = optimize_vmli_parameters(
                df, n_trials=n_trials, eval_method=eval_method, timeout=timeout
            )
        else:
            logger.warning(f"Unknown indicator: {indicator}")

    return results


def save_tuned_parameters(results: Dict, output_file: str = None) -> str:
    """
    Save tuned indicator parameters to a JSON file.

    Args:
        results: Dictionary with optimization results
        output_file: Path to output file (default: 'Data/tuned_indicators.json')

    Returns:
        Path to the saved file
    """
    if output_file is None:
        os.makedirs("Data", exist_ok=True)
        output_file = os.path.join("Data", "tuned_indicators.json")

    # Add timestamp to results
    for indicator, result in results.items():
        if "timestamps" not in result:
            result["timestamps"] = {}
        result["timestamps"]["tuned_at"] = datetime.now().isoformat()

    # Save to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved tuned parameters to {output_file}")
    return output_file


def tune_and_apply_indicators(
    df: pd.DataFrame,
    tune_werpi: bool = True,
    tune_vmli: bool = True,
    n_trials: int = 50,
    eval_method: str = "returns",
    save_results: bool = True,
) -> tuple:
    """
    Tune indicators with Optuna and then apply them to the DataFrame.

    Args:
        df: DataFrame with price data
        tune_werpi: Whether to tune WERPI
        tune_vmli: Whether to tune VMLI
        n_trials: Number of Optuna trials
        eval_method: Evaluation method ('returns', 'correlation', 'signal_quality')
        save_results: Whether to save tuned parameters to file

    Returns:
        Tuple of (DataFrame with indicators, tuning results)
    """
    indicators_to_tune = []
    if tune_werpi:
        indicators_to_tune.append("werpi")
    if tune_vmli:
        indicators_to_tune.append("vmli")

    # Run Optuna optimization
    tuning_results = optimize_indicators(
        df=df,
        indicators=indicators_to_tune,
        n_trials=n_trials,
        eval_method=eval_method,
        timeout=600,  # 10 minutes per indicator
    )

    # Save results if requested
    if save_results:
        save_tuned_parameters(tuning_results)

    # Create copies of tuned parameters for features.py
    feature_params = {}

    if "werpi" in tuning_results:
        werpi_params = tuning_results["werpi"]["params"]
        feature_params["werpi_wavelet"] = werpi_params.get("wavelet_name", "db4")
        feature_params["werpi_level"] = werpi_params.get("level", 3)
        feature_params["werpi_n_states"] = werpi_params.get("n_states", 2)
        feature_params["werpi_scale"] = werpi_params.get("scale_factor", 1.0)

    if "vmli" in tuning_results:
        vmli_params = tuning_results["vmli"]["params"]
        feature_params["vmli_window_mom"] = vmli_params.get("window_mom", 14)
        feature_params["vmli_window_vol"] = vmli_params.get("window_vol", 14)
        feature_params["vmli_smooth_period"] = vmli_params.get("smooth_period", 3)
        feature_params["vmli_winsorize_pct"] = vmli_params.get("winsorize_pct", 0.01)
        feature_params["vmli_use_ema"] = vmli_params.get("use_ema", True)

    # Apply indicators with tuned parameters
    from src.features.features import feature_engineering_with_params

    result_df = feature_engineering_with_params(df, optuna_tuned_params=tuning_results)

    return result_df, tuning_results


# Add a command-line interface to run tuning directly
if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description="Tune technical indicators")
    parser.add_argument("--data", required=True, help="Path to input data CSV")
    parser.add_argument("--output", help="Path to output CSV with indicators")
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--method",
        choices=["returns", "correlation", "signal_quality"],
        default="returns",
        help="Evaluation method",
    )
    parser.add_argument("--no-werpi", action="store_true", help="Skip WERPI tuning")
    parser.add_argument("--no-vmli", action="store_true", help="Skip VMLI tuning")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Run tuning
    result_df, tuning_results = tune_and_apply_indicators(
        df=df,
        tune_werpi=not args.no_werpi,
        tune_vmli=not args.no_vmli,
        n_trials=args.trials,
        eval_method=args.method,
    )

    # Save output if specified
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Saved output to {args.output}")

    # Print tuning results
    for indicator, result in tuning_results.items():
        print(f"\n{indicator.upper()} tuning results:")
        print(f"Performance: {result['performance']:.4f}")
        print(f"Best parameters: {result['params']}")
