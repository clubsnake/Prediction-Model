from src.features.optimized_params import load_optimized_params  # updated relative import

from features import feature_engineering_with_params  # updated relative import


def process_ticker_with_optimized_params(ticker, timeframe, df):
    """
    Process a ticker with its optimized parameters.
    """
    # Load optimized parameters for this ticker and timeframe
    params = load_optimized_params(ticker, timeframe)

    # Apply feature engineering with these parameters; include VMLI parameters from optimization
    df_with_features = feature_engineering_with_params(
        df,
        ticker=ticker,
        use_vmli=True,
        vmli_window_mom=params["vmli_window_mom"],
        vmli_window_vol=params["vmli_window_vol"],
        vmli_smooth_period=params["vmli_smooth_period"],
        vmli_winsorize_pct=params["vmli_winsorize_pct"],
        vmli_use_ema=params["vmli_use_ema"],
        # ...include other necessary parameters as needed...
    )

    return df_with_features
