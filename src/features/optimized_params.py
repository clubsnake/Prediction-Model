def save_optimized_params(ticker, timeframe, params):
    """
    Save optimized parameters for a specific ticker and timeframe.
    """
    import json
    import os

    from config.config_loader import DATA_DIR

    # Create a directory structure for parameter storage in the Models folder
    params_dir = os.path.join(DATA_DIR, "Models", "optimized_params", ticker)
    os.makedirs(params_dir, exist_ok=True)

    # Save the parameters to a JSON file
    with open(os.path.join(params_dir, f"{timeframe}_vmli_params.json"), "w") as f:
        json.dump(params, f, indent=4)


def load_optimized_params(ticker, timeframe):
    """
    Load optimized parameters for a specific ticker and timeframe.
    """
    import json
    import os

    from config.config_loader import DATA_DIR

    # Create the full path to the parameters file
    param_file = os.path.join(
        DATA_DIR, "Models", "optimized_params", ticker, f"{timeframe}_vmli_params.json"
    )

    # Use default parameters if optimized ones don't exist
    if not os.path.exists(param_file):
        try:
            from config.config_loader import (
                VMLI_SMOOTH_PERIOD,
                VMLI_USE_EMA,
                VMLI_WINDOW_MOM,
                VMLI_WINDOW_VOL,
                VMLI_WINSORIZE_PCT,
                WERPI_LEVEL,
                WERPI_N_STATES,
                WERPI_SCALE,
                WERPI_WAVELET,
            )
        except ImportError:
            # If config values aren't available, use these defaults
            VMLI_WINDOW_MOM = 14
            VMLI_WINDOW_VOL = 14
            VMLI_SMOOTH_PERIOD = 3
            VMLI_WINSORIZE_PCT = 0.01
            VMLI_USE_EMA = True
            WERPI_WAVELET = "db4"
            WERPI_LEVEL = 3
            WERPI_N_STATES = 2
            WERPI_SCALE = 1.0

        return {
            "vmli_window_mom": VMLI_WINDOW_MOM,
            "vmli_window_vol": VMLI_WINDOW_VOL,
            "vmli_smooth_period": VMLI_SMOOTH_PERIOD,
            "vmli_winsorize_pct": VMLI_WINSORIZE_PCT,
            "vmli_use_ema": VMLI_USE_EMA,
            "werpi_wavelet": WERPI_WAVELET,
            "werpi_level": WERPI_LEVEL,
            "werpi_n_states": WERPI_N_STATES,
            "werpi_scale": WERPI_SCALE,
            "timeframe": timeframe,  # Include timeframe in parameters
        }

    # Load and return the saved parameters
    with open(param_file, "r") as f:
        params = json.load(f)
        # Ensure timeframe is included in parameters
        params["timeframe"] = timeframe
        return params
