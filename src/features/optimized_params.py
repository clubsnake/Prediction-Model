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

    param_file = f"optimized_params/{ticker}/{timeframe}_vmli_params.json"

    # Use default parameters if optimized ones don't exist
    if not os.path.exists(param_file):
        from config import (
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
        }

    # Load and return the saved parameters
    with open(param_file, "r") as f:
        return json.load(f)
