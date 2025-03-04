def save_optimized_params(ticker, timeframe, params):
    """
    Save optimized parameters for a specific ticker and timeframe.
    """
    import os
    import json

    # Create a directory structure for parameter storage
    os.makedirs(f"optimized_params/{ticker}", exist_ok=True)

    # Save the parameters to a JSON file
    with open(f"optimized_params/{ticker}/{timeframe}_vmli_params.json", 'w') as f:
        json.dump(params, f, indent=4)

def load_optimized_params(ticker, timeframe):
    """
    Load optimized parameters for a specific ticker and timeframe.
    """
    import os
    import json

    param_file = f"optimized_params/{ticker}/{timeframe}_vmli_params.json"

    # Use default parameters if optimized ones don't exist
    if not os.path.exists(param_file):
        from config import (
            VMLI_WINDOW_MOM, VMLI_WINDOW_VOL, 
            VMLI_SMOOTH_PERIOD, VMLI_WINSORIZE_PCT, VMLI_USE_EMA,
            WERPI_WAVELET, WERPI_LEVEL, WERPI_N_STATES, WERPI_SCALE
        )
        return {
            'vmli_window_mom': VMLI_WINDOW_MOM,
            'vmli_window_vol': VMLI_WINDOW_VOL,
            'vmli_smooth_period': VMLI_SMOOTH_PERIOD,
            'vmli_winsorize_pct': VMLI_WINSORIZE_PCT,
            'vmli_use_ema': VMLI_USE_EMA,
            'werpi_wavelet': WERPI_WAVELET,
            'werpi_level': WERPI_LEVEL,
            'werpi_n_states': WERPI_N_STATES,
            'werpi_scale': WERPI_SCALE
        }

    # Load and return the saved parameters
    with open(param_file, 'r') as f:
        return json.load(f)
