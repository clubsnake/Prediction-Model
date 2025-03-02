# utils.py
"""
Utility functions for validation and walk-forward constraints.
"""

from config import WALK_FORWARD_MIN, WALK_FORWARD_MAX
import pandas as pd
import logging

def validate_walk_forward(window: int) -> int:
    """
    Ensure the walk-forward window is within configured min/max.
    """
    if window < WALK_FORWARD_MIN:
        return WALK_FORWARD_MIN
    elif window > WALK_FORWARD_MAX:
        return WALK_FORWARD_MAX
    return window

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has required columns and consistent naming.

    :param df: Input pandas DataFrame.
    :return: The validated or None if columns are missing.
    """
    if df is None:
        return None
        
    if df.index.name in ['Date', 'Datetime']:
        df = df.reset_index()
        
    date_columns = ['Date', 'Datetime', 'date', 'datetime']
    for col in date_columns:
        if col in df.columns:
            df = df.rename(columns={col: 'date'})
            break
            
    required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"Missing required columns. Available columns: {df.columns}")
        return None
        
    return df

def clean_memory(force_gc=False):
    """
    Release memory explicitly and optionally force garbage collection.
    Safe to call frequently.
    
    Args:
        force_gc: If True, forces Python garbage collection
    """
    import gc
    
    # Clear TensorFlow session if available
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except (ImportError, AttributeError):
        pass
    
    # Run garbage collection if requested
    if force_gc:
        gc.collect()
        
    return True

def clean_memory():
    """Release memory explicitly - call between tuning cycles"""
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        return True
    except:
        return False
