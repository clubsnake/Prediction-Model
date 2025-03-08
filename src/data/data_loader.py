"""
Data loading module with proper error handling for missing files.

This module ensures that missing data files are caught early in the process
to prevent cascading failures later in the analysis pipeline.
"""

import os
import pandas as pd
import logging
from typing import Optional, Dict, Any, Union, List
import yaml

# Add imports for memory management
from src.utils.memory_utils import adaptive_memory_clean

# Set up logging
logger = logging.getLogger(__name__)

class DataFileNotFoundError(Exception):
    """Exception raised when required data file is not found."""
    pass

class DataFormatError(Exception):
    """Exception raised when data file has incorrect format."""
    pass

def validate_file_exists(file_path: str, description: str = "data") -> None:
    """
    Validate that a file exists and raise a clear exception if it doesn't.
    
    Args:
        file_path: Path to the file to check
        description: Description of the file for error messages
        
    Raises:
        DataFileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise DataFileNotFoundError(
            f"Required {description} file not found: {file_path}"
        )
    
    if not os.path.isfile(file_path):
        raise DataFileNotFoundError(
            f"Path exists but is not a file: {file_path}"
        )
    
    # Additional check for zero-size files
    if os.path.getsize(file_path) == 0:
        raise DataFileNotFoundError(
            f"File exists but is empty: {file_path}"
        )

def load_csv_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV data with proper error handling.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame containing the data
        
    Raises:
        DataFileNotFoundError: If file doesn't exist or is empty
        DataFormatError: If file format is invalid
    """
    # First validate the file exists
    validate_file_exists(file_path, description="CSV")
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        
        # Check if DataFrame is empty
        if df.empty:
            raise DataFormatError(f"CSV file contains no data: {file_path}")
        
        return df
    
    except pd.errors.EmptyDataError:
        raise DataFormatError(f"CSV file is empty or improperly formatted: {file_path}")
    except pd.errors.ParserError as e:
        raise DataFormatError(f"Error parsing CSV file {file_path}: {str(e)}")
    except Exception as e:
        raise DataFormatError(f"Unexpected error loading CSV file {file_path}: {str(e)}")

def load_excel_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load Excel data with proper error handling.
    
    Args:
        file_path: Path to the Excel file
        **kwargs: Additional arguments for pd.read_excel
        
    Returns:
        DataFrame containing the data
        
    Raises:
        DataFileNotFoundError: If file doesn't exist or is empty
        DataFormatError: If file format is invalid
    """
    # First validate the file exists
    validate_file_exists(file_path, description="Excel")
    
    try:
        df = pd.read_excel(file_path, **kwargs)
        
        # Check if DataFrame is empty
        if df.empty:
            raise DataFormatError(f"Excel file contains no data: {file_path}")
        
        return df
    
    except Exception as e:
        raise DataFormatError(f"Error loading Excel file {file_path}: {str(e)}")

def load_yaml_data(file_path: str) -> Dict[str, Any]:
    """
    Load YAML data with proper error handling.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the data
        
    Raises:
        DataFileNotFoundError: If file doesn't exist or is empty
        DataFormatError: If file format is invalid
    """
    # First validate the file exists
    validate_file_exists(file_path, description="YAML")
    
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        if data is None:
            # YAML file is empty or only has comments
            return {}
        
        if not isinstance(data, dict):
            raise DataFormatError(f"YAML file does not contain a dictionary: {file_path}")
        
        return data
    
    except yaml.YAMLError as e:
        raise DataFormatError(f"Error parsing YAML file {file_path}: {str(e)}")
    except Exception as e:
        raise DataFormatError(f"Unexpected error loading YAML file {file_path}: {str(e)}")

# Add memory-efficient option to load_financial_data
def load_financial_data(ticker: str, timeframe: str, data_dir: str, 
                       memory_efficient: bool = True) -> pd.DataFrame:
    """
    Load financial data with memory efficiency options.
    """
    # Clean memory before loading large data
    adaptive_memory_clean("small")
    
    # Construct file path
    file_path = os.path.join(data_dir, f"{ticker}_{timeframe}.csv")
    
    # Validate file exists before attempting to load
    validate_file_exists(file_path, description=f"{ticker} {timeframe} data")
    
    try:
        if memory_efficient and os.path.getsize(file_path) > 100*1024*1024:  # 100MB
            # Use chunking for large files
            chunks = []
            for chunk in pd.read_csv(file_path, parse_dates=['date'], chunksize=100000):
                # Process each chunk
                chunks.append(chunk)
                # Clean memory after each chunk
                adaptive_memory_clean("small")
            
            # Combine chunks
            df = pd.concat(chunks)
        else:
            # Regular loading for smaller files
            df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Validate expected columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataFormatError(
                f"Missing required columns in {file_path}: {', '.join(missing_columns)}"
            )
        
        # Set date as index if it exists
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        return df
    
    except pd.errors.EmptyDataError:
        raise DataFormatError(f"Data file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise DataFormatError(f"Error parsing data file {file_path}: {str(e)}")
    except Exception as e:
        if isinstance(e, DataFormatError):
            raise
        raise DataFormatError(f"Unexpected error loading data file {file_path}: {str(e)}")

def save_financial_data(df: pd.DataFrame, ticker: str, timeframe: str, data_dir: str) -> str:
    """
    Save financial data for a specific ticker and timeframe.
    
    Args:
        df: DataFrame containing the financial data
        ticker: Ticker symbol
        timeframe: Timeframe of the data
        data_dir: Directory where data files are stored
        
    Returns:
        Path to the saved file
        
    Raises:
        IOError: If file cannot be saved
    """
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Construct file path
    file_path = os.path.join(data_dir, f"{ticker}_{timeframe}.csv")
    
    try:
        # Reset index if date is in the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Save the data
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved data for {ticker} ({timeframe}) to {file_path}")
        return file_path
    
    except Exception as e:
        raise IOError(f"Error saving data to {file_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Test with a file that should exist
        config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'user_config.yaml')
        config = load_yaml_data(config_file)
        print(f"Successfully loaded config with {len(config)} top-level keys")
        
    except (DataFileNotFoundError, DataFormatError) as e:
        print(f"Error: {e}")