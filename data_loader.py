import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Any


def validate_parquet_format(file_path: str) -> bool:
    """
    Validates that the parquet file has the required columns and data types
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Read schema
        table = pq.read_table(file_path)
        schema = table.schema
        
        # Check required columns exist
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in schema.names:
                print(f"Missing column: {col}")
                return False
        
        # Check data types
        expected_types = {
            'time': 'timestamp[ns]',
            'open': 'double',
            'high': 'double',
            'low': 'double',
            'close': 'double',
            'volume': 'int64'
        }
        
        for col, expected_type in expected_types.items():
            actual_type = str(schema.field(col).type)
            if actual_type != expected_type:
                print(f"Column {col}: expected {expected_type}, got {actual_type}")
                return False
        
        return True
    except Exception as e:
        print(f"Error validating parquet file: {e}")
        return False


def load_parquet_data(file_path: str) -> pd.DataFrame:
    """
    Loads parquet data into a pandas DataFrame
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_parquet(file_path)
    
    # Ensure proper data types
    df['time'] = pd.to_datetime(df['time'])
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    return df