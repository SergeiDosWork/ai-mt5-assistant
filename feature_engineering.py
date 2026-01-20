import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple
import math


def resample_to_m3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-minute data to 3-minute candles
    
    Args:
        df: DataFrame with 1-minute candles (time, open, high, low, close, volume)
        
    Returns:
        DataFrame with 3-minute candles
    """
    # Set time as index if it's a column
    if 'time' in df.columns:
        df = df.set_index('time')
    
    # Resample to 3 minutes using standard aggregation
    m3_data = df.resample('3T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to get time as a column again
    m3_data = m3_data.reset_index()
    
    return m3_data


def resample_to_h1_from_m3(m3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 3-minute data to 1-hour candles
    
    Args:
        m3_df: DataFrame with 3-minute candles (time, open, high, low, close, volume)
        
    Returns:
        DataFrame with 1-hour candles
    """
    # Set time as index if it's a column
    if 'time' in m3_df.columns:
        m3_df = m3_df.set_index('time')
    
    # Resample to 1 hour using standard aggregation
    h1_data = m3_df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to get time as a column again
    h1_data = h1_data.reset_index()
    
    return h1_data


def calculate_relative_changes(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate relative changes for each column in the dataframe
    Each value represents the change from the previous row
    """
    # Calculate differences between consecutive rows
    diff = df[['open', 'high', 'low', 'close', 'volume']].diff()
    
    # Replace the first row with zeros (or original values if preferred)
    diff.iloc[0] = 0.0
    
    return diff.values


def build_training_dataset(parquet_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to build training dataset from parquet file
    
    Args:
        parquet_path: Path to the parquet file with 1-minute candles
        
    Returns:
        Tuple of (X, y) where X contains features and y contains targets
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    # Validate expected columns
    required_cols = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Got: {set(df.columns)}")
    
    # Ensure time column is datetime
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Convert 1-minute data to 3-minute candles
    m3_data = resample_to_m3(df)
    
    # Calculate technical indicators on the M3 data for ALL candles
    # RSI
    m3_data['rsi'] = ta.rsi(m3_data['close'], length=14)
    
    # MACD
    macd_obj = ta.macd(m3_data['close'], fast=12, slow=26, signal=9)
    if macd_obj is not None and isinstance(macd_obj, pd.DataFrame):
        m3_data['macd_line'] = macd_obj.iloc[:, 0]  # MACD line
        m3_data['signal_line'] = macd_obj.iloc[:, 1]  # Signal line
        m3_data['histogram'] = macd_obj.iloc[:, 2]  # Histogram
    else:
        # If MACD calculation fails, fill with NaN
        m3_data['macd_line'] = np.nan
        m3_data['signal_line'] = np.nan
        m3_data['histogram'] = np.nan
    
    # Squeeze Momentum Indicator (LazyBear)
    squeeze = ta.squeeze(m3_data['high'], m3_data['low'], m3_data['close'])
    if squeeze is not None and isinstance(squeeze, pd.DataFrame):
        # Look for the specific column name in the squeeze indicator
        if 'SQZ_20_2.0_20_1.5_LB' in squeeze.columns:
            m3_data['squeeze_on'] = squeeze['SQZ_20_2.0_20_1.5_LB']
        else:
            # If the exact column name is not found, use the first available column
            col_name = squeeze.columns[0]
            m3_data['squeeze_on'] = squeeze[col_name]
    else:
        # If Squeeze calculation fails, fill with NaN
        m3_data['squeeze_on'] = np.nan
    
    # Calculate squeeze momentum by taking the difference between current and previous squeeze value
    m3_data['squeeze_momentum'] = m3_data['squeeze_on'].diff()
    
    # Prepare arrays for features and targets
    X, y = [], []
    
    # Number of M3 candles needed for 60 M3 windows + 3 future candles
    min_m3_for_window = 60  # 60 M3 candles for the main window
    min_required_m3 = min_m3_for_window + 3  # Plus 3 for future targets
    
    # Also need enough for H1 data (we need at least 10 H1 candles for our analysis)
    h1_data = resample_to_h1_from_m3(m3_data)
    if len(m3_data) < min_required_m3 or len(h1_data) < 10:
        raise ValueError(f"Not enough data. Need at least {min_required_m3} M3 candles and 10 H1 candles. "
                         f"Got {len(m3_data)} M3 and {len(h1_data)} H1 candles.")
    
    # Find the first M3 index where we have enough H1 data available
    # We need to ensure that for each M3 position i, we can find 10 H1 candles
    # that cover the time period of interest
    start_idx = min_m3_for_window
    
    # We also need to make sure there are enough H1 candles available before the M3 data
    # Find the first M3 candle that has at least 10 H1 candles before it
    first_m3_with_enough_h1 = 0
    for i, m3_time in enumerate(m3_data['time']):
        h1_before_count = len(h1_data[h1_data['time'] <= m3_time])
        if h1_before_count >= 10:
            first_m3_with_enough_h1 = i
            break
    
    # Start from the maximum of required window and H1 availability
    start_idx = max(start_idx, first_m3_with_enough_h1)
    
    # Iterate through possible positions (starting from index where we have enough data)
    for i in range(start_idx, len(m3_data) - 3):
        # Check if all indicators are available for the entire 60-candle window
        window_start_idx = i - 59
        m3_window = m3_data.iloc[window_start_idx:i + 1]  # 60 candles [i-59, i]
        
        # Check if any of the required indicators are NaN in the entire window
        has_nan_values = (
            m3_window['rsi'].isna().any() or
            m3_window['macd_line'].isna().any() or
            m3_window['signal_line'].isna().any() or
            m3_window['histogram'].isna().any() or
            m3_window['squeeze_on'].isna().any() or
            m3_window['squeeze_momentum'].isna().any()
        )
        
        if has_nan_values:
            continue
        
        # Build input tensor for the model: shape (60, 10)
        # Where each candle has: OHLC (4) + volume (1) + MACD (3) + Squeeze (2)
        ohlc = m3_window[['open', 'high', 'low', 'close']].values
        volume = m3_window['volume'].values.reshape(-1, 1)
        macd_features = m3_window[['macd_line', 'signal_line', 'histogram']].values
        squeeze_features = m3_window[['squeeze_on', 'squeeze_momentum']].values
        
        # Combine all features
        candle_features = np.concatenate([
            ohlc,
            volume,
            macd_features,
            squeeze_features
        ], axis=1)  # Shape: (60, 10)
        
        # Extract cyclical time features (for Moscow time) from the last candle in the window
        moscow_time = m3_data['time'].iloc[i]  # Time of the last candle in the window
        hour = moscow_time.hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # Build input features vector - now it will be the 60x10 tensor
        x_features = candle_features  # Shape: (60, 10)
        
        # Build target vector
        y_features = []
        
        # 1. Next 3 M3 candles as relative changes
        for j in range(1, 4):  # Next 3 candles: i+1, i+2, i+3
            if i + j >= len(m3_data):
                # If we go beyond the data, pad with zeros
                y_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # [delta_open, delta_high, delta_low, delta_close, volume]
            else:
                # Calculate relative changes compared to previous candle
                curr_row = m3_data.iloc[i + j]
                prev_close = m3_data['close'].iloc[i + j - 1]
                
                delta_open = curr_row['open'] - prev_close
                delta_high = curr_row['high'] - prev_close
                delta_low = curr_row['low'] - prev_close
                delta_close = curr_row['close'] - prev_close
                volume = curr_row['volume']
                
                y_features.extend([delta_open, delta_high, delta_low, delta_close, volume])
        
        # 2. Growth parameters
        # Find the last green candle before a red candle in the future
        future_start_idx = i + 1
        if future_start_idx < len(m3_data):
            future_data = m3_data.iloc[future_start_idx:].copy()
            
            # Find the last green candle before a red candle
            last_green_idx = None
            consecutive_greens = 0
            max_future_scan = min(50, len(m3_data) - future_start_idx)
            
            for k in range(max_future_scan):
                future_row = m3_data.iloc[future_start_idx + k]
                if future_row['close'] > future_row['open']:  # Green candle
                    last_green_idx = future_start_idx + k
                    consecutive_greens += 1
                else:  # Red candle
                    if consecutive_greens > 0:  # If we had greens before this red
                        break
                    else:  # Still looking for first green
                        continue
            
            if last_green_idx is not None and consecutive_greens > 0:
                # Calculate changes relative to the last candle in the window (at position i)
                last_green_row = m3_data.iloc[last_green_idx]
                prev_close = m3_data['close'].iloc[i]  # Reference close from window
                
                growth_delta_open = last_green_row['open'] - prev_close
                growth_delta_high = last_green_row['high'] - prev_close
                growth_delta_low = last_green_row['low'] - prev_close
                growth_delta_close = last_green_row['close'] - prev_close
                
                y_features.extend([growth_delta_open, growth_delta_high, growth_delta_low, growth_delta_close])
                y_features.append(float(consecutive_greens))
            else:
                # No growth pattern found
                y_features.extend([0.0, 0.0, 0.0, 0.0])  # growth_ohlc
                y_features.append(0.0)  # growth_length
        else:
            # No future data available
            y_features.extend([0.0, 0.0, 0.0, 0.0])  # growth_ohlc
            y_features.append(0.0)  # growth_length
        
        # Add to datasets
        X.append(x_features)
        y.append(y_features)
    
    # Convert to numpy arrays
    if len(X) == 0 or len(y) == 0:
        # Return empty arrays if no samples were generated
        return np.empty((0, 60, 10), dtype=np.float32), np.empty((0, len(y[0]) if len(y) > 0 else 0), dtype=np.float32)
    
    X_array = np.array(X, dtype=np.float32)  # Shape: (samples, 60, 10)
    y_array = np.array(y, dtype=np.float32)
    
    # Remove any rows with NaN values
    if X_array.size > 0 and y_array.size > 0:
        # Check dimensions before applying mask
        if y_array.ndim > 1:
            valid_mask = ~(np.isnan(X_array).any(axis=(1,2)) | np.isnan(y_array).any(axis=1))
            X_array = X_array[valid_mask]
            y_array = y_array[valid_mask]
    
    return X_array, y_array


def prepare_inference_features(m3_data: list, h1_data: list) -> np.ndarray:
    """
    Prepares features for inference from M3 and H1 data received from MT5
    
    Args:
        m3_data: List of 60 M3 candles
        h1_data: List of 10 H1 candles
        
    Returns:
        Features array ready for model inference
    """
    # Convert the incoming data to the format expected by the model
    # This is a simplified implementation - in a real scenario you'd do proper feature engineering
    
    # Create a combined feature vector from the M3 and H1 data
    # The exact implementation depends on how your model was trained
    
    # For now, we'll create a simple feature vector based on the OHLCV values
    features = []
    
    # Process M3 data (60 candles)
    for candle in m3_data:
        if isinstance(candle, dict):
            # If candle is a dictionary with named fields
            open_price = candle.get('open', candle.get(0, 0))
            high_price = candle.get('high', candle.get(1, 0))
            low_price = candle.get('low', candle.get(2, 0))
            close_price = candle.get('close', candle.get(3, 0))
            volume = candle.get('volume', candle.get(4, 0))
        else:
            # If candle is a list/array
            open_price = candle[0] if len(candle) > 0 else 0
            high_price = candle[1] if len(candle) > 1 else 0
            low_price = candle[2] if len(candle) > 2 else 0
            close_price = candle[3] if len(candle) > 3 else 0
            volume = candle[4] if len(candle) > 4 else 0
            
        # Normalize prices relative to the first candle's close
        first_close = m3_data[0].get('close', m3_data[0][3]) if isinstance(m3_data[0], dict) else m3_data[0][3]
        norm_factor = first_close if first_close != 0 else 1
        
        features.extend([
            open_price / norm_factor - 1,  # Normalized open
            high_price / norm_factor - 1,  # Normalized high
            low_price / norm_factor - 1,   # Normalized low
            close_price / norm_factor - 1, # Normalized close
            volume  # Volume (not normalized)
        ])
    
    # Process H1 data (10 candles) - similar approach
    for candle in h1_data:
        if isinstance(candle, dict):
            open_price = candle.get('open', candle.get(0, 0))
            high_price = candle.get('high', candle.get(1, 0))
            low_price = candle.get('low', candle.get(2, 0))
            close_price = candle.get('close', candle.get(3, 0))
            volume = candle.get('volume', candle.get(4, 0))
        else:
            open_price = candle[0] if len(candle) > 0 else 0
            high_price = candle[1] if len(candle) > 1 else 0
            low_price = candle[2] if len(candle) > 2 else 0
            close_price = candle[3] if len(candle) > 3 else 0
            volume = candle[4] if len(candle) > 4 else 0
            
        # Normalize prices relative to the first H1 candle's close
        if h1_data:
            first_h1_close = h1_data[0].get('close', h1_data[0][3]) if isinstance(h1_data[0], dict) else h1_data[0][3]
            norm_factor = first_h1_close if first_h1_close != 0 else 1
        else:
            norm_factor = 1
            
        features.extend([
            open_price / norm_factor - 1,  # Normalized open
            high_price / norm_factor - 1,  # Normalized high
            low_price / norm_factor - 1,   # Normalized low
            close_price / norm_factor - 1, # Normalized close
            volume  # Volume (not normalized)
        ])
    
    return np.array(features, dtype=np.float32)


def generate_training_samples(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function that takes a dataframe and saves it temporarily to call build_training_dataset

    Args:
        data: DataFrame with historical price data (1-minute candles)

    Returns:
        Tuple of (X, y) where X contains features and y contains targets
    """
    import tempfile
    import os

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.parquet') as temp_file:
        temp_path = temp_file.name

    try:
        # Save the dataframe to the temporary parquet file
        data.to_parquet(temp_path)

        # Call the build_training_dataset function
        X, y = build_training_dataset(temp_path)

        return X, y
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)