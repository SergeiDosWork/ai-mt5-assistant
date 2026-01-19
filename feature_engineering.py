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
    
    # Resample to 3-minute intervals
    m3_df = df.resample('3min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to make time a column again
    m3_df = m3_df.reset_index()
    
    return m3_df


def resample_to_h1_from_m3(m3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 3-minute data to 1-hour candles
    
    Args:
        m3_df: DataFrame with 3-minute candles
        
    Returns:
        DataFrame with 1-hour candles
    """
    # Set time as index if it's a column
    if 'time' in m3_df.columns:
        m3_df = m3_df.set_index('time')
    
    # Resample to 1-hour intervals (every 20 M3 candles = 1 H1 candle)
    h1_df = m3_df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to make time a column again
    h1_df = h1_df.reset_index()
    
    return h1_df


def calculate_relative_changes(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate relative changes for OHLCV data
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        
    Returns:
        Array of relative changes [delta_open, delta_high, delta_low, delta_close, volume_abs]
    """
    # Calculate previous close values shifted by 1
    prev_close = df['close'].shift(1).bfill()
    
    # Calculate relative changes
    delta_open = df['open'] - prev_close
    delta_high = df['high'] - prev_close
    delta_low = df['low'] - prev_close
    delta_close = df['close'] - prev_close
    
    # Stack features together
    features = np.column_stack([
        delta_open.values,
        delta_high.values,
        delta_low.values,
        delta_close.values,
        df['volume'].values  # absolute volume
    ])
    
    return features


def calculate_growth_targets(future_candles: pd.DataFrame) -> Tuple[list, float]:
    """
    Find the last green candle before a red candle in the future and calculate growth parameters
    
    Args:
        future_candles: DataFrame with future candles starting from position after current window
        
    Returns:
        Tuple of (growth_ohlc, growth_length) where:
        - growth_ohlc: [delta_open, delta_high, delta_low, delta_close] of the last green candle
        - growth_length: number of consecutive green candles until reversal (max 50)
    """
    # Check up to 50 candles in the future
    max_future_candles = min(50, len(future_candles))
    
    last_green_idx = None
    consecutive_greens = 0
    
    for i in range(max_future_candles):
        current_row = future_candles.iloc[i]
        # Green candle: close > open
        if current_row['close'] > current_row['open']:
            last_green_idx = i
            consecutive_greens += 1
        else:
            # Red candle found - check if preceded by green(s)
            if consecutive_greens > 0:
                # Found reversal point
                break
            else:
                # Continue looking
                continue
    
    # If no green candle was found, return defaults
    if last_green_idx is None or consecutive_greens == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    
    # Get the last green candle
    last_green_candle = future_candles.iloc[last_green_idx]
    
    # Calculate OHLC changes relative to the last candle in the current window
    # We need to pass the previous close to calculate relative changes
    # This function will be called in the main loop, so we calculate changes relative to last window close
    return [0.0, 0.0, 0.0, 0.0], float(consecutive_greens)  # Placeholder - will be calculated properly in main function


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
    
    # Convert 3-minute data to 1-hour candles
    h1_data = resample_to_h1_from_m3(m3_data)
    
    # Calculate technical indicators on the M3 data
    # RSI
    m3_data['rsi'] = ta.rsi(m3_data['close'], length=14)
    
    # MACD
    macd_obj = ta.macd(m3_data['close'], fast=12, slow=26, signal=9)
    if macd_obj is not None and isinstance(macd_obj, pd.DataFrame):
        m3_data['macd'] = macd_obj.iloc[:, 0]  # MACD line
        m3_data['macd_signal'] = macd_obj.iloc[:, 1]  # Signal line
        m3_data['macd_histogram'] = macd_obj.iloc[:, 2]  # Histogram
    else:
        # If MACD calculation fails, fill with NaN
        m3_data['macd'] = np.nan
        m3_data['macd_signal'] = np.nan
        m3_data['macd_histogram'] = np.nan
    
    # Squeeze Momentum Indicator (LazyBear)
    squeeze = ta.squeeze(m3_data['high'], m3_data['low'], m3_data['close'])
    if squeeze is not None and isinstance(squeeze, pd.DataFrame):
        # Look for the specific column name in the squeeze indicator
        if 'SQZ_20_2.0_20_1.5_LB' in squeeze.columns:
            m3_data['squeeze'] = squeeze['SQZ_20_2.0_20_1.5_LB']
        else:
            # If the exact column name is not found, use the first available column
            col_name = squeeze.columns[0]
            m3_data['squeeze'] = squeeze[col_name]
    else:
        # If Squeeze calculation fails, fill with NaN
        m3_data['squeeze'] = np.nan
    
    # Prepare arrays for features and targets
    X, y = [], []
    
    # Number of M3 candles needed for 60 M3 windows + 3 future candles + 10 H1 windows
    # Each H1 candle is composed of ~20 M3 candles, so we need at least 10*20=200 M3 candles for H1 window
    # But since we're taking 60 M3 + 3 future + buffer, we need at least 60+3+200 = 263 M3 candles
    min_m3_for_window = 60  # 60 M3 candles for the main window
    min_required_m3 = min_m3_for_window + 3  # Plus 3 for future targets
    
    # Also need enough for H1 data (we need at least 10 H1 candles for our analysis)
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
        # Skip if any of the required indicators are NaN at position i
        if (pd.isna(m3_data['rsi'].iloc[i]) or 
            pd.isna(m3_data['macd'].iloc[i]) or 
            pd.isna(m3_data['squeeze'].iloc[i])):
            continue
        
        # Get M3 window: 60 candles ending at position i
        m3_window = m3_data.iloc[i - 59:i + 1]  # 60 candles [i-59, i]
        
        # Calculate relative changes for M3 window
        m3_features = calculate_relative_changes(m3_window)
        
        # Find corresponding H1 window (last 10 H1 candles that correspond to the M3 period)
        # We need to find the H1 candles that cover the same time period as our M3 window
        current_time = m3_data['time'].iloc[i]
        
        # Find the H1 candle that includes the current M3 candle
        h1_idx = h1_data[h1_data['time'] <= current_time].index
        if len(h1_idx) == 0:
            continue  # No corresponding H1 data
        
        # Take the last 10 H1 candles ending at or before the current time
        current_h1_idx = h1_idx[-1]
        h1_start_idx = max(0, current_h1_idx - 9)  # 10 candles back
        h1_window = h1_data.iloc[h1_start_idx:current_h1_idx + 1]
        
        # Ensure we have exactly 10 H1 candles
        if len(h1_window) < 10:
            # Pad with earlier candles if available
            needed = 10 - len(h1_window)
            if h1_start_idx > 0:
                additional_start = max(0, h1_start_idx - needed)
                h1_window = pd.concat([h1_data.iloc[additional_start:h1_start_idx], h1_window])
                h1_window = h1_window.tail(10)  # Keep only last 10
            if len(h1_window) < 10:
                continue  # Not enough H1 data
        
        # Calculate relative changes for H1 window
        h1_features = calculate_relative_changes(h1_window)
        
        # Extract technical indicators at position i (the last M3 candle in the window)
        rsi_value = m3_data['rsi'].iloc[i]
        macd_value = m3_data['macd'].iloc[i]
        macd_signal = m3_data['macd_signal'].iloc[i]
        macd_hist = m3_data['macd_histogram'].iloc[i]
        squeeze_value = m3_data['squeeze'].iloc[i]
        
        # Extract cyclical time features (for Moscow time)
        moscow_time = m3_data['time'].iloc[i]  # Assuming data is already in Moscow timezone
        hour = moscow_time.hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # Build input features vector
        x_features = []
        
        # 1. Ticker name (extracted from filename - placeholder for now)
        # For now, we'll just add a placeholder; in real implementation, ticker would come from filename
        ticker_feature = 0.0  # Placeholder - would be derived from filename
        x_features.append(ticker_feature)  # Add the ticker feature
        
        # 2. 60 M3 candles (5 features each = 300 features)
        x_features.extend(m3_features.flatten())
        
        # 3. 10 H1 candles (5 features each = 50 features)
        x_features.extend(h1_features.flatten())
        
        # 4. Technical indicators (5 features: RSI, MACD, MACD_signal, MACD_hist, Squeeze)
        x_features.extend([rsi_value, macd_value, macd_signal, macd_hist, squeeze_value])
        
        # 5. Cyclical time features (2 features)
        x_features.extend([hour_sin, hour_cos])
        
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
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)
    
    # Remove any rows with NaN values
    if X_array.size > 0 and y_array.size > 0:
        # Check dimensions before applying mask
        if X_array.ndim > 1 and y_array.ndim > 1:
            valid_mask = ~(np.isnan(X_array).any(axis=1) | np.isnan(y_array).any(axis=1))
            X_array = X_array[valid_mask]
            y_array = y_array[valid_mask]
    
    return X_array, y_array


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
    
    # Save the dataframe temporarily as a parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        data.to_parquet(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # Call the main function
        X, y = build_training_dataset(temp_path)
        return X, y
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)