#!/usr/bin/env python3
"""
Debug script to understand why no samples are being generated
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from feature_engineering import resample_to_m3, resample_to_h1_from_m3
import pandas_ta as ta

def create_test_data():
    """Create test data with 1-minute candles"""
    print("Creating test data...")
    
    # Create 500 minutes of test data (more than needed for 60 M3 + 3 future + buffer)
    n_minutes = 500
    base_time = datetime.now() - timedelta(minutes=n_minutes)
    
    # Generate synthetic price data
    np.random.seed(42)
    open_prices = 100 + np.cumsum(np.random.normal(0, 0.1, n_minutes))
    
    # Generate OHLC data with realistic relationships
    data = []
    for i in range(n_minutes):
        open_val = open_prices[i]
        # Random walk with some volatility
        change = np.random.normal(0, 0.2)
        close_val = open_val + change
        high_val = max(open_val, close_val) + abs(np.random.normal(0, 0.1))
        low_val = min(open_val, close_val) - abs(np.random.normal(0, 0.1))
        
        # Ensure realistic relationships
        high_val = max(high_val, open_val, close_val)
        low_val = min(low_val, open_val, close_val)
        
        volume = int(np.random.uniform(1000, 10000))
        
        data.append({
            'time': base_time + timedelta(minutes=i),
            'open': open_val,
            'high': high_val,
            'low': low_val,
            'close': close_val,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"Created {len(df)} 1-minute candles")
    return df

def debug_conversion():
    """Debug the conversion process"""
    print("Debugging conversion process...")
    
    # Create test data
    df = create_test_data()
    print(f"Original data shape: {df.shape}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    
    # Convert 1-minute to 3-minute
    m3_data = resample_to_m3(df)
    print(f"M3 data shape: {m3_data.shape}")
    print(f"M3 time range: {m3_data['time'].min()} to {m3_data['time'].max()}")
    
    # Convert 3-minute to 1-hour
    h1_data = resample_to_h1_from_m3(m3_data)
    print(f"H1 data shape: {h1_data.shape}")
    print(f"H1 time range: {h1_data['time'].min()} to {h1_data['time'].max()}")
    
    # Check if we have enough data
    min_m3_for_window = 60  # 60 M3 candles for the main window
    min_required_m3 = min_m3_for_window + 3  # Plus 3 for future targets
    print(f"Required M3 candles: {min_required_m3}, Available: {len(m3_data)}")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    m3_data['rsi'] = ta.rsi(m3_data['close'], length=14)
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
    
    # Check for valid indicators
    print(f"RSI valid count: {m3_data['rsi'].notna().sum()}")
    print(f"MACD valid count: {m3_data['macd'].notna().sum()}")
    print(f"Squeeze valid count: {m3_data['squeeze'].notna().sum()}")
    
    # Test iteration range
    print(f"\nIteration range: {min_m3_for_window} to {len(m3_data) - 3}")
    for i in range(min_m3_for_window, min(len(m3_data) - 3, min_m3_for_window + 5)):  # Just check first few
        print(f"Position {i}: RSI={m3_data['rsi'].iloc[i]}, MACD={m3_data['macd'].iloc[i]}, Squeeze={m3_data['squeeze'].iloc[i]}")
        has_nan = (pd.isna(m3_data['rsi'].iloc[i]) or 
                  pd.isna(m3_data['macd'].iloc[i]) or 
                  pd.isna(m3_data['squeeze'].iloc[i]))
        print(f"  Has NaN indicators: {has_nan}")
    
if __name__ == "__main__":
    debug_conversion()