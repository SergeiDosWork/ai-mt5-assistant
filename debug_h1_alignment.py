#!/usr/bin/env python3
"""
Debug H1 alignment in feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from feature_engineering import resample_to_m3, resample_to_h1_from_m3, calculate_relative_changes
import pandas_ta as ta

def create_test_data():
    """Create test data with 1-minute candles"""
    print("Creating test data...")
    
    # Create 1000 minutes of test data to ensure we have plenty of H1 data
    n_minutes = 1000
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

def debug_h1_alignment():
    """Debug the H1 alignment process"""
    print("Debugging H1 alignment...")
    
    # Create test data
    df = create_test_data()
    print(f"Original data shape: {df.shape}")
    
    # Convert 1-minute to 3-minute
    m3_data = resample_to_m3(df)
    print(f"M3 data shape: {m3_data.shape}")
    print(f"M3 time range: {m3_data['time'].min()} to {m3_data['time'].max()}")
    
    # Convert 3-minute to 1-hour
    h1_data = resample_to_h1_from_m3(m3_data)
    print(f"H1 data shape: {h1_data.shape}")
    print(f"H1 time range: {h1_data['time'].min()} to {h1_data['time'].max()}")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    m3_data['rsi'] = ta.rsi(m3_data['close'], length=14)
    macd_obj = ta.macd(m3_data['close'], fast=12, slow=26, signal=9)
    if macd_obj is not None and isinstance(macd_obj, pd.DataFrame):
        m3_data['macd'] = macd_obj.iloc[:, 0]  # MACD line
        m3_data['macd_signal'] = macd_obj.iloc[:, 1]  # Signal line
        m3_data['macd_histogram'] = macd_obj.iloc[:, 2]  # Histogram
    else:
        m3_data['macd'] = np.nan
        m3_data['macd_signal'] = np.nan
        m3_data['macd_histogram'] = np.nan
    
    squeeze = ta.squeeze(m3_data['high'], m3_data['low'], m3_data['close'])
    if squeeze is not None and isinstance(squeeze, pd.DataFrame):
        if 'SQZ_20_2.0_20_1.5_LB' in squeeze.columns:
            m3_data['squeeze'] = squeeze['SQZ_20_2.0_20_1.5_LB']
        else:
            col_name = squeeze.columns[0]
            m3_data['squeeze'] = squeeze[col_name]
    else:
        m3_data['squeeze'] = np.nan
    
    # Test H1 alignment for a specific position
    min_m3_for_window = 60
    i = min_m3_for_window  # Start at position where we have enough data
    
    print(f"\nTesting H1 alignment for position {i}")
    current_time = m3_data['time'].iloc[i]
    print(f"Current M3 time: {current_time}")
    
    # Find H1 indices that are <= current_time
    h1_idx = h1_data[h1_data['time'] <= current_time].index
    print(f"H1 indices <= current_time: {list(h1_idx)}")
    
    if len(h1_idx) == 0:
        print("No H1 data available for alignment")
        return
    
    current_h1_idx = h1_idx[-1]  # Last H1 index that covers or is before current M3
    print(f"Current H1 index: {current_h1_idx}")
    
    h1_start_idx = max(0, current_h1_idx - 9)  # 10 candles back
    print(f"H1 start index: {h1_start_idx}")
    
    h1_window = h1_data.iloc[h1_start_idx:current_h1_idx + 1]
    print(f"H1 window shape: {h1_window.shape}")
    print(f"H1 window time range: {h1_window['time'].min()} to {h1_window['time'].max()}")
    
    # Check if we have enough H1 candles
    if len(h1_window) < 10:
        print(f"Only {len(h1_window)} H1 candles, trying to pad...")
        needed = 10 - len(h1_window)
        if h1_start_idx > 0:
            additional_start = max(0, h1_start_idx - needed)
            print(f"Trying to get additional H1 data from index {additional_start} to {h1_start_idx}")
            additional_h1 = h1_data.iloc[additional_start:h1_start_idx]
            print(f"Additional H1 shape: {additional_h1.shape}")
            h1_window = pd.concat([additional_h1, h1_window])
            h1_window = h1_window.tail(10)  # Keep only last 10
            print(f"After padding, H1 window shape: {h1_window.shape}")
    
    print(f"Final H1 window shape: {h1_window.shape}")
    
    if len(h1_window) >= 10:
        print("✓ Have enough H1 data")
        h1_features = calculate_relative_changes(h1_window)
        print(f"H1 features shape: {h1_features.shape}")
    else:
        print(f"✗ Only {len(h1_window)} H1 candles available, need 10")

if __name__ == "__main__":
    debug_h1_alignment()