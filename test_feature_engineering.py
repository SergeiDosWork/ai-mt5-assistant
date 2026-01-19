#!/usr/bin/env python3
"""
Test script for feature engineering module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from feature_engineering import build_training_dataset

def create_test_data():
    """Create test data with 1-minute candles"""
    print("Creating test data...")
    
    # Create 1000 minutes of test data to ensure we have enough H1 candles (1000/60 = ~17 H1 candles)
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

def test_module():
    """Test the feature engineering module"""
    print("Testing feature engineering module...")
    
    # Create test data
    df = create_test_data()
    
    # Save to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        df.to_parquet(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # Test the build_training_dataset function
        print("Running build_training_dataset...")
        X, y = build_training_dataset(temp_path)
        
        print(f"Generated training data:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        # Basic validation
        assert X.shape[0] == y.shape[0], f"X and y should have same number of samples: {X.shape[0]} vs {y.shape[0]}"
        assert X.ndim == 2, f"X should be 2D: {X.ndim}D"
        assert y.ndim == 2, f"y should be 2D: {y.ndim}D"
        
        print("✓ Basic validation passed")
        
        # Check for any NaN values
        nan_x = np.isnan(X).any()
        nan_y = np.isnan(y).any()
        if nan_x or nan_y:
            print(f"⚠ Warning: Found NaN values - X: {nan_x}, y: {nan_y}")
        else:
            print("✓ No NaN values found")
        
        # Print sample shapes to verify dimensions
        # Input features: 
        # - Ticker name: 1
        # - 60 M3 candles * 5 features = 300
        # - 10 H1 candles * 5 features = 50
        # - Technical indicators: 5 (RSI, MACD, MACD_signal, MACD_hist, Squeeze)
        # - Time features: 2 (hour_sin, hour_cos)
        expected_x_features = 1 + 300 + 50 + 5 + 2  # = 358
        print(f"Expected X features: {expected_x_features}, Actual: {X.shape[1]}")
        
        # Target features:
        # - 3 future M3 candles * 5 features = 15
        # - Growth parameters: 4 (OHLC changes) + 1 (length) = 5
        expected_y_features = 15 + 5  # = 20
        print(f"Expected y features: {expected_y_features}, Actual: {y.shape[1]}")
        
        print("✓ Test completed successfully!")
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_module()