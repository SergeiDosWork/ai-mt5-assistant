#!/usr/bin/env python3
"""
Comprehensive test for the feature engineering module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from feature_engineering import build_training_dataset, generate_training_samples
import math

def create_comprehensive_test_data():
    """Create comprehensive test data with 1-minute candles"""
    print("Creating comprehensive test data...")
    
    # Create 2000 minutes of test data to ensure we have plenty of data
    n_minutes = 2000
    base_time = datetime.now() - timedelta(minutes=n_minutes)
    
    # Generate more realistic price data
    np.random.seed(42)
    # Start with a base price level
    base_price = 100.0
    
    # Generate prices with trend and volatility
    returns = np.random.normal(0.0001, 0.005, n_minutes)  # Small positive drift with volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Generate OHLC data with realistic relationships
    data = []
    for i in range(n_minutes):
        open_val = prices[i]
        # Add some variation between open/close
        close_change = np.random.normal(0, 0.002)  # Small random movement
        close_val = open_val * (1 + close_change)
        
        # High and low around the open/close range with some volatility
        high_val = max(open_val, close_val) + abs(np.random.normal(0, 0.001))
        low_val = min(open_val, close_val) - abs(np.random.normal(0, 0.001))
        
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
    print(f"Created {len(df)} 1-minute candles with price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
    return df

def test_comprehensive():
    """Run comprehensive tests"""
    print("Running comprehensive tests...")
    
    # Create comprehensive test data
    df = create_comprehensive_test_data()
    
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
        
        # Validate feature ranges
        print(f"X value ranges: min={X.min():.6f}, max={X.max():.6f}")
        print(f"Y value ranges: min={y.min():.6f}, max={y.max():.6f}")
        
        # Validate specific feature sections
        # 1st feature should be ticker (0.0 - placeholder)
        ticker_vals = X[:, 0]
        print(f"Ticker feature (1st col) range: min={ticker_vals.min()}, max={ticker_vals.max()}, unique values: {len(np.unique(ticker_vals))}")
        
        # Features 1-300 should be M3 relative changes
        m3_features = X[:, 1:301]
        print(f"M3 features range: min={m3_features.min():.6f}, max={m3_features.max():.6f}")
        
        # Features 301-351 should be H1 relative changes  
        h1_features = X[:, 301:351]
        print(f"H1 features range: min={h1_features.min():.6f}, max={h1_features.max():.6f}")
        
        # Features 351-356 should be technical indicators
        tech_features = X[:, 351:356]
        print(f"Technical indicators range: min={tech_features.min():.6f}, max={tech_features.max():.6f}")
        
        # Features 356-358 should be time features (sin/cos - should be between -1 and 1)
        time_features = X[:, 356:358]
        print(f"Time features range: min={time_features.min():.6f}, max={time_features.max():.6f}")
        assert time_features.min() >= -1.0 and time_features.max() <= 1.0, "Time features should be between -1 and 1"
        
        # Validate target structure
        # First 15 features should be next 3 M3 candles (5 features each)
        next_candles = y[:, :15]
        print(f"Next 3 M3 candles targets range: min={next_candles.min():.6f}, max={next_candles.max():.6f}")
        
        # Last 5 features should be growth parameters (4 OHLC changes + 1 length)
        growth_params = y[:, 15:20]
        growth_ohlc = y[:, 15:19]  # OHLC changes
        growth_length = y[:, 19]   # Length
        print(f"Growth OHLC changes range: min={growth_ohlc.min():.6f}, max={growth_ohlc.max():.6f}")
        print(f"Growth length range: min={growth_length.min():.6f}, max={growth_length.max():.6f}")
        
        # Growth length should be non-negative
        assert (growth_length >= 0).all(), "Growth length should be non-negative"
        
        print("✓ All structural validations passed")
        
        print("✓ Comprehensive test completed successfully!")
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_wrapper_function():
    """Test the wrapper function"""
    print("\nTesting wrapper function (generate_training_samples)...")
    
    # Create smaller test data for wrapper test
    df = create_comprehensive_test_data().head(1000)  # Use first 1000 rows
    
    X, y = generate_training_samples(df)
    
    print(f"Wrapper function results: X shape {X.shape}, y shape {y.shape}")
    
    assert X.shape[0] == y.shape[0], "Wrapper function should produce matching shapes"
    print("✓ Wrapper function test passed")

if __name__ == "__main__":
    test_comprehensive()
    test_wrapper_function()
    print("\n🎉 All tests passed successfully!")