#!/usr/bin/env python3
"""
Final validation test to ensure the implementation matches all requirements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from feature_engineering import build_training_dataset
import math

def create_validation_test_data():
    """Create test data to validate all requirements"""
    print("Creating validation test data...")
    
    # Create sufficient data for validation
    n_minutes = 2000
    base_time = datetime.now() - timedelta(minutes=n_minutes)
    
    # Generate consistent price data
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_minutes):
        # Small random movement
        change = np.random.normal(0, 0.002)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Generate OHLC data
    data = []
    for i in range(n_minutes):
        open_val = prices[i]
        close_change = np.random.normal(0, 0.001)
        close_val = open_val * (1 + close_change)
        
        # High and low with realistic ranges
        high_val = max(open_val, close_val) + abs(np.random.normal(0, 0.0005))
        low_val = min(open_val, close_val) - abs(np.random.normal(0, 0.0005))
        
        # Ensure realistic relationships
        high_val = max(high_val, open_val, close_val)
        low_val = min(low_val, open_val, close_val)
        
        volume = int(np.random.uniform(1000, 5000))
        
        data.append({
            'time': base_time + timedelta(minutes=i),
            'open': float(open_val),
            'high': float(high_val),
            'low': float(low_val),
            'close': float(close_val),
            'volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    print(f"Created {len(df)} 1-minute candles")
    return df

def validate_requirements():
    """Validate that all requirements are met"""
    print("Validating implementation against requirements...")
    
    df = create_validation_test_data()
    
    # Save to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        df.to_parquet(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        X, y = build_training_dataset(temp_path)
        
        print(f"Dataset shapes: X={X.shape}, y={y.shape}")
        
        # Validate input features (X)
        n_samples, n_features = X.shape
        expected_n_features = 358  # 1 ticker + 300 M3 + 50 H1 + 5 indicators + 2 time
        assert n_features == expected_n_features, f"Expected {expected_n_features} features, got {n_features}"
        
        # Validate target features (y)
        _, n_targets = y.shape
        expected_n_targets = 20  # 15 future + 5 growth
        assert n_targets == expected_n_targets, f"Expected {expected_n_targets} targets, got {n_targets}"
        
        # Check that all values are finite (no inf or nan after filtering)
        assert np.isfinite(X).all(), "X contains non-finite values"
        assert np.isfinite(y).all(), "y contains non-finite values"
        
        # Validate time features are in correct range [-1, 1]
        time_features = X[:, -2:]  # Last 2 columns should be time features
        assert np.all(time_features >= -1.0) and np.all(time_features <= 1.0), \
            "Time features should be in range [-1, 1]"
        
        # Validate growth length is non-negative
        growth_lengths = y[:, -1]  # Last column should be growth length
        assert np.all(growth_lengths >= 0), "Growth length should be non-negative"
        
        # Validate that we have reasonable numbers of samples
        assert n_samples > 0, "Should have generated at least one sample"
        
        print("✅ All requirement validations passed!")
        
        # Additional detailed validation
        print("\nDetailed feature structure validation:")
        print(f"- Ticker feature (col 0): range [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}]")
        print(f"- M3 features (cols 1-300): range [{X[:, 1:301].min():.6f}, {X[:, 1:301].max():.6f}]")
        print(f"- H1 features (cols 301-350): range [{X[:, 301:351].min():.6f}, {X[:, 301:351].max():.6f}]")
        print(f"- Tech indicators (cols 351-355): range [{X[:, 351:356].min():.6f}, {X[:, 351:356].max():.6f}]")
        print(f"- Time features (cols 356-357): range [{X[:, 356:358].min():.6f}, {X[:, 356:358].max():.6f}]")
        
        print(f"- Future M3 targets (cols 0-14): range [{y[:, :15].min():.6f}, {y[:, :15].max():.6f}]")
        print(f"- Growth OHLC (cols 15-18): range [{y[:, 15:19].min():.6f}, {y[:, 15:19].max():.6f}]")
        print(f"- Growth length (col 19): range [{y[:, 19].min():.1f}, {y[:, 19].max():.1f}]")
        
        print("\n✅ Implementation successfully meets all requirements!")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def validate_function_signature():
    """Validate that the required function signature exists and works"""
    print("\nValidating function signature...")
    
    # Import and verify the function exists with correct signature
    from feature_engineering import build_training_dataset
    import inspect
    
    sig = inspect.signature(build_training_dataset)
    params = list(sig.parameters.keys())
    assert params == ['parquet_path'], f"Function should accept parquet_path parameter, got {params}"
    
    return_annotation = sig.return_annotation
    print(f"Function return type annotation: {return_annotation}")
    
    print("✅ Function signature validation passed!")

if __name__ == "__main__":
    validate_function_signature()
    validate_requirements()
    print("\n🎉 All validations completed successfully! The implementation meets all requirements.")