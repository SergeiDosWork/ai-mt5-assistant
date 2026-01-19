import pandas as pd
import numpy as np
from typing import Tuple, List


def aggregate_m3_to_h1(m3_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates M3 candles to H1 candles
    
    Args:
        m3_data: DataFrame with M3 candle data
        
    Returns:
        DataFrame with H1 candle data
    """
    # Assuming m3_data has datetime index
    if 'time' in m3_data.columns:
        m3_data = m3_data.set_index('time')
    
    # Resample to H1 (hourly) and aggregate OHLCV
    h1_data = m3_data.resample('H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to make time a column again
    h1_data = h1_data.reset_index()
    
    return h1_data


def generate_training_samples(data: pd.DataFrame, 
                           m3_window_size: int = 60, 
                           h1_window_size: int = 10,
                           prediction_horizon: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates training samples from historical data
    
    Args:
        data: DataFrame with historical price data
        m3_window_size: Size of M3 window (default 60)
        h1_window_size: Size of H1 window (default 10)
        prediction_horizon: How many periods ahead to predict (default 3)
        
    Returns:
        Tuple of (X, y) where X contains features and y contains targets
    """
    # First aggregate M3 data to H1
    h1_data = aggregate_m3_to_h1(data.copy())
    
    # Prepare features for both M3 and H1
    features_m3 = ['open', 'high', 'low', 'close', 'volume']
    features_h1 = ['open', 'high', 'low', 'close', 'volume']
    
    # Convert to numpy arrays for easier indexing
    m3_values = data[features_m3].values
    h1_values = h1_data[features_h1].values
    
    X, y = [], []
    
    # Generate samples
    for i in range(m3_window_size + prediction_horizon, len(m3_values)):
        # Extract M3 window (last 60 candles)
        m3_features = m3_values[i - m3_window_size:i]
        
        # Find corresponding H1 window (last 10 H1 candles before current time)
        current_time = data.iloc[i]['time']
        h1_start_idx = max(0, len(h1_data) - h1_window_size)
        h1_features = h1_values[h1_start_idx:h1_start_idx + h1_window_size]
        
        # Pad or truncate H1 features if needed
        if len(h1_features) < h1_window_size:
            # Pad with zeros if we don't have enough H1 data
            padding_needed = h1_window_size - len(h1_features)
            padding = np.zeros((padding_needed, len(features_h1)))
            h1_features = np.vstack([padding, h1_features])
        elif len(h1_features) > h1_window_size:
            # Take the last h1_window_size values
            h1_features = h1_features[-h1_window_size:]
            
        # Combine M3 and H1 features
        combined_features = np.concatenate([m3_features.flatten(), h1_features.flatten()])
        
        # Create target: direction of movement over next 3 M3 candles
        current_price = m3_values[i - 1, 3]  # Close price of last candle
        future_price = m3_values[i, 3]       # Close price after prediction horizon
        
        if future_price > current_price:
            target = 1  # Up
        elif future_price < current_price:
            target = -1 # Down
        else:
            target = 0  # Neutral
            
        X.append(combined_features)
        y.append(target)
    
    return np.array(X), np.array(y)


def prepare_inference_features(m3_data: List[dict], h1_data: List[dict]) -> np.ndarray:
    """
    Prepares features for inference from incoming MT5 data
    
    Args:
        m3_data: List of M3 candle data from MT5
        h1_data: List of H1 candle data from MT5
        
    Returns:
        Feature array ready for model inference
    """
    # Convert to DataFrames
    m3_df = pd.DataFrame(m3_data)
    h1_df = pd.DataFrame(h1_data)
    
    # Select features
    features_m3 = ['open', 'high', 'low', 'close', 'volume']
    features_h1 = ['open', 'high', 'low', 'close', 'volume']
    
    # Extract values
    m3_values = m3_df[features_m3].values
    h1_values = h1_df[features_h1].values
    
    # Flatten and combine
    combined_features = np.concatenate([m3_values.flatten(), h1_values.flatten()])
    
    # Reshape to match model input shape (batch_size, feature_dim)
    return combined_features.reshape(1, -1)