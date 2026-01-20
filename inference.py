from typing import Dict, Any, Tuple, List
import numpy as np
from model import load_model, predict_direction
from feature_engineering import prepare_inference_features


def process_prediction_request(request_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Processes a prediction request from MT5
    
    Args:
        request_data: Dictionary containing symbol, m3, and h1 data
        
    Returns:
        Dictionary with signal, probability, and confidence
    """
    try:
        # Extract data from request
        symbol = request_data.get('symbol')
        m3_data = request_data.get('m3', [])
        h1_data = request_data.get('h1', [])
        
        # Validate input data
        if not m3_data or not h1_data:
            raise ValueError("M3 and H1 data are required")
        
        if len(m3_data) != 60:
            raise ValueError(f"M3 data must contain exactly 60 candles, got {len(m3_data)}")
        
        if len(h1_data) != 10:
            raise ValueError(f"H1 data must contain exactly 10 candles, got {len(h1_data)}")
        
        # Prepare features for inference
        features = prepare_inference_features(m3_data, h1_data)
        
        # Load the trained model
        model, metadata = load_model()
        
        # Make prediction
        signal, probability, confidence = predict_direction(model, features)
        
        # Return prediction result
        result = {
            "signal": int(signal),
            "probability": float(probability),
            "confidence": float(confidence)
        }
        
        return result
        
    except Exception as e:
        # Log error and return error response
        print(f"Error in process_prediction_request: {str(e)}")
        raise e


def generate_forecast(current_candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates forecast for the next 3 candles based on current 60 candles
    
    Args:
        current_candles: List of 60 current candles
        
    Returns:
        Dictionary containing predicted candles and growth information
    """
    try:
        # Validate input data
        if len(current_candles) != 60:
            raise ValueError(f"Expected 60 candles, got {len(current_candles)}")
        
        # For now, implement a basic forecasting algorithm
        # In a real implementation, you would use your trained neural network to predict the next 3 candles
        # This is a placeholder implementation that simulates prediction
        
        # Get the last candle's closing price as reference
        last_close = current_candles[-1]['close'] if 'close' in current_candles[-1] else current_candles[-1][3]
        
        # Generate 3 predicted candles
        predicted_candles = []
        for i in range(1, 4):
            # Simple simulation of prediction - in reality this would come from your neural network
            # Adding some random variation based on the last close
            multiplier = 0.0001 * i  # Small increment per prediction step
            direction = (-1) ** i    # Alternate direction for demonstration
            
            pred_open = last_close + (multiplier * direction)
            pred_close = pred_open + (0.0002 * direction)
            pred_high = max(pred_open, pred_close) + 0.0001
            pred_low = min(pred_open, pred_close) - 0.0001
            
            predicted_candles.append({
                "time": f"predicted_{i}",
                "open": round(pred_open, 5),
                "high": round(pred_high, 5),
                "low": round(pred_low, 5),
                "close": round(pred_close, 5)
            })
        
        # Simulate growth duration and size (in a real implementation, these would come from the model)
        growth_duration = np.random.randint(3, 10)  # Random duration between 3-10 periods
        growth_size = round(np.random.uniform(0.001, 0.005), 5)  # Random size between 0.001-0.005
        
        return {
            "predicted_candles": predicted_candles,
            "growth_duration": growth_duration,
            "growth_size": growth_size
        }
        
    except Exception as e:
        print(f"Error in generate_forecast: {str(e)}")
        raise e