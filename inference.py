from typing import Dict, Any, Tuple
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