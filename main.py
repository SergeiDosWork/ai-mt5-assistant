from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Any
import os
import tempfile
import json
from data_loader import validate_parquet_format, load_parquet_data
from feature_engineering import generate_training_samples
from model import train_model
from inference import process_prediction_request

# Initialize FastAPI app
app = FastAPI(title="AI Trading Server")

# Serve static files and templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page with file upload form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train")
async def train_endpoint(file: UploadFile = File(...)):
    """Endpoint to upload parquet file and train the model"""
    try:
        # Validate file extension
        if not file.filename.lower().endswith('.parquet'):
            raise HTTPException(status_code=400, detail="Only .parquet files are allowed")
        
        # Save uploaded file temporarily
        temp_file_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        try:
            # Validate parquet format
            if not validate_parquet_format(temp_file_path):
                raise HTTPException(status_code=400, detail="Invalid parquet file format")
            
            # Load the data
            data = load_parquet_data(temp_file_path)
            
            # Generate training samples
            X, y = generate_training_samples(data)
            
            if len(X) == 0:
                raise HTTPException(status_code=400, detail="Not enough data to generate training samples")
            
            # Train the model
            train_model(X, y)
            
            # Return success response with metadata
            from model import load_model
            _, metadata = load_model()
            
            return {
                "status": "success",
                "message": "Model trained and saved successfully",
                "training_date": metadata['training_date'],
                "train_samples": metadata['train_samples']
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict")
async def predict_endpoint(request: Request):
    """Endpoint for receiving prediction requests from MT5"""
    try:
        # Parse JSON request
        request_data = await request.json()
        
        # Process the prediction request
        result = process_prediction_request(request_data)
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Trading Server"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)