import uvicorn
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
sys.path.append('src')

from src.basket_model.feature_store import FeatureStore
from src.basket_model.basket_model import BasketModel
from src.exceptions import UserNotFoundException, PredictionException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Metrics file
METRICS_FILE = logs_dir / "api_metrics.txt"

class PredictionRequest(BaseModel):
    user_id: str

class PredictionResponse(BaseModel):
    user_id: str
    predicted_price: float
    timestamp: str

class StatusResponse(BaseModel):
    status: str
    timestamp: str
    feature_store_size: int


# Global variables for model and feature store
feature_store = None
basket_model = None

def log_metrics(event_type: str, user_id: str = None, latency: float = None, 
                prediction: float = None, error: str = None):
    """Log metrics to file"""
    timestamp = datetime.now().isoformat()
    
    with open(METRICS_FILE, "a") as f:
        if event_type == "request":
            f.write(f"{timestamp},REQUEST,{user_id},{latency:.4f}\n")
        elif event_type == "prediction":
            f.write(f"{timestamp},PREDICTION,{user_id},{prediction:.4f},{latency:.4f}\n")
        elif event_type == "error":
            f.write(f"{timestamp},ERROR,{user_id},{error},{latency:.4f}\n")
        elif event_type == "status":
            f.write(f"{timestamp},STATUS_CHECK,{latency:.4f}\n")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global feature_store, basket_model
    
    try:
        logger.info("Initializing Feature Store...")
        feature_store = FeatureStore()
        logger.info(f"Feature Store initialized with {feature_store.feature_store.shape[0]} users")
        
        logger.info("Loading Basket Model...")
        basket_model = BasketModel()
        logger.info("Basket Model loaded successfully")
        
        # Log startup metrics
        with open(METRICS_FILE, "a") as f:
            f.write(f"{datetime.now().isoformat()},STARTUP,SUCCESS\n")
            
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        with open(METRICS_FILE, "a") as f:
            f.write(f"{datetime.now().isoformat()},STARTUP,FAILED,{str(e)}\n")
        raise
    
    yield  # App runs here
    
    # Cleanup (if needed) happens here after yield

# Initialize FastAPI app
app = FastAPI(
    title="Basket Model API",
    description="API for predicting basket prices based on user features",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Health check endpoint"""
    start_time = time.time()
    
    try:
        # Lazy initialization for test environments
        global feature_store, basket_model
        if feature_store is None or basket_model is None:
            # Initialize models inline for test environments
            feature_store = FeatureStore()
            basket_model = BasketModel()
        
        response = StatusResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            feature_store_size=feature_store.feature_store.shape[0]
        )
        
        latency = time.time() - start_time
        log_metrics("status", latency=latency)
        
        return response
        
    except Exception as e:
        latency = time.time() - start_time
        log_metrics("error", error=str(e), latency=latency)
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", response_model=PredictionResponse)
async def predict_basket_price(request: PredictionRequest):
    """Predict basket price for a given user"""
    start_time = time.time()
    user_id = request.user_id
    
    try:
        # Lazy initialization for test environments
        global feature_store, basket_model
        if feature_store is None or basket_model is None:
            # Initialize models inline for test environments
            feature_store = FeatureStore()
            basket_model = BasketModel()
        
        # Input validation
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
        # Get features for the user
        try:
            user_features = feature_store.get_features(user_id)
        except UserNotFoundException:
            latency = time.time() - start_time
            log_metrics("error", user_id=user_id, error="UserNotFound", latency=latency)
            raise HTTPException(status_code=404, detail=f"User {user_id} not found in feature store")
        
        # Convert features to numpy array for prediction
        features_array = np.array(user_features).reshape(1, -1)
        
        # Make prediction
        try:
            prediction = basket_model.predict(features_array)[0]
        except PredictionException as e:
            latency = time.time() - start_time
            log_metrics("error", user_id=user_id, error="PredictionFailed", latency=latency)
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
        # Prepare response
        response = PredictionResponse(
            user_id=user_id,
            predicted_price=float(prediction),
            timestamp=datetime.now().isoformat()
        )
        
        latency = time.time() - start_time
        log_metrics("prediction", user_id=user_id, prediction=float(prediction), latency=latency)
        
        logger.info(f"Prediction for user {user_id}: ${prediction:.2f} (latency: {latency:.4f}s)")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        latency = time.time() - start_time
        log_metrics("error", user_id=user_id, error=str(e), latency=latency)
        logger.error(f"Unexpected error for user {user_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Basket Model API", 
        "version": "1.0.0",
        "endpoints": {
            "status": "/status",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)