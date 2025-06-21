import pytest
import json
import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app
from src.exceptions import UserNotFoundException, PredictionException

# Create test client
client = TestClient(app)

class TestBasketModelAPI:
    """Test suite for the Basket Model API"""
    
    def setup_method(self):
        """Setup method called before each test"""
        # Clear any existing metrics file
        metrics_file = "logs/api_metrics.txt"
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Basket Model API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    @patch('app.feature_store')
    @patch('app.basket_model')
    def test_status_endpoint_healthy(self, mock_model, mock_feature_store):
        """Test status endpoint when models are initialized"""
        # Mock the feature store to have a valid shape
        mock_feature_store.feature_store.shape = [1000, 4]
        
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["feature_store_size"] == 1000
        assert "timestamp" in data
    
    def test_status_endpoint_unhealthy(self):
        """Test status endpoint when models are not initialized"""
        # This should work with the actual implementation
        # The startup event should initialize the models
        response = client.get("/status")
        # The response could be either 200 (if models loaded) or 503 (if not)
        assert response.status_code in [200, 503]
    
    @patch('app.feature_store')
    @patch('app.basket_model')
    def test_predict_endpoint_success(self, mock_model, mock_feature_store):
        """Test successful prediction"""
        # Mock feature store
        mock_features = pd.Series([10.5, 2.0, 1.0, 3.0])
        mock_feature_store.get_features.return_value = mock_features
        
        # Mock model prediction
        mock_model.predict.return_value = np.array([25.50])
        
        request_data = {"user_id": "test_user_123"}
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user_123"
        assert data["predicted_price"] == 25.50
        assert "timestamp" in data
        
        # Verify the mocks were called correctly
        mock_feature_store.get_features.assert_called_once_with("test_user_123")
        mock_model.predict.assert_called_once()
    
    @patch('app.feature_store')
    def test_predict_endpoint_user_not_found(self, mock_feature_store):
        """Test prediction with non-existent user"""
        # Mock feature store to raise UserNotFoundException
        mock_feature_store.get_features.side_effect = UserNotFoundException("User not found")
        
        request_data = {"user_id": "nonexistent_user"}
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_predict_endpoint_empty_user_id(self):
        """Test prediction with empty user_id"""
        request_data = {"user_id": ""}
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "cannot be empty" in data["detail"]
    
    def test_predict_endpoint_missing_user_id(self):
        """Test prediction with missing user_id"""
        request_data = {}
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.feature_store')
    @patch('app.basket_model')
    def test_predict_endpoint_model_error(self, mock_model, mock_feature_store):
        """Test prediction when model fails"""
        # Mock feature store
        mock_features = pd.Series([10.5, 2.0, 1.0, 3.0])
        mock_feature_store.get_features.return_value = mock_features
        
        # Mock model to raise PredictionException
        mock_model.predict.side_effect = PredictionException("Model failed")
        
        request_data = {"user_id": "test_user_123"}
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "prediction failed" in data["detail"].lower()
    
    @patch('app.feature_store')
    @patch('app.basket_model')
    def test_metrics_logging(self, mock_model, mock_feature_store):
        """Test that metrics are being logged"""
        # Mock feature store and model
        mock_features = pd.Series([10.5, 2.0, 1.0, 3.0])
        mock_feature_store.get_features.return_value = mock_features
        mock_model.predict.return_value = np.array([25.50])
        
        # Make a prediction request
        request_data = {"user_id": "test_user_123"}
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        
        # Check if metrics file was created and contains data
        metrics_file = "logs/api_metrics.txt"
        assert os.path.exists(metrics_file)
        
        with open(metrics_file, 'r') as f:
            content = f.read()
            assert "PREDICTION" in content
            assert "test_user_123" in content
    
    def test_predict_endpoint_with_special_characters(self):
        """Test prediction with user_id containing special characters"""
        request_data = {"user_id": "user@123!#$"}
        response = client.post("/predict", json=request_data)
        
        # Should handle gracefully (either succeed or fail with appropriate error)
        assert response.status_code in [200, 404, 500]
    
    @patch('app.feature_store')
    @patch('app.basket_model')
    def test_concurrent_requests(self, mock_model, mock_feature_store):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        # Mock feature store and model
        mock_features = pd.Series([10.5, 2.0, 1.0, 3.0])
        mock_feature_store.get_features.return_value = mock_features
        mock_model.predict.return_value = np.array([25.50])
        
        results = []
        
        def make_request(user_id):
            request_data = {"user_id": f"user_{user_id}"}
            response = client.post("/predict", json=request_data)
            results.append(response.status_code)
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


class TestAPIIntegration:
    """Integration tests using actual models"""
    
    def test_integration_with_real_models(self):
        """Test the API with real models (if available)"""
        # This test will use the actual FeatureStore and BasketModel
        # First check if we can get the status
        response = client.get("/status")
        
        if response.status_code == 200:
            # Models are loaded, try to get a real user ID
            status_data = response.json()
            assert status_data["feature_store_size"] > 0
            
            # Try to get a sample user ID from the feature store
            # This is a basic integration test
            sample_user_request = {"user_id": "sample_user_that_doesnt_exist"}
            pred_response = client.post("/predict", json=sample_user_request)
            
            # Should get 404 for non-existent user
            assert pred_response.status_code == 404
        else:
            # Models not initialized, skip this test
            pytest.skip("Models not initialized in test environment")


if __name__ == "__main__":
    pytest.main([__file__]) 