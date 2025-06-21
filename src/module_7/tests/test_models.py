import pytest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.basket_model.feature_store import FeatureStore
from src.basket_model.basket_model import BasketModel
from src.exceptions import UserNotFoundException, PredictionException


class TestFeatureStore:
    """Test suite for the FeatureStore class"""
    
    @pytest.fixture
    def feature_store(self):
        """Create a FeatureStore instance for testing"""
        return FeatureStore()
    
    def test_feature_store_initialization(self, feature_store):
        """Test that FeatureStore initializes correctly"""
        assert feature_store.feature_store is not None
        assert isinstance(feature_store.feature_store, pd.DataFrame)
        assert feature_store.feature_store.shape[0] > 0
        assert feature_store.feature_store.shape[1] == 4  # Expected number of features
        
        # Check that expected columns exist
        expected_columns = [
            'prior_basket_value',
            'prior_item_count', 
            'prior_regulars_count',
            'regulars_count'
        ]
        for col in expected_columns:
            assert col in feature_store.feature_store.columns
    
    def test_get_features_valid_user(self, feature_store):
        """Test getting features for a valid user"""
        # Get the first user ID from the feature store
        first_user_id = feature_store.feature_store.index[0]
        
        features = feature_store.get_features(first_user_id)
        
        assert isinstance(features, pd.Series)
        assert len(features) == 4
        assert not features.isnull().any()  # No null values
    
    def test_get_features_invalid_user(self, feature_store):
        """Test getting features for an invalid user"""
        invalid_user_id = "nonexistent_user_12345"
        
        with pytest.raises(UserNotFoundException):
            feature_store.get_features(invalid_user_id)
    
    def test_feature_store_index_type(self, feature_store):
        """Test that the feature store index contains string user IDs"""
        user_ids = feature_store.feature_store.index
        assert len(user_ids) > 0
        
        # Check that user IDs are strings
        sample_user_id = user_ids[0]
        assert isinstance(sample_user_id, str)
        assert len(sample_user_id) > 0
    
    def test_feature_values_are_numeric(self, feature_store):
        """Test that all feature values are numeric"""
        sample_features = feature_store.feature_store.iloc[0]
        
        for feature_name, feature_value in sample_features.items():
            assert isinstance(feature_value, (int, float, np.number))
            assert not pd.isna(feature_value)


class TestBasketModel:
    """Test suite for the BasketModel class"""
    
    @pytest.fixture
    def basket_model(self):
        """Create a BasketModel instance for testing"""
        return BasketModel()
    
    def test_basket_model_initialization(self, basket_model):
        """Test that BasketModel initializes correctly"""
        assert basket_model.model is not None
        assert hasattr(basket_model.model, 'predict')
    
    def test_predict_valid_features(self, basket_model):
        """Test prediction with valid features"""
        # Create sample features (4 features as expected by the model)
        features = np.array([[50.0, 10.0, 2.0, 3.0]])
        
        prediction = basket_model.predict(features)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1
        assert isinstance(prediction[0], (int, float, np.number))
        assert prediction[0] >= 0  # Price should be non-negative
    
    def test_predict_multiple_samples(self, basket_model):
        """Test prediction with multiple samples"""
        # Create multiple sample features
        features = np.array([
            [50.0, 10.0, 2.0, 3.0],
            [30.0, 5.0, 1.0, 2.0],
            [80.0, 15.0, 4.0, 5.0]
        ])
        
        predictions = basket_model.predict(features)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        assert all(p >= 0 for p in predictions)  # All prices should be non-negative
    
    def test_predict_wrong_feature_count(self, basket_model):
        """Test prediction with wrong number of features"""
        # Create features with wrong number of columns
        features = np.array([[50.0, 10.0]])  # Only 2 features instead of 4
        
        with pytest.raises(PredictionException):
            basket_model.predict(features)
    
    def test_predict_invalid_input_type(self, basket_model):
        """Test prediction with invalid input type"""
        # Try to predict with a list instead of numpy array
        features = [50.0, 10.0, 2.0, 3.0]
        
        # Should handle gracefully or raise PredictionException
        try:
            prediction = basket_model.predict(features)
            # If it doesn't raise an exception, it should still return valid results
            assert isinstance(prediction, np.ndarray)
        except PredictionException:
            # This is also acceptable
            pass
    
    @patch('joblib.load')
    def test_model_loading_failure(self, mock_joblib_load):
        """Test handling of model loading failure"""
        mock_joblib_load.side_effect = Exception("Failed to load model")
        
        with pytest.raises(Exception):
            BasketModel()


class TestIntegration:
    """Integration tests for FeatureStore and BasketModel together"""
    
    @pytest.fixture
    def feature_store(self):
        return FeatureStore()
    
    @pytest.fixture 
    def basket_model(self):
        return BasketModel()
    
    def test_end_to_end_prediction(self, feature_store, basket_model):
        """Test complete end-to-end prediction flow"""
        # Get a valid user ID
        user_id = feature_store.feature_store.index[0]
        
        # Get features for the user
        user_features = feature_store.get_features(user_id)
        
        # Convert to numpy array for prediction
        features_array = np.array(user_features).reshape(1, -1)
        
        # Make prediction
        prediction = basket_model.predict(features_array)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1
        assert prediction[0] >= 0
    
    def test_multiple_users_prediction(self, feature_store, basket_model):
        """Test predictions for multiple users"""
        # Get multiple user IDs
        user_ids = feature_store.feature_store.index[:5]  # First 5 users
        
        predictions = []
        for user_id in user_ids:
            user_features = feature_store.get_features(user_id)
            features_array = np.array(user_features).reshape(1, -1)
            prediction = basket_model.predict(features_array)[0]
            predictions.append(prediction)
        
        assert len(predictions) == 5
        assert all(p >= 0 for p in predictions)
        
        # Predictions should have some variance (not all identical)
        assert len(set(predictions)) > 1  # At least 2 different predictions
    
    def test_feature_consistency(self, feature_store):
        """Test that features are consistent across calls"""
        user_id = feature_store.feature_store.index[0]
        
        # Get features multiple times
        features1 = feature_store.get_features(user_id)
        features2 = feature_store.get_features(user_id)
        
        # Should be identical
        pd.testing.assert_series_equal(features1, features2)


if __name__ == "__main__":
    pytest.main([__file__]) 