"""
Demonstration script for the Basket Model API

This script demonstrates all the functionality of the FastAPI basket model API:
1. Health check endpoint
2. Successful predictions
3. Error handling
4. Metrics logging
"""

import sys
sys.path.append('src')

from fastapi.testclient import TestClient
from app import app
from src.basket_model.feature_store import FeatureStore
import json
import time

def main():
    print("🛒 Basket Model API Demonstration")
    print("=" * 50)
    
    # Initialize test client
    client = TestClient(app)
    
    # Test 1: Root endpoint
    print("\n1. Testing Root Endpoint")
    print("-" * 25)
    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Health check
    print("\n2. Testing Health Check (/status)")
    print("-" * 35)
    response = client.get("/status")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"API Status: {data['status']}")
        print(f"Feature Store Size: {data['feature_store_size']} users")
        print(f"Timestamp: {data['timestamp']}")
    else:
        print(f"Error: {response.json()}")
    
    # Test 3: Get some real user IDs
    print("\n3. Getting Real User IDs for Testing")
    print("-" * 38)
    fs = FeatureStore()
    test_users = fs.feature_store.index[:3].tolist()
    print(f"Available users: {len(fs.feature_store.index)}")
    print("Sample user IDs:")
    for i, user_id in enumerate(test_users):
        print(f"  {i+1}. {user_id[:30]}...")
    
    # Test 4: Successful predictions
    print("\n4. Testing Successful Predictions")
    print("-" * 35)
    for i, user_id in enumerate(test_users[:2]):  # Test first 2 users
        print(f"\nPrediction {i+1}:")
        start_time = time.time()
        response = client.post("/predict", json={"user_id": user_id})
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ User: {user_id[:20]}...")
            print(f"  💰 Predicted Price: ${data['predicted_price']:.2f}")
            print(f"  ⏱️  Latency: {latency:.4f}s")
            print(f"  🕐 Timestamp: {data['timestamp']}")
        else:
            print(f"  ❌ Error: {response.json()}")
    
    # Test 5: Error handling - User not found
    print("\n5. Testing Error Handling")
    print("-" * 28)
    print("Testing with non-existent user...")
    response = client.post("/predict", json={"user_id": "fake_user_12345"})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 6: Error handling - Empty user ID
    print("\nTesting with empty user ID...")
    response = client.post("/predict", json={"user_id": ""})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 7: Error handling - Missing user ID
    print("\nTesting with missing user ID...")
    response = client.post("/predict", json={})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 8: Check metrics
    print("\n6. Checking Metrics Logging")
    print("-" * 28)
    try:
        with open("logs/api_metrics.txt", "r") as f:
            metrics = f.readlines()
        
        print(f"Total metrics logged: {len(metrics)}")
        print("Recent metrics (last 5):")
        for line in metrics[-5:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                timestamp = parts[0]
                event_type = parts[1]
                print(f"  📊 {timestamp[:19]} - {event_type}")
        
        # Count event types
        event_counts = {}
        for line in metrics:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                event_type = parts[1]
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        print("\nEvent summary:")
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}")
            
    except FileNotFoundError:
        print("No metrics file found")
    
    print("\n" + "=" * 50)
    print("🎉 API Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ FastAPI web framework with automatic OpenAPI docs")
    print("✅ Machine learning model integration")
    print("✅ Feature store data loading")
    print("✅ Comprehensive error handling")
    print("✅ Request/response validation with Pydantic")
    print("✅ Metrics logging (latency, errors, predictions)")
    print("✅ Health check endpoint")
    print("✅ Robust sklearn version compatibility handling")
    print("\n📚 API Documentation available at: http://localhost:8000/docs")
    print("📊 Metrics logged to: logs/api_metrics.txt")

if __name__ == "__main__":
    main() 