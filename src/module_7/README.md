# Basket Model API

A FastAPI-based web service for predicting basket prices using machine learning. This API provides real-time price predictions based on user shopping history and behavior patterns.

## Features

- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Machine Learning Integration**: Scikit-learn model for basket price prediction
- **Feature Store**: Preprocessing pipeline for user shopping features
- **Comprehensive Error Handling**: Robust error handling with meaningful HTTP status codes
- **Metrics Logging**: Detailed logging of API performance and model predictions
- **Health Check**: Status endpoint for monitoring API health
- **Request Validation**: Pydantic models for request/response validation
- **Sklearn Compatibility**: Handles version compatibility issues between sklearn versions

## API Endpoints

### GET `/`
Root endpoint providing API information and available endpoints.

### GET `/status`
Health check endpoint that returns:
- API status (healthy/unhealthy)
- Feature store size (number of users)
- Current timestamp

### POST `/predict`
Predicts basket price for a given user.

**Request Body:**
```json
{
  "user_id": "string"
}
```

**Response:**
```json
{
  "user_id": "string",
  "predicted_price": 75.50,
  "timestamp": "2025-06-14T18:53:57.718441"
}
```

**Error Responses:**
- `400`: Invalid input (empty user_id)
- `404`: User not found in feature store
- `422`: Validation error (missing user_id)
- `500`: Internal server error

## Project Structure

```
src/module_7/
├── app.py                      # Main FastAPI application
├── src/
│   ├── exceptions.py           # Custom exception classes
│   └── basket_model/
│       ├── __init__.py
│       ├── basket_model.py     # ML model wrapper
│       ├── feature_store.py    # Feature preprocessing
│       └── utils/
│           ├── __init__.py
│           ├── features.py     # Feature engineering functions
│           └── loaders.py      # Data loading utilities
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # API endpoint tests
│   └── test_models.py         # Model component tests
├── data/                      # Training data (parquet files)
├── bin/                       # Trained model (model.joblib)
├── logs/                      # API metrics and logs
├── demo_api.py               # Comprehensive demo script
└── README.md                 # This file
```

## Dependencies

- **FastAPI**: Web framework
- **uvicorn**: ASGI server
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (via joblib)
- **pydantic**: Data validation
- **pytest**: Testing framework
- **httpx**: HTTP client for testing

## Installation

1. **Install dependencies** (from project root):
   ```bash
   poetry add fastapi uvicorn[standard] pytest httpx
   ```

2. **Data Setup**: Ensure the following files are in place:
   - `data/orders.parquet`
   - `data/regulars.parquet`
   - `data/inventory.parquet`
   - `bin/model.joblib`

## Usage

### Running the API Server

```bash
cd src/module_7
python app.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Running the Demo

```bash
cd src/module_7
python demo_api.py
```

This runs a comprehensive demonstration showing all API features.

### Running Tests

```bash
cd src/module_7
python -m pytest tests/ -v
```

## API Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/status

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "your_user_id_here"}'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/status")
print(response.json())

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"user_id": "your_user_id_here"}
)
print(response.json())
```

### Using FastAPI TestClient

```python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Test endpoints
response = client.get("/status")
response = client.post("/predict", json={"user_id": "test_user"})
```

## Model Details

The API uses a trained Random Forest Regressor that predicts basket prices based on:

- **prior_basket_value**: Previous basket value
- **prior_item_count**: Number of items in previous order
- **prior_regulars_count**: Number of regular items in previous order
- **regulars_count**: Total number of regular items for user

The feature store automatically handles:
- Data loading from parquet files
- Feature engineering and preprocessing
- Duplicate user ID resolution (uses most recent data)
- Missing value handling

## Metrics and Logging

The API automatically logs metrics to `logs/api_metrics.txt`:

- **STARTUP**: Model initialization events
- **STATUS_CHECK**: Health check requests with latency
- **PREDICTION**: Successful predictions with user ID, price, and latency
- **ERROR**: Error events with details and latency

Example metrics entry:
```
2025-06-14T18:53:57.718441,PREDICTION,user123,75.50,0.0026
```

## Error Handling

The API implements comprehensive error handling:

1. **Input Validation**: Pydantic models validate request structure
2. **Business Logic Errors**: Custom exceptions for domain-specific errors
3. **Model Errors**: Graceful handling of sklearn compatibility issues
4. **HTTP Error Codes**: Meaningful status codes for different error types
5. **Error Logging**: All errors are logged with context

## Testing

The project includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **API Tests**: HTTP endpoint testing with mocked and real components
- **Error Testing**: Comprehensive error scenario coverage

## Performance Considerations

- **Model Loading**: Models are loaded once at startup and reused
- **Feature Caching**: Feature store is computed once and cached in memory
- **Sklearn Compatibility**: Automatic handling of version mismatches
- **Lazy Initialization**: Models initialize on first request in test environments

## Troubleshooting

### Common Issues

1. **"Models not initialized"**: Check that data files are in correct locations
2. **sklearn version warnings**: These are handled automatically with compatibility patches
3. **User not found**: Ensure user ID exists in the feature store data
4. **Port already in use**: Change port in `app.py` or kill existing process

### Data Requirements

- **orders.parquet**: Must contain columns: user_id, created_at, ordered_items
- **regulars.parquet**: Must contain columns: user_id, variant_id
- **inventory.parquet**: Must contain columns: variant_id, price
- **model.joblib**: Trained sklearn model compatible with 4 features

## Production Deployment

For production deployment:

1. Use proper ASGI server like uvicorn with multiple workers
2. Add authentication and authorization
3. Implement rate limiting
4. Add monitoring and alerting
5. Use external logging service
6. Add database for persistent metrics storage
7. Implement model versioning and A/B testing

## License

This project is part of the Zrive Data Science module 7 coursework. 