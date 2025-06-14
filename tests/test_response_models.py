# tests/test_response_models.py

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from app.py instead of fastapi_pipeline
from api.app import app
from api.models.response import GARCHModelResponse

client = TestClient(app)

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    return [
        {"date": "2023-01-01", "price": 100},
        {"date": "2023-01-02", "price": 101},
        {"date": "2023-01-03", "price": 102},
        {"date": "2023-01-04", "price": 103},
        {"date": "2023-01-05", "price": 104},
    ]

def test_run_garch_response_model(sample_data):
    """Test that the run_garch endpoint returns a response matching GARCHModelResponse model."""
    input_data = {
        "p": 1,
        "q": 1,
        "data": sample_data
    }
    
    response = client.post("/api/v1/run_garch", json=input_data)
    assert response.status_code == 200, f"Response: {response.content}" # Reverted to expect 200
    
    # Check that the response has the expected structure
    data = response.json()
    assert "fitted_model" in data
    assert isinstance(data["fitted_model"], str)
    assert "forecast" in data
    assert isinstance(data["forecast"], list)
    if data["forecast"]:
        assert isinstance(data["forecast"][0], float)
    
    # Validate against the Pydantic model
    validated_data = GARCHModelResponse(**data) # Should pass now
    
    # Use model_dump() instead of dict() which is deprecated in Pydantic v2
    if hasattr(validated_data, "model_dump"):
        # Pydantic v2
        assert validated_data.model_dump() == data
    else:
        # Fallback for Pydantic v1
        assert validated_data.dict() == data
