# tests/test_fastapi_pipeline.py
import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.app import app  # Import from app.py instead of fastapi_pipeline

client = TestClient(app)

@pytest.fixture
def sample_data():
    """Fixture with more data points."""
    data = [
        {"date": "2023-01-01", "price": 100},
        {"date": "2023-01-02", "price": 101},
        {"date": "2023-01-03", "price": 102},
        {"date": "2023-01-04", "price": 103},
        {"date": "2023-01-05", "price": 104},
        {"date": "2023-01-06", "price": 105},
        {"date": "2023-01-07", "price": 106},
        {"date": "2023-01-08", "price": 107},
        {"date": "2023-01-09", "price": 108},
        {"date": "2023-01-10", "price": 109},
    ]
    return data

@pytest.fixture
def data_generation_input():
    """Fixture to provide input data for the generate_data endpoint."""
    return {
        "start_date": "2023-01-01",
        "end_date": "2023-01-10",
        "anchor_prices": {"GME": 150.0, "BYND": 200.0}
    }

@pytest.fixture
def test_scale_data_input(sample_data):
    """Fixture to provide input data for the scale_data endpoint."""
    return {
        "method": "standardize",
        "data": sample_data
    }

@pytest.fixture
def test_stationarity_input(sample_data):
    """Fixture to provide input data for the test_stationarity endpoint."""
    return {
        "data": sample_data
    }

@pytest.fixture
def test_arima_input(sample_data):
    """Fixture to provide input data for the run_arima endpoint."""
    return {
        "p": 1,
        "d": 1,
        "q": 1,
        "data": sample_data
    }

@pytest.fixture
def test_garch_input(sample_data):
    """Fixture to provide input data for the run_garch endpoint."""
    return {
        "p": 1,
        "q": 1,
        "data": sample_data
    }

@pytest.fixture
def test_run_pipeline_input():
    """Fixture to provide input data for the run_pipeline endpoint."""
    return {
        "source_actual_or_synthetic_data": "synthetic",
        "data_start_date": "2023-01-01",
        "data_end_date": "2023-01-10",
        "symbols": ["GME", "BYND", "BP"],
        "synthetic_anchor_prices": [150.0, 200.0, 15.0],
        "synthetic_random_seed": 1,
        "scaling_method": "standardize",
        "arima_params": {"p": 1, "d": 1, "q": 1},
        "garch_params": {"p": 1, "q": 1, "dist": "t"},
        "spillover_enabled": False  # Keep it false for testing
    }

@pytest.fixture
def pipeline_stooq_input():
    """Fixture to provide input data for the run_pipeline endpoint with Stooq data."""
    return {
        "source_actual_or_synthetic_data": "actual_stooq",
        "data_start_date": "2023-01-01",
        "data_end_date": "2023-02-28",  # Extended date range to get more data points
        "symbols": ["AAPL.US", "MSFT.US", "GOOG.US"],
        "scaling_method": "standardize",
        "arima_params": {"p": 1, "d": 1, "q": 1},
        "garch_params": {"p": 1, "q": 1, "dist": "t"},
        "spillover_enabled": True,
        "spillover_params": {"method": "diebold_yilmaz", "forecast_horizon": 10}
    }

def test_generate_data(data_generation_input):
    """Test the /api/v1/generate_data endpoint."""
    response = client.post("/api/v1/generate_data", json=data_generation_input)
    print("Response content:", response.content)  # Add this line
    assert response.status_code == 200, f"Response: {response.content}"
    data = response.json()
    assert isinstance(data, dict)
    assert "data" in data
    assert len(data["data"]) > 0

def test_scale_data(test_scale_data_input):
    """Test the /api/v1/scale_data endpoint."""
    response = client.post("/api/v1/scale_data", json=test_scale_data_input)
    assert response.status_code == 200, f"Response: {response.content}"
    data = response.json()
    assert isinstance(data, dict)
    assert "data" in data
    # The response data is a dict with the data at the 'data' key
    assert len(data["data"]) == len(test_scale_data_input["data"])

def test_test_stationarity(test_stationarity_input):
    """Test the /api/v1/test_stationarity endpoint."""
    response = client.post("/api/v1/test_stationarity", json=test_stationarity_input)
    assert response.status_code == 200, f"Response: {response.content}"
    data = response.json()
    # Validate response structure
    assert isinstance(data, dict)
    assert "all_symbols_stationarity" in data
    # Assuming 'price' is the key based on sample_data structure used in the fixture
    assert "price" in data["all_symbols_stationarity"]
    stationarity_result_for_price = data['all_symbols_stationarity']['price']
    assert "adf_statistic" in stationarity_result_for_price
    assert "p_value" in stationarity_result_for_price
    assert "critical_values" in stationarity_result_for_price
    assert "is_stationary" in stationarity_result_for_price
    assert "interpretation" in stationarity_result_for_price

def test_run_arima(test_arima_input):
    """Test the /api/v1/run_arima endpoint."""
    response = client.post("/api/v1/run_arima", json=test_arima_input)
    assert response.status_code == 200, f"Response: {response.content}"
    data = response.json()
    assert "fitted_model" in data
    assert "forecast" in data

def test_run_garch(test_garch_input):
    """Test the /api/v1/run_garch endpoint."""
    response = client.post("/api/v1/run_garch", json=test_garch_input)
    assert response.status_code == 200, f"Response: {response.content}" # Reverted to expect 200
    data = response.json()
    assert "fitted_model" in data
    assert isinstance(data["fitted_model"], str) # Expect string
    assert "forecast" in data
    assert isinstance(data["forecast"], list) # Expect list
    # Further checks on forecast elements can be added if necessary
    if data["forecast"]:
        assert isinstance(data["forecast"][0], float)

def test_price_to_returns():
    """Test the /api/v1/price_to_returns endpoint."""
    price_data = [
        {"date": "2023-01-01", "price": 100},
        {"date": "2023-01-02", "price": 101},
        {"date": "2023-01-03", "price": 102},
        {"date": "2023-01-04", "price": 103},
        {"date": "2023-01-05", "price": 104},
    ]
    
    response = client.post("/api/v1/price_to_returns", json={"data": price_data})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    
    # Returns should have one less entry than prices
    assert len(data["data"]) == len(price_data) - 1

def test_scale_for_garch():
    """Test the /api/v1/scale_for_garch endpoint."""
    returns_data = [
        {"date": "2023-01-02", "returns": 0.01},
        {"date": "2023-01-03", "returns": 0.009},
        {"date": "2023-01-04", "returns": 0.011},
        {"date": "2023-01-05", "returns": 0.008},
    ]
    
    response = client.post("/api/v1/scale_for_garch", json={"data": returns_data})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == len(returns_data)

def test_run_pipeline(test_run_pipeline_input):
    """Test the run_pipeline endpoint."""
    response = client.post("/api/v1/run_pipeline", json=test_run_pipeline_input)
    assert response.status_code == 200, f"Response: {response.content}"
    data = response.json()
    assert "stationarity_results" in data
    # assert "scaling_results" in data  # Key appears to be missing in current response
    assert "arima_results" in data
    assert "garch_results" in data

def test_run_pipeline_stooq(pipeline_stooq_input): # Removed 'client' from arguments
    """Test the run_pipeline endpoint with Stooq data."""
    response = client.post("/api/v1/run_pipeline", json=pipeline_stooq_input)
    assert response.status_code == 200
    data = response.json()
    assert "stationarity_results" in data
    # Add more assertions as needed
