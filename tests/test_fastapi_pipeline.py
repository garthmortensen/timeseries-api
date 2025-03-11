# tests/test_fastapi_pipeline.py
import pytest
import httpx  # allows async requests
from fastapi.testclient import TestClient

# add parent dir to PYTHONPATH so app can be imported
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi_pipeline import app

client = TestClient(app)


# fixtures are reusable objects
# they're called fixtures because they're set up before the test runs
# this one will be used as input arguments in various tests below
# im also sometimes using them to improve test readability
@pytest.fixture
def data_generation_input():
    """Fixture to provide input data for the generate_data endpoint."""
    return {
        "start_date": "2023-01-01",
        "end_date": "2023-01-10",
        "anchor_prices": {"GME": 150.0, "BYND": 200.0},
    }


def test_generate_data(data_generation_input):
    response = client.post("/generate_data", json=data_generation_input)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


@pytest.fixture
def sample_data():
    """sample data."""
    return [
        {"date": "2023-01-01", "price": 100},
        {"date": "2023-01-02", "price": 101},
        {"date": "2023-01-03", "price": 102},
        {"date": "2023-01-04", "price": 103},
        {"date": "2023-01-05", "price": 104},
    ]


@pytest.fixture
def test_scale_data_input(sample_data):
    return {"method": "standardize", "data": sample_data}


def test_scale_data(test_scale_data_input):
    response = client.post("/scale_data", json=test_scale_data_input)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_run_arima(sample_data):
    response = client.post(
        "/run_arima", json={"p": 1, "d": 1, "q": 1, "data": sample_data}
    )
    assert (
        response.status_code == 200
    ), f"Response status code: {response.status_code}, Response content: {response.content}"
    data = response.json()
    assert "fitted_model" in data
    assert "forecast" in data


def test_run_garch(sample_data):
    response = client.post("/run_garch", json={"p": 1, "q": 1, "data": sample_data})
    assert response.status_code == 200
    data = response.json()
    assert "fitted_model" in data
    assert "forecast" in data


@pytest.fixture
def test_run_pipeline_input():
    return {
        "start_date": "2023-01-01",
        "end_date": "2023-01-10",
        "anchor_prices": {"GME": 150.0, "BYND": 200.0},
        "scaling_method": "standardize",
        "arima_params": {"p": 1, "d": 1, "q": 1},
        "garch_params": {"p": 1, "q": 1},
    }


def test_run_pipeline(test_run_pipeline_input):
    response = client.post("/run_pipeline", json=test_run_pipeline_input)
    assert response.status_code == 200
    data = response.json()
    assert "stationarity_results" in data
    assert "arima_summary" in data
    assert "arima_forecast" in data
    assert "garch_summary" in data
    assert "garch_forecast" in data
