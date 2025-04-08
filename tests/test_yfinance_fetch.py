# tests/test_yfinance_fetch.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.app import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def market_data_input():
    """Fixture to provide input data for the fetch_market_data endpoint."""
    return {
        "symbols": ["^DJI", "^HSI"],
        "start_date": "2023-01-01",
        "end_date": "2023-03-01",
        "interval": "1d"
    }

@pytest.fixture
def mock_yfinance_data():
    """Fixture to create mock market data that yfinance would return."""
    # Create dates for the test period
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    
    # Create multi-index columns for two symbols
    columns = pd.MultiIndex.from_product([['Adj Close'], ["^DJI", "^HSI"]])
    
    # Create mock data with random prices
    np.random.seed(42)  # For reproducibility
    data = np.random.uniform(100, 200, size=(len(dates), 2))
    
    # Create DataFrame with proper structure
    df = pd.DataFrame(data, index=dates, columns=columns)
    return df

@patch('api.services.market_data_service.yf.download')
def test_fetch_market_data(mock_download, client, market_data_input, mock_yfinance_data):
    """Test the /api/v1/fetch_market_data endpoint."""
    # Configure the mock to return our fixture data
    mock_download.return_value = mock_yfinance_data
    
    # Mock the fetch_market_data function to return a proper tuple
    # with the first element in the format {date_str: {symbol: price}}
    with patch('api.routers.data.fetch_market_data') as mock_fetch:
        # Create a properly formatted return value
        formatted_data = {}
        for date in mock_yfinance_data.index:
            date_str = str(date)
            formatted_data[date_str] = {}
            for symbol in ["^DJI", "^HSI"]:
                if symbol in mock_yfinance_data.columns:
                    formatted_data[date_str][symbol] = mock_yfinance_data.loc[date, symbol]
        
        mock_fetch.return_value = (formatted_data, mock_yfinance_data)
        
        # Call the API endpoint
        response = client.post("/api/v1/fetch_market_data", json=market_data_input)
        
        # Check the response
        assert response.status_code == 200, f"Response: {response.content}"
        data = response.json()
        
        # Check structure
        assert isinstance(data, dict)
        assert "data" in data
        
        # Check content
        market_data = data["data"]
        assert len(market_data) > 0

@patch('api.services.market_data_service.yf.download')
def test_fetch_market_data_single_symbol(mock_download, client):
    """Test the /api/v1/fetch_market_data endpoint with a single symbol."""
    # For single symbols, yfinance returns a different structure
    # Create test data for one symbol
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    single_symbol_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, size=len(dates)),
        'High': np.random.uniform(110, 120, size=len(dates)),
        'Low': np.random.uniform(90, 100, size=len(dates)),
        'Close': np.random.uniform(100, 110, size=len(dates)),
        'Adj Close': np.random.uniform(100, 110, size=len(dates)),
        'Volume': np.random.randint(1000000, 5000000, size=len(dates))
    }, index=dates)
    
    # Configure mock
    mock_download.return_value = single_symbol_data
    
    # Input with single symbol
    input_data = {
        "symbols": ["^DJI"],
        "start_date": "2023-01-01",
        "end_date": "2023-01-10",
        "interval": "1d"
    }
    
    # Call the API endpoint
    response = client.post("/api/v1/fetch_market_data", json=input_data)
    
    # Check the response
    assert response.status_code == 200, f"Response: {response.content}"
    data = response.json()
    
    # Verify structure
    assert "data" in data
    market_data = data["data"]
    assert len(market_data) > 0
    
    first_date = list(market_data.keys())[0]
    assert "^DJI" in market_data[first_date]
    