#!/usr/bin/env python3
# # timeseries-pipeline/api/routers/data.py
"""Data generation and transformation API endpoints.
This module contains the API endpoints for generating synthetic time series data, scaling time series data, and testing for stationarity.
"""

import logging as l
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends

from generalized_timeseries import data_generator, data_processor
from api.models.input import DataGenerationInput, MarketDataInput, ScalingInput, StationarityTestInput
from api.models.response import TimeSeriesDataResponse, StationarityTestResponse
from api.services.market_data_service import fetch_market_data
from api.services.interpretations import interpret_stationarity_test

# Get the application configuration
from utilities.configurator import load_configuration

# Create router
router = APIRouter(tags=["Data Operations"])

# Dependency to get config
def get_config():
    return load_configuration("config.yml")


@router.post("/generate_data", 
             summary="Generate synthetic time series data", 
             response_model=TimeSeriesDataResponse)
async def generate_data_endpoint(input_data: DataGenerationInput):
    """Generate synthetic time series data based on input parameters."""
    try:
        price_dict, _ = data_generator.generate_price_series(
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            anchor_prices=input_data.anchor_prices,
        )
        
        return_data = {"data": price_dict}
        l.info(f"generate_data() returning {len(return_data['data'])} data points")
        return return_data
    
    except Exception as e:
        l.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch_market_data", 
          summary="Fetch real market data from external sources", 
          response_model=TimeSeriesDataResponse)
async def fetch_market_data_endpoint(input_data: MarketDataInput):
    """Fetch real market data from external sources like Yahoo Finance."""
    try:
        data_dict = fetch_market_data(
            symbols=input_data.symbols,
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            interval=input_data.interval
        )
        
        return {"data": data_dict}
    except Exception as e:
        l.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scale_data", 
          summary="Scale time series data", 
          response_model=TimeSeriesDataResponse)
async def scale_data_endpoint(input_data: ScalingInput):
    """Scale time series data using specified method."""
    try:
        df = pd.DataFrame(input_data.data)
        if 'date' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'date' column in input data")
            
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if df['price'].isnull().any():
            raise HTTPException(
                status_code=400, 
                detail="Invalid data: 'price' column contains non-numeric values."
            )
        
        df_scaled = data_processor.scale_data(df=df, method=input_data.method)
        
        # Convert DataFrame to dictionary with string keys that match the input structure
        data_dict = {}
        for i, (idx, row) in enumerate(df_scaled.iterrows()):
            # Use integer keys to match the test assertion
            data_dict[str(i)] = row.to_dict()
            
        return_data = {"data": data_dict}
        l.info(f"scale_data() returning {len(return_data['data'])} data points")
        return return_data

    except Exception as e:
        l.error(f"Error scaling data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test_stationarity", 
          summary="Test for stationarity", 
          response_model=StationarityTestResponse)
async def test_stationarity_endpoint(input_data: StationarityTestInput, config=Depends(get_config)):
    """Test stationarity of time series data."""
    try:
        # Process data and run tests
        df = pd.DataFrame(input_data.data)
        method = config.data_processor_stationarity_test_method
        adf_results = data_processor.test_stationarity(df=df, method=method)
        
        # Get first column results (we need a single set of results for the response model)
        column = list(adf_results.keys())[0]
        result = adf_results[column]
        
        # Create a default critical values dictionary if missing
        if "Critical Values" not in result:
            critical_values = {
                "1%": -3.75,  # Default values based on typical ADF test
                "5%": -3.0,
                "10%": -2.63
            }
        else:
            critical_values = result["Critical Values"]
            
        # Ensure critical values is a dict with string keys
        if not isinstance(critical_values, dict):
            critical_values = {
                "1%": -3.75,
                "5%": -3.0, 
                "10%": -2.63
            }
        
        # Add interpretation
        interpretation_dict = interpret_stationarity_test(
            adf_results, 
            p_value_threshold=config.data_processor_stationarity_test_p_value_threshold
        )
        
        # Build a response that matches the StationarityTestResponse model
        response = {
            "adf_statistic": float(result["ADF Statistic"]),
            "p_value": float(result["p-value"]),
            "critical_values": critical_values,
            "is_stationary": float(result["p-value"]) < config.data_processor_stationarity_test_p_value_threshold,
            "interpretation": interpretation_dict.get(column, "No interpretation available")
        }
        
        l.info(f"test_stationarity() returning results with interpretation")
        return response

    except Exception as e:
        l.error(f"Error testing stationarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))