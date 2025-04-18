#!/usr/bin/env python3
# timeseries-api/api/routers/data.py
"""Data generation and transformation API endpoints.
This module contains the API endpoints for generating synthetic time series data, scaling time series data, and testing for stationarity.
"""

import logging as l
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, status

from timeseries_compute import data_generator, data_processor
from timeseries_compute import data_generator, data_processor
from api.models.input import DataGenerationInput, MarketDataInput, ScalingInput, StationarityTestInput
from api.models.response import TimeSeriesDataResponse, StationarityTestResponse
from api.services.market_data_service import fetch_market_data
from api.services.interpretations import interpret_stationarity_test
from api.services.data_service import (
    generate_data_step,
    fill_missing_data_step,
    scale_data_step,
    stationarize_data_step,
    test_stationarity_step,
    convert_to_returns_step,
    scale_for_garch_step
)

# Get the application configuration
from utilities.configurator import load_configuration

# Create router
router = APIRouter(tags=["Data Operations"])

# Dependency to get config
def get_config():
    return load_configuration("config.yml")


@router.post("/generate_data", 
             summary="Generate synthetic time series data", 
             response_model=TimeSeriesDataResponse,
             responses={
                 200: {
                     "description": "Successfully generated time series data",
                     "content": {
                         "application/json": {
                             "example": {
                                 "data": {
                                     "2023-01-01": {"GME": 150.0, "BYND": 200.0},
                                     "2023-01-02": {"GME": 152.3, "BYND": 198.7}
                                 }
                             }
                         }
                     }
                 },
                 400: {
                     "description": "Bad Request - Invalid date format or other input parameters"
                 },
                 500: {
                     "description": "Internal Server Error - Failed to generate time series data"
                 }
             })
async def generate_data_endpoint(input_data: DataGenerationInput):
    """
    Generate synthetic time series data based on input parameters.
    
    This endpoint creates synthetic price series for multiple symbols over a specified date range.
    Each symbol starts from its anchor price and follows a random walk with drift.
    
    The response provides a dictionary of dates, with each date containing prices for all symbols.
    """
    try:
        price_dict, price_df = data_generator.generate_price_series(
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            anchor_prices=input_data.anchor_prices,
        )
        
        # Convert from ticker-based to date-based structure required by the API
        date_based_dict = {}
        for date, row in price_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            date_based_dict[date_str] = row.to_dict()
        
        return_data = {"data": date_based_dict}
        l.info(f"generate_data() returning {len(return_data['data'])} data points")
        return return_data
    
    except Exception as e:
        l.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch_market_data", 
          summary="Fetch real market data from external sources", 
          response_model=TimeSeriesDataResponse,
          responses={
              200: {
                  "description": "Successfully fetched market data",
                  "content": {
                      "application/json": {
                          "example": {
                              "data": {
                                  "2023-01-01": {"BYND": 150.0, "GME": 200.0},
                                  "2023-01-02": {"BYND": 152.3, "GME": 198.7}
                              }
                          }
                      }
                  }
              },
              400: {
                  "description": "Bad Request - Invalid symbols or date range"
              },
              500: {
                  "description": "Internal Server Error - Failed to fetch data from external source"
              }
          })
async def fetch_market_data_endpoint(input_data: MarketDataInput):
    """
    Fetch real market data from external sources like Yahoo Finance.
    
    This endpoint retrieves historical price data for specified symbols over a date range.
    Data is obtained from Yahoo Finance via the yfinance library.
    
    The interval parameter controls the frequency of the data (daily, weekly, monthly).
    """
    try:
        data_dict, _ = fetch_market_data(
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
          response_model=TimeSeriesDataResponse,
          responses={
              200: {
                  "description": "Successfully scaled time series data",
                  "content": {
                      "application/json": {
                          "example": {
                              "data": {
                                  "0": {"date": "2023-01-01", "price": 0.0},
                                  "1": {"date": "2023-01-02", "price": 0.5},
                                  "2": {"date": "2023-01-03", "price": -0.3}
                              }
                          }
                      }
                  }
              },
              400: {
                  "description": "Bad Request - Invalid data format or missing required columns"
              },
              500: {
                  "description": "Internal Server Error - Failed to scale data"
              }
          })
async def scale_data_endpoint(input_data: ScalingInput):
    """
    Scale time series data using the specified method.
    
    This endpoint takes raw price data and applies a scaling transformation.
    Supported methods include:
    - standardize: Transforms data to have mean=0 and standard deviation=1
    - minmax: Scales data to a range between 0 and 1
    
    The input data must contain 'date' and 'price' columns.
    """
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
          response_model=StationarityTestResponse,
          responses={
              200: {
                  "description": "Successfully tested time series for stationarity",
                  "content": {
                      "application/json": {
                          "example": {
                              "adf_statistic": -3.45,
                              "p_value": 0.032,
                              "critical_values": {"1%": -3.75, "5%": -3.0, "10%": -2.63},
                              "is_stationary": True,
                              "interpretation": "The series is stationary (p-value: 0.0320). This means the statistical properties like mean and variance remain constant over time, making it suitable for time series modeling."
                          }
                      }
                  }
              },
              400: {
                  "description": "Bad Request - Invalid data format or insufficient data points"
              },
              500: {
                  "description": "Internal Server Error - Failed to run stationarity test"
              }
          })
async def test_stationarity_endpoint(input_data: StationarityTestInput):
    """
    Test time series data for stationarity using the Augmented Dickey-Fuller test.
    
    Stationarity is a key property for time series analysis, indicating that statistical
    properties like mean, variance, and autocorrelation are constant over time.
    
    The test returns:
    - ADF statistic: More negative values suggest stationarity
    - p-value: Smaller values suggest stationarity
    - Critical values: Threshold values at different significance levels
    - is_stationary: Boolean indication based on p-value threshold
    - interpretation: Human-readable explanation of the results
    
    A p-value less than the threshold (default 0.05) indicates stationarity.
    """
    try:
        # Process data as a DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Use hardcoded default values instead of config
        p_value_threshold = 0.05  # Standard statistical significance level
        test_method = "ADF"  # Augmented Dickey-Fuller test
        
        # Delegate to the service function with explicit parameters
        response = test_stationarity_step(
            df=df, 
            test_method=test_method, 
            p_value_threshold=p_value_threshold
        )
        
        l.info(f"test_stationarity() returning results with interpretation")
        return response

    except Exception as e:
        l.error(f"Error testing stationarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/price_to_returns", 
          summary="Convert price data to log returns", 
          response_model=TimeSeriesDataResponse)
async def price_to_returns_endpoint(input_data: dict):
    """
    Convert price time series data to log returns.
    
    This endpoint takes price data and calculates the log returns,
    which are typically more suitable for statistical modeling.
    
    Returns data represents the percentage change between consecutive price observations.
    """
    try:
        df = pd.DataFrame(input_data["data"])
        
        # Ensure proper date format and set as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Convert to returns
        returns_df = data_processor.price_to_returns(df)
        
        # Convert DataFrame to dictionary for API response
        data_dict = {}
        for i, (idx, row) in enumerate(returns_df.iterrows()):
            data_dict[str(i)] = row.to_dict()
            
        return_data = {"data": data_dict}
        return return_data
    
    except Exception as e:
        l.error(f"Error converting prices to returns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/scale_for_garch", 
          summary="Scale data for GARCH modeling", 
          response_model=TimeSeriesDataResponse)
async def scale_for_garch_endpoint(input_data: dict):
    """
    Scale time series data specifically for GARCH modeling.
    
    This endpoint takes return data and applies the appropriate scaling
    for GARCH volatility modeling.
    """
    try:
        df = pd.DataFrame(input_data["data"])
        
        # Ensure proper date format and set as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Scale for GARCH
        scaled_df = data_processor.scale_for_garch(df)
        
        # Convert DataFrame to dictionary for API response
        data_dict = {}
        for i, (idx, row) in enumerate(scaled_df.iterrows()):
            data_dict[str(i)] = row.to_dict()
            
        return_data = {"data": data_dict}
        return return_data
    
    except Exception as e:
        l.error(f"Error scaling data for GARCH: {e}")
        raise HTTPException(status_code=500, detail=str(e))