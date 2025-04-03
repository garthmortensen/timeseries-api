#!/usr/bin/env python3
# # timeseries-pipeline/api/routers/models.py
"""Statistical models API endpoints.
This module contains the API endpoints for running statistical models on time series data.
"""

import logging as l
import pandas as pd
from fastapi import APIRouter, HTTPException

from generalized_timeseries import stats_model
from api.models.input import ARIMAInput, GARCHInput
from api.models.response import ARIMAModelResponse, GARCHModelResponse

router = APIRouter(tags=["Statistical Models"])


@router.post("/run_arima", 
          summary="Run ARIMA model on time series", 
          response_model=ARIMAModelResponse)
async def run_arima_endpoint(input_data: ARIMAInput):
    """Run ARIMA model on time series data."""
    try:
        df = pd.DataFrame(input_data.data)
        
        model_fits, forecasts = stats_model.run_arima(
            df_stationary=df,
            p=input_data.p,
            d=input_data.d,
            q=input_data.q,
            forecast_steps=5,
        )
        
        # Handle results for API response
        column_name = list(model_fits.keys())[0]
        model_fit = model_fits[column_name]
        
        # Extract model parameters and their p-values
        params = model_fit.params.to_dict()
        pvalues = model_fit.pvalues.to_dict()
        
        # Convert forecast values to a standard list format
        forecast_values = forecasts[column_name]
        forecast_list = forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [float(value) for value in forecast_values]
        
        # Extract model summary as string
        model_summary = str(model_fit.summary())
        
        results = {
            "fitted_model": model_summary,
            "parameters": params,
            "p_values": pvalues,
            "forecast": forecast_list
        }
        l.info(f"run_arima_endpoint() returning model and forecast")
        return results
    
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise HTTPException(status_code=500, detail=f"Error running ARIMA model: {str(e)}")


@router.post("/run_garch", 
          summary="Run GARCH model on time series", 
          response_model=GARCHModelResponse)
async def run_garch_endpoint(input_data: GARCHInput):
    """Run GARCH model on time series data."""
    try:
        # Create DataFrame from input data
        df = pd.DataFrame(input_data.data)
        
        # Convert date column to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Ensure price column is numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Check for any NaN values after conversion
        if df['price'].isnull().any():
            raise HTTPException(status_code=400, detail="Invalid data: 'price' column contains non-numeric values.")
        
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        forecast_steps = 5
        model_fits, forecasts = stats_model.run_garch(
            df_stationary=numeric_df,
            p=input_data.p,
            q=input_data.q,
            dist=input_data.dist,
            forecast_steps=forecast_steps
        )
        
        # Extract the summary from the first column's model
        column_name = list(model_fits.keys())[0]
        model_summary = str(model_fits[column_name].summary())
        
        # Get forecast for the first column
        forecast_values = forecasts[column_name]
        
        # Convert forecast values to a standard list format
        if hasattr(forecast_values, 'tolist'):
            forecast_list = forecast_values.tolist()
        else:
            forecast_list = [float(value) for value in forecast_values]
        
        results = {
            "fitted_model": model_summary,
            "forecast": forecast_list
        }
        l.info(f"run_garch_endpoint() returning model and forecast")
        return results

    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(status_code=500, detail=str(e))