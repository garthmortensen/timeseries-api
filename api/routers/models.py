#!/usr/bin/env python3
# # timeseries-pipeline/api/routers/models.py
"""Statistical models API endpoints.
This module contains the API endpoints for running statistical models on time series data.
"""

import logging as l
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from statsmodels.tsa.arima.model import ARIMA

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
        
        # Set proper datetime index for better ARIMA modeling
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Set frequency for time series
            if df.index.freq is None:
                df = df.asfreq('D')
        
        # Handle case where we need to fit our own ARIMA model to prevent warnings
        if len(df.columns) > 0 and 'price' in df.columns:
            column_name = 'price'
            
            # Use more robust model initialization to avoid warnings
            model = ARIMA(
                df[column_name], 
                order=(input_data.p, input_data.d, input_data.q),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Fit model using appropriate parameters for the version
            try:
                model_fit = model.fit(method='css')
            except:
                model_fit = model.fit()
                
            # Extract model parameters and their p-values
            params = {str(k): float(v) for k, v in model_fit.params.to_dict().items()}
            pvalues = {str(k): float(v) for k, v in model_fit.pvalues.to_dict().items()}
            
            # Create forecast with proper handling of different result types
            forecast_steps = 5
            forecast_values = model_fit.forecast(steps=forecast_steps)
            
            # Convert forecast to a proper list
            forecast_list = []
            if isinstance(forecast_values, pd.Series):
                forecast_list = [float(x) for x in forecast_values.values]
            elif isinstance(forecast_values, np.ndarray):
                forecast_list = [float(x) for x in forecast_values]
            elif np.isscalar(forecast_values):
                forecast_list = [float(forecast_values)]
            else:
                try:
                    forecast_list = [float(x) for x in forecast_values]
                except:
                    forecast_list = [0.0]  # Default value as fallback
            
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
            
        else:
            # Fall back to the original approach
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
            params = {str(k): float(v) for k, v in model_fit.params.to_dict().items()}
            pvalues = {str(k): float(v) for k, v in model_fit.pvalues.to_dict().items()}
            
            # Convert forecast values to a standard list format
            forecast_values = forecasts[column_name]
            forecast_list = []
            
            # Handle different types of forecast return values
            if np.isscalar(forecast_values):
                forecast_list = [float(forecast_values)]
            elif isinstance(forecast_values, pd.Series):
                forecast_list = [float(x) for x in forecast_values.values]
            elif isinstance(forecast_values, np.ndarray):
                forecast_list = [float(x) for x in forecast_values]
            elif isinstance(forecast_values, list):
                forecast_list = [float(x) for x in forecast_values]
            else:
                try:
                    forecast_list = [float(x) for x in forecast_values]
                except:
                    forecast_list = [0.0]  # Default value as fallback
            
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
        
        # Handle different types of forecast return values
        forecast_list = []
        
        # Check if forecast_values is a scalar (single value)
        if np.isscalar(forecast_values):
            forecast_list = [float(forecast_values)]
        # Check if it's a pandas Series
        elif isinstance(forecast_values, pd.Series):
            forecast_list = [float(x) for x in forecast_values.values]
        # Check if it's a numpy array
        elif isinstance(forecast_values, np.ndarray):
            forecast_list = [float(x) for x in forecast_values]
        # Check if it's already a list
        elif isinstance(forecast_values, list):
            forecast_list = [float(x) for x in forecast_values]
        # Fallback for any other type
        else:
            try:
                # Try to convert to a list if it's some other iterable
                forecast_list = [float(x) for x in forecast_values]
            except:
                # If all else fails, use a single value
                forecast_list = [0.0]  # Use a default value
                l.warning(f"Couldn't convert forecast values of type {type(forecast_values)}")
        
        results = {
            "fitted_model": model_summary,
            "forecast": forecast_list
        }
        l.info(f"run_garch_endpoint() returning model and forecast")
        return results

    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(status_code=500, detail=str(e))