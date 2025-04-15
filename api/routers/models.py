#!/usr/bin/env python3
# timeseries-api/api/routers/models.py
"""Statistical models API endpoints.
This module contains the API endpoints for running statistical models on time series data.
"""

import logging as l
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, status
from statsmodels.tsa.arima.model import ARIMA

from timeseries_compute import stats_model
from api.models.input import ARIMAInput, GARCHInput
from api.models.response import ARIMAModelResponse, GARCHModelResponse

router = APIRouter(tags=["Statistical Models"])


@router.post("/run_arima", 
          summary="Run ARIMA model on time series", 
          response_model=ARIMAModelResponse,
          responses={
              200: {
                  "description": "Successfully fitted ARIMA model",
                  "content": {
                      "application/json": {
                          "example": {
                              "fitted_model": "ARIMA(2,1,2) Results\nAIC: 123.45\nBIC: 134.56\n...",
                              "parameters": {"ar.L1": 0.5, "ar.L2": 0.3, "ma.L1": 0.2, "ma.L2": 0.1, "const": 0.02},
                              "p_values": {"ar.L1": 0.001, "ar.L2": 0.01, "ma.L1": 0.005, "ma.L2": 0.05, "const": 0.22},
                              "forecast": [101.2, 102.3, 103.5, 104.1, 105.2]
                          }
                      }
                  }
              },
              400: {
                  "description": "Bad Request - Invalid model parameters or insufficient data"
              },
              500: {
                  "description": "Internal Server Error - Model fitting failed"
              }
          })
async def run_arima_endpoint(input_data: ARIMAInput):
    """
    Run an ARIMA (AutoRegressive Integrated Moving Average) model on time series data.
    
    ARIMA is a statistical model for analyzing and forecasting time series data.
    It combines three components:
    - AR(p): AutoRegressive - uses the relationship between an observation and p lagged observations
    - I(d): Integrated - differencing to make the time series stationary
    - MA(q): Moving Average - uses the dependency between an observation and q lagged residuals
    
    Parameters:
    - p: Order of the AutoRegressive component (number of lag observations)
    - d: Order of differencing required to make the series stationary
    - q: Order of the Moving Average component (size of the moving average window)
    
    Returns:
    - fitted_model: Summary of the fitted model with diagnostics
    - parameters: Estimated coefficients for the model
    - p_values: Statistical significance of each parameter
    - forecast: Future predictions from the model
    """
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
          response_model=GARCHModelResponse,
          responses={
              200: {
                  "description": "Successfully fitted GARCH model",
                  "content": {
                      "application/json": {
                          "example": {
                              "fitted_model": "GARCH(1,1) Results\nAIC: 235.67\nBIC: 245.89\n...",
                              "forecast": [0.0025, 0.0028, 0.0030, 0.0027, 0.0026]
                          }
                      }
                  }
              },
              400: {
                  "description": "Bad Request - Invalid model parameters or insufficient data"
              },
              500: {
                  "description": "Internal Server Error - Model fitting failed"
              }
          })
async def run_garch_endpoint(input_data: GARCHInput):
    """
    Run a GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) model on time series data.
    
    GARCH models are used to estimate and forecast volatility in financial time series.
    They are particularly useful for asset returns that exhibit volatility clustering.
    
    Parameters:
    - p: ARCH order (lag volatility terms)
    - q: GARCH order (lag residual terms)
    - dist: Error distribution assumption (normal, t, skewed-t)
    
    The model captures how volatility evolves over time, accounting for:
    - Persistence of volatility (p)
    - Impact of past shocks on current volatility (q)
    
    Returns:
    - fitted_model: Summary of the fitted model with diagnostics
    - forecast: Predicted future volatility values
    """
    try:
        # Create DataFrame from input data
        df = pd.DataFrame(input_data.data)
        
        # Set up index and prepare data
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        if df['price'].isnull().any():
            raise HTTPException(status_code=400, detail="Invalid data: 'price' column contains non-numeric values.")
        
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        # Configure for service layer
        class SimpleConfig:
            def __init__(self, p, q, dist):
                self.stats_model_GARCH_enabled = True
                self.stats_model_GARCH_fit_p = p
                self.stats_model_GARCH_fit_q = q
                self.stats_model_GARCH_fit_dist = dist
                self.stats_model_GARCH_predict_steps = 5
        
        config = SimpleConfig(input_data.p, input_data.q, input_data.dist)
        
        # Delegate to service layer
        garch_summary, garch_forecast, _ = run_garch_step(numeric_df, config)
        
        results = {
            "fitted_model": garch_summary,
            "forecast": garch_forecast
        }
        
        l.info(f"run_garch_endpoint() returning model and forecast")
        return results

    except HTTPException as he:
        # Pass through HTTP exceptions
        raise he
    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
