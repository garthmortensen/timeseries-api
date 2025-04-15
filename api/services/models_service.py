#!/usr/bin/env python3
# timeseries-api/api/services/models_service.py
"""Statistical model service functions.
This module contains functions to run statistical models on time series data.
"""

import logging as l
from fastapi import HTTPException
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from timeseries_compute import stats_model


def run_arima_step(df_stationary, config):
    """Run ARIMA model on stationary time series data.
    
    Args:
        df_stationary (pandas.DataFrame): Stationary data frame
        config: Application configuration
        
    Returns:
        tuple: (arima_fits, arima_forecasts, arima_residuals)
    """
    if not config.stats_model_ARIMA_enabled:
        return None, None, df_stationary
    
    try:
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=df_stationary,
            p=config.stats_model_ARIMA_fit_p,
            d=config.stats_model_ARIMA_fit_d,
            q=config.stats_model_ARIMA_fit_q,
            forecast_steps=config.stats_model_ARIMA_predict_steps
        )
        
        # Extract ARIMA residuals for GARCH modeling
        arima_residuals = pd.DataFrame(index=df_stationary.index)
        for column in df_stationary.columns:
            arima_residuals[column] = arima_fits[column].resid
        
        # Generate model summary for API response
        column_name = list(arima_fits.keys())[0]
        arima_summary = str(arima_fits[column_name].summary())
        
        # Process forecasts for API response
        forecast_values = arima_forecasts[column_name]
        arima_forecast_values = process_forecast_values(forecast_values)
        
        return arima_summary, arima_forecast_values, arima_residuals
    
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error running ARIMA model: {str(e)}"
        )

def run_garch_step(df_residuals, config):
    """Run GARCH model on ARIMA residuals.
    
    Args:
        df_residuals (pandas.DataFrame): ARIMA residuals
        config: Application configuration
        
    Returns:
        tuple: (garch_summary, garch_forecast_values, conditional_volatilities)
    """
    if not config.stats_model_GARCH_enabled:
        return "GARCH not enabled", [], None
    
    try:
        garch_fits, garch_forecasts = stats_model.run_garch(
            df_stationary=df_residuals,
            p=config.stats_model_GARCH_fit_p,
            q=config.stats_model_GARCH_fit_q,
            dist=config.stats_model_GARCH_fit_dist,
            forecast_steps=config.stats_model_GARCH_predict_steps
        )
        
        # Extract conditional volatilities
        cond_vol = pd.DataFrame(index=df_residuals.index)
        for column in df_residuals.columns:
            cond_vol[column] = np.sqrt(garch_fits[column].conditional_volatility)
        
        # Generate model summary for API response
        column_name = list(garch_fits.keys())[0]
        garch_summary = str(garch_fits[column_name].summary())
        
        # Process forecasts for API response
        forecast_values = garch_forecasts[column_name]
        
        # Convert variance forecasts to volatility
        if hasattr(forecast_values, '__iter__'):
            garch_forecast_values = [float(np.sqrt(x)) for x in forecast_values]
        else:
            garch_forecast_values = [float(np.sqrt(forecast_values))]
        
        return garch_summary, garch_forecast_values, cond_vol
    
    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error running GARCH model: {str(e)}"
        )

# Utility function for processing forecast values consistently
def process_forecast_values(forecast_values):
    """Convert forecast values to a consistent list format.
    
    Args:
        forecast_values: Forecast values in various possible formats
        
    Returns:
        list: Forecast values as a list of floats
    """
    result = []
    
    # Check if forecast_values is a scalar
    if np.isscalar(forecast_values):
        result = [float(forecast_values)]
    # Check if it's a pandas Series
    elif isinstance(forecast_values, pd.Series):
        result = [float(x) for x in forecast_values.values]
    # Check if it's a numpy array
    elif isinstance(forecast_values, np.ndarray):
        result = [float(x) for x in forecast_values]
    # Check if it's already a list
    elif isinstance(forecast_values, list):
        result = [float(x) for x in forecast_values]
    # Fallback for any other type
    else:
        try:
            result = [float(x) for x in forecast_values]
        except:
            result = []
            l.warning(f"Couldn't convert forecast values of type {type(forecast_values)}")
    
    return result
