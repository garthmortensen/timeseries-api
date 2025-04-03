#!/usr/bin/env python3
# timeseries-pipeline/api/services/models_service.py
"""Statistical model service functions.
Thi module contains functions to run statistical models on time series data.
"""

import logging as l
from fastapi import HTTPException
import pandas as pd
import numpy as np

from generalized_timeseries import stats_model


def run_arima_step(df_stationary, config):
    """Run ARIMA model on stationary time series data.
    
    Args:
        df_stationary (pandas.DataFrame): Stationary data frame
        config: Application configuration
        
    Returns:
        tuple: (arima_summary, arima_forecast_values)
    """
    if not config.stats_model.ARIMA.enabled:
        return "ARIMA not enabled", []
    
    try:
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=df_stationary,
            p=config.stats_model.ARIMA.parameters_fit.p,
            d=config.stats_model.ARIMA.parameters_fit.d,
            q=config.stats_model.ARIMA.parameters_fit.q,
            forecast_steps=config.stats_model.ARIMA.parameters_predict_steps
        )
        
        if not arima_fits or len(arima_fits) <= 0:
            return "No ARIMA models fitted", []
        
        # Get the first column's model summary
        column_name = list(arima_fits.keys())[0]
        arima_summary = str(arima_fits[column_name].summary())
        
        # Get forecast for the first column
        forecast_values = arima_forecasts[column_name]
        
        # Convert forecast values to a list
        if hasattr(forecast_values, 'tolist'):
            arima_forecast_values = forecast_values.tolist()
        else:
            arima_forecast_values = []
            for x in forecast_values:
                arima_forecast_values.append(float(x))
        
        return arima_summary, arima_forecast_values
    
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error running ARIMA model: {str(e)}"
        )


def run_garch_step(df_stationary, config):
    """Run GARCH model on stationary time series data.
    
    Args:
        df_stationary (pandas.DataFrame): Stationary data frame
        config: Application configuration
        
    Returns:
        tuple: (garch_summary, garch_forecast_values)
    """
    if not config.stats_model.GARCH.enabled:
        return "GARCH not enabled", []
    
    try:
        garch_fits, garch_forecasts = stats_model.run_garch(
            df_stationary=df_stationary,
            p=config.stats_model.GARCH.parameters_fit.p,
            q=config.stats_model.GARCH.parameters_fit.q,
            dist=config.stats_model.GARCH.parameters_fit.dist,
            forecast_steps=config.stats_model.GARCH.parameters_predict_steps
        )
        
        if not garch_fits or len(garch_fits) <= 0:
            return "No GARCH models fitted", []
        
        # Get the first column's model summary
        column_name = list(garch_fits.keys())[0]
        garch_summary = str(garch_fits[column_name].summary())
        
        # Get forecast for the first column
        forecast_values = garch_forecasts[column_name]
        
        # Convert forecast values to a list
        if hasattr(forecast_values, 'tolist'):
            garch_forecast_values = forecast_values.tolist()
        else:
            garch_forecast_values = []
            for x in forecast_values:
                garch_forecast_values.append(float(x))
        
        return garch_summary, garch_forecast_values
    
    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error running GARCH model: {str(e)}"
        )