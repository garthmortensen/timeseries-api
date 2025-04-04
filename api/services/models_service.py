#!/usr/bin/env python3
# timeseries-pipeline/api/services/models_service.py
"""Statistical model service functions.
This module contains functions to run statistical models on time series data.
"""

import logging as l
from fastapi import HTTPException
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from generalized_timeseries import stats_model


def run_arima_step(df_stationary, config):
    """Run ARIMA model on stationary time series data.
    
    Args:
        df_stationary (pandas.DataFrame): Stationary data frame
        config: Application configuration
        
    Returns:
        tuple: (arima_summary, arima_forecast_values)
    """
    if not config.stats_model_ARIMA_enabled:
        return "ARIMA not enabled", []
    
    try:
        # Initialize dictionaries to store models and forecasts
        arima_fits = {}
        arima_forecasts = {}
        
        # Process the first column only - for simplicity in API responses
        if len(df_stationary.columns) > 0:
            column_name = df_stationary.columns[0]
            
            # Create DataFrame with datetime index if not already
            if not isinstance(df_stationary.index, pd.DatetimeIndex):
                if 'date' in df_stationary.columns:
                    df_temp = df_stationary.copy()
                    df_temp['date'] = pd.to_datetime(df_temp['date'])
                    df_temp.set_index('date', inplace=True)
                    series = df_temp[column_name]
                else:
                    # If no date column, create a date range
                    series = pd.Series(
                        df_stationary[column_name].values,
                        index=pd.date_range(start='2000-01-01', periods=len(df_stationary), freq='D')
                    )
            else:
                series = df_stationary[column_name]
            
            # Set frequency if not already set
            if series.index.freq is None:
                series = series.asfreq('D')
                
            # Use more robust model initialization and fitting
            model = ARIMA(
                series, 
                order=(config.stats_model_ARIMA_fit_p, config.stats_model_ARIMA_fit_d, config.stats_model_ARIMA_fit_q),
                enforce_stationarity=False,  # Don't enforce stationarity
                enforce_invertibility=False  # Don't enforce invertibility
            )
            
            # Fit the model with more robust approach - remove maxiter parameter
            try:
                arima_fit = model.fit(method='css')  # Use CSS method
            except:
                # Fallback to default method
                arima_fit = model.fit()
                
            # Store the model
            arima_fits[column_name] = arima_fit
            
            # Generate forecasts
            forecast_steps = config.stats_model_ARIMA_predict_steps
            forecast = arima_fit.forecast(steps=forecast_steps)
            arima_forecasts[column_name] = forecast
        else:
            return "No columns found in data", []
        
        if not arima_fits or len(arima_fits) <= 0:
            return "No ARIMA models fitted", []
        
        # Get the first column's model summary
        column_name = list(arima_fits.keys())[0]
        arima_summary = str(arima_fits[column_name].summary())
        
        # Get forecast for the first column
        forecast_values = arima_forecasts[column_name]
        
        # Handle different types of forecast return values
        arima_forecast_values = []
        
        # Check if forecast_values is a scalar (single value)
        if np.isscalar(forecast_values):
            arima_forecast_values = [float(forecast_values)]
        # Check if it's a pandas Series
        elif isinstance(forecast_values, pd.Series):
            arima_forecast_values = [float(x) for x in forecast_values.values]
        # Check if it's a numpy array
        elif isinstance(forecast_values, np.ndarray):
            arima_forecast_values = [float(x) for x in forecast_values]
        # Check if it's already a list
        elif isinstance(forecast_values, list):
            arima_forecast_values = [float(x) for x in forecast_values]
        # Fallback for any other type
        else:
            try:
                # Try to convert to a list if it's some other iterable
                arima_forecast_values = [float(x) for x in forecast_values]
            except:
                # If all else fails, use a single empty list
                arima_forecast_values = []
                l.warning(f"Couldn't convert forecast values of type {type(forecast_values)}")
        
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
    if not config.stats_model_GARCH_enabled:
        return "GARCH not enabled", []
    
    try:
        garch_fits, garch_forecasts = stats_model.run_garch(
            df_stationary=df_stationary,
            p=config.stats_model_GARCH_fit_p,
            q=config.stats_model_GARCH_fit_q,
            dist=config.stats_model_GARCH_fit_dist,
            forecast_steps=config.stats_model_GARCH_predict_steps
        )
        
        if not garch_fits or len(garch_fits) <= 0:
            return "No GARCH models fitted", []
        
        # Get the first column's model summary
        column_name = list(garch_fits.keys())[0]
        garch_summary = str(garch_fits[column_name].summary())
        
        # Get forecast for the first column
        forecast_values = garch_forecasts[column_name]
        
        # Handle different types of forecast return values
        garch_forecast_values = []
        
        # Check if forecast_values is a scalar (single value)
        if np.isscalar(forecast_values):
            garch_forecast_values = [float(forecast_values)]
        # Check if it's a pandas Series
        elif isinstance(forecast_values, pd.Series):
            garch_forecast_values = [float(x) for x in forecast_values.values]
        # Check if it's a numpy array
        elif isinstance(forecast_values, np.ndarray):
            garch_forecast_values = [float(x) for x in forecast_values]
        # Check if it's already a list
        elif isinstance(forecast_values, list):
            garch_forecast_values = [float(x) for x in forecast_values]
        # Fallback for any other type
        else:
            try:
                # Try to convert to a list if it's some other iterable
                garch_forecast_values = [float(x) for x in forecast_values]
            except:
                # If all else fails, use a single empty list
                garch_forecast_values = []
                l.warning(f"Couldn't convert forecast values of type {type(forecast_values)}")
        
        return garch_summary, garch_forecast_values
    
    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error running GARCH model: {str(e)}"
        )