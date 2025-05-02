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

from typing import List, Tuple, Optional, Union

def run_arima_step(df_stationary: pd.DataFrame, p: int, d: int, q: int,
                  forecast_steps: int) -> Tuple[str, List[float], pd.DataFrame]:
    """Run ARIMA model on stationary time series data."""
    try:
        # Ensure Date is set as index before passing to the library
        if 'Date' in df_stationary.columns:
            df_stationary = df_stationary.set_index('Date')
            
        # Run ARIMA models with the explicit parameters
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=df_stationary,
            p=p,
            d=d,
            q=q,
            forecast_steps=forecast_steps
        )
        
        # Extract ARIMA residuals for GARCH modeling
        arima_residuals = pd.DataFrame(index=df_stationary.index)
        for column in df_stationary.columns:
            arima_residuals[column] = arima_fits[column].resid
        
        # Get first column for API response
        column_name = list(arima_fits.keys())[0]
        arima_summary = str(arima_fits[column_name].summary())
        
        # Process forecasts for API response
        forecast_values = arima_forecasts[column_name]
        arima_forecast_values = process_forecast_values(forecast_values)
        
        return arima_summary, arima_forecast_values, arima_residuals
        
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise Exception(f"Error running ARIMA model: {str(e)}")

def run_garch_step(df_residuals: pd.DataFrame, p: int, q: int, dist: str,
                  forecast_steps: int) -> Tuple[str, List[float], Optional[pd.DataFrame]]:
    """Run GARCH model on ARIMA residuals.
    
    BEST PRACTICE: Academic research demonstrates that GARCH models effectively capture 
    volatility clustering in financial time series. The t-distribution option is particularly 
    valuable as financial returns typically exhibit fat tails that normal distributions 
    cannot adequately model. The configurable p,q parameters allow for model customization
    while defaulting to the parsimonious GARCH(1,1) specification that research shows 
    performs well in most applications.
    """
    try:
        # Ensure Date is set as index before passing to the library
        if 'Date' in df_residuals.columns:
            df_residuals = df_residuals.set_index('Date')
            
        # Run GARCH models with explicit parameters
        garch_fits, garch_forecasts = stats_model.run_garch(
            df_stationary=df_residuals,
            p=p,
            q=q,
            dist=dist,
            forecast_steps=forecast_steps
        )

        # Extract conditional volatilities
        cond_vol = pd.DataFrame(index=df_residuals.index)
        for column in df_residuals.columns:
            cond_vol[column] = np.sqrt(garch_fits[column].conditional_volatility)
        
        # Get first column for API response
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
        raise Exception(f"Error running GARCH model: {str(e)}")

def process_forecast_values(forecast_values: Union[float, np.ndarray, pd.Series, List]) -> List[float]:
    """Convert forecast values to a consistent list format."""
    if np.isscalar(forecast_values):
        return [float(forecast_values)]
    elif isinstance(forecast_values, pd.Series):
        return [float(x) for x in forecast_values.values]
    elif isinstance(forecast_values, np.ndarray):
        return [float(x) for x in forecast_values]
    elif isinstance(forecast_values, list):
        return [float(x) for x in forecast_values]
    else:
        try:
            return [float(x) for x in forecast_values]
        except:
            l.warning(f"Could not convert forecast values of type {type(forecast_values)}")
            return []
