#!/usr/bin/env python3
# timeseries-pipeline/api/services/interpretations.py
"""
Interpretation module for statistical test results.
Contains functions to create human-readable interpretations of statistical test results.
"""

import logging as l
from typing import Dict, Any


def interpret_stationarity_test(adf_results: Dict[str, Dict[str, float]], 
                               p_value_threshold: float = 0.05) -> Dict[str, str]:
    """
    Interpret Augmented Dickey-Fuller test results for stationarity.
    
    Args:
        adf_results (Dict[str, Dict[str, float]]): Dictionary of ADF test results
        p_value_threshold (float, optional): P-value threshold for significance. Defaults to 0.05.
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for each series
    """
    interpretations = {}
    
    for series_name, result in adf_results.items():
        try:
            adf_stat = result["ADF Statistic"]
            p_value = result["p-value"]
            
            if p_value < p_value_threshold:
                interpretation = (
                    f"The {series_name} series is stationary (p-value: {p_value:.4f}). "
                    f"This means the statistical properties like mean and variance "
                    f"remain constant over time, making it suitable for time series modeling. "
                    f"The ADF test statistic of {adf_stat:.4f} is below the critical threshold, "
                    f"allowing us to reject the null hypothesis of non-stationarity."
                )
            else:
                interpretation = (
                    f"The {series_name} series is non-stationary (p-value: {p_value:.4f}). "
                    f"This indicates the statistical properties change over time. "
                    f"The ADF test statistic of {adf_stat:.4f} is not low enough to reject "
                    f"the null hypothesis of non-stationarity. Consider differencing or "
                    f"transformation before modeling to achieve stationarity."
                )
                
            interpretations[series_name] = interpretation
        except KeyError as e:
            l.warning(f"Missing key in ADF results for {series_name}: {e}")
            interpretations[series_name] = f"Unable to interpret results for {series_name} due to missing data."
        except Exception as e:
            l.error(f"Error interpreting stationarity for {series_name}: {e}")
            interpretations[series_name] = f"Error interpreting results for {series_name}."
            
    return interpretations


def interpret_arima_results(model_summary: str, forecast: list) -> str:
    """
    Create a human-readable interpretation of ARIMA model results.
    
    Args:
        model_summary (str): Summary of the fitted ARIMA model
        forecast (list): List of forecasted values
        
    Returns:
        str: Human-readable interpretation of the ARIMA model results
    """
    try:
        # Extract simple trend from forecast
        if len(forecast) > 1:
            if forecast[-1] > forecast[0]:
                trend = "an increasing"
            elif forecast[-1] < forecast[0]:
                trend = "a decreasing"
            else:
                trend = "a stable"
        else:
            trend = "an unknown"
            
        interpretation = (
            f"The ARIMA model has been fitted successfully. "
            f"The forecast shows {trend} trend over the forecast horizon. "
            f"The first forecasted value is {forecast[0]:.4f} and the last is {forecast[-1]:.4f}."
        )
        
        return interpretation
    except Exception as e:
        l.error(f"Error interpreting ARIMA results: {e}")
        return "Unable to provide a detailed interpretation of the ARIMA model."


def interpret_garch_results(model_summary: str, forecast: list) -> str:
    """
    Create a human-readable interpretation of GARCH model results.
    
    Args:
        model_summary (str): Summary of the fitted GARCH model
        forecast (list): List of forecasted volatility values
        
    Returns:
        str: Human-readable interpretation of the GARCH model results
    """
    try:
        # Extract simple trend from forecast
        if len(forecast) > 1:
            if forecast[-1] > forecast[0]:
                trend = "an increasing"
                implication = "suggesting growing market uncertainty"
            elif forecast[-1] < forecast[0]:
                trend = "a decreasing"
                implication = "suggesting decreasing market uncertainty"
            else:
                trend = "a stable"
                implication = "suggesting stable market conditions"
        else:
            trend = "an unknown"
            implication = ""
            
        interpretation = (
            f"The GARCH model has been fitted successfully. "
            f"The volatility forecast shows {trend} trend {implication}. "
            f"The first forecasted volatility is {forecast[0]:.6f} and the last is {forecast[-1]:.6f}."
        )
        
        return interpretation
    except Exception as e:
        l.error(f"Error interpreting GARCH results: {e}")
        return "Unable to provide a detailed interpretation of the GARCH model."
