#!/usr/bin/env python3
# timeseries-pipeline/utilities/interpretation.py

def interpret_stationarity_test(adf_results, p_value_threshold=0.05):
    """Convert ADF test results into human-readable interpretations."""
    interpretations = {}
    
    for series_name, result in adf_results.items():
        adf_stat = result["ADF Statistic"]
        p_value = result["p-value"]
        
        if p_value < p_value_threshold:
            interpretation = (
                f"The {series_name} series is stationary (p-value: {p_value:.4f}). "
                f"This means the statistical properties like mean and variance "
                f"remain constant over time, making it suitable for time series modeling."
            )
        else:
            interpretation = (
                f"The {series_name} series is non-stationary (p-value: {p_value:.4f}). "
                f"This indicates the statistical properties change over time. "
                f"Consider differencing or transformation before modeling."
            )
            
        interpretations[series_name] = interpretation
        
    return interpretations

def interpret_arima_results(arima_fit, forecasts):
    """Convert ARIMA model results into human-readable interpretations."""
    # Implementation for ARIMA interpretations
    
def interpret_garch_results(garch_fit, forecasts):
    """Convert GARCH model results into human-readable interpretations."""
    # Implementation for GARCH interpretations


