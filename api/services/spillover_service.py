#!/usr/bin/env python3
# timeseries-api/api/services/spillover_service.py
"""Spillover analysis service functions.
This module contains functions for analyzing spillover effects between financial time series.
"""

import logging as l
import pandas as pd
from fastapi import HTTPException

from timeseries_compute import spillover_processor

def analyze_spillover_step(input_data):
    """Analyze spillover effects between time series."""
    try:
        # Convert input data to DataFrame if needed
        if isinstance(input_data.data, list):
            df = pd.DataFrame(input_data.data)
            
            # If date/time column exists, set as index
            date_cols = [col for col in df.columns if col.lower() in ('date', 'time', 'datetime')]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                df.set_index(date_cols[0], inplace=True)
        else:
            df = input_data.data
        
        # Ensure we have proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            l.warning("Converting index to DatetimeIndex")
            df.index = pd.to_datetime(df.index)

        # Add this line to select only numeric columns for analysis
        df_numeric = df.select_dtypes(include=['number'])
        
        # Calculate a safe maximum lag based on data size
        # Requested lag from input or config
        requested_max_lag = input_data.forecast_horizon if hasattr(input_data, 'forecast_horizon') else 5
        
        # For Granger causality test, a good rule of thumb is max_lag ≤ n/3
        # where n is the number of observations
        safe_max_lag = min(requested_max_lag, len(df_numeric) // 3)
        
        # Ensure max_lag is at least 1
        max_lag = max(1, safe_max_lag)
        
        if max_lag < requested_max_lag:
            l.warning(f"Adjusted max_lag from {requested_max_lag} to {max_lag} due to insufficient observations")
        
        # Call run_spillover_analysis with adjusted parameters
        result = spillover_processor.run_spillover_analysis(
            df_stationary=df_numeric,
            max_lag=max_lag
        )
        
        # Generate interpretation safely
        try:
            interpretation = interpret_spillover_results(result)
            result["interpretation"] = interpretation
        except Exception as interp_error:
            l.warning(f"Could not generate interpretation: {interp_error}")
            result["interpretation"] = "Spillover analysis complete, but detailed interpretation unavailable."
        
        # Format result with defaults for missing keys
        response = {
            "total_spillover_index": result.get("total_spillover", 0.0),
            "directional_spillover": result.get("spillover_analysis", {}).get("granger_causality", {}),
            "net_spillover": result.get("net_spillover", {}),
            "pairwise_spillover": result.get("spillover_analysis", {}).get("shock_spillover", {}),
            "interpretation": result.get("interpretation", "Spillover analysis complete.")
        }
        
        return response
    
    except Exception as e:
        l.error(f"Error analyzing spillover: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Spillover analysis failed: {str(e)}"
        )


def interpret_spillover_results(results):
    """Generate a human-readable interpretation of spillover results."""
    # Use .get() method with default values to avoid KeyError
    total_spillover = results.get("total_spillover", 0.0)
    net_spillover = results.get("net_spillover", {})
    
    # Identify top transmitters and receivers
    top_transmitters = sorted(net_spillover.items(), key=lambda x: x[1], reverse=True)[:2] if net_spillover else []
    top_receivers = sorted(net_spillover.items(), key=lambda x: x[1])[:2] if net_spillover else []
    
    # Create interpretation
    interpretation = (
        f"The system shows a total spillover index of {total_spillover:.2f}%, "
        f"indicating the overall level of interconnectedness between the variables. "
    )
    
    if top_transmitters:
        interpretation += (
            f"The main transmitters of shocks are {top_transmitters[0][0]} "
            f"(net: {top_transmitters[0][1]:.2f}%)"
        )
        if len(top_transmitters) > 1:
            interpretation += f" and {top_transmitters[1][0]} (net: {top_transmitters[1][1]:.2f}%)"
        interpretation += ". "
    
    if top_receivers:
        interpretation += (
            f"The main receivers of shocks are {top_receivers[0][0]} "
            f"(net: {top_receivers[0][1]:.2f}%)"
        )
        if len(top_receivers) > 1:
            interpretation += f" and {top_receivers[1][0]} (net: {top_receivers[1][1]:.2f}%)"
        interpretation += "."
    
    return interpretation
