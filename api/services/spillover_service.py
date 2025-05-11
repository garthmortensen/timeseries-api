#!/usr/bin/env python3
# timeseries-api/api/services/spillover_service.py
"""Spillover analysis service functions.
This module contains functions for analyzing spillover effects between financial time series.
"""

import logging as l
import pandas as pd
import numpy as np
from fastapi import HTTPException
from typing import Dict, Any, List, Optional, Union

# Standardize imports - use consistent naming throughout
from timeseries_compute import spillover_processor as spillover

def analyze_spillover_step(input_data):
    """
    Analyze spillover effects between time series.
    
    This is the main function that processes input data, runs spillover analysis,
    and generates human-readable interpretations of the results.
    
    Args:
        input_data: SpilloverInput model containing data and parameters for analysis
    
    Returns:
        Dictionary with spillover analysis results formatted for API response
    
    Raises:
        HTTPException: If analysis fails for any reason
    """
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

        # Select only numeric columns for analysis
        df_numeric = df.select_dtypes(include=['number'])
        
        # Calculate a safe maximum lag based on data size
        requested_max_lag = input_data.forecast_horizon if hasattr(input_data, 'forecast_horizon') else 5
        
        # For Granger causality test, a good rule of thumb is max_lag ≤ n/3
        # where n is the number of observations
        safe_max_lag = min(requested_max_lag, len(df_numeric) // 3)
        
        # Ensure max_lag is at least 1
        max_lag = max(1, safe_max_lag)
        
        if max_lag < requested_max_lag:
            l.warning(f"Adjusted max_lag from {requested_max_lag} to {max_lag} due to insufficient observations")
        
        # Use the standardized function name consistently
        result = spillover.run_spillover_analysis(
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


def run_granger_causality_test(
    series1: pd.Series, 
    series2: pd.Series, 
    max_lag: int = 5, 
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Run Granger causality test between two time series.
    
    This function provides a direct interface to the test_granger_causality
    functionality for use in API endpoints and the CLI pipeline.
    
    Args:
        series1: First time series (potential cause)
        series2: Second time series (potential effect)
        max_lag: Maximum lag to test
        significance_level: P-value threshold for significance
        
    Returns:
        Dictionary with test results including causality boolean and p-values
    """
    return spillover.test_granger_causality(
        series1=series1,
        series2=series2,
        max_lag=max_lag,
        significance_level=significance_level
    )


def compute_spillover_index(
    returns_data: Union[pd.DataFrame, List[Dict[str, Any]]],
    method: str = "diebold_yilmaz",
    forecast_horizon: int = 10,
    window_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute spillover indices between variables using various methodologies.
    
    This function wraps the underlying spillover analysis functions and ensures
    consistent behavior between the API endpoints and pipeline implementations.
    
    Args:
        returns_data: DataFrame or list of dictionaries with return data
        method: Analysis method (e.g., "diebold_yilmaz")
        forecast_horizon: Forecast horizon for variance decomposition
        window_size: Window size for rolling analysis (None for full sample)
        
    Returns:
        Dictionary with spillover indices and related metrics
    """
    # Convert to DataFrame if needed
    if isinstance(returns_data, list):
        df = pd.DataFrame(returns_data)
        
        # If date/time column exists, set as index
        date_cols = [col for col in df.columns if col.lower() in ('date', 'time', 'datetime')]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
            
        # Select only numeric columns
        df = df.select_dtypes(include=['number'])
    else:
        df = returns_data
    
    # Calculate spillover using the standardized function
    result = spillover.run_spillover_analysis(
        df_stationary=df,
        max_lag=min(forecast_horizon, len(df) // 3)
    )
    
    # Extract and format the results for consistency
    formatted_result = {
        "total_spillover": result.get("total_spillover", 0.0),
        "directional_spillover": result.get("spillover_analysis", {}).get("granger_causality", {}),
        "net_spillover": result.get("net_spillover", {}),
        "pairwise_spillover": result.get("spillover_analysis", {}).get("shock_spillover", {})
    }
    
    return formatted_result


def interpret_spillover_results(results):
    """
    Generate a human-readable interpretation of spillover results.
    
    This function analyzes the spillover results and creates a comprehensive
    interpretation that explains the findings in clear, business-relevant terms.
    
    Args:
        results: Dictionary with spillover analysis results
        
    Returns:
        String with human-readable interpretation of the results
    """
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
    
    # Add interpretation of strength
    if total_spillover > 50:
        interpretation += (
            "This high level of spillover suggests strong interconnections where shocks in one market "
            "significantly affect others. Diversification benefits may be limited during periods of market stress. "
        )
    elif total_spillover > 25:
        interpretation += (
            "This moderate level of spillover indicates meaningful interconnections between markets, "
            "with some potential for shock transmission but also some diversification benefits. "
        )
    else:
        interpretation += (
            "This relatively low level of spillover suggests limited interconnections, "
            "with shocks tending to remain contained within individual markets. "
            "This environment may offer good diversification opportunities. "
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
        interpretation += ". "
    
    # Add BEKK-GARCH specific interpretation if applicable
    granger_results = results.get("spillover_analysis", {}).get("granger_causality", {})
    if granger_results:
        significant_pairs = [pair for pair, result in granger_results.items() if result.get("causality", False)]
        if significant_pairs:
            interpretation += (
                f"Significant directional spillovers were detected in {len(significant_pairs)} market pair(s). "
                "This indicates specific causal relationships where volatility in one market leads to "
                "changes in another market with a time delay. "
            )
    
    return interpretation
