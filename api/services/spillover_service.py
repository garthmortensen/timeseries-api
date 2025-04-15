#!/usr/bin/env python3
# timeseries-api/api/services/spillover_service.py
"""Spillover analysis service functions.
This module contains functions for analyzing spillover effects between financial time series.
"""

import logging as l
import pandas as pd
from fastapi import HTTPException

from timeseries_compute import spillover_analyzer


def analyze_spillover_step(input_data):
    """Analyze spillover effects between time series.
    
    Args:
        input_data: Pydantic model containing input parameters
        
    Returns:
        dict: Spillover analysis results
        
    Raises:
        HTTPException: If spillover analysis fails
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
        
        # Perform spillover analysis
        result = spillover_analyzer.compute_spillover_index(
            returns_data=df,
            method=input_data.method,
            forecast_horizon=input_data.forecast_horizon,
            window_size=input_data.window_size
        )
        
        # Generate interpretation
        interpretation = interpret_spillover_results(result)
        result["interpretation"] = interpretation
        
        # Format result for API response
        response = {
            "total_spillover_index": result["total_spillover"],
            "directional_spillover": result["directional_spillover"],
            "net_spillover": result["net_spillover"],
            "pairwise_spillover": result["pairwise_spillover"],
            "interpretation": result["interpretation"]
        }
        
        return response
    
    except Exception as e:
        l.error(f"Error analyzing spillover: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Spillover analysis failed: {str(e)}"
        )


def interpret_spillover_results(results):
    """Generate a human-readable interpretation of spillover results.
    
    Args:
        results (dict): Spillover analysis results
        
    Returns:
        str: Human-readable interpretation
    """
    total_spillover = results["total_spillover"]
    net_spillover = results["net_spillover"]
    
    # Identify top transmitters and receivers
    top_transmitters = sorted(net_spillover.items(), key=lambda x: x[1], reverse=True)[:2]
    top_receivers = sorted(net_spillover.items(), key=lambda x: x[1])[:2]
    
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
