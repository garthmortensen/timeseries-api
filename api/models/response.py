#!/usr/bin/env python3
# timeseries-api/api/models/response.py
"""Response Pydantic models for API response validation.
This module contains Pydantic models for validating API response data.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class TimeSeriesDataResponse(BaseModel):
    """Response model for time series data endpoints."""
    data: List[Dict[str, Any]] = Field(
        ..., 
        description="Time series data as a list of records with date and symbol values"
    )


class StationarityTestResponse(BaseModel):
    """Response model for stationarity test results."""
    all_symbols_stationarity: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Stationarity test results for each symbol"
    )
    
    series_stats: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Comprehensive statistics for each series (mean, std, skew, kurtosis)"
    )


class ARIMAModelResponse(BaseModel):
    """Response model for ARIMA model results."""
    fitted_model: str = Field(..., description="Summary of the fitted ARIMA model")
    parameters: Dict[str, float] = Field(..., description="Model parameters")
    p_values: Dict[str, float] = Field(..., description="P-values for model parameters")
    forecast: List[float] = Field(..., description="Forecasted values")


class GARCHModelResponse(BaseModel):
    """Response model for GARCH model results."""
    fitted_model: str = Field(..., description="Summary of the fitted GARCH model")
    forecast: List[float] = Field(..., description="Forecasted volatility values")


class SpilloverResponse(BaseModel):
    """Response model for spillover analysis."""
    total_spillover_index: float = Field(
        ..., 
        description="Overall system-wide spillover"
    )
    directional_spillover: Dict[str, Any] = Field(
        ..., 
        description="Spillover from each variable to others and from others to each variable"
    )
    net_spillover: Dict[str, float] = Field(
        ..., 
        description="Net spillover (directional to others minus directional from others)"
    )
    pairwise_spillover: Dict[str, Any] = Field(
        ..., 
        description="Pairwise spillover between each pair of variables"
    )
    interpretation: str = Field(
        ..., 
        description="Human-readable interpretation of spillover results"
    )

class PipelineResponse(BaseModel):
    """Response model for the complete pipeline."""
    original_data: List[Dict[str, Any]] = Field(
        ..., 
        description="The original time series data (synthetic or fetched)"
    )
    returns_data: List[Dict[str, Any]] = Field(
        ..., 
        description="Log returns data converted from original prices"
    )
    scaled_data: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Scaled time series data (included if data is not stationary)"
    )
    pre_garch_data: List[Dict[str, Any]] = Field(
        ..., 
        description="Data before GARCH processing (ARIMA residuals)"
    )
    post_garch_data: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Data after GARCH processing (conditional volatilities)"
    )
    stationarity_results: StationarityTestResponse = Field(
        ..., 
        description="Results of stationarity tests"
    )
    
    series_stats: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Comprehensive statistics for each series (mean, std, skew, kurtosis)"
    )
    arima_results: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="ARIMA results for all symbols"
    )
    garch_results: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="GARCH results for all symbols"
    )
    spillover_results: Optional[SpilloverResponse] = Field(
        None,
        description="Results of spillover analysis (if enabled)"
    )
    
    granger_causality_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Results of Granger causality analysis (if enabled)"
    )