#!/usr/bin/env python3
# timeseries-api/api/routers/spillover.py
"""Spillover analysis API endpoints.
This module contains API endpoints for spillover analysis between multiple time series.
"""

import logging as l
from fastapi import APIRouter, HTTPException, Depends, Request

# Import rate limiting
from api.middleware import limiter, HEAVY_COMPUTATION_PER_MINUTE, HEAVY_COMPUTATION_PER_HOUR

from api.models.input import SpilloverInput
from api.models.response import SpilloverResponse
from api.services.spillover_service import analyze_spillover_step

# Get the application configuration
from utilities.configurator import load_configuration

# Create router
router = APIRouter(tags=["Spillover Analysis"])

# Dependency to get config
def get_config():
    return load_configuration("config.yml")


@router.post(
    "/analyze_spillover",
    summary="Analyze spillover effects between time series",
    response_model=SpilloverResponse,
    responses={
        200: {
            "description": "Successfully completed spillover analysis",
        },
        429: {
            "description": "Rate limit exceeded - too many requests",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Rate limit exceeded",
                        "message": "You have exceeded the rate limit for this API. This is a free service, please use it responsibly.",
                        "retry_after": 3600,
                        "limits": {
                            "heavy_computation": "5/minute, 20/hour"
                        }
                    }
                }
            }
        }
    }
)
@limiter.limit(HEAVY_COMPUTATION_PER_MINUTE)
@limiter.limit(HEAVY_COMPUTATION_PER_HOUR)
async def analyze_spillover_endpoint(request: Request, input_data: SpilloverInput):
    """
    Analyze spillover effects between multiple time series.
    
    This endpoint examines how shocks in one time series propagate to others over time,
    which is crucial for understanding risk transmission in financial markets.
    
    The analysis provides metrics on directional and net spillovers between series.
    """
    try:
        result = analyze_spillover_step(input_data)
        l.info(f"analyze_spillover() returning complete analysis")
        return result
    
    except Exception as e:
        l.error(f"Error analyzing spillover: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rolling_spillover",
    summary="Compute rolling window spillover analysis",
    response_model=SpilloverResponse
)
async def rolling_spillover_endpoint(input_data: SpilloverInput):
    """
    Perform rolling window spillover analysis to capture time-varying dynamics.
    
    This endpoint calculates spillover metrics across a rolling window of observations,
    allowing the detection of changes in interconnectedness over time.
    
    Returns time series of total, directional, and net spillover indices.
    """
    try:
        result = analyze_spillover_step(input_data)
        return result
    
    except Exception as e:
        l.error(f"Error in rolling spillover analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
