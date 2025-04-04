#!/usr/bin/env python3
# timeseries-pipeline/api/routers/pipeline.py
"""End-to-end pipeline API endpoint.
This module defines the FastAPI router for the end-to-end pipeline.
"""

import logging as l
import time
from fastapi import APIRouter, HTTPException

from api.models.input import PipelineInput
from api.models.response import PipelineResponse
from api.services.data_service import (
    generate_data_step,
    fill_missing_data_step,
    scale_data_step,
    stationarize_data_step,
    test_stationarity_step
)
from api.services.models_service import run_arima_step, run_garch_step

# Get the application configuration
from utilities.configurator import load_configuration
config = load_configuration("config.yml")

router = APIRouter(tags=["Pipeline"])


@router.post("/run_pipeline", 
          summary="Execute the entire pipeline",
          response_model=PipelineResponse)
async def run_pipeline_endpoint(pipeline_input: PipelineInput):
    """
    Execute the complete time series analysis pipeline.
    
    This endpoint:
    1. Generates synthetic time series data
    2. Fills any missing values
    3. Scales the data
    4. Makes the data stationary
    5. Tests for stationarity
    6. Runs ARIMA and GARCH models
    7. Returns model results and forecasts
    """
    t1 = time.perf_counter()

    try:
        # Enable models for pipeline run
        config.stats_model_ARIMA_enabled = True
        config.stats_model_GARCH_enabled = True
        
        # Execute pipeline steps sequentially
        df = generate_data_step(pipeline_input, config)
        df_filled = fill_missing_data_step(df, config)
        df_scaled = scale_data_step(df_filled, config)
        df_stationary = stationarize_data_step(df_scaled, config)
        stationarity_results = test_stationarity_step(df_stationary, config)
        
        # Run models
        arima_summary, arima_forecast = run_arima_step(df_stationary, config)
        garch_summary, garch_forecast = run_garch_step(df_stationary, config)

        # Record execution time
        execution_time = time.perf_counter() - t1
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        l.info(
            f"Pipeline execution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        )

        # Return results
        return {
            "stationarity_results": stationarity_results,
            "arima_summary": arima_summary,
            "arima_forecast": arima_forecast,
            "garch_summary": garch_summary,
            "garch_forecast": garch_forecast,
        }
    except Exception as e:
        l.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Pipeline failed: {str(e)}"
        )
    