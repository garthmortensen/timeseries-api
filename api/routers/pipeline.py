#!/usr/bin/env python3
# timeseries-api/api/routers/pipeline.py
"""End-to-end pipeline API endpoint.
"""

import logging as l
import time
from fastapi import APIRouter, HTTPException, status

from api.models.input import PipelineInput
from api.models.response import PipelineResponse
from api.services.data_service import (
    generate_data_step,
    fill_missing_data_step,
    convert_to_returns_step,
    scale_for_garch_step,
    test_stationarity_step
)
from api.services.models_service import run_arima_step, run_garch_step

# Get the application configuration
from utilities.configurator import load_configuration
config = load_configuration("config.yml")

router = APIRouter(tags=["Pipeline"])

@router.post("/run_pipeline", 
          summary="Execute the complete time series analysis pipeline",
          description="""
          Run the end-to-end time series analysis pipeline with a single API call.
          
          This endpoint performs a complete workflow:
          1. Generate synthetic data or fetch real market data
          2. Convert prices to returns
          3. Test for stationarity
          4. Scale data for GARCH modeling
          5. Fit ARIMA models for conditional mean
          6. Extract ARIMA residuals
          7. Fit GARCH models for volatility forecasting
          8. Return all results including forecasts
          
          All parameters have sensible defaults defined in the configuration.
          """,
          response_model=PipelineResponse)
async def run_pipeline_endpoint(pipeline_input: PipelineInput):
    """
    Execute the complete time series analysis pipeline.
    
    This endpoint runs a full econometric modeling workflow that:
    
    1. Sources data (either synthetically generated or from external market sources)
    2. Converts price data to log returns
    3. Tests for stationarity in the returns
    4. Scales the returns for GARCH modeling
    5. Fits an ARIMA model for return forecasting
    6. Extracts ARIMA residuals as input for GARCH
    7. Fits a GARCH model for volatility forecasting
    
    The pipeline offers significant customization through its parameters:
    - Data source selection (synthetic vs. real market data)
    - Data processing configuration (scaling method, filling strategy)
    - Model hyperparameters (ARIMA and GARCH orders)
    
    Returns a comprehensive result including:
    - Stationarity test results with interpretation
    - ARIMA model summary and forecasts
    - GARCH model summary and volatility forecasts
    """
    t1 = time.perf_counter()

    try:
        # Override configuration with input parameters
        config.source_actual_or_synthetic_data = pipeline_input.source_actual_or_synthetic_data
        config.data_start_date = pipeline_input.data_start_date
        config.data_end_date = pipeline_input.data_end_date
        config.symbols = pipeline_input.symbols or config.symbols
        config.synthetic_anchor_prices = pipeline_input.synthetic_anchor_prices or config.synthetic_anchor_prices
        config.synthetic_random_seed = pipeline_input.synthetic_random_seed or config.synthetic_random_seed
        config.data_processor_scaling_method = pipeline_input.scaling_method

        # Update ARIMA and GARCH model parameters
        config.stats_model_ARIMA_fit_p = pipeline_input.arima_params.get('p', config.stats_model_ARIMA_fit_p)
        config.stats_model_ARIMA_fit_d = pipeline_input.arima_params.get('d', config.stats_model_ARIMA_fit_d)
        config.stats_model_ARIMA_fit_q = pipeline_input.arima_params.get('q', config.stats_model_ARIMA_fit_q)
        
        config.stats_model_GARCH_fit_p = pipeline_input.garch_params.get('p', config.stats_model_GARCH_fit_p)
        config.stats_model_GARCH_fit_q = pipeline_input.garch_params.get('q', config.stats_model_GARCH_fit_q)
        config.stats_model_GARCH_fit_dist = pipeline_input.garch_params.get('dist', config.stats_model_GARCH_fit_dist)

        # Enable models for pipeline run
        config.stats_model_ARIMA_enabled = True
        config.stats_model_GARCH_enabled = True
        
        # Execute pipeline steps with updated workflow
        df_prices = generate_data_step(pipeline_input, config)
        # New workflow: convert prices to returns before further processing
        df_returns = convert_to_returns_step(df_prices, config)
        stationarity_results = test_stationarity_step(df_returns, config)
        df_scaled = scale_for_garch_step(df_returns, config)
        
        # Run ARIMA models and get residuals for GARCH input
        arima_summary, arima_forecast, arima_residuals = run_arima_step(df_scaled, config)
        
        # Run GARCH models on ARIMA residuals
        garch_summary, garch_forecast, cond_vol = run_garch_step(arima_residuals, config)

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