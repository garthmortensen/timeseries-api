#!/usr/bin/env python3
# timeseries-api/api/routers/pipeline.py
"""End-to-end pipeline API endpoint.
"""

import logging as l
import time
from fastapi import APIRouter, HTTPException
import datetime
import json

# database
from fastapi import Depends
from sqlalchemy.orm import Session
import json
from api.database import get_db, PipelineRun, PipelineResult

from api.models.input import PipelineInput, SpilloverInput
from api.models.response import PipelineResponse
from api.services.data_service import (
    generate_data_step,
    fill_missing_data_step,
    convert_to_returns_step,
    scale_for_garch_step,
    test_stationarity_step
)
from api.services.models_service import run_arima_step, run_garch_step
from api.services.spillover_service import analyze_spillover_step

# Get the application configuration
from utilities.configurator import load_configuration
config = load_configuration("config.yml")

router = APIRouter(tags=["Pipeline"])

# In api/routers/pipeline.py
# Add this import at the top
from api.services.interpretations import interpret_arima_results, interpret_garch_results


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
          8. Run spillover analysis if enabled
          9. Return all results including forecasts and human-readable interpretations
          10. Store results in the database for future reference
          
          All parameters have sensible defaults defined in the configuration.
          """,
          response_model=PipelineResponse)
async def run_pipeline_endpoint(pipeline_input: PipelineInput, db: Session = Depends(get_db)):
    """Execute the complete time series analysis pipeline with explicit parameters."""
    t1 = time.perf_counter()

    try:
        # Extract data source parameters
        source_type = pipeline_input.source_actual_or_synthetic_data
        start_date = pipeline_input.data_start_date
        end_date = pipeline_input.data_end_date
        symbols = pipeline_input.symbols or config.symbols

        # Create pipeline run record
        pipeline_run = PipelineRun(
            name=f"Pipeline run for {', '.join(symbols)}",
            status="running",
            source_type=source_type,
            start_date=start_date,
            end_date=end_date
        )
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)
        
        # Extract ARIMA parameters
        arima_p = pipeline_input.arima_params.get('p', config.stats_model_ARIMA_fit_p)
        arima_d = pipeline_input.arima_params.get('d', config.stats_model_ARIMA_fit_d)
        arima_q = pipeline_input.arima_params.get('q', config.stats_model_ARIMA_fit_q)
        arima_forecast_steps = config.stats_model_ARIMA_predict_steps
        
        # Extract GARCH parameters
        garch_p = pipeline_input.garch_params.get('p', config.stats_model_GARCH_fit_p)
        garch_q = pipeline_input.garch_params.get('q', config.stats_model_GARCH_fit_q)
        garch_dist = pipeline_input.garch_params.get('dist', config.stats_model_GARCH_fit_dist)
        garch_forecast_steps = config.stats_model_GARCH_predict_steps
        
        # Extract scaling parameters
        scaling_method = pipeline_input.scaling_method
        
        # Extract synthetic data parameters (if applicable)
        anchor_prices = None
        if source_type == "synthetic":
            synthetic_prices = pipeline_input.synthetic_anchor_prices or config.synthetic_anchor_prices
            anchor_prices = dict(zip(symbols, synthetic_prices))
            random_seed = pipeline_input.synthetic_random_seed or config.synthetic_random_seed
        
        # 1. Data acquisition: Either generate synthetic data or fetch actual market data
        # Research emphasizes using reliable financial time series data as the foundation
        # for any volatility modeling
        df_prices = generate_data_step(
            source_type=source_type, 
            start_date=start_date, 
            end_date=end_date, 
            symbols=symbols,
            anchor_prices=anchor_prices,
            random_seed=random_seed if source_type == "synthetic" else None
        )
        
        # 2. Convert prices to log returns
        # Studies emphasize the importance of proper data transformation,
        # especially converting prices to log returns before GARCH modeling, as returns exhibit
        # more stationary behavior than raw price series
        df_returns = convert_to_returns_step(df=df_prices)
        
        if df_returns.isnull().any().any():
            l.warning("Missing values detected in returns data - applying appropriate filling strategy")
            df_returns = fill_missing_data_step(df_returns, config.data_processor_missing_values_strategy)

        # 3. Test for stationarity using ADF test
        # Research shows that stationarity testing is a critical preliminary step
        # before applying time series models. Non-stationary data can lead to spurious regression
        # and invalid statistical inferences
        stationarity_results = test_stationarity_step(
            df=df_returns, 
            test_method="ADF", 
            p_value_threshold=config.data_processor_stationarity_test_p_value_threshold
        )

        if not stationarity_results["is_stationary"]:
            l.warning("Data is not stationary after transformation - results may be unreliable")
        
        # 4. Scale data for GARCH modeling
        # This preprocessing ensures numerical stability and comparable magnitude across series
        df_scaled = scale_for_garch_step(df=df_returns)
        
        # 5. Run ARIMA models to capture conditional mean dynamics
        # The standard approach in financial econometrics is to first model the
        # conditional mean with ARIMA to remove autocorrelation in returns, then model the
        # volatility of residuals. This two-stage approach is well-established in academic literature

        if df_returns.shape[0] < 30:  # Check for sufficient data points
            arima_p = min(arima_p, 1)  # Reduce model complexity for small samples
            arima_q = min(arima_q, 1)
            l.info("Reduced ARIMA order due to limited sample size")
            
        arima_summary, arima_forecast, arima_residuals = run_arima_step(
            df_stationary=df_scaled,
            p=arima_p,
            d=arima_d,
            q=arima_q,
            forecast_steps=arima_forecast_steps
        )
        
        # Generate human-readable interpretation of ARIMA results
        arima_interpretation = interpret_arima_results(arima_summary, arima_forecast)
        
        # 6. Run GARCH models on ARIMA residuals
        # Studies typically use either normal or Student's t distributions
        # for GARCH modeling, with t-distribution often preferred because financial returns frequently
        # exhibit fat tails. Research shows that simple GARCH(1,1) models often perform as well as
        # more complex specifications for many financial time series
        garch_summary, garch_forecast, _ = run_garch_step(
            df_residuals=arima_residuals,
            p=garch_p,
            q=garch_q,
            dist=garch_dist,
            forecast_steps=garch_forecast_steps
        )
        
        # Generate human-readable interpretation of GARCH results
        garch_interpretation = interpret_garch_results(garch_summary, garch_forecast)

        # 7. Run spillover analysis if enabled
        # Research highlights the importance of studying volatility transmission
        # across financial markets for risk management and portfolio diversification
        spillover_results = None
        if pipeline_input.spillover_enabled:
            spillover_params = pipeline_input.spillover_params
            spillover_input = SpilloverInput(
                data=df_returns.reset_index().to_dict('records'),
                method=spillover_params.get('method', 'diebold_yilmaz'),
                forecast_horizon=spillover_params.get('forecast_horizon', 10),
                window_size=spillover_params.get('window_size', None)
            )
            # Run spillover analysis
            spillover_results = analyze_spillover_step(spillover_input)

        # After executing the pipeline, store results in the database
        for symbol in symbols:
            # Store stationarity results
            stationarity_db = PipelineResult(
                pipeline_run_id=pipeline_run.id,
                symbol=symbol,
                result_type="stationarity",
                is_stationary=stationarity_results["is_stationary"],
                adf_statistic=stationarity_results["adf_statistic"],
                p_value=stationarity_results["p_value"],
                interpretation=stationarity_results["interpretation"]
            )
            db.add(stationarity_db)
            
            # Store ARIMA results
            arima_db = PipelineResult(
                pipeline_run_id=pipeline_run.id,
                symbol=symbol,
                result_type="arima",
                model_summary=arima_summary,
                forecast=json.dumps(arima_forecast),
                interpretation=arima_interpretation
            )
            db.add(arima_db)
            
            # Store GARCH results
            garch_db = PipelineResult(
                pipeline_run_id=pipeline_run.id,
                symbol=symbol,
                result_type="garch",
                model_summary=garch_summary,
                forecast=json.dumps(garch_forecast),
                interpretation=garch_interpretation
            )
            db.add(garch_db)
        
        # Update pipeline status
        pipeline_run.status = "completed"
        pipeline_run.end_time = datetime.datetime.utcnow()
        db.commit()

        # Record execution time
        log_execution_time(t1)

        # Return results with interpretations
        return {
            "stationarity_results": stationarity_results,
            "arima_summary": arima_summary,
            "arima_forecast": arima_forecast,
            "arima_interpretation": arima_interpretation,
            "garch_summary": garch_summary,
            "garch_forecast": garch_forecast,
            "garch_interpretation": garch_interpretation,
            "spillover_results": spillover_results
        }
    except Exception as e:
        # Update pipeline status on error
        if 'pipeline_run' in locals():
            pipeline_run.status = "failed"
            pipeline_run.end_time = datetime.datetime.utcnow()
            db.commit()
            
        l.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Pipeline failed: {str(e)}"
        )

def log_execution_time(start_time: float) -> None:
    """Log pipeline execution time in a readable format."""
    execution_time = time.perf_counter() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    l.info(
        f"Pipeline execution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    )

