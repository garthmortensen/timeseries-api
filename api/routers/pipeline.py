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
# Add import for calculate_stats
from timeseries_compute.stats_model import calculate_stats

# Get the application configuration
from utilities.configurator import load_configuration
config = load_configuration("config.yml")

router = APIRouter(tags=["Pipeline"])

# In api/routers/pipeline.py
# Add this import at the top
from api.services.interpretations import interpret_arima_results, interpret_garch_results
# Import for granger causality
from api.services.spillover_service import perform_granger_causality

@router.post("/run_pipeline", 
          summary="Execute the complete time series analysis pipeline",
          description="""
          Run the end-to-end time series analysis pipeline with a single API call.
          
          This endpoint performs a complete workflow:
          1. Generate synthetic data or fetch real market data
          2. Fill missing data (if configured)
          3. Convert prices to returns
          4. Test for stationarity (if configured)
          5. Scale data for GARCH modeling
          6. Fit ARIMA models for conditional mean
          7. Extract ARIMA residuals
          8. Fit GARCH models for volatility forecasting
          9. Run spillover analysis and Granger causality if enabled
          10. Return all results including forecasts and human-readable interpretations
          11. Store results in the database for future reference
          
          All parameters have sensible defaults defined in the configuration.
          """,
          response_model=PipelineResponse,
          responses={
              200: {
                  "description": "Successfully executed pipeline",
                  "content": {
                      "application/json": {
                          "example": {
                              "original_data": [
                                  {"date": "2023-01-01", "GME": 150.0},
                                  {"date": "2023-01-02", "GME": 152.3}
                              ],
                              "returns_data": [
                                  {"date": "2023-01-02", "GME": 0.0153}
                              ],
                              # Include abbreviated examples of other fields
                              "stationarity_results": {
                                  "adf_statistic": -3.45,
                                  "p_value": 0.032,
                                  "critical_values": {"1%": -3.75, "5%": -3.0, "10%": -2.63},
                                  "is_stationary": True,
                                  "interpretation": "The series is stationary (p-value: 0.0320)."
                              },
                              "arima_summary": "ARIMA(1,1,1) Model Results...",
                              "arima_forecast": [0.002, 0.003, 0.0025],
                              "arima_interpretation": "The ARIMA model shows an increasing trend...",
                              "garch_summary": "GARCH(1,1) Model Results...",
                              "garch_forecast": [0.0025, 0.0028, 0.0030],
                              "garch_interpretation": "The GARCH model predicts stable volatility..."
                          }
                      }
                  }
              }
          })
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
        
        # Fill missing data if configured
        if config.data_processor_missing_values_enabled:
            l.info("Applying missing data processing as configured")
            df_prices = fill_missing_data_step(df_prices, config.data_processor_missing_values_strategy)
        
        # 2. Convert prices to log returns
        # Studies emphasize the importance of proper data transformation,
        # especially converting prices to log returns before GARCH modeling, as returns exhibit
        # more stationary behavior than raw price series
        df_returns = convert_to_returns_step(df=df_prices)
        
        # Check again for missing values in returns
        if df_returns.isnull().any().any():
            l.warning("Missing values detected in returns data - applying appropriate filling strategy")
            df_returns = fill_missing_data_step(df_returns, config.data_processor_missing_values_strategy)

        # 3. Test for stationarity using ADF test only if configured
        stationarity_results = {"is_stationary": True}  # Default if test is skipped
        if config.data_processor_stationary_enabled:
            # Research shows that stationarity testing is a critical preliminary step
            # before applying time series models. Non-stationary data can lead to spurious regression
            # and invalid statistical inferences
            stationarity_results = test_stationarity_step(
                df=df_returns, 
                test_method="ADF", 
                p_value_threshold=config.data_processor_stationarity_test_p_value_threshold
            )
        else:
            l.info("Stationarity testing skipped per configuration")
            stationarity_results = {
                "is_stationary": True,  # Assume stationary
                "adf_statistic": None,
                "p_value": None,
                "critical_values": None,
                "interpretation": "Stationarity testing skipped per configuration."
            }
            
        # Calculate comprehensive statistics if data is stationary
        series_stats = {}
        if stationarity_results["is_stationary"]:
            l.info("Calculating comprehensive statistics for stationary data")
            for column in df_returns.columns:
                if column != 'Date':
                    series_stats[column] = calculate_stats(df_returns[column])
                    l.info(f"Statistics for {column}: Mean={series_stats[column]['mean']:.6f}, "
                           f"Std={series_stats[column]['std']:.6f}, Skew={series_stats[column]['skew']:.6f}, "
                           f"Kurt={series_stats[column]['kurt']:.6f}")
            
            # Add statistics to stationarity results
            stationarity_results["series_stats"] = series_stats

        # 4. Scale data for GARCH modeling
        # This preprocessing ensures numerical stability and comparable magnitude across series
        df_scaled = scale_for_garch_step(df=df_returns)
        
        # Ensure Date is set as index before passing to ARIMA
        if 'Date' in df_scaled.columns:
            df_scaled = df_scaled.set_index('Date')

        # 5. Run ARIMA models to capture conditional mean dynamics
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
        # Now capture the conditional volatilities (third return value)
        garch_summary, garch_forecast, cond_vol = run_garch_step(
            df_residuals=arima_residuals,
            p=garch_p,
            q=garch_q,
            dist=garch_dist,
            forecast_steps=garch_forecast_steps
        )
        
        # Generate human-readable interpretation of GARCH results
        garch_interpretation = interpret_garch_results(garch_summary, garch_forecast)

        # 7. Run spillover analysis if enabled
        spillover_results = None
        granger_causality_results = None
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
            
            # Run Granger causality analysis
            granger_causality_results = perform_granger_causality(
                df_returns, 
                max_lag=spillover_params.get('max_lag', 5),
                alpha=spillover_params.get('alpha', 0.05)
            )

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

        # Convert DataFrames to dictionaries for JSON serialization
        original_data_dict = df_prices.reset_index().to_dict('records')
        returns_data_dict = df_returns.reset_index().to_dict('records')
        pre_garch_data_dict = arima_residuals.reset_index().to_dict('records')
        post_garch_data_dict = cond_vol.reset_index().to_dict('records') if cond_vol is not None else None

        # Include scaled data only if the data is not stationary
        scaled_data_dict = None
        if not stationarity_results["is_stationary"]:
            scaled_data_dict = df_scaled.reset_index().to_dict('records')

        # Return expanded results with all the requested data
        return {
            "original_data": original_data_dict,
            "returns_data": returns_data_dict,
            "scaled_data": scaled_data_dict,
            "pre_garch_data": pre_garch_data_dict,
            "post_garch_data": post_garch_data_dict,
            "stationarity_results": stationarity_results,
            "series_stats": series_stats if stationarity_results["is_stationary"] else None,  # Add calculated statistics
            "arima_summary": arima_summary,
            "arima_forecast": arima_forecast,
            "arima_interpretation": arima_interpretation,
            "garch_summary": garch_summary,
            "garch_forecast": garch_forecast,
            "garch_interpretation": garch_interpretation,
            "spillover_results": spillover_results,
            "granger_causality_results": granger_causality_results
        }
    except Exception as e:
        # Get more detailed error information
        import traceback
        error_trace = traceback.format_exc()
        error_location = f"{e.__class__.__name__} in {e.__traceback__.tb_frame.f_code.co_filename} at line {e.__traceback__.tb_lineno}"
        
        error_message = (
            f"Pipeline error: {str(e)}\n"
            f"Error type: {e.__class__.__name__}\n"
            f"Error location: {error_location}"
        )
        
        l.error(error_message)
        l.debug(f"Full traceback:\n{error_trace}")
        
        # Update pipeline status on error with more details
        if 'pipeline_run' in locals():
            pipeline_run.status = "failed"
            pipeline_run.end_time = datetime.datetime.utcnow()
            db.commit()
        
        raise HTTPException(
            status_code=500, 
            detail={
                "message": f"Pipeline failed: {str(e)}",
                "error_type": e.__class__.__name__,
                "error_location": error_location
            }
        )

def log_execution_time(start_time: float) -> None:
    """Log pipeline execution time in a readable format."""
    execution_time = time.perf_counter() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    l.info(
        f"Pipeline execution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    )

