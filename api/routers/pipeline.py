#!/usr/bin/env python3
# timeseries-api/api/routers/pipeline.py
"""End-to-end pipeline API endpoint.
"""

import datetime
import json
import logging as l
import time
import traceback
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

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
from api.services.interpretations import interpret_arima_results, interpret_garch_results
from api.services.models_service import run_arima_step, run_garch_step
from api.services.spillover_service import analyze_spillover_step, perform_granger_causality, get_var_results_from_spillover
from timeseries_compute.stats_model import calculate_stats
from utilities.configurator import load_configuration
from utilities.export_util import export_data

# Get the application configuration
config = load_configuration("config.yml")

router = APIRouter(tags=["Pipeline"])

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

        # Create pipeline run record only if db is enabled
        if db is not None:
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
        else:
            pipeline_run = None
        
        # Extract ARIMA parameters
        arima_p = pipeline_input.arima_params.get('p', config.stats_model_ARIMA_fit_p)
        arima_d = pipeline_input.arima_params.get('d', config.stats_model_ARIMA_fit_d)
        arima_q = pipeline_input.arima_params.get('q', config.stats_model_ARIMA_fit_q)
        arima_forecast_steps = pipeline_input.arima_params.get('forecast_steps', config.stats_model_ARIMA_predict_steps)
        
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
        df_prices = generate_data_step(
            source_type=source_type, 
            start_date=start_date, 
            end_date=end_date, 
            symbols=symbols,
            anchor_prices=anchor_prices,
            random_seed=random_seed if source_type == "synthetic" else None
        )
        # Export price data
        export_data(df_prices, name="api_price_data")
        
        # Fill missing data if configured
        if config.data_processor_missing_values_enabled:
            l.info("Applying missing data processing as configured")
            df_prices = fill_missing_data_step(df_prices, config.data_processor_missing_values_strategy)
            # Export processed price data
            export_data(df_prices, name="api_processed_price_data")
        
        # 2. Convert prices to log returns
        # Studies emphasize the importance of proper data transformation,
        # especially converting prices to log returns before GARCH modeling, as returns exhibit
        # more stationary behavior than raw price series
        df_returns = convert_to_returns_step(df=df_prices)
        # Export returns data
        export_data(df_returns, name="api_returns_data")
        
        # Check again for missing values in returns
        if df_returns.isnull().any().any():
            l.warning("Missing values detected in returns data - applying appropriate filling strategy")
            df_returns = fill_missing_data_step(df_returns, config.data_processor_missing_values_strategy)
            # Export cleaned returns data
            export_data(df_returns, name="api_cleaned_returns_data")

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
            # Export stationarity results
            export_data(stationarity_results, name="api_stationarity_results")
        else:
            l.info("Stationarity testing skipped per configuration")
            # Create a mock structure matching the new format
            mock_results = {}
            for symbol in symbols:
                mock_results[symbol] = {
                    "adf_statistic": None,
                    "p_value": None,
                    "critical_values": None,
                    "is_stationary": True,
                    "interpretation": "Stationarity testing skipped per configuration."
                }
            stationarity_results = {
                "all_symbols_stationarity": mock_results
            }
    
        # Calculate comprehensive statistics if data is stationary
        # So one series may be stationary while others are not. Tricky
        series_stats = {}
        # Check if any symbol is stationary (or all if we want to be more strict)
        any_stationary = False
        if "all_symbols_stationarity" in stationarity_results:
            any_stationary = any(result["is_stationary"] for result in stationarity_results["all_symbols_stationarity"].values())
        else:
            # Fallback for when stationarity testing is disabled
            any_stationary = stationarity_results.get("is_stationary", True)
            
        if any_stationary:
            l.info("Calculating comprehensive statistics for stationary data")
            for column in df_returns.columns:
                if column != 'Date':
                    series_stats[column] = calculate_stats(df_returns[column])
                    l.info(f"Statistics for {column}: Mean={series_stats[column]['mean']:.6f}, "
                           f"Std={series_stats[column]['std']:.6f}, Skew={series_stats[column]['skew']:.6f}, "
                           f"Kurt={series_stats[column]['kurt']:.6f}")
            
            # Add statistics to stationarity results
            stationarity_results["series_stats"] = series_stats
            # Export series statistics
            export_data(series_stats, name="api_series_stats")

        # 4. Scale data for GARCH modeling
        # This preprocessing ensures numerical stability and comparable magnitude across series
        df_scaled = scale_for_garch_step(df=df_returns)
        # Export scaled data
        export_data(df_scaled, name="api_scaled_data")
        
        # Ensure Date is set as index before passing to ARIMA
        if 'Date' in df_scaled.columns:
            df_scaled = df_scaled.set_index('Date')

        # 5. Run ARIMA models to capture conditional mean dynamics
        if df_returns.shape[0] < 30:  # Check for sufficient data points
            arima_p = min(arima_p, 1)  # Reduce model complexity for small samples
            arima_q = min(arima_q, 1)
            l.info("Reduced ARIMA order due to limited sample size")
            
        all_arima_summaries, all_arima_forecasts, arima_residuals = run_arima_step(
            df_stationary=df_scaled,
            p=arima_p,
            d=arima_d,
            q=arima_q,
            forecast_steps=arima_forecast_steps
        )
        # Export ARIMA results
        export_data({"summaries": all_arima_summaries, "forecasts": all_arima_forecasts}, name="api_arima_results")
        export_data(arima_residuals, name="api_arima_residuals")
        
        # Generate human-readable interpretations of ARIMA results for all symbols
        all_arima_interpretations = {}
        for symbol in all_arima_summaries.keys():
            all_arima_interpretations[symbol] = interpret_arima_results(
                all_arima_summaries[symbol], 
                all_arima_forecasts[symbol]
            )
        # Export ARIMA interpretations
        export_data(all_arima_interpretations, name="api_arima_interpretations")
        
        # 6. Run GARCH models on ARIMA residuals
        # Now capture the conditional volatilities (third return value)
        all_garch_summaries, all_garch_forecasts, cond_vol = run_garch_step(
            df_residuals=arima_residuals,
            p=garch_p,
            q=garch_q,
            dist=garch_dist,
            forecast_steps=garch_forecast_steps
        )
        # Export GARCH results and conditional volatilities
        export_data({"summaries": all_garch_summaries, "forecasts": all_garch_forecasts}, name="api_garch_results")
        if cond_vol is not None:
            export_data(cond_vol, name="api_conditional_volatility")
        
        # Generate human-readable interpretations of GARCH results for all symbols
        all_garch_interpretations = {}
        for symbol in all_garch_summaries.keys():
            all_garch_interpretations[symbol] = interpret_garch_results(
                all_garch_summaries[symbol], 
                all_garch_forecasts[symbol]
            )
        # Export GARCH interpretations
        export_data(all_garch_interpretations, name="api_garch_interpretations")

        # 7. Run spillover analysis if enabled
        # --- SPILLOVER ANALYSIS INPUT SELECTION ---
        # We must use returns for Diebold-Yilmaz (DY) spillover, but GARCH-based spillover requires
        # the conditional volatility series. Using the wrong input will give meaningless results.
        spillover_results = None
        granger_causality_results = None
        var_results = None  # Add VAR results
        
        if pipeline_input.spillover_enabled:
            spillover_params = pipeline_input.spillover_params
            spillover_method = spillover_params.get('method', 'diebold_yilmaz')
            if spillover_method == 'diebold_yilmaz':
                spillover_data = df_returns.reset_index().to_dict('records')
            elif spillover_method == 'garch_spillover' and cond_vol is not None:
                spillover_data = cond_vol.reset_index().to_dict('records')
            else:
                # Default to returns if method is unknown or GARCH volatility is missing
                spillover_data = df_returns.reset_index().to_dict('records')
            spillover_input = SpilloverInput(
                data=spillover_data,
                method=spillover_method,
                forecast_horizon=spillover_params.get('forecast_horizon', 10),
                window_size=spillover_params.get('window_size', None)
            )
            spillover_results = analyze_spillover_step(spillover_input)
            export_data(spillover_results, name="api_spillover_results")
            
            # Extract VAR results from spillover analysis
            if spillover_results:
                var_results = get_var_results_from_spillover(spillover_results, df_returns.columns.tolist())
                export_data(var_results, name="api_var_results")
            
            granger_causality_results = perform_granger_causality(
                df_returns, 
                max_lag=spillover_params.get('max_lag', 5),
                alpha=spillover_params.get('alpha', 0.05)
            )
            export_data(granger_causality_results, name="api_granger_causality_results")

        # After executing the pipeline, store results in the database
        if db is not None and pipeline_run is not None:
            for symbol in symbols:
                # Store stationarity results - get results for this specific symbol
                symbol_stationarity = stationarity_results.get("all_symbols_stationarity", {}).get(symbol, {})
                stationarity_db = PipelineResult(
                    pipeline_run_id=pipeline_run.id,
                    symbol=symbol,
                    result_type="stationarity",
                    is_stationary=symbol_stationarity.get("is_stationary", True),
                    adf_statistic=symbol_stationarity.get("adf_statistic"),
                    p_value=symbol_stationarity.get("p_value"),
                    interpretation=symbol_stationarity.get("interpretation", "No results available")
                )
                db.add(stationarity_db)
                
                # Store ARIMA results
                arima_db = PipelineResult(
                    pipeline_run_id=pipeline_run.id,
                    symbol=symbol,
                    result_type="arima",
                    model_summary=all_arima_summaries.get(symbol, "No summary available"),
                    forecast=json.dumps(all_arima_forecasts.get(symbol, [])),
                    interpretation=all_arima_interpretations.get(symbol, "No interpretation available")
                )
                db.add(arima_db)
                
                # Store GARCH results
                garch_db = PipelineResult(
                    pipeline_run_id=pipeline_run.id,
                    symbol=symbol,
                    result_type="garch",
                    model_summary=all_garch_summaries.get(symbol, "No summary available"),
                    forecast=json.dumps(all_garch_forecasts.get(symbol, [])),
                    interpretation=all_garch_interpretations.get(symbol, "No interpretation available")
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
        # Check if any symbol is non-stationary
        any_non_stationary = False
        if "all_symbols_stationarity" in stationarity_results:
            any_non_stationary = any(not result["is_stationary"] for result in stationarity_results["all_symbols_stationarity"].values())
        
        if any_non_stationary:
            scaled_data_dict = df_scaled.reset_index().to_dict('records')
        # Return expanded results with all the requested data
        pipeline_results = {
            "original_data": original_data_dict,
            "returns_data": returns_data_dict,
            "scaled_data": scaled_data_dict,
            "pre_garch_data": pre_garch_data_dict,
            "post_garch_data": post_garch_data_dict,
            "stationarity_results": stationarity_results,
            "series_stats": series_stats,
            "arima_results": {
                "all_symbols_arima": {
                    symbol: {
                        "summary": all_arima_summaries[symbol],
                        "forecast": all_arima_forecasts[symbol],
                        "interpretation": all_arima_interpretations[symbol]
                    }
                    for symbol in all_arima_summaries.keys()
                }
            },
            "garch_results": {
                "all_symbols_garch": {
                    symbol: {
                        "summary": all_garch_summaries[symbol],
                        "forecast": all_garch_forecasts[symbol],
                        "interpretation": all_garch_interpretations[symbol]
                    }
                    for symbol in all_garch_summaries.keys()
                }
            },
            "spillover_results": spillover_results,
            "granger_causality_results": granger_causality_results,
            "var_results": var_results
        }

        l.info(f"pipeline_results: {pipeline_results}")
        
        # Export complete pipeline results
        export_data(pipeline_results, name="api_pipeline_complete_results")
        return pipeline_results
    except Exception as e:
        # Get more detailed error information
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
        if db is not None and 'pipeline_run' in locals() and pipeline_run is not None:
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

