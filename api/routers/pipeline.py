#!/usr/bin/env python3
# timeseries-api/api/routers/pipeline.py
"""End-to-end pipeline API endpoint.
"""

import datetime
import json
import logging as l
import time
import traceback
from typing import Dict, Any
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
from api.utils.json_handling import round_for_json
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
        raw_data_source_metadata = {}
        raw_api_records = []  # Initialize to prevent unbound variable
        api_provider = "Unknown"  # Initialize with default
        api_endpoint = "Unknown"  # Initialize with default
        df_prices = None  # Initialize to prevent unbound variable
        
        if source_type == "synthetic":
            synthetic_prices = pipeline_input.synthetic_anchor_prices or config.synthetic_anchor_prices
            anchor_prices = dict(zip(symbols, synthetic_prices))
            random_seed = pipeline_input.synthetic_random_seed or config.synthetic_random_seed
            
            # Capture synthetic data generation metadata
            raw_data_source_metadata = {
                "data_source": "synthetic",
                "generation_parameters": {
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "anchor_prices": anchor_prices,
                    "random_seed": random_seed
                },
                "generation_timestamp": datetime.datetime.utcnow().isoformat(),
                "data_lineage": "Synthetic data generated using geometric Brownian motion"
            }
        
        # 1. Enhanced Data acquisition with comprehensive ETL metadata
        if source_type == "synthetic":
            df_prices = generate_data_step(
                source_type=source_type, 
                start_date=start_date, 
                end_date=end_date, 
                symbols=symbols,
                anchor_prices=anchor_prices,
                random_seed=random_seed
            )
            # For synthetic data, create mock raw API records
            raw_api_records = df_prices.reset_index().to_dict('records')
        else:
            # For actual market data, capture both processed DataFrame and raw API records
            from api.services.market_data_service import fetch_market_data_yfinance, fetch_market_data_stooq
            
            try:
                if source_type == "actual_yfinance":
                    raw_api_records, df_prices = fetch_market_data_yfinance(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    api_provider = "Yahoo Finance (yfinance)"
                    api_endpoint = f"yfinance.download({symbols}, start='{start_date}', end='{end_date}')"
                elif source_type == "actual_stooq":
                    raw_api_records, df_prices = fetch_market_data_stooq(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    api_provider = "Stooq"
                    api_endpoint = f"pandas_datareader.DataReader({symbols}, 'stooq', start='{start_date}', end='{end_date}')"
                
                # Capture comprehensive raw data source metadata for actual market data
                raw_data_source_metadata = {
                    "data_source": source_type,
                    "api_provider": api_provider,
                    "api_endpoint": api_endpoint,
                    "request_parameters": {
                        "symbols": symbols,
                        "start_date": start_date,
                        "end_date": end_date,
                        "data_type": "adjusted_close_prices"
                    },
                    "fetch_timestamp": datetime.datetime.utcnow().isoformat(),
                    "raw_records_count": len(raw_api_records),
                    "date_range_actual": {
                        "first_date": raw_api_records[0]["date"] if raw_api_records else None,
                        "last_date": raw_api_records[-1]["date"] if raw_api_records else None
                    },
                    "symbols_retrieved": list(df_prices.columns) if hasattr(df_prices, 'columns') else symbols,
                    "data_quality": {
                        "total_observations": len(raw_api_records),
                        "missing_values_detected": df_prices.isnull().sum().to_dict() if hasattr(df_prices, 'isnull') else {},
                        "data_completeness_ratio": (1.0 - df_prices.isnull().sum().sum() / (len(df_prices) * len(df_prices.columns))) if hasattr(df_prices, 'isnull') else 1.0
                    },
                    "data_lineage": f"Raw market data fetched from {api_provider} API"
                }
                
            except Exception as e:
                l.error(f"Error fetching market data: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")
        
        # Export raw API data for ETL audit trail
        export_data(raw_api_records, name="raw_api_data_source")
        export_data(raw_data_source_metadata, name="raw_data_source_metadata")
        
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
        stationarity_results: Dict[str, Any] = {}  # Default if test is skipped
        if config.data_processor_stationary_enabled:
            # Research shows that stationarity testing is a critical preliminary step
            # before applying time series models. Non-stationary data can lead to spurious regression
            # and invalid statistical inferences
            results = test_stationarity_step(
                df=df_returns, 
                test_method="ADF", 
                p_value_threshold=config.data_processor_stationarity_test_p_value_threshold
            )
            stationarity_results = {
                "all_symbols_stationarity": results
            }
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
            symbol_stationarity_results = stationarity_results["all_symbols_stationarity"]
            if isinstance(symbol_stationarity_results, dict):
                any_stationary = any(result.get("is_stationary", True) for result in symbol_stationarity_results.values())
        else:
            # Fallback for when stationarity testing is disabled
            any_stationary = True
            
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
            # Extract the string summary and forecast list from the structured data
            symbol_summary = all_arima_summaries[symbol]
            symbol_forecast = all_arima_forecasts[symbol]
            
            # Get the full summary string for interpretation
            model_summary_str = symbol_summary.get('full_summary', str(symbol_summary))
            
            # Get the point forecasts list
            forecast_list = symbol_forecast.get('point_forecasts', [])
            if not isinstance(forecast_list, list):
                forecast_list = [forecast_list] if forecast_list is not None else []
            
            # Get residuals if available for better model quality assessment
            residuals_dict = symbol_summary.get('residuals', {})
            residuals_list = list(residuals_dict.values()) if residuals_dict else None
            
            all_arima_interpretations[symbol] = interpret_arima_results(
                model_summary=model_summary_str,
                forecast=forecast_list,
                residuals=residuals_list
            )
        # Export ARIMA interpretations
        export_data(all_arima_interpretations, name="api_arima_interpretations")
        
        # 6. Run GARCH models on ARIMA residuals
        # Now capture the conditional volatilities (third return value) and fitted models (fourth return value)
        all_garch_summaries, all_garch_forecasts, cond_vol, fitted_garch_models = run_garch_step(
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
            # Extract the string summary and forecast list from the GARCH data
            symbol_summary = all_garch_summaries[symbol]
            symbol_forecast = all_garch_forecasts[symbol]
            
            # GARCH summaries are already strings (from statsmodels), but forecasts are lists
            model_summary_str = symbol_summary if isinstance(symbol_summary, str) else str(symbol_summary)
            
            # GARCH forecasts should already be lists of volatility values
            forecast_list = symbol_forecast if isinstance(symbol_forecast, list) else [symbol_forecast] if symbol_forecast is not None else []
            
            all_garch_interpretations[symbol] = interpret_garch_results(
                model_summary=model_summary_str,
                forecast=forecast_list
            )
        # Export GARCH interpretations
        export_data(all_garch_interpretations, name="api_garch_interpretations")

        # 7. Run spillover analysis if enabled
        # --- ENHANCED SPILLOVER ANALYSIS WITH MULTIVARIATE GARCH ---
        # Now includes CCC-GARCH, DCC-GARCH, and comprehensive spillover methods
        spillover_results = None
        granger_causality_results = None
        var_results = None
        multivariate_garch_results = None  # Add comprehensive MGARCH results
        
        if pipeline_input.spillover_enabled:
            spillover_params = pipeline_input.spillover_params
            spillover_method = spillover_params.get('method', 'diebold_yilmaz')
            
            # Standard spillover analysis
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
            
            # NEW: Comprehensive Multivariate GARCH Analysis
            # This captures CCC-GARCH, DCC-GARCH, and advanced correlation analysis
            try:
                from timeseries_compute.stats_model import run_multivariate_garch
                from api.services.interpretations import (
                    interpret_multivariate_garch_results,
                    interpret_correlation_dynamics,
                    interpret_portfolio_risk_metrics
                )
                
                l.info("Running comprehensive multivariate GARCH analysis with CCC and DCC models")
                multivariate_garch_results = run_multivariate_garch(
                    df_stationary=df_returns,  # Use original returns for MGARCH
                    arima_fits=None,  # Let MGARCH fit its own ARIMA models to avoid type issues
                    garch_fits=fitted_garch_models,  # Use fitted GARCH models (not summaries)
                    lambda_val=spillover_params.get('dcc_lambda', 0.95)  # EWMA decay factor
                )
                
                # Generate comprehensive interpretations for multivariate GARCH results
                mgarch_interpretations = interpret_multivariate_garch_results(
                    mgarch_results=multivariate_garch_results,
                    variable_names=list(df_returns.columns),
                    lambda_val=spillover_params.get('dcc_lambda', 0.95)
                )
                
                # Generate additional specialized interpretations
                additional_interpretations = {}
                
                # Interpret dynamic correlation patterns if DCC results are available
                if "dcc_correlation" in multivariate_garch_results and multivariate_garch_results["dcc_correlation"] is not None:
                    correlation_dynamics_interp = interpret_correlation_dynamics(
                        correlation_series=multivariate_garch_results["dcc_correlation"],
                        asset_names=list(df_returns.columns),
                        analysis_period=f"{start_date} to {end_date}"
                    )
                    additional_interpretations["correlation_dynamics"] = correlation_dynamics_interp
                
                # Interpret portfolio risk metrics if covariance matrix is available
                if "cc_covariance_matrix" in multivariate_garch_results and multivariate_garch_results["cc_covariance_matrix"] is not None:
                    portfolio_risk_interp = interpret_portfolio_risk_metrics(
                        cov_matrix=multivariate_garch_results["cc_covariance_matrix"],
                        asset_names=list(df_returns.columns),
                        equal_weights=True
                    )
                    additional_interpretations["portfolio_risk_metrics"] = portfolio_risk_interp
                
                # Add interpretations to the results
                multivariate_garch_results["interpretations"] = mgarch_interpretations
                multivariate_garch_results["additional_interpretations"] = additional_interpretations
                
                # Export comprehensive MGARCH results with interpretations
                export_data(multivariate_garch_results, name="api_multivariate_garch_results")
                export_data(mgarch_interpretations, name="api_mgarch_interpretations")
                export_data(additional_interpretations, name="api_mgarch_additional_interpretations")
                
                l.info("Multivariate GARCH analysis completed successfully")
                l.info(f"MGARCH results include: {list(multivariate_garch_results.keys())}")
                l.info(f"Generated {len(mgarch_interpretations)} primary interpretations and {len(additional_interpretations)} additional interpretations")
                
            except Exception as mgarch_error:
                l.warning(f"Multivariate GARCH analysis failed: {mgarch_error}")
                multivariate_garch_results = {
                    "error": f"Multivariate GARCH analysis failed: {str(mgarch_error)}",
                    "cc_correlation": None,
                    "dcc_correlation": None,
                    "cc_covariance_matrix": None,
                    "dcc_covariance": None,
                    "interpretations": {
                        "error": f"Multivariate GARCH interpretations unavailable due to analysis failure: {str(mgarch_error)}",
                        "CCC_GARCH": "CCC-GARCH interpretation unavailable",
                        "DCC_GARCH": "DCC-GARCH interpretation unavailable",
                        "Covariance_Analysis": "Covariance analysis unavailable",
                        "Portfolio_Risk_Assessment": "Portfolio risk assessment unavailable",
                        "Overall_MGARCH_Summary": "Overall MGARCH summary unavailable"
                    },
                    "additional_interpretations": {
                        "correlation_dynamics": "Correlation dynamics interpretation unavailable due to analysis failure",
                        "portfolio_risk_metrics": "Portfolio risk metrics interpretation unavailable due to analysis failure"
                    }
                }
            
            # Extract VAR results from spillover analysis
            if spillover_results:
                var_results = get_var_results_from_spillover(spillover_results, df_returns.columns.tolist())
                export_data(var_results, name="api_var_results")
            
            # Enhanced Granger causality analysis
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
            any_non_stationary = any(not result.get("is_stationary") for result in stationarity_results["all_symbols_stationarity"].values())
        
        if any_non_stationary:
            scaled_data_dict = df_scaled.reset_index().to_dict('records')
        # Return expanded results with comprehensive ETL metadata and raw API data
        pipeline_results = {
            # ===== RAW DATA SOURCE (ETL Best Practice) =====
            "raw_data_source": {
                "raw_api_records": raw_api_records,
                "source_metadata": raw_data_source_metadata
            },
            
            # ===== PROCESSED DATA AT EACH PIPELINE STAGE =====
            "original_data": original_data_dict,
            "returns_data": returns_data_dict,
            "scaled_data": scaled_data_dict,
            "pre_garch_data": pre_garch_data_dict,
            "post_garch_data": post_garch_data_dict,
            
            # ===== STATISTICAL ANALYSIS RESULTS =====
            "stationarity_results": stationarity_results,
            
            # ===== MODEL RESULTS =====
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
            
            # ===== SPILLOVER AND CAUSALITY ANALYSIS =====
            "spillover_results": spillover_results,
            "granger_causality_results": granger_causality_results,
            "var_results": var_results,
            
            # ===== NEW: ADVANCED MULTIVARIATE GARCH ANALYSIS =====
            "multivariate_garch_results": multivariate_garch_results,
            
            # ===== PIPELINE EXECUTION METADATA =====
            "pipeline_metadata": {
                "execution_timestamp": datetime.datetime.utcnow().isoformat(),
                "execution_time_seconds": time.perf_counter() - t1,
                "configuration_used": {
                    "arima_params": {
                        "p": arima_p,
                        "d": arima_d, 
                        "q": arima_q,
                        "forecast_steps": arima_forecast_steps
                    },
                    "garch_params": {
                        "p": garch_p,
                        "q": garch_q,
                        "dist": garch_dist,
                        "forecast_steps": garch_forecast_steps
                    },
                    "scaling_method": scaling_method,
                    "missing_values_enabled": config.data_processor_missing_values_enabled,
                    "missing_values_strategy": config.data_processor_missing_values_strategy if config.data_processor_missing_values_enabled else None,
                    "stationarity_test_enabled": config.data_processor_stationary_enabled,
                    "spillover_enabled": pipeline_input.spillover_enabled
                },
                "data_processing_summary": {
                    "input_symbols_requested": symbols,
                    "input_date_range": {
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "data_transformations_applied": [
                        "price_to_returns",
                        "garch_scaling" if df_scaled is not None else None,
                        "missing_values_treatment" if config.data_processor_missing_values_enabled else None,
                        "stationarity_testing" if config.data_processor_stationary_enabled else None
                    ],
                    "models_fitted": [
                        f"ARIMA({arima_p},{arima_d},{arima_q})",
                        f"GARCH({garch_p},{garch_q})",
                        "Spillover Analysis" if pipeline_input.spillover_enabled else None,
                        "Granger Causality" if pipeline_input.spillover_enabled else None,
                        "Multivariate GARCH (CCC-GARCH + DCC-GARCH)" if pipeline_input.spillover_enabled and multivariate_garch_results else None
                    ]
                }
            }
        }

        # Convert numpy types to native Python types for JSON serialization
        pipeline_results = round_for_json(pipeline_results)
        
        l.info(f"pipeline_results: {pipeline_results}")
        
        # Export complete pipeline results
        export_data(pipeline_results, name="api_pipeline_complete_results")
        return pipeline_results
    except Exception as e:
        # Get more detailed error information
        error_trace = traceback.format_exc()
        
        # Safe error location extraction
        error_location = f"{e.__class__.__name__}"
        if e.__traceback__ is not None:
            tb = e.__traceback__
            if hasattr(tb, 'tb_frame') and hasattr(tb, 'tb_lineno'):
                error_location += f" in {tb.tb_frame.f_code.co_filename} at line {tb.tb_lineno}"
        
        error_message = (
            f"Pipeline error: {str(e)}\n"
            f"Error type: {e.__class__.__name__}\n"
            f"Error location: {error_location}"
        )
        
        l.error(error_message)
        l.debug(f"Full traceback:\n{error_trace}")
        
        # Update pipeline status on error with more details
        pipeline_run_exists = 'pipeline_run' in locals() and pipeline_run is not None
        if db is not None and pipeline_run_exists:
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

