#!/usr/bin/env python3
# timeseries-api/api/cli_pipeline.py

# import parent directory modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

import time  # stopwatch

# handle relative directory imports for chronicler
import logging as l
from utilities.chronicler import init_chronicler

from utilities.configurator import load_configuration
from timeseries_compute import data_generator, data_processor, stats_model

from api.services.interpretations import interpret_granger_causality
from timeseries_compute.spillover_processor import test_granger_causality


def main():
    """Main function to run the pipeline."""
    t1 = time.perf_counter()
    chronicler = init_chronicler()
    l.info("\n\n+++++pipeline: start+++++")
    try:
        # Load configuration
        config_file = "config.yml"
        config = load_configuration(config_file=config_file)

        # Create a dictionary mapping symbols to their anchor prices
        anchor_prices = dict(zip(config.symbols, config.synthetic_anchor_prices))
        l.info(f"Using anchor prices: {anchor_prices}")

        # Generate price data
        l.info("\n\n+++++pipeline: generate_price_series()+++++")
        _, price_df = data_generator.generate_price_series(
            start_date=config.data_start_date,
            end_date=config.data_end_date,
            anchor_prices=anchor_prices,
        )
        
        # Check if missing value processing is enabled in config
        if config.data_processor_missing_values_enabled:
            # Check and handle missing values in price data
            missing_count = price_df.isnull().sum().sum()
            if missing_count > 0:
                l.info(f"\n\n+++++pipeline: handling {missing_count} missing values in price data+++++")
                # Get missing values strategy from config
                missing_strategy = getattr(config, "data_processor_missing_values_strategy", "ffill")
                price_df = data_processor.handle_missing_values(price_df, strategy=missing_strategy)
                l.info(f"Applied '{missing_strategy}' strategy to handle missing values")
                remaining_missing = price_df.isnull().sum().sum()
                if remaining_missing > 0:
                    l.warning(f"There are still {remaining_missing} missing values after processing")
        else:
            # Log if missing values exist but handling is disabled
            missing_count = price_df.isnull().sum().sum()
            if missing_count > 0:
                l.warning(f"Detected {missing_count} missing values in price data but missing value handling is disabled")

        # Calculate log returns
        l.info("\n\n+++++pipeline: price_to_returns()+++++")
        returns_df = data_processor.price_to_returns(price_df)

        # Test for stationarity
        l.info("\n\n+++++pipeline: test_stationarity()+++++")
        adf_results = data_processor.test_stationarity(returns_df)
        for col, result in adf_results.items():
            l.info(f"{col}: p-value={result['p-value']:.4e} "
                   f"{'(Stationary)' if result['p-value'] < 0.05 else '(Non-stationary)'}")
        
        # Generate and log stationarity interpretations immediately
        from api.services.interpretations import interpret_stationarity_test
        l.info("\n----- Stationarity Test Interpretations -----")
        stationarity_interpretations = interpret_stationarity_test(adf_results)
        for series, interpretation in stationarity_interpretations.items():
            l.info(f"\n{series}:\n{interpretation}")
        
        # Scale data for GARCH modeling
        l.info("\n\n+++++pipeline: scale_for_garch()+++++")
        scaled_returns_df = data_processor.scale_for_garch(returns_df)

        # Run ARIMA models if enabled
        if config.stats_model_ARIMA_enabled:
            l.info("\n\n+++++pipeline: run_arima()+++++")
            arima_fits, arima_forecasts = stats_model.run_arima(
                df_stationary=scaled_returns_df,
                p=config.stats_model_ARIMA_fit_p,
                d=config.stats_model_ARIMA_fit_d,
                q=config.stats_model_ARIMA_fit_q,
                forecast_steps=config.stats_model_ARIMA_predict_steps,
            )
            
            # Extract ARIMA residuals for GARCH modeling
            arima_residuals = pd.DataFrame(index=scaled_returns_df.index)
            for column in scaled_returns_df.columns:
                arima_residuals[column] = arima_fits[column].resid
                
            # Generate and log ARIMA interpretations immediately
            from api.services.interpretations import interpret_arima_results
            l.info("\n----- ARIMA Model Interpretations -----")
            for column in scaled_returns_df.columns:
                if column in arima_fits and column in arima_forecasts:
                    forecast_list = arima_forecasts[column].tolist() if hasattr(arima_forecasts[column], 'tolist') else [arima_forecasts[column]]
                    arima_interp = interpret_arima_results(str(arima_fits[column].summary()), forecast_list)
                    l.info(f"\n{column} ARIMA Model:\n{arima_interp}")
        else:
            arima_residuals = scaled_returns_df  # Use scaled returns if ARIMA is disabled

        # Run GARCH models if enabled
        if config.stats_model_GARCH_enabled:
            l.info("\n\n+++++pipeline: run_garch()+++++")
            garch_fits, garch_forecasts = stats_model.run_garch(
                df_stationary=arima_residuals,
                p=config.stats_model_GARCH_fit_p,
                q=config.stats_model_GARCH_fit_q,
                dist=config.stats_model_GARCH_fit_dist,
                forecast_steps=config.stats_model_GARCH_predict_steps,
            )
            
            # Extract conditional volatilities
            cond_vol = pd.DataFrame(index=arima_residuals.index)
            for column in arima_residuals.columns:
                cond_vol[column] = np.sqrt(garch_fits[column].conditional_volatility)
            
            # Display volatility forecasts
            l.info("GARCH volatility forecasts:")
            for col, forecast in garch_forecasts.items():
                if hasattr(forecast, '__iter__'):
                    # Convert variance forecasts to volatility
                    forecast_vols = np.sqrt(forecast)
                    l.info(f"  {col} volatility forecast: {', '.join([f'{v:.6f}' for v in forecast_vols])}")
                else:
                    l.info(f"  {col}: {np.sqrt(forecast):.6f}")
            
            # Generate and log GARCH interpretations immediately
            from api.services.interpretations import interpret_garch_results
            l.info("\n----- GARCH Model Interpretations -----")
            for column in arima_residuals.columns:
                if column in garch_fits and column in garch_forecasts:
                    forecast_list = garch_forecasts[column].tolist() if hasattr(garch_forecasts[column], 'tolist') else [garch_forecasts[column]]
                    # Convert variance forecasts to volatility
                    volatility_forecasts = [np.sqrt(v) for v in forecast_list]
                    garch_interp = interpret_garch_results(str(garch_fits[column].summary()), volatility_forecasts)
                    l.info(f"\n{column} GARCH Model:\n{garch_interp}")

        if config.spillover_analysis_enabled:
            l.info("\n\n+++++pipeline: analyze_spillover()+++++")
            from api.models.input import SpilloverInput
            from api.services.spillover_service import analyze_spillover_step
            from api.services.interpretations import interpret_spillover_results
            
            spillover_input = SpilloverInput(
                data=returns_df.reset_index().to_dict('records'),
                method=config.spillover_analysis_method,
                forecast_horizon=config.spillover_analysis_forecast_horizon,
                window_size=config.spillover_analysis_window_size
            )
            
            spillover_results = analyze_spillover_step(spillover_input)
            
            l.info(f"Total spillover index: {spillover_results['total_spillover_index']:.2f}%")
            l.info(f"Net spillover: {spillover_results['net_spillover']}")
            
            # Generate interpretations for the spillover results immediately
            interpretations = interpret_spillover_results(spillover_results, significance_threshold=0.1)
            l.info("\n----- Spillover Analysis Interpretations -----")
            for key, interpretation in interpretations.items():
                l.info(f"\n{key}:\n{interpretation}")
        
        # Run Granger causality tests if enabled and multiple assets are available
        if len(returns_df.columns) > 1 and hasattr(config, "granger_causality_enabled") and config.granger_causality_enabled:
            l.info("\n----- Granger Causality Tests -----")
            max_lag = config.granger_causality_max_lag if hasattr(config, "granger_causality_max_lag") else 5
            granger_results = {}

            for source in returns_df.columns:
                for target in returns_df.columns:
                    if source != target:
                        try:
                            # Run the Granger causality test with proper arguments
                            test_result = test_granger_causality(
                                series1=returns_df[source],
                                series2=returns_df[target],
                                max_lag=max_lag
                            )
                            
                            # Store the full test result directly
                            granger_results[f"{source}->{target}"] = test_result
                            
                            # Log the result
                            causality = test_result.get("causality", False)
                            p_values = test_result.get("p_values", {})
                            optimal_lag = test_result.get("optimal_lag")
                            l.info(f"Granger causality {source}->{target}: causality={causality}, optimal_lag={optimal_lag}")
                            
                        except Exception as e:
                            l.warning(f"Error in Granger causality test for {source}->{target}: {e}")
            
            # Generate and log Granger causality interpretations immediately
            granger_interpretations = interpret_granger_causality(granger_results)
            for relation, interpretation in granger_interpretations.items():
                l.info(f"\n{relation}:\n{interpretation}")

    except Exception as e:
        l.exception(f"\nError in pipeline:\n{e}")
        raise

    # Log execution time
    l.info("\n\n+++++pipeline: complete+++++")
    execution_time = time.perf_counter() - t1
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    l.info(
        f"\nexecution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    )


if __name__ == "__main__":
    main()
