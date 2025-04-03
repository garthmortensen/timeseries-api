#!/usr/bin/env python3
# timeseries-pipeline/api/cli_pipeline.py

# import parent directory modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time  # stopwatch

# handle relative directory imports for chronicler
import logging as l
from utilities.chronicler import init_chronicler

from utilities.configurator import load_configuration
from generalized_timeseries import data_generator, data_processor, stats_model


def main():
    """Main function to run the pipeline."""
    t1 = time.perf_counter()
    chronicler = init_chronicler()
    l.info("\n\n+++++pipeline: start+++++")
    try:
        l.info("\n\n+++++pipeline: load_configuration()+++++")
        config_file = "config.yml"
        config = load_configuration(config_file=config_file)

        # Build anchor_prices dictionary from flat config fields
        anchor_prices = {
            "GME": config.data_generator_anchor_prices_GME,
            "BYND": config.data_generator_anchor_prices_BYND,
            "BYD": config.data_generator_anchor_prices_BYD,
        }

        # Generate price data
        l.info("\n\n+++++pipeline: generate_price_series()+++++")
        _, price_df = (
            data_generator.generate_price_series(  # _ is shorthand for throwaway variable
                start_date=config.data_generator_start_date,
                end_date=config.data_generator_end_date,
                anchor_prices=anchor_prices,
            )
        )

        # Fill data
        l.info("\n\n+++++pipeline: fill_data()+++++")
        if config.data_processor_missing_values_enabled:
            strategy = config.data_processor_missing_values_strategy
            df_filled = data_processor.fill_data(df=price_df, strategy=strategy)
        else:
            df_filled = price_df

        # Scale data
        l.info("\n\n+++++pipeline: scale_data()+++++")
        method = config.data_processor_scaling_method
        df_scaled = data_processor.scale_data(df=df_filled, method=method)

        # Stationarize data
        l.info("\n\n+++++pipeline: stationarize_data()+++++")
        if config.data_processor_stationary_enabled:
            method = config.data_processor_stationary_method
            df_stationary = data_processor.stationarize_data(df=df_scaled, method=method)
        else:
            df_stationary = df_scaled

        # Test stationarity
        l.info("\n\n+++++pipeline: test_stationarity()+++++")
        method = config.data_processor_stationarity_test_method
        adf_results = data_processor.test_stationarity(df=df_stationary, method=method)

        # Log stationarity results
        l.info("\n\n+++++pipeline: log_stationarity()+++++")
        p_value_threshold = config.data_processor_stationarity_test_p_value_threshold
        data_processor.log_stationarity(
            adf_results=adf_results, p_value_threshold=p_value_threshold
        )

        # Run ARIMA model if enabled
        if config.stats_model_ARIMA_enabled:
            l.info("\n\n+++++pipeline: run_arima()+++++")
            arima_fit, arima_forecast = stats_model.run_arima(
                df_stationary=df_stationary,
                p=config.stats_model_ARIMA_fit_p,
                d=config.stats_model_ARIMA_fit_d,
                q=config.stats_model_ARIMA_fit_q,
                forecast_steps=config.stats_model_ARIMA_predict_steps,
            )

        # Run GARCH model if enabled
        if config.stats_model_GARCH_enabled:
            l.info("\n\n+++++pipeline: run_garch()+++++")
            garch_fit, garch_forecast = stats_model.run_garch(
                df_stationary=df_stationary,
                p=config.stats_model_GARCH_fit_p,
                q=config.stats_model_GARCH_fit_q,
                dist=config.stats_model_GARCH_fit_dist,
                forecast_steps=config.stats_model_GARCH_predict_steps,
            )

    except Exception as e:
        l.exception(f"\nError in pipeline:\n{e}")
        raise

    l.info("\n\n+++++pipeline: complete+++++")
    execution_time = time.perf_counter() - t1
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    l.info(
        f"\nexecution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    )


if __name__ == "__main__":
    main()
