#!/usr/bin/env python3
# run_pipeline.py

import time  # stopwatch

# handle relative directory imports for chronicler
import logging as l
from src.chronicler import init_chronicler

from src.configurator import load_configuration
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

        # Generate price data
        l.info("\n\n+++++pipeline: generate_price_series()+++++")
        price_dict, price_df = data_generator.generate_price_series(config=config)

        # Fill data
        l.info("\n\n+++++pipeline: fill_data()+++++")
        df_filled = data_processor.fill_data(df=price_df, config=config)

        # Scale data
        l.info("\n\n+++++pipeline: scale_data()+++++")
        df_scaled = data_processor.scale_data(df=df_filled, config=config)

        # Stationarize data
        l.info("\n\n+++++pipeline: stationarize_data()+++++")
        df_stationary = data_processor.stationarize_data(df=df_scaled, config=config)

        # Test stationarity
        l.info("\n\n+++++pipeline: test_stationarity()+++++")
        adf_results = data_processor.test_stationarity(df=df_stationary, config=config)

        # Log stationarity results
        l.info("\n\n+++++pipeline: log_stationarity()+++++")
        data_processor.log_stationarity(df=adf_results, config=config)

        l.info("\n\n+++++pipeline: modeling+++++")
        if config.stats_model.ARIMA.enabled:
            l.info("\n\n+++++pipeline: run_arima()+++++")
            arima_fit, arima_forecast = stats_model.run_arima(df_stationary, config)

        if config.stats_model.GARCH.enabled:
            l.info("\n\n+++++pipeline: run_garch()+++++")
            garch_fit, garch_forecast = stats_model.run_garch(df_stationary, config)

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
