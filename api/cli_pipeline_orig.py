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

        # Generate price data
        l.info("\n\n+++++pipeline: generate_price_series()+++++")
        _, price_df = (
            data_generator.generate_price_series(  # _ is shorthand for throwaway variable
                start_date=config.data_generator.start_date,
                end_date=config.data_generator.end_date,
                anchor_prices=config.data_generator.anchor_prices,
            )
        )

        # Fill data
        l.info("\n\n+++++pipeline: fill_data()+++++")
        strategy = config.data_processor.handle_missing_values.strategy
        df_filled = data_processor.fill_data(df=price_df, strategy=strategy)

        # Scale data
        l.info("\n\n+++++pipeline: scale_data()+++++")
        method = config.data_processor.scaling.method
        df_scaled = data_processor.scale_data(df=df_filled, method=method)

        # Stationarize data
        l.info("\n\n+++++pipeline: stationarize_data()+++++")
        method = config.data_processor.make_stationary.method
        df_stationary = data_processor.stationarize_data(df=df_scaled, method=method)

        # Test stationarity
        l.info("\n\n+++++pipeline: test_stationarity()+++++")
        method = config.data_processor.test_stationarity.method
        adf_results = data_processor.test_stationarity(df=df_stationary, method=method)

        # Log stationarity results
        l.info("\n\n+++++pipeline: log_stationarity()+++++")
        p_value_threshold = config.data_processor.test_stationarity.p_value_threshold
        data_processor.log_stationarity(
            adf_results=adf_results, p_value_threshold=p_value_threshold
        )

        if config.stats_model.ARIMA.enabled:
            l.info("\n\n+++++pipeline: run_arima()+++++")
            arima_params = config.stats_model.ARIMA.parameters_fit
            forecast_steps = config.stats_model.ARIMA.parameters_predict_steps
            arima_fit, arima_forecast = stats_model.run_arima(
                df_stationary=df_stationary,
                p=arima_params.p,
                d=arima_params.d,
                q=arima_params.q,
                forecast_steps=forecast_steps,
            )

        if config.stats_model.GARCH.enabled:
            l.info("\n\n+++++pipeline: run_garch()+++++")
            garch_params = config.stats_model.GARCH.parameters_fit
            forecast_steps = config.stats_model.GARCH.parameters_predict_steps
            garch_fit, garch_forecast = stats_model.run_garch(
                df_stationary=df_stationary,
                p=garch_params.p,
                q=garch_params.q,
                dist=garch_params.dist,
                forecast_steps=forecast_steps,
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
