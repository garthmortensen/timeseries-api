#!/usr/bin/env python3
# timeseries-api/api/services/__init__.py
"""Service functions for the API.
This module contains the service functions for the API. These service functions are used to perform the data processing and model training steps.
They're called services because they provide a service to the API, which is to process the data and train the models.
"""

from .data_service import (
    generate_data_step,
    fill_missing_data_step,
    scale_data_step,
    stationarize_data_step,
    test_stationarity_step,
    convert_to_returns_step,
    scale_for_garch_step
)

from .models_service import (
    run_arima_step,
    run_garch_step
)

from .spillover_service import analyze_spillover_step

__all__ = [
    "generate_data_step",
    "fill_missing_data_step",
    "scale_data_step",
    "stationarize_data_step",
    "test_stationarity_step",
    "convert_to_returns_step",
    "scale_for_garch_step",
    "run_arima_step",
    "run_garch_step",
    "analyze_spillover_step"
]
