#!/usr/bin/env python3
# # timeseries-api/api/models/__init__.py
"""API data models.
This module contains the data models used for the API.
__init__ is the entry point of the package, and it imports all the data models
"""

from .input import (
    DataGenerationInput,
    ScalingInput,
    StationarityTestInput,
    ARIMAInput,
    GARCHInput,
    PipelineInput
)

from .response import (
    TimeSeriesDataResponse,
    StationarityTestResponse,
    ARIMAModelResponse,
    GARCHModelResponse,
    PipelineResponse
)

# __all__ is used by the `from <module> import *` syntax
__all__ = [
    "DataGenerationInput",
    "ScalingInput",
    "StationarityTestInput",
    "ARIMAInput",
    "GARCHInput",
    "PipelineInput",
    "TimeSeriesDataResponse",
    "StationarityTestResponse",
    "ARIMAModelResponse",
    "GARCHModelResponse",
    "PipelineResponse"
]