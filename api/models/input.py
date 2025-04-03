#!/usr/bin/env python3
# # timeseries-pipeline/api/models/input.py
"""Input Pydantic models for API request validation.
These models are used to validate the input data for the API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class DataGenerationInput(BaseModel):
    """Input model for data generation endpoint."""
    start_date: str
    end_date: str
    anchor_prices: dict


class ScalingInput(BaseModel):
    """Input model for data scaling endpoint."""
    method: str
    data: list


class StationarityTestInput(BaseModel):
    """Input model for stationarity test endpoint."""
    data: list


class ARIMAInput(BaseModel):
    """Input model for ARIMA model endpoint."""
    p: int
    d: int
    q: int
    data: list


class GARCHInput(BaseModel):
    """Input model for GARCH model endpoint."""
    p: int
    q: int
    data: list
    dist: Optional[str] = "normal"


class PipelineInput(BaseModel):
    """Input model for full pipeline endpoint."""
    start_date: str = Field(
        ..., description="Start date for data generation (YYYY-MM-DD)"
    )
    end_date: str = Field(..., description="End date for data generation (YYYY-MM-DD)")
    anchor_prices: dict = Field(..., description="Symbol-prices for data generation")
    scaling_method: str = Field(
        default="standardize", description="Scaling method"
    )
    arima_params: dict = Field(
        default={"p": 1, "d": 1, "q": 1}, description="ARIMA parameters"
    )
    garch_params: dict = Field(
        default={"p": 1, "q": 1, "dist": "t"}, description="GARCH parameters"
    )
    