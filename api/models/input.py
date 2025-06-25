#!/usr/bin/env python3
# timeseries-api/api/models/input.py
"""Input Pydantic models for API request validation.
These models are used to validate the input data for the API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum

# Get the application configuration for default values
from utilities.configurator import load_configuration
config = load_configuration("config.yml")

class DataSourceType(str, Enum):
    SYNTHETIC = "synthetic"
    ACTUAL_YFINANCE = "actual_yfinance"
    ACTUAL_STOOQ = "actual_stooq"

class DataGenerationInput(BaseModel):
    """Input model for data generation endpoint."""
    start_date: str
    end_date: str
    anchor_prices: dict

class MarketDataInput(BaseModel):
    """Input model for market data fetch endpoint."""
    symbols: List[str]
    start_date: str
    end_date: str
    interval: str = "1d"  # daily data by default

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
    dist: Optional[str] = "t"



class PipelineInput(BaseModel):
    """Input model for full pipeline endpoint."""
    source_actual_or_synthetic_data: DataSourceType = Field(
        default=DataSourceType.SYNTHETIC,
        description="Source of data to use: synthetic, actual_yfinance, or actual_stooq"
    )
    data_start_date: str = Field(
        default=config.data_start_date, 
        description="Start date for data generation/fetching (YYYY-MM-DD)"
    )
    data_end_date: str = Field(
        default=config.data_end_date, 
        description="End date for data generation/fetching (YYYY-MM-DD)"
    )
    symbols: Optional[List[str]] = Field(
        default=config.symbols, 
        description="Symbols to generate or fetch data for"
    )
    synthetic_anchor_prices: Optional[List[float]] = Field(
        default=config.synthetic_anchor_prices, 
        description="Anchor prices for synthetic data generation"
    )
    synthetic_random_seed: Optional[int] = Field(
        default=config.synthetic_random_seed, 
        description="Random seed for synthetic data generation"
    )
    scaling_method: str = Field(
        default=config.data_processor_scaling_method, 
        description="Scaling method"
    )
    arima_params: dict = Field(
        default={
            "p": config.stats_model_ARIMA_fit_p,
            "d": config.stats_model_ARIMA_fit_d,
            "q": config.stats_model_ARIMA_fit_q,
            "forecast_steps": config.stats_model_ARIMA_predict_steps
        }, 
        description="ARIMA parameters"
    )
    garch_params: dict = Field(
        default={
            "p": config.stats_model_GARCH_fit_p,
            "q": config.stats_model_GARCH_fit_q,
            "dist": config.stats_model_GARCH_fit_dist
        }, 
        description="GARCH parameters"
    )
    spillover_enabled: bool = Field(
        default=False,
        description="Enable spillover analysis"
    )
    spillover_params: dict = Field(
        default={
            "method": "diebold_yilmaz",
            "forecast_horizon": 10,
            "window_size": None
        },
        description="Spillover analysis parameters"
    )

class SpilloverInput(BaseModel):
    """Input model for spillover analysis endpoint."""
    data: List[Dict[str, Any]]
    method: str = Field(
        default="diebold_yilmaz", 
        description="Method for spillover calculation"
    )
    forecast_horizon: int = Field(
        default=10, 
        description="Forecast horizon for variance decomposition"
    )
    window_size: Optional[int] = Field(
        default=None,
        description="Window size for rolling analysis (None for full sample)"
    )