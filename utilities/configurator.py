#!/usr/bin/env python3
# timeseries-api/utilities/configurator.py

import os
import yaml
import logging as l
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

def read_config_from_fs(config_filename: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file from the config directory.

    Args:
        config_filename (str): The name of the configuration file.

    Returns:
        Dict[str, Any]: The parsed configuration file as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../config/", config_filename)
    with open(config_path, "r") as f:
        try:
            contents = yaml.safe_load(f)
            l.info(f"yml contents:\n{contents}")
            return contents
        except Exception as e:
            l.error(f"Error loading config: {e}")
            raise

class Config(BaseModel):
    """Flat configuration structure using Pydantic for validation."""
    # Metadata
    metadata_version: float = Field(default=1.0)
    metadata_environment: str = Field(default="dev")
    
    # Data Source Selection
    source_actual_or_synthetic_data: str = Field(default="synthetic", pattern="^(actual|synthetic)$")
    data_start_date: str = Field(default="2023-01-01")
    data_end_date: str = Field(default="2023-02-01")
    symbols: List[str] = Field(default=["GME", "BYND", "BYD"])
    
    # Synthetic Data Options
    synthetic_anchor_prices: List[float] = Field(default=[150.0, 200.0, 15.0])
    synthetic_random_seed: int = Field(default=1)
    
    # Data Processor - Missing Values
    data_processor_missing_values_enabled: bool = Field(default=True)
    data_processor_missing_values_strategy: str = Field(default="forward_fill")
    
    # Data Processor - Returns Conversion
    data_processor_returns_conversion_enabled: bool = Field(default=True)
    
    # Data Processor - Stationary
    data_processor_stationary_enabled: bool = Field(default=True)
    data_processor_stationary_method: str = Field(default="difference")
    
    # Data Processor - Stationarity Test
    data_processor_stationarity_test_method: str = Field(default="ADF")
    data_processor_stationarity_test_p_value_threshold: float = Field(default=0.05)
    
    # Data Processor - Scaling
    data_processor_scaling_method: str = Field(default="standardize")
    
    # ARIMA Model
    stats_model_ARIMA_enabled: bool = Field(default=False)
    stats_model_ARIMA_fit_p: int = Field(default=1)
    stats_model_ARIMA_fit_d: int = Field(default=1)
    stats_model_ARIMA_fit_q: int = Field(default=1)
    stats_model_ARIMA_predict_steps: int = Field(default=5)
    stats_model_ARIMA_residuals_as_garch_input: bool = Field(default=True)
    
    # GARCH Model
    stats_model_GARCH_enabled: bool = Field(default=False)
    stats_model_GARCH_fit_p: int = Field(default=1)
    stats_model_GARCH_fit_q: int = Field(default=1)
    stats_model_GARCH_fit_dist: str = Field(default="normal")
    stats_model_GARCH_predict_steps: int = Field(default=5)
    stats_model_GARCH_volatility_format: str = Field(default="standard_deviation")

    # Spillover Analysis
    spillover_analysis_enabled: bool = Field(default=False)
    spillover_analysis_method: str = Field(default="diebold_yilmaz")
    spillover_analysis_forecast_horizon: int = Field(default=10)
    spillover_analysis_window_size: Optional[int] = Field(default=None)

    # Grandger Causality
    granger_causality_enabled: bool = Field(default=True)
    granger_causality_max_lag: int = Field(default=5)
    granger_causality_p_value_threshold: float = Field(default=0.05)

    # Network Metrics
    network_metrics_enabled: bool = Field(default=False)
    spillover_significance_threshold: float = Field(default=0.1)

def load_configuration(config_file: str) -> Config:
    """
    Load and validate the YAML configuration file.

    Args:
        config_file (str): The name of the configuration file.

    Returns:
        Config: The validated configuration object.
    """
    l.info(f"# Loading config_file: {config_file}")
    config_dict = read_config_from_fs(config_file)
    return Config(**config_dict)
