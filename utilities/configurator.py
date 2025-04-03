#!/usr/bin/env python3
# timeseries-pipeline/utilities/configurator.py

import os
import yaml
import logging as l
from pydantic import BaseModel, Field
from typing import Dict, Any

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
    
    # Data Generator
    data_generator_enabled: bool = Field(default=True)
    data_generator_random_seed: int = Field(default=1)
    data_generator_start_date: str = Field(default="2023-01-01")
    data_generator_end_date: str = Field(default="2023-12-31")
    data_generator_anchor_prices_GME: float = Field(default=150.0)
    data_generator_anchor_prices_BYND: float = Field(default=200.0)
    data_generator_anchor_prices_BYD: float = Field(default=15.0)
    
    # Data Processor - Missing Values
    data_processor_missing_values_enabled: bool = Field(default=True)
    data_processor_missing_values_strategy: str = Field(default="forward_fill")
    
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
    
    # GARCH Model
    stats_model_GARCH_enabled: bool = Field(default=False)
    stats_model_GARCH_fit_p: int = Field(default=1)
    stats_model_GARCH_fit_q: int = Field(default=1)
    stats_model_GARCH_fit_dist: str = Field(default="normal")
    stats_model_GARCH_predict_steps: int = Field(default=5)

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
