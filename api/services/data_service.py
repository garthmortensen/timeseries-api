#!/usr/bin/env python3
# timeseries-api/api/services/data_service.py
"""Data generation and transformation service functions.
This module contains functions for generating synthetic time series data and pulling actual data.
"""

import logging as l
import pandas as pd
from fastapi import HTTPException
from typing import Dict, Any

from timeseries_compute import data_generator, data_processor
from .interpretations import interpret_stationarity_test
from api.services.market_data_service import fetch_market_data


def generate_data_step(pipeline_input: Dict[str, Any], config) -> pd.DataFrame:
    """Generate or fetch time series data step."""
    try:
        if config.source_actual_or_synthetic_data == "synthetic":
            anchor_prices = dict(zip(
                config.symbols, 
                config.synthetic_anchor_prices
            ))
            
            _, df = data_generator.generate_price_series(
                start_date=config.data_start_date,
                end_date=config.data_end_date,
                anchor_prices=anchor_prices,
                random_seed=config.synthetic_random_seed
            )
            return df
        elif config.source_actual_or_synthetic_data == "actual":
            _, df = fetch_market_data(
                symbols=config.symbols,
                start_date=config.data_start_date,
                end_date=config.data_end_date,
            )
            return df
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data source: {config.source_actual_or_synthetic_data}"
            )
    except Exception as e:
        l.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def convert_to_returns_step(df: pd.DataFrame, config) -> pd.DataFrame:
    """Convert price data to log returns."""
    return data_processor.price_to_returns(df)


def scale_for_garch_step(df: pd.DataFrame, config) -> pd.DataFrame:
    """Scale time series data for GARCH modeling."""
    return data_processor.scale_for_garch(df)

def scale_data_step(df, config):
    """Scale time series data using the specified method.
    
    Args:
        df (pandas.DataFrame): Input data frame
        config: Application configuration
        
    Returns:
        pandas.DataFrame: Scaled data frame
    """
    try:
        # First convert prices to returns if needed
        if config.data_processor_returns_conversion_enabled:
            df_returns = data_processor.price_to_returns(df)
        else:
            df_returns = df
        
        # Then scale using the method specified in config
        return data_processor.scale_data(
            df=df_returns, 
            method=config.data_processor_scaling_method
        )
    except Exception as e:
        l.error(f"Error scaling data: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error scaling data: {str(e)}"
        )


def stationarize_data_step(df, config):
    """Make time series data stationary.
    
    Args:
        df (pandas.DataFrame): Input data frame
        config: Application configuration
        
    Returns:
        pandas.DataFrame: Stationary data frame
    """
    if config.data_processor_stationary_enabled:
        try:
            return data_processor.stationarize_data(
                df=df, 
                method=config.data_processor_stationary_method
            )
        except Exception as e:
            l.error(f"Error making data stationary: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error making data stationary: {str(e)}"
            )
    return df

def fill_missing_data_step(df: pd.DataFrame, config) -> pd.DataFrame:
    """Fill missing data in time series."""
    return data_processor.fill_data(
        df=df, 
        strategy=config.data_processor_missing_values_strategy
    ) if config.data_processor_missing_values_enabled else df


def test_stationarity_step(df: pd.DataFrame, config) -> Dict[str, Any]:
    """Test stationarity of time series data."""
    try:
        adf_results = data_processor.test_stationarity(
            df=df, 
            method=config.data_processor_stationarity_test_method
        )
        
        # Get first column results
        column = list(adf_results.keys())[0]
        result = adf_results[column]
        
        # Get critical values with fallback defaults
        critical_values = result.get("Critical Values", {
            "1%": -3.75,
            "5%": -3.0,
            "10%": -2.63
        })
        
        # Add interpretation
        interpretation_dict = interpret_stationarity_test(
            adf_results, 
            p_value_threshold=config.data_processor_stationarity_test_p_value_threshold
        )
        
        # Build response
        return {
            "adf_statistic": float(result["ADF Statistic"]),
            "p_value": float(result["p-value"]),
            "critical_values": critical_values,
            "is_stationary": float(result["p-value"]) < config.data_processor_stationarity_test_p_value_threshold,
            "interpretation": interpretation_dict.get(column, "No interpretation available")
        }
    except Exception as e:
        l.error(f"Error testing stationarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))
