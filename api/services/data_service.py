#!/usr/bin/env python3
# # timeseries-pipeline/api/services/data_service.py
"""Data generation and transformation service functions.
This module contains functions for generating synthetic time series data and
"""

import logging as l
from fastapi import HTTPException
import pandas as pd

from generalized_timeseries import data_generator, data_processor
from .interpretations import interpret_stationarity_test


def generate_data_step(pipeline_input, config):
    """Generate synthetic data step.
    
    Args:
        pipeline_input: Pydantic model containing input parameters
        config: Application configuration
        
    Returns:
        pandas.DataFrame: Generated time series data
        
    Raises:
        HTTPException: If data generation is disabled or fails
    """
    if not config.data_generator_enabled:
        raise HTTPException(
            status_code=400,
            detail="Data generation is disabled in the configuration.",
        )
    
    try:
        # Build anchor_prices dictionary from flat config fields if not provided
        if not pipeline_input.anchor_prices:
            anchor_prices = {
                "GME": config.data_generator_anchor_prices_GME,
                "BYND": config.data_generator_anchor_prices_BYND,
                "BYD": config.data_generator_anchor_prices_BYD,
            }
        else:
            anchor_prices = pipeline_input.anchor_prices
            
        _, df = data_generator.generate_price_series(
            start_date=pipeline_input.start_date,
            end_date=pipeline_input.end_date,
            anchor_prices=anchor_prices,
        )
        return df
    except Exception as e:
        l.error(f"Error generating data: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating data: {str(e)}"
        )


def fill_missing_data_step(df, config):
    """Fill missing data in time series.
    
    Args:
        df (pandas.DataFrame): Input data frame
        config: Application configuration
        
    Returns:
        pandas.DataFrame: Data frame with filled missing values
    """
    if config.data_processor_missing_values_enabled:
        return data_processor.fill_data(
            df=df, 
            strategy=config.data_processor_missing_values_strategy
        )
    return df


def scale_data_step(df, config):
    """Scale time series data.
    
    Args:
        df (pandas.DataFrame): Input data frame
        config: Application configuration
        
    Returns:
        pandas.DataFrame: Scaled data frame
    """
    return data_processor.scale_data(
        df=df, 
        method=config.data_processor_scaling_method
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
        return data_processor.stationarize_data(
            df=df, 
            method=config.data_processor_stationary_method
        )
    return df


def test_stationarity_step(df, config):
    """Test stationarity of time series data.
    
    Args:
        df (pandas.DataFrame): Input data frame
        config: Application configuration
        
    Returns:
        dict: Stationarity test results with interpretation
    """
    try:
        adf_results = data_processor.test_stationarity(
            df=df, 
            method=config.data_processor_stationarity_test_method
        )
        
        # Log stationarity results
        data_processor.log_stationarity(
            adf_results=adf_results,
            p_value_threshold=config.data_processor_stationarity_test_p_value_threshold
        )
        
        # Get first column results (we need a single set of results for the response model)
        column = list(adf_results.keys())[0]
        result = adf_results[column]
        
        # Create a default critical values dictionary if missing
        # This handles the error with the 'Critical Values' key
        if "Critical Values" not in result:
            critical_values = {
                "1%": -3.75,  # Default values based on typical ADF test
                "5%": -3.0,
                "10%": -2.63
            }
        else:
            critical_values = result["Critical Values"]
            
        # Ensure critical values is a dict with string keys
        if not isinstance(critical_values, dict):
            critical_values = {
                "1%": -3.75,
                "5%": -3.0, 
                "10%": -2.63
            }
        
        # Add interpretation
        interpretation_dict = interpret_stationarity_test(
            adf_results, 
            p_value_threshold=config.data_processor_stationarity_test_p_value_threshold
        )
        
        # Build a response that matches the StationarityTestResponse model
        response = {
            "adf_statistic": float(result["ADF Statistic"]),
            "p_value": float(result["p-value"]),
            "critical_values": critical_values,
            "is_stationary": float(result["p-value"]) < config.data_processor_stationarity_test_p_value_threshold,
            "interpretation": interpretation_dict.get(column, "No interpretation available")
        }
        
        return response
    except Exception as e:
        l.error(f"Error testing stationarity: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error testing stationarity: {str(e)}"
        )
    