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
    if not config.data_generator.enabled:
        raise HTTPException(
            status_code=400,
            detail="Data generation is disabled in the configuration.",
        )
    
    try:
        _, df = data_generator.generate_price_series(
            start_date=pipeline_input.start_date,
            end_date=pipeline_input.end_date,
            anchor_prices=pipeline_input.anchor_prices,
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
    if config.data_processor.handle_missing_values.enabled:
        return data_processor.fill_data(
            df=df, 
            strategy=config.data_processor.handle_missing_values.strategy
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
    if config.data_processor.scaling.enabled:
        return data_processor.scale_data(
            df=df, 
            method=config.data_processor.scaling.method
        )
    return df


def stationarize_data_step(df, config):
    """Make time series data stationary.
    
    Args:
        df (pandas.DataFrame): Input data frame
        config: Application configuration
        
    Returns:
        pandas.DataFrame: Stationary data frame
    """
    if config.data_processor.make_stationary.enabled:
        return data_processor.stationarize_data(
            df=df, 
            method=config.data_processor.make_stationary.method
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
        stationarity_results = data_processor.test_stationarity(
            df=df, 
            method=config.data_processor.test_stationarity.method
        )
        
        # Log stationarity results
        data_processor.log_stationarity(
            adf_results=stationarity_results,
            p_value_threshold=config.data_processor.test_stationarity.p_value_threshold
        )
        
        # Add interpretation
        interpretation = interpret_stationarity_test(
            stationarity_results, 
            p_value_threshold=config.data_processor.test_stationarity.p_value_threshold
        )
        
        # Add interpretation to results
        stationarity_results["interpretation"] = interpretation
        
        return stationarity_results
    except Exception as e:
        l.error(f"Error testing stationarity: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error testing stationarity: {str(e)}"
        )