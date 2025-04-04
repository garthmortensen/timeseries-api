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

from api.services.market_data_service import fetch_market_data



def generate_data_step(pipeline_input, config):
    """Generate or fetch time series data step.
    
    Args:
        pipeline_input: Pydantic model containing input parameters
        config: Application configuration
        
    Returns:
        pandas.DataFrame: Generated or fetched time series data
        
    Raises:
        HTTPException: If data generation or fetching fails
    """
    # Handle data source selection 
    if config.source_actual_or_synthetic_data == "synthetic":
        try:
            # Use configuration parameters for synthetic data
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
        except Exception as e:
            l.error(f"Error generating synthetic data: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating data: {str(e)}"
            )
    
    elif config.source_actual_or_synthetic_data == "actual":
        try:
            # Fetch market data using configuration parameters
            data_dict = fetch_market_data(
                symbols=config.symbols,
                start_date=config.data_start_date,
                end_date=config.data_end_date,
            )
            
            # Convert the market data dict to a DataFrame
            df = pd.DataFrame.from_dict(data_dict, orient='index')
            
            return df
        except Exception as e:
            l.error(f"Error fetching market data: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error fetching market data: {str(e)}"
            )
    
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid data source: {config.source_actual_or_synthetic_data}"
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
    