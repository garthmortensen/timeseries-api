#!/usr/bin/env python3
# timeseries-api/api/services/data_service.py
"""Data generation and transformation service functions.
This module contains functions for generating synthetic time series data and pulling actual data.
"""

import logging as l
import pandas as pd
from fastapi import HTTPException
from typing import Dict, Any, Optional, List

from timeseries_compute import data_generator, data_processor
from .interpretations import interpret_stationarity_test
from api.services.market_data_service import fetch_market_data_yfinance, fetch_market_data_stooq


def generate_data_step(source_type: str, start_date: str, end_date: str, 
                      symbols: List[str], anchor_prices: Optional[Dict[str, float]]=None, 
                      random_seed: Optional[int]=None) -> pd.DataFrame:
    """Generate or fetch time series data based on parameters."""
    try:
        if source_type == "synthetic":
            _, df = data_generator.generate_price_series(
                start_date=start_date,
                end_date=end_date,
                anchor_prices=anchor_prices,
                random_seed=random_seed
            )
            return df
        elif source_type == "actual_yfinance":
            _, df = fetch_market_data_yfinance(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
            return df
        elif source_type == "actual_stooq":
            _, df = fetch_market_data_stooq(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
            return df
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data source: {source_type}"
            )
    except Exception as e:
        l.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_to_returns_step(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price data to log returns.
    
    BEST PRACTICE: Academic research consistently shows that financial price series should
    be transformed to returns before modeling. Log returns specifically are preferred as they:
    1. Are approximately normally distributed for small changes
    2. Can be interpreted as continuously compounded returns
    3. Are additive over time, making multi-period analysis more straightforward
    4. Help achieve the stationarity required for time series modeling
    """
    return data_processor.price_to_returns(df)

def fill_missing_data_step(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Fill missing values in time series data.
    
    Args:
        df (pandas.DataFrame): Input data frame with potentially missing values
        strategy (str, optional): Strategy to handle missing values.
            Options: 'drop', 'forward_fill', 'backward_fill', 'interpolate'.
            Defaults to "drop".
        
    Returns:
        pandas.DataFrame: Data frame with missing values addressed
    """
    try:
        if strategy == "drop":
            return df.dropna()
        elif strategy == "forward_fill" or strategy == "ffill":
            return df.ffill()
        elif strategy == "backward_fill" or strategy == "bfill":
            return df.bfill()
        elif strategy == "interpolate":
            return df.interpolate()
        else:
            l.warning(f"Unknown missing value strategy '{strategy}', using 'drop' instead")
            return df.dropna()
    except Exception as e:
        l.error(f"Error filling missing data: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error filling missing data: {str(e)}"
        )

def scale_data_step(df: pd.DataFrame, method: str = "standardize", convert_to_returns: bool = True) -> pd.DataFrame:
    """Scale time series data using the specified method.
    
    Args:
        df (pandas.DataFrame): Input data frame
        method (str, optional): Scaling method. Options: 'standardize', 'minmax'. 
            Defaults to "standardize".
        convert_to_returns (bool, optional): Whether to convert prices to returns first.
            Defaults to True.
        
    Returns:
        pandas.DataFrame: Scaled data frame
    """
    try:
        # First convert prices to returns if needed
        if convert_to_returns:
            df_returns = data_processor.price_to_returns(df)
        else:
            df_returns = df
        
        # Then scale using the specified method
        return data_processor.scale_data(
            df=df_returns, 
            method=method
        )
    except Exception as e:
        l.error(f"Error scaling data: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error scaling data: {str(e)}"
        )

def scale_for_garch_step(df: pd.DataFrame) -> pd.DataFrame:
    """Scale time series data for GARCH modeling."""
    return data_processor.scale_for_garch(df)

def stationarize_data_step(df: pd.DataFrame, method: str = "difference", enabled: bool = True) -> pd.DataFrame:
    """Make time series data stationary.
    
    Args:
        df (pandas.DataFrame): Input data frame
        method (str, optional): Method for making data stationary.
            Options: 'difference', 'log', 'percentage_change'.
            Defaults to "difference".
        enabled (bool, optional): Whether to perform stationarization.
            Defaults to True.
        
    Returns:
        pandas.DataFrame: Stationary data frame
    """
    if enabled:
        try:
            return data_processor.stationarize_data(
                df=df, 
                method=method
            )
        except Exception as e:
            l.error(f"Error making data stationary: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error making data stationary: {str(e)}"
            )
    return df

def stationarize_data_step(df: pd.DataFrame, method: str = "difference", enabled: bool = True) -> pd.DataFrame:
    """Make time series data stationary.
    
    Args:
        df (pandas.DataFrame): Input data frame
        method (str, optional): Method for making data stationary.
            Options: 'difference', 'log', 'percentage_change'.
            Defaults to "difference".
        enabled (bool, optional): Whether to perform stationarization.
            Defaults to True.
        
    Returns:
        pandas.DataFrame: Stationary data frame
    """
    if enabled:
        try:
            return data_processor.stationarize_data(
                df=df, 
                method=method
            )
        except Exception as e:
            l.error(f"Error making data stationary: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error making data stationary: {str(e)}"
            )
    return df

def test_stationarity_step(df: pd.DataFrame, test_method: str, 
                          p_value_threshold: float) -> Dict[str, Any]:
    """Test stationarity of time series data."""
    try:
        # Ensure Date is set as index before passing to the library
        if 'Date' in df.columns:
            df = df.set_index('Date')
            
        adf_results = data_processor.test_stationarity(
            df=df, 
            method=test_method
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
            p_value_threshold=p_value_threshold
        )
        
        # Build response
        return {
            "adf_statistic": float(result["ADF Statistic"]),
            "p_value": float(result["p-value"]),
            "critical_values": critical_values,
            "is_stationary": float(result["p-value"]) < p_value_threshold,
            "interpretation": interpretation_dict.get(column, "No interpretation available")
        }
    except Exception as e:
        l.error(f"Error testing stationarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))
