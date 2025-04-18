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
from api.services.market_data_service import fetch_market_data


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
        elif source_type == "actual":
            _, df = fetch_market_data(
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

def scale_for_garch_step(df: pd.DataFrame) -> pd.DataFrame:
    """Scale time series data for GARCH modeling."""
    return data_processor.scale_for_garch(df)

def test_stationarity_step(df: pd.DataFrame, test_method: str, 
                          p_value_threshold: float) -> Dict[str, Any]:
    """Test stationarity of time series data."""
    try:
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
