#!/usr/bin/env python3
# timeseries-api/api/graphql/types.py
"""GraphQL types using Graphene for the Timeseries API."""

import graphene
from graphene import ObjectType, String, Float, Int, Boolean, List, Field
import json

class TimeSeriesDataPointType(ObjectType):
    """A single data point in a time series."""
    date = String(required=True)
    values = graphene.JSONString()

class StationarityTestType(ObjectType):
    """Stationarity test results for a single symbol."""
    is_stationary = Boolean(required=True)
    adf_statistic = Float()
    p_value = Float()
    critical_values = graphene.JSONString()
    interpretation = String(required=True)

class SeriesStatsType(ObjectType):
    """Statistical measures for a time series."""
    mean = Float(required=True)
    std = Float(required=True)
    skew = Float(required=True)
    kurtosis = Float(required=True)
    min = Float(required=True)
    max = Float(required=True)
    median = Float(required=True)
    n = Int(required=True)
    annualized_vol = Float()

class StationarityResultsType(ObjectType):
    """Complete stationarity test results."""
    all_symbols_stationarity = graphene.JSONString(required=True)
    series_stats = graphene.JSONString()

class ARIMAModelType(ObjectType):
    """ARIMA model results for a single symbol."""
    fitted_model = String(required=True)
    parameters = graphene.JSONString(required=True)
    p_values = graphene.JSONString(required=True)
    forecast = List(Float, required=True)
    interpretation = String(required=True)
    summary = String(required=True)

class GARCHModelType(ObjectType):
    """GARCH model results for a single symbol."""
    fitted_model = String(required=True)
    forecast = List(Float, required=True)
    interpretation = String(required=True)
    summary = String(required=True)

class SpilloverAnalysisType(ObjectType):
    """Spillover analysis results."""
    total_spillover_index = Float(required=True)
    directional_spillover = graphene.JSONString(required=True)
    net_spillover = graphene.JSONString(required=True)
    pairwise_spillover = graphene.JSONString(required=True)
    interpretation = String(required=True)

class PipelineResultsType(ObjectType):
    """Complete pipeline analysis results."""
    original_data = List(TimeSeriesDataPointType, required=True)
    returns_data = List(TimeSeriesDataPointType, required=True)
    scaled_data = List(TimeSeriesDataPointType)
    pre_garch_data = List(TimeSeriesDataPointType, required=True)
    post_garch_data = List(TimeSeriesDataPointType)
    stationarity_results = Field(StationarityResultsType, required=True)
    series_stats = graphene.JSONString()
    arima_results = graphene.JSONString(required=True)
    garch_results = graphene.JSONString(required=True)
    spillover_results = Field(SpilloverAnalysisType)
    granger_causality_results = graphene.JSONString()

# Input types
class PipelineInputType(graphene.InputObjectType):
    """Input for pipeline analysis."""
    source_actual_or_synthetic_data = String(default_value="synthetic")
    data_start_date = String(required=True)
    data_end_date = String(required=True)
    symbols = List(String)
    synthetic_anchor_prices = List(Float)
    synthetic_random_seed = Int()
    scaling_method = String(default_value="standard")
    arima_params = graphene.JSONString(required=True)
    garch_params = graphene.JSONString(required=True)
    spillover_enabled = Boolean(default_value=False)
    spillover_params = graphene.JSONString(required=True)

class MarketDataInputType(graphene.InputObjectType):
    """Input for market data fetching."""
    symbols = List(String, required=True)
    start_date = String(required=True)
    end_date = String(required=True)
    interval = String(default_value="1d")

class StationarityTestInputType(graphene.InputObjectType):
    """Input for stationarity testing."""
    data = graphene.JSONString(required=True)

class ARIMAInputType(graphene.InputObjectType):
    """Input for ARIMA modeling."""
    p = Int(required=True)
    d = Int(required=True)
    q = Int(required=True)
    data = graphene.JSONString(required=True)

class GARCHInputType(graphene.InputObjectType):
    """Input for GARCH modeling."""
    p = Int(required=True)
    q = Int(required=True)
    data = graphene.JSONString(required=True)
    dist = String(default_value="t")

class SpilloverInputType(graphene.InputObjectType):
    """Input for spillover analysis."""
    data = graphene.JSONString(required=True)
    method = String(default_value="diebold_yilmaz")
    forecast_horizon = Int(default_value=10)
    window_size = Int()