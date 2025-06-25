#!/usr/bin/env python3
# timeseries-api/api/graphql/types.py
"""GraphQL types using Graphene for the Timeseries API."""

import graphene
from graphene import ObjectType, String, Float, Int, Boolean, List, Field
import json

# ============================================================================
# Basic Data Types
# ============================================================================

class TimeSeriesDataPointType(ObjectType):
    """A single data point in a time series."""
    date = String(required=True)
    # Replace JSONString with proper typed fields based on actual data structure
    open = Float()
    high = Float()
    low = Float()
    close = Float()
    volume = Int()
    returns = Float()
    scaled = Float()

class CriticalValuesType(ObjectType):
    """Critical values for statistical tests."""
    one_percent = Float()
    five_percent = Float() 
    ten_percent = Float()

class StationarityTestType(ObjectType):
    """Stationarity test results for a single symbol."""
    is_stationary = Boolean(required=True)
    adf_statistic = Float()
    p_value = Float()
    critical_values = Field(CriticalValuesType)
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

# ============================================================================
# ARIMA Model Types
# ============================================================================

class ARIMAParametersType(ObjectType):
    """ARIMA model parameters."""
    ar_l1 = Float()
    ar_l2 = Float()
    ar_l3 = Float()
    ma_l1 = Float()
    ma_l2 = Float()
    ma_l3 = Float()
    const = Float()
    sigma2 = Float()

class ARIMAPValuesType(ObjectType):
    """ARIMA model p-values."""
    ar_l1 = Float()
    ar_l2 = Float()
    ar_l3 = Float()
    ma_l1 = Float()
    ma_l2 = Float()
    ma_l3 = Float()
    const = Float()

class ARIMAModelType(ObjectType):
    """ARIMA model results for a single symbol."""
    fitted_model = String(required=True)
    parameters = Field(ARIMAParametersType, required=True)
    p_values = Field(ARIMAPValuesType, required=True)
    forecast = List(Float, required=True)
    interpretation = String(required=True)
    summary = String(required=True)
    aic = Float()
    bic = Float()
    llf = Float()

# ============================================================================
# GARCH Model Types
# ============================================================================

class GARCHParametersType(ObjectType):
    """GARCH model parameters."""
    omega = Float()
    alpha_1 = Float()
    beta_1 = Float()
    nu = Float()  # degrees of freedom for t-distribution

class GARCHModelType(ObjectType):
    """GARCH model results for a single symbol."""
    fitted_model = String(required=True)
    parameters = Field(GARCHParametersType)
    forecast = List(Float, required=True)
    interpretation = String(required=True)
    summary = String(required=True)
    aic = Float()
    bic = Float()
    llf = Float()

# ============================================================================
# Spillover Analysis Types
# ============================================================================

class DirectionalSpilloverType(ObjectType):
    """Directional spillover for a single asset."""
    to_others = Float(required=True)
    from_others = Float(required=True)

class PairwiseSpilloverType(ObjectType):
    """Pairwise spillover relationship."""
    from_asset = String(required=True)
    to_asset = String(required=True)
    spillover_value = Float(required=True)
    r_squared = Float()
    significant_lags = List(Int)

class SpilloverIndicesType(ObjectType):
    """Comprehensive spillover indices."""
    total_connectedness_index = Float(required=True)
    interpretation = String()
    calculation_method = String()

class FEVDAnalysisType(ObjectType):
    """FEVD (Forecast Error Variance Decomposition) analysis."""
    fevd_horizon = Int(required=True)
    fevd_normalized = Boolean(required=True)
    fevd_row_sums = List(Float, required=True)

class VARModelDetailsType(ObjectType):
    """VAR model details from spillover analysis."""
    var_fitted_successfully = Boolean(required=True)
    var_lag_order = Int()
    var_ic_used = String()
    var_stability_check = Boolean()

class GrangerCausalityResultType(ObjectType):
    """Granger causality test result for a specific relationship."""
    causality = Boolean(required=True)
    causality_1pct = Boolean()
    causality_5pct = Boolean()
    optimal_lag = Int()
    optimal_lag_1pct = Int()
    optimal_lag_5pct = Int()
    min_p_value = Float()
    p_values = List(Float)

class MethodologyParamsType(ObjectType):
    """Methodology parameters used in spillover analysis."""
    spillover_method = String(required=True)
    var_lag_selection_criterion = String()
    max_lags_considered = Int()
    selected_lag_order = Int()
    forecast_horizon = Int()

class DirectionalSpilloverEntry(ObjectType):
    """Entry for directional spillover mapping."""
    asset = String(required=True)
    spillover = Field(DirectionalSpilloverType, required=True)

class NetSpilloverEntry(ObjectType):
    """Entry for net spillover mapping."""
    asset = String(required=True)
    net_value = Float(required=True)

class SpilloverAnalysisType(ObjectType):
    """Complete spillover analysis results with proper typing."""
    total_spillover_index = Float(required=True)
    directional_spillovers = List(DirectionalSpilloverEntry)
    net_spillovers = List(NetSpilloverEntry)
    pairwise_spillovers = List(PairwiseSpilloverType)
    spillover_indices = Field(SpilloverIndicesType)
    fevd_analysis = Field(FEVDAnalysisType)
    var_model_details = Field(VARModelDetailsType)
    methodology = Field(MethodologyParamsType)
    interpretation = String(required=True)

# ============================================================================
# Granger Causality Types  
# ============================================================================

class GrangerRelationship(ObjectType):
    """Granger causality relationship entry."""
    relationship = String(required=True)  # e.g., "AAPL->MSFT"
    result = Field(GrangerCausalityResultType, required=True)

class GrangerInterpretation(ObjectType):
    """Granger causality interpretation entry."""
    relationship = String(required=True)
    interpretation = String(required=True)

class GrangerMetadata(ObjectType):
    """Granger causality analysis metadata."""
    max_lag = Int(required=True)
    n_pairs_tested = Int(required=True)
    significance_levels = List(String, required=True)

class GrangerCausalityAnalysisType(ObjectType):
    """Complete Granger causality analysis results."""
    causality_results = List(GrangerRelationship, required=True)
    interpretations = List(GrangerInterpretation)
    metadata = Field(GrangerMetadata)

# ============================================================================
# Stationarity Results Types
# ============================================================================

class SymbolStationarityResult(ObjectType):
    """Symbol-specific stationarity result."""
    symbol = String(required=True)
    test_result = Field(StationarityTestType, required=True)

class SymbolSeriesStats(ObjectType):
    """Symbol-specific series statistics."""
    symbol = String(required=True)
    stats = Field(SeriesStatsType, required=True)

class StationarityResultsType(ObjectType):
    """Complete stationarity test results with proper typing."""
    symbol_results = List(SymbolStationarityResult, required=True)
    series_stats = List(SymbolSeriesStats)

# ============================================================================
# Model Results by Symbol
# ============================================================================

class SymbolARIMAResult(ObjectType):
    """Symbol-specific ARIMA result."""
    symbol = String(required=True)
    result = Field(ARIMAModelType, required=True)

class SymbolGARCHResult(ObjectType):
    """Symbol-specific GARCH result."""
    symbol = String(required=True)
    result = Field(GARCHModelType, required=True)

# ============================================================================
# Pipeline Results Type
# ============================================================================

class PipelineResultsType(ObjectType):
    """Complete pipeline analysis results with proper typing throughout."""
    original_data = List(TimeSeriesDataPointType, required=True)
    returns_data = List(TimeSeriesDataPointType, required=True)
    scaled_data = List(TimeSeriesDataPointType)
    pre_garch_data = List(TimeSeriesDataPointType, required=True)
    post_garch_data = List(TimeSeriesDataPointType)
    
    stationarity_results = Field(StationarityResultsType, required=True)
    
    # Replace JSONString with proper typed results
    arima_results = List(SymbolARIMAResult, required=True)
    garch_results = List(SymbolGARCHResult, required=True)
    
    spillover_results = Field(SpilloverAnalysisType)
    granger_causality_results = Field(GrangerCausalityAnalysisType)

# ============================================================================
# Input Types (with proper parameter typing)
# ============================================================================

class TimeSeriesDataPointInputType(graphene.InputObjectType):
    """Input type for a single data point in a time series."""
    date = String(required=True)
    open = Float()
    high = Float()
    low = Float()
    close = Float()
    volume = Int()
    returns = Float()
    scaled = Float()

class ARIMAParamsInputType(graphene.InputObjectType):
    """ARIMA model parameters input."""
    p = Int(required=True)
    d = Int(required=True) 
    q = Int(required=True)
    include_const = Boolean(default_value=True)

class GARCHParamsInputType(graphene.InputObjectType):
    """GARCH model parameters input."""
    p = Int(required=True)
    q = Int(required=True)
    dist = String(default_value="t")
    mean = String(default_value="Zero")

class SpilloverParamsInputType(graphene.InputObjectType):
    """Spillover analysis parameters input."""
    method = String(default_value="diebold_yilmaz")
    forecast_horizon = Int(default_value=10)
    window_size = Int()
    var_lag_selection_method = String(default_value="aic")

class PipelineInputType(graphene.InputObjectType):
    """Input for pipeline analysis with proper parameter typing."""
    source_actual_or_synthetic_data = String(default_value="synthetic")
    data_start_date = String(required=True)
    data_end_date = String(required=True)
    symbols = List(String)
    synthetic_anchor_prices = List(Float)
    synthetic_random_seed = Int()
    scaling_method = String(default_value="standard")
    
    # Replace JSONString with proper typed parameters
    arima_params = Field(ARIMAParamsInputType, required=True)
    garch_params = Field(GARCHParamsInputType, required=True)
    spillover_enabled = Boolean(default_value=False)
    spillover_params = Field(SpilloverParamsInputType, required=True)

class MarketDataInputType(graphene.InputObjectType):
    """Input for market data fetching."""
    symbols = List(String, required=True)
    start_date = String(required=True)
    end_date = String(required=True)
    interval = String(default_value="1d")

class StationarityTestInputType(graphene.InputObjectType):
    """Input for stationarity testing."""
    # Use the proper input type instead of output type
    data = List(TimeSeriesDataPointInputType, required=True)

class ARIMAInputType(graphene.InputObjectType):
    """Input for ARIMA modeling."""
    p = Int(required=True)
    d = Int(required=True)
    q = Int(required=True)
    data = List(TimeSeriesDataPointInputType, required=True)

class GARCHInputType(graphene.InputObjectType):
    """Input for GARCH modeling."""
    p = Int(required=True)
    q = Int(required=True)
    data = List(TimeSeriesDataPointInputType, required=True)
    dist = String(default_value="t")

class SpilloverInputType(graphene.InputObjectType):
    """Input for spillover analysis."""
    data = List(TimeSeriesDataPointInputType, required=True)
    method = String(default_value="diebold_yilmaz")
    forecast_horizon = Int(default_value=10)
    window_size = Int()