#!/usr/bin/env python3
# timeseries-api/api/graphql/resolvers.py
"""GraphQL resolvers using Graphene for the Timeseries API."""

import graphene
import logging as l
import json
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .types import (
    PipelineResultsType, PipelineInputType, MarketDataInputType, 
    StationarityTestInputType, ARIMAInputType, GARCHInputType, SpilloverInputType,
    TimeSeriesDataPointType, TimeSeriesDataPointInputType, StationarityResultsType, SpilloverAnalysisType,
    ARIMAModelType, GARCHModelType, StationarityTestType, SeriesStatsType,
    CriticalValuesType, ARIMAParametersType, ARIMAPValuesType, GARCHParametersType,
    DirectionalSpilloverType, PairwiseSpilloverType, SpilloverIndicesType,
    GrangerCausalityAnalysisType, GrangerCausalityResultType
)
from api.services.data_service import (
    generate_data_step, convert_to_returns_step, test_stationarity_step, scale_for_garch_step
)
from api.services.models_service import run_arima_step, run_garch_step
from api.services.spillover_service import analyze_spillover_step
from api.services.interpretations import interpret_arima_results, interpret_garch_results
from timeseries_compute.stats_model import calculate_stats
from utilities.configurator import load_configuration

config = load_configuration("config.yml")

def convert_dataframe_to_graphql_points(df, date_col='Date'):
    """Convert DataFrame to GraphQL TimeSeriesDataPoint format with proper typing."""
    if df is None or df.empty:
        return []
    
    # Ensure we have a proper DataFrame
    df_reset = df.copy()
    if date_col not in df_reset.columns:
        df_reset = df_reset.reset_index()
        if 'index' in df_reset.columns:
            df_reset = df_reset.rename(columns={'index': date_col})
        elif df.index.name:
            df_reset[date_col] = df_reset.index
    
    points = []
    for _, row in df_reset.iterrows():
        try:
            date = str(row[date_col]) if date_col in row else str(row.name)
            
            # Create properly typed data point
            point_data = {'date': date}
            
            # Map common column names to proper fields
            for col in row.index:
                if col != date_col and pd.notna(row[col]):
                    val = float(row[col]) if isinstance(row[col], (int, float)) else None
                    
                    # Map to proper field names
                    if col.lower() in ['open', 'high', 'low', 'close']:
                        point_data[col.lower()] = val
                    elif col.lower() == 'volume':
                        point_data['volume'] = int(val) if val is not None else None
                    elif 'return' in col.lower():
                        point_data['returns'] = val
                    elif 'scaled' in col.lower():
                        point_data['scaled'] = val
            
            points.append(TimeSeriesDataPointType(**point_data))
        except Exception as e:
            l.warning(f"Error converting row to GraphQL point: {e}")
            continue
    
    return points

def transform_stationarity_results(results: Dict[str, Any]) -> StationarityResultsType:
    """Transform stationarity results to properly typed GraphQL structure."""
    symbol_results = []
    series_stats = []
    
    all_symbols = results.get("all_symbols_stationarity", {})
    stats_data = results.get("series_stats", {})
    
    for symbol, test_data in all_symbols.items():
        # Create critical values object
        critical_vals = test_data.get("critical_values", {})
        critical_values = None
        if critical_vals:
            critical_values = CriticalValuesType(
                one_percent=critical_vals.get("1%"),
                five_percent=critical_vals.get("5%"),
                ten_percent=critical_vals.get("10%")
            )
        
        # Create stationarity test result
        test_result = StationarityTestType(
            is_stationary=test_data.get("is_stationary", False),
            adf_statistic=test_data.get("adf_statistic"),
            p_value=test_data.get("p_value"),
            critical_values=critical_values,
            interpretation=test_data.get("interpretation", "")
        )
        
        symbol_results.append(type("SymbolStationarityResult", (), {
            "symbol": symbol,
            "test_result": test_result
        })())
        
        # Add series stats if available
        if symbol in stats_data:
            stats = stats_data[symbol]
            series_stat = SeriesStatsType(
                mean=float(stats.get("mean", 0)),
                std=float(stats.get("std", 0)),
                skew=float(stats.get("skew", 0)),
                kurtosis=float(stats.get("kurtosis", 0)),
                min=float(stats.get("min", 0)),
                max=float(stats.get("max", 0)),
                median=float(stats.get("median", 0)),
                n=int(stats.get("n", 0)),
                annualized_vol=float(stats.get("annualized_vol", 0)) if stats.get("annualized_vol") else None
            )
            
            series_stats.append(type("SymbolSeriesStats", (), {
                "symbol": symbol,
                "stats": series_stat
            })())
    
    return StationarityResultsType(
        symbol_results=symbol_results,
        series_stats=series_stats
    )

def transform_arima_results(arima_data: Dict[str, Any]) -> List:
    """Transform ARIMA results to properly typed GraphQL structure."""
    arima_results = []
    
    all_symbols_arima = arima_data.get("all_symbols_arima", {})
    
    for symbol, result_data in all_symbols_arima.items():
        summary = result_data.get("summary", {})
        forecast = result_data.get("forecast", [])
        interpretation = result_data.get("interpretation", "")
        
        # Extract parameters from summary
        params = summary.get("params", {})
        p_values = summary.get("pvalues", {})
        
        parameters = ARIMAParametersType(
            ar_l1=params.get("ar.L1"),
            ar_l2=params.get("ar.L2"), 
            ar_l3=params.get("ar.L3"),
            ma_l1=params.get("ma.L1"),
            ma_l2=params.get("ma.L2"),
            ma_l3=params.get("ma.L3"),
            const=params.get("const"),
            sigma2=params.get("sigma2")
        )
        
        p_vals = ARIMAPValuesType(
            ar_l1=p_values.get("ar.L1"),
            ar_l2=p_values.get("ar.L2"),
            ar_l3=p_values.get("ar.L3"),
            ma_l1=p_values.get("ma.L1"),
            ma_l2=p_values.get("ma.L2"),
            ma_l3=p_values.get("ma.L3"),
            const=p_values.get("const")
        )
        
        arima_model = ARIMAModelType(
            fitted_model=str(summary.get("model", "")),
            parameters=parameters,
            p_values=p_vals,
            forecast=[float(f) for f in forecast],
            interpretation=interpretation,
            summary=str(summary),
            aic=summary.get("aic"),
            bic=summary.get("bic"),
            llf=summary.get("llf")
        )
        
        arima_results.append(type("SymbolARIMAResult", (), {
            "symbol": symbol,
            "result": arima_model
        })())
    
    return arima_results

def transform_garch_results(garch_data: Dict[str, Any]) -> List:
    """Transform GARCH results to properly typed GraphQL structure."""
    garch_results = []
    
    all_symbols_garch = garch_data.get("all_symbols_garch", {})
    
    for symbol, result_data in all_symbols_garch.items():
        summary = result_data.get("summary", {})
        forecast = result_data.get("forecast", [])
        interpretation = result_data.get("interpretation", "")
        
        # Extract parameters from summary
        params = summary.get("params", {})
        
        parameters = GARCHParametersType(
            omega=params.get("omega"),
            alpha_1=params.get("alpha[1]"),
            beta_1=params.get("beta[1]"),
            nu=params.get("nu")
        )
        
        garch_model = GARCHModelType(
            fitted_model=str(summary.get("model", "")),
            parameters=parameters,
            forecast=[float(f) for f in forecast],
            interpretation=interpretation,
            summary=str(summary),
            aic=summary.get("aic"),
            bic=summary.get("bic"),
            llf=summary.get("llf")
        )
        
        garch_results.append(type("SymbolGARCHResult", (), {
            "symbol": symbol,
            "result": garch_model
        })())
    
    return garch_results

def transform_spillover_results(spillover_data: Dict[str, Any]) -> SpilloverAnalysisType:
    """Transform spillover results to properly typed GraphQL structure."""
    # Directional spillovers
    directional_spillovers = []
    directional_data = spillover_data.get("directional_spillover", {})
    for asset, values in directional_data.items():
        if isinstance(values, dict):
            spillover = DirectionalSpilloverType(
                to_others=float(values.get("to", 0)),
                from_others=float(values.get("from", 0))
            )
            directional_spillovers.append(type("DirectionalSpilloverEntry", (), {
                "asset": asset,
                "spillover": spillover
            })())
    
    # Net spillovers
    net_spillovers = []
    net_data = spillover_data.get("net_spillover", {})
    for asset, value in net_data.items():
        net_spillovers.append(type("NetSpilloverEntry", (), {
            "asset": asset,
            "net_value": float(value)
        })())
    
    # Pairwise spillovers
    pairwise_spillovers = []
    pairwise_data = spillover_data.get("pairwise_spillover", {})
    for relationship, data in pairwise_data.items():
        if "_to_" in relationship:
            from_asset, to_asset = relationship.split("_to_")
            if isinstance(data, dict):
                pairwise_spillovers.append(PairwiseSpilloverType(
                    from_asset=from_asset,
                    to_asset=to_asset,
                    spillover_value=float(data.get("spillover_value", 0)),
                    r_squared=data.get("r_squared"),
                    significant_lags=data.get("significant_lags", [])
                ))
    
    # Enhanced spillover details
    spillover_indices = None
    if "spillover_indices" in spillover_data:
        indices_data = spillover_data["spillover_indices"]
        spillover_indices = SpilloverIndicesType(
            total_connectedness_index=float(indices_data.get("total_connectedness_index", {}).get("value", 0)),
            interpretation=indices_data.get("total_connectedness_index", {}).get("interpretation"),
            calculation_method=indices_data.get("total_connectedness_index", {}).get("calculation_method")
        )
    
    return SpilloverAnalysisType(
        total_spillover_index=float(spillover_data.get("total_spillover_index", 0)),
        directional_spillovers=directional_spillovers,
        net_spillovers=net_spillovers,
        pairwise_spillovers=pairwise_spillovers,
        spillover_indices=spillover_indices,
        interpretation=spillover_data.get("interpretation", "")
    )

def transform_granger_causality_results(granger_data: Dict[str, Any]) -> GrangerCausalityAnalysisType:
    """Transform Granger causality results to properly typed GraphQL structure."""
    causality_results = []
    interpretations = []
    
    causality_data = granger_data.get("causality_results", {})
    interp_data = granger_data.get("interpretations", {})
    
    for relationship, result in causality_data.items():
        causality_result = GrangerCausalityResultType(
            causality=result.get("causality", False),
            causality_1pct=result.get("causality_1pct"),
            causality_5pct=result.get("causality_5pct"),
            optimal_lag=result.get("optimal_lag"),
            optimal_lag_1pct=result.get("optimal_lag_1pct"),
            optimal_lag_5pct=result.get("optimal_lag_5pct"),
            min_p_value=result.get("min_p_value"),
            p_values=result.get("p_values", [])
        )
        
        causality_results.append(type("GrangerRelationship", (), {
            "relationship": relationship,
            "result": causality_result
        })())
        
        if relationship in interp_data:
            interpretations.append(type("GrangerInterpretation", (), {
                "relationship": relationship,
                "interpretation": interp_data[relationship]
            })())
    
    metadata = None
    if "metadata" in granger_data:
        meta = granger_data["metadata"]
        metadata = type("GrangerMetadata", (), {
            "max_lag": meta.get("max_lag", 5),
            "n_pairs_tested": meta.get("n_pairs_tested", 0),
            "significance_levels": meta.get("significance_levels", ["1%", "5%"])
        })()
    
    return GrangerCausalityAnalysisType(
        causality_results=causality_results,
        interpretations=interpretations,
        metadata=metadata
    )

def convert_input_points_to_dataframe(data_points):
    """Convert GraphQL input data points to DataFrame for processing."""
    data_dict = {}
    dates = []
    
    for point in data_points:
        dates.append(point.date)
        
        # Build data dictionary from available fields
        for field in ['open', 'high', 'low', 'close', 'volume', 'returns', 'scaled']:
            value = getattr(point, field, None)
            if value is not None:
                if field not in data_dict:
                    data_dict[field] = []
                data_dict[field].append(value)
    
    df = pd.DataFrame(data_dict, index=pd.to_datetime(dates))
    return df

class Query(graphene.ObjectType):
    """GraphQL Query root."""
    
    health = graphene.String()
    fetch_market_data = graphene.List(TimeSeriesDataPointType, input=graphene.Argument(MarketDataInputType, required=True))
    test_stationarity = graphene.Field(StationarityResultsType, input=graphene.Argument(StationarityTestInputType, required=True))
    
    def resolve_health(self, info):
        """Health check endpoint."""
        return "Timeseries GraphQL API is healthy"
    
    def resolve_fetch_market_data(self, info, input):
        """Fetch market data for given symbols and date range."""
        try:
            df = generate_data_step(
                source_type="actual_yfinance",
                start_date=input.start_date,
                end_date=input.end_date,
                symbols=input.symbols
            )
            return convert_dataframe_to_graphql_points(df)
        except Exception as e:
            l.error(f"Error fetching market data: {e}")
            raise Exception(f"Failed to fetch market data: {str(e)}")
    
    def resolve_test_stationarity(self, info, input):
        """Test stationarity of time series data."""
        try:
            # Convert GraphQL input to DataFrame using new helper function
            df = convert_input_points_to_dataframe(input.data)
            
            results = test_stationarity_step(df, "ADF", 0.05)
            return transform_stationarity_results(results)
            
        except Exception as e:
            l.error(f"Error in stationarity test: {e}")
            raise Exception(f"Stationarity test failed: {str(e)}")

class Mutation(graphene.ObjectType):
    """GraphQL Mutation root."""
    
    run_pipeline = graphene.Field(PipelineResultsType, input=graphene.Argument(PipelineInputType, required=True))
    run_arima_model = graphene.Field(ARIMAModelType, input=graphene.Argument(ARIMAInputType, required=True))
    run_garch_model = graphene.Field(GARCHModelType, input=graphene.Argument(GARCHInputType, required=True))
    run_spillover_analysis = graphene.Field(SpilloverAnalysisType, input=graphene.Argument(SpilloverInputType, required=True))
    
    def resolve_run_arima_model(self, info, input):
        """Run ARIMA model on provided data."""
        try:
            # Convert GraphQL input to DataFrame using new helper function
            df = convert_input_points_to_dataframe(input.data)
            
            summaries, forecasts, residuals = run_arima_step(
                df_stationary=df,
                p=input.p,
                d=input.d,
                q=input.q,
                forecast_steps=10
            )
            
            # Transform to properly typed result
            if summaries and len(summaries) > 0:
                symbol = list(summaries.keys())[0]
                summary = summaries[symbol]
                forecast = forecasts[symbol]
                interpretation = interpret_arima_results(summary, forecast)
                
                params = summary.get("params", {})
                p_values = summary.get("pvalues", {})
                
                parameters = ARIMAParametersType(
                    ar_l1=params.get("ar.L1"),
                    ar_l2=params.get("ar.L2"), 
                    ar_l3=params.get("ar.L3"),
                    ma_l1=params.get("ma.L1"),
                    ma_l2=params.get("ma.L2"),
                    ma_l3=params.get("ma.L3"),
                    const=params.get("const"),
                    sigma2=params.get("sigma2")
                )
                
                p_vals = ARIMAPValuesType(
                    ar_l1=p_values.get("ar.L1"),
                    ar_l2=p_values.get("ar.L2"),
                    ar_l3=p_values.get("ar.L3"),
                    ma_l1=p_values.get("ma.L1"),
                    ma_l2=p_values.get("ma.L2"),
                    ma_l3=p_values.get("ma.L3"),
                    const=p_values.get("const")
                )
                
                return ARIMAModelType(
                    fitted_model=str(summary.get("model", "")),
                    parameters=parameters,
                    p_values=p_vals,
                    forecast=[float(f) for f in forecast],
                    interpretation=interpretation,
                    summary=str(summary),
                    aic=summary.get("aic"),
                    bic=summary.get("bic"),
                    llf=summary.get("llf")
                )
            
        except Exception as e:
            l.error(f"ARIMA model error: {e}")
            raise Exception(f"ARIMA model failed: {str(e)}")
    
    def resolve_run_garch_model(self, info, input):
        """Run GARCH model on provided data."""
        try:
            # Convert GraphQL input to DataFrame using new helper function
            df = convert_input_points_to_dataframe(input.data)
            
            summaries, forecasts, cond_vol = run_garch_step(
                df_residuals=df,
                p=input.p,
                q=input.q,
                dist=input.dist,
                forecast_steps=10
            )
            
            # Transform to properly typed result
            if summaries and len(summaries) > 0:
                symbol = list(summaries.keys())[0]
                summary = summaries[symbol]
                forecast = forecasts[symbol]
                interpretation = interpret_garch_results(summary, forecast)
                
                params = summary.get("params", {})
                
                parameters = GARCHParametersType(
                    omega=params.get("omega"),
                    alpha_1=params.get("alpha[1]"),
                    beta_1=params.get("beta[1]"),
                    nu=params.get("nu")
                )
                
                return GARCHModelType(
                    fitted_model=str(summary.get("model", "")),
                    parameters=parameters,
                    forecast=[float(f) for f in forecast],
                    interpretation=interpretation,
                    summary=str(summary),
                    aic=summary.get("aic"),
                    bic=summary.get("bic"),
                    llf=summary.get("llf")
                )
            
        except Exception as e:
            l.error(f"GARCH model error: {e}")
            raise Exception(f"GARCH model failed: {str(e)}")
    
    def resolve_run_spillover_analysis(self, info, input):
        """Run spillover analysis on provided data."""
        try:
            # Convert GraphQL input to DataFrame using new helper function
            df = convert_input_points_to_dataframe(input.data)
            
            from api.models.input import SpilloverInput as PydanticSpilloverInput
            spillover_input = PydanticSpilloverInput(
                data=df.reset_index().to_dict('records'),
                method=input.method,
                forecast_horizon=input.forecast_horizon,
                window_size=input.window_size
            )
            
            spillover_results_dict = analyze_spillover_step(spillover_input)
            
            if spillover_results_dict:
                return transform_spillover_results(spillover_results_dict)
                
        except Exception as e:
            l.error(f"Spillover analysis error: {e}")
            raise Exception(f"Spillover analysis failed: {str(e)}")

    def resolve_run_pipeline(self, info, input):
        """Execute the complete time series analysis pipeline."""
        try:
            import time
            
            t1 = time.perf_counter()
            symbols = input.symbols or config.symbols
            
            # Generate/fetch data
            anchor_prices = None
            if input.source_actual_or_synthetic_data == "synthetic":
                synthetic_prices = input.synthetic_anchor_prices or config.synthetic_anchor_prices
                anchor_prices = dict(zip(symbols, synthetic_prices))
            
            df_prices = generate_data_step(
                source_type=input.source_actual_or_synthetic_data,
                start_date=input.data_start_date,
                end_date=input.data_end_date,
                symbols=symbols,
                anchor_prices=anchor_prices,
                random_seed=input.synthetic_random_seed if input.source_actual_or_synthetic_data == "synthetic" else None
            )
            
            # Convert to returns
            df_returns = convert_to_returns_step(df_prices)
            
            # Test stationarity
            stationarity_results = test_stationarity_step(df_returns, "ADF", 0.05)
            
            # Scale data
            df_scaled = scale_for_garch_step(df_returns)
            if 'Date' in df_scaled.columns:
                df_scaled = df_scaled.set_index('Date')
            
            # Run ARIMA
            all_arima_summaries, all_arima_forecasts, arima_residuals = run_arima_step(
                df_stationary=df_scaled,
                p=input.arima_params.p,
                d=input.arima_params.d, 
                q=input.arima_params.q,
                forecast_steps=config.stats_model_ARIMA_predict_steps
            )
            
            # Generate ARIMA interpretations
            all_arima_interpretations = {}
            for symbol in all_arima_summaries.keys():
                all_arima_interpretations[symbol] = interpret_arima_results(
                    all_arima_summaries[symbol], 
                    all_arima_forecasts[symbol]
                )
            
            # Run GARCH
            all_garch_summaries, all_garch_forecasts, cond_vol = run_garch_step(
                df_residuals=arima_residuals,
                p=input.garch_params.p,
                q=input.garch_params.q,
                dist=input.garch_params.dist,
                forecast_steps=config.stats_model_GARCH_predict_steps
            )
            
            # Generate GARCH interpretations
            all_garch_interpretations = {}
            for symbol in all_garch_summaries.keys():
                all_garch_interpretations[symbol] = interpret_garch_results(
                    all_garch_summaries[symbol], 
                    all_garch_forecasts[symbol]
                )
            
            # Handle spillover analysis
            spillover_results = None
            granger_results = None
            if input.spillover_enabled:
                from api.models.input import SpilloverInput as PydanticSpilloverInput
                spillover_input = PydanticSpilloverInput(
                    data=df_returns.reset_index().to_dict('records'),
                    method=input.spillover_params.method,
                    forecast_horizon=input.spillover_params.forecast_horizon,
                    window_size=input.spillover_params.window_size
                )
                spillover_results_dict = analyze_spillover_step(spillover_input)
                
                if spillover_results_dict:
                    spillover_results = transform_spillover_results(spillover_results_dict)
                    
                    # Transform Granger causality results if present
                    if "granger_causality_results" in spillover_results_dict:
                        granger_results = transform_granger_causality_results(
                            spillover_results_dict["granger_causality_results"]
                        )
            
            # Transform results to properly typed GraphQL structures
            arima_data = {
                "all_symbols_arima": {
                    symbol: {
                        "summary": all_arima_summaries[symbol],
                        "forecast": all_arima_forecasts[symbol],
                        "interpretation": all_arima_interpretations[symbol]
                    }
                    for symbol in all_arima_summaries.keys()
                }
            }
            
            garch_data = {
                "all_symbols_garch": {
                    symbol: {
                        "summary": all_garch_summaries[symbol], 
                        "forecast": all_garch_forecasts[symbol],
                        "interpretation": all_garch_interpretations[symbol]
                    }
                    for symbol in all_garch_summaries.keys()
                }
            }
            
            return PipelineResultsType(
                original_data=convert_dataframe_to_graphql_points(df_prices),
                returns_data=convert_dataframe_to_graphql_points(df_returns),
                scaled_data=convert_dataframe_to_graphql_points(df_scaled) if df_scaled is not None else None,
                pre_garch_data=convert_dataframe_to_graphql_points(arima_residuals),
                post_garch_data=convert_dataframe_to_graphql_points(cond_vol) if cond_vol is not None else None,
                stationarity_results=transform_stationarity_results(stationarity_results),
                arima_results=transform_arima_results(arima_data),
                garch_results=transform_garch_results(garch_data),
                spillover_results=spillover_results,
                granger_causality_results=granger_results
            )
            
        except Exception as e:
            l.error(f"GraphQL pipeline error: {e}")
            raise Exception(f"Pipeline execution failed: {str(e)}")

# Create the GraphQL schema
schema = graphene.Schema(query=Query, mutation=Mutation)