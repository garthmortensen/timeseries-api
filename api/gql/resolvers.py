#!/usr/bin/env python3
# timeseries-api/api/graphql/resolvers.py
"""GraphQL resolvers using Graphene for the Timeseries API."""

import graphene
import logging as l
import json
from typing import List, Optional
import pandas as pd

from .types import (
    PipelineResultsType, PipelineInputType, MarketDataInputType, 
    StationarityTestInputType, ARIMAInputType, GARCHInputType, SpilloverInputType,
    TimeSeriesDataPointType, StationarityResultsType, SpilloverAnalysisType
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
    """Convert DataFrame to GraphQL TimeSeriesDataPoint format."""
    if df is None or df.empty:
        return []
    
    # Ensure we have a proper DataFrame
    if date_col in df.columns:
        df_reset = df.copy()
    else:
        df_reset = df.reset_index()
        if 'index' in df_reset.columns:
            df_reset = df_reset.rename(columns={'index': date_col})
        elif df.index.name:
            df_reset[date_col] = df_reset.index
    
    points = []
    for _, row in df_reset.iterrows():
        try:
            date = str(row[date_col]) if date_col in row else str(row.name)
            values = {}
            for col in row.index:
                if col != date_col:
                    val = row[col]
                    if pd.isna(val):
                        values[col] = None
                    elif isinstance(val, (int, float)):
                        values[col] = float(val)
                    else:
                        values[col] = str(val)
            
            points.append(TimeSeriesDataPointType(date=date, values=json.dumps(values)))
        except Exception as e:
            l.warning(f"Error converting row to GraphQL point: {e}")
            continue
    
    return points

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
            data = json.loads(input.data)
            df = pd.DataFrame(data)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            
            results = test_stationarity_step(df, "ADF", 0.05)
            
            return StationarityResultsType(
                all_symbols_stationarity=json.dumps(results.get("all_symbols_stationarity", {})),
                series_stats=json.dumps(results.get("series_stats")) if results.get("series_stats") else None
            )
        except Exception as e:
            l.error(f"Error in stationarity test: {e}")
            raise Exception(f"Stationarity test failed: {str(e)}")

class Mutation(graphene.ObjectType):
    """GraphQL Mutation root."""
    
    run_pipeline = graphene.Field(PipelineResultsType, input=graphene.Argument(PipelineInputType, required=True))
    run_arima_model = graphene.JSONString(input=graphene.Argument(ARIMAInputType, required=True))
    run_garch_model = graphene.JSONString(input=graphene.Argument(GARCHInputType, required=True))
    
    def resolve_run_pipeline(self, info, input):
        """Execute the complete time series analysis pipeline."""
        try:
            import time
            
            t1 = time.perf_counter()
            symbols = input.symbols or config.symbols
            
            # Parse JSON inputs safely - handle both string and dict inputs
            def safe_json_parse(value, param_name):
                if isinstance(value, dict):
                    return value
                elif isinstance(value, str):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError as e:
                        l.error(f"Invalid JSON in {param_name}: {e}")
                        raise Exception(f"Invalid JSON format in {param_name}")
                else:
                    raise Exception(f"{param_name} must be a JSON string or dictionary")
            
            arima_params = safe_json_parse(input.arima_params, "arima_params")
            garch_params = safe_json_parse(input.garch_params, "garch_params") 
            spillover_params = safe_json_parse(input.spillover_params, "spillover_params")
            
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
            
            # Calculate series stats with proper serialization
            series_stats = {}
            for column in df_returns.columns:
                if column != 'Date':
                    try:
                        stats = calculate_stats(df_returns[column])
                        # Convert numpy types to native Python types for JSON serialization
                        if isinstance(stats, dict):
                            series_stats[column] = {k: float(v) if pd.notna(v) and isinstance(v, (int, float)) else str(v) for k, v in stats.items()}
                        else:
                            series_stats[column] = str(stats)
                    except Exception as e:
                        l.warning(f"Error calculating stats for {column}: {e}")
                        series_stats[column] = {"error": str(e)}
            
            # Scale data
            df_scaled = scale_for_garch_step(df_returns)
            if 'Date' in df_scaled.columns:
                df_scaled = df_scaled.set_index('Date')
            
            # Run ARIMA
            all_arima_summaries, all_arima_forecasts, arima_residuals = run_arima_step(
                df_stationary=df_scaled,
                p=arima_params.get('p', 1),
                d=arima_params.get('d', 1), 
                q=arima_params.get('q', 1),
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
                p=garch_params.get('p', 1),
                q=garch_params.get('q', 1),
                dist=garch_params.get('dist', 'normal'),
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
            if input.spillover_enabled:
                from api.models.input import SpilloverInput as PydanticSpilloverInput
                spillover_input = PydanticSpilloverInput(
                    data=df_returns.reset_index().to_dict('records'),
                    method=spillover_params.get('method', 'diebold_yilmaz'),
                    forecast_horizon=spillover_params.get('forecast_horizon', 10),
                    window_size=spillover_params.get('window_size')
                )
                spillover_results_dict = analyze_spillover_step(spillover_input)
                
                if spillover_results_dict:
                    spillover_results = SpilloverAnalysisType(
                        total_spillover_index=spillover_results_dict.get('total_spillover_index', 0.0),
                        directional_spillover=json.dumps(spillover_results_dict.get('directional_spillover', {})),
                        net_spillover=json.dumps(spillover_results_dict.get('net_spillover', {})),
                        pairwise_spillover=json.dumps(spillover_results_dict.get('pairwise_spillover', {})),
                        interpretation=spillover_results_dict.get('interpretation', '')
                    )
            
            # Convert results to GraphQL format
            return PipelineResultsType(
                original_data=convert_dataframe_to_graphql_points(df_prices),
                returns_data=convert_dataframe_to_graphql_points(df_returns),
                scaled_data=convert_dataframe_to_graphql_points(df_scaled) if df_scaled is not None else None,
                pre_garch_data=convert_dataframe_to_graphql_points(arima_residuals),
                post_garch_data=convert_dataframe_to_graphql_points(cond_vol) if cond_vol is not None else None,
                stationarity_results=StationarityResultsType(
                    all_symbols_stationarity=json.dumps(stationarity_results.get("all_symbols_stationarity", {})),
                    series_stats=json.dumps(series_stats)
                ),
                series_stats=json.dumps(series_stats),
                arima_results=json.dumps({
                    "all_symbols_arima": {
                        symbol: {
                            "summary": all_arima_summaries[symbol],
                            "forecast": all_arima_forecasts[symbol],
                            "interpretation": all_arima_interpretations[symbol]
                        }
                        for symbol in all_arima_summaries.keys()
                    }
                }),
                garch_results=json.dumps({
                    "all_symbols_garch": {
                        symbol: {
                            "summary": all_garch_summaries[symbol], 
                            "forecast": all_garch_forecasts[symbol],
                            "interpretation": all_garch_interpretations[symbol]
                        }
                        for symbol in all_garch_summaries.keys()
                    }
                }),
                spillover_results=spillover_results,
                granger_causality_results=None
            )
            
        except Exception as e:
            l.error(f"GraphQL pipeline error: {e}")
            raise Exception(f"Pipeline execution failed: {str(e)}")
    
    def resolve_run_arima_model(self, info, input):
        """Run ARIMA model on provided data."""
        try:
            data = json.loads(input.data)
            df = pd.DataFrame(data)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            
            summaries, forecasts, residuals = run_arima_step(
                df_stationary=df,
                p=input.p,
                d=input.d,
                q=input.q,
                forecast_steps=10
            )
            
            result = {
                "summaries": summaries,
                "forecasts": forecasts,
                "residuals": residuals.to_dict('records') if residuals is not None else None
            }
            
            return json.dumps(result)
            
        except Exception as e:
            l.error(f"ARIMA model error: {e}")
            raise Exception(f"ARIMA model failed: {str(e)}")
    
    def resolve_run_garch_model(self, info, input):
        """Run GARCH model on provided data."""
        try:
            data = json.loads(input.data)
            df = pd.DataFrame(data)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            
            summaries, forecasts, cond_vol = run_garch_step(
                df_residuals=df,
                p=input.p,
                q=input.q,
                dist=input.dist,
                forecast_steps=10
            )
            
            result = {
                "summaries": summaries,
                "forecasts": forecasts,
                "conditional_volatility": cond_vol.to_dict('records') if cond_vol is not None else None
            }
            
            return json.dumps(result)
            
        except Exception as e:
            l.error(f"GARCH model error: {e}")
            raise Exception(f"GARCH model failed: {str(e)}")

# Create the GraphQL schema
schema = graphene.Schema(query=Query, mutation=Mutation)