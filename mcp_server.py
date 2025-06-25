#!/usr/bin/env python3
"""MCP Server for Timeseries API using FastMCP.

This server wraps the FastAPI timeseries endpoints and exposes them as MCP tools
for LLM agents to interact with directly.
"""

from mcp.server.fastmcp import FastMCP
import json
import requests
from typing import List, Dict, Any, Optional
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("spillover")

# Base URL for your FastAPI server
API_BASE_URL = os.getenv("TIMESERIES_API_URL", "http://localhost:8001")

def make_api_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to make API requests with error handling."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=payload,
            timeout=300  # 5 minute timeout for long computations
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise

def make_graphql_request(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper function to make GraphQL requests."""
    try:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
            
        response = requests.post(
            f"{API_BASE_URL}/v1/graphql",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        
        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")
            
        return result.get("data", {})
    except requests.exceptions.RequestException as e:
        logger.error(f"GraphQL request failed: {e}")
        raise

# GraphQL Tools
@mcp.tool()
def graphql_health_check() -> Dict[str, Any]:
    """Check the health status of the timeseries API using GraphQL.
    
    Returns:
        Dict containing health status from GraphQL endpoint
    """
    query = """
    query HealthCheck {
        health
    }
    """
    return make_graphql_request(query)

@mcp.tool()
def graphql_fetch_market_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Dict[str, Any]:
    """Fetch market data using GraphQL endpoint.
    
    Args:
        symbols: List of stock symbols to fetch
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1d, 1h, etc.)
        
    Returns:
        Dict containing market data from GraphQL endpoint
    """
    query = """
    query FetchMarketData($input: MarketDataInputType!) {
        fetchMarketData(input: $input) {
            date
            values
        }
    }
    """
    variables = {
        "input": {
            "symbols": symbols,
            "startDate": start_date,
            "endDate": end_date,
            "interval": interval
        }
    }
    return make_graphql_request(query, variables)

@mcp.tool()
def graphql_test_stationarity(data: Dict[str, Any]) -> Dict[str, Any]:
    """Test stationarity using GraphQL endpoint.
    
    Args:
        data: Time series data as JSON object
        
    Returns:
        Dict containing stationarity test results from GraphQL
    """
    query = """
    query TestStationarity($input: StationarityTestInputType!) {
        testStationarity(input: $input) {
            allSymbolsStationarity
            seriesStats
        }
    }
    """
    variables = {
        "input": {
            "data": json.dumps(data)
        }
    }
    return make_graphql_request(query, variables)

@mcp.tool()
def graphql_run_complete_pipeline(
    symbols: List[str] = ["GME", "BYND"],
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    source_data: str = "synthetic",
    anchor_prices: Optional[List[float]] = None,
    random_seed: Optional[int] = None,
    scaling_method: str = "standard",
    arima_params: Dict[str, Any] = {"p": 1, "d": 1, "q": 1},
    garch_params: Dict[str, Any] = {"p": 1, "q": 1, "dist": "normal"},
    spillover_enabled: bool = False,
    spillover_params: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """Execute complete pipeline using GraphQL mutation.
    
    This provides the same functionality as the REST endpoint but through GraphQL,
    offering more flexible data fetching with the ability to specify exactly which
    fields you want returned.
    
    Args:
        symbols: List of stock symbols to analyze
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source_data: Data source ("synthetic", "actual_yahoo", "actual_stooq")
        anchor_prices: Starting prices for synthetic data
        random_seed: Random seed for reproducible synthetic data
        scaling_method: Scaling method for data processing
        arima_params: ARIMA model parameters
        garch_params: GARCH model parameters
        spillover_enabled: Whether to run spillover analysis
        spillover_params: Parameters for spillover analysis
        
    Returns:
        Dict containing complete pipeline results from GraphQL
    """
    query = """
    mutation RunPipeline($input: PipelineInputType!) {
        runPipeline(input: $input) {
            originalData {
                date
                values
            }
            returnsData {
                date
                values
            }
            scaledData {
                date
                values
            }
            preGarchData {
                date
                values
            }
            postGarchData {
                date
                values
            }
            stationarityResults {
                allSymbolsStationarity
                seriesStats
            }
            seriesStats
            arimaResults
            garchResults
            spilloverResults {
                totalSpilloverIndex
                directionalSpillover
                netSpillover
                pairwiseSpillover
                interpretation
            }
            grangerCausalityResults
        }
    }
    """
    
    variables = {
        "input": {
            "sourceActualOrSyntheticData": source_data,
            "dataStartDate": start_date,
            "dataEndDate": end_date,
            "symbols": symbols,
            "syntheticAnchorPrices": anchor_prices or [150.0, 200.0],
            "syntheticRandomSeed": random_seed,
            "scalingMethod": scaling_method,
            "arimaParams": json.dumps(arima_params),
            "garchParams": json.dumps(garch_params),
            "spilloverEnabled": spillover_enabled,
            "spilloverParams": json.dumps(spillover_params)
        }
    }
    return make_graphql_request(query, variables)

@mcp.tool()
def graphql_run_arima_model(
    data: Dict[str, Any],
    p: int = 1,
    d: int = 1,
    q: int = 1
) -> Dict[str, Any]:
    """Run ARIMA model using GraphQL mutation.
    
    Args:
        data: Time series data as JSON object
        p: ARIMA p parameter (autoregressive order)
        d: ARIMA d parameter (differencing order)
        q: ARIMA q parameter (moving average order)
        
    Returns:
        Dict containing ARIMA model results from GraphQL
    """
    query = """
    mutation RunArima($input: ARIMAInputType!) {
        runArimaModel(input: $input)
    }
    """
    variables = {
        "input": {
            "p": p,
            "d": d,
            "q": q,
            "data": json.dumps(data)
        }
    }
    return make_graphql_request(query, variables)

@mcp.tool()
def graphql_run_garch_model(
    data: Dict[str, Any],
    p: int = 1,
    q: int = 1,
    dist: str = "t"
) -> Dict[str, Any]:
    """Run GARCH model using GraphQL mutation.
    
    Args:
        data: Time series data as JSON object (typically returns data)
        p: GARCH p parameter
        q: GARCH q parameter
        dist: Distribution assumption ("normal", "t", "skewt")
        
    Returns:
        Dict containing GARCH model results from GraphQL
    """
    query = """
    mutation RunGarch($input: GARCHInputType!) {
        runGarchModel(input: $input)
    }
    """
    variables = {
        "input": {
            "p": p,
            "q": q,
            "data": json.dumps(data),
            "dist": dist
        }
    }
    return make_graphql_request(query, variables)

@mcp.tool()
def graphql_custom_query(
    query: str,
    variables: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a custom GraphQL query or mutation.
    
    This tool allows LLM agents to write and execute custom GraphQL queries
    for more advanced or specific use cases not covered by the predefined tools.
    
    Args:
        query: The GraphQL query or mutation string
        variables: Optional variables for the GraphQL query
        
    Returns:
        Dict containing the GraphQL response data
    """
    return make_graphql_request(query, variables)

# Data Operations Tools
@mcp.tool()
def generate_synthetic_data(
    symbols: List[str] = ["GME", "BYND"],
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    num_points: int = 252
) -> Dict[str, Any]:
    """Generate synthetic time series data for multiple symbols.
    
    Args:
        symbols: List of stock symbols (e.g., ['GME', 'BYND'])
        start_date: Start date for data generation (YYYY-MM-DD)
        end_date: End date for data generation (YYYY-MM-DD)
        num_points: Number of data points to generate
        
    Returns:
        Dict containing synthetic time series data with date and price columns
    """
    payload = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "num_points": num_points
    }
    return make_api_request("/api/v1/generate_data", payload)

@mcp.tool()
def fetch_market_data(
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Fetch real market data from Yahoo Finance.
    
    Args:
        symbols: List of stock symbols to fetch data for
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dict containing real market data from Yahoo Finance
    """
    payload = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date
    }
    return make_api_request("/api/v1/fetch_market_data", payload)

@mcp.tool()
def fetch_stooq_data(
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Fetch real market data from Stooq.
    
    Args:
        symbols: List of stock symbols to fetch data for
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dict containing real market data from Stooq
    """
    payload = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date
    }
    return make_api_request("/api/v1/fetch_stooq_data", payload)

@mcp.tool()
def scale_data(
    data: List[Dict[str, Any]],
    method: str = "standard",
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Scale time series data using various methods.
    
    Args:
        data: Time series data to scale
        method: Scaling method ('standard', 'minmax', 'robust')
        columns: Specific columns to scale (None for all numeric columns)
        
    Returns:
        Dict containing scaled time series data
    """
    payload = {
        "data": data,
        "method": method,
        "columns": columns
    }
    return make_api_request("/api/v1/scale_data", payload)

@mcp.tool()
def test_stationarity(
    data: List[Dict[str, Any]],
    target_column: str = "GME"
) -> Dict[str, Any]:
    """Test time series data for stationarity using ADF test.
    
    Args:
        data: Time series data with date and value columns
        target_column: Column to test for stationarity
        
    Returns:
        Dict containing stationarity test results (ADF statistic, p-value, critical values)
    """
    payload = {
        "data": data,
        "target_column": target_column
    }
    return make_api_request("/api/v1/test_stationarity", payload)

@mcp.tool()
def convert_to_returns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert price data to log returns.
    
    Args:
        data: Time series price data
        
    Returns:
        Dict containing converted returns data
    """
    payload = {"data": data}
    return make_api_request("/api/v1/price_to_returns", payload)

@mcp.tool()
def scale_for_garch(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scale data specifically for GARCH modeling.
    
    Args:
        data: Time series data to scale for GARCH
        
    Returns:
        Dict containing GARCH-scaled data
    """
    payload = {"data": data}
    return make_api_request("/api/v1/scale_for_garch", payload)

# Statistical Models Tools
@mcp.tool()
def run_arima_model(
    data: List[Dict[str, Any]],
    target_column: str = "GME",
    order: List[int] = [1, 1, 1],
    forecast_steps: int = 10
) -> Dict[str, Any]:
    """Fit ARIMA model to time series data.
    
    Args:
        data: Time series data
        target_column: Column to model
        order: ARIMA order [p, d, q]
        forecast_steps: Number of steps to forecast
        
    Returns:
        Dict containing ARIMA model results, parameters, and forecasts
    """
    payload = {
        "data": data,
        "target_column": target_column,
        "order": order,
        "forecast_steps": forecast_steps
    }
    return make_api_request("/api/v1/run_arima", payload)

@mcp.tool()
def run_garch_model(
    data: List[Dict[str, Any]],
    target_column: str = "GME",
    p: int = 1,
    q: int = 1,
    forecast_steps: int = 10
) -> Dict[str, Any]:
    """Fit GARCH model for volatility forecasting.
    
    Args:
        data: Time series data (typically returns)
        target_column: Column to model
        p: GARCH p parameter
        q: GARCH q parameter
        forecast_steps: Number of steps to forecast
        
    Returns:
        Dict containing GARCH model results and volatility forecasts
    """
    payload = {
        "data": data,
        "target_column": target_column,
        "p": p,
        "q": q,
        "forecast_steps": forecast_steps
    }
    return make_api_request("/api/v1/run_garch", payload)

# Spillover Analysis Tools
@mcp.tool()
def analyze_spillover(
    data: List[Dict[str, Any]],
    lag_order: int = 1,
    forecast_horizon: int = 10
) -> Dict[str, Any]:
    """Analyze spillover effects between multiple time series.
    
    Args:
        data: Time series data with multiple columns
        lag_order: VAR lag order
        forecast_horizon: Forecast horizon for spillover analysis
        
    Returns:
        Dict containing spillover analysis results including spillover table and metrics
    """
    payload = {
        "data": data,
        "lag_order": lag_order,
        "forecast_horizon": forecast_horizon
    }
    return make_api_request("/api/v1/analyze_spillover", payload)

@mcp.tool()
def rolling_spillover(
    data: List[Dict[str, Any]],
    window_size: int = 100,
    lag_order: int = 1,
    forecast_horizon: int = 10
) -> Dict[str, Any]:
    """Compute rolling window spillover analysis.
    
    Args:
        data: Time series data with multiple columns
        window_size: Rolling window size
        lag_order: VAR lag order
        forecast_horizon: Forecast horizon for spillover analysis
        
    Returns:
        Dict containing rolling spillover analysis results
    """
    payload = {
        "data": data,
        "window_size": window_size,
        "lag_order": lag_order,
        "forecast_horizon": forecast_horizon
    }
    return make_api_request("/api/v1/rolling_spillover", payload)

# Complete Pipeline Tool
@mcp.tool()
def run_complete_pipeline(
    symbols: List[str] = ["GME", "BYND"],
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    use_real_data: bool = False,
    run_spillover: bool = True,
    test_stationarity: bool = True
) -> Dict[str, Any]:
    """Execute the complete end-to-end time series analysis pipeline.
    
    This is the most comprehensive tool that runs the entire analysis workflow:
    1. Generate synthetic data or fetch real market data
    2. Fill missing data (if configured)
    3. Convert prices to returns
    4. Test for stationarity (if configured)
    5. Scale data for GARCH modeling
    6. Fit ARIMA models for conditional mean
    7. Extract ARIMA residuals
    8. Fit GARCH models for volatility forecasting
    9. Run spillover analysis and Granger causality if enabled
    10. Return all results including forecasts and interpretations
    
    Args:
        symbols: List of stock symbols to analyze
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_real_data: Whether to use real market data or synthetic data
        run_spillover: Whether to include spillover analysis
        test_stationarity: Whether to test for stationarity
        
    Returns:
        Dict containing complete pipeline analysis results with all models and forecasts
    """
    payload = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "use_real_data": use_real_data,
        "run_spillover": run_spillover,
        "test_stationarity": test_stationarity
    }
    return make_api_request("/api/v1/run_pipeline", payload)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Timeseries API MCP Server")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8001",
        help="Base URL of the Timeseries API (default: http://localhost:8001)"
    )
    
    args = parser.parse_args()
    
    # Update the API base URL if provided
    global API_BASE_URL
    API_BASE_URL = args.api_url
    
    logger.info(f"Starting MCP server, connecting to API at: {API_BASE_URL}")
    mcp.run(transport="stdio")