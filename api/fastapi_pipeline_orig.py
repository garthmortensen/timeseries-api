#!/usr/bin/env python3
# timeseries-pipeline/api/fastapi_pipeline.py

# import parent directory modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging as l
import time  # stopwatch
import pandas as pd
import numpy as np
import math  # rounding overcomes macOS json serialization issue
import json

from utilities.configurator import load_configuration
from utilities.chronicler import init_chronicler

from generalized_timeseries import data_generator, data_processor, stats_model
from utilities.interpretation import (
    interpret_stationarity_test,
    interpret_arima_results,
    interpret_garch_results,
    interpret_conditional_correlation,
    interpret_conditional_volatility,
    interpret_portfolio_risk
)

from pydantic import (
    BaseModel,
    Field
)  # BaseModel is for input data validation, Field is for metadata
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException  # FastAPI framework's error handling
from fastapi.responses import JSONResponse

import uvicorn  # ASGI server (async server gateway interface, for async web frameworks, like FastAPI)


chronicler = init_chronicler()

# load default config
try:
    config = load_configuration("config.yml")
except Exception as e:
    l.error(f"error loading configuration: {e}")
    raise  # stop script

# round values and handle special cases for json serialization
# without this, tests will fail on macOS due to numpy float serialization
def round_for_json(obj, decimals=6):
    if isinstance(obj, dict):
        # recursively round values in dict
        return {k: round_for_json(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        # recursively round values in list
        return [round_for_json(item, decimals) for item in obj]
    elif isinstance(obj, np.ndarray):
        # recursively round values in numpy array
        return [round_for_json(x, decimals) for x in obj]
    elif isinstance(obj, float):
        # round floats to avoid json serialization issues
        if math.isnan(obj) or math.isinf(obj):
            return None
        # round to specified decimal places
        return round(obj, decimals)
    else:
        return obj

# Create a custom JSON encoder, which handles NaN and Inf values
# this is necessary for macOS, where NaN and Inf values are not serializable to JSON
class RoundingJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

# Create a custom JSON response class, which uses the custom encoder
class RoundingJSONResponse(JSONResponse):
    """All endpoints automatically use the custom response class,
    which rounds all float values and handles special values like NaN and infinity without
    requiring changing individual endpoints.
    """
    def render(self, content):
        # round all values
        rounded_content = round_for_json(content)
        # use the custom encoder for serialization
        return json.dumps(rounded_content, cls=RoundingJSONEncoder).encode()

# =====================================================
# Response Models - Added for OpenAPI documentation and response validation
# =====================================================

class TimeSeriesDataResponse(BaseModel):
    """Response model for time series data endpoints."""
    data: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Time series data indexed by date"
    )


class StationarityTestResponse(BaseModel):
    """Response model for stationarity test results."""
    adf_statistic: float = Field(..., description="ADF test statistic")
    p_value: float = Field(..., description="P-value of the test")
    critical_values: Dict[str, float] = Field(
        ..., 
        description="Critical values at different significance levels"
    )
    is_stationary: bool = Field(
        ..., 
        description="Whether the time series is considered stationary"
    )
    interpretation: str = Field(..., description="Human-readable interpretation of results")


class ARIMAModelResponse(BaseModel):
    """Response model for ARIMA model results."""
    fitted_model: str = Field(..., description="Summary of the fitted ARIMA model")
    parameters: Dict[str, float] = Field(..., description="Model parameters")
    p_values: Dict[str, float] = Field(..., description="P-values for model parameters")
    forecast: List[float] = Field(..., description="Forecasted values")


class GARCHModelResponse(BaseModel):
    """Response model for GARCH model results."""
    fitted_model: str = Field(..., description="Summary of the fitted GARCH model")
    forecast: List[float] = Field(..., description="Forecasted volatility values")


class PipelineResponse(BaseModel):
    """Response model for the complete pipeline."""
    stationarity_results: StationarityTestResponse = Field(
        ..., 
        description="Results of stationarity tests"
    )
    arima_summary: str = Field(..., description="ARIMA model summary")
    arima_forecast: List[float] = Field(..., description="ARIMA model forecast")
    garch_summary: str = Field(..., description="GARCH model summary")
    garch_forecast: List[float] = Field(..., description="GARCH model forecast")

# =====================================================
# FastAPI App initialization
# =====================================================

# individual endpoints for generate_data, scale_data, etc.
# as well as end-to-end "run_pipeline" endpoint
app = FastAPI(
    title="Timeseries Pipeline API",
    version="0.0.1",
    description="Econometric time series modeling API with ARIMA and GARCH capabilities",
    summary="A statistical time series analysis API for financial and econometric modeling",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},  # collapse all models by default
    default_response_class=RoundingJSONResponse  # custom response class for rounding
)

# =====================================================
# Input Models
# =====================================================

class DataGenerationInput(BaseModel):
    start_date: str
    end_date: str
    anchor_prices: dict


class ScalingInput(BaseModel):
    method: str
    data: list


class StationarityTestInput(BaseModel):
    data: list


class ARIMAInput(BaseModel):
    p: int
    d: int
    q: int
    data: list


class GARCHInput(BaseModel):
    p: int
    q: int
    data: list
    dist: Optional[str] = "normal"


# Endpoints: pipeline
class PipelineInput(BaseModel):
    # captures all config fields in one place for readability and validation
    # In pydantic, ... is shorthand for required field. because of Field, this field is required
    # its like making a required field in a schema or html form
    start_date: str = Field(
        ..., description="Start date for data generation (YYYY-MM-DD)"
    )
    end_date: str = Field(..., description="End date for data generation (YYYY-MM-DD)")
    anchor_prices: dict = Field(..., description="Symbol-prices for data generation")
    scaling_method: str = Field(
        default=config.data_processor.scaling.method, description="Scaling method"
    )
    arima_params: dict = Field(
        default=config.stats_model.ARIMA.parameters_fit, description="ARIMA parameters"
    )
    garch_params: dict = Field(
        default=config.stats_model.GARCH.parameters_fit, description="GARCH parameters"
    )

# =====================================================
# API Endpoints with Response Models
# =====================================================

@app.post("/generate_data", 
          summary="Generate synthetic time series data", 
          response_model=TimeSeriesDataResponse)
def generate_data(input_data: DataGenerationInput):
    try:
        _, price_df = data_generator.generate_price_series(
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            anchor_prices=input_data.anchor_prices,
        )  # _ is shorthand for throwaway variable

        return_data = {"data": price_df.to_dict(orient="index")}
        l.info(f"generate_data() returning:\n{return_data}")
        return return_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error


@app.post("/scale_data", 
          summary="Scale time series data", 
          response_model=TimeSeriesDataResponse)
def scale_data(input_data: ScalingInput):
    try:
        df = pd.DataFrame(input_data.data)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if df['price'].isnull().any():
            raise HTTPException(status_code=400, detail="Invalid data: 'price' column contains non-numeric values.")
        df_scaled = data_processor.scale_data(df=df, method=input_data.method)
        
        return_data = {"data": df_scaled.to_dict(orient="index")}
        l.info(f"scale_data() returning:\n{return_data}")
        return return_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error

@app.post("/test_stationarity", 
          summary="Test for stationarity", 
          response_model=StationarityTestResponse)
def test_stationarity(input_data: StationarityTestInput):
    try:
        # Existing code to process data and run tests
        df = pd.DataFrame(input_data.data)
        method = config.data_processor.test_stationarity.method
        results = data_processor.test_stationarity(df=df, method=method)
        
        # Add interpretation
        interpretation = interpret_stationarity_test(
            results, 
            p_value_threshold=config.data_processor.test_stationarity.p_value_threshold
        )
        
        # Add interpretation to results
        results["interpretation"] = interpretation
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_arima", 
          summary="Run ARIMA model on time series", 
          response_model=ARIMAModelResponse)
def run_arima_endpoint(input_data: ARIMAInput):
    try:
        df = pd.DataFrame(input_data.data)
        
        model_fits, forecasts = stats_model.run_arima(
            df_stationary=df,
            p=input_data.p,
            d=input_data.d,
            q=input_data.q,
            forecast_steps=5,
        )
        
        # Handle results for API response
        column_name = list(model_fits.keys())[0]
        model_fit = model_fits[column_name]
        
        # Extract model parameters and their p-values
        params = model_fit.params.to_dict()
        pvalues = model_fit.pvalues.to_dict()
        
        # Convert forecast values to a standard list format
        forecast_values = forecasts[column_name]
        forecast_list = forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [float(value) for value in forecast_values]
        
        # Extract model summary as string
        model_summary = str(model_fit.summary())
        
        results = {
            "fitted_model": model_summary,
            "parameters": params,
            "p_values": pvalues,
            "forecast": forecast_list
        }
        l.info(f"run_arima_endpoint() returning:\n{results}")
        return results
    
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise HTTPException(status_code=500, detail=f"Error running ARIMA model: {str(e)}")


@app.post("/run_garch", 
          summary="Run GARCH model on time series", 
          response_model=GARCHModelResponse)
def run_garch_endpoint(input_data: GARCHInput):
    try:
        # Create DataFrame from input data
        df = pd.DataFrame(input_data.data)
        
        # Convert date column to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Ensure price column is numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Check for any NaN values after conversion
        if df['price'].isnull().any():
            raise HTTPException(status_code=400, detail="Invalid data: 'price' column contains non-numeric values.")
        
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        forecast_steps = 5
        model_fits, forecasts = stats_model.run_garch(
            df_stationary=numeric_df,
            p=input_data.p,
            q=input_data.q,
            dist=input_data.dist,
            forecast_steps=forecast_steps
        )
        
        # Extract the summary from the first column's model
        column_name = list(model_fits.keys())[0]  # from model_fits, get the first column name
        model_summary = str(model_fits[column_name].summary())  # Get the model summary
        
        # Get forecast for the first column
        forecast_values = forecasts[column_name]
        
        # Convert forecast values to a standard list format
        forecast_list = []
        if hasattr(forecast_values, 'tolist'):
            # If it's a numpy array or pandas series
            forecast_list = forecast_values.tolist()
        else:
            # If it's another iterable type
            for value in forecast_values:
                forecast_list.append(float(value))
        
        results = {
            "fitted_model": model_summary,
            "forecast": forecast_list
        }
        l.info(f"run_garch_endpoint() returning:\n{results}")
        return results

    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# Pipeline Step Functions
# =====================================================

# Step functions - each handles one part of the pipeline
def generate_data_step(pipeline_input, config):
    """Generate synthetic data step"""
    if not config.data_generator.enabled:
        raise HTTPException(
            status_code=400,
            detail="Data generation is disabled in the configuration.",
        )
    
    _, df = data_generator.generate_price_series(
        start_date=pipeline_input.start_date,
        end_date=pipeline_input.end_date,
        anchor_prices=pipeline_input.anchor_prices,
    )
    return df

def fill_missing_data_step(df, config):
    """Fill missing data step"""
    if config.data_processor.handle_missing_values.enabled:
        return data_processor.fill_data(
            df=df, 
            strategy=config.data_processor.handle_missing_values.strategy
        )
    return df

def scale_data_step(df, config):
    """Scale data step"""
    if config.data_processor.scaling.enabled:
        return data_processor.scale_data(
            df=df, 
            method=config.data_processor.scaling.method
        )
    return df

def stationarize_data_step(df, config):
    """Stationarize data step"""
    if config.data_processor.make_stationary.enabled:
        return data_processor.stationarize_data(
            df=df, 
            method=config.data_processor.make_stationary.method
        )
    return df

def test_stationarity_step(df, config):
    """Test stationarity step"""
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

def run_arima_step(df_stationary, config):
    """Run ARIMA model step"""
    if not config.stats_model.ARIMA.enabled:
        return "ARIMA not enabled", []
    
    arima_fits, arima_forecasts = stats_model.run_arima(
        df_stationary=df_stationary,
        p=config.stats_model.ARIMA.parameters_fit.p,
        d=config.stats_model.ARIMA.parameters_fit.d,
        q=config.stats_model.ARIMA.parameters_fit.q,
        forecast_steps=config.stats_model.ARIMA.parameters_predict_steps
    )
    
    if not arima_fits or len(arima_fits) <= 0:
        return "No ARIMA models fitted", []
    
    # Get the first column's model summary
    column_name = list(arima_fits.keys())[0]
    arima_summary = str(arima_fits[column_name].summary())
    
    # Get forecast for the first column
    forecast_values = arima_forecasts[column_name]
    
    # Convert forecast values to a list in a more readable way
    if hasattr(forecast_values, 'tolist'):
        # If it's a numpy array or similar object with tolist() method
        arima_forecast_values = forecast_values.tolist()
    else:
        # Otherwise, manually convert each value to float and build a list
        arima_forecast_values = []
        for x in forecast_values:
            arima_forecast_values.append(float(x))
    
    return arima_summary, arima_forecast_values

def run_garch_step(df_stationary, config):
    """Run GARCH model step"""
    if not config.stats_model.GARCH.enabled:
        return "GARCH not enabled", []
    
    garch_fits, garch_forecasts = stats_model.run_garch(
        df_stationary=df_stationary,
        p=config.stats_model.GARCH.parameters_fit.p,
        q=config.stats_model.GARCH.parameters_fit.q,
        dist=config.stats_model.GARCH.parameters_fit.dist,
        forecast_steps=config.stats_model.GARCH.parameters_predict_steps
    )
    
    if not garch_fits or len(garch_fits) <= 0:
        return "No GARCH models fitted", []
    
    # Get the first column's model summary
    column_name = list(garch_fits.keys())[0]
    garch_summary = str(garch_fits[column_name].summary())
    
    # Get forecast for the first column
    forecast_values = garch_forecasts[column_name]
    
    # Convert forecast values to a list in a more readable way
    if hasattr(forecast_values, 'tolist'):
        # If it's a numpy array or similar object with tolist() method
        garch_forecast_values = forecast_values.tolist()
    else:
        # Otherwise, manually convert each value to float and build a list
        garch_forecast_values = []
        for x in forecast_values:
            garch_forecast_values.append(float(x))
    
    return garch_summary, garch_forecast_values


@app.post("/v1/run_pipeline", 
          summary="Execute the entire pipeline",
          response_model=PipelineResponse)
def run_pipeline_v1(pipeline_input: PipelineInput):
    """Generate data, scale it, test stationarity, then run ARIMA and GARCH.
    Functionality is logic gated by config file."""

    t1 = time.perf_counter()

    try:
        # Execute pipeline steps sequentially
        df = generate_data_step(pipeline_input, config)
        df_filled = fill_missing_data_step(df, config)
        df_scaled = scale_data_step(df_filled, config)
        df_stationary = stationarize_data_step(df_scaled, config)
        stationarity_results = test_stationarity_step(df_stationary, config)
        
        # Run models
        arima_summary, arima_forecast = run_arima_step(df_stationary, config)
        garch_summary, garch_forecast = run_garch_step(df_stationary, config)

        # Record execution time
        execution_time = time.perf_counter() - t1
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        l.info(
            f"\nexecution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        )

        # Return results
        return {
            "stationarity_results": stationarity_results,
            "arima_summary": arima_summary,
            "arima_forecast": arima_forecast,
            "garch_summary": garch_summary,
            "garch_forecast": garch_forecast,
        }
    except Exception as e:
        l.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Pipeline failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

