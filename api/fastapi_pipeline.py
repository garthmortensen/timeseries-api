#!/usr/bin/env python3
# fastapi_pipeline.py

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

from pydantic import (
    BaseModel,
)  # BaseModel is for input data validation, ensuring correct data types. helps fail fast and clearly
from pydantic import Field  # field is for metadata. used here for description
from fastapi import FastAPI, HTTPException  # FastAPI framework's error handling
from fastapi.responses import JSONResponse


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
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    default_response_class=RoundingJSONResponse  # custom response class for rounding
)


# Endpoints: modular
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


# Modular endpoints
@app.post("/generate_data", summary="Generate synthetic time series data")
def generate_data(input_data: DataGenerationInput):
    try:
        _, price_df = data_generator.generate_price_series(
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            anchor_prices=input_data.anchor_prices,
        )  # _ is shorthand for throwaway variable
        return price_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error


@app.post("/scale_data", summary="Scale time series data")
def scale_data(input_data: ScalingInput):
    try:
        df = pd.DataFrame(input_data.data)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if df['price'].isnull().any():
            raise HTTPException(status_code=400, detail="Invalid data: 'price' column contains non-numeric values.")
        df_scaled = data_processor.scale_data(df=df, method=input_data.method)
        return df_scaled.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error


@app.post("/test_stationarity", summary="Test for stationarity")
def test_stationarity(input_data: StationarityTestInput):
    try:
        df = pd.DataFrame(input_data.data)
        method = config.data_processor.test_stationarity.method
        results = data_processor.test_stationarity(df=df, method=method)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_arima", summary="Run ARIMA model on time series")
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
        model_summary = str(model_fits[column_name].summary())
        forecast_values = forecasts[column_name]
        
        return {
            "fitted_model": model_summary,
            "forecast": forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [float(x) for x in forecast_values]
        }
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise HTTPException(status_code=500, detail=f"Error running ARIMA model: {str(e)}")


@app.post("/run_garch", summary="Run GARCH model on time series")
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
            dist=input_data.dist if hasattr(input_data, "dist") else "normal",
            forecast_steps=forecast_steps
        )
        
        # Extract the summary from the first column's model
        column_name = list(model_fits.keys())[0]  # Get the first column name
        model_summary = str(model_fits[column_name].summary())
        
        # Get forecast for the first column
        forecast_values = forecasts[column_name]
        
        return {
            "fitted_model": model_summary,
            "forecast": forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [float(x) for x in forecast_values]
        }
    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints: pipeline
class PipelineInput(BaseModel):
    # captures all config fields in one place for readability and validation
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


@app.post("/run_pipeline", summary="Execute the entire pipeline")
def run_pipeline(pipeline_input: PipelineInput):
    """Generate data, scale it, test stationarity, then run ARIMA and GARCH.
    Functionality is logic gated by config file."""

    t1 = time.perf_counter()

    try:
        # Step 1: Generate synthetic data
        if config.data_generator.enabled:
            _, df = data_generator.generate_price_series(
                start_date=pipeline_input.start_date,
                end_date=pipeline_input.end_date,
                anchor_prices=pipeline_input.anchor_prices,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Data generation is disabled in the configuration.",
            )

        # Step 2: Fill missing data
        if config.data_processor.handle_missing_values.enabled:
            df_filled = data_processor.fill_data(
                df=df, 
                strategy=config.data_processor.handle_missing_values.strategy
            )
        else:
            df_filled = df

        # Step 3: Scale data
        if config.data_processor.scaling.enabled:
            df_scaled = data_processor.scale_data(
                df=df_filled, 
                method=config.data_processor.scaling.method
            )
        else:
            df_scaled = df_filled

        # Step 4: Stationarize data
        if config.data_processor.make_stationary.enabled:
            df_stationary = data_processor.stationarize_data(
                df=df_scaled, 
                method=config.data_processor.make_stationary.method
            )
        else:
            df_stationary = df_scaled

        # Step 5: Test stationarity
        stationarity_results = data_processor.test_stationarity(
            df=df_stationary, 
            method=config.data_processor.test_stationarity.method
        )

        # Step 6: Log stationarity results
        data_processor.log_stationarity(
            adf_results=stationarity_results,
            p_value_threshold=config.data_processor.test_stationarity.p_value_threshold
        )

        # Step 7: ARIMA
        arima_summary = "ARIMA not enabled"
        arima_forecast_values = []
        
        if config.stats_model.ARIMA.enabled:
            arima_fits, arima_forecasts = stats_model.run_arima(
                df_stationary=df_stationary,
                p=config.stats_model.ARIMA.parameters_fit.p,
                d=config.stats_model.ARIMA.parameters_fit.d,
                q=config.stats_model.ARIMA.parameters_fit.q,
                forecast_steps=config.stats_model.ARIMA.parameters_predict_steps
            )
            
            if arima_fits and len(arima_fits) > 0:
                # Get the first column's model summary
                column_name = list(arima_fits.keys())[0]
                arima_summary = str(arima_fits[column_name].summary())
                
                # Get forecast for the first column
                forecast_values = arima_forecasts[column_name]
                arima_forecast_values = forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [float(x) for x in forecast_values]

        # Step 8: GARCH
        garch_summary = "GARCH not enabled"
        garch_forecast_values = []
        
        if config.stats_model.GARCH.enabled:
            garch_fits, garch_forecasts = stats_model.run_garch(
                df_stationary=df_stationary,
                p=config.stats_model.GARCH.parameters_fit.p,
                q=config.stats_model.GARCH.parameters_fit.q,
                dist=config.stats_model.GARCH.parameters_fit.dist,
                forecast_steps=config.stats_model.GARCH.parameters_predict_steps
            )
            
            if garch_fits and len(garch_fits) > 0:
                # Get the first column's model summary
                column_name = list(garch_fits.keys())[0]
                garch_summary = str(garch_fits[column_name].summary())
                
                # Get forecast for the first column
                forecast_values = garch_forecasts[column_name]
                garch_forecast_values = forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [float(x) for x in forecast_values]

        execution_time = time.perf_counter() - t1
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        l.info(
            f"\nexecution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        )

        return {
            "stationarity_results": stationarity_results,
            "arima_summary": arima_summary,
            "arima_forecast": arima_forecast_values,
            "garch_summary": garch_summary,
            "garch_forecast": garch_forecast_values,
        }
    except Exception as e:
        l.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Pipeline failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
