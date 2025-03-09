import logging as l
import time  # stopwatch
import pandas as pd

from src.configurator import load_configuration
from src.chronicler import init_chronicler
from generalized_timeseries import data_generator, data_processor, stats_model

from pydantic import BaseModel  # BaseModel is for input data validation, ensuring correct data types. helps fail fast and clearly
from pydantic import Field  # field is for metadata. used here for description
from fastapi import FastAPI, HTTPException  # FastAPI framework's error handling

chronicler = init_chronicler()

# load default config
try:
    config = load_configuration("config.yml")
except Exception as e:
    l.error(f"error loading configuration: {e}")
    raise  # stop script

# individual endpoints for generate_data, scale_data, etc.
# as well as end-to-end "run_pipeline" endpoint
app = FastAPI(title="Timeseries Pipeline API", version="0.1.0")


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
        l.info("Generating synthetic data...")
        config.data_generator.start_date = input_data.start_date
        config.data_generator.end_date = input_data.end_date
        config.data_generator.anchor_prices = input_data.anchor_prices
        _, price_df = data_generator.generate_price_series(
            config=config
        )  # _ is shorthand for throwaway variable
        return price_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error


@app.post("/scale_data", summary="Scale time series data")
def scale_data(input_data: ScalingInput):
    try:
        df = pd.DataFrame(input_data.data)
        config.data_processor.scaling.method = input_data.method
        df_scaled = data_processor.scale_data(df=df, config=config)
        return df_scaled.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error


@app.post("/test_stationarity", summary="Test for stationarity")
def test_stationarity(input_data: StationarityTestInput):
    try:
        df = pd.DataFrame(input_data.data)
        results = data_processor.test_stationarity(df, config=config)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error


@app.post("/run_arima", summary="Run ARIMA model on time series")
def run_arima_endpoint(input_data: ARIMAInput):
    try:
        df = pd.DataFrame(input_data.data)
        
        # Create a temporary config with the right structure
        from types import SimpleNamespace
        temp_config = SimpleNamespace()
        temp_config.stats_model = SimpleNamespace()
        temp_config.stats_model.ARIMA = SimpleNamespace()
        temp_config.stats_model.ARIMA.parameters_fit = {"p": input_data.p, "d": input_data.d, "q": input_data.q}
        temp_config.stats_model.ARIMA.parameters_predict_steps = config.stats_model.ARIMA.parameters_predict_steps
        
        fit, forecast = stats_model.run_arima(df, temp_config)
        return {"fitted_model": str(fit.summary()), "forecast": forecast.tolist()}
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise HTTPException(status_code=500, detail=f"Error running ARIMA model: {str(e)}")

@app.post("/run_garch", summary="Run GARCH model on time series")
def run_garch_endpoint(input_data: GARCHInput):
    try:
        df = pd.DataFrame(input_data.data)
        fit, forecast = stats_model.run_garch(
            df, {"p": input_data.p, "q": input_data.q}
        )
        return {"fitted_model": str(fit.summary()), "forecast": forecast.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # internal server error


# Endpoints: pipeline
class PipelineInput(BaseModel):
    # captures all config fields in one place for readability and validation
    start_date: str = Field(..., description="Start date for data generation (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for data generation (YYYY-MM-DD)")
    anchor_prices: dict = Field(..., description="Symbol-prices for data generation")
    scaling_method: str = Field(default=config.data_processor.scaling.method, description="Scaling method")
    arima_params: dict = Field(default=config.stats_model.ARIMA.parameters_fit, description="ARIMA parameters")
    garch_params: dict = Field(default=config.stats_model.GARCH.parameters_fit, description="GARCH parameters")



@app.post("/run_pipeline", summary="Execute the entire pipeline")
def run_pipeline(pipeline_input: PipelineInput):
    """Generate data, scale it, test stationarity, then run ARIMA and GARCH.
    Functionality is logic gated by config file."""

    t1 = time.perf_counter()

    try:
        # Step 1: Generate synthetic data
        if config.data_generator.enabled:
            config.data_generator.start_date = pipeline_input.start_date
            config.data_generator.end_date = pipeline_input.end_date
            config.data_generator.anchor_prices = pipeline_input.anchor_prices
            _, df = data_generator.generate_price_series(
                config=config
            )  # _ is shorthand for throwaway variable
        else:
            raise HTTPException(
                status_code=400,  # bad request
                detail="Data generation is disabled in the configuration.",
            )

        # Step 2: Fill missing data
        if config.data_processor.handle_missing_values.enabled:
            df_filled = data_processor.fill_data(df=df, config=config)
        else:
            df_filled = df

        # Step 3: Scale data
        if config.data_processor.scaling.enabled:
            df_scaled = data_processor.scale_data(df=df_filled, config=config)
        else:
            df_scaled = df_filled

        # Step 4: Stationarize data
        if config.data_processor.make_stationary.enabled:
            df_stationary = data_processor.stationarize_data(
                df=df_scaled, config=config
            )
        else:
            df_stationary = df_scaled

        # Step 5: Test stationarity
        stationarity_results = data_processor.test_stationarity(
            df=df_stationary, config=config
        )

        # Step 6: Log stationarity results
        data_processor.log_stationarity(df=stationarity_results, config=config)

        # Step 7: ARIMA
        if config.stats_model.ARIMA.enabled:
            arima_fit, arima_forecast = stats_model.run_arima(df_stationary, config)
        else:
            arima_fit, arima_forecast = {}, {}

        # Step 8: GARCH
        if config.stats_model.GARCH.enabled:
            garch_fit, garch_forecast = stats_model.run_garch(df_stationary, config)
        else:
            garch_fit, garch_forecast = {}, {}

        execution_time = time.perf_counter() - t1
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        l.info(
            f"\nexecution time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        )

        return {
            "stationarity_results": stationarity_results,
            "arima_summary": (
                str(arima_fit.summary()) if arima_fit else "ARIMA not enabled"
            ),
            "arima_forecast": arima_forecast.tolist() if arima_forecast else [],
            "garch_summary": (
                str(garch_fit.summary()) if garch_fit else "GARCH not enabled"
            ),
            "garch_forecast": garch_forecast.tolist() if garch_forecast else [],
        }
    except Exception as e:
        l.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Pipeline failed: {str(e)}"
        )  # internal server error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
