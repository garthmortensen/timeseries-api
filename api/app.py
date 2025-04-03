#!/usr/bin/env python3
# timeseries-pipeline/api/app.py
"""FastAPI application initialization and configuration.
This module initializes the FastAPI application and sets up the API routers.
"""

import logging as l
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize logger
from utilities.chronicler import init_chronicler
chronicler = init_chronicler()

# Load configuration
from utilities.configurator import load_configuration
try:
    config = load_configuration("config.yml")
except Exception as e:
    l.error(f"Error loading configuration: {e}")
    raise

# Import FastAPI and related modules
from fastapi import FastAPI
import uvicorn

# Import custom modules
from api.utils.json_handling import RoundingJSONResponse
from api.routers import data_router, models_router, pipeline_router

# Create FastAPI app
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

# Add routers
app.include_router(data_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api/v1")

# Add v1 endpoint alias for backward compatibility
@app.post("/run_pipeline", include_in_schema=False)
async def run_pipeline_alias(pipeline_input):
    """Legacy endpoint that redirects to /api/v1/run_pipeline."""
    return await pipeline_router.endpoints["run_pipeline"](pipeline_input)


# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for API health check."""
    return {"status": "healthy", "message": "Timeseries Pipeline API is running"}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)