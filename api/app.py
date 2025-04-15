#!/usr/bin/env python3
# timeseries-api/api/app.py
"""FastAPI application initialization and configuration.
This module initializes the FastAPI application and sets up the API routers.
"""

import logging as l
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorama import Fore, Style

# Initialize logger
from utilities.chronicler import init_chronicler
chronicler = init_chronicler(use_json=os.getenv("ENVIRONMENT", "").lower() in ["prod", "production", "cloud"])

# Load configuration
from utilities.configurator import load_configuration
try:
    config = load_configuration("config.yml")
except Exception as e:
    l.error(f"Error loading configuration: {e}")
    raise

# Import FastAPI and related modules
from fastapi import FastAPI, Response
import uvicorn

# Import custom modules
from api.utils.json_handling import RoundingJSONResponse
from api.routers import data_router, models_router, pipeline_router

ascii_banner = f"""
   ▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▄▖ ▗▄▄▖
     █    █  ▐▛▚▞▜▌▐▌   ▐▌   ▐▌   ▐▌ ▐▌  █  ▐▌   ▐▌   
     █    █  ▐▌  ▐▌▐▛▀▀▘ ▝▀▚▖▐▛▀▀▘▐▛▀▚▖  █  ▐▛▀▀▘ ▝▀▚▖
     █  ▗▄█▄▖▐▌  ▐▌▐▙▄▄▖▗▄▄▞▘▐▙▄▄▖▐▌ ▐▌▗▄█▄▖▐▙▄▄▖▗▄gm▘
         ▗▄▄▖▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▖   ▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖
         ▐▌ ▐▌ █  ▐▌ ▐▌▐▌   ▐▌     █  ▐▛▚▖▐▌▐▌   
         ▐▛▀▘  █  ▐▛▀▘ ▐▛▀▀▘▐▌     █  ▐▌ ▝▜▌▐▛▀▀▘
         ▐▌  ▗▄█▄▖▐▌   ▐▙▄▄▖▐▙▄▄▖▗▄█▄▖▐▌  ▐▌▐▙▄▄▖
"""

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_url = "http://localhost:8001"

    l.info(f"{Fore.GREEN}{ascii_banner}")

    # Log interactive docs URLs
    l.info(f"Swagger UI:     {Fore.YELLOW}{base_url}{app.docs_url}")
    l.info(f"ReDoc:          {Fore.YELLOW}{base_url}{app.redoc_url}")
    l.info(f"OpenAPI schema: {Fore.YELLOW}{base_url}{app.openapi_url}")

    # log all endpoints but avoid logging the interactive docs
    skip_paths = {app.docs_url, app.redoc_url, app.openapi_url}
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            if route.path in skip_paths:
                continue
            methods = ",".join(route.methods)
            l.info(f"{methods}  {Fore.CYAN}{base_url}{route.path}")

    yield
    l.info(f"{Fore.RED}Timeseries API API is shutting down")

# Create FastAPI app
app = FastAPI(
    title="Timeseries API API",
    version="0.0.1",
    description="Econometric time series modeling API with ARIMA and GARCH capabilities",
    summary="A statistical time series analysis for financial and econometric modeling",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},  # collapse all models by default
    default_response_class=RoundingJSONResponse,  # custom response class for rounding
    lifespan=lifespan  # adds custom startup/shutdown logging
)

# ignore favicon requests
@app.get("/favicon.ico")
async def ignore_favicon():
    return Response(status_code=204)

# Add routers
app.include_router(data_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(pipeline_router, prefix="/api/v1")

# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for API health check."""
    return {"status": "healthy", "message": "Timeseries API API is running"}



if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    l.info(f"Starting timeseries-api FastAPI application on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
    # uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
