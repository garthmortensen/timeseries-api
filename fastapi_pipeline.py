#!/usr/bin/env python3
# timeseries-pipeline/fastapi_pipeline.py
"""Main application entry point for the FastAPI time series pipeline application."""

import sys
import os
import logging as l
from api.app import app
import uvicorn

if __name__ == "__main__":
    l.info("Starting timeseries-pipeline FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8001)
