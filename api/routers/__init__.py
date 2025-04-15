#!/usr/bin/env python3
# timeseries-api/api/routers/__init__.py
"""API routers, grouped by functionality."""

from .data import router as data_router
from .models import router as models_router
from .pipeline import router as pipeline_router

# __all__ is used by `from api.routers import *`
__all__ = ["data_router", "models_router", "pipeline_router",]
