#!/usr/bin/env python3
# timeseries-api/api/utils/json_handling.py
"""JSON handling utilities for the API.
Thi module provides utilities for handling JSON responses in the API. 
MacOS has issues with serializing NaN and Inf values to JSON. 
This module provides a custom JSON encoder and response class that rounds float values and handles special cases like NaN and infinity.
This ensures the API tests pass on all platforms.
"""

import json
import math
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse

# Round values and handle special cases for json serialization
# without this, tests will fail on macOS due to numpy float serialization
def round_for_json(obj, decimals=6):
    """
    Round float values in objects and handle special cases for JSON serialization.
    
    Args:
        obj: Object to process (dict, list, float, etc.)
        decimals (int): Number of decimal places to round to
        
    Returns:
        Processed object with rounded values
    """
    if isinstance(obj, dict):
        # recursively round values in dict
        return {k: round_for_json(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        # recursively round values in list
        return [round_for_json(item, decimals) for item in obj]
    elif isinstance(obj, np.ndarray):
        # recursively round values in numpy array
        return [round_for_json(x, decimals) for x in obj]
    elif isinstance(obj, (float, np.float32, np.float64)):
        # round floats to avoid json serialization issues
        if math.isnan(obj) or math.isinf(obj):
            return None
        # round to specified decimal places
        return round(float(obj), decimals)
    elif isinstance(obj, (np.int32, np.int64)):
        # convert numpy ints to Python ints
        return int(obj)
    elif isinstance(obj, pd.Timestamp):
        # convert pandas Timestamp to ISO format string
        return obj.isoformat()
    else:
        return obj


# Create a custom JSON encoder, which handles NaN and Inf values
# this is necessary for macOS, where NaN and Inf values are not serializable to JSON
class RoundingJSONEncoder(json.JSONEncoder):
    """JSON encoder that rounds float values and handles NaN and Infinity."""
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
            return float(obj)
        return super().default(obj)


# Create a custom JSON response class, which uses the custom encoder
class RoundingJSONResponse(JSONResponse):
    """
    JSON response that rounds float values and handles special values.
    
    All endpoints automatically use the custom response class,
    which rounds all float values and handles special values like NaN and infinity without
    requiring changing individual endpoints.
    """
    def render(self, content):
        # round all values
        rounded_content = round_for_json(content)
        # use the custom encoder for serialization
        return json.dumps(rounded_content, cls=RoundingJSONEncoder).encode()
