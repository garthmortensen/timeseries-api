#!/usr/bin/env python3
# ./save_openapi_json.py

import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app


def save_openapi_spec():
    """Save the current OpenAPI specification to a JSON file."""
    # Generate the OpenAPI specification
    openapi_spec = app.openapi()
    
    # Save to file
    openapi_path = os.path.join('api', 'openapi.json')
    with open(openapi_path, 'w') as f:
        json.dump(openapi_spec, f, indent=2)
    
    print(f"OpenAPI specification saved to {openapi_path}")

if __name__ == "__main__":
    save_openapi_spec()
