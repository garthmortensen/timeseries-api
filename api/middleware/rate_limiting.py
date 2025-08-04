#!/usr/bin/env python3
# timeseries-api/api/middleware/rate_limiting.py
"""Rate limiting middleware using SlowAPI to prevent abuse and control costs."""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
import logging as l

# RATE LIMITING CONFIGURATION - Adjust these variables as needed
# =================================================================

# Global rate limits (applies to all endpoints)
GLOBAL_REQUESTS_PER_MINUTE = "6/minute"  # x requests per minute per IP
GLOBAL_REQUESTS_PER_HOUR = "20/hour"    # x requests per hour per IP

# Heavy computation endpoints (pipeline, models, spillover)
HEAVY_COMPUTATION_PER_MINUTE = "6/minute"   # Very restrictive for expensive operations
HEAVY_COMPUTATION_PER_HOUR = "20/hour"     # x heavy operations per hour max

# Data endpoints (generate_data, fetch_market_data)
DATA_ENDPOINTS_PER_MINUTE = "6/minute"    # Moderate limit for data operations
DATA_ENDPOINTS_PER_HOUR = "20/hour"       # x data requests per hour

# Health check and light endpoints
LIGHT_ENDPOINTS_PER_MINUTE = "20/minute"   # More generous for health checks

# =================================================================

def get_client_ip(request: Request):
    """
    Get client IP address, considering proxy headers for cloud deployment.
    """
    # Check for common proxy headers first (for Cloud Run, load balancers, etc.)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection IP
    return get_remote_address(request)

# Create the limiter instance
limiter = Limiter(
    key_func=get_client_ip,
    default_limits=[GLOBAL_REQUESTS_PER_MINUTE, GLOBAL_REQUESTS_PER_HOUR],
    storage_uri="memory://",  # Simple in-memory storage
)

def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for rate limit exceeded errors with helpful message.
    """
    client_ip = get_client_ip(request)
    l.warning(f"Rate limit exceeded for IP {client_ip} on {request.url.path}")
    
    return HTTPException(
        status_code=429,
        detail={
            "error": "Rate limit exceeded",
            "message": "You have exceeded the rate limit for this API. This is a free service, please use it responsibly.",
            "retry_after": exc.retry_after,
            "limits": {
                "global": f"{GLOBAL_REQUESTS_PER_MINUTE}, {GLOBAL_REQUESTS_PER_HOUR}",
                "heavy_computation": f"{HEAVY_COMPUTATION_PER_MINUTE}, {HEAVY_COMPUTATION_PER_HOUR}",
                "data_endpoints": f"{DATA_ENDPOINTS_PER_MINUTE}, {DATA_ENDPOINTS_PER_HOUR}"
            },
            "note": "Rate limits help keep this free service available for everyone. Please space out your requests."
        }
    )