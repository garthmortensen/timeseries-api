# Use the official Python image
FROM python:3.13-slim

# Create a non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} timeseriesapiapp && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash timeseriesapiapp

# Set working directory and give our user ownership
WORKDIR /app
RUN mkdir -p /app && chown timeseriesapiapp:timeseriesapiapp /app

# Switch to non-root user
USER timeseriesapiapp

# Copy requirements file (as the user)
COPY --chown=timeseriesapiapp:timeseriesapiapp requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Install gunicorn and uvicorn for production
RUN pip install --no-cache-dir --user gunicorn uvicorn

# Add .local/bin to PATH to ensure installed executables are found
ENV PATH="/home/timeseriesapiapp/.local/bin:${PATH}"

# Copy application files (as the user)
COPY --chown=timeseriesapiapp:timeseriesapiapp ./ /app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variable for frontend to reach API (example, adjust as needed)
ENV API_URL="http://timeseries-api:8000"

# Run the FastAPI app with uvicorn in production mode
# Docker recommended: this json format avoids shell string parsing issues
CMD ["uvicorn", "fastapi_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
