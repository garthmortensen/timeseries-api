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

# Add .local/bin to PATH to ensure installed executables are found
ENV PATH="/home/timeseriesapiapp/.local/bin:${PATH}"

# Copy application files (as the user)
COPY --chown=timeseriesapiapp:timeseriesapiapp ./ /app

EXPOSE 8000

# Run the FastAPI app
# Docker recommended: this json format avoids shell string parsing issues
CMD ["python", "fastapi_pipeline.py"]
