# lightweight and fast
FROM python:3.13-slim

# bc running as a non-root, create a user that matches host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} timeseriesapiapp && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash timeseriesapiapp

# set containers working dir
WORKDIR /app

# copy everything in source into app working dir first
COPY ./ /app

# Install uv as root for system-wide availability
USER root
RUN pip install uv
RUN chown -R timeseriesapiapp:timeseriesapiapp /app

# Switch to non-root user for dependency installation
USER timeseriesapiapp

# Install dependencies using uv (for API service, not package)
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Set environment variables
ENV API_URL="http://timeseries-api:8080"
ENV PORT=8080
ENV PATH="/app/.venv/bin:${PATH}"

# Run the FastAPI app with uvicorn in production mode
# Docker recommended: this json format avoids shell string parsing issues
CMD ["uvicorn", "fastapi_pipeline:app", "--host", "0.0.0.0", "--port", "8080"]

# docker build -t timeseries-api:latest ./
# docker run -p 8080:8080 timeseries-api:latest
# -p for port mapping from host to container

# to run the container in interactive mode, without using uvicorn as the entrypoint
# docker run -it --entrypoint /bin/bash timeseries-api:latest
