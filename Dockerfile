# Use the official Python image
FROM python:3.13-slim

# Create a non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} timeseriespipelineapp && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash timeseriespipelineapp

# Set working directory and give our user ownership
WORKDIR /app
RUN mkdir -p /app && chown timeseriespipelineapp:timeseriespipelineapp /app

# Switch to non-root user
USER timeseriespipelineapp

# Copy requirements file (as the user)
COPY --chown=timeseriespipelineapp:timeseriespipelineapp requirements-minimal.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Add .local/bin to PATH to ensure installed executables are found
ENV PATH="/home/timeseriespipelineapp/.local/bin:${PATH}"

# Copy application files (as the user)
COPY --chown=timeseriespipelineapp:timeseriespipelineapp ./ /app

EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "fastapi_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
