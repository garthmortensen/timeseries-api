# lightweight and fast
# FROM python:3.9-slim
FROM python:3.13-slim

# bc running as a non-root, create a user that matches host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} timeseriespipelineapp && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash timeseriespipelineapp
USER timeseriespipelineapp

# set containers working dir
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cp everything in
COPY ./ /app

EXPOSE 8000

# run the FastAPI app with Uvicorn
CMD ["uvicorn", "fastapi_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
