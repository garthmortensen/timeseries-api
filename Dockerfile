# lightweight and fast
# FROM python:3.9-slim
FROM python:3.13-slim

# bc running as a non-root, create a user that matches host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} coolcat && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash coolcat
USER coolcat

# set containers working dir
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cp everything in
# COPY . .
# copy everything in souce into app working dir
COPY ./ /app

# Make port 8000 available to the host
EXPOSE 8000

# run the FastAPI app with Uvicorn
CMD ["uvicorn", "fastapi_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t fastapi_pipeline .
# docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --no-cache -t fastapi_pipeline-app .
# docker run -it --user $(id -u):$(id -g) -v $(pwd):/app:Z -p 5000:5000 --name fastapi_pipeline-container fastapi_pipeline-app /bin/bash


# list all containers
# docker ps -a
# docker rmi fensfisifnsfis

# list all images
# docker images
# docker rm djadkjada
