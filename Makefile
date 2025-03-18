print-% : ; @echo $* = $($*)

PROJECT_NAME = pipeline-timeseries
SHELL = /bin/bash
PYTHON ?= $(shell command -v python3 || command -v python)
COMMIT_HASH = $(shell git log -1 --format=%h || echo "dev")

.PHONY: docker-clean
docker-clean:
	docker container prune -f
	docker image prune -f
	docker network prune -f

.PHONY: docker-build
docker-build:
	docker build --build-arg USER_ID=$(shell id -u) --build-arg GROUP_ID=$(shell id -g) -t $(PROJECT_NAME):latest -t $(PROJECT_NAME):$(COMMIT_HASH) .

.PHONY: docker-run
docker-run:
	docker run -d -p 8000:8000 --name $(PROJECT_NAME)-container $(PROJECT_NAME):latest

.PHONY: docker-run-interactive
docker-run-interactive:
	docker run -it --user $(shell id -u):$(shell id -g) -v $(shell pwd):/app:Z -p 8000:8000 --name $(PROJECT_NAME)-interactive $(PROJECT_NAME):latest /bin/bash

.PHONY: docker-stop
docker-stop:
	docker stop $(PROJECT_NAME)-container || true

.PHONY: docker-rm
docker-rm:
	docker rm $(PROJECT_NAME)-container || true

# Application commands
.PHONY: run-local
run-local:
	$(PYTHON) -m pip install -q fastapi uvicorn
	uvicorn fastapi_pipeline:app --host 0.0.0.0 --port 8000 --reload

.PHONY: run-cli
run-cli:
	$(PYTHON) cli_pipeline.py

# Convenience targets
.PHONY: rebuild
rebuild: docker-stop docker-rm docker-clean docker-build docker-run

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  docker-clean          - Clean Docker resources"
	@echo "  docker-build          - Build Docker image"
	@echo "  docker-run            - Run Docker container"
	@echo "  docker-run-interactive - Run Docker container with shell"
	@echo "  docker-stop           - Stop Docker container"
	@echo "  docker-rm             - Remove Docker container"
	@echo "  run-local             - Run FastAPI app locally"
	@echo "  run-cli               - Run CLI pipeline"
	@echo "  rebuild               - Rebuild and run Docker container"

.PHONY: default
default: help
