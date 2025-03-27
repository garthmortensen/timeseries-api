# Timeseries Pipeline

![CI/CD](https://github.com/garthmortensen/timeseries-pipeline/actions/workflows/cicd.yml/badge.svg)
[![codecov](https://codecov.io/gh/garthmortensen/timeseries-pipeline/graph/badge.svg?token=L1L5OBSF3Z)](https://codecov.io/gh/garthmortensen/timeseries-pipeline)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a55633cfb8324f379b0b5ec16f03c268)](https://app.codacy.com/gh/garthmortensen/timeseries-pipeline/dashboard)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-pipeline--timeseries-blue)](https://hub.docker.com/r/goattheprofessionalmeower/timeseries-pipeline)

```ascii
   ▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▄▖ ▗▄▄▖
     █    █  ▐▛▚▞▜▌▐▌   ▐▌   ▐▌   ▐▌ ▐▌  █  ▐▌   ▐▌   
     █    █  ▐▌  ▐▌▐▛▀▀▘ ▝▀▚▖▐▛▀▀▘▐▛▀▚▖  █  ▐▛▀▀▘ ▝▀▚▖
     █  ▗▄█▄▖▐▌  ▐▌▐▙▄▄▖▗▄▄▞▘▐▙▄▄▖▐▌ ▐▌▗▄█▄▖▐▙▄▄▖▗▄gm▘
         ▗▄▄▖▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▖   ▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖
         ▐▌ ▐▌ █  ▐▌ ▐▌▐▌   ▐▌     █  ▐▛▚▖▐▌▐▌   
         ▐▛▀▘  █  ▐▛▀▘ ▐▛▀▀▘▐▌     █  ▐▌ ▝▜▌▐▛▀▀▘
         ▐▌  ▗▄█▄▖▐▌   ▐▙▄▄▖▐▙▄▄▖▗▄█▄▖▐▌  ▐▌▐▙▄▄▖
```

TODO: toss vertically into air for the full up and down crash pad effect

TODO: Add async webhooks. Webhooks are HTTP callbacks that are triggered by specific events. They're a way to notify other systems when something happens.

When a model takes minutes or hours to run, you don't want to keep an HTTP connection open that long. Instead:

1. The client (Django) makes a request to start processing.
1. Your API immediately returns a job ID and status "processing".
1. When processing completes, your API calls a webhook URL provided by the client.
1. The Django application receives the webhook with the results.

Benefits

Django can provide immediate feedback to users
Users don't have to keep browser tabs open during processing
Processing continues even if users close their browser
The UI can update dynamically when results arrive
Failed jobs can be properly handled and reported

Reproduce [thesis work](https://github.com/garthmortensen/finance/tree/master/15_thesis) as a production-grade api pipeline.

Take pdf writings and convert entirely. Then add supplementary generalized code.

## Features

- FastAPI endpoints for time series analysis
- OpenAPI response model for illustrating API contract
- ARIMA and GARCH modeling capabilities
- Data generation, scaling, and stationarity testing
- Docker containerization
- GitHub Actions CI/CD pipeline
- Comprehensive test suite

## Quick Start

Pull the Docker image:

```bash
docker pull goattheprofessionalmeower/timeseries-pipeline
```

Run the container:

```bash
docker run -d -p 8000:8000 --name timeseries-pipeline-container goattheprofessionalmeower/timeseries-pipeline:latest
```

## API Endpoints

| Endpoint | HTTP Verb | Description |
|----------|-----------|-------------|
| `/generate_data` | POST | Generate synthetic time series data |
| `/scale_data` | POST | Scale time series data |
| `/test_stationarity` | POST | Test for stationarity |
| `/run_arima` | POST | Run ARIMA model on time series |
| `/run_garch` | POST | Run GARCH model on time series |
| `/run_pipeline` | POST | Execute the full pipeline |

## Development

### Prerequisites

- Python 3.11+
- Docker (optional)

### Local Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/garthmortensen/timeseries-pipeline.git
   cd timeseries-pipeline
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI app:

   ```bash
   python -m fastapi_pipeline.py
   # or
   make run-local
   ```

5. Access the API documentation:

   - Swagger: http://localhost:8001/docs
   - ReDoc: http://localhost:8001/redoc
   - OpenAPI spec: http://localhost:8001/api/openapi.json

### Configuration

The application uses a YAML configuration file at `config/config.yml`. You can customize:

- Data generation parameters
- Data processing strategies
- Model parameters for ARIMA and GARCH

### Testing

Run smoke tests (after launching app):

```bash
./smoketest.sh
```

Run the test suite:

```bash
pytest .
```

## Docker

Build the Docker image:

```bash
make docker-build
```

Run with Docker:

```bash
make docker-run
```

For interactive shell:

```bash
make docker-run-interactive
```

## Project Structure

```
timeseries-pipeline/
├── api/                    # API implementation
│   ├── cli_pipeline.py     # Command-line interface
│   └── fastapi_pipeline.py # FastAPI implementation
├── config/                 # Configuration files
│   └── config.yml          # Main configuration
├── tests/                  # Test suite
├── utilities/              # Utility modules
│   ├── chronicler.py       # Logging utilities
│   └── configurator.py     # Configuration utilities
├── .github/                # GitHub Actions workflows
├── Dockerfile              # Docker configuration
├── Makefile                # Development shortcuts
└── smoketest.sh            # API smoke tests
```

## CI/CD Pipeline

The project uses GitHub Actions for:

- Running tests on multiple Python versions and platforms
- Building and pushing Docker images
- Code coverage reporting

## License

[MIT License](LICENSE). Have at it.
