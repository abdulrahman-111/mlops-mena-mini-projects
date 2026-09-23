# ProdML — NYC Green Taxi Ride Duration Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED)](https://www.docker.com/)

An end-to-end MLOps learning project that turns an exploratory NYC Green Taxi ride-duration notebook into an installable, tested, containerized ML inference service. This repository is developed incrementally as part of **The MLOps Practitioner — Module 1: From Notebook to Production-Ready Service**.

The baseline predicts trip duration in minutes using a pickup–dropoff route (`PU_DO`) and `trip_distance`. A scikit-learn `DictVectorizer` encodes the features, and `LinearRegression` produces the prediction. The service exposes the model through FastAPI with validated requests, structured JSON logs, request correlation IDs, and Docker support.

> **Scope:** This README describes the Module 1 implementation. Training orchestration, model registries, continuous training, and production monitoring belong to later modules.

## Features

- Reproducible baseline training and validation with MAE and RMSE.
- Editable Python package with separate modules for data, features, training, configuration, and prediction.
- Pydantic request validation and four FastAPI endpoints: `/health`, `/metadata`, `/predict`, and `/predict/batch`.
- One-time model loading during application startup.
- Structured JSON logging with correlation IDs returned in the `X-Request-ID` response header.
- Trusted Pickle model persistence, ONNX export, prediction parity checks, and inference latency benchmarking.
- Pytest-based unit and API tests, with an **80% coverage target**.
- Multi-stage Docker build, non-root runtime user, health check, and Docker Compose configuration.

## Architecture

```text
NYC TLC Green Taxi Parquet
           |
           v
    Data preparation
    (src/prodml/data.py)
           |
           v
    Feature engineering
  (src/prodml/features.py)
           |
           v
    DictVectorizer +
    LinearRegression
           |
           +--------------------+
           |                    |
           v                    v
    models/model.pkl      models/model.onnx
           |              (export / benchmark)
           v
  DurationPredictor
           |
           v
  FastAPI + Pydantic
           |
           v
     JSON response

Structured JSON logging and correlation IDs cover the serving path.
```

## Repository layout

```text
.
├── data/                          # Local dataset (not committed)
├── docker/
│   ├── Dockerfile                 # Multi-stage production image
│   ├── Dockerfile.single          # Image-size comparison
│   └── docker-compose.yml
├── models/                        # Generated model artifacts
│   ├── model.pkl
│   └── model.onnx
├── notebooks/
│   └── 00-baseline.ipynb          # Original exploratory baseline
├── reports/
│   └── module-1.md                # Measurements and evaluation
├── src/prodml/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── config.py
│   ├── data.py
│   ├── export.py
│   ├── features.py
│   ├── logging_conf.py
│   ├── predict.py
│   └── train.py
├── tests/
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

## Prerequisites

- Python **3.11** recommended (the project declares Python `>=3.10`).
- Git and a Python virtual environment.
- Docker and Docker Compose for containerized deployment.
- One month of [NYC TLC Green Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page); the example configuration uses **January 2023**.

Run the local development commands below from the **repository root**, unless otherwise stated.

## Local setup

### 1. Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows, activate the virtual environment with the activation command appropriate to your shell.

### 2. Add the training dataset

Download the January 2023 Green Taxi Parquet file from NYC TLC and save it as:

```text
data/green_tripdata_2023-01.parquet
```

The default training configuration reads this path. The raw dataset is intended to remain local rather than being committed to Git.

### 3. Train the model

```bash
python -m prodml.train
# Equivalent console script: prodml-train
```

Training reads the Parquet dataset, creates trip duration in minutes, filters duration to **1–60 minutes**, builds the `PU_DO` feature, and makes a reproducible **80/20 train/validation split** with `random_state=42`. It fits `DictVectorizer` on training data only, trains `LinearRegression`, evaluates MAE and RMSE, and saves:

```text
models/model.pkl
```

Compare the packaged training MAE with the original `notebooks/00-baseline.ipynb` result. The Module 1 refactoring goal is a difference of no more than **±0.05 MAE**. Record your measured results in `reports/module-1.md`.

### 4. Start the API

```bash
uvicorn prodml.api.main:app --reload --port 8000
```

Open the interactive API documentation at **http://localhost:8000/docs**.

> The service loads `models/model.pkl` at startup. Train the model first, or set `PRODML_MODEL_PATH` to the path of a trusted artifact.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Reports healthy only when the model is loaded. |
| `GET` | `/metadata` | Returns version, training date, features, framework, metrics, and artifact SHA-256 hash. |
| `POST` | `/predict` | Predicts duration for one trip. |
| `POST` | `/predict/batch` | Predicts durations for multiple trips. |

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_location_id": 74,
    "dropoff_location_id": 236,
    "trip_distance": 4.2
  }'
```

Response shape (values are illustrative, not measured results):

```json
{
  "prediction": 14.72,
  "model_version": "0.1.0",
  "correlation_id": "REQUEST-UUID",
  "latency_ms": 3.81
}
```

The response also includes the HTTP header `X-Request-ID` with the request's correlation ID.

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "pickup_location_id": 74,
        "dropoff_location_id": 236,
        "trip_distance": 4.2
      },
      {
        "pickup_location_id": 75,
        "dropoff_location_id": 42,
        "trip_distance": 2.5
      }
    ]
  }'
```

The response contains `predictions` (a list of numbers), `model_version`, `correlation_id`, and `latency_ms`.

### Validation

Pickup and dropoff location IDs must be positive integers. `trip_distance` must be greater than `0` and less than `200`. Invalid requests return **HTTP 422** with validation details; an unavailable model results in **HTTP 503**.

## Serialization: Pickle and ONNX

The Python API serves a **trusted Pickle artifact** containing both the trained estimator and fitted vectorizer. ONNX export is provided to compare portability and inference performance.

```bash
python -m prodml.export
# Equivalent console script: prodml-export
```

The export workflow creates `models/model.onnx`, compares scikit-learn and ONNX Runtime predictions on **500 validation rows** with an absolute tolerance of `1e-4`, and benchmarks mean and p95 inference latency. Record actual results in `reports/module-1.md`.

**Benchmark boundary:** Both runtimes receive the same numeric feature matrix produced by the saved `DictVectorizer`. The ONNX artifact in this implementation represents the estimator, **not the entire raw-JSON-to-prediction pipeline**.

**Security:** Unpickling can execute code. Load only model files generated by a trusted training process; never load arbitrary user-supplied `.pkl` files.

## Structured logging

The service writes JSON logs to stdout. Each application log contains `timestamp`, `level`, `logger`, `message`, and `correlation_id`; prediction logs can also include the model version, latency, or prediction value.

A request-scoped correlation ID is stored using Python `contextvars`. It ties together request-start, prediction, and request-completion events and is returned via `X-Request-ID`.

```bash
PRODML_LOG_LEVEL=DEBUG uvicorn prodml.api.main:app --port 8000
```

Use `DEBUG` when investigating feature preparation. In routine serving, the default level is `INFO`. Avoid logging credentials, sensitive request contents, or large raw datasets.

## Testing and quality checks

```bash
pytest
ruff check src tests
black --check src tests
pre-commit run --all-files
```

The test suite is designed to cover feature construction, cleaning and reproducible splits, predictor behavior, valid and invalid HTTP requests, batch inference, and serialization parity. The configured project coverage target is **80%**; coverage is a measure of code executed during tests, not a guarantee of correctness.

Install Git hooks, if desired:

```bash
pre-commit install
```

For the full 500-row ONNX parity and latency exercise, run `python -m prodml.export` after training; it complements the smaller automated serialization unit test.

## Docker

Train the model locally before building: the multi-stage Dockerfile copies `models/` into the runtime image.

### Build and run

```bash
docker build -f docker/Dockerfile -t prodml-api:0.1.0 .
docker run --rm -p 8000:8000 prodml-api:0.1.0
```

In another terminal:

```bash
curl http://localhost:8000/health
```

The production image uses a non-root `appuser` and an HTTP health check.

### Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

The Compose setup mounts the local `models/` directory read-only. To stop it:

```bash
docker compose -f docker/docker-compose.yml down
```

### Compare image sizes

```bash
docker build -f docker/Dockerfile.single -t prodml-api:single .
docker build -f docker/Dockerfile -t prodml-api:multi .
docker images prodml-api
```

Record the measured image sizes in `reports/module-1.md`. The multi-stage image is intended to omit development-only dependencies; measure the actual difference rather than assuming a specific percentage reduction.



## Configuration

Configuration is provided by `pydantic-settings`. Environment variables use the `PRODML_` prefix:

| Variable | Default | Description |
|---|---|---|
| `PRODML_DATA_PATH` | `data/green_tripdata_2023-01.parquet` | Local training dataset. |
| `PRODML_MODEL_PATH` | `models/model.pkl` | Trusted Pickle model path. |
| `PRODML_ONNX_PATH` | `models/model.onnx` | ONNX export location. |
| `PRODML_MODEL_VERSION` | `0.1.0` | Reported model version. |
| `PRODML_TEST_SIZE` | `0.2` | Validation split fraction. |
| `PRODML_RANDOM_STATE` | `42` | Reproducible split seed. |
| `PRODML_LOG_LEVEL` | `INFO` | Application logging level. |

The project can also read these values from a local `.env` file. Do not commit secrets in `.env`.

## Evaluation and Module 1 report

Use `reports/module-1.md` to record **measured**, not example, results:

- Notebook MAE/RMSE vs. packaged training MAE/RMSE.
- Maximum prediction difference for the 500-row ONNX parity check.
- Mean and p95 latency for scikit-learn vs. ONNX Runtime.
- Single-stage vs. multi-stage Docker image sizes.
- JSON, Protobuf, Pickle, and ONNX comparison and the chosen serving format.
- MLOps maturity assessment and the next missing capabilities.

## Development workflow

Module 1 is implemented on the `module-1-packaging` feature branch. Open one pull request for the module, request peer review, then merge and tag the completed release `v0.1.0` once the Definition of Done is met.

## Limitations and next steps

This baseline uses a small feature set and a simple linear model to focus on production engineering. It does not yet include automated retraining, experiment tracking, a model registry, continuous deployment, or production drift monitoring. These will be introduced as the same repository evolves through later MLOps modules.

## Data attribution

Trip records: [New York City Taxi & Limousine Commission — TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Check NYC TLC's data documentation and usage conditions before redistributing derived data or artifacts.
