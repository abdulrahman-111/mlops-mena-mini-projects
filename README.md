# ProdML — Ride Duration Service

A production-oriented machine-learning service for predicting
NYC green taxi trip duration.

The project converts an exploratory notebook into:

- an installable Python package
- a tested FastAPI application
- structured JSON logging
- Pickle and ONNX model artifacts
- a Dockerized production service

## Quickstart

Development
pip install -e ".[dev]"

Train:
python -m prodml.train

Test:
pytest

Run locally:

uvicorn prodml.api.main:app --reload --port 8000
