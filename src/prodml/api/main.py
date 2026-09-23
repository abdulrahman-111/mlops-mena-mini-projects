import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import BatchPredictionRequest, BatchPredictionResponse, PredictionRequest, PredictionResponse
from prodml.config import settings
from prodml.logging_conf import get_correlation_id, get_logger, reset_correlation_id, set_correlation_id, setup_logging
from prodml.predict import DurationPredictor

setup_logging(settings.log_level)

logger = get_logger(__name__)


MODEL_PATH = Path(os.getenv("PRODML_MODEL_PATH", settings.model_path))


# 1. the application lifecycle.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None

    try:
        app.state.predictor = DurationPredictor.load(MODEL_PATH)
        logger.info("Model loaded successfully", extra={"model_path": str(MODEL_PATH)})

    except Exception:
        logger.exception("Model loading failed", extra={"model_path": str(MODEL_PATH)})

    yield

    app.state.predictor = None
    logger.info("Application shutting down")


app = FastAPI(title="Taxi Duration API", version="0.1.0", lifespan=lifespan)


# REQUEST LOGGING MIDDLEWARE
@app.middleware("http")
async def logging_middleware(request: Request, call_next):

    correlation_id = str(uuid4())
    token = set_correlation_id(correlation_id=correlation_id)

    start = perf_counter()

    logger.info(
        "request.started",
        extra={
            "method": request.method,
            "endpoint": request.url.path,
        },
    )

    try:
        response = await call_next(request)

    except Exception:

        logger.exception(
            "request.failed",
            extra={
                "endpoint": request.url.path,
            },
        )
        # exception 500
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "correlation_id": correlation_id,
            },
        )

    try:
        request_latency_ms = (perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = correlation_id

        logger.info(
            "request.completed",
            extra={
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "request_latency_ms": round(request_latency_ms, 3),
            },
        )

        return response

    finally:
        reset_correlation_id(token)


## #VALIDATION ERROR LOGGING  422 -> pydantic


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc=RequestValidationError):

    logger.error(
        "request.validation_failed",
        extra={
            "endpoint": request.url.path,
        },
    )

    # Retain FastAPI's standard informative 422 response.
    return await request_validation_exception_handler(
        request,
        exc,
    )


# 3. Retrieve the loaded predictor.
def get_predictor(request: Request) -> DurationPredictor:

    predictor = request.app.state.predictor

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    return predictor


@app.get("/health")
def health(request: Request):
    predictor = get_predictor(request=request)

    return {"status": "healthy", "model_version": predictor.metadata["model_version"]}


@app.get("/metadata")
def metadata(request: Request):
    return get_predictor(request).metadata


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:

    predictor = get_predictor(request=request)

    # Warning for an unusual, but valid, distance.
    if payload.trip_distance > 100:
        logger.warning(
            "prediction.outside_training_range",
            extra={
                "trip_distance": payload.trip_distance,
            },
        )

    # The @timed decorator measures inference latency.
    # Our middleware logs total server-side request latency,
    # while the existing @timed decorator logs inference latency.
    prediction, latency_ms = predictor.predict_one(
        features={
            "pickup_id": payload.pickup_id,
            "dropoff_id": payload.dropoff_id,
            "trip_distance": payload.trip_distance,
        }
    )

    model_version = predictor.metadata["model_version"]

    logger.info(
        "prediction.served",
        extra={
            "prediction": prediction,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 3),
        },
    )

    return PredictionResponse(
        predcition=round(prediction, 3),
        model_version=model_version,
        correlation_id=get_correlation_id(),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch")
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:

    predictor = get_predictor(request)

    items = [item.model_dump() for item in payload.items]  # list of dict

    start = perf_counter()

    predictions = predictor.predict_batch(items)

    latency_ms = (perf_counter() - start) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=predictor.metadata["model_version"],
        correlation_id=get_correlation_id(),
        latency_ms=round(
            latency_ms,
            3,
        ),
    )
