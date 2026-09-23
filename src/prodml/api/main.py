import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from prodml.api.schemas import PredictionRequest, PredictionResponse
from prodml.config import settings
from prodml.logging_conf import get_logger, setup_logging
from prodml.predict import DurationPredictor

setup_logging(settings.log_level)

logger = get_logger(__name__)


MODEL_PATH = Path(os.getenv("PRODML_MODEL_PATH", settings.model_path))


# 1. Define the application lifecycle.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None

    try:
        app.state.predictor = DurationPredictor.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Model loading failed")

    yield

    app.state.predictor = None
    logger.info("Application shutting down")


app = FastAPI(title="Taxi Duration API", version="0.1.0", lifespan=lifespan)


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

    prediction, latency_ms = predictor.predict_one(
        features={
            "pickup_id": payload.pickup_id,
            "dropoff_id": payload.dropoff_id,
            "trip_distance": payload.trip_distance,
        }
    )

    return PredictionResponse(
        predcition=round(prediction, 3),
        model_version=predictor.metadata["model_version"],
        latency_ms=round(latency_ms, 3),
    )
