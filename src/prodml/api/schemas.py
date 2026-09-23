from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    trip_distance: float = Field(ge=0)
    pickup_id: Annotated[int, Field(ge=0)]
    dropoff_id: int = Field(ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "pickup_id": 74,
                    "dropoff_id": 236,
                    "trip_distance": 4.2,
                }
            ]
        }
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(ge=0)
    model_version: str
    correlation_id: str
    latency_ms: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float
