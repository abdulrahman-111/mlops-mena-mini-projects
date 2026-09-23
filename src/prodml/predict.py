import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any

from prodml.features import make_prediction_features
from prodml.logging_conf import get_logger, timed

logger = get_logger(__name__)

"""
If two files are identical: same hash
If even one byte changes: different hash
"""


def file_sha256(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as f:
        while chunk := f.read(8192):
            digest.update(chunk)

    return digest.hexdigest()


class DurationPredictor:
    """Ride-duration model inference interface."""

    # receives already-created components
    def __init__(
        self,
        vectorizer: Any,
        model: Any,
        metadata: dict[str, Any],
    ) -> None:

        self._vetorizer = vectorizer
        self._model = model
        self._metadata = metadata

    # A class method to act as an alternative constructor
    # we give it pickle_file to load
    @classmethod
    def load(cls, path: Path):
        """Load a trusted model artifact."""

        with path.open("rb") as f:
            loaded_artifact = pickle.load(f)

        model = loaded_artifact["model"]
        vectorizer = loaded_artifact["vectorizer"]
        metadata = dict(loaded_artifact["metadata"])

        metadata["artifact_hash"] = file_sha256(path)

        # return new instance
        return cls(
            vectorizer,
            model,
            metadata,
        )

    @classmethod
    def from_onnx():
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @timed  ## my decorator will log the time of request , and return latency
    def predict_one(self, features: dict[str:Any]) -> float:

        model_features = make_prediction_features(
            pickup_id=features["pickup_id"],
            dropoff_id=features["dropoff_id"],
            trip_distance=features["trip_distance"],
        )

        logger.debug("prediction.features", extra={"features": model_features})

        model_input = self._vetorizer.transform([model_features])

        prediction = self._model.predict(model_input)[0]
        return float(prediction)

    def predict_batch(self, items: list[dict[str:Any]]) -> dict[float]:
        """Predict multiple trips in one model call."""

        model_features = [
            make_prediction_features(
                pickup_id=item["pickup_id"],
                dropoff_id=item["dropoff_id"],
                trip_distance=item["trip_distance"],
            )
            for item in items
        ]

        model_input = self._vetorizer.transform(model_features)

        predcitions = self._model.predict(model_input)

        return [float(value) for value in predcitions]
