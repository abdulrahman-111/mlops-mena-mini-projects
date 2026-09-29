import pickle
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.api import main as api_main
from prodml.predict import DurationPredictor


@pytest.fixture
def sample_features():
    return {
        "pickup_id": 74,
        "dropoff_id": 236,
        "trip_distance": 4.0,
    }


@pytest.fixture
def artifact_path(
    tmp_path,
):
    records = [
        {
            "PU_DO": "74_236",
            "trip_distance": 4.0,
        },
        {
            "PU_DO": "75_42",
            "trip_distance": 2.0,
        },
        {
            "PU_DO": "74_236",
            "trip_distance": 8.0,
        },
        {
            "PU_DO": "75_42",
            "trip_distance": 5.0,
        },
    ]

    targets = [
        12.0,
        7.0,
        22.0,
        14.0,
    ]

    vectorizer = DictVectorizer()

    matrix = vectorizer.fit_transform(records)

    model = LinearRegression()

    model.fit(matrix, targets)

    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "metadata": {
            "model_version": "test",
            "training_date": (datetime.now(timezone.utc).isoformat()),
            "feature_names": [
                "PU_DO",
                "trip_distance",
            ],
            "framework": ("scikit-learn-test"),
            "metrics": {
                "mae": 0.0,
                "rmse": 0.0,
            },
        },
    }

    path = tmp_path / "model.pkl"

    with path.open("wb") as f:
        pickle.dump(artifact, f)

    return path


@pytest.fixture
def trained_model(
    artifact_path,
):
    return DurationPredictor.load(artifact_path)


@pytest.fixture
def client(artifact_path, monkeypatch):
    """Start FastAPI with the temporary test model."""

    # Keep settings consistent with the temporary model.
    monkeypatch.setattr(
        api_main.settings,
        "model_path",
        artifact_path,
    )

    # IMPORTANT:
    # main.py uses its own MODEL_PATH variable,
    # which was calculated when the module was imported.
    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        artifact_path,
    )

    # Entering TestClient executes FastAPI's lifespan,
    # which now loads the temporary model.
    with TestClient(api_main.app) as test_client:
        yield test_client
