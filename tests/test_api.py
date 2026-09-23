def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_metadata(
    client,
):
    response = client.get("/metadata")

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "test"

    assert "artifact_hash" in body


def test_predict_success(
    client,
):
    response = client.post(
        "/predict",
        json={
            "pickup_id": 74,
            "dropoff_id": 236,
            "trip_distance": 4.0,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "model_version" in body
    assert "correlation_id" in body
    assert "latency_ms" in body

    assert "X-Request-ID" in response.headers


def test_invalid_distance_returns_422(
    client,
):
    response = client.post(
        "/predict",
        json={
            "pickup_id": 74,
            "dropoff_id": 236,
            "trip_distance": -1,
        },
    )

    assert response.status_code == 422


def test_predict_with_mock(client, monkeypatch):
    predictor = client.app.state.predictor

    # Your current API unpacks:
    # prediction, latency_ms = predictor.predict_one(...)
    #
    # Therefore the mock must return TWO values.
    monkeypatch.setattr(
        predictor,
        "predict_one",
        lambda features: (12.34, 0.1),
    )

    response = client.post(
        "/predict",
        json={
            "pickup_id": 74,
            "dropoff_id": 236,
            "trip_distance": 4.0,
        },
    )

    assert response.status_code == 200

    body = response.json()

    # Temporary API spelling; see test_predict_success.
    assert body["prediction"] == 12.34
    assert body["latency_ms"] == 0.1
    assert body["model_version"] == "test"


def test_batch_prediction(client):

    response = client.post(
        "/predict/batch",
        json={
            "items": [
                {
                    "pickup_id": 74,
                    "dropoff_id": 236,
                    "trip_distance": 4.0,
                },
                {
                    "pickup_id": 75,
                    "dropoff_id": 42,
                    "trip_distance": 2.0,
                },
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["predictions"]) == 2
    assert body["model_version"] == "test"
