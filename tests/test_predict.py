def test_prediction_returns_float(
    trained_model,
    sample_features,
):

    prediction, _ = trained_model.predict_one(sample_features)

    assert isinstance(prediction, float)


def test_prediction_is_deterministic(
    trained_model,
    sample_features,
):

    first, _ = trained_model.predict_one(sample_features)

    second, _ = trained_model.predict_one(sample_features)

    assert first == second


def test_unseen_route_does_not_crash(
    trained_model,
):

    prediction, _ = trained_model.predict_one(
        {
            "pickup_id": 999,
            "dropoff_id": 998,
            "trip_distance": 3.0,
        }
    )

    assert isinstance(prediction, float)
