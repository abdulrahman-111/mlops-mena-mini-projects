import pytest

from prodml.features import (
    make_prediction_features,
)


@pytest.mark.parametrize(
    (
        "pickup",
        "dropoff",
        "distance",
        "expected_route",
    ),
    [
        (
            74,
            236,
            4.2,
            "74_236",
        ),
    ],
)
def test_prediction_features(
    pickup,
    dropoff,
    distance,
    expected_route,
):

    result = make_prediction_features(
        pickup,
        dropoff,
        distance,
    )

    assert result["PU_DO"] == expected_route

    assert result["trip_distance"] == float(distance)
