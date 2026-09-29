import pandas as pd

from prodml.data import (
    add_duration,
    clean_trips,
    split_trips,
)


def sample_dataframe():
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 12:00:00",
            ],
            "lpep_dropoff_datetime": [
                "2026-01-01 10:15:00",
                "2026-01-01 11:00:30",
                "2026-01-01 14:00:00",
            ],
            "PULocationID": [
                74,
                75,
                76,
            ],
            "DOLocationID": [
                236,
                42,
                43,
            ],
            "trip_distance": [
                4.0,
                2.0,
                10.0,
            ],
        }
    )


def test_add_duration():
    df = add_duration(sample_dataframe())

    assert df.iloc[0]["duration"] == 15


def test_clean_duration():
    df = add_duration(sample_dataframe())

    cleaned = clean_trips(
        df,
        min_duration=1,
        max_duration=60,
    )

    assert len(cleaned) == 1


def test_split_is_reproducible():
    df = pd.DataFrame({"value": range(100)})

    train_1, val_1 = split_trips(
        df,
        0.2,
        42,
    )

    train_2, val_2 = split_trips(
        df,
        0.2,
        42,
    )

    assert train_1.index.tolist() == train_2.index.tolist()

    assert val_1.index.tolist() == val_2.index.tolist()
