from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load NYC green taxi data from a Parquet file."""
    df = pd.read_parquet(path=path)
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return df


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Create trip duration in minutes."""

    result = df.copy()

    result["lpep_pickup_datetime"] = pd.to_datetime(result["lpep_pickup_datetime"])

    result["lpep_dropoff_datetime"] = pd.to_datetime(result["lpep_dropoff_datetime"])

    result["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60

    return result


def clean_trips(
    df: pd.DataFrame,
    min_duration: float,
    max_duration: float,
) -> pd.DataFrame:
    """Remove invalid or unusable training rows."""

    result = df.dropna(
        subset=[
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "duration",
        ]
    ).copy()

    result = result[
        result["duration"].between(
            min_duration,
            max_duration,
        )
    ].copy()

    return result


def split_trips(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible training and validation splits."""

    train_df, validation_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    return train_df, validation_df
