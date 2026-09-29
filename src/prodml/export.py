import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType

from prodml.config import Settings, settings
from prodml.data import add_duration, clean_trips, load_data, split_trips
from prodml.features import add_route_feature, df_to_dict
from prodml.logging_conf import get_logger, setup_logging

logger = get_logger(__name__)


def export_sklearn_model(model: Any, n_features: int, output_path: Path) -> Path:
    """Export Sklearn model to ONNX"""

    initial_types = [("features", DoubleTensorType([None, n_features]))]

    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=17)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_bytes(onnx_model.SerializeToString())

    return output_path


def create_onnx_session(path: Path) -> ort.InferenceSession:

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def predict_onnx(session: ort.InferenceSession, matrix: np.ndarray) -> np.ndarray:

    input_name = session.get_inputs()[0].name

    result = session.run(None, {input_name: matrix})[0]

    return np.asarray(result).reshape(-1)


def benchmark_serializations(model: Any, session: ort.InferenceSession, matrix: np.ndarray) -> dict[str, float]:
    """Benchmark per-row model inference."""

    pickle_times: list[float] = []
    onnx_times: list[float] = []

    input_name = session.get_inputs()[0].name

    for index in range(len(matrix)):
        row = matrix[index : index + 1]  # to remain 2d shape

        start = perf_counter()

        model.predict(row)
        pickle_times.append((perf_counter() - start) * 1000)

        start = perf_counter()
        session.run(None, {input_name: row})

        onnx_times.append((perf_counter() - start) * 1000)

    return {
        "pickle_mean_ms": float(np.mean(pickle_times)),
        "pickle_p95_ms": float(np.percentile(pickle_times, 95)),
        "onnx_mean_ms": float(np.mean(onnx_times)),
        "onnx_p95_ms": float(
            np.percentile(
                onnx_times,
                95,
            )
        ),
    }


# build data used in benchmark
def build_validation_matrix(config: Settings, vectorizer: Any, number_of_rows: int = 500) -> np.ndarray:

    df = load_data(config.data_path)
    df = add_duration(df)

    df = clean_trips(df, config.min_duration, config.max_duration)

    df = add_route_feature(df)

    _, validation_df = split_trips(
        df,
        config.test_size,
        config.random_state,
    )

    sample = validation_df.head(number_of_rows)

    records = df_to_dict(sample)

    matrix = vectorizer.transform(records)

    return matrix.toarray().astype(np.float64)


def run_export(
    config: Settings = settings,
) -> dict[str, float]:

    # Only load trusted artifacts produced
    # by your own training pipeline.

    with config.model_path.open("rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    matrix = build_validation_matrix(config, vectorizer, number_of_rows=500)

    export_sklearn_model(model=model, n_features=matrix.shape[1], output_path=config.onnx_path)

    session = create_onnx_session(config.onnx_path)

    pickle_predictions = model.predict(matrix)

    onnx_predictions = predict_onnx(session, matrix)

    # parity_test
    max_difference = float(np.max(np.abs(pickle_predictions - onnx_predictions)))

    if not np.allclose(pickle_predictions, onnx_predictions, atol=1e-4):

        raise AssertionError(
            "Pickle and ONNX predictions " f"are not equivalent. " f"Maximum difference: " f"{max_difference}"
        )

    logger.info(
        "serialization.parity_passed",
        extra={
            "rows": len(matrix),
            "max_absolute_difference": (max_difference),
        },
    )

    benchmark = benchmark_serializations(model, session, matrix)

    logger.info(
        "serialization.benchmark",
        extra=benchmark,
    )

    return benchmark


def main() -> None:
    setup_logging(settings.log_level)

    run_export(settings)


if __name__ == "__main__":
    main()
