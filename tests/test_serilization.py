import pickle

import numpy as np

from prodml.export import (
    create_onnx_session,
    export_sklearn_model,
    predict_onnx,
)


def test_pickle_onnx_parity(
    artifact_path,
    tmp_path,
):

    with artifact_path.open("rb") as f:

        artifact = pickle.load(f)

    vectorizer = artifact["vectorizer"]

    model = artifact["model"]

    records = [
        {
            "PU_DO": "74_236",
            "trip_distance": 4.0,
        },
        {
            "PU_DO": "75_42",
            "trip_distance": 2.0,
        },
    ]

    matrix = vectorizer.transform(records).toarray().astype(np.float64)

    onnx_path = tmp_path / "model.onnx"

    export_sklearn_model(
        model,
        matrix.shape[1],
        onnx_path,
    )

    session = create_onnx_session(onnx_path)

    pickle_predictions = model.predict(matrix)

    onnx_predictions = predict_onnx(
        session,
        matrix,
    )

    assert np.allclose(
        pickle_predictions,
        onnx_predictions,
        atol=1e-4,
    )
