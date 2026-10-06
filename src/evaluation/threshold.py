import numpy as np


def percentile_threshold(scores, percentile=99.0):
    scores = np.asarray(scores, dtype=np.float32)

    if scores.size == 0:
        raise ValueError("scores cannot be empty")

    if not 0 < percentile < 100:
        raise ValueError(
            "percentile must be between 0 and 100"
        )

    return float(
        np.percentile(
            scores,
            percentile,
        )
    )


def predict_from_threshold(
    scores,
    threshold,
):
    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    return (
        scores >= threshold
    ).astype(np.int32)
