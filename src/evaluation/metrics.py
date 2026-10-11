import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def image_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    return float(
        roc_auc_score(labels, scores)
    )


def image_average_precision(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    return float(
        average_precision_score(labels, scores)
    )


def classification_metrics(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    return {
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
    }


def pixel_auroc(masks, anomaly_maps):
    masks = np.asarray(masks)
    anomaly_maps = np.asarray(anomaly_maps)

    labels = masks.reshape(-1)
    scores = anomaly_maps.reshape(-1)

    return float(
        roc_auc_score(labels, scores)
    )
