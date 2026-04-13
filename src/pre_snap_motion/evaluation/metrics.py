from __future__ import annotations

from math import sqrt

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        accuracy = y_true[mask].mean()
        confidence = y_prob[mask].mean()
        ece += abs(accuracy - confidence) * (mask.sum() / len(y_true))
    return float(ece)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, bins: int
) -> dict[str, float | None]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": _safe_auroc(y_true, y_prob),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": expected_calibration_error(y_true, y_prob, bins),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
