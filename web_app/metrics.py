from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def amex_metric(y_true, y_pred) -> tuple[float, float, float]:
    """Return official AMEX metric M, normalized weighted Gini G, and top-4% capture D."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cutoff = int(0.04 * np.sum(weights))
    top = labels[np.cumsum(weights) <= cutoff]
    capture = top[:, 0].sum() / labels[:, 0].sum()

    gini = [0.0, 0.0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weights = np.where(labels[:, 0] == 0, 20, 1)
        random = np.cumsum(weights / weights.sum())
        lorentz = np.cumsum(labels[:, 0] * weights) / np.sum(labels[:, 0] * weights)
        gini[i] = np.sum((lorentz - random) * weights)

    normalized_gini = gini[1] / gini[0]
    return 0.5 * (normalized_gini + capture), normalized_gini, capture


def model_scores(model, x_eval):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_eval)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_eval)
    return model.predict(x_eval)


def score_predictions(y_true, y_pred) -> dict[str, float]:
    amex_m, gini, capture = amex_metric(y_true, y_pred)
    return {
        "amex_M": amex_m,
        "gini_G": gini,
        "capture_D@4pct": capture,
        "roc_auc": roc_auc_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred),
    }


def amex_sklearn_scorer(estimator, x_eval, y_eval) -> float:
    return amex_metric(y_eval, model_scores(estimator, x_eval))[0]
