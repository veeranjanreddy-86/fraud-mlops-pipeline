"""Evaluation metrics and threshold selection for imbalanced binary classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float) -> float:
    """Highest recall achievable at any threshold whose precision >= target_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= target_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def threshold_for_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float) -> float:
    """Lowest threshold (i.e. maximum recall) whose precision meets the target.

    Falls back to the threshold with the best precision if the target is unreachable.
    """
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    # precision has len(thresholds) + 1; the final point (precision=1, recall=0) has no threshold.
    precision = precision[:-1]
    ok = np.where(precision >= target_precision)[0]
    if ok.size:
        return float(thresholds[ok[0]])
    return float(thresholds[int(np.argmax(precision))])


def threshold_for_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fn_cost: float = 250.0,
    fp_cost: float = 10.0,
) -> float:
    """Threshold minimising expected cost = FN * fn_cost + FP * fp_cost.

    ``fn_cost`` approximates the average fraud loss (chargeback + ops); ``fp_cost`` approximates
    the friction of a false decline (support call, abandoned basket, churn risk).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    scores, labels = y_score[order], y_true[order]
    # Flagging the top-k rows: cumulative TP/FP for every k (only at distinct score boundaries).
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    fn = labels.sum() - tp
    boundary = np.r_[scores[1:] != scores[:-1], True]
    cost = np.where(boundary, fn * fn_cost + fp * fp_cost, np.inf)
    best = int(np.argmin(cost))
    if labels.sum() * fn_cost <= cost[best]:  # flagging nothing is cheapest
        return float(scores[0] + 1e-9)
    return float(scores[best])


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    target_precision: float,
) -> dict[str, float]:
    """Threshold-free ranking metrics plus point metrics at the operating threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_precision": recall_at_precision(y_true, y_score, target_precision),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alert_rate": float(y_pred.mean()),
    }
