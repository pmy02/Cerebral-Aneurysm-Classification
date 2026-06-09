"""Evaluation metrics for the binary and multi-label tasks.

These produce the numbers that populate the Results table in the README.
Nothing here invents values -- they are computed from predictions and labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy / precision / recall / F1 / AUROC for binary detection."""
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"accuracy": float((y_pred == y_true).mean()),
           "precision": float(p), "recall": float(r), "f1": float(f1)}
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = float("nan")  # undefined if only one class present
    return out


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       thresholds: np.ndarray | float = 0.5,
                       label_names: list[str] | None = None) -> dict:
    """Macro / micro F1, mAP, and per-class F1 for multi-label location."""
    y_pred = (y_prob >= thresholds).astype(int)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    try:
        out["mAP"] = float(average_precision_score(y_true, y_prob, average="macro"))
    except ValueError:
        out["mAP"] = float("nan")
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    names = label_names or [f"class_{i}" for i in range(len(per_class))]
    out["per_class_f1"] = dict(zip(names, map(float, per_class)))
    return out


def percentile_thresholds(probs: np.ndarray, q: float = 90.0) -> np.ndarray:
    """Per-class threshold at the given percentile (decision rule, see README)."""
    return np.percentile(probs, q, axis=0)
