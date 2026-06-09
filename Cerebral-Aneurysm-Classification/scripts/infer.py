"""Full inference pipeline implementing the documented decision rule.

Steps (see README "Decision rule"):
  1. Binary detection -> rows above the median threshold are aneurysm-bearing;
     others have all location labels set to 0.
  2. Positive rows -> multi-label location, thresholded at each location's 90th
     percentile of train predictions.
  3. Per patient (8 views): average anterior/posterior over their 4 views each,
     take the larger average as the patient-level aneurysm score.
"""
from __future__ import annotations

import numpy as np


def aggregate_per_patient(view_scores: np.ndarray, views_per_side: int = 4) -> np.ndarray:
    """Reduce 8 per-view scores to one patient score.

    Args:
        view_scores: (N*8,) array ordered as 4 anterior then 4 posterior views.
        views_per_side: number of views per circulation (default 4).

    Returns:
        (N,) patient-level scores = max(mean(anterior), mean(posterior)).
    """
    scores = view_scores.reshape(-1, 2 * views_per_side)
    ant = scores[:, :views_per_side].mean(axis=1)
    pos = scores[:, views_per_side:].mean(axis=1)
    return np.maximum(ant, pos)


def apply_decision_rule(binary_prob: np.ndarray, binary_thr: float,
                        location_prob: np.ndarray,
                        location_thr: np.ndarray) -> np.ndarray:
    """Combine binary detection and multi-label location into final labels.

    Args:
        binary_prob: (N,) aneurysm scores.
        binary_thr: scalar threshold (median of train binary predictions).
        location_prob: (N, C) location scores.
        location_thr: (C,) per-location thresholds (90th percentile).

    Returns:
        (N, C) binary location predictions; all-zero rows for negative cases.
    """
    positive = binary_prob >= binary_thr
    preds = (location_prob >= location_thr).astype(int)
    preds[~positive] = 0
    return preds
