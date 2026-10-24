"""
lift.py — Lift at top-k with a bootstrap confidence interval.

Lift@k = hit rate among the top k fraction of customers by score, divided by
the overall hit rate. The interval uses a Poisson bootstrap over customers:
each replicate reweights customers instead of resampling, so the scores are
sorted only once.
"""

from typing import Tuple

import numpy as np


def lift_at_k(y, score, k: float = 0.1) -> float:
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    n_top = max(1, int(round(k * len(y))))
    return float(y[order[:n_top]].mean() / y.mean())


def bootstrap_lift_ci(y, score, k: float = 0.1, n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    y_sorted = y[order]
    rng = np.random.default_rng(seed)
    lifts = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.poisson(1.0, size=len(y)).astype(float)
        cw = np.cumsum(w)
        n_top = np.searchsorted(cw, k * cw[-1], side="left") + 1
        top_rate = np.dot(w[:n_top], y_sorted[:n_top]) / cw[n_top - 1]
        lifts[b] = top_rate / (np.dot(w, y_sorted) / cw[-1])
    lo, hi = np.quantile(lifts, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
