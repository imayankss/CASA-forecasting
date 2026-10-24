"""
test_lift.py — Unit tests for src/evaluation/lift.py
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.lift import lift_at_k, bootstrap_lift_ci


def test_hand_computed_lift():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    score = np.arange(10, 0, -1, dtype=float)   # first customer ranked highest
    # top 10% = 1 customer, hit rate 1.0; base rate 0.2 -> lift 5
    assert lift_at_k(y, score, k=0.1) == pytest.approx(5.0)


def test_random_scores_lift_near_one():
    rng = np.random.default_rng(0)
    y = rng.random(200_000) < 0.05
    assert lift_at_k(y, rng.random(200_000), k=0.1) == pytest.approx(1.0, abs=0.1)


def test_ci_brackets_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    score = rng.random(50_000)
    y = rng.random(50_000) < 0.02 + 0.08 * score     # informative score
    point = lift_at_k(y, score, k=0.1)
    lo, hi = bootstrap_lift_ci(y, score, k=0.1, n_boot=200, seed=7)
    assert lo < point < hi
    assert (lo, hi) == bootstrap_lift_ci(y, score, k=0.1, n_boot=200, seed=7)
