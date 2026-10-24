"""
test_drift_detection.py — KS drift test on growth rates, including the train -> live step.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.advanced.drift_detection import DriftDetector


def test_single_live_quarter_does_not_crash():
    idx = pd.period_range("2000Q1", periods=30, freq="Q").to_timestamp()
    train = pd.Series(100 * np.exp(0.02 * np.arange(30)), index=idx)
    live = pd.Series([train.iloc[-1] * 1.02], index=[train.index[-1] + pd.offsets.QuarterBegin()])
    stat, p, _ = DriftDetector(train, np.zeros(5))._ks_test(live)
    assert np.isfinite(stat) and np.isfinite(p)


def test_jump_at_train_live_boundary_counts():
    # Train growth alternates 1% / 3%; live continues the pattern, with or without
    # a 30% drop at the boundary. Ignoring the boundary step makes both live samples identical.
    g = np.tile([0.01, 0.03], 15)
    idx = pd.period_range("2000Q1", periods=31, freq="Q").to_timestamp()
    train = pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(g)])), index=idx)
    live_idx = pd.period_range("2007Q4", periods=3, freq="Q").to_timestamp()
    det = DriftDetector(train, np.zeros(5))
    smooth, _, _ = det._ks_test(pd.Series(train.iloc[-1] * np.exp(np.cumsum([0.01, 0.03, 0.01])), index=live_idx))
    jump, _, _ = det._ks_test(pd.Series(train.iloc[-1] * 0.7 * np.exp(np.cumsum([0, 0.03, 0.01])), index=live_idx))
    assert jump > smooth
