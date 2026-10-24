"""
test_temporal.py — Unit tests for src/evaluation/temporal.py (leakage guards)
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.temporal import (
    LeakageError, assert_temporal_split, assert_features_known_at
)

Q = pd.date_range("2020-03-31", periods=8, freq="QE")


class TestTemporalSplit:
    def test_ordered_split_passes(self):
        assert_temporal_split(Q[:6], Q[6:])

    def test_overlap_raises(self):
        with pytest.raises(LeakageError):
            assert_temporal_split(Q[:6], Q[5:])

    def test_test_before_train_raises(self):
        with pytest.raises(LeakageError):
            assert_temporal_split(Q[4:], Q[:2])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            assert_temporal_split(Q[:0], Q[4:])


class TestFeaturesKnownAt:
    def test_past_features_pass(self):
        feats = pd.DataFrame({"x": range(4)}, index=Q[:4])
        assert_features_known_at(feats, origin=Q[3])

    def test_future_feature_raises(self):
        feats = pd.DataFrame({"x": range(5)}, index=Q[:5])
        with pytest.raises(LeakageError):
            assert_features_known_at(feats, origin=Q[3])

    def test_timestamp_column(self):
        feats = pd.DataFrame({"known_at": Q[:5], "x": range(5)})
        with pytest.raises(LeakageError):
            assert_features_known_at(feats, origin=Q[2], time_col="known_at")
