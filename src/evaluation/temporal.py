"""
temporal.py — Leakage guards for time-ordered evaluation.

Every reported metric must come from periods strictly later than the data
the model was trained on, and every feature must be known at the forecast
origin. These guards raise instead of warning so a leak cannot pass silently.
"""

from typing import Optional

import pandas as pd


class LeakageError(AssertionError):
    """Raised when training or feature data reaches into the evaluation period."""


def assert_temporal_split(train_index, test_index) -> None:
    """Fail unless every training timestamp is earlier than every test timestamp."""
    train_index = pd.Index(train_index)
    test_index = pd.Index(test_index)
    if len(train_index) == 0 or len(test_index) == 0:
        raise ValueError("train and test must both be non-empty")
    if train_index.max() >= test_index.min():
        raise LeakageError(
            f"train ends {train_index.max()} but test starts {test_index.min()}"
        )


def assert_features_known_at(
    features: pd.DataFrame, origin, time_col: Optional[str] = None
) -> None:
    """Fail if any feature row is timestamped after the forecast origin.

    Uses the DataFrame index unless `time_col` names a timestamp column.
    """
    stamps = features[time_col] if time_col else features.index.to_series()
    latest = stamps.max()
    if latest > origin:
        raise LeakageError(f"feature known at {latest} is after origin {origin}")
