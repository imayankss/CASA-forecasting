"""
test_data_loader.py — Unit tests for src/data_loader.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_casa_data, get_train_test, get_log_series
from src.config import DATA_FILE


@pytest.fixture(scope="module")
def df():
    return load_casa_data(DATA_FILE)


class TestLoadCasaData:
    def test_loads_without_error(self, df):
        assert df is not None
        assert len(df) > 0

    def test_has_deposit_column(self, df):
        assert "Deposit_Amount" in df.columns

    def test_index_is_datetime(self, df):
        assert hasattr(df.index, "year"), "Index should be DatetimeIndex"

    def test_has_derived_features(self, df):
        for col in ["Log_Deposit", "YoY_Growth"]:
            assert col in df.columns, f"Missing: {col}"

    def test_deposits_positive(self, df):
        assert (df["Deposit_Amount"].dropna() > 0).all()

    def test_sorted_ascending(self, df):
        assert df.index.is_monotonic_increasing


class TestTrainTestSplit:
    def test_sizes(self, df):
        train, test = get_train_test(df, train_frac=0.80)
        total = len(train) + len(test)
        n     = len(df["Deposit_Amount"].dropna())
        assert total == n

    def test_chronological(self, df):
        train, test = get_train_test(df)
        assert train.index[-1] < test.index[0]

    def test_custom_fraction(self, df):
        train, test = get_train_test(df, train_frac=0.75)
        n = len(df["Deposit_Amount"].dropna())
        assert len(train) == int(n * 0.75)

    def test_no_overlap(self, df):
        train, test = get_train_test(df)
        common = train.index.intersection(test.index)
        assert len(common) == 0


class TestLogSeries:
    def test_log_series_positive(self, df):
        ls = get_log_series(df)
        assert (ls > 0).all()

    def test_log_series_length(self, df):
        ls = get_log_series(df)
        assert len(ls) > 0
