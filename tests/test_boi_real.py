"""Checks on the real Bank of India quarterly CASA dataset built from NSE filings.

Run ``python run_boi_real.py`` first to build data/real/boi_casa_quarterly.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "real" / "boi_casa_quarterly.csv"

EXPECTED_COLUMNS = [
    "quarter_end",
    "fy_quarter",
    "casa_cr",
    "ca_cr",
    "sa_cr",
    "casa_ratio_reported",
    "domestic_deposits_cr",
    "global_deposits_cr",
    "source_url",
    "source_page",
]

# Figures verified by hand against the filed PDFs (docs/research/boi-public-casa.md).
VERIFIED_CASA = {
    "2015-03-31": 114386,
    "2019-03-31": 181765,
    "2020-12-31": 214826,
    "2023-03-31": 252149,
    "2024-12-31": 277190,
}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        pytest.fail(f"{CSV_PATH} missing; run `python run_boi_real.py` first")
    with CSV_PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == EXPECTED_COLUMNS
        return list(reader)


@pytest.mark.parametrize("quarter_end,casa", sorted(VERIFIED_CASA.items()))
def test_verified_quarters_match(rows, quarter_end, casa):
    by_date = {r["quarter_end"]: r for r in rows}
    assert quarter_end in by_date, f"{quarter_end} missing from CSV"
    assert float(by_date[quarter_end]["casa_cr"]) == casa


def test_quarter_end_unique_and_sorted(rows):
    dates = [r["quarter_end"] for r in rows]
    assert len(dates) == len(set(dates))
    assert dates == sorted(dates)


def test_every_row_has_source_url(rows):
    for r in rows:
        assert r["source_url"].startswith("https://"), r
        assert r["source_page"].strip(), r


def test_quarter_ends_are_calendar_quarter_ends(rows):
    for r in rows:
        assert r["quarter_end"][5:] in {"03-31", "06-30", "09-30", "12-31"}, r
