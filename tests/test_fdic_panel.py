"""Tests for the FDIC Call Report deposit panel (src/panel/fdic.py).

All fixtures are built inline; nothing here touches the network.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.panel import fdic


def _quarter_ends(start: str, n: int) -> list[str]:
    return fdic.quarter_ends(start, "20991231")[:n]


def _raw_rows(cert: int, repdtes: list[str], ys: list[float], name: str = "BANK") -> pl.DataFrame:
    """Raw /financials-like rows where TRN/MMDA/OTH split y 50/30/20 and the identity holds."""
    trn = [round(v * 0.5) for v in ys]
    mmda = [round(v * 0.3) for v in ys]
    oth = [v - a - b for v, a, b in zip(ys, trn, mmda)]
    time = [1000.0] * len(ys)
    return pl.DataFrame(
        {
            "CERT": [cert] * len(ys),
            "REPDTE": repdtes,
            "NAME": [name] * len(ys),
            "STALP": ["CA"] * len(ys),
            "TRN": [float(x) for x in trn],
            "NTRSMMDA": [float(x) for x in mmda],
            "NTRSOTH": [float(x) for x in oth],
            "NTRTIME": time,
            "DEPDOM": [float(v) + t for v, t in zip(ys, time)],
            "DEP": [float(v) + t for v, t in zip(ys, time)],
            "ASSET": [float(v) * 1.3 for v in ys],
        }
    )


# --------------------------------------------------------------------------------------
# CASA identity
# --------------------------------------------------------------------------------------


def test_quarter_ends_cover_2001q1_to_2026q2():
    q = fdic.quarter_ends("20010331", "20260630")
    assert len(q) == 102
    assert q[0] == "20010331" and q[-1] == "20260630"
    assert q[1] == "20010630" and q[3] == "20011231"


def test_casa_is_trn_plus_mmda_plus_other_savings():
    df = pl.DataFrame(
        {
            "TRN": [352_272_000.0, 10.0, None],
            "NTRSMMDA": [546_395_000.0, 20.0, 5.0],
            "NTRSOTH": [355_601_000.0, 30.0, 5.0],
            "NTRTIME": [100_800_000.0, 40.0, 1.0],
            "DEPDOM": [1_355_068_000.0, 100.0, 11.0],
        }
    )
    out = fdic.add_casa(df)
    assert out["y"].to_list()[:2] == [1_254_268_000.0, 60.0]
    # A missing component makes y missing rather than silently understating it.
    assert out["y"][2] is None


def test_identity_helper_matches_depdom_minus_time_within_rounding():
    # JPM 2019Q4 numbers from the research note: exact identity.
    # Row 2: off by $1k (rounding) -> still holds. Row 3: off by $5k -> fails.
    df = pl.DataFrame(
        {
            "TRN": [352_272_000.0, 100.0, 100.0],
            "NTRSMMDA": [546_395_000.0, 50.0, 50.0],
            "NTRSOTH": [355_601_000.0, 25.0, 25.0],
            "NTRTIME": [100_800_000.0, 10.0, 10.0],
            "DEPDOM": [1_355_068_000.0, 186.0, 190.0],
        }
    )
    df = fdic.add_casa(df)
    assert fdic.identity_holds(df, tol=1.0).to_list() == [True, True, False]
    assert fdic.identity_hit_rate(df, tol=1.0) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------------------
# Panel build
# --------------------------------------------------------------------------------------


def _empty_history() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "CERT": pl.Int64,
            "CHANGECODE": pl.Int64,
            "EFFDATE": pl.Date,
            "SUR_CERT": pl.Int64,
            "OUT_CERT": pl.Int64,
            "ACQ_CERT": pl.Int64,
            "FRM_CERT": pl.Int64,
        }
    )


def test_panel_filters_short_banks_and_keeps_failed_banks():
    q = _quarter_ends("20010331", 102)
    long_alive = _raw_rows(1, q, [1000.0 * 1.01**i for i in range(102)])
    # "Failed" bank: reports 2010Q1..2022Q4 (52 quarters) then disappears (like SVB).
    q_failed = [d for d in q if "20100331" <= d <= "20221231"]
    failed = _raw_rows(2, q_failed, [5000.0] * len(q_failed))
    short = _raw_rows(3, q[:20], [700.0] * 20)
    raw = pl.concat([long_alive, failed, short])

    panel, summary = fdic.build_panel(raw, _empty_history(), min_quarters=40)

    assert panel.columns == ["unique_id", "ds", "y"]
    assert panel.schema["unique_id"] == pl.String
    assert panel.schema["ds"] == pl.Date
    assert panel.schema["y"] == pl.Float64
    assert set(panel["unique_id"].unique()) == {"1", "2"}
    failed_ds = panel.filter(pl.col("unique_id") == "2")["ds"]
    assert failed_ds.max() == dt.date(2022, 12, 31)
    assert summary["bank_count"] == 2
    assert summary["dropped"]["fewer_than_min_quarters"] == 1


def test_panel_cuts_at_extreme_break_and_keeps_recent_segment():
    q = _quarter_ends("20010331", 102)
    ys = [1000.0] * 50 + [5000.0] * 52  # log jump of ln(5) ~ 1.61 > 1.0 at index 50
    raw = _raw_rows(10, q, ys)
    panel, summary = fdic.build_panel(raw, _empty_history(), min_quarters=40)
    s = panel.filter(pl.col("unique_id") == "10")
    assert s.height == 52
    assert s["ds"].min() == dt.date.fromisoformat(f"{q[50][:4]}-{q[50][4:6]}-{q[50][6:]}")
    assert summary["cuts"]["extreme_break"] == 1


def test_panel_drops_non_positive_y_and_splits_at_gap():
    q = _quarter_ends("20010331", 102)
    ys = [1000.0] * 102
    ys[30] = 0.0  # non-positive -> dropped, leaves a gap -> split
    raw = _raw_rows(11, q, ys)
    panel, summary = fdic.build_panel(raw, _empty_history(), min_quarters=40)
    s = panel.filter(pl.col("unique_id") == "11")
    assert (s["y"] > 0).all()
    assert s.height == 102 - 31  # recent segment after the gap
    assert summary["dropped"]["non_positive_or_missing_y_rows"] == 1


def test_panel_cuts_at_material_merger_but_not_tiny_one():
    q = _quarter_ends("20010331", 102)
    acquirer = _raw_rows(20, q, [10_000.0] * 102)
    # Big target (40% of acquirer) merged in 2015-05-01 -> first affected quarter 2015-06-30.
    q_big = [d for d in q if d <= "20150331"]
    big = _raw_rows(21, q_big, [4_000.0] * len(q_big))
    # Tiny target (1%) merged in 2020-02-15 -> should not cut with a 5% threshold.
    q_tiny = [d for d in q if d <= "20191231"]
    tiny = _raw_rows(22, q_tiny, [100.0] * len(q_tiny))
    raw = pl.concat([acquirer, big, tiny])
    history = pl.DataFrame(
        {
            "CERT": [21, 22],
            "CHANGECODE": [223, 223],
            "EFFDATE": [dt.date(2015, 5, 1), dt.date(2020, 2, 15)],
            "SUR_CERT": [20, 20],
            "OUT_CERT": [21, 22],
            "ACQ_CERT": [20, 20],
            "FRM_CERT": [None, None],
        },
        schema_overrides={"FRM_CERT": pl.Int64},
    )

    panel, summary = fdic.build_panel(raw, history, min_quarters=40, merger_min_share=0.05)
    s = panel.filter(pl.col("unique_id") == "20")
    assert s["ds"].min() == dt.date(2015, 6, 30)
    assert s["ds"].max() == dt.date(2026, 6, 30)
    assert summary["cuts"]["merger"] == 1

    # With no materiality threshold every merger cuts: segments 2001Q1-2015Q1 (57 q),
    # 2015Q2-2019Q4 (19 q) and 2020Q1-2026Q2 (26 q). The most recent segment with >= 40
    # quarters is the first one.
    panel0, summary0 = fdic.build_panel(raw, history, min_quarters=40, merger_min_share=0.0)
    s0 = panel0.filter(pl.col("unique_id") == "20")
    assert s0["ds"].min() == dt.date(2001, 3, 31)
    assert s0["ds"].max() == dt.date(2015, 3, 31)
    assert summary0["cuts"]["merger"] == 2
    # With a 20-quarter minimum the recent 2020Q1+ segment wins.
    panel20, _ = fdic.build_panel(raw, history, min_quarters=20, merger_min_share=0.0)
    s20 = panel20.filter(pl.col("unique_id") == "20")
    assert s20["ds"].min() == dt.date(2020, 3, 31)


def test_industry_totals_sum_all_reporters_before_filtering():
    q = _quarter_ends("20010331", 3)
    raw = pl.concat([_raw_rows(1, q, [100.0, 200.0, 300.0]), _raw_rows(2, q[:1], [-5.0])])
    ind = fdic.industry_totals(raw)
    assert ind.columns[:3] == ["ds", "y", "n_banks"]
    assert ind["y"].to_list() == [95.0, 200.0, 300.0]
    assert ind["n_banks"].to_list() == [2, 1, 1]
