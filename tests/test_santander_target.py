"""
test_santander_target.py — CASA-acquisition target on a tiny fixture.
"""

import sys
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.propensity.santander import CASA_PRODUCTS, casa_targets

MONTHS = [date(2016, 1, 28), date(2016, 2, 28), date(2016, 3, 28)]


def _rows(cust, flags_by_month):
    rows = []
    for d, flags in zip(MONTHS, flags_by_month):
        if flags is None:
            continue
        row = {"ncodpers": cust, "fecha_dato": d, **{c: 0 for c in CASA_PRODUCTS}}
        row.update(flags)
        rows.append(row)
    return rows


def _fixture():
    rows = []
    # 1: opens a current account in Feb, keeps it in Mar
    rows += _rows(1, [{}, {"ind_cco_fin_ult1": 1}, {"ind_cco_fin_ult1": 1}])
    # 2: already holds savings, never adds anything new
    rows += _rows(2, [{"ind_ahor_fin_ult1": 1}] * 3)
    # 3: drops payroll in Feb, re-adds it in Mar; missing flags count as not held
    rows += _rows(3, [{"ind_cno_fin_ult1": 1}, {"ind_cno_fin_ult1": None}, {"ind_cno_fin_ult1": 1}])
    # 4: absent in Feb, so neither transition involving Feb can be scored
    rows += _rows(4, [{}, None, {"ind_ecue_fin_ult1": 1}])
    return pl.DataFrame(rows).lazy()


def test_adds_not_holds():
    out = casa_targets(_fixture()).collect().sort(["ncodpers", "fecha_dato"])
    got = {(r["ncodpers"], r["fecha_dato"]): r["casa_add"] for r in out.iter_rows(named=True)}
    assert got == {
        (1, MONTHS[0]): 1, (1, MONTHS[1]): 0,
        (2, MONTHS[0]): 0, (2, MONTHS[1]): 0,
        (3, MONTHS[0]): 0, (3, MONTHS[1]): 1,
    }


def test_target_is_dated_at_feature_month():
    out = casa_targets(_fixture()).collect()
    assert out["fecha_dato"].max() == MONTHS[1]
