"""FDIC BankFind Call Report ingest and US bank CASA panel construction.

US CASA analog (see docs/research/panel-deposit-data-and-tools.md, section 1):

    y = TRN + NTRSMMDA + NTRSOTH  (= DEPDOM - NTRTIME), in $ thousands.

Only the sum is modelled: the 2020 Regulation D change moved balances between the
savings and transaction components, but not the sum.

Pipeline:
    ingest_financials()  one parquet per quarter-end REPDTE (cached, resumable)
    ingest_history()     merger / failure structure events from /history
    build_panel()        long panel (unique_id, ds, y) cut at mergers and extreme breaks
    industry_totals()    sum of y across every reporter per quarter (before filtering)
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

API_BASE = "https://api.fdic.gov/banks"
PAGE_LIMIT = 10_000

FIELDS = [
    "CERT",
    "REPDTE",
    "NAME",
    "STALP",
    "TRN",
    "NTRSMMDA",
    "NTRSOTH",
    "DEPDOM",
    "NTRTIME",
    "DEP",
    "ASSET",
]
NUMERIC_FIELDS = ["TRN", "NTRSMMDA", "NTRSOTH", "DEPDOM", "NTRTIME", "DEP", "ASSET"]
RAW_SCHEMA: dict[str, Any] = {
    "CERT": pl.Int64,
    "REPDTE": pl.String,
    "NAME": pl.String,
    "STALP": pl.String,
    **{f: pl.Float64 for f in NUMERIC_FIELDS},
}

# Structure-change codes that move deposits into a surviving institution (research note,
# "Merger and structure history"). 150 (cert change) is cached but does not cut a series.
MERGER_CODES = (211, 213, 216, 217, 221, 222, 223, 224, 810, 811)
HISTORY_CODES = (150,) + MERGER_CODES
HISTORY_FIELDS = [
    "ID",
    "CERT",
    "INSTNAME",
    "CHANGECODE",
    "CHANGECODE_DESC",
    "EFFDATE",
    "SUR_CERT",
    "OUT_CERT",
    "ACQ_CERT",
    "FRM_CERT",
    "OUT_INSTNAME",
]
HISTORY_SCHEMA: dict[str, Any] = {
    "ID": pl.String,
    "CERT": pl.Int64,
    "INSTNAME": pl.String,
    "CHANGECODE": pl.Int64,
    "CHANGECODE_DESC": pl.String,
    "EFFDATE": pl.String,
    "SUR_CERT": pl.Int64,
    "OUT_CERT": pl.Int64,
    "ACQ_CERT": pl.Int64,
    "FRM_CERT": pl.Int64,
    "OUT_INSTNAME": pl.String,
}


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------


def quarter_ends(start: str, end: str) -> list[str]:
    """Quarter-end dates as ``yyyymmdd`` strings, inclusive of both bounds."""
    s = dt.date(int(start[:4]), int(start[4:6]), int(start[6:]))
    e = dt.date(int(end[:4]), int(end[4:6]), int(end[6:]))
    ends = {3: 31, 6: 30, 9: 30, 12: 31}
    out = []
    for year in range(s.year, e.year + 1):
        for month, day in ends.items():
            d = dt.date(year, month, day)
            if s <= d <= e:
                out.append(d.strftime("%Y%m%d"))
    return out


def _qidx(date_expr: pl.Expr) -> pl.Expr:
    """Integer quarter index (consecutive quarters differ by 1)."""
    return date_expr.dt.year().cast(pl.Int64) * 4 + (date_expr.dt.month().cast(pl.Int64) - 1) // 3


# --------------------------------------------------------------------------------------
# CASA identity
# --------------------------------------------------------------------------------------


def add_casa(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``y = TRN + NTRSMMDA + NTRSOTH``. A missing component gives a missing y."""
    return df.with_columns((pl.col("TRN") + pl.col("NTRSMMDA") + pl.col("NTRSOTH")).alias("y"))


def identity_holds(df: pl.DataFrame, tol: float = 1.0) -> pl.Series:
    """True where ``y`` equals ``DEPDOM - NTRTIME`` within ``tol`` ($ thousands)."""
    return df.select(
        ((pl.col("y") - (pl.col("DEPDOM") - pl.col("NTRTIME"))).abs() <= tol)
        .fill_null(False)
        .alias("identity_ok")
    ).to_series()


def identity_hit_rate(df: pl.DataFrame, tol: float = 1.0) -> float:
    """Share of rows where the CASA identity holds within ``tol``."""
    if df.height == 0:
        return float("nan")
    return float(identity_holds(df, tol).mean())


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


def _get_json(session, endpoint: str, params: dict, max_retries: int = 6) -> dict:
    """GET with exponential backoff on 429, 5xx and connection errors."""
    import requests

    url = f"{API_BASE}/{endpoint}"
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == max_retries:
                raise
            retry_after = None
            if status == 429 and exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            log.warning("%s %s failed (%s); retry %d in %.0fs", endpoint, params, exc, attempt + 1, wait)
            time.sleep(wait)
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def _paged(session, endpoint: str, params: dict, sleep: float) -> tuple[list[dict], int]:
    """Fetch every page of a query; returns (records, meta.total)."""
    records: list[dict] = []
    offset = 0
    total = None
    while True:
        payload = _get_json(session, endpoint, {**params, "limit": PAGE_LIMIT, "offset": offset})
        total = int(payload["meta"]["total"])
        batch = [row["data"] for row in payload["data"]]
        records.extend(batch)
        offset += len(batch)
        time.sleep(sleep)
        if not batch or offset >= total:
            break
    return records, total


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "CASA-forecasting research panel (python-requests)"
    return s


# --------------------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------------------


def fetch_quarter(session, repdte: str, sleep: float = 1.0) -> pl.DataFrame:
    params = {
        "filters": f"REPDTE:{repdte}",
        "fields": ",".join(FIELDS),
        "sort_by": "CERT",
        "sort_order": "ASC",
        "format": "json",
    }
    records, total = _paged(session, "financials", params, sleep)
    df = pl.from_dicts(
        [{k: r.get(k) for k in FIELDS} for r in records],
        schema={k: (pl.Float64 if k in NUMERIC_FIELDS else RAW_SCHEMA[k]) for k in FIELDS},
        strict=False,
    )
    df = df.unique(subset=["CERT", "REPDTE"], keep="first", maintain_order=True)
    if df.height != total:
        raise RuntimeError(f"{repdte}: got {df.height} unique rows, API reports {total}")
    return df


def ingest_financials(
    raw_dir: Path,
    start: str = "20010331",
    end: str = "20260630",
    sleep: float = 1.0,
    force: bool = False,
) -> list[Path]:
    """Download one parquet per quarter into ``raw_dir``; skips quarters already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    paths = []
    for repdte in quarter_ends(start, end):
        path = raw_dir / f"financials_{repdte}.parquet"
        paths.append(path)
        if path.exists() and not force:
            continue
        t0 = time.time()
        df = fetch_quarter(session, repdte, sleep=sleep)
        tmp = path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp)
        tmp.replace(path)
        log.info("%s: %d rows (%.1fs)", repdte, df.height, time.time() - t0)
    return paths


def ingest_history(
    path: Path, start_year: int = 2000, end_year: int | None = None, sleep: float = 1.0
) -> pl.DataFrame:
    """Merger, failure and cert-change events from /history, one call per EFFDATE year."""
    end_year = end_year or dt.date.today().year
    session = _session()
    codes = " OR ".join(str(c) for c in HISTORY_CODES)
    records: list[dict] = []
    for year in range(start_year, end_year + 1):
        params = {
            "filters": f"CHANGECODE:({codes}) AND EFFDATE:[{year}-01-01 TO {year}-12-31]",
            "fields": ",".join(HISTORY_FIELDS),
            "sort_by": "EFFDATE",
            "sort_order": "ASC",
        }
        recs, total = _paged(session, "history", params, sleep)
        if len(recs) != total:
            raise RuntimeError(f"history {year}: got {len(recs)} rows, API reports {total}")
        records.extend(recs)
    df = pl.from_dicts(
        [{k: r.get(k) for k in HISTORY_FIELDS} for r in records], schema=HISTORY_SCHEMA, strict=False
    )
    df = (
        df.unique(subset=["ID"], keep="first", maintain_order=True)
        .with_columns(pl.col("EFFDATE").str.slice(0, 10).str.to_date("%Y-%m-%d"))
        .sort("EFFDATE", "ID")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def load_raw(raw_dir: Path) -> pl.DataFrame:
    files = sorted(raw_dir.glob("financials_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cached quarters in {raw_dir}")
    return pl.concat([pl.read_parquet(f).cast(RAW_SCHEMA) for f in files], how="vertical")


# --------------------------------------------------------------------------------------
# Panel build
# --------------------------------------------------------------------------------------


def _with_dates(raw: pl.DataFrame) -> pl.DataFrame:
    return add_casa(raw).with_columns(pl.col("REPDTE").str.to_date("%Y%m%d").alias("ds")).with_columns(
        _qidx(pl.col("ds")).alias("q")
    )


def industry_totals(raw: pl.DataFrame) -> pl.DataFrame:
    """Sum of y across all reporting banks per quarter, before any filtering."""
    df = _with_dates(raw)
    return (
        df.group_by("ds")
        .agg(
            pl.col("y").sum().alias("y"),
            pl.len().cast(pl.Int64).alias("n_banks"),
            pl.col("DEPDOM").sum().alias("DEPDOM"),
            pl.col("NTRTIME").sum().alias("NTRTIME"),
            pl.col("TRN").sum().alias("TRN"),
            pl.col("NTRSMMDA").sum().alias("NTRSMMDA"),
            pl.col("NTRSOTH").sum().alias("NTRSOTH"),
        )
        .sort("ds")
    )


def merger_cuts(
    obs: pl.DataFrame, history: pl.DataFrame, min_share: float = 0.10
) -> pl.DataFrame:
    """Quarters at which a survivor's series must restart because it absorbed another bank.

    ``obs`` holds positive-y observations with columns CERT, q, y. A merger effective on
    EFFDATE first shows in the survivor's report for the quarter containing EFFDATE, so the
    new segment starts at that quarter. Mergers whose targets together held less than
    ``min_share`` of the survivor's y in the prior quarter are ignored. Targets with no
    report before the merger count as material.
    """
    empty = pl.DataFrame(schema={"CERT": pl.Int64, "q": pl.Int64})
    if history.height == 0:
        return empty
    ev = (
        history.filter(pl.col("CHANGECODE").is_in(list(MERGER_CODES)))
        .with_columns(pl.coalesce("SUR_CERT", "ACQ_CERT").alias("SUR"))
        .filter(pl.col("SUR").is_not_null() & pl.col("OUT_CERT").is_not_null())
        .filter(pl.col("SUR") != pl.col("OUT_CERT"))
        .select(
            pl.col("SUR").cast(pl.Int64),
            pl.col("OUT_CERT").cast(pl.Int64).alias("OUT"),
            _qidx(pl.col("EFFDATE")).alias("q"),
        )
        .unique()
    )
    if ev.height == 0:
        return empty

    obs_sorted = obs.select("CERT", "q", "y").sort("q")
    ev = ev.with_columns((pl.col("q") - 1).alias("q_prev")).sort("q_prev")
    # Last reported y of the target before the merger quarter.
    ev = ev.join_asof(
        obs_sorted.rename({"CERT": "OUT", "q": "q_out", "y": "y_out"}),
        left_on="q_prev",
        right_on="q_out",
        by="OUT",
        strategy="backward",
        check_sortedness=False,
    )
    per_q = ev.group_by("SUR", "q", "q_prev").agg(
        pl.col("y_out").sum().alias("y_out"),
        pl.col("y_out").is_null().any().alias("unknown_target"),
    )
    per_q = per_q.sort("q_prev").join_asof(
        obs_sorted.rename({"CERT": "SUR", "q": "q_sur", "y": "y_sur"}),
        left_on="q_prev",
        right_on="q_sur",
        by="SUR",
        strategy="backward",
        check_sortedness=False,
    )
    material = per_q.filter(
        pl.col("unknown_target")
        | pl.col("y_sur").is_null()
        | (pl.col("y_out") >= min_share * pl.col("y_sur"))
    )
    return material.select(pl.col("SUR").alias("CERT"), "q").unique()


def build_panel(
    raw: pl.DataFrame,
    history: pl.DataFrame,
    min_quarters: int = 40,
    max_abs_dlog: float = 1.0,
    merger_min_share: float = 0.10,
) -> tuple[pl.DataFrame, dict]:
    """Build the long CASA panel and a summary of what was dropped and why.

    Steps: drop non-positive or missing y; drop banks with fewer than ``min_quarters``
    observations; split each bank's series at gaps, material mergers (from ``history``) and
    quarter-on-quarter |log change| > ``max_abs_dlog``; keep the most recent segment with at
    least ``min_quarters`` quarters. Banks that stop reporting (failures) are kept.
    """
    df = _with_dates(raw).select("CERT", "ds", "q", "y")
    n_rows_in = df.height
    n_banks_in = df["CERT"].n_unique()

    df = df.filter(pl.col("y").is_not_null() & (pl.col("y") > 0))
    dropped_rows_nonpos = n_rows_in - df.height

    # Merger materiality needs every target's size, so compute cuts before dropping short banks.
    cuts = merger_cuts(df, history, min_share=merger_min_share).with_columns(
        pl.lit(True).alias("merger_cut")
    )

    counts = df.group_by("CERT").len()
    short = counts.filter(pl.col("len") < min_quarters)["CERT"]
    banks_no_positive = n_banks_in - counts.height
    df = df.filter(~pl.col("CERT").is_in(short.implode()))
    df = (
        df.sort("CERT", "q")
        .join(cuts, on=["CERT", "q"], how="left")
        .with_columns(
            pl.col("merger_cut").fill_null(False),
            (pl.col("CERT") != pl.col("CERT").shift(1)).fill_null(True).alias("first"),
            (pl.col("q") - pl.col("q").shift(1)).alias("dq"),
            (pl.col("y").log() - pl.col("y").log().shift(1)).abs().alias("adlog"),
        )
        .with_columns(
            (~pl.col("first") & (pl.col("dq") != 1)).alias("gap_cut"),
            (~pl.col("first") & (pl.col("dq") == 1) & (pl.col("adlog") > max_abs_dlog)).alias(
                "break_cut"
            ),
        )
        .with_columns((pl.col("merger_cut") & ~pl.col("first")).alias("merger_cut"))
        .with_columns(
            (pl.col("first") | pl.col("gap_cut") | pl.col("break_cut") | pl.col("merger_cut"))
            .cast(pl.Int64)
            .cum_sum()
            .alias("seg")
        )
    )
    cut_counts = {
        "merger": int(df["merger_cut"].sum()),
        "extreme_break": int((df["break_cut"] & ~df["merger_cut"]).sum()),
        "gap": int((df["gap_cut"] & ~df["merger_cut"]).sum()),
        "banks_with_merger_cut": int(df.filter("merger_cut")["CERT"].n_unique()),
        "banks_with_extreme_break": int(df.filter("break_cut")["CERT"].n_unique()),
    }

    seg_len = df.group_by("CERT", "seg").agg(pl.len().alias("n"), pl.col("q").max().alias("qmax"))
    best = (
        seg_len.filter(pl.col("n") >= min_quarters)
        .sort("qmax", descending=True)
        .group_by("CERT", maintain_order=True)
        .first()
        .select("CERT", "seg")
    )
    banks_after_len = df["CERT"].n_unique()
    no_clean = banks_after_len - best.height

    panel = (
        df.join(best, on=["CERT", "seg"], how="inner")
        .select(
            pl.col("CERT").cast(pl.String).alias("unique_id"),
            pl.col("ds"),
            pl.col("y").cast(pl.Float64),
        )
        .sort("unique_id", "ds")
    )

    n_banks = panel["unique_id"].n_unique()
    trimmed_rows = df.height - panel.height
    summary = {
        "row_count": panel.height,
        "bank_count": n_banks,
        "date_range": [
            str(panel["ds"].min()) if panel.height else None,
            str(panel["ds"].max()) if panel.height else None,
        ],
        "input": {"rows": n_rows_in, "banks": n_banks_in},
        "params": {
            "y": "TRN + NTRSMMDA + NTRSOTH ($ thousands)",
            "min_quarters": min_quarters,
            "max_abs_dlog": max_abs_dlog,
            "merger_min_share": merger_min_share,
            "merger_codes": list(MERGER_CODES),
            "segment_rule": "most recent clean segment with >= min_quarters quarters",
            "survivorship": "failed / merged-out banks kept (no survivorship filter)",
        },
        "dropped": {
            "non_positive_or_missing_y_rows": dropped_rows_nonpos,
            "banks_with_no_positive_y": banks_no_positive,
            "fewer_than_min_quarters": int(short.len()),
            "no_clean_segment_ge_min_quarters": no_clean,
            "rows_outside_kept_segment": trimmed_rows,
        },
        "cuts": cut_counts,
    }
    return panel, summary


def quarter_label(d: dt.date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"
