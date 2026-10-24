"""Pull FDIC Call Report deposits and build the US bank CASA panel.

    .venv/bin/python run_panel_ingest.py                 # ingest (resumable) + build
    .venv/bin/python run_panel_ingest.py --skip-ingest   # rebuild from the cache only

Outputs (all under data/external/fdic/, gitignored):
    raw/financials_<REPDTE>.parquet  one file per quarter-end
    history.parquet                  merger / failure / cert-change events
    panel.parquet                    unique_id (CERT), ds, y  ($ thousands)
    panel_summary.json               counts, date range, drops and cuts
    industry.parquet                 sum of y over all reporters per quarter (unfiltered)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import polars as pl

from src.panel import fdic

OUT_DIR = Path(__file__).resolve().parent / "data" / "external" / "fdic"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="20010331")
    p.add_argument("--end", default="20260630")
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between API calls")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--refresh-history", action="store_true")
    p.add_argument("--min-quarters", type=int, default=40)
    p.add_argument("--max-abs-dlog", type=float, default=1.0)
    p.add_argument("--merger-min-share", type=float, default=0.10,
                   help="ignore mergers whose targets held less than this share of the survivor y")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_dir = OUT_DIR / "raw"
    hist_path = OUT_DIR / "history.parquet"

    if not args.skip_ingest:
        fdic.ingest_financials(raw_dir, args.start, args.end, sleep=args.sleep)
        if args.refresh_history or not hist_path.exists():
            fdic.ingest_history(hist_path, sleep=args.sleep)

    raw = fdic.load_raw(raw_dir)
    raw = raw.filter(pl.col("REPDTE").is_between(pl.lit(args.start), pl.lit(args.end)))
    history = pl.read_parquet(hist_path)
    print(f"raw: {raw.height:,} bank-quarter rows, {raw['CERT'].n_unique():,} banks, "
          f"{raw['REPDTE'].n_unique()} quarters; history: {history.height:,} events")

    # CASA identity on the real data.
    with_y = fdic.add_casa(raw)
    nonnull = with_y.filter(pl.col("y").is_not_null())
    hit_1k = fdic.identity_hit_rate(with_y, tol=1.0)
    hit_0 = fdic.identity_hit_rate(with_y, tol=0.0)
    hit_1k_nn = fdic.identity_hit_rate(nonnull, tol=1.0)
    # Four separately rounded $k items vs a rounded total: up to ~$2-3k of pure rounding.
    hit_3k_nn = fdic.identity_hit_rate(nonnull, tol=3.0)
    n_null = with_y.height - nonnull.height
    print(f"identity y == DEPDOM - NTRTIME: within $1k {hit_1k:.4%} of all rows "
          f"({hit_1k_nn:.4%} of rows with y), within $3k {hit_3k_nn:.4%} of rows with y, "
          f"exact {hit_0:.4%}; null y rows {n_null:,}")

    industry = fdic.industry_totals(raw)
    industry.write_parquet(OUT_DIR / "industry.parquet")
    ind = {r["ds"]: r["y"] for r in industry.iter_rows(named=True)}
    q4, q1 = dt.date(2022, 12, 31), dt.date(2023, 3, 31)
    drop_2023q1 = None
    if q4 in ind and q1 in ind:
        drop_2023q1 = {
            "2022Q4_bn": ind[q4] / 1e6,
            "2023Q1_bn": ind[q1] / 1e6,
            "change_bn": (ind[q1] - ind[q4]) / 1e6,
            "change_pct": (ind[q1] / ind[q4] - 1) * 100,
        }
        print(f"industry y 2022Q4 -> 2023Q1: {drop_2023q1['2022Q4_bn']:,.0f} -> "
              f"{drop_2023q1['2023Q1_bn']:,.0f} $bn ({drop_2023q1['change_bn']:+,.0f} $bn, "
              f"{drop_2023q1['change_pct']:+.2f}%)")

    panel, summary = fdic.build_panel(
        raw,
        history,
        min_quarters=args.min_quarters,
        max_abs_dlog=args.max_abs_dlog,
        merger_min_share=args.merger_min_share,
    )
    panel.write_parquet(OUT_DIR / "panel.parquet")

    for cert, label in [("24735", "SVB"), ("57053", "Signature"), ("59017", "First Republic")]:
        s = panel.filter(pl.col("unique_id") == cert)
        summary.setdefault("failed_bank_checks", {})[label] = (
            {"cert": cert, "quarters": s.height, "last": str(s["ds"].max())} if s.height else
            {"cert": cert, "quarters": 0}
        )
    summary["identity"] = {
        "within_1k_share": hit_1k,
        "within_1k_share_nonnull_y": hit_1k_nn,
        "within_3k_share_nonnull_y": hit_3k_nn,
        "exact_share": hit_0,
        "null_y_rows": n_null,
    }
    summary["raw"] = {
        "rows": raw.height,
        "banks": raw["CERT"].n_unique(),
        "quarters": raw["REPDTE"].n_unique(),
        "history_events": history.height,
    }
    summary["industry_2023q1_change"] = drop_2023q1
    (OUT_DIR / "panel_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
