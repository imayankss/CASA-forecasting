"""Build data/real/boi_casa_quarterly.csv from Bank of India's NSE filings.

Usage: python run_boi_real.py

Downloads (and caches under data/external/boi_filings/) BOI's press releases
and investor presentations, extracts quarter-end CASA figures and writes one
row per quarter-end from 31-Mar-2015 to 30-Jun-2026.
"""
from __future__ import annotations

import logging

from src.real_data.boi_filings import FIGURES, NSEClient, build_rows, quarter_ends, verify_manual, write_csv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = NSEClient()
    for problem in verify_manual(client):
        logging.warning("manual value not found in text layer (image slide?): %s", problem)
    rows = build_rows(client)
    path = write_csv(rows)
    have = {r.quarter_end for r in rows}
    missing = [q.isoformat() for q in quarter_ends() if q not in have]
    print(f"wrote {len(rows)} rows to {path}")
    print(f"quarter-ends missing: {missing or 'none'}")
    for f in FIGURES:
        blank = [r.quarter_end.isoformat() for r in rows if f not in r.values]
        if blank:
            print(f"  {f} blank for {len(blank)} quarters: {blank[0]} .. {blank[-1]}")


if __name__ == "__main__":
    main()
