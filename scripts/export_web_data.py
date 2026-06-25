"""CLI wrapper for exporting static JSON used by the Next.js dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export.web_exporter import WebDataExporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CASA dashboard JSON files.")
    parser.add_argument("--root-dir", default=str(ROOT), help="Project root directory.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "web" / "public" / "data"),
        help="Directory where JSON files should be generated.",
    )
    args = parser.parse_args()

    exporter = WebDataExporter(root_dir=args.root_dir, output_dir=args.output_dir)
    generated = exporter.export_all()
    print(f"\nGenerated {len(generated)} web data files.")


if __name__ == "__main__":
    main()
