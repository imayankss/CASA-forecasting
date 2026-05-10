"""
markdown_report.py — GitHub-ready Markdown forecasting report.

Produces a portable .md file suitable for:
  • GitHub repository documentation
  • Confluence / Notion pages
  • Email / Slack summaries
  • Quick executive reviews

Usage
-----
    from src.reporting.markdown_report import generate_markdown_report
    path = generate_markdown_report(pipeline, full_insights, out_dir="reports")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.exceptions import ReportGenerationError, ReportExportError

_log = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# MARKDOWN HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _h1(text: str) -> str:
    return f"\n# {text}\n"

def _h2(text: str) -> str:
    return f"\n## {text}\n"

def _h3(text: str) -> str:
    return f"\n### {text}\n"

def _hr() -> str:
    return "\n---\n"

def _badge(label: str, value: str, colour: str = "blue") -> str:
    label_enc = label.replace(" ", "_").replace("%", "%25")
    value_enc = value.replace(" ", "_").replace("%", "%25").replace("#", "")
    return f"![{label}](https://img.shields.io/badge/{label_enc}-{value_enc}-{colour})"

def _df_to_md(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Convert DataFrame to Markdown table, including the index."""
    df2 = df.copy()
    for col in df2.select_dtypes(include=float).columns:
        df2[col] = df2[col].map(lambda x: f"{x:{float_fmt}}")
    return df2.to_markdown()


def _metric_badge_row(metrics: dict, model_name: str) -> str:
    mape  = metrics.get("MAPE",  0)
    rmse  = metrics.get("RMSE_%",0)
    r2    = metrics.get("R2",    0)
    stab  = metrics.get("Stability", 0)
    mc    = "brightgreen" if mape < 3 else "green" if mape < 6 else "yellow"
    return (
        f"{_badge('MAPE', f'{mape:.2f}%', mc)}  "
        f"{_badge('RMSE%', f'{rmse:.2f}%', 'blue')}  "
        f"{_badge('R2', f'{r2:.4f}', 'blueviolet')}  "
        f"{_badge('Stability', f'{stab:.0f}/100', 'informational')}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def _cover_section(full_insights: dict, pipeline) -> str:
    best  = full_insights.get("best_model", "—")
    ts    = full_insights.get("timestamp", "—")
    bm    = pipeline.insights().get("best_metrics", {})
    n_m   = len(pipeline.results)
    n_obs = len(pipeline.train) + len(pipeline.test)

    mape_str = f"{bm.get('MAPE', 0):.2f}pct"
    badges = (
        f"{_badge('Python', '3.10+', 'blue')}  "
        f"{_badge('Models', str(n_m), 'green')}  "
        f"{_badge('Best_Model', best, 'brightgreen')}  "
        f"{_badge('MAPE', mape_str, 'success')}  "
        f"{_badge('License', 'MIT', 'lightgrey')}"
    )

    kpi_table = (
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| 🏆 Best Model | **{best}** |\n"
        f"| 📉 Best MAPE | `{bm.get('MAPE',0):.2f}%` |\n"
        f"| 📐 Best R² | `{bm.get('R2',0):.4f}` |\n"
        f"| 📊 Models Evaluated | `{n_m}` |\n"
        f"| 🗓️ Data Points | `{n_obs} quarters` |\n"
        f"| ⏱️ Report Generated | `{ts}` |\n"
    )

    return (
        "# 📊 BOI CASA Deposit Forecasting Report\n\n"
        "> **Bank of India · Data Science Division**  \n"
        "> Time-Series Forecasting — CASA Deposits (2015–2024)\n\n"
        f"{badges}\n\n"
        f"{_hr()}\n"
        f"## 🎯 At a Glance\n\n{kpi_table}\n"
    )


def _executive_summary_section(full_insights: dict) -> str:
    out = _h2("1. Executive Summary")
    for para in full_insights["executive_summary"].split("\n\n"):
        out += f"\n{para.strip()}\n"
    return out


def _leaderboard_section(pipeline) -> str:
    lb   = pipeline.leaderboard()
    cols = [c for c in ["Rank","MAE_%","RMSE_%","MAPE","R2",
                         "Stability","Composite_Score","Train_Time_s"] if c in lb.columns]
    out  = _h2("2. Model Performance Leaderboard")
    out += "\n" + _df_to_md(lb[cols]) + "\n"
    out += (
        "\n> **Composite Score** = weighted combination of "
        "MAPE (35%), RMSE % (25%), MAE % (20%), Stability (20%). "
        "Higher is better.\n"
    )
    return out


def _forecast_interpretation_section(full_insights: dict) -> str:
    out  = _h2("3. Forecast Interpretation")
    out += f"\n{full_insights.get('forecast_interpretation', '—')}\n"
    out += _h3("Performance Context")
    out += f"\n{full_insights.get('performance_explanation', '—')}\n"
    return out


def _model_explanations_section(full_insights: dict, pipeline) -> str:
    out  = _h2("4. Model Explanations")
    exps = full_insights.get("model_explanations", {})
    results = pipeline.results
    for i, (mname, mtext) in enumerate(exps.items(), 1):
        out += _h3(f"4.{i} {mname}")
        metrics = results.get(mname, {}).get("metrics", {})
        if metrics:
            out += "\n" + _metric_badge_row(metrics, mname) + "\n\n"
        for line in mtext.split("\n"):
            out += f"{line}\n"
        out += "\n"
    return out


def _diagnostics_section(pipeline) -> str:
    out  = _h2("5. Residual Diagnostics")
    diags = pipeline.diagnostics
    if not diags:
        return out + "\n_No diagnostics available._\n"

    header = "| Model | Shapiro-Wilk | Jarque-Bera | Ljung-Box | Health Score |\n"
    sep    = "|---|---|---|---|---|\n"
    rows   = ""
    for name, d in diags.items():
        sw  = "✅ Normal"   if d.get("shapiro",{}).get("normal")          else "❌ Non-normal"
        jb  = "✅ Normal"   if d.get("jarque_bera",{}).get("normal")      else "❌ Non-normal"
        lbx = "✅ No AutoCorr" if d.get("ljung_box",{}).get("no_autocorr") else "❌ Autocorr."
        h   = d.get("health_score", 0)
        rows += f"| **{name}** | {sw} | {jb} | {lbx} | {h:.0f}/100 — {d.get('health_label','—')} |\n"

    out += f"\n{header}{sep}{rows}\n"
    out += (
        "> **Health Score** = % of diagnostic tests passed. "
        "75–100 = Excellent · 50–74 = Good · 25–49 = Fair · 0–24 = Poor.\n"
    )
    return out


def _cv_section(pipeline) -> str:
    out = _h2("6. Walk-Forward Cross-Validation")
    cv  = pipeline.cv_summary()
    if cv.empty:
        return out + "\n_Cross-validation was not run._\n"
    out += "\n" + _df_to_md(cv) + "\n"
    out += (
        "\n> **CV_MAPE_Mean** = average MAPE across all folds. "
        "**CV_MAPE_Std** = standard deviation across folds — lower indicates more stable generalisation.\n"
    )
    return out


def _banking_section(full_insights: dict) -> str:
    out = _h2("7. Banking Domain Analysis")
    for para in full_insights.get("banking_narrative", "—").split("\n\n"):
        out += f"\n{para.strip()}\n"
    return out


def _recommendations_section(full_insights: dict) -> str:
    out  = _h2("8. Recommendations")
    out += "\n"
    for line in full_insights.get("recommendations", "—").split("\n"):
        line = line.strip()
        if line:
            out += f"{line}\n\n"
    return out


def _appendix_section(pipeline) -> str:
    """Append a raw metrics table for technical readers."""
    out  = _h2("Appendix — Full Metrics Table")
    mdf  = pipeline.comparison.metrics_df()
    cols = [c for c in ["MAE","RMSE","MAPE","SMAPE","R2","Bias_%",
                         "Stability","Theil_U"] if c in mdf.columns]
    out += "\n" + _df_to_md(mdf[cols]) + "\n"
    return out


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(
    pipeline,
    full_insights: Dict,
    out_dir: str | Path = "reports",
    filename: Optional[str] = None,
) -> Path:
    """
    Generate a comprehensive GitHub-ready Markdown report.

    Parameters
    ----------
    pipeline      : completed ForecastingPipeline
    full_insights : output of build_full_insights(pipeline)
    out_dir       : directory to save the .md file
    filename      : optional custom filename

    Returns
    -------
    Path to the saved Markdown file.
    """
    _log.info("Generating Markdown report …")

    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"boi_casa_report_{ts}.md"
        out_path = out_dir / filename

        sections = [
            _cover_section(full_insights, pipeline),
            _hr(),
            _executive_summary_section(full_insights),
            _hr(),
            _leaderboard_section(pipeline),
            _forecast_interpretation_section(full_insights),
            _hr(),
            _model_explanations_section(full_insights, pipeline),
            _hr(),
            _diagnostics_section(pipeline),
            _cv_section(pipeline),
            _hr(),
            _banking_section(full_insights),
            _recommendations_section(full_insights),
            _hr(),
            _appendix_section(pipeline),
            _hr(),
            (
                f"\n*Report generated by BOI CASA Forecasting Platform v1.0 "
                f"· {full_insights.get('timestamp', '')}*\n"
            ),
        ]

        content = "\n".join(sections)
        out_path.write_text(content, encoding="utf-8")

    except Exception as exc:
        raise ReportGenerationError("Markdown", str(exc)) from exc

    _log.info("Markdown saved → %s  (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path
