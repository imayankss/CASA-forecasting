"""
html_report.py — Self-contained dark-themed HTML forecasting report.

Produces a single portable .html file with:
  • Inline CSS (Bloomberg dark theme)
  • All text sections
  • Embedded Plotly interactive charts (forecast comparison, metric bars, radar)
  • Model leaderboard table
  • Residual diagnostics summary table
  • CV results table
  • Downloadable from any browser — no server needed

Usage
-----
    from src.reporting.html_report import generate_html_report
    path = generate_html_report(pipeline, full_insights, out_dir="reports")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.exceptions import ReportGenerationError, ReportExportError

_log = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (inline Plotly JSON)
# ═════════════════════════════════════════════════════════════════════════════

def _plotly_json(fig) -> str:
    """Serialise a Plotly figure to a JSON string for inline embedding."""
    import plotly.io as pio
    return pio.to_json(fig)


def _forecast_chart_json(pipeline) -> str:
    import plotly.graph_objects as go
    train, test = pipeline.train, pipeline.test
    results = pipeline.results

    PAL = {"ARIMA":"#3B82F6","SARIMA":"#10B981","SARIMAX":"#F59E0B",
           "AutoARIMA":"#EF4444","Prophet":"#EC4899"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index.astype(str).tolist(), y=train.values.tolist(),
        name="Train Actual", line=dict(color="#94A3B8", width=2), mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=test.index.astype(str).tolist(), y=test.values.tolist(),
        name="Test Actual",
        line=dict(color="#F1F5F9", width=3, dash="dot"),
        mode="lines+markers", marker=dict(size=9, symbol="diamond"),
    ))
    for name, rec in results.items():
        fc  = rec["forecast"]
        ci  = rec["ci"]
        col = PAL.get(name, "#3B82F6")
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fig.add_trace(go.Scatter(
            x=fc.index.astype(str).tolist(), y=fc.values.tolist(),
            name=name, line=dict(color=col, width=2.5),
            mode="lines+markers", marker=dict(size=6),
        ))
        ci_x = list(ci.index.astype(str)) + list(ci.index.astype(str))[::-1]
        ci_y = list(ci["upper"]) + list(ci["lower"])[::-1]
        fig.add_trace(go.Scatter(
            x=ci_x, y=ci_y, fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{name} CI", showlegend=False,
        ))

    fig.update_layout(
        title="<b>Forecast vs Actual — All Models</b>",
        paper_bgcolor="#080E1A", plot_bgcolor="#0F1D35",
        font=dict(color="#F1F5F9", family="Inter, sans-serif"),
        xaxis=dict(title="Quarter", gridcolor="#1E2D4A"),
        yaxis=dict(title="Deposit Amount (₹)", tickformat=",.0f", gridcolor="#1E2D4A"),
        legend=dict(bgcolor="#0D1526", bordercolor="#1E2D4A"),
        hovermode="x unified", height=480,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return _plotly_json(fig)


def _metric_bar_json(pipeline) -> str:
    import plotly.graph_objects as go
    lb      = pipeline.leaderboard()
    models  = lb.index.tolist()
    metrics = ["MAE_%", "RMSE_%", "MAPE"]
    PAL     = ["#3B82F6", "#10B981", "#F59E0B"]
    fig     = go.Figure()
    for col, colour in zip(metrics, PAL):
        if col not in lb.columns:
            continue
        fig.add_trace(go.Bar(
            name=col.replace("_", " "),
            x=models, y=lb[col].tolist(),
            marker_color=colour, opacity=0.88,
            text=[f"{v:.2f}%" for v in lb[col]],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="<b>Model Error Metrics Comparison (%)</b>",
        paper_bgcolor="#080E1A", plot_bgcolor="#0F1D35",
        font=dict(color="#F1F5F9"), height=400,
        xaxis=dict(gridcolor="#1E2D4A"),
        yaxis=dict(gridcolor="#1E2D4A", title="Error (%)"),
        legend=dict(bgcolor="#0D1526", bordercolor="#1E2D4A"),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return _plotly_json(fig)


def _radar_json(pipeline) -> str:
    import plotly.graph_objects as go
    lb     = pipeline.leaderboard()
    cols   = [c for c in ["MAE_%","RMSE_%","MAPE","Stability","R2"] if c in lb.columns][:5]
    norm   = lb[cols].copy().fillna(lb[cols].max())
    for c in cols:
        rng = norm[c].max() - norm[c].min() + 1e-9
        if c in ("Stability", "R2"):
            norm[c] = (norm[c] - norm[c].min()) / rng
        else:
            norm[c] = 1 - (norm[c] - norm[c].min()) / rng

    PAL    = {"ARIMA":"#3B82F6","SARIMA":"#10B981","SARIMAX":"#F59E0B",
              "AutoARIMA":"#EF4444","Prophet":"#EC4899"}
    labels = [c.replace("_", " ") for c in cols]
    fig    = go.Figure()
    for model in lb.index:
        vals = norm.loc[model].tolist()
        vals_closed = vals + [vals[0]]
        col = PAL.get(model, "#3B82F6")
        r, g, b = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels + [labels[0]],
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(color=col, width=2), name=model,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#1E2D4A"),
            angularaxis=dict(gridcolor="#1E2D4A"),
            bgcolor="#0F1D35",
        ),
        title="<b>Model Radar Chart — Higher = Better</b>",
        paper_bgcolor="#080E1A",
        font=dict(color="#F1F5F9"), height=460,
        legend=dict(bgcolor="#0D1526", bordercolor="#1E2D4A"),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return _plotly_json(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TABLE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _html_table(df: pd.DataFrame, highlight_col: Optional[str] = None) -> str:
    """Render a DataFrame as a styled HTML table."""
    rows_html = ""
    for i, (idx, row) in enumerate(df.iterrows()):
        bg = "#0F1D35" if i % 2 == 0 else "#0D1526"
        row_html = f'<tr style="background:{bg};">'
        row_html += f'<td style="padding:8px 12px;font-weight:600;color:#60A5FA;">{idx}</td>'
        for col, val in row.items():
            cell_col = "#F1F5F9"
            if highlight_col and col == highlight_col:
                cell_col = "#10B981"
            if isinstance(val, float):
                val_str = f"{val:.3f}"
            elif isinstance(val, bool):
                val_str = "✓" if val else "✗"
            else:
                val_str = str(val)
            row_html += (
                f'<td style="padding:8px 12px;color:{cell_col};'
                f'text-align:center;">{val_str}</td>'
            )
        row_html += "</tr>"
        rows_html += row_html

    headers = "".join(
        f'<th style="padding:10px 12px;background:#0C1529;'
        f'color:#60A5FA;text-align:center;'
        f'border-bottom:2px solid #1E3A5F;font-size:0.82rem;'
        f'letter-spacing:0.06em;text-transform:uppercase;">{c}</th>'
        for c in ["Model"] + df.columns.tolist()
    )
    return (
        '<table style="width:100%;border-collapse:collapse;'
        'font-size:0.88rem;font-family:Inter,sans-serif;">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION RENDERERS → HTML strings
# ═════════════════════════════════════════════════════════════════════════════

def _section(title: str, content: str, section_id: str = "") -> str:
    return f"""
    <section id="{section_id}" style="margin-bottom:48px;">
        <h2 style="color:#60A5FA;font-size:1.4rem;font-weight:700;
                   border-bottom:2px solid #1E3A5F;padding-bottom:10px;
                   margin-bottom:20px;">{title}</h2>
        {content}
    </section>
    """


def _chart_div(chart_json: str, div_id: str) -> str:
    return f"""
    <div id="{div_id}" style="border-radius:12px;overflow:hidden;
         border:1px solid #1E2D4A;margin:20px 0;"></div>
    <script>
        Plotly.react("{div_id}", {chart_json});
    </script>
    """


def _kpi_card(value: str, label: str, colour: str = "#60A5FA") -> str:
    return f"""
    <div style="background:#0F1D35;border:1px solid #1E3A5F;border-radius:12px;
                padding:22px 18px;text-align:center;flex:1;min-width:130px;">
        <div style="font-size:1.6rem;font-weight:700;color:{colour};
                    font-family:'JetBrains Mono',monospace;">{value}</div>
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;
                    letter-spacing:0.1em;margin-top:6px;">{label}</div>
    </div>
    """


# ═════════════════════════════════════════════════════════════════════════════
# FULL HTML ASSEMBLY
# ═════════════════════════════════════════════════════════════════════════════

def generate_html_report(
    pipeline,
    full_insights: Dict,
    out_dir: str | Path = "reports",
    filename: Optional[str] = None,
) -> Path:
    """
    Generate a self-contained dark-themed HTML forecasting report.

    Parameters
    ----------
    pipeline      : completed ForecastingPipeline
    full_insights : output of build_full_insights(pipeline)
    out_dir       : directory to save the HTML file
    filename      : optional custom filename

    Returns
    -------
    Path to the saved HTML file.
    """
    _log.info("Generating HTML report …")

    try:
        out_dir  = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"boi_casa_report_{ts}.html"
        out_path = out_dir / filename

        lb       = pipeline.leaderboard()
        cv       = pipeline.cv_summary()
        diags    = pipeline.diagnostics
        best     = full_insights.get("best_model", "—")
        ts_str   = full_insights.get("timestamp", "—")
        bm       = pipeline.insights().get("best_metrics", {})

        # ── Charts ────────────────────────────────────────────────────────
        fc_json  = _forecast_chart_json(pipeline)
        bar_json = _metric_bar_json(pipeline)
        rad_json = _radar_json(pipeline)

        # ── KPI row ───────────────────────────────────────────────────────
        kpi_html = f"""
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:32px;">
            {_kpi_card(f"{bm.get('MAPE',0):.2f}%",    "Best MAPE",       "#10B981")}
            {_kpi_card(f"{bm.get('R2',0):.4f}",        "Best R²",         "#3B82F6")}
            {_kpi_card(f"{bm.get('RMSE_%',0):.2f}%",   "Best RMSE %",     "#F59E0B")}
            {_kpi_card(f"{bm.get('Stability',0):.1f}",  "Stability /100",  "#8B5CF6")}
            {_kpi_card(str(len(pipeline.results)),       "Models Trained",  "#EC4899")}
        </div>
        """

        # ── Leaderboard table ─────────────────────────────────────────────
        lb_cols  = [c for c in ["Rank","MAE_%","RMSE_%","MAPE","R2",
                                 "Stability","Composite_Score"] if c in lb.columns]
        lb_table = _html_table(lb[lb_cols], highlight_col="Composite_Score")

        # ── Diagnostics table ─────────────────────────────────────────────
        diag_rows = ""
        for name, d in diags.items():
            sw  = "✓" if d.get("shapiro",{}).get("normal")          else "✗"
            jb  = "✓" if d.get("jarque_bera",{}).get("normal")      else "✗"
            lbx = "✓" if d.get("ljung_box",{}).get("no_autocorr")   else "✗"
            h   = d.get("health_score", 0)
            hc  = "#10B981" if h >= 75 else "#F59E0B" if h >= 50 else "#EF4444"
            diag_rows += f"""
            <tr style="border-bottom:1px solid #1E2D4A;">
                <td style="padding:10px 14px;color:#60A5FA;font-weight:600;">{name}</td>
                <td style="padding:10px;text-align:center;
                           color:{'#10B981' if sw=='✓' else '#EF4444'};">{sw}</td>
                <td style="padding:10px;text-align:center;
                           color:{'#10B981' if jb=='✓' else '#EF4444'};">{jb}</td>
                <td style="padding:10px;text-align:center;
                           color:{'#10B981' if lbx=='✓' else '#EF4444'};">{lbx}</td>
                <td style="padding:10px;text-align:center;color:{hc};font-weight:600;">
                    {h:.0f}/100 — {d.get('health_label','—')}</td>
            </tr>"""
        diag_table = f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
            <thead>
                <tr style="background:#0C1529;color:#60A5FA;font-size:0.78rem;
                            text-transform:uppercase;letter-spacing:0.06em;">
                    <th style="padding:10px 14px;text-align:left;">Model</th>
                    <th style="padding:10px;">Shapiro-Wilk</th>
                    <th style="padding:10px;">Jarque-Bera</th>
                    <th style="padding:10px;">Ljung-Box</th>
                    <th style="padding:10px;">Health Score</th>
                </tr>
            </thead>
            <tbody>{diag_rows}</tbody>
        </table>"""

        # ── CV table ──────────────────────────────────────────────────────
        cv_html = _html_table(cv) if not cv.empty else "<p>CV was not run.</p>"

        # ── Model explanations ────────────────────────────────────────────
        model_exp_html = ""
        for mname, mtext in full_insights.get("model_explanations", {}).items():
            mtext_html = "".join(
                f"<p style='margin:4px 0;color:#CBD5E1;line-height:1.6;'>{l}</p>"
                if l.strip() else "<br>"
                for l in mtext.split("\n")
            )
            model_exp_html += f"""
            <div style="background:#0F1D35;border:1px solid #1E3A5F;
                        border-left:4px solid #2563EB;border-radius:0 10px 10px 0;
                        padding:20px 24px;margin-bottom:20px;">
                <h3 style="color:#60A5FA;margin:0 0 12px;font-size:1.05rem;">{mname}</h3>
                {mtext_html}
            </div>"""

        # ── Recommendations ───────────────────────────────────────────────
        recs_html = "".join(
            f"<li style='margin-bottom:10px;color:#CBD5E1;line-height:1.6;'>{r}</li>"
            for r in full_insights.get("recommendations", "").split("\n") if r.strip()
        )

        # ── Full HTML ──────────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BOI CASA Deposit Forecasting Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&
family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#080E1A; color:#E2E8F0; font-family:'Inter',sans-serif;
          line-height:1.6; }}
  a {{ color:#60A5FA; text-decoration:none; }}
  a:hover {{ color:#93C5FD; }}
  .container {{ max-width:1100px; margin:0 auto; padding:40px 28px; }}
  h1 {{ font-size:2rem; font-weight:800; color:#F1F5F9; letter-spacing:-0.03em;
        margin-bottom:6px; }}
  h2 {{ font-size:1.35rem; font-weight:700; color:#60A5FA; }}
  h3 {{ font-size:1.05rem; font-weight:600; color:#CBD5E1; }}
  p  {{ color:#CBD5E1; line-height:1.7; margin-bottom:12px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:999px;
            font-size:0.7rem; font-weight:600; letter-spacing:0.06em;
            text-transform:uppercase; border:1px solid; }}
  .badge-blue   {{ background:rgba(37,99,235,.15); color:#60A5FA; border-color:#2563EB; }}
  .badge-green  {{ background:rgba(16,185,129,.15); color:#10B981; border-color:#10B981; }}
  nav {{ background:#0D1526; border-bottom:1px solid #1E2D4A; padding:14px 28px;
         position:sticky; top:0; z-index:100; display:flex; gap:20px;
         font-size:0.82rem; }}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <span style="color:#60A5FA;font-weight:700;margin-right:8px;">📊 BOI CASA</span>
  <a href="#summary">Summary</a>
  <a href="#leaderboard">Leaderboard</a>
  <a href="#forecast">Forecast</a>
  <a href="#models">Models</a>
  <a href="#diagnostics">Diagnostics</a>
  <a href="#cv">Cross-Val</a>
  <a href="#banking">Banking</a>
  <a href="#recs">Recommendations</a>
</nav>

<div class="container">

<!-- HEADER -->
<div style="padding:40px 0 32px;border-bottom:1px solid #1E2D4A;margin-bottom:40px;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:10px;">
    <span style="font-size:2.2rem;">📊</span>
    <h1>BOI CASA Deposit Forecasting Report</h1>
  </div>
  <p style="color:#64748B;margin-bottom:12px;">
    Bank of India · Data Science Division
    &nbsp;·&nbsp; Generated: <b style="color:#94A3B8;">{ts_str}</b>
    &nbsp;·&nbsp; Best Model: <span class="badge badge-green">{best}</span>
  </p>
  {kpi_html}
</div>

<!-- 1. EXECUTIVE SUMMARY -->
{_section("1. Executive Summary",
    "".join(f"<p>{p.strip()}</p>" for p in
            full_insights["executive_summary"].split("\n\n")),
    "summary")}

<!-- 2. LEADERBOARD -->
{_section("2. Model Performance Leaderboard",
    lb_table + "<p style='font-size:0.8rem;color:#64748B;margin-top:8px;'>"
    "Composite Score = weighted MAPE (35%) + RMSE % (25%) + MAE % (20%) + Stability (20%).</p>",
    "leaderboard")}

<!-- FORECAST CHART -->
{_section("3. Forecast vs Actual",
    _chart_div(fc_json, "fc-chart") +
    "<p style='font-size:0.82rem;color:#64748B;'>" +
    full_insights.get("forecast_interpretation","") + "</p>",
    "forecast")}

<!-- METRIC CHARTS -->
{_section("4. Model Comparison Charts",
    _chart_div(bar_json, "bar-chart") +
    _chart_div(rad_json, "radar-chart") +
    "<p style='font-size:0.82rem;color:#64748B;'>" +
    full_insights.get("performance_explanation","") + "</p>",
    "comparison")}

<!-- MODEL EXPLANATIONS -->
{_section("5. Model Explanations", model_exp_html, "models")}

<!-- DIAGNOSTICS -->
{_section("6. Residual Diagnostics", diag_table, "diagnostics")}

<!-- CROSS-VALIDATION -->
{_section("7. Walk-Forward Cross-Validation",
    cv_html + "<p style='font-size:0.8rem;color:#64748B;margin-top:8px;'>"
    "Expanding-window CV — lower CV_MAPE_Mean with low Std indicates robust generalisation.</p>",
    "cv")}

<!-- BANKING NARRATIVE -->
{_section("8. Banking Domain Analysis",
    "".join(f"<p>{p.strip()}</p>" for p in
            full_insights.get("banking_narrative","").split("\n\n")),
    "banking")}

<!-- RECOMMENDATIONS -->
{_section("9. Recommendations",
    f"<ol style='padding-left:20px;'>{recs_html}</ol>",
    "recs")}

<!-- FOOTER -->
<footer style="border-top:1px solid #1E2D4A;padding-top:24px;margin-top:40px;
               color:#334155;font-size:0.78rem;text-align:center;">
  BOI CASA Deposit Forecasting Platform v1.0 ·
  Confidential — Internal Use Only ·
  BOI Data Science Internship Project
</footer>

</div>
</body>
</html>"""

        out_path.write_text(html, encoding="utf-8")

    except Exception as exc:
        raise ReportGenerationError("HTML", str(exc)) from exc

    _log.info("HTML saved → %s  (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path
