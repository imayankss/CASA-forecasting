"""
pdf_report.py — Professional PDF Forecasting Report Generator.

Uses ReportLab to produce a multi-section, branded PDF including:
  • Cover page with KPI summary
  • Executive summary
  • Model performance table
  • Per-model explanation and metrics
  • Forecast interpretation
  • Banking narrative
  • Residual diagnostics summary
  • Cross-validation results
  • Recommendations

Usage
-----
    from src.reporting.pdf_report import generate_pdf_report
    path = generate_pdf_report(pipeline, out_dir="reports")
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.exceptions import ReportGenerationError, ReportExportError

_log = get_logger(__name__)

# ── Colours (R,G,B in 0–1 scale for ReportLab) ───────────────────────────────
_NAVY   = (0.047, 0.082, 0.161)   # #0C1529
_BLUE   = (0.149, 0.306, 0.922)   # #264EEB
_TEAL   = (0.039, 0.722, 0.510)   # #0AB882
_AMBER  = (0.961, 0.620, 0.043)   # #F59E0B
_RED    = (0.937, 0.267, 0.267)   # #EF4444
_WHITE  = (1.0,   1.0,   1.0)
_LIGHT  = (0.949, 0.953, 0.961)   # #F2F3F5
_MUTED  = (0.580, 0.627, 0.714)   # #94A0B6
_DARK   = (0.133, 0.157, 0.220)   # #222838


def _rl_colour(rgb: tuple):
    """Convert (r, g, b) 0-1 tuple to ReportLab Color."""
    from reportlab.lib.colors import Color
    return Color(*rgb)


# ═════════════════════════════════════════════════════════════════════════════
# STYLE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _get_styles():
    """Return a dict of ReportLab ParagraphStyles."""
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

    base = getSampleStyleSheet()
    c_navy  = _rl_colour(_NAVY)
    c_blue  = _rl_colour(_BLUE)
    c_white = _rl_colour(_WHITE)
    c_muted = _rl_colour(_MUTED)
    c_dark  = _rl_colour(_DARK)

    styles = {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold", fontSize=28,
            textColor=c_white, leading=36,
            alignment=TA_CENTER, spaceAfter=6,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontName="Helvetica", fontSize=13,
            textColor=_rl_colour(_MUTED), leading=18,
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            fontName="Helvetica", fontSize=10,
            textColor=_rl_colour(_MUTED), leading=14,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontName="Helvetica-Bold", fontSize=18,
            textColor=c_navy, spaceBefore=18, spaceAfter=8,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=13,
            textColor=c_blue, spaceBefore=12, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontName="Helvetica-BoldOblique", fontSize=11,
            textColor=c_dark, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontName="Helvetica", fontSize=10,
            textColor=c_dark, leading=15, spaceAfter=8,
            alignment=TA_JUSTIFY,
        ),
        "body_bullet": ParagraphStyle(
            "body_bullet",
            fontName="Helvetica", fontSize=10,
            textColor=c_dark, leading=14,
            leftIndent=16, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=c_muted, alignment=TA_CENTER, spaceAfter=6,
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value",
            fontName="Helvetica-Bold", fontSize=18,
            textColor=c_blue, alignment=TA_CENTER, leading=22,
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label",
            fontName="Helvetica", fontSize=8,
            textColor=c_muted, alignment=TA_CENTER, leading=10,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica", fontSize=8,
            textColor=c_muted, alignment=TA_CENTER,
        ),
        "mono": ParagraphStyle(
            "mono",
            fontName="Courier", fontSize=9,
            textColor=c_dark, leading=13, spaceAfter=4,
        ),
        "verdict_pass": ParagraphStyle(
            "verdict_pass",
            fontName="Helvetica-Bold", fontSize=10,
            textColor=_rl_colour(_TEAL),
        ),
        "verdict_fail": ParagraphStyle(
            "verdict_fail",
            fontName="Helvetica-Bold", fontSize=10,
            textColor=_rl_colour(_RED),
        ),
    }
    return styles


# ═════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _hr(canvas, doc, y_offset: float = 0, colour=None, width: float = 1):
    """Draw a horizontal rule on the current page."""
    from reportlab.lib.units import mm
    colour = colour or _rl_colour(_BLUE)
    canvas.setStrokeColor(colour)
    canvas.setLineWidth(width)
    margin = doc.leftMargin
    page_w = doc.pagesize[0] - doc.rightMargin - margin
    canvas.line(margin, y_offset, margin + page_w, y_offset)


def _coloured_table_style(header_bg=None, row_alt=None):
    """Return a ReportLab TableStyle with dark header and alternating rows."""
    from reportlab.platypus import TableStyle
    from reportlab.lib.colors import Color

    header_bg = header_bg or _rl_colour(_NAVY)
    row_alt   = row_alt   or _rl_colour(_LIGHT)

    return TableStyle([
        # Header
        ("BACKGROUND",  (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",   (0, 0), (-1, 0), _rl_colour(_WHITE)),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("TOPPADDING",    (0, 0), (-1, 0), 7),
        # Data rows
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_rl_colour(_WHITE), row_alt]),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.4, _rl_colour(_MUTED)),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 0), (0, -1), "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def _build_cover(styles: dict, full_insights: dict, pipeline) -> list:
    """Return flowables for the cover page."""
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
    )
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import Color

    S = styles
    elems = []

    elems.append(Spacer(1, 40))
    elems.append(Paragraph("BOI CASA DEPOSIT", S["cover_title"]))
    elems.append(Paragraph("Forecasting &amp; Analytics Report", S["cover_title"]))
    elems.append(Spacer(1, 8))
    elems.append(HRFlowable(
        width="80%", thickness=2,
        color=_rl_colour(_BLUE), spaceAfter=12
    ))
    elems.append(Paragraph("Bank of India · Data Science Division", S["cover_sub"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        f"Report generated: {full_insights.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}",
        S["cover_meta"]
    ))
    elems.append(Spacer(1, 36))

    # KPI cards row
    lb   = pipeline.leaderboard()
    best = full_insights.get("best_model", "—")
    bm   = pipeline.insights().get("best_metrics", {})

    kpi_data = [
        [
            Paragraph(f"{bm.get('MAPE', 0):.2f}%",  S["kpi_value"]),
            Paragraph(f"{bm.get('R2', 0):.4f}",     S["kpi_value"]),
            Paragraph(f"{len(pipeline.results)}",    S["kpi_value"]),
            Paragraph(str(len(pipeline.train) + len(pipeline.test)), S["kpi_value"]),
        ],
        [
            Paragraph("Best MAPE",      S["kpi_label"]),
            Paragraph("Best R²",        S["kpi_label"]),
            Paragraph("Models Trained", S["kpi_label"]),
            Paragraph("Data Points",    S["kpi_label"]),
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[110, 110, 110, 110])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), _rl_colour(_NAVY)),
        ("TEXTCOLOR",   (0, 0), (-1, -1), _rl_colour(_WHITE)),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0,0), (-1, -1), 14),
        ("GRID",        (0, 0), (-1, -1), 0.5, _rl_colour(_BLUE)),
        ("ROUNDEDCORNERS", [6]),
    ]))
    elems.append(kpi_table)
    elems.append(Spacer(1, 32))

    # Best model callout
    elems.append(Paragraph(
        f"🏆 Best Model: <b>{best}</b> — "
        f"Composite Score {lb.loc[best, 'Composite_Score']:.1f}/100",
        S["body"]
    ))

    elems.append(Spacer(1, 24))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_MUTED)))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(
        "CONFIDENTIAL — Internal Use Only · BOI Data Science Internship Project",
        S["footer"]
    ))
    elems.append(PageBreak())
    return elems


def _build_executive_summary(styles: dict, full_insights: dict) -> list:
    """Return flowables for the Executive Summary section."""
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    S = styles
    elems = []
    elems.append(Paragraph("1. Executive Summary", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))

    for para in full_insights["executive_summary"].split("\n\n"):
        elems.append(Paragraph(para.strip(), S["body"]))
        elems.append(Spacer(1, 4))
    return elems


def _build_leaderboard(styles: dict, lb: pd.DataFrame) -> list:
    """Return flowables for the Model Leaderboard section."""
    from reportlab.platypus import Paragraph, Spacer, Table, HRFlowable
    S = styles
    elems = []
    elems.append(Paragraph("2. Model Performance Leaderboard", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))

    cols_display = [c for c in ["Rank", "MAE_%", "RMSE_%", "MAPE", "R2",
                                 "Stability", "Composite_Score"] if c in lb.columns]
    headers = [c.replace("_", " ") for c in cols_display]

    table_data = [headers]
    for idx, row in lb[cols_display].iterrows():
        table_data.append([
            str(int(row["Rank"])) if "Rank" in cols_display else "—",
            *[f"{v:.3f}" if isinstance(v, float) else str(v)
              for v in row[cols_display[1:]]]
        ])
    # Insert model name as first column
    table_data[0].insert(1, "Model")
    for i, idx in enumerate(lb.index, 1):
        table_data[i].insert(1, str(idx))

    col_widths = [35, 80] + [55] * (len(cols_display) - 1)
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(_coloured_table_style())
    elems.append(t)
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(
        "Composite Score = weighted combination of MAPE (35%), RMSE % (25%), "
        "MAE % (20%), and Stability (20%), normalised across all models.",
        S["caption"]
    ))
    return elems


def _build_model_detail(styles: dict, full_insights: dict) -> list:
    """Return flowables for per-model explanation section."""
    from reportlab.platypus import Paragraph, Spacer, HRFlowable, Table, TableStyle, PageBreak
    S = styles
    elems = []
    elems.append(Paragraph("3. Model Explanations", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))

    model_exp = full_insights.get("model_explanations", {})
    for i, (model, text) in enumerate(model_exp.items(), 1):
        elems.append(Paragraph(f"3.{i}  {model}", S["h2"]))
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                elems.append(Spacer(1, 3))
            elif line.startswith("•"):
                elems.append(Paragraph(line, S["body_bullet"]))
            else:
                elems.append(Paragraph(line, S["body"]))
        elems.append(Spacer(1, 6))
    return elems


def _build_forecast_section(styles: dict, full_insights: dict) -> list:
    """Return flowables for forecast interpretation section."""
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    S = styles
    elems = []
    elems.append(Paragraph("4. Forecast Interpretation", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))
    elems.append(Paragraph(full_insights.get("forecast_interpretation", "—"), S["body"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Performance Context", S["h2"]))
    elems.append(Paragraph(full_insights.get("performance_explanation", "—"), S["body"]))
    return elems


def _build_diagnostics_section(styles: dict, pipeline) -> list:
    """Return flowables for residual diagnostics summary."""
    from reportlab.platypus import Paragraph, Spacer, Table, HRFlowable
    S = styles
    elems = []
    elems.append(Paragraph("5. Residual Diagnostics", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))

    diags = pipeline.diagnostics
    if not diags:
        elems.append(Paragraph("No diagnostics available.", S["body"]))
        return elems

    # Summary table
    headers = ["Model", "Shapiro-Wilk", "Jarque-Bera", "Ljung-Box", "Health"]
    rows    = [headers]
    for name, d in diags.items():
        sw  = "✓ Normal" if d.get("shapiro", {}).get("normal")     else "✗ Non-normal"
        jb  = "✓ Normal" if d.get("jarque_bera", {}).get("normal") else "✗ Non-normal"
        lb_ = "✓ OK"     if d.get("ljung_box", {}).get("no_autocorr") else "✗ Autocorr."
        h   = f"{d.get('health_score', 0):.0f}/100 — {d.get('health_label', '—')}"
        rows.append([name, sw, jb, lb_, h])

    t = Table(rows, colWidths=[80, 90, 90, 90, 110])
    t.setStyle(_coloured_table_style())
    elems.append(t)
    elems.append(Spacer(1, 8))
    return elems


def _build_cv_section(styles: dict, pipeline) -> list:
    """Return flowables for cross-validation results."""
    from reportlab.platypus import Paragraph, Spacer, Table, HRFlowable
    S = styles
    elems = []
    elems.append(Paragraph("6. Walk-Forward Cross-Validation", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))

    cv = pipeline.cv_summary()
    if cv.empty:
        elems.append(Paragraph("Cross-validation was not run.", S["body"]))
        return elems

    headers = ["Model"] + cv.columns.tolist()
    rows    = [headers]
    for idx, row in cv.iterrows():
        rows.append([str(idx)] + [f"{v:.3f}" if isinstance(v, float) else str(v)
                                   for v in row])

    col_w = [90] + [100] * len(cv.columns)
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(_coloured_table_style())
    elems.append(t)
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(
        "Walk-forward CV uses an expanding window strategy. "
        "Lower CV_MAPE_Mean and CV_MAPE_Std indicate more robust and consistent models.",
        S["caption"]
    ))
    return elems


def _build_banking_section(styles: dict, full_insights: dict) -> list:
    """Return flowables for banking narrative."""
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    S = styles
    elems = []
    elems.append(Paragraph("7. Banking Domain Analysis", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))
    for para in full_insights.get("banking_narrative", "—").split("\n\n"):
        elems.append(Paragraph(para.strip(), S["body"]))
        elems.append(Spacer(1, 4))
    return elems


def _build_recommendations(styles: dict, full_insights: dict) -> list:
    """Return flowables for the recommendations section."""
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    S = styles
    elems = []
    elems.append(Paragraph("8. Recommendations", S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colour(_BLUE), spaceAfter=10))
    for line in full_insights.get("recommendations", "—").split("\n"):
        line = line.strip()
        if line:
            elems.append(Paragraph(line, S["body_bullet"]))
            elems.append(Spacer(1, 3))
    return elems


# ═════════════════════════════════════════════════════════════════════════════
# HEADER / FOOTER CANVAS
# ═════════════════════════════════════════════════════════════════════════════

def _on_page(canvas, doc, best_model: str):
    """Draw header and footer on every page except the cover."""
    from reportlab.lib.units import mm

    if doc.page == 1:
        return

    w, h = doc.pagesize
    canvas.saveState()

    # Header bar
    canvas.setFillColor(_rl_colour(_NAVY))
    canvas.rect(0, h - 22, w, 22, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(_rl_colour(_WHITE))
    canvas.drawString(doc.leftMargin, h - 15, "BOI CASA Deposit Forecasting Report")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(w - doc.rightMargin, h - 15, f"Best Model: {best_model}")

    # Footer
    canvas.setFillColor(_rl_colour(_MUTED))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, 14, "Confidential — BOI Data Science Division")
    canvas.drawRightString(w - doc.rightMargin, 14, f"Page {doc.page}")

    canvas.restoreState()


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    pipeline,
    full_insights: Dict,
    out_dir: str | Path = "reports",
    filename: Optional[str] = None,
) -> Path:
    """
    Generate a professional multi-section PDF report.

    Parameters
    ----------
    pipeline      : completed ForecastingPipeline instance
    full_insights : output of build_full_insights(pipeline)
    out_dir       : directory to save the PDF
    filename      : optional custom filename (default: boi_casa_report_<date>.pdf)

    Returns
    -------
    Path to the saved PDF file.

    Raises
    ------
    ReportGenerationError   if assembly fails
    ReportExportError       if the file cannot be written
    """
    from reportlab.platypus import (
        SimpleDocTemplate, Spacer, PageBreak, HRFlowable
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm

    _log.info("Generating PDF report …")

    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"boi_casa_report_{ts}.pdf"

        out_path = out_dir / filename
        styles   = _get_styles()
        lb       = pipeline.leaderboard()
        best     = full_insights.get("best_model", "—")

        # ── Document setup ────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=28 * mm,
            bottomMargin=20 * mm,
            title="BOI CASA Deposit Forecasting Report",
            author="BOI Data Science Division",
            subject="Time-Series Forecasting — CASA Deposits",
        )

        # ── Assemble flowables ────────────────────────────────────────────
        story: list = []
        story += _build_cover(styles, full_insights, pipeline)
        story += _build_executive_summary(styles, full_insights)
        story.append(PageBreak())
        story += _build_leaderboard(styles, lb)
        story.append(Spacer(1, 12))
        story += _build_forecast_section(styles, full_insights)
        story.append(PageBreak())
        story += _build_model_detail(styles, full_insights)
        story.append(PageBreak())
        story += _build_diagnostics_section(styles, pipeline)
        story += _build_cv_section(styles, pipeline)
        story.append(PageBreak())
        story += _build_banking_section(styles, full_insights)
        story += _build_recommendations(styles, full_insights)

        # ── Build PDF ─────────────────────────────────────────────────────
        doc.build(
            story,
            onFirstPage=lambda c, d: _on_page(c, d, best),
            onLaterPages=lambda c, d: _on_page(c, d, best),
        )

    except Exception as exc:
        raise ReportGenerationError("PDF", str(exc)) from exc

    _log.info("PDF saved → %s  (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path
