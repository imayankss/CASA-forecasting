"""
charts.py — Reusable Plotly chart components for the Streamlit dashboard.
All charts use the Bloomberg dark theme palette.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# ── Palette ──────────────────────────────────────────────────────────────────
BG    = "#080E1A"
SURF  = "#0D1526"
SURF2 = "#0F1D35"
GRID  = "#1E2D4A"
T1    = "#F1F5F9"
T2    = "#94A3B8"
BLUE  = "#2563EB"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED   = "#EF4444"
PURP  = "#7C3AED"
PINK  = "#EC4899"
TEAL  = "#14B8A6"

MODEL_PAL: Dict[str, str] = {
    "ARIMA":     "#3B82F6",
    "ARMA":      "#8B5CF6",
    "SARIMA":    "#10B981",
    "SARIMAX":   "#F59E0B",
    "AutoARIMA": "#EF4444",
    "Prophet":   "#EC4899",
    "Actual":    "#E2E8F0",
}

def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=BG, plot_bgcolor=SURF2,
        font=dict(family="Inter, sans-serif", color=T1, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showspikes=True, spikecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor=SURF, bordercolor=GRID, borderwidth=1,
                    font=dict(size=11, color=T1)),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=50),
    )
    base.update(kwargs)
    return base


def _hex_rgba(hex_: str, a: float = 0.15) -> str:
    h = hex_.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ─────────────────────────────────────────────────────────────────────────────
# 1. TREND CHART
# ─────────────────────────────────────────────────────────────────────────────

def trend_chart(df: pd.DataFrame, target: str = "Deposit_Amount") -> go.Figure:
    s = df[target].dropna()
    x = np.arange(len(s))
    z = np.polyfit(x, s.values, 2)
    trend = np.poly1d(z)(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index.astype(str), y=s.values, name="CASA Deposits",
        line=dict(color=BLUE, width=2.5),
        mode="lines+markers", marker=dict(size=5, color=BLUE),
        fill="tozeroy", fillcolor=_hex_rgba(BLUE, 0.07),
    ))
    fig.add_trace(go.Scatter(
        x=s.index.astype(str), y=trend, name="Trend",
        line=dict(color=AMBER, width=2, dash="dash"),
        mode="lines",
    ))
    fig.update_layout(
        title="<b>CASA Deposit Trend</b>",
        yaxis_title="Deposit Amount (₹)",
        xaxis_title="Quarter",
        **_base_layout(height=420),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORECAST VS ACTUAL
# ─────────────────────────────────────────────────────────────────────────────

def forecast_vs_actual_chart(
    train: pd.Series,
    test:  pd.Series,
    forecasts: Dict[str, pd.Series],
    ci_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> go.Figure:
    fig = go.Figure()
    ci_map = ci_map or {}

    fig.add_trace(go.Scatter(
        x=train.index.astype(str), y=train.values,
        name="Train Actual", line=dict(color=T2, width=2), mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=test.index.astype(str), y=test.values,
        name="Test Actual",
        line=dict(color=T1, width=3, dash="dot"),
        mode="lines+markers", marker=dict(size=9, symbol="diamond", color=T1),
    ))

    for model, fc in forecasts.items():
        col = MODEL_PAL.get(model, BLUE)
        fig.add_trace(go.Scatter(
            x=fc.index.astype(str), y=fc.values, name=model,
            line=dict(color=col, width=2.5),
            mode="lines+markers", marker=dict(size=6),
        ))
        if model in ci_map:
            ci = ci_map[model]
            fig.add_trace(go.Scatter(
                x=list(ci.index.astype(str)) + list(ci.index.astype(str))[::-1],
                y=list(ci["upper"]) + list(ci["lower"])[::-1],
                fill="toself", fillcolor=_hex_rgba(col, 0.12),
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{model} CI", showlegend=False,
            ))

    # Split line
    split_x = str(train.index[-1])[:10]
    fig.add_vline(x=split_x, line_dash="dot", line_color=GRID,
                  annotation_text="Train | Test", annotation_font_color=T2)

    fig.update_layout(
        title="<b>Forecast vs Actual — All Models</b>",
        yaxis_title="Deposit Amount (₹)",
        xaxis_title="Quarter",
        **_base_layout(height=500),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL COMPARISON BAR
# ─────────────────────────────────────────────────────────────────────────────

def model_comparison_bar(metrics_df: pd.DataFrame) -> go.Figure:
    metric_cols = [c for c in metrics_df.columns if "%" in c or c == "MAPE"]
    models = metrics_df.index.tolist()
    palette = [BLUE, GREEN, AMBER, RED, PURP, PINK, TEAL]

    fig = go.Figure()
    for i, col in enumerate(metric_cols[:3]):
        fig.add_trace(go.Bar(
            name=col, x=models,
            y=metrics_df[col].fillna(0),
            marker_color=palette[i],
            text=[f"{v:.2f}%" for v in metrics_df[col].fillna(0)],
            textposition="outside",
            marker_line_color=BG, marker_line_width=1.5,
        ))

    # Best model highlight
    mape_col = next((c for c in metrics_df.columns if "MAPE" in c), None)
    if mape_col:
        best = metrics_df[mape_col].idxmin()
        best_x = models.index(best)
        fig.add_vrect(
            x0=best_x - 0.45, x1=best_x + 0.45,
            fillcolor=_hex_rgba(GREEN, 0.08),
            line_color=GREEN, line_width=1,
            annotation_text="🏆 Best", annotation_position="top left",
            annotation_font_color=GREEN,
        )

    fig.update_layout(
        barmode="group", title="<b>Model Error Metrics Comparison (%)</b>",
        yaxis_title="Error (%)", xaxis_title="Model",
        **_base_layout(height=460),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def radar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    cols = [c for c in metrics_df.columns if "%" in c or c in ("MAPE", "Stability")][:5]
    norm = metrics_df[cols].copy().fillna(metrics_df[cols].max())
    for c in cols:
        if c == "Stability":
            norm[c] = (norm[c] - norm[c].min()) / (norm[c].max() - norm[c].min() + 1e-9)
        else:
            norm[c] = 1 - (norm[c] - norm[c].min()) / (norm[c].max() - norm[c].min() + 1e-9)

    labels = [c.replace(" (%)", "").replace("(%)", "") for c in cols]
    fig = go.Figure()

    for i, model in enumerate(metrics_df.index):
        vals = norm.loc[model].tolist()
        vals += [vals[0]]
        col = list(MODEL_PAL.values())[i % len(MODEL_PAL)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]],
            fill="toself", fillcolor=_hex_rgba(col, 0.10),
            line=dict(color=col, width=2), name=model,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID,
                            tickfont=dict(color=T2, size=9)),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(color=T1, size=11)),
            bgcolor=SURF2,
        ),
        title="<b>Model Radar — Higher is Better</b>",
        **_base_layout(height=480),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. RESIDUAL DIAGNOSTICS — 4-PANEL
# ─────────────────────────────────────────────────────────────────────────────

def residual_dashboard(residuals: np.ndarray, model_name: str = "Model") -> go.Figure:
    from statsmodels.tsa.stattools import acf
    r = np.array(residuals, dtype=float)
    r = r[~np.isnan(r)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuals over Time", f"ACF (lags={min(15, len(r)//2-1)})",
            "Distribution", "Q-Q Plot",
        ),
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )

    # Panel 1 — residuals over time
    fig.add_trace(go.Scatter(
        y=r, mode="lines+markers", name="Residuals",
        line=dict(color=BLUE, width=1.5), marker=dict(size=3),
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=RED, row=1, col=1)

    # Panel 2 — ACF
    max_lags = min(15, len(r) // 2 - 1)
    acf_vals = acf(r, nlags=max_lags, fft=True)
    lags = list(range(len(acf_vals)))
    ci_line = 1.96 / np.sqrt(len(r))
    for lag, val in zip(lags, acf_vals):
        c = GREEN if abs(val) <= ci_line else RED
        fig.add_trace(go.Bar(x=[lag], y=[val], marker_color=c,
                              showlegend=False), row=1, col=2)
    fig.add_hline(y=ci_line, line_dash="dash", line_color=AMBER, row=1, col=2)
    fig.add_hline(y=-ci_line, line_dash="dash", line_color=AMBER, row=1, col=2)

    # Panel 3 — histogram + KDE
    fig.add_trace(go.Histogram(
        x=r, nbinsx=14, name="Residuals",
        marker_color=_hex_rgba(BLUE, 0.6),
        marker_line_color=BLUE, marker_line_width=1,
        histnorm="probability density",
    ), row=2, col=1)
    xr = np.linspace(r.min() - r.std(), r.max() + r.std(), 200)
    mu, sigma = np.mean(r), np.std(r)
    kde_y = stats.norm.pdf(xr, mu, sigma)
    fig.add_trace(go.Scatter(x=xr, y=kde_y, mode="lines", name="Normal fit",
                              line=dict(color=AMBER, width=2.5)), row=2, col=1)

    # Panel 4 — Q-Q
    (osm, osr), (slope, intercept, _) = stats.probplot(r)
    fig.add_trace(go.Scatter(
        x=list(osm), y=list(osr), mode="markers", name="Q-Q",
        marker=dict(color=BLUE, size=6, opacity=0.7),
    ), row=2, col=2)
    fit_line = np.array(osm) * slope + intercept
    fig.add_trace(go.Scatter(
        x=list(osm), y=list(fit_line), mode="lines", name="Ideal",
        line=dict(color=AMBER, width=2),
    ), row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor=GRID, row=i, col=j)
            fig.update_yaxes(gridcolor=GRID, row=i, col=j)

    fig.update_layout(
        title=f"<b>Residual Diagnostics — {model_name}</b>",
        showlegend=False,
        **_base_layout(height=620),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. SEASONAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_decomp_chart(series: pd.Series, period: int = 4) -> go.Figure:
    from statsmodels.tsa.seasonal import seasonal_decompose
    dec = seasonal_decompose(series.dropna(), model="additive", period=period)

    components = [
        ("Observed",  dec.observed,  T1),
        ("Trend",     dec.trend,     AMBER),
        ("Seasonal",  dec.seasonal,  GREEN),
        ("Residual",  dec.resid,     RED),
    ]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=[c[0] for c in components],
                        vertical_spacing=0.06)

    for i, (name, comp, col) in enumerate(components, 1):
        c = comp.dropna()
        fig.add_trace(go.Scatter(
            x=c.index.astype(str), y=c.values, name=name,
            mode="lines", line=dict(color=col, width=2),
            fill="tozeroy", fillcolor=_hex_rgba(col, 0.07),
        ), row=i, col=1)
        fig.update_xaxes(gridcolor=GRID, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID, row=i, col=1)

    fig.update_layout(
        title="<b>Seasonal Decomposition</b>",
        showlegend=False,
        **_base_layout(height=700),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. ROLLING STATS
# ─────────────────────────────────────────────────────────────────────────────

def rolling_stats_chart(series: pd.Series, windows: List[int] = [4, 8]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index.astype(str), y=series.values,
        name="Actual", line=dict(color=T1, width=2), mode="lines",
    ))
    palette = [BLUE, GREEN, AMBER]
    for w, col in zip(windows, palette):
        rm = series.rolling(w).mean()
        fig.add_trace(go.Scatter(
            x=rm.index.astype(str), y=rm.values,
            name=f"Rolling Mean (w={w}Q)",
            line=dict(color=col, width=2, dash="dash"), mode="lines",
        ))

    fig.update_layout(
        title="<b>Rolling Statistics</b>",
        yaxis_title="Deposit Amount (₹)",
        **_base_layout(height=420),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. GROWTH HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def growth_heatmap_chart(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["Year"]    = data.index.year
    data["Quarter"] = data.index.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    pivot = data.pivot_table(values="YoY_Growth", index="Year", columns="Quarter")
    pivot = pivot[["Q1", "Q2", "Q3", "Q4"]]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "N/A"
               for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(color=T1, size=12),
        hoverongaps=False,
        colorbar=dict(title="YoY %", tickfont=dict(color=T1)),
    ))
    fig.update_layout(
        title="<b>Quarterly YoY Growth Heatmap</b>",
        xaxis_title="Quarter", yaxis_title="Year",
        **_base_layout(height=420),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. LEADERBOARD TABLE
# ─────────────────────────────────────────────────────────────────────────────

def leaderboard_chart(lb: pd.DataFrame) -> go.Figure:
    display_cols = [c for c in ["MAE_%", "RMSE_%", "MAPE", "R2",
                                "Stability", "Composite_Score"] if c in lb.columns]
    models  = lb.index.tolist()
    colors_  = [_hex_rgba(GREEN, 0.2) if i == 0 else _hex_rgba(BLUE, 0.07)
                for i in range(len(models))]

    header = ["<b>Model</b>"] + [f"<b>{c}</b>" for c in display_cols]
    cells  = [models]
    for c in display_cols:
        cells.append([f"{v:.3f}" if isinstance(v, float) else str(v)
                      for v in lb[c]])

    fig = go.Figure(go.Table(
        header=dict(
            values=header,
            fill_color=SURF, font=dict(color=T2, size=11),
            align="center", line_color=GRID,
        ),
        cells=dict(
            values=cells,
            fill_color=[colors_] * len(cells),
            font=dict(color=[T1] * len(models), size=12),
            align="center", line_color=GRID,
            height=36,
        ),
    ))
    fig.update_layout(
        title="<b>Model Leaderboard</b>",
        **_base_layout(height=max(250, 80 + 36 * len(models))),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. DISTRIBUTION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def distribution_chart(series: pd.Series) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Distribution", "Box Plot"))

    fig.add_trace(go.Histogram(
        x=series.values, name="Distribution", nbinsx=16,
        marker_color=_hex_rgba(BLUE, 0.6),
        marker_line_color=BLUE, marker_line_width=1,
        histnorm="probability density",
    ), row=1, col=1)

    xr = np.linspace(series.min(), series.max(), 200)
    kde = stats.norm.pdf(xr, series.mean(), series.std())
    fig.add_trace(go.Scatter(x=xr, y=kde, mode="lines",
                              line=dict(color=AMBER, width=2),
                              name="KDE"), row=1, col=1)

    fig.add_trace(go.Box(
        y=series.values, name="Box Plot",
        marker_color=BLUE,
        line_color=BLUE,
        boxmean=True,
    ), row=1, col=2)

    for j in [1, 2]:
        fig.update_xaxes(gridcolor=GRID, row=1, col=j)
        fig.update_yaxes(gridcolor=GRID, row=1, col=j)

    fig.update_layout(
        title="<b>Deposit Amount Distribution</b>",
        showlegend=False,
        **_base_layout(height=400),
    )
    return fig
