"""
plots.py — Professional visualisation suite for BOI CASA Deposit Forecasting.

Generates:
  1.  Actual deposit trend with trend-line overlay
  2.  Forecast vs Actual (multi-model)
  3.  Confidence interval chart
  4.  Residual diagnostics dashboard (4-panel)
  5.  Seasonal decomposition plot
  6.  Model comparison bar chart (MAE / RMSE / MAPE)
  7.  Model comparison radar chart
  8.  Quarterly YoY growth heatmap
  9.  Error metric heatmap
  10. Interactive Plotly — Forecast vs Actual
  11. Interactive Plotly — Model comparison
  12. Interactive Plotly — Seasonal decomposition
  13. Interactive Plotly — 3-D metric surface

All static charts are saved as high-DPI PNGs; Plotly charts as HTML.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# THEME / CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#0F172A"
SURFACE    = "#1E293B"
BORDER     = "#334155"
TEXT_MAIN  = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
ACCENT     = "#2563EB"
GREEN      = "#10B981"
AMBER      = "#F59E0B"
RED        = "#EF4444"
PURPLE     = "#8B5CF6"
PINK       = "#EC4899"
TEAL       = "#14B8A6"

MODEL_COLORS: Dict[str, str] = {
    "ARIMA":     "#3B82F6",
    "ARMA":      "#8B5CF6",
    "SARIMA":    "#10B981",
    "SARIMAX":   "#F59E0B",
    "AutoARIMA": "#EF4444",
    "Prophet":   "#EC4899",
    "Actual":    "#F1F5F9",
}

DPI = 160


def _dark_style() -> None:
    """Apply a consistent dark-mode style to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_MAIN,
        "axes.titlecolor":   TEXT_MAIN,
        "axes.grid":         True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        BORDER,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "xtick.color":       TEXT_MUTED,
        "ytick.color":       TEXT_MUTED,
        "text.color":        TEXT_MAIN,
        "legend.facecolor":  SURFACE,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT_MAIN,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
    })


def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  Saved → {path.name}")


def _fmt_crore(val: float, _pos=None) -> str:
    """Format y-axis tick as ₹ Crore."""
    return f"₹{val/1e7:.1f}Cr"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ACTUAL DEPOSIT TREND
# ─────────────────────────────────────────────────────────────────────────────

def plot_deposit_trend(
    df: pd.DataFrame,
    target: str = "Deposit_Amount",
    out_dir: Path = Path("."),
) -> Path:
    """Line chart of actual CASA deposits with a polynomial trend overlay."""
    _dark_style()
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=DARK_BG)

    series = df[target].dropna()
    x      = np.arange(len(series))

    # Main line
    ax.plot(series.index, series.values, color=MODEL_COLORS["Actual"],
            lw=2.5, marker="o", ms=5, zorder=3, label="CASA Deposits")

    # Filled area
    ax.fill_between(series.index, series.values,
                    alpha=0.12, color=ACCENT)

    # Polynomial trend
    z    = np.polyfit(x, series.values, 2)
    p    = np.poly1d(z)
    ax.plot(series.index, p(x), "--", color=AMBER, lw=2,
            alpha=0.85, label="Polynomial Trend")

    # Quarterly annotations for min/max
    peak_idx = series.idxmax()
    trough_idx = series.idxmin()
    ax.annotate(f"Peak\n{_fmt_crore(series[peak_idx])}",
                xy=(peak_idx, series[peak_idx]),
                xytext=(0, 28), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
                fontsize=9, color=GREEN, ha="center")
    ax.annotate(f"Trough\n{_fmt_crore(series[trough_idx])}",
                xy=(trough_idx, series[trough_idx]),
                xytext=(0, -38), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=9, color=RED, ha="center")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_crore))
    ax.set_title("BOI CASA Deposit Trend  (2015 Q1 – 2024 Q4)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Deposit Amount", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    fig.autofmt_xdate(rotation=45)

    path = out_dir / "01_deposit_trend.png"
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FORECAST VS ACTUAL — MULTI-MODEL
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_vs_actual(
    actual_train: pd.Series,
    actual_test: pd.Series,
    forecasts: Dict[str, pd.Series],
    out_dir: Path = Path("."),
) -> Path:
    """Side-by-side comparison of all model forecasts against held-out actuals."""
    _dark_style()
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=DARK_BG)

    # Training history
    ax.plot(actual_train.index, actual_train.values,
            color=MODEL_COLORS["Actual"], lw=2, label="Train Actual", zorder=4)

    # Actual test
    ax.plot(actual_test.index, actual_test.values,
            color=TEXT_MAIN, lw=2.5, marker="D", ms=7,
            linestyle="--", label="Test Actual", zorder=5)

    # Shaded train / test boundary
    split_date = actual_train.index[-1]
    ax.axvline(split_date, color=BORDER, lw=1.5, linestyle=":")
    ax.text(split_date, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else actual_train.min() * 0.99,
            "  Train | Test", color=TEXT_MUTED, fontsize=9, va="bottom")

    # Model forecasts
    for model_name, fc in forecasts.items():
        col = MODEL_COLORS.get(model_name, ACCENT)
        ax.plot(fc.index, fc.values, color=col, lw=2,
                marker="o", ms=5, label=model_name, zorder=3)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_crore))
    ax.set_title("Forecast vs Actual — All Models", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Deposit Amount", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.autofmt_xdate(rotation=45)

    path = out_dir / "02_forecast_vs_actual.png"
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONFIDENCE INTERVAL CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_intervals(
    actual: pd.Series,
    best_forecast: pd.Series,
    ci_df: pd.DataFrame,
    model_name: str = "SARIMAX",
    out_dir: Path = Path("."),
) -> Path:
    """Plot forecast with shaded 95% confidence bands."""
    _dark_style()
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=DARK_BG)

    col = MODEL_COLORS.get(model_name, ACCENT)

    ax.plot(actual.index, actual.values,
            color=MODEL_COLORS["Actual"], lw=2.5, label="Actual", zorder=4)

    ax.plot(best_forecast.index, best_forecast.values,
            color=col, lw=2.5, marker="o", ms=6, label=f"{model_name} Forecast", zorder=5)

    ax.fill_between(ci_df.index, ci_df["lower"], ci_df["upper"],
                    alpha=0.22, color=col, label="95% Confidence Interval")
    ax.plot(ci_df.index, ci_df["lower"], "--", color=col, lw=1.2, alpha=0.7)
    ax.plot(ci_df.index, ci_df["upper"], "--", color=col, lw=1.2, alpha=0.7)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_crore))
    ax.set_title(f"{model_name} — Forecast with 95% Confidence Interval",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Deposit Amount", fontsize=12)
    ax.legend(fontsize=10)
    fig.autofmt_xdate(rotation=45)

    path = out_dir / "03_confidence_intervals.png"
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RESIDUAL DIAGNOSTICS — 4-PANEL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def plot_residual_diagnostics(
    residuals: np.ndarray,
    model_name: str = "Model",
    out_dir: Path = Path("."),
) -> Path:
    """4-panel residual diagnostics: time-plot, ACF, histogram, Q-Q."""
    from statsmodels.graphics.tsaplots import plot_acf

    _dark_style()
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    col = ACCENT

    # ── Panel 1: Residuals over time ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(residuals, color=col, lw=1.5, alpha=0.9)
    ax1.axhline(0, color=RED, lw=1.2, linestyle="--")
    ax1.fill_between(range(len(residuals)), residuals, 0,
                     where=(residuals > 0), alpha=0.15, color=GREEN)
    ax1.fill_between(range(len(residuals)), residuals, 0,
                     where=(residuals < 0), alpha=0.15, color=RED)
    ax1.set_title("Residuals over Time", fontweight="bold")
    ax1.set_xlabel("Observation")
    ax1.set_ylabel("Residual")

    # ── Panel 2: ACF ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    max_lags = min(15, len(residuals) // 2 - 1)
    plot_acf(residuals, lags=max_lags, ax=ax2,
             color=col, vlines_kwargs={"colors": col})
    ax2.set_facecolor(SURFACE)
    ax2.set_title("Autocorrelation Function (ACF)", fontweight="bold")

    # ── Panel 3: Histogram + KDE ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=12, color=col, alpha=0.55,
             edgecolor=BORDER, density=True)
    xr = np.linspace(residuals.min() - residuals.std(),
                     residuals.max() + residuals.std(), 200)
    mu, sigma = np.mean(residuals), np.std(residuals)
    ax3.plot(xr, stats.norm.pdf(xr, mu, sigma),
             color=AMBER, lw=2.2, label=f"N({mu:.0f}, {sigma:.0f})")
    ax3.set_title("Residual Distribution", fontweight="bold")
    ax3.set_xlabel("Residual Value")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=9)

    # ── Panel 4: Q-Q plot ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
    ax4.scatter(osm, osr, color=col, s=28, alpha=0.7, zorder=3)
    fit_line = np.array(osm) * slope + intercept
    ax4.plot(osm, fit_line, color=AMBER, lw=2)
    ax4.set_title("Q-Q Plot (Normality Check)", fontweight="bold")
    ax4.set_xlabel("Theoretical Quantiles")
    ax4.set_ylabel("Sample Quantiles")

    fig.suptitle(f"Residual Diagnostics — {model_name}",
                 fontsize=15, fontweight="bold", y=1.01, color=TEXT_MAIN)

    path = out_dir / "04_residual_diagnostics.png"
    _save(fig, path, tight=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SEASONAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def plot_seasonal_decomposition(
    series: pd.Series,
    period: int = 4,
    out_dir: Path = Path("."),
) -> Path:
    """Additive seasonal decomposition into trend, seasonal, residual."""
    _dark_style()
    dec = seasonal_decompose(series.dropna(), model="additive", period=period)

    components = {
        "Observed":  dec.observed,
        "Trend":     dec.trend,
        "Seasonal":  dec.seasonal,
        "Residual":  dec.resid,
    }
    colors = [MODEL_COLORS["Actual"], AMBER, GREEN, RED]

    fig, axes = plt.subplots(4, 1, figsize=(16, 13), facecolor=DARK_BG,
                             sharex=True)
    for ax, (name, comp), col in zip(axes, components.items(), colors):
        ax.plot(comp.index, comp.values, color=col, lw=1.8)
        ax.fill_between(comp.index, comp.values,
                        alpha=0.08, color=col)
        ax.set_ylabel(name, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            _fmt_crore if name in ("Observed", "Trend") else
            mticker.ScalarFormatter()))

    axes[0].set_title("Seasonal Decomposition — BOI CASA Deposits",
                      fontsize=14, fontweight="bold", pad=12)
    axes[-1].set_xlabel("Quarter", fontsize=11)
    fig.autofmt_xdate(rotation=40)

    path = out_dir / "05_seasonal_decomposition.png"
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    out_dir: Path = Path("."),
) -> Path:
    """Grouped bar chart comparing MAE%, RMSE%, MAPE% across all models."""
    _dark_style()

    models   = metrics_df.index.tolist()
    mae_col  = [c for c in metrics_df.columns if "MAE"  in c and "%" in c][0]
    rmse_col = [c for c in metrics_df.columns if "RMSE" in c and "%" in c][0]
    mape_col = [c for c in metrics_df.columns if "MAPE" in c][0]

    x     = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK_BG)

    bars_mae  = ax.bar(x - width, metrics_df[mae_col],  width, color=ACCENT,  label="MAE %",  alpha=0.88)
    bars_rmse = ax.bar(x,         metrics_df[rmse_col], width, color=GREEN,   label="RMSE %", alpha=0.88)
    bars_mape = ax.bar(x + width, metrics_df[mape_col], width, color=AMBER,   label="MAPE %", alpha=0.88)

    def _annotate(bars, ax_):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax_.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                     f"{h:.1f}%", ha="center", va="bottom",
                     fontsize=8.5, color=TEXT_MAIN)

    _annotate(bars_mae, ax)
    _annotate(bars_rmse, ax)
    _annotate(bars_mape, ax)

    # Highlight best model (lowest MAPE)
    best_idx = metrics_df[mape_col].dropna().idxmin()
    best_pos = models.index(best_idx)
    ax.axvspan(best_pos - 0.48, best_pos + 0.48, alpha=0.07, color=GREEN, zorder=0)
    ax.text(best_pos, metrics_df[mape_col].max() * 1.05,
            "🏆 Best", ha="center", fontsize=11, color=GREEN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Error (%)", fontsize=12)
    ax.set_title("Model Comparison — MAE / RMSE / MAPE (%)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, metrics_df[[mae_col, rmse_col, mape_col]].max().max() * 1.18)

    path = out_dir / "06_model_comparison_bar.png"
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MODEL COMPARISON RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_radar(
    metrics_df: pd.DataFrame,
    out_dir: Path = Path("."),
) -> Path:
    """Spider/radar chart for multi-metric model comparison (normalised 0-1)."""
    _dark_style()

    # Normalise metrics so lower = better → invert for radar (higher = better)
    cols = [c for c in metrics_df.columns if "%" in c or "MAPE" in c]
    norm = metrics_df[cols].copy().fillna(metrics_df[cols].max())
    inverted = 1 - (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

    n_vars  = len(cols)
    angles  = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]
    labels  = [c.replace(" (%)", "").replace("(%)", "") for c in cols]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True),
                           facecolor=DARK_BG)
    ax.set_facecolor(SURFACE)
    ax.spines["polar"].set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.yaxis.set_tick_params(labelcolor="none")
    ax.set_rlabel_position(20)

    model_names = inverted.index.tolist()
    for i, model in enumerate(model_names):
        vals  = inverted.loc[model].tolist() + [inverted.loc[model].tolist()[0]]
        col   = list(MODEL_COLORS.values())[i % len(MODEL_COLORS)]
        ax.plot(angles, vals, color=col, lw=2, label=model)
        ax.fill(angles, vals, color=col, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=TEXT_MAIN)
    ax.set_title("Model Radar Chart\n(Higher = Better)",
                 fontsize=13, fontweight="bold", pad=18, color=TEXT_MAIN)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=9)

    path = out_dir / "07_model_radar.png"
    _save(fig, path, tight=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  QUARTERLY YoY GROWTH HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_quarterly_growth_heatmap(
    df: pd.DataFrame,
    out_dir: Path = Path("."),
) -> Path:
    """Heatmap: year × quarter with YoY growth (%) as cell values."""
    _dark_style()

    data = df.copy()
    data["Year"]    = data.index.year
    data["Quarter"] = data.index.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})

    pivot = data.pivot_table(values="YoY_Growth", index="Year", columns="Quarter")
    pivot = pivot[["Q1", "Q2", "Q3", "Q4"]]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
    cmap = sns.diverging_palette(10, 130, s=85, l=40, as_cmap=True)

    sns.heatmap(
        pivot, ax=ax, cmap=cmap, center=0, annot=True, fmt=".1f",
        annot_kws={"size": 11, "color": TEXT_MAIN},
        linewidths=1, linecolor=BORDER,
        cbar_kws={"label": "YoY Growth (%)", "shrink": 0.8},
    )
    ax.set_facecolor(DARK_BG)
    ax.set_title("Quarterly YoY Growth Heatmap — BOI CASA Deposits",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Year", fontsize=11)

    path = out_dir / "08_growth_heatmap.png"
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ERROR METRIC HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    out_dir: Path = Path("."),
) -> Path:
    """Colour-coded heatmap of all model × metric combinations."""
    _dark_style()

    pct_cols = [c for c in metrics_df.columns if "%" in c or "MAPE" in c]
    data     = metrics_df[pct_cols].fillna(metrics_df[pct_cols].max().max())

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    sns.heatmap(
        data, ax=ax, cmap=cmap, annot=True, fmt=".2f",
        annot_kws={"size": 11, "color": TEXT_MAIN},
        linewidths=0.8, linecolor=BORDER,
        cbar_kws={"label": "Error (%)", "shrink": 0.85},
    )
    ax.set_title("Error Metric Heatmap — All Models",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    path = out_dir / "09_metric_heatmap.png"
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 10.  INTERACTIVE PLOTLY — FORECAST vs ACTUAL
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert a hex color string to an rgba() string for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plotly_forecast_vs_actual(
    actual_train: pd.Series,
    actual_test:  pd.Series,
    forecasts:    Dict[str, pd.Series],
    ci_map:       Dict[str, pd.DataFrame],
    out_dir: Path = Path("."),
) -> Path:
    """Interactive Plotly chart: train history + test actuals + all model forecasts."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Training actuals
    fig.add_trace(go.Scatter(
        x=actual_train.index.astype(str), y=actual_train.values,
        name="Train Actual",
        line=dict(color="#94A3B8", width=2),
        mode="lines",
    ))

    # Test actuals
    fig.add_trace(go.Scatter(
        x=actual_test.index.astype(str), y=actual_test.values,
        name="Test Actual",
        line=dict(color="#F1F5F9", width=3, dash="dash"),
        mode="lines+markers",
        marker=dict(size=9, symbol="diamond"),
    ))

    # Forecasts + CI bands
    for model_name, fc in forecasts.items():
        col = MODEL_COLORS.get(model_name, "#3B82F6")

        fig.add_trace(go.Scatter(
            x=fc.index.astype(str), y=fc.values,
            name=model_name,
            line=dict(color=col, width=2.5),
            mode="lines+markers",
            marker=dict(size=7),
        ))

        if model_name in ci_map:
            ci = ci_map[model_name]
            fig.add_trace(go.Scatter(
                x=list(ci.index.astype(str)) + list(ci.index.astype(str))[::-1],
                y=list(ci["upper"]) + list(ci["lower"])[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(col, 0.12),
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{model_name} CI",
                showlegend=False,
            ))

    fig.update_layout(
        title=dict(
            text="<b>BOI CASA Deposits — Forecast vs Actual (Interactive)</b>",
            font=dict(size=18, color="#F1F5F9"),
            x=0.02,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F1F5F9", family="Inter, sans-serif"),
        legend=dict(
            bgcolor="#1E293B", bordercolor="#334155",
            borderwidth=1, font=dict(size=11),
        ),
        xaxis=dict(
            title="Quarter", gridcolor="#334155",
            showspikes=True, spikecolor="#475569",
        ),
        yaxis=dict(
            title="Deposit Amount (₹)",
            tickformat=",.0f",
            gridcolor="#334155",
        ),
        hovermode="x unified",
        height=560,
    )

    path = out_dir / "10_interactive_forecast.html"
    fig.write_html(str(path))
    print(f"  ✓  Saved → {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 11.  INTERACTIVE PLOTLY — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plotly_model_comparison(
    metrics_df: pd.DataFrame,
    out_dir: Path = Path("."),
) -> Path:
    """Interactive grouped bar chart for model metrics."""
    import plotly.graph_objects as go

    pct_cols = [c for c in metrics_df.columns if "%" in c or "MAPE" in c]
    models   = metrics_df.index.tolist()

    fig = go.Figure()
    palette = [ACCENT, GREEN, AMBER, RED, PURPLE, PINK, TEAL]
    for i, col in enumerate(pct_cols):
        fig.add_trace(go.Bar(
            name=col.replace(" (%)", "").replace("(%)", ""),
            x=models,
            y=metrics_df[col],
            marker_color=palette[i % len(palette)],
            text=[f"{v:.2f}%" for v in metrics_df[col]],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Model Performance Comparison — Error Metrics (%)</b>",
            font=dict(size=17, color="#F1F5F9"), x=0.02,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F1F5F9"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1),
        xaxis=dict(title="Model", gridcolor="#334155"),
        yaxis=dict(title="Error (%)", gridcolor="#334155"),
        height=520,
    )

    path = out_dir / "11_interactive_comparison.html"
    fig.write_html(str(path))
    print(f"  ✓  Saved → {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 12.  INTERACTIVE PLOTLY — SEASONAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def plotly_seasonal_decomposition(
    series: pd.Series,
    period: int = 4,
    out_dir: Path = Path("."),
) -> Path:
    """Interactive 4-panel seasonal decomposition with Plotly subplots."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dec = seasonal_decompose(series.dropna(), model="additive", period=period)

    comps = {
        "Observed":  (dec.observed,  "#F1F5F9"),
        "Trend":     (dec.trend,     AMBER),
        "Seasonal":  (dec.seasonal,  GREEN),
        "Residual":  (dec.resid,     RED),
    }

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=list(comps.keys()),
                        vertical_spacing=0.07)

    for row, (name, (comp, col)) in enumerate(comps.items(), start=1):
        comp_clean = comp.dropna()
        fig.add_trace(
            go.Scatter(
                x=comp_clean.index.astype(str),
                y=comp_clean.values,
                mode="lines",
                line=dict(color=col, width=2),
                name=name,
                fill="tozeroy",
                fillcolor=_hex_to_rgba(col, 0.09),
            ),
            row=row, col=1,
        )

    fig.update_layout(
        title=dict(
            text="<b>Interactive Seasonal Decomposition — BOI CASA Deposits</b>",
            font=dict(size=17, color="#F1F5F9"), x=0.02,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F1F5F9"),
        showlegend=False,
        height=800,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#334155", row=i, col=1)
        fig.update_yaxes(gridcolor="#334155", row=i, col=1)

    path = out_dir / "12_interactive_decomposition.html"
    fig.write_html(str(path))
    print(f"  ✓  Saved → {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 13.  INTERACTIVE PLOTLY — ROLLING FORECAST ANIMATION (CANDLESTICK STYLE)
# ─────────────────────────────────────────────────────────────────────────────

def plotly_animated_forecast(
    actual: pd.Series,
    forecasts: Dict[str, pd.Series],
    ci_map: Dict[str, pd.DataFrame],
    out_dir: Path = Path("."),
) -> Path:
    """Animated Plotly chart that reveals each model's forecast step by step."""
    import plotly.graph_objects as go

    models     = list(forecasts.keys())
    all_dates  = sorted(set(actual.index) | set().union(*[set(f.index) for f in forecasts.values()]))
    str_dates  = [str(d)[:10] for d in all_dates]

    frames = []
    for step in range(1, len(models) + 1):
        traces = [
            go.Scatter(
                x=actual.index.astype(str), y=actual.values,
                name="Actual", line=dict(color="#F1F5F9", width=2.5),
                mode="lines",
            )
        ]
        for i, m in enumerate(models[:step]):
            fc  = forecasts[m]
            col = MODEL_COLORS.get(m, ACCENT)
            traces.append(go.Scatter(
                x=fc.index.astype(str), y=fc.values,
                name=m, line=dict(color=col, width=2.5),
                mode="lines+markers", marker=dict(size=7),
            ))
            if m in ci_map:
                ci = ci_map[m]
                traces.append(go.Scatter(
                    x=list(ci.index.astype(str)) + list(ci.index.astype(str))[::-1],
                    y=list(ci["upper"]) + list(ci["lower"])[::-1],
                    fill="toself",
                    fillcolor=_hex_to_rgba(col, 0.13),
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{m} CI", showlegend=False,
                ))
        frames.append(go.Frame(data=traces, name=str(step)))

    # Initial (only actual)
    initial = [go.Scatter(
        x=actual.index.astype(str), y=actual.values,
        name="Actual", line=dict(color="#F1F5F9", width=2.5),
    )]

    fig = go.Figure(data=initial, frames=frames)
    fig.update_layout(
        title=dict(
            text="<b>Animated Model Reveal — BOI CASA Forecast</b>",
            font=dict(size=17, color="#F1F5F9"), x=0.02,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F1F5F9"),
        xaxis=dict(title="Quarter", gridcolor="#334155"),
        yaxis=dict(title="Deposit Amount (₹)", tickformat=",.0f", gridcolor="#334155"),
        height=540,
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 900, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
            bgcolor="#1E293B", font=dict(color="#F1F5F9"),
            x=0.02, y=1.08, xanchor="left",
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                          "mode": "immediate"}],
                        label=models[int(f.name) - 1] if int(f.name) > 0 else "Start",
                        method="animate") for f in frames],
            x=0.02, len=0.96,
            bgcolor="#334155",
            font=dict(color="#F1F5F9"),
        )],
    )

    path = out_dir / "13_animated_forecast.html"
    fig.write_html(str(path))
    print(f"  ✓  Saved → {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_visualizations(
    df: pd.DataFrame,
    actual_train: pd.Series,
    actual_test: pd.Series,
    forecasts: Dict[str, pd.Series],
    ci_map: Dict[str, pd.DataFrame],
    residuals_map: Dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    out_dir: Path = Path("visualizations/output"),
) -> List[Path]:
    """
    Run every visualisation function and return a list of saved file paths.

    Parameters
    ----------
    df            : Full CASA DataFrame (with YoY_Growth etc.)
    actual_train  : Training-period actuals
    actual_test   : Test-period actuals
    forecasts     : {model_name: forecast_series}
    ci_map        : {model_name: DataFrame(lower, upper)}
    residuals_map : {model_name: np.ndarray}
    metrics_df    : DataFrame of percentage metrics indexed by model name
    out_dir       : Directory to save outputs
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Generating BOI CASA Forecasting Visualisations    ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    actual_full = pd.concat([actual_train, actual_test])

    # Best model for CI plot (lowest MAPE)
    mape_col = [c for c in metrics_df.columns if "MAPE" in c][0]
    best     = metrics_df[mape_col].dropna().idxmin()
    best_fc, best_ci = forecasts.get(best), ci_map.get(best)

    saved.append(plot_deposit_trend(df, out_dir=out_dir))
    saved.append(plot_forecast_vs_actual(actual_train, actual_test, forecasts, out_dir=out_dir))

    if best_fc is not None and best_ci is not None:
        saved.append(plot_confidence_intervals(actual_full, best_fc, best_ci,
                                               model_name=best, out_dir=out_dir))

    first_model = next(iter(residuals_map))
    saved.append(plot_residual_diagnostics(residuals_map[first_model],
                                           model_name=first_model, out_dir=out_dir))

    saved.append(plot_seasonal_decomposition(actual_full, out_dir=out_dir))
    saved.append(plot_model_comparison_bar(metrics_df, out_dir=out_dir))
    saved.append(plot_model_radar(metrics_df, out_dir=out_dir))
    saved.append(plot_quarterly_growth_heatmap(df, out_dir=out_dir))
    saved.append(plot_metric_heatmap(metrics_df, out_dir=out_dir))
    saved.append(plotly_forecast_vs_actual(actual_train, actual_test,
                                           forecasts, ci_map, out_dir=out_dir))
    saved.append(plotly_model_comparison(metrics_df, out_dir=out_dir))
    saved.append(plotly_seasonal_decomposition(actual_full, out_dir=out_dir))
    saved.append(plotly_animated_forecast(actual_full, forecasts, ci_map, out_dir=out_dir))

    print(f"\n✅  All {len(saved)} visualisations saved to: {out_dir.resolve()}\n")
    return saved
