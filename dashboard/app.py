"""
app.py — BOI CASA Deposit Forecasting Dashboard
Bloomberg-style financial analytics platform built with Streamlit.

Run:
    streamlit run dashboard/app.py
"""

import io
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="BOI CASA Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = ROOT / "dashboard" / "styles" / "theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Lazy imports (avoid import errors if libraries missing) ───────────────────
from src.data_loader  import load_casa_data, get_train_test
from src.evaluation.diagnostics import run_full_diagnostics, acf_pacf_values
from src.evaluation.comparison import ModelComparison
from src.pipelines.forecasting_pipeline import FITTERS
from dashboard.components.charts import (
    trend_chart, forecast_vs_actual_chart, model_comparison_bar,
    radar_chart, residual_dashboard, seasonal_decomp_chart,
    rolling_stats_chart, growth_heatmap_chart, leaderboard_chart,
    distribution_chart,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "df":           None,
        "train":        None,
        "test":         None,
        "results":      {},
        "comparison":   None,
        "pipeline_ran": False,
        "date_col":     "Quarter_Year",
        "target_col":   "Deposit_Amount",
        "freq":         "Q",
        "horizon":      4,
        "train_frac":   0.80,
        "conf_level":   0.95,
        "selected_models": ["ARIMA", "SARIMA", "SARIMAX", "AutoARIMA", "HoltWinters", "Prophet"],
        "page":         "🏠 Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px;'>
            <div style='font-size:2.2rem;'>📊</div>
            <div style='font-size:1.1rem; font-weight:700; color:#60A5FA; letter-spacing:0.04em;'>
                BOI CASA
            </div>
            <div style='font-size:0.7rem; color:#64748B; letter-spacing:0.12em; 
                        text-transform:uppercase; margin-top:2px;'>
                Forecasting Platform
            </div>
        </div>
        <hr style='border-color:#1E2D4A; margin:10px 0 20px;'>
        """, unsafe_allow_html=True)

        pages = [
            "🏠 Home",
            "📁 Upload Dataset",
            "🔍 Exploratory Analysis",
            "🤖 Forecasting Models",
            "⚖️ Model Comparison",
            "🧪 Residual Diagnostics",
            "📤 Export Results",
            "⚙️ Settings",
        ]
        page = st.radio("Navigation", pages,
                        index=pages.index(st.session_state.page),
                        label_visibility="collapsed")
        st.session_state.page = page

        st.markdown("<hr style='border-color:#1E2D4A; margin:20px 0;'>", unsafe_allow_html=True)

        # Dataset status
        if st.session_state.df is not None:
            df = st.session_state.df
            n  = len(df)
            st.markdown(f"""
            <div style='background:#0F1D35; border:1px solid #1E3A5F; border-radius:10px; 
                        padding:14px; font-size:0.8rem;'>
                <div style='color:#64748B; text-transform:uppercase; 
                            letter-spacing:0.08em; font-size:0.68rem; margin-bottom:8px;'>
                    Dataset Loaded
                </div>
                <div style='color:#10B981; font-weight:600;'>✓ {n} observations</div>
                <div style='color:#94A3B8; margin-top:4px;'>
                    {df.index.min().strftime('%Y-Q%q') if hasattr(df.index.min(), 'strftime') else str(df.index.min())[:7]}
                    →
                    {df.index.max().strftime('%Y-Q%q') if hasattr(df.index.max(), 'strftime') else str(df.index.max())[:7]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#0F1D35; border:1px solid #1E3A5F; border-radius:10px; 
                        padding:14px; font-size:0.8rem;'>
                <div style='color:#64748B; text-transform:uppercase; 
                            letter-spacing:0.08em; font-size:0.68rem; margin-bottom:8px;'>
                    Dataset Status
                </div>
                <div style='color:#F59E0B;'>⚠ No dataset loaded</div>
            </div>
            """, unsafe_allow_html=True)

        # Quick load demo data
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("⚡ Load Demo Data", use_container_width=True):
            _load_demo()
            st.rerun()

        # Model run status
        if st.session_state.pipeline_ran:
            n_models = len(st.session_state.results)
            st.markdown(f"""
            <div style='margin-top:12px; background:#0A1F0A; border:1px solid #14532D; 
                        border-radius:10px; padding:12px; font-size:0.8rem;'>
                <div style='color:#10B981; font-weight:600;'>✅ {n_models} models trained</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='position:fixed; bottom:20px; left:0; width:260px; 
                    text-align:center; font-size:0.65rem; color:#334155;'>
            BOI CASA Forecasting Platform v1.0<br>
            Built with Streamlit + Plotly
        </div>
        """, unsafe_allow_html=True)

    return page


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_demo():
    demo_path = ROOT / "data" / "raw" / "boi_casa_deposits.csv"
    df = load_casa_data(str(demo_path))
    _set_dataset(df)
    st.success("✅ Demo dataset loaded successfully!")


def _set_dataset(df: pd.DataFrame):
    train, test = get_train_test(df, train_frac=st.session_state.train_frac)
    st.session_state.df    = df
    st.session_state.train = train
    st.session_state.test  = test


# The legacy Streamlit app intentionally imports FITTERS from the core pipeline.
# Model training logic lives in src/pipelines/forecasting_pipeline.py only.


def run_selected_models(selected: list) -> Dict:
    train = st.session_state.train
    test  = st.session_state.test
    results = {}
    prog = st.progress(0)
    status = st.empty()

    for i, name in enumerate(selected):
        status.markdown(f"⚙️ Training **{name}** …  ({i+1}/{len(selected)})")
        prog.progress((i) / len(selected))
        t0 = time.time()
        try:
            out = FITTERS[name](train, test, st.session_state.df)
            out["train_time"] = round(time.time() - t0, 2)
            results[name] = out
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {e}")
        prog.progress((i + 1) / len(selected))

    prog.empty()
    status.empty()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── HOME ──────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <h1 style='font-size:2.2rem; margin-bottom:0;'>
        📊 BOI CASA Deposit Forecasting
    </h1>
    <p style='color:#64748B; font-size:1rem; margin-top:4px;'>
        Enterprise-grade time-series forecasting platform • Bank of India Analytics
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D4A;'>", unsafe_allow_html=True)

    # KPI row
    df = st.session_state.df
    if df is not None:
        s = df["Deposit_Amount"]
        yoy = df["YoY_Growth"].dropna()
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Latest Deposit", f"₹{s.iloc[-1]/1e7:.2f}Cr",
                      delta=f"{yoy.iloc[-1]:+.1f}% YoY")
        with c2:
            st.metric("Peak Deposit",  f"₹{s.max()/1e7:.2f}Cr")
        with c3:
            st.metric("Avg YoY Growth", f"{yoy.mean():.2f}%")
        with c4:
            st.metric("Periods (Total)", f"{len(s)} Qtrs")
        with c5:
            if st.session_state.pipeline_ran:
                results = st.session_state.results
                best = min(results, key=lambda k: results[k]["metrics"]["MAPE"])
                st.metric("Best MAPE", f"{results[best]['metrics']['MAPE']:.2f}%",
                          delta=f"({best})")
            else:
                st.metric("Models Trained", "—")
    else:
        # Placeholder KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val in zip(
            [c1, c2, c3, c4, c5],
            ["Latest Deposit", "Peak Deposit", "Avg YoY Growth", "Data Points", "Best MAPE"],
            ["—", "—", "—", "—", "—"]
        ):
            with col:
                st.metric(label, val)

    st.markdown("<br>", unsafe_allow_html=True)

    # Info cards row
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("## 📖 Platform Overview")
        st.markdown("""
        This platform provides an end-to-end forecasting workflow for **Bank of India CASA deposits**:

        | Stage | Description |
        |---|---|
        | **Data Ingestion** | Upload CSV/XLSX or use the built-in demo dataset |
        | **EDA** | Interactive trend, seasonality, and distribution analysis |
        | **Modelling** | Train ARIMA, SARIMA, SARIMAX, AutoARIMA, HoltWinters, Prophet |
        | **Evaluation** | MAE, RMSE, MAPE, R², Stability Score, Composite Rank |
        | **Diagnostics** | Residual plots, ACF, Q-Q, ADF, Shapiro-Wilk, Jarque-Bera |
        | **Export** | CSV forecasts, metrics, leaderboard |
        """)

    with col2:
        st.markdown("## 🤖 Supported Models")
        models_info = {
            "ARIMA":     ("Classic univariate model",    "#3B82F6"),
            "SARIMA":    ("Seasonal ARIMA (quarterly)", "#10B981"),
            "SARIMAX":   ("SARIMA + exogenous vars",     "#F59E0B"),
            "AutoARIMA": ("Grid-search order selection", "#EF4444"),
            "HoltWinters": ("Trend + quarterly seasonality", "#8B5CF6"),
            "Prophet":   ("Meta additive model",         "#EC4899"),
        }
        for model, (desc, col) in models_info.items():
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:12px; 
                        padding:10px 14px; margin-bottom:8px;
                        background:#0F1D35; border:1px solid #1E3A5F; 
                        border-left:3px solid {col}; border-radius:8px;'>
                <div style='flex:1;'>
                    <span style='color:{col}; font-weight:600; font-size:0.88rem;'>{model}</span>
                    <span style='color:#64748B; font-size:0.78rem; margin-left:10px;'>{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown("## 🔄 Forecasting Pipeline")
    st.markdown("""
    <div style='background:#0D1526; border:1px solid #1E2D4A; border-radius:12px; 
                padding:24px; text-align:center; font-size:0.9rem; color:#94A3B8;'>
        <span style='color:#3B82F6; font-weight:600;'>📁 Data Upload</span>
        &nbsp;→&nbsp;
        <span style='color:#10B981; font-weight:600;'>🔍 EDA</span>
        &nbsp;→&nbsp;
        <span style='color:#F59E0B; font-weight:600;'>✂️ Train/Test Split</span>
        &nbsp;→&nbsp;
        <span style='color:#EF4444; font-weight:600;'>🤖 Model Training</span>
        &nbsp;→&nbsp;
        <span style='color:#8B5CF6; font-weight:600;'>📊 Evaluation</span>
        &nbsp;→&nbsp;
        <span style='color:#EC4899; font-weight:600;'>🧪 Diagnostics</span>
        &nbsp;→&nbsp;
        <span style='color:#14B8A6; font-weight:600;'>📤 Export</span>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(trend_chart(df), use_container_width=True)


# ── UPLOAD ────────────────────────────────────────────────────────────────────
def page_upload():
    st.markdown("# 📁 Upload Dataset")
    st.markdown("Upload your time-series CSV or XLSX file, or use the demo dataset.")
    st.markdown("<hr style='border-color:#1E2D4A;'>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload File", "⚡ Use Demo Data"])

    with tab1:
        uploaded = st.file_uploader(
            "Drop your CSV or XLSX here",
            type=["csv", "xlsx"],
            help="File must contain a date column and a numeric target column.",
        )

        if uploaded:
            try:
                with st.spinner("Reading file …"):
                    if uploaded.name.endswith(".xlsx"):
                        df_raw = pd.read_excel(uploaded)
                    else:
                        df_raw = pd.read_csv(uploaded)

                st.success(f"✅ File loaded: **{uploaded.name}** — {len(df_raw)} rows, {len(df_raw.columns)} columns")

                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("📅 Date column", df_raw.columns.tolist())
                with col2:
                    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
                    target_col = st.selectbox("🎯 Target column", num_cols)

                st.markdown("**Preview (first 10 rows):**")
                st.dataframe(df_raw.head(10), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Rows",    len(df_raw))
                c2.metric("Columns", len(df_raw.columns))
                c3.metric("Missing", int(df_raw.isnull().sum().sum()))

                # Missing value warnings
                miss = df_raw.isnull().sum()
                miss = miss[miss > 0]
                if not miss.empty:
                    st.warning(f"⚠️ Missing values detected: {miss.to_dict()}")

                if st.button("✅ Confirm & Load Dataset", type="primary"):
                    with st.spinner("Processing …"):
                        try:
                            df_raw[date_col] = pd.PeriodIndex(df_raw[date_col], freq="Q").to_timestamp()
                            df_raw.set_index(date_col, inplace=True)
                            df_raw.sort_index(inplace=True)
                            df_raw["Deposit_Amount"] = df_raw[target_col].astype(float)
                            if "Log_Deposit" not in df_raw.columns:
                                df_raw["Log_Deposit"] = np.log(df_raw["Deposit_Amount"])
                            if "YoY_Growth" not in df_raw.columns:
                                df_raw["YoY_Growth"] = df_raw["Deposit_Amount"].pct_change(4) * 100
                            _set_dataset(df_raw)
                            st.success("✅ Dataset loaded into the platform!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error parsing dataset: {e}")

            except Exception as e:
                st.error(f"❌ Could not read file: {e}")

    with tab2:
        st.markdown("""
        <div style='background:#0F1D35; border:1px solid #1E3A5F; border-radius:12px; padding:24px;'>
            <div style='font-size:1rem; font-weight:600; color:#F1F5F9; margin-bottom:12px;'>
                📊 BOI CASA Deposit Demo Dataset
            </div>
            <div style='color:#94A3B8; font-size:0.88rem; line-height:1.6;'>
                Quarterly CASA deposit data from 2015 Q1 to 2024 Q4 (40 periods).<br>
                Includes pre-computed features: YoY growth, log transform, 
                seasonal dummies, rolling averages.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Load Demo Dataset", type="primary"):
            with st.spinner("Loading demo data …"):
                _load_demo()
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("## 📋 Dataset Summary")
        s = df["Deposit_Amount"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations",  len(df))
        c2.metric("Mean Deposit",  f"₹{s.mean()/1e7:.2f}Cr")
        c3.metric("Max Deposit",   f"₹{s.max()/1e7:.2f}Cr")
        c4.metric("Min Deposit",   f"₹{s.min()/1e7:.2f}Cr")
        st.dataframe(df[["Deposit_Amount", "Log_Deposit", "YoY_Growth"]].describe().round(2),
                     use_container_width=True)


# ── EDA ───────────────────────────────────────────────────────────────────────
def page_eda():
    st.markdown("# 🔍 Exploratory Data Analysis")
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first (sidebar → Load Demo Data).")
        return

    df = st.session_state.df
    s  = df["Deposit_Amount"]

    tabs = st.tabs(["📈 Trend", "🌊 Seasonality", "📊 Distribution",
                     "🔄 Rolling Stats", "🗓️ Growth Heatmap"])

    with tabs[0]:
        st.plotly_chart(trend_chart(df), use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Growth",   f"{(s.iloc[-1]/s.iloc[0]-1)*100:+.1f}%")
        col2.metric("CAGR (est.)",    f"{((s.iloc[-1]/s.iloc[0])**(1/9)-1)*100:.2f}%")
        col3.metric("Volatility",     f"{s.std()/s.mean()*100:.2f}%")
        col4.metric("Peak Quarter",   str(s.idxmax())[:7])

    with tabs[1]:
        period = st.slider("Seasonal period (quarters)", 2, 8, 4)
        st.plotly_chart(seasonal_decomp_chart(s, period=period), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(distribution_chart(s), use_container_width=True)
        with st.expander("📐 Descriptive Statistics"):
            st.dataframe(s.describe().round(2).to_frame("Deposit_Amount"),
                         use_container_width=True)

    with tabs[3]:
        w1 = st.slider("Window 1 (quarters)", 2, 8, 4)
        w2 = st.slider("Window 2 (quarters)", 2, 12, 8)
        st.plotly_chart(rolling_stats_chart(s, windows=[w1, w2]), use_container_width=True)

    with tabs[4]:
        if "YoY_Growth" in df.columns:
            st.plotly_chart(growth_heatmap_chart(df), use_container_width=True)
        else:
            st.info("YoY Growth column not available in this dataset.")


# ── FORECASTING ───────────────────────────────────────────────────────────────
def page_forecasting():
    st.markdown("# 🤖 Forecasting Models")
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first.")
        return

    st.markdown("<hr style='border-color:#1E2D4A;'>", unsafe_allow_html=True)

    # Config panel
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("### ⚙️ Forecast Configuration")
        selected = st.multiselect(
            "Select Models to Train",
            options=list(FITTERS.keys()),
            default=st.session_state.selected_models,
        )
        st.session_state.selected_models = selected

        col_a, col_b = st.columns(2)
        with col_a:
            frac = st.slider("Train fraction", 0.6, 0.9, st.session_state.train_frac, 0.05)
            if frac != st.session_state.train_frac:
                st.session_state.train_frac = frac
                _set_dataset(st.session_state.df)
        with col_b:
            st.session_state.horizon = st.slider(
                "Forecast horizon (quarters)", 1, 12, st.session_state.horizon
            )

        train = st.session_state.train
        test  = st.session_state.test
        st.markdown(f"""
        <div style='background:#0D1526; border:1px solid #1E2D4A; border-radius:8px; 
                    padding:12px 16px; font-size:0.82rem; color:#94A3B8; margin-top:8px;'>
            📐 Train: <b style='color:#F1F5F9'>{len(train)} quarters</b> &nbsp;|&nbsp; 
            Test: <b style='color:#F1F5F9'>{len(test)} quarters</b>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📋 Model Descriptions")
        desc_map = {
            "ARIMA":     "Auto-Regressive Integrated Moving Average — classical univariate model.",
            "SARIMA":    "ARIMA with seasonal components — handles Q4 effects well.",
            "SARIMAX":   "SARIMA with real macro exogenous variables.",
            "AutoARIMA": "pmdarima grid-search — automatically selects optimal p,d,q.",
            "HoltWinters": "Exponential smoothing with additive trend and quarterly seasonality.",
            "Prophet":   "Meta additive model — handles holidays, trend changepoints.",
        }
        for model in selected:
            st.markdown(f"""
            <div style='margin-bottom:8px; padding:10px 14px; 
                        background:#0F1D35; border:1px solid #1E3A5F; border-radius:8px;'>
                <b style='color:#60A5FA; font-size:0.85rem;'>{model}</b><br>
                <span style='color:#94A3B8; font-size:0.78rem;'>{desc_map.get(model, "")}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
        if not selected:
            st.error("Please select at least one model.")
            return
        with st.spinner(""):
            results = run_selected_models(selected)
        if results:
            st.session_state.results      = results
            st.session_state.pipeline_ran = True
            # Build comparison object
            comp = ModelComparison()
            for name, rec in results.items():
                comp.add(name, st.session_state.test.values,
                         rec["forecast"].values, rec["residuals"],
                         train_time=rec.get("train_time", 0))
            st.session_state.comparison = comp
            st.success(f"✅ {len(results)} model(s) trained successfully!")
            st.rerun()

    # Show results
    if st.session_state.pipeline_ran and st.session_state.results:
        results = st.session_state.results
        train   = st.session_state.train
        test    = st.session_state.test

        # Metric cards
        st.markdown("---")
        st.markdown("### 📊 Model Metrics")
        cols = st.columns(len(results))
        for col, (name, rec) in zip(cols, results.items()):
            with col:
                mape = rec["metrics"]["MAPE"]
                r2   = rec["metrics"]["R2"]
                col.metric(f"**{name}**", f"MAPE {mape:.2f}%",
                           delta=f"R²={r2:.3f}", delta_color="normal")

        # Forecast chart
        st.markdown("---")
        forecasts = {k: v["forecast"] for k, v in results.items()}
        ci_map    = {k: v["ci"]       for k, v in results.items()}
        st.plotly_chart(
            forecast_vs_actual_chart(train, test, forecasts, ci_map),
            use_container_width=True
        )

        # Individual model detail
        st.markdown("---")
        st.markdown("### 🔎 Model Detail")
        selected_detail = st.selectbox("Inspect model:", list(results.keys()))
        rec = results[selected_detail]
        m   = rec["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE %",       f"{m['MAE_%']:.2f}%")
        c2.metric("RMSE %",      f"{m['RMSE_%']:.2f}%")
        c3.metric("MAPE",        f"{m['MAPE']:.2f}%")
        c4.metric("R²",          f"{m['R2']:.4f}")
        c5.metric("Stability",   f"{m['Stability']:.1f}/100")

        import plotly.graph_objects as go
        fc   = rec["forecast"]
        ci   = rec["ci"]
        col  = {"ARIMA":"#3B82F6","SARIMA":"#10B981","SARIMAX":"#F59E0B",
                 "AutoARIMA":"#EF4444","HoltWinters":"#8B5CF6",
                 "Prophet":"#EC4899"}.get(selected_detail, "#3B82F6")
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=train.index.astype(str), y=train.values,
                                  name="Train", line=dict(color="#94A3B8", width=2)))
        fig.add_trace(go.Scatter(x=test.index.astype(str), y=test.values,
                                  name="Actual", line=dict(color="#F1F5F9", width=2.5, dash="dot"),
                                  mode="lines+markers", marker=dict(size=8, symbol="diamond")))
        fig.add_trace(go.Scatter(x=fc.index.astype(str), y=fc.values,
                                  name=f"{selected_detail} Forecast",
                                  line=dict(color=col, width=2.5), mode="lines+markers",
                                  marker=dict(size=7)))
        r,g,b = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
        fig.add_trace(go.Scatter(
            x=list(ci.index.astype(str)) + list(ci.index.astype(str))[::-1],
            y=list(ci["upper"]) + list(ci["lower"])[::-1],
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True,
        ))
        fig.update_layout(
            title=f"<b>{selected_detail} — Forecast vs Actual + 95% CI</b>",
            paper_bgcolor="#080E1A", plot_bgcolor="#0F1D35",
            font=dict(color="#F1F5F9"), hovermode="x unified",
            xaxis=dict(gridcolor="#1E2D4A"), yaxis=dict(gridcolor="#1E2D4A"),
            legend=dict(bgcolor="#0D1526", bordercolor="#1E2D4A"),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
def page_comparison():
    st.markdown("# ⚖️ Model Comparison")
    if not st.session_state.pipeline_ran or not st.session_state.results:
        st.warning("⚠️ Please train models first (Forecasting Models page).")
        return

    comp    = st.session_state.comparison
    lb      = comp.leaderboard()
    results = st.session_state.results
    insights = comp.generate_insights()

    # Leaderboard KPIs
    best = insights["best_model"]
    best_m = results[best]["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Model",       best)
    c2.metric("Best MAPE",            f"{best_m['MAPE']:.2f}%")
    c3.metric("Composite Score",      f"{lb.loc[best,'Composite_Score']:.1f}/100")
    c4.metric("R² Score",             f"{best_m['R2']:.4f}")

    st.markdown("<hr style='border-color:#1E2D4A;'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Leaderboard", "📊 Bar Chart",
                                       "🕸️ Radar", "💡 Insights"])

    with tab1:
        display_cols = ["Rank"] + [c for c in ["MAE_%", "RMSE_%", "MAPE", "R2",
                                                "Stability", "Composite_Score",
                                                "Train_Time_s"] if c in lb.columns]
        st.dataframe(
            lb[display_cols].style
              .highlight_min(subset=["MAPE", "MAE_%", "RMSE_%"], color="#0A2A0A", axis=0)
              .highlight_max(subset=["Composite_Score", "R2", "Stability"], color="#0A1F35", axis=0)
              .format(precision=3),
            use_container_width=True,
        )
        st.plotly_chart(leaderboard_chart(lb), use_container_width=True)

    with tab2:
        metrics_display = comp.metrics_df()
        pct_cols = [c for c in metrics_display.columns if "%" in c or c == "MAPE"]
        if pct_cols:
            st.plotly_chart(model_comparison_bar(metrics_display[pct_cols]),
                            use_container_width=True)

    with tab3:
        st.plotly_chart(radar_chart(comp.metrics_df()), use_container_width=True)

    with tab4:
        st.markdown(f"""
        <div style='background:#0A1F35; border:1px solid #1E3A5F; border-radius:12px; 
                    padding:20px; margin-bottom:20px;'>
            <div style='font-size:1.1rem; font-weight:600; color:#60A5FA; margin-bottom:10px;'>
                🏦 Executive Summary
            </div>
            <p style='color:#E2E8F0; line-height:1.7;'>{insights["executive_summary"]}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Model Recommendations")
            for rec in insights.get("model_recommendations", []):
                st.markdown(f"- {rec}")
        with col2:
            st.markdown("#### 🏛️ Banking Insights")
            for ins in insights.get("banking_insights", []):
                st.markdown(f"- {ins}")


# ── RESIDUAL DIAGNOSTICS ──────────────────────────────────────────────────────
def page_diagnostics():
    st.markdown("# 🧪 Residual Diagnostics")
    if not st.session_state.pipeline_ran or not st.session_state.results:
        st.warning("⚠️ Please train models first.")
        return

    results = st.session_state.results
    model   = st.selectbox("Select model for diagnostics:", list(results.keys()))
    rec     = results[model]
    residuals = np.array(rec["residuals"], dtype=float)
    residuals = residuals[~np.isnan(residuals)]

    # 4-panel chart
    st.plotly_chart(residual_dashboard(residuals, model_name=model),
                    use_container_width=True)

    # Statistical tests
    st.markdown("---")
    st.markdown("### 🔬 Statistical Test Results")

    diag = run_full_diagnostics(residuals, series=st.session_state.train, model_name=model)

    health = diag["health_score"]
    health_color = "#10B981" if health >= 75 else "#F59E0B" if health >= 50 else "#EF4444"

    st.markdown(f"""
    <div style='display:inline-block; background:{health_color}22; 
                border:1px solid {health_color}; border-radius:10px;
                padding:10px 20px; margin-bottom:20px;'>
        <span style='color:{health_color}; font-weight:700; font-size:1.1rem;'>
            Diagnostic Health: {health:.0f}/100 — {diag["health_label"]}
        </span>
    </div>
    """, unsafe_allow_html=True)

    tests = ["shapiro", "jarque_bera", "ljung_box", "arch", "adf"]
    test_labels = {
        "shapiro":    ("Shapiro-Wilk",    "normal"),
        "jarque_bera":("Jarque-Bera",     "normal"),
        "ljung_box":  ("Ljung-Box",       "no_autocorr"),
        "arch":       ("ARCH-LM",         "no_arch"),
        "adf":        ("ADF (Stationarity)", "stationary"),
    }

    cols = st.columns(len([t for t in tests if t in diag]))
    ci = 0
    for test_key in tests:
        if test_key not in diag:
            continue
        r = diag[test_key]
        passed_key = test_labels[test_key][1]
        passed     = r.get(passed_key, False)
        col_c = "#10B981" if passed else "#EF4444"
        with cols[ci]:
            st.markdown(f"""
            <div style='background:#0F1D35; border:1px solid {col_c}55; 
                        border-radius:10px; padding:14px; text-align:center;'>
                <div style='font-size:0.72rem; color:#64748B; text-transform:uppercase; 
                            letter-spacing:0.08em;'>{test_labels[test_key][0]}</div>
                <div style='font-size:1.4rem; margin:6px 0;'>
                    {'✅' if passed else '❌'}
                </div>
                <div style='color:{col_c}; font-size:0.78rem; font-weight:600;'>
                    {r.get("verdict", "—")}
                </div>
                <div style='color:#64748B; font-size:0.72rem; margin-top:4px;'>
                    p={r.get("p_value", "—")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        ci += 1

    # Interpretation
    st.markdown("<br>", unsafe_allow_html=True)
    for test_key in tests:
        if test_key not in diag:
            continue
        r = diag[test_key]
        interp = r.get("interpretation", "")
        if interp:
            with st.expander(f"📖 {test_labels[test_key][0]} interpretation"):
                st.info(interp)

    # Residual stats
    with st.expander("📊 Residual Summary Statistics"):
        stats_d = diag.get("stats", {})
        if stats_d:
            st.dataframe(pd.DataFrame([stats_d]).T.rename(columns={0: "Value"}).round(4),
                         use_container_width=True)


# ── EXPORT ────────────────────────────────────────────────────────────────────
def page_export():
    st.markdown("# 📤 Export Results")
    if not st.session_state.pipeline_ran or not st.session_state.results:
        st.warning("⚠️ Please train models first to export results.")
        return

    results  = st.session_state.results
    comp     = st.session_state.comparison
    test     = st.session_state.test
    insights = comp.generate_insights()

    st.markdown("### 📦 Available Exports")
    c1, c2, c3 = st.columns(3)

    # ── Forecasts CSV ──────────────────────────────────────────────────
    with c1:
        fc_rows = {k: v["forecast"] for k, v in results.items()}
        fc_df   = pd.DataFrame(fc_rows)
        fc_df["Actual"] = test.values[:len(fc_df)]
        buf = io.BytesIO()
        fc_df.to_csv(buf)
        st.download_button(
            "📥 Download Forecasts CSV",
            data=buf.getvalue(),
            file_name="boi_casa_forecasts.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(fc_df)} forecast periods | {len(results)} models")

    # ── Metrics CSV ────────────────────────────────────────────────────
    with c2:
        lb  = comp.leaderboard()
        buf2 = io.BytesIO()
        lb.to_csv(buf2)
        st.download_button(
            "📥 Download Leaderboard CSV",
            data=buf2.getvalue(),
            file_name="boi_casa_leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(lb)} models | with composite scores")

    # ── Markdown Report ────────────────────────────────────────────────
    with c3:
        md_lines = [
            "# BOI CASA Deposit Forecasting — Report\n",
            f"**Generated:** {insights['timestamp']}\n",
            f"**Models evaluated:** {insights['n_models']}\n\n",
            "## Executive Summary\n",
            insights["executive_summary"] + "\n\n",
            "## Model Leaderboard\n",
            comp.to_markdown() + "\n\n",
            "## Recommendations\n",
        ]
        for r in insights.get("model_recommendations", []):
            md_lines.append(f"- {r}\n")
        md_lines.append("\n## Banking Insights\n")
        for ins in insights.get("banking_insights", []):
            md_lines.append(f"- {ins}\n")

        md_content = "".join(md_lines)
        st.download_button(
            "📥 Download Markdown Report",
            data=md_content.encode(),
            file_name="boi_casa_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.caption("Executive summary + leaderboard + insights")

    # Preview
    st.markdown("---")
    st.markdown("### 📋 Forecast Preview")
    st.dataframe(fc_df.round(0), use_container_width=True)

    st.markdown("### 🏆 Leaderboard Preview")
    display_cols = [c for c in ["Rank","MAPE","MAE_%","RMSE_%","R2",
                                 "Stability","Composite_Score"] if c in lb.columns]
    st.dataframe(lb[display_cols].round(3), use_container_width=True)


# ── SETTINGS ──────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown("# ⚙️ Settings & About")

    tab1, tab2 = st.tabs(["⚙️ Configuration", "ℹ️ About"])

    with tab1:
        st.markdown("### 🔧 Pipeline Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Seasonal Period", [4, 12, 52], index=0,
                         help="4=Quarterly, 12=Monthly, 52=Weekly")
            st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
        with col2:
            st.slider("AutoARIMA max_p", 1, 6, 4)
            st.slider("AutoARIMA max_q", 1, 6, 4)

        st.markdown("### 🗑️ Session Management")
        if st.button("🔄 Reset All Session Data", type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    with tab2:
        st.markdown("""
        ## 📊 BOI CASA Deposit Forecasting Platform
        **Version:** 1.0.0  
        **Built with:** Python · Streamlit · Plotly · Statsmodels · Prophet · pmdarima

        ### 🎯 Project Purpose
        This platform was developed as part of a **Bank of India data science internship**
        to build a production-ready forecasting system for CASA deposit prediction.

        ### 🤖 Models
        | Model | Library | Type |
        |---|---|---|
        | ARIMA | statsmodels | Classical |
        | SARIMA | statsmodels | Seasonal |
        | SARIMAX | statsmodels | Exogenous |
        | AutoARIMA | pmdarima | Auto-selection |
        | HoltWinters | statsmodels | Exponential smoothing |
        | Prophet | prophet (Meta) | Additive |

        ### 📐 Metrics
        MAE, RMSE, MAPE, SMAPE, R², Forecast Bias, Stability Score, Composite Score

        ### 🔬 Statistical Tests
        ADF (Stationarity), Shapiro-Wilk, Jarque-Bera, Ljung-Box, ARCH-LM

        ### 👤 Author
        Data Science Internship Project — BOI Analytics  
        GitHub: [github.com/yourusername/boi-casa-forecasting](https://github.com)
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    route = {
        "🏠 Home":               page_home,
        "📁 Upload Dataset":     page_upload,
        "🔍 Exploratory Analysis": page_eda,
        "🤖 Forecasting Models": page_forecasting,
        "⚖️ Model Comparison":   page_comparison,
        "🧪 Residual Diagnostics": page_diagnostics,
        "📤 Export Results":     page_export,
        "⚙️ Settings":          page_settings,
    }
    fn = route.get(page, page_home)
    fn()


if __name__ == "__main__":
    main()
