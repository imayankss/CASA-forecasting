"""
web_exporter.py — Convert pipeline CSV/JSON outputs into static dashboard JSON.

The exporter is intentionally tolerant of optional advanced artifacts. Missing
confidence, anomaly, or drift outputs produce safe empty JSON structures so the
Next.js dashboard can still render.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


class WebDataExporter:
    """Export processed CASA forecasting artifacts for the static web app."""

    def __init__(self, root_dir: Optional[Path | str] = None, output_dir: Optional[Path | str] = None) -> None:
        self.root_dir = Path(root_dir or Path(__file__).resolve().parents[2])
        self.raw_dir = self.root_dir / "data" / "raw"
        self.processed_dir = self.root_dir / "data" / "processed"
        self.reports_dir = self.root_dir / "reports"
        self.output_dir = Path(output_dir or self.root_dir / "web" / "public" / "data")
        self.generated_files: List[Path] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_all(self) -> List[Path]:
        """Generate all dashboard JSON files and return their paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        raw = self._load_deposits()
        leaderboard = self._leaderboard_rows()
        metrics = self._metrics_rows()
        cv_summary = self._cv_rows()
        confidence = self._confidence_rows()
        forecast = self._forecast_rows()
        anomalies = self._anomaly_rows()
        drift = self._drift_payload()

        self._write_json("manifest.json", self._manifest(raw, leaderboard))
        self._write_json("kpis.json", self._kpis(raw, leaderboard, forecast, confidence, drift))
        self._write_json("forecast.json", forecast)
        self._write_json("leaderboard.json", leaderboard)
        self._write_json("model_metrics.json", metrics)
        self._write_json("cv_summary.json", cv_summary)
        self._write_json("confidence_scores.json", confidence)
        self._write_json("insights.json", self._insights(leaderboard, confidence, anomalies, drift))
        self._write_json("anomalies.json", anomalies)
        self._write_json("drift.json", drift)

        return self.generated_files

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_deposits(self) -> pd.DataFrame:
        path = self.raw_dir / "boi_casa_deposits.csv"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "Quarter_Year" in df.columns:
            df["period"] = pd.PeriodIndex(df["Quarter_Year"], freq="Q").to_timestamp()
        elif "Period" in df.columns:
            df["period"] = pd.to_datetime(df["Period"], errors="coerce")
        return df.sort_values("period") if "period" in df.columns else df

    def _read_csv(self, relative_path: str) -> pd.DataFrame:
        path = self.root_dir / relative_path
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _read_json(self, relative_path: str) -> Dict[str, Any]:
        path = self.root_dir / relative_path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _manifest(self, raw: pd.DataFrame, leaderboard: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = leaderboard[0] if leaderboard else {}
        source_files = {
            "deposits": self._source("data/raw/boi_casa_deposits.csv"),
            "leaderboard": self._source("data/processed/model_leaderboard.csv"),
            "metrics": self._source("data/processed/model_metrics.csv"),
            "forecast": self._source("data/processed/forecast_results.csv"),
            "cv_summary": self._source("data/processed/cv_summary.csv"),
            "confidence": self._source("data/processed/confidence_scores.csv"),
            "registry": self._source("reports/model_registry.json"),
        }

        data_start = None
        data_end = None
        if not raw.empty and "period" in raw.columns:
            data_start = self._format_period(raw["period"].iloc[0])
            data_end = self._format_period(raw["period"].iloc[-1])

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_name": "CASA Intelligence Dashboard",
            "data_start": data_start,
            "data_end": data_end,
            "row_count": int(len(raw)) if not raw.empty else 0,
            "model_count": len(leaderboard),
            "best_model": best.get("model"),
            "best_mape": best.get("mape"),
            "best_r2": best.get("r2"),
            "source_files": source_files,
        }

    def _kpis(
        self,
        raw: pd.DataFrame,
        leaderboard: List[Dict[str, Any]],
        forecast: List[Dict[str, Any]],
        confidence: List[Dict[str, Any]],
        drift: Dict[str, Any],
    ) -> Dict[str, Any]:
        latest_deposit = None
        latest_period = None
        if not raw.empty and "Deposit_Amount" in raw.columns:
            latest_deposit = self._to_float(raw["Deposit_Amount"].iloc[-1])
            latest_period = self._format_period(raw["period"].iloc[-1]) if "period" in raw.columns else None

        best = leaderboard[0] if leaderboard else {}
        latest_forecast = forecast[-1] if forecast else {}
        best_model = best.get("model")
        forecast_value = latest_forecast.get("ensemble") or latest_forecast.get(best_model)
        if forecast_value is None:
            model_keys = [
                key for key, value in latest_forecast.items()
                if key not in {"period", "actual", "lower_ci", "upper_ci"} and isinstance(value, (int, float))
            ]
            forecast_value = latest_forecast.get(model_keys[0]) if model_keys else None

        growth_pct = None
        if latest_deposit and forecast_value:
            growth_pct = round((forecast_value / latest_deposit - 1) * 100, 2)

        latest_conf = confidence[-1] if confidence else {}
        confidence_score = latest_conf.get("confidence_score")
        confidence_label = latest_conf.get("confidence_label") or self._confidence_label(confidence_score)
        risk_status = self._risk_status(confidence_score, drift)

        return {
            "current_deposit": latest_deposit,
            "latest_period": latest_period,
            "forecast_next_period": forecast_value,
            "forecast_period": latest_forecast.get("period"),
            "forecast_growth_pct": growth_pct,
            "best_model": best_model,
            "best_mape": best.get("mape"),
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "risk_status": risk_status,
        }

    def _forecast_rows(self) -> List[Dict[str, Any]]:
        df = self._read_csv("data/processed/forecast_results.csv")
        if df.empty:
            return []

        period_col = self._period_column(df)
        actual_col = "Actual_Test" if "Actual_Test" in df.columns else "Actual" if "Actual" in df.columns else None
        ensemble = self._read_csv("data/processed/ensemble_forecast.csv")
        confidence = self._read_csv("data/processed/confidence_scores.csv")

        ensemble_lookup = self._lookup_by_period(ensemble)
        confidence_lookup = self._lookup_by_period(confidence)

        rows = []
        for _, row in df.iterrows():
            period_raw = row.get(period_col)
            period = self._format_period(period_raw)
            out: Dict[str, Any] = {
                "period": period,
                "actual": self._to_float(row.get(actual_col)) if actual_col else None,
            }

            for col in df.columns:
                if col in {period_col, actual_col} or col.startswith("Unnamed"):
                    continue
                value = self._to_float(row.get(col))
                if value is not None:
                    out[col] = value

            ens = ensemble_lookup.get(period, {})
            conf = confidence_lookup.get(period, {})
            if ens:
                out["ensemble"] = self._to_float(ens.get("Ensemble_Forecast"))
                out["lower_ci"] = self._to_float(ens.get("CI_Lower"))
                out["upper_ci"] = self._to_float(ens.get("CI_Upper"))
            else:
                out["ensemble"] = out.get("Ensemble")
                out["lower_ci"] = self._to_float(conf.get("lower_95"))
                out["upper_ci"] = self._to_float(conf.get("upper_95"))

            rows.append(out)
        return rows

    def _leaderboard_rows(self) -> List[Dict[str, Any]]:
        df = self._read_csv("data/processed/model_leaderboard.csv")
        if df.empty:
            return []
        df = self._ensure_model_column(df)
        if "Rank" in df.columns:
            df = df.sort_values("Rank")

        rows = []
        for _, row in df.iterrows():
            rank = int(row.get("Rank", len(rows) + 1))
            rows.append({
                "rank": rank,
                "model": str(row.get("Model")),
                "mape": self._to_float(row.get("MAPE")),
                "rmse_pct": self._to_float(row.get("RMSE_%")),
                "mae_pct": self._to_float(row.get("MAE_%")),
                "r2": self._to_float(row.get("R2")),
                "adjusted_r2": self._to_float(row.get("Adjusted_R2")),
                "direction_accuracy": self._to_float(row.get("Direction_Accuracy")),
                "stability": self._to_float(row.get("Stability")),
                "composite_score": self._to_float(row.get("Composite_Score")),
                "train_time_s": self._to_float(row.get("Train_Time_s")),
                "recommended": bool(row.get("Recommended", rank == 1)),
                "badges": self._model_badges(rank, row),
            })
        return rows

    def _metrics_rows(self) -> List[Dict[str, Any]]:
        df = self._read_csv("data/processed/model_metrics.csv")
        if df.empty:
            return []
        df = self._ensure_model_column(df)
        return self._records(df)

    def _cv_rows(self) -> List[Dict[str, Any]]:
        df = self._read_csv("data/processed/cv_summary.csv")
        if df.empty:
            return []
        df = self._ensure_model_column(df)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "model": str(row.get("Model")),
                "cv_mape_mean": self._to_float(row.get("CV_MAPE_Mean")),
                "cv_mape_std": self._to_float(row.get("CV_MAPE_Std")),
                "cv_rmse_mean": self._to_float(row.get("CV_RMSE_%_Mean")),
                "cv_r2_mean": self._to_float(row.get("CV_R2_Mean")),
                "cv_direction_accuracy": self._to_float(row.get("CV_Dir_Acc_Mean")),
                "folds": self._to_int(row.get("CV_Folds")),
            })
        return rows

    def _confidence_rows(self) -> List[Dict[str, Any]]:
        df = self._read_csv("data/processed/confidence_scores.csv")
        if df.empty:
            return []
        if "period" not in df.columns:
            df = df.rename(columns={df.columns[0]: "period"})
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "period": self._format_period(row.get("period")),
                "forecast": self._to_float(row.get("forecast")),
                "lower_95": self._to_float(row.get("lower_95")),
                "upper_95": self._to_float(row.get("upper_95")),
                "confidence_score": self._to_float(row.get("confidence_score")),
                "confidence_label": row.get("confidence_label"),
                "drift_penalty": self._to_float(row.get("drift_penalty")),
            })
        return rows

    def _anomaly_rows(self) -> List[Dict[str, Any]]:
        df = self._read_csv("data/processed/anomaly_report.csv")
        if df.empty:
            return []
        period_col = self._period_column(df)
        rows = []
        for _, row in df.iterrows():
            votes = self._to_int(row.get("votes")) or 0
            rows.append({
                "period": self._format_period(row.get(period_col)),
                "deposit_amount": self._to_float(row.get("Deposit_Amount")),
                "detection_method": row.get("methods") or "unknown",
                "severity": "High" if votes >= 2 else "Medium",
                "explanation": "Deposit movement was flagged by anomaly detectors.",
            })
        return rows

    def _drift_payload(self) -> Dict[str, Any]:
        json_payload = self._read_json("data/processed/drift.json")
        if json_payload:
            return self._safe(json_payload)

        df = self._read_csv("data/processed/drift_report.csv")
        if not df.empty:
            row = df.iloc[0]
            return {
                "drift_detected": bool(row.get("overall_drift", False)),
                "drift_type": row.get("severity") or "None",
                "test_statistic": self._to_float(row.get("ks_statistic")),
                "p_value": self._to_float(row.get("ks_p_value")),
                "cusum_signal": bool(row.get("cusum_signal", False)),
                "residual_drift": bool(row.get("residual_drift", False)),
                "model_health_score": self._model_health_score(row),
                "interpretation": self._drift_interpretation(bool(row.get("overall_drift", False))),
            }

        return {
            "drift_detected": False,
            "drift_type": "Unavailable",
            "test_statistic": None,
            "p_value": None,
            "cusum_signal": False,
            "residual_drift": False,
            "model_health_score": None,
            "interpretation": "No drift artifact was found. Run run_advanced.py to generate monitoring output.",
        }

    def _insights(
        self,
        leaderboard: List[Dict[str, Any]],
        confidence: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]],
        drift: Dict[str, Any],
    ) -> Dict[str, Any]:
        best = leaderboard[0] if leaderboard else {}
        model_count = len(leaderboard)
        best_model = best.get("model") or "No model"
        best_mape = best.get("mape")
        avg_conf = None
        if confidence:
            scores = [row["confidence_score"] for row in confidence if row.get("confidence_score") is not None]
            avg_conf = round(float(np.mean(scores)), 1) if scores else None

        return {
            "executive_summary": (
                f"{best_model} leads the exported leaderboard across {model_count} evaluated models"
                + (f" with {best_mape:.2f}% MAPE." if best_mape is not None else ".")
            ),
            "best_model_reason": "The recommended model has the highest composite ranking in the exported leaderboard.",
            "risk_interpretation": self._risk_interpretation(avg_conf, anomalies, drift),
            "model_recommendations": [
                f"Use {best_model} as the default planning view until fresh pipeline outputs change the leaderboard.",
                "Review confidence scores by quarter before using forecasts for treasury decisions.",
                "Regenerate web data after every pipeline run to keep dashboard claims aligned with artifacts.",
            ],
            "banking_insights": [
                "CASA deposits show quarterly movement, so model selection should account for both accuracy and stability.",
                "Direction accuracy and confidence bands help distinguish useful forecasts from low-error but fragile fits.",
                "Anomaly and drift checks make the project closer to a monitored production forecasting workflow.",
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_json(self, filename: str, payload: Any) -> None:
        path = self.output_dir / filename
        path.write_text(json.dumps(self._safe(payload), indent=2))
        self.generated_files.append(path)
        print(f"generated {path}")

    def _source(self, relative_path: str) -> Dict[str, Any]:
        path = self.root_dir / relative_path
        return {
            "path": relative_path,
            "exists": path.exists(),
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if path.exists() else None,
        }

    def _ensure_model_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Model" not in df.columns:
            first = df.columns[0]
            if first.startswith("Unnamed") or first == "":
                df = df.rename(columns={first: "Model"})
        return df

    def _period_column(self, df: pd.DataFrame) -> str:
        for col in ["Period", "period", "Quarter_Year", "Date", "Unnamed: 0"]:
            if col in df.columns:
                return col
        return df.columns[0]

    def _lookup_by_period(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if df.empty:
            return {}
        period_col = self._period_column(df)
        out = {}
        for _, row in df.iterrows():
            out[self._format_period(row.get(period_col))] = row.to_dict()
        return out

    def _records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        return [self._safe(row) for row in df.to_dict(orient="records")]

    def _safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._safe(v) for v in value]
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if np.isnan(value) or np.isinf(value) else float(value)
        if isinstance(value, float):
            return None if np.isnan(value) or np.isinf(value) else value
        if pd.isna(value):
            return None
        return value

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or pd.isna(value):
                return None
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    def _to_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or pd.isna(value):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _format_period(self, value: Any) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        text = str(value)
        try:
            if "Q" in text and len(text) <= 8:
                period = pd.Period(text, freq="Q")
                return f"{period.year} Q{period.quarter}"
            ts = pd.to_datetime(value)
            return f"{ts.year} Q{ts.quarter}"
        except (ValueError, TypeError):
            return text[:10]

    def _model_badges(self, rank: int, row: pd.Series) -> List[str]:
        badges = []
        if rank == 1:
            badges.append("Production Candidate")
            badges.append("Best Accuracy")
        if self._to_float(row.get("Stability")) and self._to_float(row.get("Stability")) >= 75:
            badges.append("Most Stable")
        if self._to_float(row.get("Train_Time_s")) is not None and self._to_float(row.get("Train_Time_s")) <= 0.5:
            badges.append("Fastest")
        if self._to_float(row.get("MAPE")) is not None and self._to_float(row.get("MAPE")) >= 5:
            badges.append("Risky")
        return badges

    def _confidence_label(self, score: Optional[float]) -> Optional[str]:
        if score is None:
            return None
        if score >= 75:
            return "High"
        if score >= 45:
            return "Medium"
        return "Low"

    def _risk_status(self, score: Optional[float], drift: Dict[str, Any]) -> str:
        if drift.get("drift_detected"):
            return "Elevated"
        if score is None:
            return "Unknown"
        if score >= 70:
            return "Stable"
        if score >= 45:
            return "Watch"
        return "Elevated"

    def _risk_interpretation(
        self,
        avg_confidence: Optional[float],
        anomalies: Iterable[Dict[str, Any]],
        drift: Dict[str, Any],
    ) -> str:
        if drift.get("drift_detected"):
            return "Drift monitoring indicates elevated model risk. Retraining should be prioritized."
        if list(anomalies):
            return "Anomalous deposit periods exist and should be reviewed before executive use."
        if avg_confidence is None:
            return "Confidence artifacts are not available yet. Run advanced scoring for richer risk context."
        return f"Average exported confidence is {avg_confidence:.1f}/100, indicating a {self._confidence_label(avg_confidence)} planning view."

    def _drift_interpretation(self, drift_detected: bool) -> str:
        return (
            "Drift detected. Schedule retraining before using the forecast operationally."
            if drift_detected
            else "No drift detected in the exported monitoring artifact."
        )

    def _model_health_score(self, row: pd.Series) -> Optional[float]:
        alerts = sum(bool(row.get(key, False)) for key in ["ks_drift", "cusum_signal", "residual_drift"])
        return round(100 - alerts * 25, 1)
