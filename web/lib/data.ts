import { promises as fs } from "fs";
import path from "path";

import type {
  AnomalyRow,
  ConfidenceRow,
  CvRow,
  DashboardData,
  DriftPayload,
  ForecastRow,
  Insights,
  Kpis,
  LeaderboardRow,
  Manifest
} from "@/lib/types";

const dataDir = path.join(process.cwd(), "public", "data");

async function readJson<T>(fileName: string, fallback: T): Promise<T> {
  try {
    const file = await fs.readFile(path.join(dataDir, fileName), "utf8");
    return JSON.parse(file) as T;
  } catch {
    return fallback;
  }
}

const emptyManifest: Manifest = {
  generated_at: "",
  project_name: "CASA Intelligence Dashboard",
  data_start: null,
  data_end: null,
  row_count: 0,
  model_count: 0,
  best_model: null,
  best_mape: null,
  best_r2: null,
  source_files: {}
};

const emptyKpis: Kpis = {
  current_deposit: null,
  latest_period: null,
  forecast_next_period: null,
  forecast_growth_pct: null,
  best_model: null,
  best_mape: null,
  confidence_score: null,
  confidence_label: null,
  risk_status: "Unknown"
};

const emptyInsights: Insights = {
  executive_summary: "Run the Python export step to populate the dashboard.",
  best_model_reason: "No leaderboard artifact has been exported yet.",
  risk_interpretation: "Risk artifacts are unavailable.",
  model_recommendations: [],
  banking_insights: []
};

const emptyDrift: DriftPayload = {
  drift_detected: false,
  drift_type: "Unavailable",
  test_statistic: null,
  p_value: null,
  cusum_signal: false,
  residual_drift: false,
  model_health_score: null,
  interpretation: "No drift artifact was found."
};

export async function getDashboardData(): Promise<DashboardData> {
  const [
    manifest,
    kpis,
    forecast,
    leaderboard,
    modelMetrics,
    cvSummary,
    confidenceScores,
    insights,
    anomalies,
    drift
  ] = await Promise.all([
    readJson<Manifest>("manifest.json", emptyManifest),
    readJson<Kpis>("kpis.json", emptyKpis),
    readJson<ForecastRow[]>("forecast.json", []),
    readJson<LeaderboardRow[]>("leaderboard.json", []),
    readJson<Record<string, unknown>[]>("model_metrics.json", []),
    readJson<CvRow[]>("cv_summary.json", []),
    readJson<ConfidenceRow[]>("confidence_scores.json", []),
    readJson<Insights>("insights.json", emptyInsights),
    readJson<AnomalyRow[]>("anomalies.json", []),
    readJson<DriftPayload>("drift.json", emptyDrift)
  ]);

  return {
    manifest,
    kpis,
    forecast,
    leaderboard,
    modelMetrics,
    cvSummary,
    confidenceScores,
    insights,
    anomalies,
    drift
  };
}

