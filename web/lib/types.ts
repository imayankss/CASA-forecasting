export type SourceFile = {
  path: string;
  exists: boolean;
  modified_at: string | null;
};

export type Manifest = {
  generated_at: string;
  project_name: string;
  data_start: string | null;
  data_end: string | null;
  row_count: number;
  model_count: number;
  best_model: string | null;
  best_mape: number | null;
  best_r2: number | null;
  source_files: Record<string, SourceFile>;
};

export type Kpis = {
  current_deposit: number | null;
  latest_period: string | null;
  forecast_next_period: number | null;
  forecast_period?: string | null;
  forecast_growth_pct: number | null;
  best_model: string | null;
  best_mape: number | null;
  confidence_score: number | null;
  confidence_label: string | null;
  risk_status: string;
};

export type ForecastRow = {
  period: string | null;
  actual: number | null;
  lower_ci?: number | null;
  upper_ci?: number | null;
  ensemble?: number | null;
  [model: string]: string | number | null | undefined;
};

export type LeaderboardRow = {
  rank: number;
  model: string;
  mape: number | null;
  rmse_pct: number | null;
  mae_pct: number | null;
  r2: number | null;
  adjusted_r2: number | null;
  direction_accuracy: number | null;
  stability: number | null;
  composite_score: number | null;
  train_time_s?: number | null;
  recommended: boolean;
  badges: string[];
};

export type CvRow = {
  model: string;
  cv_mape_mean: number | null;
  cv_mape_std: number | null;
  cv_rmse_mean: number | null;
  cv_r2_mean: number | null;
  cv_direction_accuracy: number | null;
  folds: number | null;
};

export type ConfidenceRow = {
  period: string | null;
  forecast: number | null;
  lower_95: number | null;
  upper_95: number | null;
  confidence_score: number | null;
  confidence_label: string | null;
  drift_penalty: number | null;
};

export type AnomalyRow = {
  period: string | null;
  deposit_amount: number | null;
  detection_method: string;
  severity: string;
  explanation: string;
};

export type DriftPayload = {
  drift_detected: boolean;
  drift_type: string;
  test_statistic: number | null;
  p_value: number | null;
  cusum_signal: boolean;
  residual_drift: boolean;
  model_health_score: number | null;
  interpretation: string;
};

export type Insights = {
  executive_summary: string;
  best_model_reason: string;
  risk_interpretation: string;
  model_recommendations: string[];
  banking_insights: string[];
};

export type DashboardData = {
  manifest: Manifest;
  kpis: Kpis;
  forecast: ForecastRow[];
  leaderboard: LeaderboardRow[];
  modelMetrics: Record<string, unknown>[];
  cvSummary: CvRow[];
  confidenceScores: ConfidenceRow[];
  insights: Insights;
  anomalies: AnomalyRow[];
  drift: DriftPayload;
};

