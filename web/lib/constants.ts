import {
  Activity,
  BarChart3,
  BrainCircuit,
  Gauge,
  LineChart,
  Layers3,
  Map,
  Radar,
  SlidersHorizontal
} from "lucide-react";

export const navItems = [
  { href: "/", label: "Overview", icon: Gauge },
  { href: "/forecast", label: "Forecast", icon: LineChart },
  { href: "/models", label: "Models", icon: BarChart3 },
  { href: "/risk", label: "Risk", icon: Radar },
  { href: "/diagnostics", label: "Diagnostics", icon: Activity },
  { href: "/seasonality", label: "Seasonality", icon: Layers3 },
  { href: "/scenario", label: "Scenario", icon: SlidersHorizontal },
  { href: "/methodology", label: "Methodology", icon: Map }
];

export const modelColors: Record<string, string> = {
  ARIMA: "#38bdf8",
  SARIMA: "#34d399",
  SARIMAX: "#f59e0b",
  AutoARIMA: "#fb7185",
  HoltWinters: "#a78bfa",
  Prophet: "#f472b6",
  ensemble: "#e5eefc",
  actual: "#22c55e"
};

export const modelFamilies = [
  {
    title: "Classical Baselines",
    models: "ARIMA, SARIMA",
    note: "Fast statistical anchors for trend and quarterly seasonality.",
    icon: BrainCircuit
  },
  {
    title: "Exogenous Forecasting",
    models: "SARIMAX",
    note: "Adds macro context such as repo rate, inflation, and GDP growth.",
    icon: Activity
  },
  {
    title: "Automated and Additive",
    models: "AutoARIMA, Prophet",
    note: "Useful benchmark families for automatic order and trend discovery.",
    icon: LineChart
  },
  {
    title: "Smoothing and Ensemble",
    models: "HoltWinters, Ensemble",
    note: "Seasonal smoothing and weighted model combination for planning stability.",
    icon: Layers3
  }
];

