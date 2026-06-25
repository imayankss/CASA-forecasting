import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const steps = [
  ["Load", "Read CASA deposits and macro indicators, then engineer lags, growth, and volatility features."],
  ["Split", "Use chronological train/test splits so future observations never influence training."],
  ["Model", "Evaluate statistical, seasonal, exogenous, smoothing, automated, and ensemble methods."],
  ["Validate", "Run walk-forward validation and compare error, direction accuracy, stability, and fit."],
  ["Monitor", "Export confidence, anomaly, and drift artifacts for dashboard consumption."]
];

export function MethodologyTimeline() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Forecasting Workflow</CardTitle>
        <CardDescription>Single-source-of-truth ML pipeline feeding a static web dashboard.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-5">
          {steps.map(([title, body], index) => (
            <div key={title} className="relative rounded-lg border border-slate-700/70 bg-slate-950/45 p-4">
              <div className="mb-3 flex h-8 w-8 items-center justify-center rounded-md border border-cyan-300/35 bg-cyan-400/10 text-sm font-semibold text-cyan-100">
                {index + 1}
              </div>
              <p className="font-semibold text-white">{title}</p>
              <p className="mt-2 text-sm leading-6 text-slate-400">{body}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

