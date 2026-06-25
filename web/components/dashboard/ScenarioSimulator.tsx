"use client";

import { useMemo, useState } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency, formatPct } from "@/lib/formatters";
import type { Kpis } from "@/lib/types";

const controls = [
  { key: "repo", label: "RBI Repo Rate", min: 3, max: 9, step: 0.25, value: 6.5 },
  { key: "cpi", label: "CPI Inflation", min: 2, max: 9, step: 0.25, value: 4.8 },
  { key: "gdp", label: "GDP Growth", min: -2, max: 10, step: 0.25, value: 6.8 },
  { key: "industry", label: "CASA Industry Ratio", min: 30, max: 50, step: 0.5, value: 39 },
  { key: "cd", label: "Credit Deposit Ratio", min: 60, max: 90, step: 0.5, value: 76 }
] as const;

type ControlKey = (typeof controls)[number]["key"];
type State = Record<ControlKey, number>;

export function ScenarioSimulator({ kpis }: { kpis: Kpis }) {
  const [values, setValues] = useState<State>(
    Object.fromEntries(controls.map((control) => [control.key, control.value])) as State
  );

  const scenarios = useMemo(() => {
    const base = kpis.forecast_next_period ?? kpis.current_deposit ?? 0;
    const pressure =
      (values.gdp - 6.5) * 0.45 +
      (values.industry - 39) * 0.2 -
      (values.repo - 6.5) * 0.32 -
      (values.cpi - 4.8) * 0.22 -
      (values.cd - 76) * 0.08;

    return [
      { label: "Optimistic", delta: pressure + 2.4, tone: "success" as const },
      { label: "Base", delta: pressure, tone: "default" as const },
      { label: "Stress", delta: pressure - 2.8, tone: "warning" as const }
    ].map((scenario) => ({
      ...scenario,
      forecast: base * (1 + scenario.delta / 100)
    }));
  }, [kpis.current_deposit, kpis.forecast_next_period, values]);

  return (
    <div className="grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
      <Card>
        <CardHeader>
          <CardTitle>Macro Assumptions</CardTitle>
          <CardDescription>Illustrative simulator. Connect to SARIMAX future exog for production use.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {controls.map((control) => (
            <label key={control.key} className="block">
              <div className="mb-2 flex items-center justify-between text-sm">
                <span className="font-medium text-slate-200">{control.label}</span>
                <span className="text-slate-400">{values[control.key].toFixed(2)}%</span>
              </div>
              <input
                type="range"
                min={control.min}
                max={control.max}
                step={control.step}
                value={values[control.key]}
                onChange={(event) =>
                  setValues((current) => ({ ...current, [control.key]: Number(event.target.value) }))
                }
                className="h-2 w-full accent-cyan-300"
              />
            </label>
          ))}
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-1">
        {scenarios.map((scenario) => {
          const positive = scenario.delta >= 0;
          const Icon = positive ? TrendingUp : TrendingDown;
          return (
            <Card key={scenario.label}>
              <CardContent className="p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm text-slate-400">{scenario.label}</p>
                    <p className="mt-2 text-2xl font-semibold text-white">{formatCurrency(scenario.forecast)}</p>
                  </div>
                  <Badge variant={scenario.tone}>
                    <Icon className="mr-1 h-3 w-3" />
                    {formatPct(scenario.delta)}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

