"use client";

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { modelColors } from "@/lib/constants";
import { formatCurrency } from "@/lib/formatters";
import type { ForecastRow } from "@/lib/types";

export function ForecastChart({ rows, compact = false }: { rows: ForecastRow[]; compact?: boolean }) {
  const modelKeys = useMemo(() => {
    const keys = new Set<string>();
    rows.forEach((row) => {
      Object.keys(row).forEach((key) => {
        if (!["period", "actual", "lower_ci", "upper_ci"].includes(key) && typeof row[key] === "number") {
          keys.add(key);
        }
      });
    });
    return Array.from(keys);
  }, [rows]);
  const [selectedModel, setSelectedModel] = useState(modelKeys.includes("ensemble") ? "ensemble" : modelKeys[0] ?? "ensemble");

  const chartData = rows.map((row) => {
    const lower = typeof row.lower_ci === "number" ? row.lower_ci : null;
    const upper = typeof row.upper_ci === "number" ? row.upper_ci : null;
    return {
      period: row.period ?? "-",
      actual: row.actual,
      forecast: typeof row[selectedModel] === "number" ? row[selectedModel] : null,
      ciBase: lower,
      ciBand: lower !== null && upper !== null ? upper - lower : null
    };
  });

  return (
    <Card className="h-full">
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
          <CardTitle>{compact ? "Forecast Preview" : "Forecast vs Actual"}</CardTitle>
          <CardDescription>
            Exported holdout forecasts with confidence interval band when available.
          </CardDescription>
        </div>
        {!compact && modelKeys.length > 0 ? (
          <select
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
            className="h-10 rounded-md border border-slate-700 bg-slate-950/70 px-3 text-sm text-slate-100 outline-none focus:border-cyan-300"
          >
            {modelKeys.map((model) => (
              <option key={model} value={model}>
                {model === "ensemble" ? "Ensemble" : model}
              </option>
            ))}
          </select>
        ) : null}
      </CardHeader>
      <CardContent>
        <div className={compact ? "h-72" : "h-[420px]"}>
          {rows.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">
              No forecast export found.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ left: 8, right: 18, top: 8, bottom: 8 }}>
                <defs>
                  <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.34} />
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid className="chart-grid" vertical={false} />
                <XAxis dataKey="period" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(value) => `${Math.round(Number(value) / 1000)}k`}
                  width={56}
                />
                <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                {!compact ? <Legend wrapperStyle={{ color: "#cbd5e1", fontSize: 12 }} /> : null}
                <Area dataKey="ciBase" stackId="ci" stroke="none" fill="transparent" isAnimationActive={false} />
                <Area
                  dataKey="ciBand"
                  stackId="ci"
                  name="Confidence band"
                  stroke="none"
                  fill="url(#forecastFill)"
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Actual"
                  stroke={modelColors.actual}
                  strokeWidth={2.4}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  name={selectedModel === "ensemble" ? "Ensemble" : selectedModel}
                  stroke={modelColors[selectedModel] ?? "#22d3ee"}
                  strokeWidth={2.4}
                  dot={{ r: 3 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
