"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber, formatPct } from "@/lib/formatters";
import type { LeaderboardRow } from "@/lib/types";

type MetricKey = "mape" | "rmse_pct" | "r2" | "composite_score";

const labels: Record<MetricKey, string> = {
  mape: "MAPE",
  rmse_pct: "RMSE %",
  r2: "R2",
  composite_score: "Composite"
};

export function ModelComparisonChart({
  rows,
  metric = "mape",
  title
}: {
  rows: LeaderboardRow[];
  metric?: MetricKey;
  title?: string;
}) {
  const chartData = rows.map((row) => ({
    model: row.model,
    value: row[metric] ?? 0
  }));

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{title ?? `${labels[metric]} Comparison`}</CardTitle>
        <CardDescription>Model selection view from exported leaderboard metrics.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72">
          {rows.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">
              No model metrics exported.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ left: 8, right: 14, top: 8, bottom: 8 }}>
                <CartesianGrid className="chart-grid" vertical={false} />
                <XAxis dataKey="model" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} width={44} />
                <Tooltip
                  formatter={(value) =>
                    metric === "mape" || metric === "rmse_pct"
                      ? formatPct(Number(value))
                      : formatNumber(Number(value), 3)
                  }
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]} fill="#22d3ee" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

