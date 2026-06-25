"use client";

import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { LeaderboardRow } from "@/lib/types";

export function MetricRadar({ model }: { model?: LeaderboardRow }) {
  const data = model
    ? [
        { metric: "Accuracy", value: invert(model.mape, 8) },
        { metric: "RMSE", value: invert(model.rmse_pct, 8) },
        { metric: "R2", value: clamp((model.r2 ?? 0) * 100) },
        { metric: "Direction", value: model.direction_accuracy ?? 50 },
        { metric: "Stability", value: model.stability ?? 50 },
        { metric: "Composite", value: model.composite_score ?? 50 }
      ]
    : [];

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Model Profile</CardTitle>
        <CardDescription>{model ? `${model.model} normalized performance profile.` : "No model selected."}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          {model ? (
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={data} outerRadius="72%">
                <PolarGrid stroke="rgba(148, 163, 184, 0.18)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
                <Tooltip formatter={(value) => `${Number(value).toFixed(1)}/100`} />
                <Radar dataKey="value" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.28} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">
              No leaderboard export found.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function invert(value: number | null, max: number) {
  if (value === null || value === undefined) return 50;
  return clamp(100 - (value / max) * 100);
}

function clamp(value: number) {
  return Math.min(100, Math.max(0, value));
}

