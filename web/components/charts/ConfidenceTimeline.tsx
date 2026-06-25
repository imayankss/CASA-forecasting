"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge, type BadgeVariant } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency, formatScore } from "@/lib/formatters";
import type { ConfidenceRow } from "@/lib/types";

export function ConfidenceTimeline({ rows }: { rows: ConfidenceRow[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Confidence Timeline</CardTitle>
        <CardDescription>Per-quarter forecast confidence and interval expansion.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {rows.length === 0 ? (
            <div className="rounded-md border border-slate-700/70 bg-slate-950/45 p-4 text-sm text-slate-500">
              No confidence score export found.
            </div>
          ) : (
            rows.map((row) => (
              <div key={row.period} className="rounded-md border border-slate-700/70 bg-slate-950/45 p-4">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium text-slate-200">{row.period}</p>
                  <Badge variant={badgeVariant(row.confidence_score)}>{row.confidence_label ?? "Unknown"}</Badge>
                </div>
                <p className="mt-3 text-xl font-semibold text-white">{formatCurrency(row.forecast)}</p>
                <p className="mt-1 text-sm text-slate-500">{formatScore(row.confidence_score)}</p>
              </div>
            ))
          )}
        </div>

        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={rows} margin={{ left: 8, right: 16, top: 10, bottom: 8 }}>
              <defs>
                <linearGradient id="confidenceArea" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity={0.32} />
                  <stop offset="100%" stopColor="#34d399" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid className="chart-grid" vertical={false} />
              <XAxis dataKey="period" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} width={44} />
              <Tooltip formatter={(value) => formatScore(Number(value))} />
              <Area
                type="monotone"
                dataKey="confidence_score"
                name="Confidence"
                stroke="#34d399"
                strokeWidth={2}
                fill="url(#confidenceArea)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function badgeVariant(score: number | null): BadgeVariant {
  if (score === null || score === undefined) return "muted";
  if (score >= 70) return "success";
  if (score >= 45) return "warning";
  return "danger";
}
