"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { axis, grid, quarter, tooltip } from "@/components/results/chart";

type Row = Record<string, number | string>;

const MODELS: { key: string; label: string; baseline?: boolean }[] = [
  { key: "Drift", label: "Drift", baseline: true },
  { key: "GlobalLGBM_FDIC", label: "LightGBM (FDIC-trained)" },
  { key: "HoltWinters", label: "Holt-Winters" },
  { key: "Chronos2", label: "Chronos-2" },
  { key: "SeasonalNaive", label: "Seasonal naive", baseline: true }
];

const crore = (v: number) => `₹${(v / 1e5).toFixed(2)}L cr`;

export function BoiExplorer({ history, forecasts, mape }: { history: Row[]; forecasts: Row[]; mape: Record<string, number> }) {
  const [model, setModel] = useState("GlobalLGBM_FDIC");
  const meta = MODELS.find((m) => m.key === model)!;
  const origin = String(forecasts[0].ds);
  const lastTrain = history.filter((r) => String(r.quarter_end) < origin).at(-1)!;

  // History from 2019 plus the holdout; the forecast line starts at the last training quarter.
  const data = [
    ...history
      .filter((r) => String(r.quarter_end) >= "2019-01-01" && String(r.quarter_end) < origin)
      .map((r) => ({ q: quarter(String(r.quarter_end)), actual: Number(r.casa_cr), forecast: undefined as number | undefined })),
    ...forecasts.map((r) => ({ q: quarter(String(r.ds)), actual: Number(r.actual), forecast: Number(r[model]) }))
  ];
  data[data.length - forecasts.length - 1].forecast = Number(lastTrain.casa_cr);

  const errors = forecasts.map((r) => Math.abs(Number(r[model]) - Number(r.actual)) / Number(r.actual) * 100);
  const worst = Math.max(...errors);

  return (
    <div className="panel">
      <div className="controls">
        <span className="label">Model</span>
        <div className="chips" role="group" aria-label="Choose a model">
          {MODELS.map((m) => (
            <button key={m.key} type="button" className="chip" aria-pressed={m.key === model} onClick={() => setModel(m.key)}>
              {m.label}
              <span className="sub">{mape[m.key]?.toFixed(2)}%</span>
            </button>
          ))}
        </div>
      </div>
      <div className="chart">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
            <CartesianGrid {...grid} vertical={false} />
            <XAxis dataKey="q" {...axis} interval={3} tickFormatter={(q: string) => q.slice(0, 4)} />
            <YAxis {...axis} domain={["auto", "auto"]} tickFormatter={crore} width={72} />
            <Tooltip {...tooltip} formatter={(v) => crore(Number(v))} />
            <ReferenceLine x={quarter(origin)} stroke="var(--axis)" strokeDasharray="3 3" label={{ value: "holdout →", fill: "var(--text-muted)", fontSize: 11, position: "insideTopLeft" }} />
            <Line dataKey="actual" name="BOI actual" stroke="var(--text-primary)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line dataKey="forecast" name={meta.label} stroke={meta.baseline ? "var(--series-2)" : "var(--series-1)"} strokeWidth={2} strokeDasharray={meta.baseline ? "5 4" : undefined} dot={{ r: 2.5 }} connectNulls={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--text-primary)" }} />BOI reported CASA</span>
        <span><i style={{ background: meta.baseline ? "var(--series-2)" : "var(--series-1)" }} />{meta.label}{meta.baseline ? " (baseline)" : ""}</span>
      </div>
      <div className="cells">
        <div className="cell">
          <span className="label">Holdout MAPE</span>
          <span className={`value ${meta.baseline ? "baseline" : "model"}`}>{mape[model].toFixed(2)}%</span>
          <span className="note">{meta.label}, {forecasts.length} quarters</span>
        </div>
        <div className="cell">
          <span className="label">Drift baseline</span>
          <span className="value baseline">{mape.Drift.toFixed(2)}%</span>
          <span className="note">a straight line through the history</span>
        </div>
        <div className="cell">
          <span className="label">Worst quarter</span>
          <span className={`value ${worst > 5 ? "miss" : ""}`}>{worst.toFixed(1)}%</span>
          <span className="note">largest single-quarter miss</span>
        </div>
      </div>
      <p className="hint">committed forecasts only — origin {quarter(lastTrain.quarter_end as string)}, nothing refit in the browser</p>
    </div>
  );
}
