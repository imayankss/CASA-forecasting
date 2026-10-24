"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { axis, grid, quarter, tooltip } from "@/components/results/chart";

type Row = Record<string, number | string>;

const MODELS: { key: string; label: string; color: string; dash?: string }[] = [
  { key: "Naive", label: "No change", color: "var(--series-2)" },
  { key: "GlobalLGBM", label: "Global LightGBM", color: "var(--series-1)" },
  { key: "Chronos2", label: "Chronos-2", color: "var(--series-1)", dash: "5 4" },
  { key: "AutoETS", label: "AutoETS", color: "var(--text-secondary)" },
  { key: "AutoTheta", label: "AutoTheta", color: "var(--text-secondary)", dash: "5 4" },
  { key: "SeasonalNaive", label: "Seasonal naive", color: "var(--axis)", dash: "2 3" }
];

export function OriginsChart({ rows }: { rows: Row[] }) {
  const [on, setOn] = useState<string[]>(["Naive", "GlobalLGBM", "AutoETS"]);
  const data = rows.map((r) => ({ ...r, q: quarter(String(r.origin)) }));
  const flip = (k: string) => setOn((s) => (s.includes(k) ? s.filter((x) => x !== k) : [...s, k]));

  return (
    <div className="panel">
      <div className="controls">
        <span className="label">Show</span>
        <div className="chips" role="group" aria-label="Choose models">
          {MODELS.map((m) => (
            <button key={m.key} type="button" className="chip" aria-pressed={on.includes(m.key)} onClick={() => flip(m.key)}>
              <i className="swatch" style={{ borderTopColor: m.color, borderTopStyle: m.dash ? "dashed" : "solid" }} />
              {m.label}
            </button>
          ))}
        </div>
      </div>
      <div className="chart">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid {...grid} vertical={false} />
            {/* Origins whose 1-4 quarter windows include the 2023Q1 deposit outflows */}
            <ReferenceArea x1="2022Q1" x2="2022Q4" fill="var(--surface-3)" fillOpacity={0.8} label={{ value: "windows covering 2023Q1 outflows", fill: "var(--text-muted)", fontSize: 11, position: "insideTop" }} />
            <XAxis dataKey="q" {...axis} />
            <YAxis {...axis} domain={[0.75, 3]} ticks={[1, 1.5, 2, 2.5, 3]} width={36} />
            <Tooltip {...tooltip} formatter={(v, name) => [Number(v).toFixed(3), MODELS.find((m) => m.key === name)?.label ?? name]} />
            {MODELS.filter((m) => on.includes(m.key)).map((m) => (
              <Line key={m.key} dataKey={m.key} stroke={m.color} strokeDasharray={m.dash} strokeWidth={m.key === "Naive" || m.key === "GlobalLGBM" ? 2 : 1.5} dot={{ r: 2.5 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="hint">mean MASE per forecast origin, 1–4 quarters ahead, ~4,000 banks each — lower is better</p>
    </div>
  );
}
