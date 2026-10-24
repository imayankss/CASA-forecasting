"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { axis, grid, tooltip } from "@/components/results/chart";
import { readout, type GainsRow } from "@/lib/frontier";

const TIERS = [1, 2, 3, 5, 10];
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

export function CapacityFrontier({ gains }: { gains: GainsRow[] }) {
  const [k, setK] = useState(1);
  const r = readout(gains, k);
  const totalAdds = gains.reduce((s, g) => s + g.adds, 0);
  const curve = [{ x: 0, model: 0, random: 0 }, ...gains.map((g) => ({ x: g.decile * 10, model: g.cumulative_share_of_adds * 100, random: g.decile * 10 }))];
  const show = (v: number | undefined, f: (v: number) => string) => (v === undefined ? "—" : f(v));

  return (
    <div className="panel">
      <div className="controls">
        <span className="label">Call capacity</span>
        <div className="slider">
          <input type="range" min={1} max={10} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="Share of customers contacted" />
          <span className="mono">top {k * 10}% of customers</span>
        </div>
      </div>
      <div className="controls">
        <span className="label">Measured tiers</span>
        <div className="chips" role="group" aria-label="Measured tiers">
          {TIERS.map((t) => (
            <button key={t} type="button" className="chip" aria-pressed={t === k} onClick={() => setK(t)}>
              {t * 10}%
            </button>
          ))}
        </div>
      </div>
      <div className="chart">
        <ResponsiveContainer>
          <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid {...grid} vertical={false} />
            <XAxis dataKey="x" type="number" domain={[0, 100]} ticks={[0, 20, 40, 60, 80, 100]} unit="%" {...axis} />
            <YAxis domain={[0, 100]} unit="%" width={44} {...axis} />
            <Tooltip {...tooltip} formatter={(v, name) => [`${Number(v).toFixed(1)}%`, name === "model" ? "LightGBM ranking" : "random calling"]} labelFormatter={(x) => `top ${x}% contacted`} />
            <ReferenceLine x={k * 10} stroke="var(--series-1)" strokeDasharray="3 3" />
            <Line dataKey="random" stroke="var(--series-2)" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
            <Line dataKey="model" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--series-1)" }} />share of new openers reached, ranked by LightGBM</span>
        <span><i style={{ background: "var(--series-2)" }} />random calling</span>
      </div>
      <div className="cells">
        <div className="cell">
          <span className="label">Openers reached</span>
          <span className="value model">{show(r?.share, (v) => pct(v))}</span>
          <span className="note">{show(r?.adds, (v) => v.toLocaleString())} of {totalAdds.toLocaleString()} new CASA openers</span>
        </div>
        <div className="cell">
          <span className="label">Customers contacted</span>
          <span className="value">{show(r?.customers, (v) => v.toLocaleString())}</span>
          <span className="note">the capacity you chose</span>
        </div>
        <div className="cell">
          <span className="label">Hit rate</span>
          <span className="value">{show(r?.hitRate, (v) => pct(v, 2))}</span>
          <span className="note">share of calls that reach an opener</span>
        </div>
        <div className="cell">
          <span className="label">Lift vs random</span>
          <span className="value model">{show(r?.lift, (v) => `${v.toFixed(2)}×`)}</span>
          <span className="note">hit rate over the base rate</span>
        </div>
        <div className="cell">
          <span className="label">Openers missed</span>
          <span className="value miss">{show(r?.adds, (v) => (totalAdds - v).toLocaleString())}</span>
          <span className="note">never contacted</span>
        </div>
      </div>
      <p className="hint">measured deciles only — no interpolation, nothing recomputed from raw data</p>
    </div>
  );
}
