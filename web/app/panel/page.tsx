import { Building2 } from "lucide-react";

import { DataTable } from "@/components/dashboard/DataTable";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { getDashboardData, readJson } from "@/lib/data";

type Row = Record<string, number | string>;
type Panel = {
  panel_rows: number;
  banks: number;
  origins: string[];
  leaderboard: Row[];
  paired_bootstrap_vs_AutoETS: { a: string; mean_diff: number; ci95: number[] }[];
  stress_2023Q1: Record<string, { MASE_h1_2023Q1: number; MASE_h1_other: number }>;
  aggregate_ape_pct_h1: Record<string, number>;
};
type Boi = { series: Row[]; leaderboard: Row[]; transfer_mape: Row[] };

const fmt = (v: unknown, d = 3) => (typeof v === "number" ? v.toFixed(d) : String(v ?? "-"));

export default async function PanelPage() {
  const data = await getDashboardData();
  const panel = await readJson<Panel | null>("panel.json", null);
  const boi = await readJson<Boi | null>("boi_real.json", null);

  return (
    <DashboardShell manifest={data.manifest} dataLabel="Real public data">
      <div className="space-y-8">
        <section>
          <Badge variant="success">
            <Building2 className="mr-1 h-3 w-3" />
            Real data: FDIC Call Reports and BOI filings
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">Deposit forecasting on real banks</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            {panel
              ? `${panel.banks.toLocaleString()} US banks, ${panel.panel_rows.toLocaleString()} bank-quarters. Forecasts from ${panel.origins.length} origins (${panel.origins[0]} to ${panel.origins[panel.origins.length - 1]}), 1-4 quarters ahead. Target: transaction + savings deposits, the US analogue of CASA.`
              : "Run python run_panel.py and the export step."}
          </p>
        </section>

        {panel && (
          <>
            <DataTable
              columns={["Model", "MASE mean", "MASE median", "Beats seasonal naive", "Bank-origins"]}
              rows={panel.leaderboard.map((r) => [r.model, fmt(r.MASE_mean), fmt(r.MASE_median), fmt(r.share_beats_SeasonalNaive, 2), r.series_origins])}
            />
            <DataTable
              columns={["Model vs AutoETS", "Mean MASE difference", "95% CI (bootstrap over banks)"]}
              rows={panel.paired_bootstrap_vs_AutoETS.map((r) => [r.a, fmt(r.mean_diff), `${fmt(r.ci95[0])} to ${fmt(r.ci95[1])}`])}
            />
            <DataTable
              columns={["Model", "1-step MASE in 2023Q1 outflows", "1-step MASE other quarters", "Sum-of-banks error (%)"]}
              rows={Object.entries(panel.stress_2023Q1).map(([m, v]) => [m, fmt(v.MASE_h1_2023Q1), fmt(v.MASE_h1_other), fmt(panel.aggregate_ape_pct_h1[m], 2)])}
            />
          </>
        )}

        {boi && (
          <section className="space-y-4">
            <h2 className="text-xl font-semibold text-white">Bank of India: real published CASA ({boi.series.length} quarters)</h2>
            <p className="text-sm text-slate-400">Holdout 2024Q1-2026Q2. Each quarter is cited to BOI&apos;s NSE filing in data/real/boi_casa_quarterly.csv.</p>
            <DataTable
              columns={["Model (fit on BOI only)", "MAPE %", "MASE", "R2"]}
              rows={boi.leaderboard.map((r) => [r.model, fmt(r.MAPE, 2), fmt(r.MASE), fmt(r.R2)])}
            />
            <DataTable
              columns={["Transfer comparison", "MAPE %"]}
              rows={boi.transfer_mape.map((r) => [r.model, fmt(r.MAPE_pct, 2)])}
            />
          </section>
        )}
      </div>
    </DashboardShell>
  );
}
