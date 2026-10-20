import { Users } from "lucide-react";

import { DataTable } from "@/components/dashboard/DataTable";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { getDashboardData, readJson } from "@/lib/data";

type M = { lift_at_10pct: number; lift_at_10pct_ci95: number[]; lift_at_1pct: number; roc_auc: number; pr_auc: number };
type Prop = {
  base_rate_test: number;
  test_customers: number;
  models: Record<string, M>;
  new_adds_only: { customers: number; adds: number; base_rate: number; share_of_adds_excluded: number; models: Record<string, M> };
  gains_new_adds: Record<string, number>[];
};

const f = (v: number, d = 2) => v.toFixed(d);
const rows = (ms: Record<string, M>) =>
  Object.entries(ms).map(([n, m]) => [n, f(m.lift_at_10pct), `${f(m.lift_at_10pct_ci95[0])} to ${f(m.lift_at_10pct_ci95[1])}`, f(m.lift_at_1pct, 1), f(m.roc_auc, 3), f(m.pr_auc, 3)]);
const cols = ["Model", "Lift @ top 10%", "95% CI", "Lift @ top 1%", "ROC AUC", "PR AUC"];

export default async function PropensityPage() {
  const data = await getDashboardData();
  const p = await readJson<Prop | null>("propensity.json", null);

  return (
    <DashboardShell manifest={data.manifest} dataLabel="Real public data">
      <div className="space-y-8">
        <section>
          <Badge variant="success">
            <Users className="mr-1 h-3 w-3" />
            Public data: Santander Product Recommendation (Kaggle)
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">CASA-acquisition lead scoring</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Who opens a current or savings-type account next month? 13.6M customer-months, 957k customers. Trained on
            Jun-2015 to Feb-2016, tested on May-2016 labels. Kaggle notes the sample contains no real Santander customers.
          </p>
        </section>
        {p && (
          <>
            <h2 className="text-xl font-semibold text-white">
              Genuinely new adds: {p.new_adds_only.customers.toLocaleString()} customers, base rate {f(p.new_adds_only.base_rate * 100)}%
            </h2>
            <p className="text-sm text-slate-400">
              Excludes customers re-opening an account held in the last 5 months ({f(p.new_adds_only.share_of_adds_excluded * 100, 1)}% of all adds).
            </p>
            <DataTable columns={cols} rows={rows(p.new_adds_only.models)} />
            <h2 className="text-xl font-semibold text-white">All adds: {p.test_customers.toLocaleString()} customers, base rate {f(p.base_rate_test * 100)}%</h2>
            <DataTable columns={cols} rows={rows(p.models)} />
            <h2 className="text-xl font-semibold text-white">Gains by decile (new adds, LightGBM)</h2>
            <DataTable
              columns={["Decile", "Customers", "Adds", "Lift", "Cumulative share of adds"]}
              rows={p.gains_new_adds.map((g) => [g.decile, g.customers.toLocaleString(), g.adds, f(g.lift), `${f(g.cumulative_share_of_adds * 100, 1)}%`])}
            />
          </>
        )}
      </div>
    </DashboardShell>
  );
}
