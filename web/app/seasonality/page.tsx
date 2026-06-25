import { Layers3, TrendingDown, TrendingUp } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { ForecastChart } from "@/components/charts/ForecastChart";
import { GrowthHeatmap } from "@/components/charts/GrowthHeatmap";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getDashboardData } from "@/lib/data";
import { formatCurrency } from "@/lib/formatters";

export default async function SeasonalityPage() {
  const data = await getDashboardData();
  const observed = data.forecast
    .filter((row) => typeof row.actual === "number")
    .map((row) => ({ period: row.period, value: row.actual as number }))
    .sort((a, b) => b.value - a.value);
  const best = observed[0];
  const worst = observed[observed.length - 1];

  return (
    <DashboardShell manifest={data.manifest}>
      <div className="space-y-6">
        <section>
          <Badge variant="default">
            <Layers3 className="mr-1 h-3 w-3" />
            Seasonality
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">Seasonality and Growth Analytics</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Quarter-level movement, holdout trend behavior, and period extremes.
          </p>
        </section>

        <div className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
          <GrowthHeatmap rows={data.forecast} />
          <div className="grid gap-5">
            <SummaryCard icon={TrendingUp} title="Strongest Exported Quarter" period={best?.period} value={best?.value} />
            <SummaryCard icon={TrendingDown} title="Weakest Exported Quarter" period={worst?.period} value={worst?.value} />
          </div>
        </div>

        <ForecastChart rows={data.forecast} compact />

        <Card>
          <CardHeader>
            <CardTitle>Seasonality Read</CardTitle>
            <CardDescription>Banking deposits often move with quarter-end balance sheet behavior and macro liquidity.</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-6 text-slate-300">
              The dashboard keeps the visual layer descriptive: it shows exported historical holdout movement and leaves
              model training, decomposition, and diagnostics in the Python pipeline.
            </p>
          </CardContent>
        </Card>
      </div>
    </DashboardShell>
  );
}

function SummaryCard({
  icon: Icon,
  title,
  period,
  value
}: {
  icon: LucideIcon;
  title: string;
  period?: string | null;
  value?: number | null;
}) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md border border-cyan-300/30 bg-cyan-400/10">
            <Icon className="h-5 w-5 text-cyan-200" />
          </div>
          <div>
            <p className="text-sm text-slate-400">{title}</p>
            <p className="mt-1 text-xl font-semibold text-white">{period ?? "-"}</p>
          </div>
        </div>
        <p className="mt-4 text-2xl font-semibold text-white">{formatCurrency(value)}</p>
      </CardContent>
    </Card>
  );
}
