import { Banknote, Gauge, LineChart, ShieldCheck, Target, TrendingUp } from "lucide-react";

import { ModelCard } from "@/components/cards/ModelCard";
import { RiskCard } from "@/components/cards/RiskCard";
import { InsightCard } from "@/components/cards/InsightCard";
import { KpiCard } from "@/components/cards/KpiCard";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { ArchitectureDiagram } from "@/components/dashboard/ArchitectureDiagram";
import { MethodologyTimeline } from "@/components/dashboard/MethodologyTimeline";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { FadeIn } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { getDashboardData } from "@/lib/data";
import { formatPct } from "@/lib/formatters";

export default async function OverviewPage() {
  const data = await getDashboardData();
  const bestModel = data.leaderboard[0];

  return (
    <DashboardShell manifest={data.manifest}>
      <div className="space-y-6">
        <FadeIn>
          <section className="grid gap-5 lg:grid-cols-[1.25fr_0.75fr]">
            <div>
              <Badge variant="default">Banking Forecasting Intelligence</Badge>
              <h1 className="mt-4 max-w-4xl text-4xl font-semibold tracking-normal text-white lg:text-5xl">
                CASA Intelligence Dashboard
              </h1>
              <p className="mt-4 max-w-3xl text-base leading-7 text-slate-400">
                Banking deposit forecasting, model comparison, risk monitoring, and confidence intelligence.
              </p>
            </div>
            <div className="rounded-lg border border-slate-700/70 bg-slate-950/45 p-5">
              <p className="text-xs uppercase tracking-wide text-slate-500">Export Snapshot</p>
              <p className="mt-2 text-2xl font-semibold text-white">{data.manifest.data_end ?? "Pending export"}</p>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                {data.manifest.row_count} quarterly rows, {data.manifest.model_count} exported models.
              </p>
            </div>
          </section>
        </FadeIn>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <FadeIn delay={0.04}>
            <KpiCard
              title="Current CASA Deposit"
              value={data.kpis.current_deposit}
              detail={data.kpis.latest_period ?? undefined}
              icon={Banknote}
              format="currency"
            />
          </FadeIn>
          <FadeIn delay={0.08}>
            <KpiCard
              title="Latest Forecast"
              value={data.kpis.forecast_next_period}
              detail={data.kpis.forecast_period ?? undefined}
              trend={formatPct(data.kpis.forecast_growth_pct)}
              variant={data.kpis.forecast_growth_pct !== null && data.kpis.forecast_growth_pct < 0 ? "warning" : "success"}
              icon={TrendingUp}
              format="currency"
            />
          </FadeIn>
          <FadeIn delay={0.12}>
            <KpiCard title="Best Model" textValue={data.kpis.best_model} trend={formatPct(data.kpis.best_mape)} icon={Target} />
          </FadeIn>
          <FadeIn delay={0.16}>
            <KpiCard
              title="Confidence Score"
              value={data.kpis.confidence_score}
              detail={data.kpis.confidence_label ?? undefined}
              icon={Gauge}
              format="score"
            />
          </FadeIn>
          <FadeIn delay={0.2}>
            <KpiCard title="Risk Status" textValue={data.kpis.risk_status} icon={ShieldCheck} variant="success" />
          </FadeIn>
          <FadeIn delay={0.24}>
            <KpiCard title="Model Count" value={data.manifest.model_count} detail="Exported leaderboard" icon={LineChart} />
          </FadeIn>
        </div>

        <div className="grid gap-5 xl:grid-cols-[1.35fr_0.65fr]">
          <FadeIn delay={0.1}>
            <ForecastChart rows={data.forecast} compact />
          </FadeIn>
          <div className="grid gap-5">
            <FadeIn delay={0.14}>
              <ModelCard model={bestModel} />
            </FadeIn>
            <FadeIn delay={0.18}>
              <RiskCard kpis={data.kpis} drift={data.drift} />
            </FadeIn>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <InsightCard
            title="Executive Insight"
            description={data.insights.executive_summary}
            items={data.insights.banking_insights}
            icon={Banknote}
          />
          <InsightCard
            title="Model Recommendations"
            description={data.insights.best_model_reason}
            items={data.insights.model_recommendations}
            icon={Target}
          />
        </div>

        <MethodologyTimeline />
        <ArchitectureDiagram />
      </div>
    </DashboardShell>
  );
}
