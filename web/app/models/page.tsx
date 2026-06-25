import { Award, BarChart3 } from "lucide-react";

import { ModelCard } from "@/components/cards/ModelCard";
import { MetricRadar } from "@/components/charts/MetricRadar";
import { ModelComparisonChart } from "@/components/charts/ModelComparisonChart";
import { LeaderboardTable } from "@/components/dashboard/LeaderboardTable";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getDashboardData } from "@/lib/data";

export default async function ModelsPage() {
  const data = await getDashboardData();
  const bestModel = data.leaderboard[0];

  return (
    <DashboardShell manifest={data.manifest}>
      <div className="space-y-6">
        <section>
          <Badge variant="violet">
            <BarChart3 className="mr-1 h-3 w-3" />
            Model Lab
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">Leaderboard and Model Comparison</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Compare accuracy, stability, fit, ranking, and production-candidate signals from exported pipeline results.
          </p>
        </section>

        <div className="grid gap-5 xl:grid-cols-[0.8fr_1.2fr]">
          <ModelCard model={bestModel} />
          <MetricRadar model={bestModel} />
        </div>

        <LeaderboardTable rows={data.leaderboard} />

        <Tabs defaultValue="accuracy" className="w-full">
          <TabsList>
            <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            <TabsTrigger value="error">Error</TabsTrigger>
            <TabsTrigger value="fit">Fit</TabsTrigger>
            <TabsTrigger value="score">Score</TabsTrigger>
          </TabsList>
          <TabsContent value="accuracy">
            <ModelComparisonChart rows={data.leaderboard} metric="mape" title="MAPE Ranking" />
          </TabsContent>
          <TabsContent value="error">
            <ModelComparisonChart rows={data.leaderboard} metric="rmse_pct" title="RMSE Comparison" />
          </TabsContent>
          <TabsContent value="fit">
            <ModelComparisonChart rows={data.leaderboard} metric="r2" title="R2 Comparison" />
          </TabsContent>
          <TabsContent value="score">
            <ModelComparisonChart rows={data.leaderboard} metric="composite_score" title="Composite Score Ranking" />
          </TabsContent>
        </Tabs>

        <div className="rounded-lg border border-slate-700/70 bg-slate-950/45 p-5">
          <div className="flex items-start gap-3">
            <Award className="mt-0.5 h-5 w-5 text-amber-300" />
            <div>
              <p className="font-semibold text-white">Recommendation</p>
              <p className="mt-2 text-sm leading-6 text-slate-400">{data.insights.best_model_reason}</p>
            </div>
          </div>
        </div>
      </div>
    </DashboardShell>
  );
}

