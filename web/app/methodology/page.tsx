import { BookOpen, CheckCircle2 } from "lucide-react";

import { ArchitectureDiagram } from "@/components/dashboard/ArchitectureDiagram";
import { MethodologyTimeline } from "@/components/dashboard/MethodologyTimeline";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { modelFamilies } from "@/lib/constants";
import { getDashboardData } from "@/lib/data";

const metrics = [
  ["MAPE", "Business-readable percentage error for comparing model accuracy."],
  ["RMSE %", "Penalizes larger misses and makes error scale comparable."],
  ["R2", "Shows explanatory fit on the exported holdout period."],
  ["Direction Accuracy", "Checks whether the model gets quarter-to-quarter movement right."],
  ["Stability", "Summarizes residual behavior for risk-sensitive planning."]
];

export default async function MethodologyPage() {
  const data = await getDashboardData();

  return (
    <DashboardShell manifest={data.manifest}>
      <div className="space-y-6">
        <section>
          <Badge variant="default">
            <BookOpen className="mr-1 h-3 w-3" />
            Methodology
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">Forecasting Methodology</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            How the CASA forecasting pipeline validates, ranks, monitors, and exports model outputs.
          </p>
        </section>

        <MethodologyTimeline />
        <ArchitectureDiagram />

        <Tabs defaultValue="models">
          <TabsList>
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="readiness">Readiness</TabsTrigger>
          </TabsList>
          <TabsContent value="models">
            <div className="grid gap-4 md:grid-cols-2">
              {modelFamilies.map((family) => {
                const Icon = family.icon;
                return (
                  <Card key={family.title}>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Icon className="h-4 w-4 text-cyan-200" />
                        {family.title}
                      </CardTitle>
                      <CardDescription>{family.models}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm leading-6 text-slate-300">{family.note}</p>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>
          <TabsContent value="metrics">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {metrics.map(([name, body]) => (
                <Card key={name}>
                  <CardHeader>
                    <CardTitle>{name}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-6 text-slate-300">{body}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
          <TabsContent value="readiness">
            <Card>
              <CardHeader>
                <CardTitle>Production-Like Signals</CardTitle>
                <CardDescription>What makes the project more than a notebook.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3 md:grid-cols-2">
                {[
                  "Chronological split and walk-forward validation",
                  "Train-only exogenous scaling",
                  "Model registry and exported leaderboard",
                  "Confidence scoring by forecast period",
                  "Anomaly and drift monitoring hooks",
                  "Static dashboard data contract"
                ].map((item) => (
                  <div key={item} className="flex items-center gap-3 rounded-md border border-slate-700/70 bg-slate-950/45 p-3 text-sm text-slate-300">
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-300" />
                    {item}
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardShell>
  );
}

