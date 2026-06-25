import { SlidersHorizontal } from "lucide-react";

import { ScenarioSimulator } from "@/components/dashboard/ScenarioSimulator";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getDashboardData } from "@/lib/data";

export default async function ScenarioPage() {
  const data = await getDashboardData();

  return (
    <DashboardShell manifest={data.manifest}>
      <div className="space-y-6">
        <section>
          <Badge variant="violet">
            <SlidersHorizontal className="mr-1 h-3 w-3" />
            Scenario Simulator
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">Macro Scenario Simulator</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Illustrative optimistic, base, and stress views from macro sensitivity assumptions.
          </p>
        </section>

        <ScenarioSimulator kpis={data.kpis} />

        <Card>
          <CardHeader>
            <CardTitle>Interpretation</CardTitle>
            <CardDescription>First-version simulator boundary.</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-6 text-slate-300">
              This simulator is frontend-only and rule-based. It is useful for portfolio storytelling, while the
              production path is to pass future macro assumptions into SARIMAX exogenous inputs and export those
              scenario forecasts from Python.
            </p>
          </CardContent>
        </Card>
      </div>
    </DashboardShell>
  );
}

