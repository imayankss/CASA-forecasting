import { AlertTriangle, CheckCircle2, Gauge } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatScore } from "@/lib/formatters";
import type { DriftPayload, Kpis } from "@/lib/types";

export function RiskCard({ kpis, drift }: { kpis: Kpis; drift: DriftPayload }) {
  const elevated = kpis.risk_status === "Elevated" || drift.drift_detected;
  const Icon = elevated ? AlertTriangle : CheckCircle2;

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-4 w-4 text-cyan-200" />
              Risk Status
            </CardTitle>
            <CardDescription>Confidence and monitoring view from exported artifacts.</CardDescription>
          </div>
          <Badge variant={elevated ? "danger" : kpis.risk_status === "Watch" ? "warning" : "success"}>
            <Icon className="mr-1 h-3 w-3" />
            {kpis.risk_status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-3xl font-semibold text-white">{formatScore(kpis.confidence_score)}</p>
          <p className="mt-1 text-sm text-slate-400">{kpis.confidence_label ?? "Confidence unavailable"}</p>
        </div>
        <div className="rounded-md border border-slate-700/70 bg-slate-950/45 p-3 text-sm leading-6 text-slate-300">
          {drift.interpretation}
        </div>
      </CardContent>
    </Card>
  );
}

