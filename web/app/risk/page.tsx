import { AlertTriangle, Radar, ShieldCheck } from "lucide-react";

import { RiskCard } from "@/components/cards/RiskCard";
import { ConfidenceTimeline } from "@/components/charts/ConfidenceTimeline";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getDashboardData } from "@/lib/data";
import { formatCurrency, formatNumber } from "@/lib/formatters";

export default async function RiskPage() {
  const data = await getDashboardData();

  return (
    <DashboardShell manifest={data.manifest}>
      <div className="space-y-6">
        <section>
          <Badge variant="warning">
            <Radar className="mr-1 h-3 w-3" />
            Confidence and Risk
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">Risk Intelligence</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Forecast confidence, anomaly status, and drift monitoring from exported advanced artifacts.
          </p>
        </section>

        <div className="grid gap-5 lg:grid-cols-[0.65fr_1.35fr]">
          <RiskCard kpis={data.kpis} drift={data.drift} />
          <ConfidenceTimeline rows={data.confidenceScores} />
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-emerald-300" />
                Drift Monitoring
              </CardTitle>
              <CardDescription>Distribution, residual, and CUSUM monitoring summary.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2">
              <Metric label="Drift Type" value={data.drift.drift_type} />
              <Metric label="KS Statistic" value={formatNumber(data.drift.test_statistic, 4)} />
              <Metric label="P Value" value={formatNumber(data.drift.p_value, 4)} />
              <Metric
                label="Health Score"
                value={data.drift.model_health_score !== null ? `${data.drift.model_health_score}/100` : "-"}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-amber-300" />
                Risk Interpretation
              </CardTitle>
              <CardDescription>Business-facing reading of monitoring artifacts.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="rounded-md border border-slate-700/70 bg-slate-950/45 p-4 text-sm leading-6 text-slate-300">
                {data.insights.risk_interpretation}
              </p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Anomaly Detection</CardTitle>
            <CardDescription>Flagged anomalous quarters if anomaly artifacts are present.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Period</TableHead>
                  <TableHead>Deposit</TableHead>
                  <TableHead>Method</TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead>Explanation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.anomalies.map((row) => (
                  <TableRow key={`${row.period}-${row.detection_method}`}>
                    <TableCell>{row.period}</TableCell>
                    <TableCell>{formatCurrency(row.deposit_amount)}</TableCell>
                    <TableCell>{row.detection_method}</TableCell>
                    <TableCell>
                      <Badge variant={row.severity === "High" ? "danger" : "warning"}>{row.severity}</Badge>
                    </TableCell>
                    <TableCell>{row.explanation}</TableCell>
                  </TableRow>
                ))}
                {data.anomalies.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="py-8 text-center text-slate-500">
                      No anomaly artifact exported.
                    </TableCell>
                  </TableRow>
                ) : null}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </DashboardShell>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-700/70 bg-slate-950/45 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-lg font-semibold text-white">{value}</p>
    </div>
  );
}
