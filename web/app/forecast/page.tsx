import { Calculator, LineChart } from "lucide-react";

import { ForecastChart } from "@/components/charts/ForecastChart";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getDashboardData } from "@/lib/data";
import { formatCurrency, formatPct } from "@/lib/formatters";

export default async function ForecastPage() {
  const data = await getDashboardData();
  const bestModel = data.kpis.best_model ?? "ensemble";
  const errorRows = data.forecast.map((row) => {
    const forecast = typeof row.ensemble === "number" ? row.ensemble : typeof row[bestModel] === "number" ? row[bestModel] as number : null;
    const errorPct = row.actual && forecast ? ((forecast - row.actual) / row.actual) * 100 : null;
    return { ...row, forecast, errorPct };
  });

  return (
    <DashboardShell manifest={data.manifest}>
      <div className="space-y-6">
        <section>
          <Badge variant="default">
            <LineChart className="mr-1 h-3 w-3" />
            Forecast Explorer
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold text-white">Forecast vs Actual View</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Historical holdout actuals, model forecasts, and exported interval bands.
          </p>
        </section>

        <ForecastChart rows={data.forecast} />

        <div className="grid gap-5 lg:grid-cols-[1fr_0.45fr]">
          <Card>
            <CardHeader>
              <CardTitle>Forecast Table</CardTitle>
              <CardDescription>Best available forecast uses ensemble when exported, otherwise the best model.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table className="min-w-[720px]">
                  <TableHeader>
                    <TableRow>
                      <TableHead>Period</TableHead>
                      <TableHead>Actual</TableHead>
                      <TableHead>Forecast</TableHead>
                      <TableHead>Lower CI</TableHead>
                      <TableHead>Upper CI</TableHead>
                      <TableHead>Error</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {errorRows.map((row) => (
                      <TableRow key={row.period}>
                        <TableCell>{row.period}</TableCell>
                        <TableCell>{formatCurrency(row.actual)}</TableCell>
                        <TableCell>{formatCurrency(row.forecast)}</TableCell>
                        <TableCell>{formatCurrency(row.lower_ci)}</TableCell>
                        <TableCell>{formatCurrency(row.upper_ci)}</TableCell>
                        <TableCell>{formatPct(row.errorPct)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calculator className="h-4 w-4 text-cyan-200" />
                Error Summary
              </CardTitle>
              <CardDescription>Signed error for the selected exported forecast basis.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {errorRows.slice(-4).map((row) => (
                <div key={row.period} className="rounded-md border border-slate-700/70 bg-slate-950/45 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-medium text-slate-200">{row.period}</span>
                    <Badge variant={row.errorPct !== null && Math.abs(row.errorPct) <= 3 ? "success" : "warning"}>
                      {formatPct(row.errorPct)}
                    </Badge>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardShell>
  );
}

