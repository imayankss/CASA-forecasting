import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatPct } from "@/lib/formatters";
import type { ForecastRow } from "@/lib/types";

type HeatmapCell = {
  year: string;
  quarter: string;
  growth: number | null;
};

export function GrowthHeatmap({ rows }: { rows: ForecastRow[] }) {
  const cells = buildCells(rows);
  const years = Array.from(new Set(cells.map((cell) => cell.year)));
  const quarters = ["Q1", "Q2", "Q3", "Q4"];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Quarterly Growth Heatmap</CardTitle>
        <CardDescription>Year-over-year movement derived from exported forecast periods.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="grid min-w-[560px] grid-cols-[90px_repeat(4,1fr)] gap-2">
            <div />
            {quarters.map((quarter) => (
              <div key={quarter} className="text-center text-xs font-medium uppercase tracking-wide text-slate-500">
                {quarter}
              </div>
            ))}
            {years.map((year) => (
              <Row key={year} year={year} quarters={quarters} cells={cells.filter((cell) => cell.year === year)} />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Row({ year, quarters, cells }: { year: string; quarters: string[]; cells: HeatmapCell[] }) {
  return (
    <>
      <div className="flex items-center text-sm font-medium text-slate-300">{year}</div>
      {quarters.map((quarter) => {
        const cell = cells.find((item) => item.quarter === quarter);
        return (
          <div
            key={quarter}
            className="rounded-md border border-slate-700/70 px-3 py-4 text-center text-sm font-medium text-white"
            style={{ background: heatColor(cell?.growth ?? null) }}
          >
            {formatPct(cell?.growth ?? null, 1)}
          </div>
        );
      })}
    </>
  );
}

function buildCells(rows: ForecastRow[]): HeatmapCell[] {
  const values = rows
    .map((row) => {
      const parsed = parsePeriod(row.period);
      const value = typeof row.actual === "number" ? row.actual : typeof row.ensemble === "number" ? row.ensemble : null;
      return parsed && value ? { ...parsed, value } : null;
    })
    .filter(Boolean) as Array<{ year: string; quarter: string; value: number }>;

  return values.map((current, index) => {
    const prior = values
      .slice(0, index)
      .reverse()
      .find((candidate) => candidate.quarter === current.quarter);
    const growth = prior ? ((current.value / prior.value) - 1) * 100 : null;
    return { year: current.year, quarter: current.quarter, growth };
  });
}

function parsePeriod(period?: string | null) {
  if (!period) return null;
  const match = period.match(/(\d{4})\s+Q([1-4])/);
  if (!match) return null;
  return { year: match[1], quarter: `Q${match[2]}` };
}

function heatColor(value: number | null) {
  if (value === null) return "rgba(51, 65, 85, 0.45)";
  if (value >= 4) return "rgba(16, 185, 129, 0.45)";
  if (value >= 0) return "rgba(34, 211, 238, 0.28)";
  if (value <= -4) return "rgba(244, 63, 94, 0.42)";
  return "rgba(245, 158, 11, 0.32)";
}

