import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatPct, formatScore } from "@/lib/formatters";
import type { LeaderboardRow } from "@/lib/types";

export function LeaderboardTable({ rows }: { rows: LeaderboardRow[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Leaderboard</CardTitle>
        <CardDescription>Ranked by exported composite score from the Python pipeline.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table className="min-w-[860px]">
            <TableHeader>
              <TableRow>
                <TableHead>Rank</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>MAPE</TableHead>
                <TableHead>RMSE %</TableHead>
                <TableHead>R2</TableHead>
                <TableHead>Direction</TableHead>
                <TableHead>Stability</TableHead>
                <TableHead>Score</TableHead>
                <TableHead>Flags</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.model}>
                  <TableCell className="font-semibold text-white">#{row.rank}</TableCell>
                  <TableCell className="font-medium text-white">{row.model}</TableCell>
                  <TableCell>{formatPct(row.mape)}</TableCell>
                  <TableCell>{formatPct(row.rmse_pct)}</TableCell>
                  <TableCell>{row.r2?.toFixed(3) ?? "-"}</TableCell>
                  <TableCell>{formatPct(row.direction_accuracy, 1)}</TableCell>
                  <TableCell>{formatScore(row.stability)}</TableCell>
                  <TableCell>{formatScore(row.composite_score)}</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1.5">
                      {row.recommended ? <Badge variant="success">Recommended</Badge> : null}
                      {row.badges.slice(0, 2).map((badge) => (
                        <Badge key={badge} variant="muted">
                          {badge}
                        </Badge>
                      ))}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {rows.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={9} className="py-8 text-center text-slate-500">
                    No leaderboard export found.
                  </TableCell>
                </TableRow>
              ) : null}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

