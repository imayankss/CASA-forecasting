import { Award, Clock3, ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatPct, formatScore } from "@/lib/formatters";
import type { LeaderboardRow } from "@/lib/types";

export function ModelCard({ model }: { model?: LeaderboardRow }) {
  if (!model) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Best Model</CardTitle>
          <CardDescription>No leaderboard export found.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Award className="h-4 w-4 text-amber-300" />
              {model.model}
            </CardTitle>
            <CardDescription>Recommended by exported composite ranking.</CardDescription>
          </div>
          <Badge variant="success">Rank #{model.rank}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-3">
          <Metric label="MAPE" value={formatPct(model.mape)} />
          <Metric label="R2" value={model.r2?.toFixed(3) ?? "-"} />
          <Metric label="Score" value={formatScore(model.composite_score)} />
        </div>
        <div className="flex flex-wrap gap-2">
          {model.badges.map((badge) => (
            <Badge key={badge} variant={badge.includes("Stable") ? "violet" : "default"}>
              {badge}
            </Badge>
          ))}
          {model.train_time_s !== null && model.train_time_s !== undefined ? (
            <Badge variant="muted">
              <Clock3 className="mr-1 h-3 w-3" />
              {model.train_time_s.toFixed(2)}s
            </Badge>
          ) : null}
          <Badge variant="success">
            <ShieldCheck className="mr-1 h-3 w-3" />
            Candidate
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-700/70 bg-slate-950/45 p-3">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-white">{value}</p>
    </div>
  );
}

