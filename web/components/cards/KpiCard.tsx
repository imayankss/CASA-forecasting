import type { LucideIcon } from "lucide-react";

import { AnimatedCounter } from "@/components/motion/AnimatedCounter";
import { Card, CardContent } from "@/components/ui/card";
import { Badge, type BadgeVariant } from "@/components/ui/badge";

type KpiCardProps = {
  title: string;
  value?: number | null;
  textValue?: string | null;
  detail?: string;
  trend?: string;
  icon: LucideIcon;
  format?: "currency" | "percent" | "score" | "number";
  variant?: BadgeVariant;
};

export function KpiCard({
  title,
  value,
  textValue,
  detail,
  trend,
  icon: Icon,
  format,
  variant = "default"
}: KpiCardProps) {
  return (
    <Card className="min-h-36">
      <CardContent className="flex h-full flex-col justify-between p-5">
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{title}</p>
          <div className="flex h-9 w-9 items-center justify-center rounded-md border border-slate-600/45 bg-slate-900/60">
            <Icon className="h-4 w-4 text-cyan-200" />
          </div>
        </div>
        <div>
          <p className="mt-4 break-words text-2xl font-semibold text-white">
            {textValue ?? <AnimatedCounter value={value} format={format} />}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {trend ? <Badge variant={variant}>{trend}</Badge> : null}
            {detail ? <span className="text-xs text-slate-500">{detail}</span> : null}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
