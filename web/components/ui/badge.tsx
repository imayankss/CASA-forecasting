import * as React from "react";

import { cn } from "@/lib/utils";

const variants = {
  default: "border-cyan-400/35 bg-cyan-400/10 text-cyan-100",
  success: "border-emerald-400/35 bg-emerald-400/10 text-emerald-100",
  warning: "border-amber-400/35 bg-amber-400/10 text-amber-100",
  danger: "border-rose-400/35 bg-rose-400/10 text-rose-100",
  muted: "border-slate-400/25 bg-slate-400/10 text-slate-300",
  violet: "border-violet-400/35 bg-violet-400/10 text-violet-100"
};

export type BadgeVariant = keyof typeof variants;

export function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: BadgeVariant }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-1 text-xs font-medium",
        variants[variant],
        className
      )}
      {...props}
    />
  );
}

