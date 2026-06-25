"use client";

import Link from "next/link";
import { CalendarClock, Database, Landmark } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { navItems } from "@/lib/constants";
import type { Manifest } from "@/lib/types";

export function Topbar({ manifest }: { manifest: Manifest }) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/55 px-4 py-3 backdrop-blur-xl lg:px-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 lg:hidden">
          <div className="flex h-9 w-9 items-center justify-center rounded-md border border-cyan-300/25 bg-cyan-400/10">
            <Landmark className="h-4 w-4 text-cyan-200" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white">CASA Intelligence</p>
            <p className="text-xs text-slate-500">Forecasting dashboard</p>
          </div>
        </div>

        <div className="hidden items-center gap-2 lg:flex">
          <Badge variant="success">
            <Database className="mr-1 h-3 w-3" />
            {manifest.model_count} models
          </Badge>
          <Badge variant="muted">
            <CalendarClock className="mr-1 h-3 w-3" />
            {manifest.data_start ?? "-"} to {manifest.data_end ?? "-"}
          </Badge>
        </div>

        <nav className="flex max-w-full gap-1 overflow-x-auto lg:hidden">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-md border border-slate-700/60 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-300"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

