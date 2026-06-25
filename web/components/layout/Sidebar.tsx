"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Landmark } from "lucide-react";

import { navItems } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-72 shrink-0 border-r border-slate-800/80 bg-slate-950/35 px-4 py-5 backdrop-blur-xl lg:block">
      <div className="mb-8 flex items-center gap-3 px-2">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-cyan-300/30 bg-cyan-400/10">
          <Landmark className="h-5 w-5 text-cyan-200" />
        </div>
        <div>
          <p className="text-sm font-semibold text-white">CASA Intelligence</p>
          <p className="text-xs text-slate-500">Deposit Forecasting</p>
        </div>
      </div>

      <nav className="space-y-1">
        {navItems.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm text-slate-400 transition-colors",
                active
                  ? "border border-cyan-300/25 bg-cyan-400/12 text-cyan-50"
                  : "hover:bg-slate-800/50 hover:text-slate-100"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

