import type { ReactNode } from "react";

import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";
import type { Manifest } from "@/lib/types";

export function DashboardShell({
  children,
  manifest,
  dataLabel = "Synthetic demo data"
}: {
  children: ReactNode;
  manifest: Manifest;
  dataLabel?: string;
}) {
  return (
    <div className="min-h-screen lg:flex">
      <Sidebar />
      <div className="min-w-0 flex-1">
        <Topbar manifest={manifest} dataLabel={dataLabel} />
        <main className="mx-auto w-full max-w-7xl px-4 py-6 lg:px-8 lg:py-8">{children}</main>
      </div>
    </div>
  );
}

