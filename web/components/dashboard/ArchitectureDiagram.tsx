import { ArrowRight, Database, FileJson, Monitor, Workflow } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const nodes = [
  { label: "Raw CASA + Macro CSV", icon: Database },
  { label: "Python Forecasting Pipeline", icon: Workflow },
  { label: "Static JSON Export", icon: FileJson },
  { label: "Next.js Dashboard", icon: Monitor }
];

export function ArchitectureDiagram() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Product Architecture</CardTitle>
        <CardDescription>No backend API in v1. Python exports static artifacts consumed by Next.js.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
          {nodes.map((node, index) => {
            const Icon = node.icon;
            return (
              <div key={node.label} className="contents">
                <div className="flex min-h-28 items-center gap-3 rounded-lg border border-slate-700/70 bg-slate-950/45 p-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-cyan-300/30 bg-cyan-400/10">
                    <Icon className="h-5 w-5 text-cyan-200" />
                  </div>
                  <p className="text-sm font-medium text-white">{node.label}</p>
                </div>
                {index < nodes.length - 1 ? (
                  <div className="hidden items-center justify-center text-slate-500 md:flex">
                    <ArrowRight className="h-5 w-5" />
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

