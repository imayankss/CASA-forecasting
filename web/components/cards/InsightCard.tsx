import type { LucideIcon } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function InsightCard({
  title,
  description,
  items,
  icon: Icon
}: {
  title: string;
  description?: string;
  items: string[];
  icon: LucideIcon;
}) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-cyan-200" />
          {title}
        </CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      <CardContent>
        <ul className="space-y-3 text-sm leading-6 text-slate-300">
          {items.length > 0 ? (
            items.map((item) => (
              <li key={item} className="border-l border-cyan-300/30 pl-3">
                {item}
              </li>
            ))
          ) : (
            <li className="text-slate-500">No exported insight available.</li>
          )}
        </ul>
      </CardContent>
    </Card>
  );
}

