import * as React from "react";
import { Slot } from "@radix-ui/react-slot";

import { cn } from "@/lib/utils";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  asChild?: boolean;
  variant?: "primary" | "secondary" | "ghost";
};

export function Button({
  className,
  variant = "primary",
  asChild = false,
  ...props
}: ButtonProps) {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp
      className={cn(
        "inline-flex h-10 items-center justify-center gap-2 rounded-md px-4 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:pointer-events-none disabled:opacity-50",
        variant === "primary" &&
          "border border-cyan-300/35 bg-cyan-400/15 text-cyan-50 hover:bg-cyan-400/25",
        variant === "secondary" &&
          "border border-slate-300/20 bg-slate-300/10 text-slate-100 hover:bg-slate-300/15",
        variant === "ghost" && "text-slate-300 hover:bg-slate-300/10 hover:text-white",
        className
      )}
      {...props}
    />
  );
}

