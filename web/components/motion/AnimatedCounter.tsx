"use client";

import { animate, useMotionValue, useTransform, motion } from "framer-motion";
import { useEffect } from "react";

type AnimatedCounterProps = {
  value?: number | null;
  format?: "currency" | "percent" | "score" | "number";
};

export function AnimatedCounter({ value, format = "number" }: AnimatedCounterProps) {
  const motionValue = useMotionValue(0);
  const rounded = useTransform(motionValue, (latest) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "-";
    if (format === "currency") return `Rs ${(latest / 10_000_000).toFixed(2)} Cr`;
    if (format === "percent") return `${latest.toFixed(2)}%`;
    if (format === "score") return `${Math.round(latest)}/100`;
    return Math.round(latest).toLocaleString("en-IN");
  });

  useEffect(() => {
    if (value === null || value === undefined || Number.isNaN(value)) return;
    const controls = animate(motionValue, value, { duration: 0.9, ease: "easeOut" });
    return controls.stop;
  }, [motionValue, value]);

  return <motion.span>{rounded}</motion.span>;
}
