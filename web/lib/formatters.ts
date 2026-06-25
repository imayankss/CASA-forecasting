export function formatCurrency(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (Math.abs(value) >= 10_000_000) return `Rs ${(value / 10_000_000).toFixed(2)} Cr`;
  if (Math.abs(value) >= 100_000) return `Rs ${(value / 100_000).toFixed(2)} L`;
  return `Rs ${value.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

export function formatPct(value?: number | null, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(digits)}%`;
}

export function formatNumber(value?: number | null, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toLocaleString("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  });
}

export function formatScore(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Math.round(value)}/100`;
}

