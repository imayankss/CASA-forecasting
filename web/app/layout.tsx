import type { Metadata } from "next";

import "@/app/globals.css";

export const metadata: Metadata = {
  title: "CASA Intelligence Dashboard",
  description: "Banking deposit forecasting, model comparison, risk monitoring, and confidence intelligence."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}

