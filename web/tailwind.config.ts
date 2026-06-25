import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#07111f",
        foreground: "#e5eefc",
        muted: "#94a3b8",
        border: "rgba(148, 163, 184, 0.22)",
        panel: "rgba(15, 23, 42, 0.72)",
        cyan: {
          400: "#22d3ee",
          500: "#06b6d4"
        },
        emerald: {
          400: "#34d399",
          500: "#10b981"
        },
        violet: {
          400: "#a78bfa",
          500: "#8b5cf6"
        }
      },
      boxShadow: {
        glow: "0 0 40px rgba(34, 211, 238, 0.14)",
        panel: "0 24px 70px rgba(0, 0, 0, 0.34)"
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "Inter", "system-ui", "sans-serif"]
      }
    }
  },
  plugins: []
};

export default config;

