import type { Config } from "tailwindcss";

// "Monitoring console" theme. Colors extend (not replace) Tailwind's defaults.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0B0F14", // page background (named "canvas" so text-base stays the font-size utility)
        surface: "#11161D", // cards / panels
        elevated: "#1A212B", // hover / raised surfaces
        line: "#232B36", // borders
        ink: {
          DEFAULT: "#E6EAEE", // primary text
          muted: "#8B96A3", // secondary text
        },
        accent: "#2DD4BF", // brand: clean water / detection active
        spill: "#F59E0B", // spill detected (amber, deliberately not red)
        clear: "#4ADE80", // no spill detected
      },
      fontFamily: {
        sans: ['"Inter"', "ui-sans-serif", "system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
        mono: ["ui-monospace", '"SF Mono"', "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 12px 32px -16px rgba(0,0,0,0.6)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        rise: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.6s linear infinite",
        rise: "rise 0.35s ease-out both",
      },
    },
  },
  plugins: [],
} satisfies Config;
