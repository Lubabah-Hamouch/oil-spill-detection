import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // Fail early: the API URL is baked into the bundle, so a build without it ships a UI that can't reach
  // the API. loadEnv reads .env files for this mode plus VITE_* variables from the environment.
  // "." is the working directory, where Vite looks for .env files by default.
  const apiUrl = loadEnv(mode, ".", "VITE_").VITE_API_URL?.trim();
  if (!apiUrl) {
    throw new Error("VITE_API_URL must be set at build time (e.g. http://localhost:8000 for local dev).");
  }

  return {
    plugins: [react()],
    server: { port: 5173 },
  };
});
