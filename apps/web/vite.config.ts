import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Static build: no server functions, no proxy. The app must work from `dist/` alone.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2022",
  },
  worker: {
    format: "es",
  },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
    setupFiles: ["tests/setup.ts"],
  },
});
