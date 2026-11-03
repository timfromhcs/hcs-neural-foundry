/**
 * Shipped demo model manifest.
 *
 * The numbers in this file are produced by `scripts/build_web_demo.py`, which trains and evaluates the
 * model with the Python engine on this machine and records the result. The browser displays them as
 * *desktop-engine measurements* — it never pretends to have produced them itself.
 */

import type { DemoManifest } from "./types";

let cache: DemoManifest | null | undefined;

export async function loadDemoManifest(): Promise<DemoManifest | null> {
  if (cache !== undefined) return cache;
  try {
    const response = await fetch("/models/demo-manifest.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = (await response.json()) as DemoManifest;
    if (!payload.id || !payload.onnx?.path || typeof payload.parameterCount !== "number") {
      throw new Error("manifest is missing required fields");
    }
    cache = payload;
    return payload;
  } catch {
    cache = null;
    return null;
  }
}

export function resetDemoCache(): void {
  cache = undefined;
}
