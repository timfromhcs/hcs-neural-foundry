/**
 * ONNX Runtime Web integration.
 *
 * Rules enforced here:
 *  1. A provider is only reported as active after a real inference ran on it.
 *  2. WebGPU is attempted, then WASM; the fallback reason is returned, not swallowed.
 *  3. The runtime and its WASM binaries are served from this origin (see scripts/copy-ort.mjs),
 *     so the page does not silently depend on a CDN.
 */

export type ProviderPreference = "auto" | "wasm" | "webgpu";
export type ActiveProvider = "webgpu" | "wasm";

export interface BackendStatus {
  provider: ActiveProvider;
  requested: ProviderPreference;
  webgpuSupported: boolean;
  detail: string;
  loadMs: number;
  fallbackReason?: string;
}

export interface InferenceResult {
  provider: ActiveProvider;
  logits: number[];
  dims: number[];
  ms: number;
  tokenCount: number;
}

type OrtModule = typeof import("onnxruntime-web");

const MODEL_PATH = "/models/demo-char-lm.onnx";

let ortPromise: Promise<OrtModule> | null = null;
let sessionPromise: Promise<{ session: import("onnxruntime-web").InferenceSession; provider: ActiveProvider }> | null =
  null;
let cachedStatus: BackendStatus | null = null;

async function loadOrt(): Promise<OrtModule> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web").then((module) => {
      // Vite rewrites the runtime's own wasm URL to a bundled asset, so no CDN is involved.
      module.env.wasm.numThreads = 1;
      module.env.logLevel = "error";
      return module;
    });
  }
  return ortPromise;
}

function webgpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && Boolean(navigator.gpu);
}

async function createSession(preference: ProviderPreference) {
  const ort = await loadOrt();
  const attempts: ActiveProvider[] = preference === "wasm" ? ["wasm"] : preference === "webgpu" ? ["webgpu"] : ["webgpu", "wasm"];
  const failures: string[] = [];
  for (const provider of attempts) {
    if (provider === "webgpu" && !webgpuAvailable()) {
      failures.push("navigator.gpu is unavailable");
      continue;
    }
    try {
      const started = performance.now();
      const session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });
      // Prove the provider works with a real inference before advertising it.
      await runOnce(ort, session);
      return {
        session,
        provider,
        loadMs: performance.now() - started,
        failures,
      };
    } catch (error) {
      failures.push(`${provider}: ${(error as Error).message}`);
    }
  }
  throw new Error(`no ONNX provider could run the model (${failures.join("; ")})`);
}

async function runOnce(ort: OrtModule, session: import("onnxruntime-web").InferenceSession) {
  const tokens = BigInt64Array.from([1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n]);
  const input = new ort.Tensor("int64", tokens, [1, 8]);
  const feeds: Record<string, import("onnxruntime-web").Tensor> = { token_ids: input };
  await session.run(feeds);
}

async function ensureSession(preference: ProviderPreference) {
  if (!sessionPromise) {
    sessionPromise = createSession(preference).then(({ session, provider, loadMs, failures }) => {
      cachedStatus = {
        provider,
        requested: preference,
        webgpuSupported: webgpuAvailable(),
        detail: provider === "webgpu" ? "WebGPU execution provider" : "WASM execution provider (CPU)",
        loadMs,
        fallbackReason: failures.length ? failures.join("; ") : undefined,
      };
      return { session, provider };
    });
    sessionPromise.catch(() => {
      sessionPromise = null;
    });
  }
  return sessionPromise;
}

export async function backendStatus(preference: ProviderPreference = "auto"): Promise<BackendStatus> {
  if (cachedStatus && cachedStatus.requested === preference) return cachedStatus;
  await ensureSession(preference);
  return cachedStatus!;
}

export async function runDemoModel(tokenIds: number[], preference: ProviderPreference = "auto"): Promise<InferenceResult> {
  const ort = await loadOrt();
  const { session, provider } = await ensureSession(preference);
  const tokens = BigInt64Array.from(tokenIds.map((value) => BigInt(value)));
  const input = new ort.Tensor("int64", tokens, [1, tokenIds.length]);
  const started = performance.now();
  const output = await session.run({ token_ids: input });
  const elapsed = performance.now() - started;
  const first = output["logits"] ?? Object.values(output)[0];
  if (!first) throw new Error("model returned no output tensor");
  const data = first.data as Float32Array;
  return {
    provider,
    logits: Array.from(data),
    dims: first.dims as number[],
    ms: elapsed,
    tokenCount: tokenIds.length,
  };
}

export function resetSession(): void {
  sessionPromise = null;
  cachedStatus = null;
}

/** Deterministic prompt encoder for the demo model: characters map into the model's vocabulary. */
export function encodePrompt(text: string, vocabSize: number): number[] {
  const ids = Array.from(text).map((character) => {
    const code = character.codePointAt(0) ?? 63;
    return code % vocabSize;
  });
  return ids.length ? ids.slice(-16) : [1];
}

export function argmax(values: number[]): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i]! > bestValue) {
      bestValue = values[i]!;
      best = i;
    }
  }
  return best;
}
