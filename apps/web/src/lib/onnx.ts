/**
 * ONNX Runtime Web integration.
 *
 * Rules enforced here:
 *  1. A provider is only reported as active after a real inference ran on it.
 *  2. WebGPU is attempted, then WASM; the fallback reason is returned, not swallowed.
 *  3. The runtime and its WASM binaries are served from this origin, so the page does not
 *     silently depend on a CDN.
 *  4. We import the native WebGPU build. ONNX Runtime deprecated WebGL and JSEP in v1.29.0
 *     (microsoft/onnxruntime#29716); the bare `onnxruntime-web` entry point still resolves to the
 *     JSEP backend, which is on its way out. `onnxruntime-web/webgpu` ships the native EP and
 *     still contains the WASM fallback, so one import serves both paths.
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
  adapter?: string;
}

export interface InferenceResult {
  provider: ActiveProvider;
  logits: number[];
  dims: number[];
  ms: number;
  tokenCount: number;
}

type OrtModule = typeof import("onnxruntime-web/webgpu");

const MODEL_PATH = "/models/demo-char-lm.onnx";

/**
 * A provider attempt must never hang the UI. Switching providers in one page can leave the ONNX
 * Runtime Web backend half-initialised (observed: WASM → WebGPU never resolves), so every attempt is
 * bounded and the timeout is reported like any other failure rather than spinning forever.
 */
const PROVIDER_TIMEOUT_MS = 20_000;

function withTimeout<T>(work: Promise<T>, label: string, ms = PROVIDER_TIMEOUT_MS): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} did not settle within ${ms} ms`));
    }, ms);
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

let ortPromise: Promise<OrtModule> | null = null;
let sessionPromise: Promise<{
  session: import("onnxruntime-web/webgpu").InferenceSession;
  provider: ActiveProvider;
}> | null = null;
let cachedStatus: BackendStatus | null = null;

async function loadOrt(): Promise<OrtModule> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/webgpu").then((module) => {
      // Vite rewrites the runtime's own wasm URL to a bundled asset, so no CDN is involved.
      module.env.wasm.numThreads = 1;
      module.env.logLevel = "fatal"; // ORT logs provider warnings at error level; keep the console clean
      return module;
    });
  }
  return ortPromise;
}

function webgpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && Boolean(navigator.gpu);
}

/**
 * `navigator.gpu` existing is not the same as an adapter being usable — a headless or
 * software-blocklisted environment exposes the API and returns no adapter. Report what we actually
 * got, never the API's mere presence.
 */
async function describeAdapter(): Promise<string | null> {
  if (!webgpuAvailable()) return null;
  try {
    // `GPUAdapter.info` is newer than some bundled DOM typings, so read it defensively.
    const gpu = navigator.gpu as unknown as {
      requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string } } | null>;
    };
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    const vendor = adapter.info?.vendor ?? "unknown vendor";
    const architecture = adapter.info?.architecture ?? "unknown architecture";
    return `${vendor} / ${architecture}`;
  } catch {
    return null;
  }
}

async function createSession(preference: ProviderPreference) {
  const ort = await loadOrt();
  const attempts: ActiveProvider[] =
    preference === "wasm" ? ["wasm"] : preference === "webgpu" ? ["webgpu"] : ["webgpu", "wasm"];
  const failures: string[] = [];
  const adapter = await describeAdapter();
  for (const provider of attempts) {
    if (provider === "webgpu" && !webgpuAvailable()) {
      failures.push("navigator.gpu is unavailable");
      continue;
    }
    if (provider === "webgpu" && !adapter) {
      failures.push("navigator.gpu exists but requestAdapter() returned no adapter");
      continue;
    }
    try {
      const started = performance.now();
      const session = await withTimeout(
        ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: [provider],
          graphOptimizationLevel: "all",
          // ORT emits "some nodes were not assigned to the preferred EP" at error level even though
          // it is advisory. 3 = error: real failures still reach the console, advisory noise does not.
          logSeverityLevel: 3,
        }),
        `${provider} session creation`,
      );
      // Prove the provider works with a real inference before advertising it.
      await withTimeout(runOnce(ort, session), `${provider} warm-up inference`);
      return {
        session,
        provider,
        adapter,
        loadMs: performance.now() - started,
        failures,
      };
    } catch (error) {
      failures.push(`${provider}: ${(error as Error).message}`);
    }
  }
  throw new Error(`no ONNX provider could run the model (${failures.join("; ")})`);
}

async function runOnce(ort: OrtModule, session: import("onnxruntime-web/webgpu").InferenceSession) {
  const tokens = BigInt64Array.from([1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n]);
  const input = new ort.Tensor("int64", tokens, [1, 8]);
  const feeds: Record<string, import("onnxruntime-web/webgpu").Tensor> = { token_ids: input };
  await session.run(feeds);
}

async function ensureSession(preference: ProviderPreference) {
  // A session is bound to the provider it was created with. Changing the preference must rebuild it,
  // otherwise the Settings control would appear to work while the old provider keeps running.
  if (sessionPromise && cachedStatus && cachedStatus.requested !== preference) {
    resetSession();
  }
  if (!sessionPromise) {
    sessionPromise = createSession(preference).then(({ session, provider, loadMs, failures, adapter }) => {
      const adapterNote = adapter ? ` on ${adapter}` : "";
      cachedStatus = {
        provider,
        requested: preference,
        webgpuSupported: webgpuAvailable(),
        detail:
          provider === "webgpu"
            ? `WebGPU execution provider (native EP)${adapterNote}`
            : "WASM execution provider (CPU)",
        loadMs,
        fallbackReason: failures.length ? failures.join("; ") : undefined,
        ...(adapter ? { adapter } : {}),
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
