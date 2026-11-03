/**
 * The contract under test: a provider is reported only after a real inference ran on it, and a
 * fallback is always explained. These tests exist because the WebGPU path moved from the deprecated
 * JSEP build to the native WebGPU EP, and the claim must not outlive the evidence.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({
  created: [] as string[],
  webgpu: "ok" as "ok" | "throw" | "hang",
  wasm: "ok" as "ok" | "throw",
}));

vi.mock("onnxruntime-web/webgpu", () => {
  class Tensor {
    constructor(
      public type: string,
      public data: unknown,
      public dims: number[],
    ) {}
  }
  return {
    env: { wasm: { numThreads: 1 }, logLevel: "fatal" },
    Tensor,
    InferenceSession: {
      create: async (_path: string, options: { executionProviders: string[] }) => {
        const provider = options.executionProviders[0]!;
        state.created.push(provider);
        if (state.webgpu === "throw" && provider === "webgpu") throw new Error("webgpu could not initialise");
        if (state.webgpu === "hang" && provider === "webgpu") await new Promise(() => {});
        if (state.wasm === "throw" && provider === "wasm") throw new Error("wasm could not initialise");
        return {
          run: async () => ({
            logits: { data: new Float32Array([0.1, 0.2, 0.7]), dims: [1, 1, 3] },
          }),
        };
      },
    },
  };
});

import { argmax, backendStatus, resetSession, runDemoModel } from "../src/lib/onnx";

function installWebGpu(adapterPresent: boolean) {
  Object.defineProperty(navigator, "gpu", {
    configurable: true,
    value: {
      requestAdapter: async () => (adapterPresent ? { info: { vendor: "test-vendor", architecture: "test-arch" } } : null),
    },
  });
}

function removeWebGpu() {
  Object.defineProperty(navigator, "gpu", { configurable: true, value: undefined });
}

async function statusFor(preference: "auto" | "wasm" | "webgpu") {
  resetSession();
  return backendStatus(preference);
}

describe("onnx runtime provider reporting", () => {
  beforeEach(() => {
    state.created = [];
    state.webgpu = "ok";
    state.wasm = "ok";
    resetSession();
  });

  afterEach(() => {
    removeWebGpu();
  });

  it("reports WASM and names the reason when navigator.gpu is absent", async () => {
    removeWebGpu();
    const status = await statusFor("auto");
    expect(status.provider).toBe("wasm");
    expect(status.webgpuSupported).toBe(false);
    expect(status.fallbackReason).toContain("navigator.gpu");
    // WASM only: no WebGPU session is even attempted without the API.
    expect(state.created).toEqual(["wasm"]);
  });

  it("treats an API without an adapter as unsupported rather than guessing", async () => {
    installWebGpu(false);
    const status = await statusFor("auto");
    expect(status.provider).toBe("wasm");
    expect(status.fallbackReason).toContain("requestAdapter");
  });

  it("falls back to WASM when the WebGPU session fails, and says why", async () => {
    installWebGpu(true);
    state.webgpu = "throw";
    const status = await statusFor("auto");
    expect(status.provider).toBe("wasm");
    expect(status.fallbackReason).toContain("webgpu could not initialise");
    expect(state.created).toEqual(["webgpu", "wasm"]);
  });

  it("reports WebGPU only after a real inference succeeded on it", async () => {
    installWebGpu(true);
    const status = await statusFor("webgpu");
    expect(status.provider).toBe("webgpu");
    expect(status.detail).toContain("native EP");
    expect(status.adapter).toContain("test-vendor");
    expect(state.created).toEqual(["webgpu"]);
  });

  it("runs the demo model and returns logits from the active provider", async () => {
    installWebGpu(true);
    const result = await runDemoModel([1, 2, 3], "webgpu");
    expect(result.provider).toBe("webgpu");
    expect(result.logits).toHaveLength(3);
    expect(argmax(result.logits)).toBe(2);
  });

  it("rebuilds the session when the provider preference changes", async () => {
    installWebGpu(true);
    const auto = await backendStatus("auto");
    expect(auto.provider).toBe("webgpu");
    // Changing the preference must actually switch providers, not reuse the cached session.
    const wasm = await backendStatus("wasm");
    expect(wasm.provider).toBe("wasm");
    expect(state.created).toEqual(["webgpu", "wasm"]);
  });

  it("bounds a hanging provider attempt instead of spinning forever", async () => {
    vi.useFakeTimers();
    try {
      installWebGpu(true);
      state.webgpu = "hang";
      const pending = backendStatus("auto");
      await vi.advanceTimersByTimeAsync(20_100);
      const status = await pending;
      expect(status.provider).toBe("wasm");
      expect(status.fallbackReason).toContain("did not settle");
    } finally {
      vi.useRealTimers();
    }
  });

  it("surfaces a hard error when every provider fails", async () => {
    installWebGpu(true);
    state.webgpu = "throw";
    state.wasm = "throw";
    resetSession();
    await expect(backendStatus("auto")).rejects.toThrow(/no ONNX provider could run the model/);
  });
});
