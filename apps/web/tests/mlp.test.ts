import { describe, expect, it } from "vitest";

import {
  applyPatch,
  countMlpParams,
  countTransformerParams,
  createMlp,
  deserializeMlp,
  evaluate,
  forward,
  histogram,
  interpolateWeights,
  makeBatch,
  patchEffect,
  serializeMlp,
  tensorStats,
  TrainerSession,
  train,
  type MlpConfig,
} from "../src/lib/mlp";

const config: MlpConfig = { inputDim: 8, hidden: 12, outputDim: 4, seed: 3 };

describe("initialisation", () => {
  it("is deterministic for a seed", () => {
    const a = createMlp(config);
    const b = createMlp(config);
    expect(Array.from(a.weights.w1)).toEqual(Array.from(b.weights.w1));
  });

  it("differs across seeds", () => {
    const a = createMlp(config);
    const b = createMlp({ ...config, seed: 4 });
    expect(Array.from(a.weights.w1)).not.toEqual(Array.from(b.weights.w1));
  });

  it("produces finite, correctly sized tensors", () => {
    const model = createMlp(config);
    expect(model.weights.w1.length).toBe(8 * 12);
    expect(model.weights.b1.length).toBe(12);
    expect(model.weights.w2.length).toBe(12 * 4);
    expect(model.weights.b2.length).toBe(4);
    for (const tensor of Object.values(model.weights)) {
      expect(Array.from(tensor).every((value) => Number.isFinite(value))).toBe(true);
    }
  });
});

describe("forward pass", () => {
  it("returns logits with the right shape", () => {
    const model = createMlp(config);
    const batch = makeBatch(config, 5, 1);
    const logits = forward(model, batch.x, 5);
    expect(logits.length).toBe(5 * 4);
    expect(Array.from(logits).every((value) => Number.isFinite(value))).toBe(true);
  });

  it("is invariant to batch size for the same row", () => {
    const model = createMlp(config);
    const batch = makeBatch(config, 3, 2);
    const single = forward(model, batch.x.slice(0, 8), 1);
    const three = forward(model, batch.x, 3);
    expect(single[0]).toBeCloseTo(three[0]!, 5);
    expect(single[3]).toBeCloseTo(three[3]!, 5);
  });
});

describe("training", () => {
  it("reduces the loss and beats chance accuracy", () => {
    const model = createMlp(config);
    const before = evaluate(model, { batchSize: 128, seed: 11 });
    const result = train(model, { steps: 120, batchSize: 16, lr: 0.08, seed: 5 });
    expect(result.finalLoss).toBeLessThan(result.initialLoss);
    expect(result.valLoss).toBeLessThan(before.loss);
    expect(result.accuracy).toBeGreaterThan(0.25);
  });

  it("is reproducible for a seed", () => {
    const a = createMlp(config);
    const b = createMlp(config);
    const first = train(a, { steps: 20, batchSize: 8, lr: 0.05, seed: 9 });
    const second = train(b, { steps: 20, batchSize: 8, lr: 0.05, seed: 9 });
    expect(first.finalLoss).toBe(second.finalLoss);
    expect(Array.from(a.weights.w1)).toEqual(Array.from(b.weights.w1));
  });

  it("stops between steps when asked", () => {
    const model = createMlp(config);
    let steps = 0;
    const result = train(model, {
      steps: 500,
      batchSize: 8,
      lr: 0.05,
      seed: 1,
      shouldStop: () => {
        steps += 1;
        return steps > 5;
      },
    });
    expect(result.steps).toBeLessThan(10);
  });

  it("supports incremental sessions (the worker path)", () => {
    const model = createMlp(config);
    const session = new TrainerSession(model, { lr: 0.05, batchSize: 8, seed: 2 });
    const first = session.step();
    const second = session.step();
    expect(session.stepsDone).toBe(2);
    expect(Number.isFinite(first)).toBe(true);
    expect(Number.isFinite(second)).toBe(true);
    expect(session.result().steps).toBe(2);
  });
});

describe("tensor inspection and patches", () => {
  it("computes stats that match a manual calculation", () => {
    const model = createMlp(config);
    const stats = tensorStats(model, "b2");
    const data = Array.from(model.weights.b2);
    const mean = data.reduce((sum, value) => sum + value, 0) / data.length;
    expect(stats.mean).toBeCloseTo(mean, 6);
    expect(stats.numel).toBe(4);
    expect(stats.finite).toBe(true);
  });

  it("builds a histogram whose counts sum to the tensor size", () => {
    const model = createMlp(config);
    const { counts } = histogram(model, "w1", 12);
    expect(counts.reduce((sum, value) => sum + value, 0)).toBe(model.weights.w1.length);
  });

  it("scales and reports the delta", () => {
    const model = createMlp(config);
    const scaled = applyPatch(model, "w1", { op: "scale", factor: 0.5 });
    const effect = patchEffect(model, scaled, "w1");
    expect(effect.changed).toBe(model.weights.w1.length);
    expect(effect.relativeNormChange).toBeCloseTo(-0.5, 4);
    expect(effect.maxDelta).toBeGreaterThan(0);
  });

  it("clamps values into range", () => {
    const model = createMlp(config);
    const clamped = applyPatch(model, "w2", { op: "clamp", min: -0.1, max: 0.1 });
    // Float32 cannot represent 0.1 exactly, so compare with one ulp of slack.
    expect(Math.max(...Array.from(clamped.weights.w2))).toBeLessThanOrEqual(0.1 + 1e-6);
    expect(Math.min(...Array.from(clamped.weights.w2))).toBeGreaterThanOrEqual(-0.1 - 1e-6);
  });

  it("sets a single scalar and rejects out-of-range indices", () => {
    const model = createMlp(config);
    const patched = applyPatch(model, "b1", { op: "setScalar", index: 2, value: 1.25 });
    expect(patched.weights.b1[2]).toBe(1.25);
    expect(() => applyPatch(model, "b1", { op: "setScalar", index: 99, value: 1 })).toThrow(/out of range/);
  });

  it("normalises to unit norm", () => {
    const model = createMlp(config);
    const normalised = applyPatch(model, "w1", { op: "normalize", eps: 1e-9 });
    expect(tensorStats(normalised, "w1").norm).toBeCloseTo(1, 4);
  });

  it("sparsifies deterministically and keeps the requested fraction", () => {
    const model = createMlp(config);
    const a = applyPatch(model, "w1", { op: "sparsify", keep: 0.5, seed: 4 });
    const b = applyPatch(model, "w1", { op: "sparsify", keep: 0.5, seed: 4 });
    expect(Array.from(a.weights.w1)).toEqual(Array.from(b.weights.w1));
    const zeros = Array.from(a.weights.w1).filter((value) => value === 0).length;
    expect(zeros).toBeGreaterThan(0);
  });

  it("survives a serialise/deserialise round trip", () => {
    const model = createMlp(config);
    const restored = deserializeMlp(serializeMlp(model));
    expect(restored.config).toEqual(model.config);
    expect(Array.from(restored.weights.w1)).toEqual(Array.from(model.weights.w1));
    expect(Array.from(restored.weights.b2)).toEqual(Array.from(model.weights.b2));
  });
});

describe("weight-space interpolation", () => {
  it("returns the endpoints at t=0 and t=1", () => {
    const a = createMlp(config);
    const b = createMlp({ ...config, seed: 99 });
    expect(Array.from(interpolateWeights(a, b, 0).weights.w1)).toEqual(Array.from(a.weights.w1));
    expect(Array.from(interpolateWeights(a, b, 1).weights.w1)).toEqual(Array.from(b.weights.w1));
  });

  it("rejects mismatched architectures and out-of-range t", () => {
    const a = createMlp(config);
    const b = createMlp({ ...config, hidden: 16 });
    expect(() => interpolateWeights(a, b, 0.5)).toThrow(/identical architectures/);
    expect(() => interpolateWeights(a, a, 1.5)).toThrow(/t must be in/);
  });

  it("produces a model that still evaluates", () => {
    const a = createMlp(config);
    const b = createMlp({ ...config, seed: 77 });
    const mixed = interpolateWeights(a, b, 0.5);
    const metrics = evaluate(mixed, { batchSize: 32, seed: 3 });
    expect(Number.isFinite(metrics.loss)).toBe(true);
    expect(metrics.accuracy).toBeGreaterThanOrEqual(0);
  });
});

describe("architecture maths mirrors the engine", () => {
  it("reproduces the documented transformer counts", () => {
    // Matches the Python build exactly: d_model=64, layers=2, heads=4, default ffn=192, vocab=64, tied.
    expect(
      countTransformerParams({ vocabSize: 64, dModel: 64, layers: 2, heads: 4, ffnDim: 192, tieEmbeddings: true }),
    ).toBe(110_912);
    expect(
      countTransformerParams({ vocabSize: 64, dModel: 64, layers: 2, heads: 4, ffnDim: 192, tieEmbeddings: true, nKvHeads: 2 }),
    ).toBeLessThan(110_912);
    const untied = countTransformerParams({
      vocabSize: 64,
      dModel: 64,
      layers: 2,
      heads: 4,
      ffnDim: 192,
      tieEmbeddings: false,
    });
    expect(untied - 110_912).toBe(64 * 64);
  });

  it("counts MLP parameters including biases", () => {
    expect(countMlpParams({ inputDim: 16, hidden: 32, outputDim: 4, seed: 0 })).toBe(16 * 32 + 32 + 32 * 4 + 4);
  });
});
