/**
 * Browser-side tiny MLP: real forward pass, real backprop, real patches.
 *
 * This exists so the web build can do *something* honestly: train a small model on a synthetic task
 * in a worker, inspect its weights, edit them, and measure the effect. It is deliberately limited to
 * one hidden layer — anything larger belongs to the desktop engine, not to a demo.
 */

export interface MlpConfig {
  inputDim: number;
  hidden: number;
  outputDim: number;
  seed: number;
}

export interface MlpWeights {
  w1: Float32Array; // [inputDim, hidden]
  b1: Float32Array; // [hidden]
  w2: Float32Array; // [hidden, outputDim]
  b2: Float32Array; // [outputDim]
}

export interface TinyMlp {
  config: MlpConfig;
  weights: MlpWeights;
}

export type TensorName = "w1" | "b1" | "w2" | "b2";

export interface Batch {
  x: Float32Array; // [n, inputDim]
  y: Int32Array; // [n]
}

/** Deterministic, fast PRNG so a seed reproduces exactly the same initialisation in any browser. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number): number {
  // Box–Muller; never returns 0 because u1 is in (0, 1].
  const u1 = Math.max(rand(), 1e-12);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function createMlp(config: MlpConfig): TinyMlp {
  const rand = mulberry32(config.seed);
  const scale1 = Math.sqrt(2 / config.inputDim);
  const scale2 = Math.sqrt(2 / config.hidden);
  const w1 = new Float32Array(config.inputDim * config.hidden);
  for (let i = 0; i < w1.length; i += 1) w1[i] = gaussian(rand) * scale1;
  const b1 = new Float32Array(config.hidden);
  const w2 = new Float32Array(config.hidden * config.outputDim);
  for (let i = 0; i < w2.length; i += 1) w2[i] = gaussian(rand) * scale2;
  const b2 = new Float32Array(config.outputDim);
  return { config, weights: { w1, b1, w2, b2 } };
}

export function tensors(model: TinyMlp): Record<TensorName, Float32Array> {
  return model.weights;
}

export function tensorShape(model: TinyMlp, name: TensorName): [number, number] | [number] {
  const c = model.config;
  switch (name) {
    case "w1":
      return [c.inputDim, c.hidden];
    case "b1":
      return [c.hidden];
    case "w2":
      return [c.hidden, c.outputDim];
    case "b2":
      return [c.outputDim];
  }
}

/** Teacher used to generate the synthetic task. Fixed per seed so results are reproducible. */
export function teacherMatrix(inputDim: number, outputDim: number, taskSeed: number): Float32Array {
  const rand = mulberry32(taskSeed * 7919 + inputDim * 31 + outputDim);
  const matrix = new Float32Array(inputDim * outputDim);
  const scale = 1 / Math.sqrt(inputDim);
  for (let i = 0; i < matrix.length; i += 1) matrix[i] = gaussian(rand) * scale;
  return matrix;
}

export function makeBatch(
  config: MlpConfig,
  batchSize: number,
  seed: number,
  split: "train" | "val" = "train",
  taskSeed = 20261103,
): Batch {
  const rand = mulberry32(seed + (split === "val" ? 10_000_000 : 0));
  const x = new Float32Array(batchSize * config.inputDim);
  const y = new Int32Array(batchSize);
  const matrix = teacherMatrix(config.inputDim, config.outputDim, taskSeed);
  for (let row = 0; row < batchSize; row += 1) {
    for (let col = 0; col < config.inputDim; col += 1) {
      x[row * config.inputDim + col] = gaussian(rand);
    }
    // label = argmax of the teacher projection, with 10% label noise
    let best = 0;
    let bestValue = -Infinity;
    for (let out = 0; out < config.outputDim; out += 1) {
      let value = 0;
      for (let col = 0; col < config.inputDim; col += 1) {
        value += x[row * config.inputDim + col]! * matrix[col * config.outputDim + out]!;
      }
      if (value > bestValue) {
        bestValue = value;
        best = out;
      }
    }
    if (rand() < 0.1) best = Math.floor(rand() * config.outputDim);
    y[row] = Math.min(best, config.outputDim - 1);
  }
  return { x, y };
}

export function forward(model: TinyMlp, x: Float32Array, rows: number): Float32Array {
  const { inputDim, hidden, outputDim } = model.config;
  const { w1, b1, w2, b2 } = model.weights;
  const out = new Float32Array(rows * outputDim);
  const preActivation = new Float32Array(hidden);
  for (let r = 0; r < rows; r += 1) {
    for (let h = 0; h < hidden; h += 1) {
      let sum = b1[h]!;
      for (let i = 0; i < inputDim; i += 1) sum += x[r * inputDim + i]! * w1[i * hidden + h]!;
      preActivation[h] = sum;
    }
    for (let o = 0; o < outputDim; o += 1) {
      let sum = b2[o]!;
      for (let h = 0; h < hidden; h += 1) {
        const act = preActivation[h]! > 0 ? preActivation[h]! : 0;
        sum += act * w2[h * outputDim + o]!;
      }
      out[r * outputDim + o] = sum;
    }
  }
  return out;
}

function softmaxCrossEntropy(logits: Float32Array, y: Int32Array, rows: number, classes: number) {
  let loss = 0;
  const probabilities = new Float32Array(rows * classes);
  for (let r = 0; r < rows; r += 1) {
    let max = -Infinity;
    for (let c = 0; c < classes; c += 1) max = Math.max(max, logits[r * classes + c]!);
    let sum = 0;
    for (let c = 0; c < classes; c += 1) {
      const value = Math.exp(logits[r * classes + c]! - max);
      probabilities[r * classes + c] = value;
      sum += value;
    }
    for (let c = 0; c < classes; c += 1) probabilities[r * classes + c]! /= sum;
    loss -= Math.log(Math.max(probabilities[r * classes + y[r]!]!, 1e-9));
  }
  return { loss: loss / rows, probabilities };
}

export interface TrainOptions {
  steps: number;
  batchSize: number;
  lr: number;
  seed: number;
  taskSeed?: number;
  onStep?: (update: { step: number; loss: number }) => void;
  shouldStop?: () => boolean;
}

export interface TrainResult {
  steps: number;
  initialLoss: number;
  finalLoss: number;
  valLoss: number;
  accuracy: number;
}

/**
 * Incremental SGD + momentum trainer.
 *
 * Being incremental matters: the browser worker must be able to stop between steps and report progress,
 * which a single blocking loop cannot do.
 */
export class TrainerSession {
  private readonly momentum: {
    w1: Float32Array;
    b1: Float32Array;
    w2: Float32Array;
    b2: Float32Array;
  };
  private readonly gradW1: Float32Array;
  private readonly gradB1: Float32Array;
  private readonly gradW2: Float32Array;
  private readonly gradB2: Float32Array;
  readonly taskSeed: number;
  stepsDone = 0;
  firstLoss = 0;
  lastLoss = 0;

  constructor(
    readonly model: TinyMlp,
    readonly options: { lr: number; batchSize: number; seed: number; taskSeed?: number; beta?: number },
  ) {
    const { w1, b1, w2, b2 } = model.weights;
    this.momentum = {
      w1: new Float32Array(w1.length),
      b1: new Float32Array(b1.length),
      w2: new Float32Array(w2.length),
      b2: new Float32Array(b2.length),
    };
    this.gradW1 = new Float32Array(w1.length);
    this.gradB1 = new Float32Array(b1.length);
    this.gradW2 = new Float32Array(w2.length);
    this.gradB2 = new Float32Array(b2.length);
    this.taskSeed = options.taskSeed ?? 20261103;
  }

  /** One optimiser step. Returns the training loss for this batch. */
  step(): number {
    const model = this.model;
    const { inputDim, hidden, outputDim } = model.config;
    const { w1, b1, w2, b2 } = model.weights;
    const batchSize = this.options.batchSize;
    const lr = this.options.lr;
    const beta = this.options.beta ?? 0.9;

    const batch = makeBatch(model.config, batchSize, this.options.seed * 1_000_003 + this.stepsDone, "train", this.taskSeed);
    const logits = forward(model, batch.x, batchSize);
    const { loss, probabilities } = softmaxCrossEntropy(logits, batch.y, batchSize, outputDim);
    if (this.stepsDone === 0) this.firstLoss = loss;
    this.lastLoss = loss;
    this.stepsDone += 1;

    this.gradW1.fill(0);
    this.gradB1.fill(0);
    this.gradW2.fill(0);
    this.gradB2.fill(0);

    for (let r = 0; r < batchSize; r += 1) {
      const dLogits = new Float32Array(outputDim);
      for (let o = 0; o < outputDim; o += 1) {
        dLogits[o] = (probabilities[r * outputDim + o]! - (batch.y[r] === o ? 1 : 0)) / batchSize;
      }
      const activations = new Float32Array(hidden);
      for (let h = 0; h < hidden; h += 1) {
        let sum = b1[h]!;
        for (let i = 0; i < inputDim; i += 1) sum += batch.x[r * inputDim + i]! * w1[i * hidden + h]!;
        activations[h] = sum > 0 ? sum : 0;
      }
      for (let h = 0; h < hidden; h += 1) {
        for (let o = 0; o < outputDim; o += 1) {
          this.gradW2[h * outputDim + o]! += activations[h]! * dLogits[o]!;
        }
      }
      for (let o = 0; o < outputDim; o += 1) this.gradB2[o]! += dLogits[o]!;

      const dHidden = new Float32Array(hidden);
      for (let h = 0; h < hidden; h += 1) {
        let sum = 0;
        for (let o = 0; o < outputDim; o += 1) sum += dLogits[o]! * w2[h * outputDim + o]!;
        dHidden[h] = activations[h]! > 0 ? sum : 0;
      }
      for (let i = 0; i < inputDim; i += 1) {
        for (let h = 0; h < hidden; h += 1) {
          this.gradW1[i * hidden + h]! += batch.x[r * inputDim + i]! * dHidden[h]!;
        }
      }
      for (let h = 0; h < hidden; h += 1) this.gradB1[h]! += dHidden[h]!;
    }

    applySgd(w1, this.gradW1, this.momentum.w1, lr, beta);
    applySgd(b1, this.gradB1, this.momentum.b1, lr, beta);
    applySgd(w2, this.gradW2, this.momentum.w2, lr, beta);
    applySgd(b2, this.gradB2, this.momentum.b2, lr, beta);
    return loss;
  }

  result(): TrainResult {
    const evaluation = evaluate(this.model, { batchSize: 64, seed: 999_331, taskSeed: this.taskSeed });
    return {
      steps: this.stepsDone,
      initialLoss: this.firstLoss,
      finalLoss: this.lastLoss,
      valLoss: evaluation.loss,
      accuracy: evaluation.accuracy,
    };
  }
}

export function train(model: TinyMlp, options: TrainOptions): TrainResult {
  const session = new TrainerSession(model, {
    lr: options.lr,
    batchSize: options.batchSize,
    seed: options.seed,
    taskSeed: options.taskSeed,
  });
  for (let step = 0; step < options.steps; step += 1) {
    if (options.shouldStop?.()) break;
    const loss = session.step();
    options.onStep?.({ step, loss });
  }
  return session.result();
}

function applySgd(weights: Float32Array, grads: Float32Array, velocity: Float32Array, lr: number, beta: number) {
  for (let i = 0; i < weights.length; i += 1) {
    velocity[i] = beta * velocity[i]! + grads[i]!;
    weights[i] = weights[i]! - lr * velocity[i]!;
  }
}

export function evaluate(
  model: TinyMlp,
  options: { batchSize?: number; seed?: number; taskSeed?: number } = {},
): { loss: number; accuracy: number } {
  const batchSize = options.batchSize ?? 64;
  const batch = makeBatch(model.config, batchSize, options.seed ?? 999_331, "val", options.taskSeed);
  const logits = forward(model, batch.x, batchSize);
  const { loss } = softmaxCrossEntropy(logits, batch.y, batchSize, model.config.outputDim);
  let correct = 0;
  for (let r = 0; r < batchSize; r += 1) {
    let best = 0;
    let bestValue = -Infinity;
    for (let c = 0; c < model.config.outputDim; c += 1) {
      const value = logits[r * model.config.outputDim + c]!;
      if (value > bestValue) {
        bestValue = value;
        best = c;
      }
    }
    if (best === batch.y[r]) correct += 1;
  }
  return { loss, accuracy: correct / batchSize };
}

// ------------------------------------------------------------------------------------------------
// Tensor operations — the browser equivalent of the engine's closed patch operation set
// ------------------------------------------------------------------------------------------------

export interface TensorStats {
  name: TensorName;
  shape: [number, number] | [number];
  numel: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  norm: number;
  zeros: number;
  finite: boolean;
}

export function tensorStats(model: TinyMlp, name: TensorName): TensorStats {
  const data = model.weights[name];
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let zeros = 0;
  let finite = true;
  for (let i = 0; i < data.length; i += 1) {
    const value = data[i]!;
    if (!Number.isFinite(value)) finite = false;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    if (value === 0) zeros += 1;
  }
  const mean = data.length ? sum / data.length : 0;
  let variance = 0;
  for (let i = 0; i < data.length; i += 1) variance += (data[i]! - mean) ** 2;
  const std = data.length ? Math.sqrt(variance / data.length) : 0;
  let normSquared = 0;
  for (let i = 0; i < data.length; i += 1) normSquared += data[i]! ** 2;
  return {
    name,
    shape: tensorShape(model, name),
    numel: data.length,
    min: data.length ? min : 0,
    max: data.length ? max : 0,
    mean,
    std,
    norm: Math.sqrt(normSquared),
    zeros: data.length ? zeros / data.length : 0,
    finite,
  };
}

export function histogram(model: TinyMlp, name: TensorName, bins = 24): { edges: number[]; counts: number[] } {
  const data = model.weights[name];
  if (!data.length) return { edges: [], counts: [] };
  let min = Infinity;
  let max = -Infinity;
  for (const value of data) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (min === max) return { edges: [min, max], counts: [data.length] };
  const width = (max - min) / bins;
  const counts = new Array(bins).fill(0) as number[];
  for (const value of data) {
    const index = Math.min(bins - 1, Math.floor((value - min) / width));
    counts[index] = (counts[index] ?? 0) + 1;
  }
  const edges = Array.from({ length: bins + 1 }, (_, index) => min + index * width);
  return { edges, counts };
}

export type PatchOp =
  | { op: "scale"; factor: number }
  | { op: "clamp"; min?: number; max?: number }
  | { op: "setScalar"; index: number; value: number }
  | { op: "normalize"; eps: number }
  | { op: "sparsify"; keep: number; seed: number };

export function applyPatch(model: TinyMlp, name: TensorName, patch: PatchOp): TinyMlp {
  const data = new Float32Array(model.weights[name]);
  switch (patch.op) {
    case "scale":
      for (let i = 0; i < data.length; i += 1) data[i] = data[i]! * patch.factor;
      break;
    case "clamp":
      for (let i = 0; i < data.length; i += 1) {
        let value = data[i]!;
        if (patch.min !== undefined) value = Math.max(value, patch.min);
        if (patch.max !== undefined) value = Math.min(value, patch.max);
        data[i] = value;
      }
      break;
    case "setScalar": {
      if (!Number.isInteger(patch.index) || patch.index < 0 || patch.index >= data.length) {
        throw new Error(`index ${patch.index} is out of range for ${name} (${data.length} elements)`);
      }
      data[patch.index] = patch.value;
      break;
    }
    case "normalize": {
      let norm = 0;
      for (const value of data) norm += value * value;
      norm = Math.sqrt(norm);
      const divisor = Math.max(norm, patch.eps);
      for (let i = 0; i < data.length; i += 1) data[i] = data[i]! / divisor;
      break;
    }
    case "sparsify": {
      const rand = mulberry32(patch.seed);
      for (let i = 0; i < data.length; i += 1) if (rand() > patch.keep) data[i] = 0;
      break;
    }
  }
  for (let i = 0; i < data.length; i += 1) {
    if (!Number.isFinite(data[i]!)) throw new Error("patch produced a non-finite value and was rejected");
  }
  return { config: model.config, weights: { ...model.weights, [name]: data } };
}

export function patchEffect(before: TinyMlp, after: TinyMlp, name: TensorName) {
  const a = before.weights[name];
  const b = after.weights[name];
  let changed = 0;
  let maxDelta = 0;
  let normBefore = 0;
  let normAfter = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) changed += 1;
    maxDelta = Math.max(maxDelta, Math.abs(b[i]! - a[i]!));
    normBefore += a[i]! ** 2;
    normAfter += b[i]! ** 2;
  }
  return {
    changed,
    maxDelta,
    normBefore: Math.sqrt(normBefore),
    normAfter: Math.sqrt(normAfter),
    relativeNormChange: normBefore > 0 ? Math.sqrt(normAfter) / Math.sqrt(normBefore) - 1 : 0,
  };
}

export function serializeMlp(model: TinyMlp): string {
  return JSON.stringify({
    config: model.config,
    w1: Array.from(model.weights.w1),
    b1: Array.from(model.weights.b1),
    w2: Array.from(model.weights.w2),
    b2: Array.from(model.weights.b2),
  });
}

export function deserializeMlp(text: string): TinyMlp {
  const parsed = JSON.parse(text) as {
    config: MlpConfig;
    w1: number[];
    b1: number[];
    w2: number[];
    b2: number[];
  };
  return {
    config: parsed.config,
    weights: {
      w1: new Float32Array(parsed.w1),
      b1: new Float32Array(parsed.b1),
      w2: new Float32Array(parsed.w2),
      b2: new Float32Array(parsed.b2),
    },
  };
}

// ------------------------------------------------------------------------------------------------
// Architecture preview maths (mirrors the Python formula so the browser cannot show a different count)
// ------------------------------------------------------------------------------------------------

export interface TransformerPreview {
  vocabSize: number;
  dModel: number;
  layers: number;
  heads: number;
  ffnDim: number;
  tieEmbeddings: boolean;
  /** Grouped-query attention. Defaults to ``heads`` (no grouping), like the engine's config. */
  nKvHeads?: number;
}

/** Exact parameter count for the decoder architecture implemented in the Python engine. */
export function countTransformerParams(c: TransformerPreview): number {
  const ffn = c.ffnDim;
  const kvHeads = c.nKvHeads ?? c.heads;
  const headDim = c.dModel / c.heads;
  const perBlock = 2 * c.dModel * c.dModel + 2 * c.dModel * kvHeads * headDim + 3 * c.dModel * ffn + 2 * c.dModel;
  let total = c.vocabSize * c.dModel + c.layers * perBlock + c.dModel;
  if (!c.tieEmbeddings) total += c.vocabSize * c.dModel;
  return Math.round(total);
}

export function countMlpParams(c: MlpConfig): number {
  return c.inputDim * c.hidden + c.hidden + c.hidden * c.outputDim + c.outputDim;
}


/** Linear interpolation in weight space between two models of the same architecture. */
export function interpolateWeights(a: TinyMlp, b: TinyMlp, t: number): TinyMlp {
  if (
    a.config.inputDim !== b.config.inputDim ||
    a.config.hidden !== b.config.hidden ||
    a.config.outputDim !== b.config.outputDim
  ) {
    throw new Error('interpolation requires identical architectures');
  }
  if (!(t >= 0 && t <= 1)) throw new Error('t must be in [0, 1]');
  const mix = (left: Float32Array, right: Float32Array) => {
    const out = new Float32Array(left.length);
    for (let i = 0; i < left.length; i += 1) out[i] = (1 - t) * left[i]! + t * right[i]!;
    return out;
  };
  return {
    config: a.config,
    weights: {
      w1: mix(a.weights.w1, b.weights.w1),
      b1: mix(a.weights.b1, b.weights.b1),
      w2: mix(a.weights.w2, b.weights.w2),
      b2: mix(a.weights.b2, b.weights.b2),
    },
  };
}
