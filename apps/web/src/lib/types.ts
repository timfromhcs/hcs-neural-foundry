/** Shared types for the browser workspace. Mirrors the engine's wire vocabulary, not its internals. */

export type Maturity = "stable" | "experimental" | "unsupported";
export type Capability = "full" | "limited" | "experimental" | "unavailable";

export type RunKind = "training" | "evaluation" | "evolution" | "generation" | "export" | "patch";
export type RunState = "queued" | "running" | "completed" | "failed" | "cancelled" | "invalid" | "interrupted";

export interface ModelRecord {
  id: string;
  name: string;
  family: "mlp" | "transformer-lm";
  /** The architecture payload, kept as JSON so it can be exported and rebuilt. */
  config: Record<string, unknown>;
  parameterCount: number;
  architectureHash: string;
  createdAt: string;
  notes: string;
  source: "definition" | "browser-trained" | "shipped-example" | "browser-patched";
  origin: "created" | "imported" | "generated" | "evolved" | "patched";
  /** Present for MLPs trained in this browser: the actual weights, JSON encoded. */
  weights?: string;
  maturity: Maturity;
}

export interface CheckpointRecord {
  id: string;
  modelId: string;
  label: string;
  /** Content hash of the serialised weights. */
  sha256: string;
  sizeBytes: number;
  origin: "init" | "training" | "patch";
  createdAt: string;
  parentSha256?: string;
  weights?: string;
}

export interface MetricPoint {
  step: number;
  loss: number;
  valLoss?: number;
  lr: number;
  elapsedMs: number;
}

export interface RunRecord {
  id: string;
  modelId: string;
  kind: RunKind;
  state: RunState;
  startedAt: string;
  finishedAt?: string;
  config: Record<string, unknown>;
  metrics: MetricPoint[];
  summary: Record<string, unknown>;
  provenance: {
    runtime: "browser-worker" | "desktop-engine";
    provider?: string;
    seed?: number;
    datasetSha256?: string;
    architectureHash?: string;
    evaluatorVersion?: string;
    note?: string;
  };
}

export interface EventRecord {
  id: string;
  at: string;
  level: "info" | "warning" | "error";
  scope: string;
  message: string;
}

export interface SettingsRecord {
  key: string;
  value: unknown;
}

export interface DemoManifest {
  id: string;
  name: string;
  family: "transformer-lm";
  maturity: Maturity;
  config: Record<string, unknown>;
  parameterCount: number;
  architectureHash: string;
  onnx: {
    path: string;
    sha256: string;
    bytes: number;
    opset: number;
    parity: { maxAbsDiff: number; tolerance: number; provider: string };
  };
  training: {
    steps: number;
    batchSize: number;
    blockSize: number;
    finalLoss: number;
    valLoss: number;
    seconds: number;
    device: string;
    seed: number;
    date: string;
  };
  evaluation: {
    suite: string;
    metrics: { name: string; value: number; unit?: string; definition: string }[];
  };
  corpus: { sha256: string; chars: number };
  claim: string;
}

export interface HistoryBundle {
  schemaVersion: "1.0";
  exportedAt: string;
  models: ModelRecord[];
  checkpoints: CheckpointRecord[];
  runs: RunRecord[];
  events: EventRecord[];
}
