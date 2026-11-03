import { beforeEach, describe, expect, it } from "vitest";

import { isHistoryBundle, Workspace } from "../src/lib/db";
import type { CheckpointRecord, ModelRecord, RunRecord } from "../src/lib/types";

function model(id: string, createdAt: string): ModelRecord {
  return {
    id,
    name: `Model ${id}`,
    family: "mlp",
    config: { inputDim: 4, hidden: 8, outputDim: 2, seed: 1 },
    parameterCount: 4 * 8 + 8 + 8 * 2 + 2,
    architectureHash: `hash-${id}`,
    createdAt,
    notes: "",
    source: "browser-trained",
    origin: "created",
    weights: "{}",
    maturity: "stable",
  };
}

function checkpoint(id: string, modelId: string, createdAt: string): CheckpointRecord {
  return {
    id,
    modelId,
    label: "final",
    sha256: `sha-${id}`,
    sizeBytes: 128,
    origin: "training",
    createdAt,
  };
}

function run(id: string, modelId: string, startedAt: string): RunRecord {
  return {
    id,
    modelId,
    kind: "training",
    state: "completed",
    startedAt,
    config: {},
    metrics: [],
    summary: { finalLoss: 0.42 },
    provenance: { runtime: "browser-worker" },
  };
}

describe("browser workspace", () => {
  let db: Workspace;

  beforeEach(async () => {
    db = await Workspace.open();
    await db.clearAll();
  });

  it("stores and lists models newest first", async () => {
    await db.putModel(model("a", "2026-01-01T00:00:00Z"));
    await db.putModel(model("b", "2026-02-01T00:00:00Z"));
    const models = await db.listModels();
    expect(models.map((entry) => entry.id)).toEqual(["b", "a"]);
  });

  it("filters checkpoints by model and returns the newest as latest", async () => {
    await db.putCheckpoint(checkpoint("c1", "a", "2026-01-01T00:00:00Z"));
    await db.putCheckpoint(checkpoint("c2", "a", "2026-03-01T00:00:00Z"));
    await db.putCheckpoint(checkpoint("c3", "b", "2026-02-01T00:00:00Z"));
    expect((await db.listCheckpoints("a")).map((entry) => entry.id)).toEqual(["c2", "c1"]);
    expect((await db.latestCheckpoint("a"))?.id).toBe("c2");
  });

  it("round-trips settings", async () => {
    expect(await db.getSetting("theme", "dark")).toBe("dark");
    await db.setSetting("theme", "light");
    expect(await db.getSetting("theme", "dark")).toBe("light");
  });

  it("exports and imports a complete bundle", async () => {
    await db.putModel(model("a", "2026-01-01T00:00:00Z"));
    await db.putCheckpoint(checkpoint("c1", "a", "2026-01-01T00:00:00Z"));
    await db.putRun(run("r1", "a", "2026-01-01T00:00:00Z"));
    const bundle = await db.exportBundle();
    expect(bundle.schemaVersion).toBe("1.0");
    expect(bundle.models).toHaveLength(1);

    await db.clearAll();
    expect(await db.listModels()).toHaveLength(0);

    await db.importBundle(bundle);
    expect((await db.listModels()).map((entry) => entry.id)).toEqual(["a"]);
    expect(await db.listCheckpoints("a")).toHaveLength(1);
    expect(await db.listRuns("a")).toHaveLength(1);
  });

  it("rejects foreign JSON as a history bundle", () => {
    expect(isHistoryBundle({ hello: "world" })).toBe(false);
    expect(isHistoryBundle({ schemaVersion: "2.0", models: [] })).toBe(false);
    expect(isHistoryBundle({ schemaVersion: "1.0", models: [] })).toBe(true);
  });
});
