/** Weight Studio for browser-trained MLPs: real tensors, real patches, real provenance. */

import { useEffect, useMemo, useState } from "react";

import { useI18n, useT } from "../i18n";
import { formatNumber, shortHash } from "../lib/format";
import {
  applyPatch,
  deserializeMlp,
  histogram,
  patchEffect,
  serializeMlp,
  tensorShape,
  tensorStats,
  type PatchOp,
  type TensorName,
  type TinyMlp,
} from "../lib/mlp";
import { useApp } from "../state";
import { Alert, Card, Field, Heatmap, BarChart, PageHead, Stat, StatusPill } from "../components/ui";
import type { CheckpointRecord, ModelRecord } from "../lib/types";

const OPERATIONS = ["scale", "clamp", "setScalar", "normalize", "sparsify"] as const;
type Operation = (typeof OPERATIONS)[number];

const TENSORS: TensorName[] = ["w1", "b1", "w2", "b2"];

export function WeightStudioPage({ onNavigate }: { onNavigate: (route: string) => void }) {
  const t = useT();
  const { locale } = useI18n();
  const { db, refresh, version } = useApp();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [tensor, setTensor] = useState<TensorName>("w1");
  const [operation, setOperation] = useState<Operation>("scale");
  const [factor, setFactor] = useState(0.98);
  const [min, setMin] = useState(-0.5);
  const [max, setMax] = useState(0.5);
  const [index, setIndex] = useState(0);
  const [value, setValue] = useState(0);
  const [keep, setKeep] = useState(0.5);
  const [seed, setSeed] = useState(1);
  const [effect, setEffect] = useState<ReturnType<typeof patchEffect> | null>(null);
  const [applied, setApplied] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!db) return;
    void (async () => {
      const list = (await db.listModels()).filter((model) => Boolean(model.weights));
      setModels(list);
      if (!modelId && list[0]) setModelId(list[0].id);
    })();
  }, [db, version, modelId]);

  useEffect(() => {
    if (!db || !modelId) return;
    void (async () => setCheckpoints(await db.listCheckpoints(modelId)))();
  }, [db, modelId, version]);

  const model = models.find((entry) => entry.id === modelId);
  const current: TinyMlp | null = useMemo(() => {
    if (!model?.weights) return null;
    try {
      return deserializeMlp(model.weights);
    } catch {
      return null;
    }
  }, [model]);

  const stats = current ? tensorStats(current, tensor) : null;
  const hist = current ? histogram(current, tensor, 20) : null;

  const patch: PatchOp = useMemo(() => {
    switch (operation) {
      case "scale":
        return { op: "scale", factor };
      case "clamp":
        return { op: "clamp", min, max };
      case "setScalar":
        return { op: "setScalar", index, value };
      case "normalize":
        return { op: "normalize", eps: 1e-6 };
      case "sparsify":
        return { op: "sparsify", keep, seed };
    }
  }, [operation, factor, min, max, index, value, keep, seed]);

  const preview = () => {
    setError(null);
    setApplied(null);
    if (!current) return;
    try {
      const patched = applyPatch(current, tensor, patch);
      setEffect(patchEffect(current, patched, tensor));
    } catch (cause) {
      setEffect(null);
      setError((cause as Error).message);
    }
  };

  const apply = async () => {
    if (!db || !current || !model) return;
    setError(null);
    try {
      const patched = applyPatch(current, tensor, patch);
      const delta = patchEffect(current, patched, tensor);
      const serialized = serializeMlp(patched);
      const sha = await digest(serialized);
      const createdAt = new Date().toISOString();
      const parent = checkpoints[0];
      const checkpoint: CheckpointRecord = {
        id: `ckpt_${Date.now().toString(36)}`,
        modelId: model.id,
        label: `patch:${operation}`,
        sha256: sha,
        sizeBytes: new TextEncoder().encode(serialized).length,
        origin: "patch",
        createdAt,
        parentSha256: parent?.sha256,
        weights: serialized,
      };
      await db.putCheckpoint(checkpoint);
      await db.putModel({ ...model, weights: serialized, source: "browser-patched" });
      await db.putRun({
        id: `run_${Date.now().toString(36)}`,
        modelId: model.id,
        kind: "patch",
        state: "completed",
        startedAt: createdAt,
        finishedAt: createdAt,
        config: { tensor, operation, patch },
        metrics: [],
        summary: { ...delta, checkpointSha256: sha },
        provenance: {
          runtime: "browser-worker",
          architectureHash: model.architectureHash,
          note: "browser patch: immutable checkpoint written before the model head moved",
        },
      });
      await db.logEvent({
        id: `evt_${Date.now().toString(36)}`,
        at: createdAt,
        level: "info",
        scope: `patch:${model.id}`,
        message: `${operation} on ${tensor} changed ${delta.changed} elements`,
      });
      setEffect(delta);
      setApplied(sha);
      refresh();
    } catch (cause) {
      setError((cause as Error).message);
    }
  };

  const rollback = async (checkpoint: CheckpointRecord) => {
    if (!db || !model) return;
    if (!checkpoint.weights) return;
    await db.putModel({ ...model, weights: checkpoint.weights });
    await db.logEvent({
      id: `evt_${Date.now().toString(36)}`,
      at: new Date().toISOString(),
      level: "warning",
      scope: `model:${model.id}`,
      message: `head moved to ${checkpoint.sha256.slice(0, 12)}`,
    });
    refresh();
  };

  if (models.length === 0) {
    return (
      <>
        <PageHead title={t("weights.title")} subtitle={t("weights.subtitle")} />
        <Alert tone="warn" title={t("weights.noModel")}>
          <button type="button" className="btn small" onClick={() => onNavigate("training-lab")}>
            {t("train.start")}
          </button>
        </Alert>
      </>
    );
  }

  return (
    <>
      <PageHead title={t("weights.title")} subtitle={t("weights.subtitle")} />

      <div className="split">
        <Card title={t("weights.tensors")}>
          <div className="grid grid-2">
            <Field label={t("model.name")} htmlFor="model">
              <select id="model" value={modelId} onChange={(event) => setModelId(event.target.value)}>
                {models.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={t("weights.tensors")} htmlFor="tensor">
              <select id="tensor" value={tensor} onChange={(event) => setTensor(event.target.value as TensorName)}>
                {TENSORS.map((name) => (
                  <option key={name} value={name}>
                    {name} {current ? JSON.stringify(tensorShape(current, name)) : ''}
                  </option>
                ))}
              </select>
            </Field>
          </div>

          {stats && (
            <div className="grid grid-4" style={{ marginTop: "var(--space-4)" }}>
              <Stat label={t("weights.mean")} value={formatNumber(stats.mean, locale, 4)} />
              <Stat label={t("weights.std")} value={formatNumber(stats.std, locale, 4)} />
              <Stat label={t("weights.norm")} value={formatNumber(stats.norm, locale, 3)} />
              <Stat label={t("weights.finite")} value={stats.finite ? "true" : "false"} />
            </div>
          )}

          {current && (
            <div style={{ marginTop: "var(--space-4)" }}>
              <h4 style={{ marginBottom: "var(--space-2)" }}>{t("weights.heatmap")}</h4>
              <Heatmap
                values={Array.from(current.weights[tensor]).slice(0, 64)}
                rows={8}
                cols={8}
                title={`${tensor} ${JSON.stringify(tensorShape(current, tensor))}`}
              />
            </div>
          )}

          {hist && (
            <div style={{ marginTop: "var(--space-4)" }}>
              <h4 style={{ marginBottom: "var(--space-2)" }}>{t("weights.histogram")}</h4>
              <BarChart values={hist.counts} title={t("weights.histogram")} labels={{ x: "value", y: "count" }} />
            </div>
          )}
        </Card>

        <Card title={t("weights.operation")}>
          <div className="stack">
            <Field label={t("weights.operation")} htmlFor="op">
              <select id="op" value={operation} onChange={(event) => setOperation(event.target.value as Operation)}>
                {OPERATIONS.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </Field>

            {operation === "scale" && (
              <Field label="factor" htmlFor="factor">
                <input id="factor" type="number" step="0.01" value={factor} onChange={(event) => setFactor(Number(event.target.value))} />
              </Field>
            )}
            {operation === "clamp" && (
              <div className="grid grid-2">
                <Field label="min" htmlFor="min">
                  <input id="min" type="number" step="0.1" value={min} onChange={(event) => setMin(Number(event.target.value))} />
                </Field>
                <Field label="max" htmlFor="max">
                  <input id="max" type="number" step="0.1" value={max} onChange={(event) => setMax(Number(event.target.value))} />
                </Field>
              </div>
            )}
            {operation === "setScalar" && (
              <div className="grid grid-2">
                <Field label="index" htmlFor="index" hint={stats ? `0…${stats.numel - 1}` : undefined}>
                  <input id="index" type="number" min={0} value={index} onChange={(event) => setIndex(Number(event.target.value))} />
                </Field>
                <Field label="value" htmlFor="value">
                  <input id="value" type="number" step="0.01" value={value} onChange={(event) => setValue(Number(event.target.value))} />
                </Field>
              </div>
            )}
            {operation === "sparsify" && (
              <div className="grid grid-2">
                <Field label="keep" htmlFor="keep" hint="0 < keep ≤ 1">
                  <input id="keep" type="number" step="0.05" min={0.05} max={1} value={keep} onChange={(event) => setKeep(Number(event.target.value))} />
                </Field>
                <Field label="seed" htmlFor="pseed">
                  <input id="pseed" type="number" min={0} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
                </Field>
              </div>
            )}
            {operation === "normalize" && <p className="tiny faint">Divides by the tensor L2 norm with epsilon 1e-6.</p>}

            <div className="row">
              <button type="button" className="btn" onClick={preview}>
                {t("weights.propose")}
              </button>
              <button type="button" className="btn primary" onClick={() => void apply()}>
                {t("weights.apply")}
              </button>
            </div>

            {error && <Alert tone="bad">{error}</Alert>}
            {effect && (
              <div className="stack">
                <h4>{t("weights.diff")}</h4>
                <table>
                  <tbody>
                    <tr>
                      <td>{t("weights.changed")}</td>
                      <td className="num">{formatNumber(effect.changed, locale)}</td>
                    </tr>
                    <tr>
                      <td>{t("weights.maxDelta")}</td>
                      <td className="num">{effect.maxDelta.toExponential(3)}</td>
                    </tr>
                    <tr>
                      <td>{t("weights.normChange")}</td>
                      <td className="num">{(effect.relativeNormChange * 100).toFixed(3)}%</td>
                    </tr>
                  </tbody>
                </table>
                {applied && (
                  <Alert tone="ok">
                    {t("weights.newCheckpoint")}: <code className="inline">{shortHash(applied)}</code>
                  </Alert>
                )}
              </div>
            )}

            <Alert tone="warn">{t("weights.safety")}</Alert>
          </div>
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("weights.provenance")} subtitle={t("library.checkpoints")}>
          {checkpoints.length === 0 ? (
            <p className="muted">{t("history.empty")}</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("library.label")}</th>
                    <th>{t("library.hash")}</th>
                    <th>{t("library.origin")}</th>
                    <th>{t("library.created")}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {checkpoints.map((checkpoint, position) => (
                    <tr key={checkpoint.id}>
                      <td>
                        {checkpoint.label} {position === 0 && <StatusPill tone="ok">head</StatusPill>}
                      </td>
                      <td className="mono tiny">{shortHash(checkpoint.sha256)}</td>
                      <td>{checkpoint.origin}</td>
                      <td className="tiny faint">{checkpoint.createdAt.slice(0, 19).replace("T", " ")}</td>
                      <td>
                        <button type="button" className="btn small" onClick={() => void rollback(checkpoint)}>
                          {t("weights.rollback")}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

async function digest(text: string): Promise<string> {
  const data = new TextEncoder().encode(text);
  if (crypto?.subtle) {
    const hash = await crypto.subtle.digest("SHA-256", data);
    return Array.from(new Uint8Array(hash))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }
  let value = 0;
  for (const byte of data) value = (value * 31 + byte) >>> 0;
  return `weak-${value.toString(16)}`;
}
