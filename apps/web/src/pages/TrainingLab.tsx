/** Training Lab: a real SGD run in a Web Worker, persisted with provenance. */

import { useEffect, useMemo, useRef, useState } from "react";

import { useI18n, useT } from "../i18n";
import { formatDuration, formatNumber } from "../lib/format";
import type { MlpConfig } from "../lib/mlp";
import { useApp } from "../state";
import { Alert, Card, Field, LineChart, PageHead, Stat, StatusPill } from "../components/ui";
import type { MetricPoint, ModelRecord } from "../lib/types";

type State = "idle" | "running" | "done" | "stopped" | "error";

interface WorkerMessage {
  type: "progress" | "done" | "error";
  step?: number;
  loss?: number;
  elapsedMs?: number;
  result?: { steps: number; initialLoss: number; finalLoss: number; valLoss: number; accuracy: number };
  weights?: string;
  stopped?: boolean;
  message?: string;
}

export function TrainingLabPage() {
  const t = useT();
  const { locale } = useI18n();
  const { db, refresh } = useApp();
  const workerRef = useRef<Worker | null>(null);
  const [state, setState] = useState<State>("idle");
  const [hidden, setHidden] = useState(24);
  const [steps, setSteps] = useState(200);
  const [batchSize, setBatchSize] = useState(16);
  const [lr, setLr] = useState(0.05);
  const [seed, setSeed] = useState(7);
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [result, setResult] = useState<{ finalLoss: number; valLoss: number; accuracy: number; steps: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedModelId, setSavedModelId] = useState<string | null>(null);

  const config: MlpConfig = useMemo(
    () => ({ inputDim: 16, hidden, outputDim: 4, seed }),
    [hidden, seed],
  );

  useEffect(() => () => workerRef.current?.terminate(), []);

  const start = () => {
    setError(null);
    setResult(null);
    setMetrics([]);
    setSavedModelId(null);
    setState("running");
    const worker = new Worker(new URL("../workers/mlp.worker.ts", import.meta.url), { type: "module" });
    workerRef.current?.terminate();
    workerRef.current = worker;
    const startedAt = performance.now();
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === "progress" && message.step !== undefined && message.loss !== undefined) {
        setMetrics((current) => [
          ...current,
          { step: message.step!, loss: message.loss!, lr, elapsedMs: message.elapsedMs ?? performance.now() - startedAt },
        ]);
      }
      if (message.type === "error") {
        setError(message.message ?? "worker failed");
        setState("error");
        worker.terminate();
      }
      if (message.type === "done" && message.result && message.weights) {
        const final = message.result;
        const trainLoss = final.finalLoss;
        setMetrics((current) =>
          current.length
            ? [...current.slice(0, -1), { ...current[current.length - 1]!, loss: trainLoss }]
            : current,
        );
        setResult({
          finalLoss: trainLoss,
          valLoss: final.valLoss,
          accuracy: final.accuracy,
          steps: final.steps,
        });
        setState(message.stopped ? "stopped" : "done");
        worker.terminate();
        void persist(final, message.weights, message.elapsedMs ?? 0, startedAt);
      }
    };
    worker.postMessage({ type: "start", config, steps, batchSize, lr, seed });
  };

  const persist = async (
    final: { steps: number; initialLoss: number; finalLoss: number; valLoss: number; accuracy: number },
    weights: string,
    elapsedMs: number,
    startedAt: number,
  ) => {
    if (!db) return;
    const createdAt = new Date().toISOString();
    const id = `browser-mlp-${Date.now().toString(36)}`;
    const sizeBytes = new TextEncoder().encode(weights).length;
    const sha = await digest(weights);
    const model: ModelRecord = {
      id,
      name: `Browser MLP ${hidden}×16`,
      family: "mlp",
      config: { ...config, seed },
      parameterCount: config.inputDim * hidden + hidden + hidden * config.outputDim + config.outputDim,
      architectureHash: await digest(JSON.stringify(config)),
      createdAt,
      notes: "Trained in this browser with SGD. Weights are stored locally and are editable in the Weight Studio.",
      source: "browser-trained",
      origin: "created",
      weights,
      maturity: "stable",
    };
    await db.putModel(model);
    await db.putCheckpoint({
      id: `ckpt_${Date.now().toString(36)}`,
      modelId: id,
      label: "final",
      sha256: sha,
      sizeBytes,
      origin: "training",
      createdAt,
      weights,
    });
    await db.putRun({
      id: `run_${Date.now().toString(36)}`,
      modelId: id,
      kind: "training",
      state: "completed",
      startedAt: new Date(startedAt).toISOString(),
      finishedAt: createdAt,
      config: { steps, batchSize, lr, seed, hidden, inputDim: 16, outputDim: 4 },
      metrics,
      summary: { ...final, elapsedMs, parameters: model.parameterCount, bytes: sizeBytes },
      provenance: {
        runtime: "browser-worker",
        seed,
        architectureHash: model.architectureHash,
        note: "SGD with momentum on a synthetic classification task; measures pipeline correctness, not capability",
      },
    });
    await db.logEvent({
      id: `evt_${Date.now().toString(36)}`,
      at: createdAt,
      level: "info",
      scope: `training:${id}`,
      message: `browser training completed with val_loss ${final.valLoss.toFixed(4)}`,
    });
    setSavedModelId(id);
    refresh();
  };

  return (
    <>
      <PageHead title={t("train.title")} subtitle={t("train.subtitle")} />

      <div className="split">
        <Card title={t("train.title")}>
          <div className="grid grid-3">
            <Field label={t("train.hidden")} htmlFor="hidden">
              <input id="hidden" type="number" min={4} max={128} value={hidden} onChange={(event) => setHidden(Number(event.target.value))} />
            </Field>
            <Field label={t("train.steps")} htmlFor="steps">
              <input id="steps" type="number" min={1} max={5000} value={steps} onChange={(event) => setSteps(Number(event.target.value))} />
            </Field>
            <Field label={t("train.batch")} htmlFor="batch">
              <input id="batch" type="number" min={1} max={256} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
            </Field>
            <Field label={t("train.lr")} htmlFor="lr">
              <input id="lr" type="number" step="0.001" min={0.0001} max={1} value={lr} onChange={(event) => setLr(Number(event.target.value))} />
            </Field>
            <Field label={t("eval.seed")} htmlFor="seed">
              <input id="seed" type="number" min={0} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </Field>
            <Field label={t("train.status")}>
              <div>
                <StatusPill tone={state === "done" ? "ok" : state === "error" ? "bad" : state === "running" ? "warn" : "neutral"}>
                  {state === "done" ? t("train.done") : state === "running" ? t("train.running") : state === "error" ? t("history.stateFailed") : state === "stopped" ? t("history.stateCancelled") : t("train.idle")}
                </StatusPill>
              </div>
            </Field>
          </div>
          <div className="row" style={{ marginTop: "var(--space-4)" }}>
            <button type="button" className="btn primary" onClick={start} disabled={state === "running"}>
              {t("train.start")}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => workerRef.current?.postMessage({ type: "stop" })}
              disabled={state !== "running"}
            >
              {t("train.stop")}
            </button>
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <Alert tone="warn" title={t("train.modeBrowser")}>
              {t("train.limitation")}
            </Alert>
          </div>
          {error && (
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="bad">{error}</Alert>
            </div>
          )}
          {savedModelId && (
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="ok">
                {t("train.provenance")} <code className="inline">{savedModelId}</code>
              </Alert>
            </div>
          )}
        </Card>

        <Card title={t("eval.results")}>
          <div className="grid grid-2">
            <Stat label={t("train.loss")} value={result ? result.finalLoss.toFixed(4) : "—"} hint={t("common.measured")} />
            <Stat label={t("train.valLoss")} value={result ? result.valLoss.toFixed(4) : "—"} hint={t("common.measured")} />
            <Stat label={t("train.accuracy")} value={result ? `${(result.accuracy * 100).toFixed(1)}%` : "—"} hint={`${t("common.baseline")} 25%`} />
            <Stat label={t("train.step")} value={result ? formatNumber(result.steps, locale) : formatNumber(metrics.length, locale)} />
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <LineChart
              points={metrics.map((point) => ({ x: point.step, y: point.loss }))}
              labels={{ x: t("train.step"), y: t("train.loss") }}
              title={t("train.loss")}
            />
          </div>
          {metrics.length > 0 && (
            <p className="tiny faint">
              {t("train.elapsed")}: {formatDuration(metrics[metrics.length - 1]!.elapsedMs, locale)} ·{" "}
              {formatNumber(metrics.length, locale)} {t("common.measured")}
            </p>
          )}
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("train.modeDesktop")} subtitle={t("train.provenance")}>
          <pre className="code">{`hcs-foundry model create --name "My LM" --params 250k
hcs-foundry train --model my-lm --steps 400 --batch-size 16 --block-size 32
hcs-foundry eval --model my-lm --suite text-smoke
hcs-foundry export --model my-lm --format onnx`}</pre>
          <div style={{ marginTop: "var(--space-3)" }}>
            <Alert>
              {t("parity.note")} {t("common.notAvailable")}: PyTorch training, LoRA, evolution and ONNX export only exist in
              the desktop engine.
            </Alert>
          </div>
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
