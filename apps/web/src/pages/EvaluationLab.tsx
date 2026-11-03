/** Evaluation Lab: run a real evaluation on a browser model, or read the shipped desktop measurements. */

import { useEffect, useMemo, useState } from "react";

import { useI18n, useT } from "../i18n";
import { formatNumber } from "../lib/format";
import { deserializeMlp, evaluate } from "../lib/mlp";
import { useApp } from "../state";
import { Alert, Card, Field, PageHead, Stat, StatusPill } from "../components/ui";
import type { ModelRecord, RunRecord } from "../lib/types";

interface Result {
  suite: string;
  loss: number;
  accuracy: number;
  examples: number;
  seed: number;
  definitions: { name: string; value: number; definition: string; baseline?: number }[];
}

export function EvaluationLabPage() {
  const t = useT();
  const { locale } = useI18n();
  const { db, demo, version } = useApp();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState("");
  const [batches, setBatches] = useState(8);
  const [seed, setSeed] = useState(1234);
  const [result, setResult] = useState<Result | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!db) return;
    void (async () => {
      const list = (await db.listModels()).filter((model) => Boolean(model.weights));
      setModels(list);
      if (!modelId && list[0]) setModelId(list[0].id);
      const all = await db.listRuns();
      setRuns(all.filter((run) => run.kind === "evaluation" || run.kind === "training").slice(0, 8));
    })();
  }, [db, modelId, version]);

  const model = useMemo(() => models.find((entry) => entry.id === modelId), [models, modelId]);

  const runEvaluation = async () => {
    setError(null);
    if (!db || !model?.weights) {
      setError(t("weights.noModel"));
      return;
    }
    try {
      const mlp = deserializeMlp(model.weights);
      const examples = batches * 64;
      let lossSum = 0;
      let correct = 0;
      let total = 0;
      const perBatch: { batch: number; loss: number; accuracy: number }[] = [];
      for (let batch = 0; batch < batches; batch += 1) {
        const metrics = evaluate(mlp, { batchSize: 64, seed: seed + batch * 17 });
        lossSum += metrics.loss;
        correct += metrics.accuracy * 64;
        total += 64;
        perBatch.push({ batch, loss: metrics.loss, accuracy: metrics.accuracy });
      }
      const loss = lossSum / batches;
      const accuracy = correct / total;
      const evaluation: Result = {
        suite: "browser-mlp-smoke",
        loss,
        accuracy,
        examples,
        seed,
        definitions: [
          {
            name: "val_loss",
            value: loss,
            definition: "mean cross-entropy over held-out batches of the synthetic classification task",
            baseline: Math.log(mlp.config.outputDim),
          },
          {
            name: "val_accuracy",
            value: accuracy,
            definition: "held-out top-1 accuracy on the same batches",
            baseline: 1 / mlp.config.outputDim,
          },
          { name: "examples_evaluated", value: examples, definition: "number of held-out examples (64 per batch)" },
        ],
      };
      setResult(evaluation);
      const createdAt = new Date().toISOString();
      await db.putRun({
        id: `run_${Date.now().toString(36)}`,
        modelId: model.id,
        kind: "evaluation",
        state: "completed",
        startedAt: createdAt,
        finishedAt: createdAt,
        config: { suite: evaluation.suite, batches, seed },
        metrics: [],
        summary: { ...evaluation, perBatch },
        provenance: {
          runtime: "browser-worker",
          seed,
          evaluatorVersion: "browser-1",
          architectureHash: model.architectureHash,
          note: "evaluated in the browser against freshly generated held-out batches",
        },
      });
      await db.logEvent({
        id: `evt_${Date.now().toString(36)}`,
        at: createdAt,
        level: "info",
        scope: `evaluation:${model.id}`,
        message: `browser evaluation val_loss=${loss.toFixed(4)} accuracy=${accuracy.toFixed(3)}`,
      });
    } catch (cause) {
      setError((cause as Error).message);
    }
  };

  return (
    <>
      <PageHead title={t("eval.title")} subtitle={t("eval.subtitle")} />

      <div className="split">
        <Card title={t("eval.suite")}>
          <div className="grid grid-3">
            <Field label={t("model.name")} htmlFor="eval-model">
              <select id="eval-model" value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={models.length === 0}>
                {models.length === 0 && <option value="">{t("weights.noModel")}</option>}
                {models.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={t("eval.batches")} htmlFor="eval-batches">
              <input id="eval-batches" type="number" min={1} max={64} value={batches} onChange={(event) => setBatches(Number(event.target.value))} />
            </Field>
            <Field label={t("eval.seed")} htmlFor="eval-seed">
              <input id="eval-seed" type="number" min={0} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </Field>
          </div>
          <div className="row" style={{ marginTop: "var(--space-4)" }}>
            <button type="button" className="btn primary" onClick={() => void runEvaluation()}>
              {t("eval.run")}
            </button>
            <StatusPill tone="neutral">browser-mlp-smoke v1</StatusPill>
          </div>
          {error && (
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="bad">{error}</Alert>
            </div>
          )}
          <div style={{ marginTop: "var(--space-4)" }}>
            <Alert>{t("eval.baselines")}</Alert>
          </div>
        </Card>

        <Card title={t("eval.results")}>
          {result ? (
            <div className="stack">
              <div className="grid grid-2">
                <Stat label={t("train.valLoss")} value={result.loss.toFixed(4)} />
                <Stat label={t("train.accuracy")} value={`${(result.accuracy * 100).toFixed(1)}%`} hint={`${t("common.baseline")} ${(100 / 4).toFixed(0)}%`} />
                <Stat label="examples" value={formatNumber(result.examples, locale)} />
                <Stat label={t("eval.seed")} value={formatNumber(result.seed, locale)} />
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>{t("eval.metric")}</th>
                      <th className="num">{t("eval.value")}</th>
                      <th>{t("eval.definition")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.definitions.map((metric) => (
                      <tr key={metric.name}>
                        <td className="mono tiny">{metric.name}</td>
                        <td className="num">
                          {metric.value.toFixed(4)}
                          {metric.baseline !== undefined && <span className="tiny faint"> ({t("common.baseline")} {metric.baseline.toFixed(3)})</span>}
                        </td>
                        <td className="tiny">{metric.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <p className="muted">{t("eval.noRuns")}</p>
          )}
        </Card>
      </div>

      {demo && (
        <div style={{ marginTop: "var(--space-4)" }}>
          <Card title={`${demo.name} · ${t("eval.results")}`} subtitle={t("common.measured")}>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("eval.metric")}</th>
                    <th className="num">{t("eval.value")}</th>
                    <th>{t("eval.definition")}</th>
                  </tr>
                </thead>
                <tbody>
                  {demo.evaluation.metrics.map((metric) => (
                    <tr key={metric.name}>
                      <td className="mono tiny">{metric.name}</td>
                      <td className="num">{metric.value.toFixed(4)}</td>
                      <td className="tiny">{metric.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="warn">
                {t("gen.boundary")}: {demo.claim}
              </Alert>
            </div>
          </Card>
        </div>
      )}

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("eval.compare")}>
          {runs.length === 0 ? (
            <p className="muted">{t("history.empty")}</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("history.kind")}</th>
                    <th>{t("history.state")}</th>
                    <th className="num">{t("train.loss")}</th>
                    <th className="num">{t("train.accuracy")}</th>
                    <th>{t("history.when")}</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr key={run.id}>
                      <td>{run.kind}</td>
                      <td>
                        <StatusPill tone={run.state === "completed" ? "ok" : run.state === "invalid" ? "bad" : "warn"}>{run.state}</StatusPill>
                      </td>
                      <td className="num">
                        {typeof run.summary.loss === "number"
                          ? (run.summary.loss as number).toFixed(4)
                          : typeof run.summary.valLoss === "number"
                            ? (run.summary.valLoss as number).toFixed(4)
                            : "—"}
                      </td>
                      <td className="num">
                        {typeof run.summary.accuracy === "number" ? `${((run.summary.accuracy as number) * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="tiny faint">{run.startedAt.slice(0, 19).replace("T", " ")}</td>
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
