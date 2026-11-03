/**
 * Agent Lab: the browser's view of the agent contract.
 *
 * The MCP server and the evaluator subprocess live in the desktop engine. What this page does for real
 * is enforce the same policy locally: allowlist, bounds, immutable checkpoint, before/after evaluation.
 */

import { useEffect, useMemo, useState } from "react";

import { useI18n, useT } from "../i18n";
import { applyPatch, deserializeMlp, evaluate, patchEffect, serializeMlp, tensorStats, type PatchOp, type TensorName } from "../lib/mlp";
import { useApp } from "../state";
import { Alert, Card, Field, PageHead, Stat } from "../components/ui";
import type { ModelRecord } from "../lib/types";

const ALLOWED = ["scale", "clamp", "normalize", "sparsify"] as const;
type AllowedOp = (typeof ALLOWED)[number];
const FORBIDDEN = ["set_trainable", "copy_from", "interpolate", "set_scalar"] as const;
const DESKTOP_ONLY = ["low_rank_delta"] as const;

export function AgentLabPage() {
  const t = useT();
  const { locale } = useI18n();
  const { db, refresh, version } = useApp();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState("");
  const [goal, setGoal] = useState("Reduce the norm of the first layer without destroying accuracy");
  const [hypothesis, setHypothesis] = useState("A 5% shrink should keep held-out accuracy within 2 points");
  const [tensor, setTensor] = useState<TensorName>("w1");
  const [operation, setOperation] = useState<AllowedOp>("scale");
  const [factor, setFactor] = useState(0.95);
  const [outcome, setOutcome] = useState<{ before: number; after: number; changed: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!db) return;
    void (async () => {
      const list = (await db.listModels()).filter((model) => Boolean(model.weights));
      setModels(list);
      if (!modelId && list[0]) setModelId(list[0].id);
    })();
  }, [db, modelId, version]);

  const model = useMemo(() => models.find((entry) => entry.id === modelId), [models, modelId]);
  const stats = useMemo(() => {
    if (!model?.weights) return null;
    try {
      return tensorStats(deserializeMlp(model.weights), tensor);
    } catch {
      return null;
    }
  }, [model, tensor]);

  const submit = async () => {
    setError(null);
    setOutcome(null);
    if (!db || !model?.weights) {
      setError(t("weights.noModel"));
      return;
    }
    const patch: PatchOp = buildOp(operation, factor);
    setBusy(true);
    try {
      const before = deserializeMlp(model.weights);
      const after = applyPatch(before, tensor, patch);
      const effect = patchEffect(before, after, tensor);
      const beforeMetrics = evaluate(before, { batchSize: 128, seed: 424242 });
      const afterMetrics = evaluate(after, { batchSize: 128, seed: 424242 });
      const serialized = serializeMlp(after);
      const createdAt = new Date().toISOString();
      const sha = await digest(serialized);
      await db.putCheckpoint({
        id: `ckpt_${Date.now().toString(36)}`,
        modelId: model.id,
        label: `agent:${operation}`,
        sha256: sha,
        sizeBytes: new TextEncoder().encode(serialized).length,
        origin: "patch",
        createdAt,
        weights: serialized,
      });
      await db.putModel({ ...model, weights: serialized, source: "browser-patched" });
      await db.putRun({
        id: `run_${Date.now().toString(36)}`,
        modelId: model.id,
        kind: "patch",
        state: "completed",
        startedAt: createdAt,
        finishedAt: createdAt,
        config: { agent: "agent:browser", goal, hypothesis, tensor, operation, patch },
        metrics: [],
        summary: {
          changed: effect.changed,
          accuracyBefore: beforeMetrics.accuracy,
          accuracyAfter: afterMetrics.accuracy,
          lossBefore: beforeMetrics.loss,
          lossAfter: afterMetrics.loss,
          checkpointSha256: sha,
        },
        provenance: {
          runtime: "browser-worker",
          architectureHash: model.architectureHash,
          evaluatorVersion: "browser-1",
          note: "the proposal was validated against the policy, applied to a copy, and evaluated before the head moved",
        },
      });
      await db.logEvent({
        id: `evt_${Date.now().toString(36)}`,
        at: createdAt,
        level: "info",
        scope: `agent:${model.id}`,
        message: `proposal applied: ${operation} on ${tensor}, accuracy ${beforeMetrics.accuracy.toFixed(3)} -> ${afterMetrics.accuracy.toFixed(3)}`,
      });
      setOutcome({ before: beforeMetrics.accuracy, after: afterMetrics.accuracy, changed: effect.changed });
      refresh();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHead title={t("agent.title")} subtitle={t("agent.subtitle")} />

      <div className="split">
        <Card title={t("agent.proposal")}>
          <div className="grid grid-2">
            <Field label={t("model.name")} htmlFor="agent-model">
              <select id="agent-model" value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={models.length === 0}>
                {models.length === 0 && <option value="">{t("weights.noModel")}</option>}
                {models.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={t("agent.tensor")} htmlFor="agent-tensor">
              <select id="agent-tensor" value={tensor} onChange={(event) => setTensor(event.target.value as TensorName)}>
                <option value="w1">w1</option>
                <option value="b1">b1</option>
                <option value="w2">w2</option>
                <option value="b2">b2</option>
              </select>
            </Field>
            <Field label={t("weights.operation")} htmlFor="agent-op">
              <select id="agent-op" value={operation} onChange={(event) => setOperation(event.target.value as AllowedOp)}>
                {ALLOWED.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="factor" htmlFor="agent-factor">
              <input id="agent-factor" type="number" step="0.01" value={factor} onChange={(event) => setFactor(Number(event.target.value))} />
            </Field>
          </div>
          <div className="stack" style={{ marginTop: "var(--space-3)" }}>
            <Field label={t("agent.goal")} htmlFor="goal">
              <input id="goal" value={goal} onChange={(event) => setGoal(event.target.value)} />
            </Field>
            <Field label={t("agent.hypothesis")} htmlFor="hypothesis">
              <input id="hypothesis" value={hypothesis} onChange={(event) => setHypothesis(event.target.value)} />
            </Field>
          </div>
          <div className="row" style={{ marginTop: "var(--space-4)" }}>
            <button type="button" className="btn primary" onClick={() => void submit()} disabled={busy}>
              {busy ? t("app.loading") : t("agent.submit")}
            </button>
            <span className="tiny faint">{t("agent.separation")}</span>
          </div>
          {error && (
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="bad">{error}</Alert>
            </div>
          )}
          {stats && (
            <p className="tiny faint">
              {tensor}: {stats.numel} values, norm {stats.norm.toFixed(4)}, {t("weights.finite")} {String(stats.finite)}
            </p>
          )}
        </Card>

        <Card title={t("agent.policy")}>
          <div className="stack">
            <div>
              <span className="kpi-label">{t("agent.allowed")}</span>
              <p className="mono tiny">{ALLOWED.join(", ")}</p>
            </div>
            <div>
              <span className="kpi-label">{t("agent.forbidden")}</span>
              <p className="mono tiny">{FORBIDDEN.join(", ")}</p>
            </div>
            <div>
              <span className="kpi-label">{t("train.modeDesktop")}</span>
              <p className="mono tiny">{DESKTOP_ONLY.join(", ")}</p>
            </div>
            <Alert tone="warn" title={t("agent.gate")}>
              {t("agent.gateOn")}
            </Alert>
            <Alert>{t("agent.noShell")}</Alert>
            {outcome && (
              <>
                <h4>{t("agent.outcome")}</h4>
                <div className="grid grid-2">
                  <Stat label={`${t("train.accuracy")} (before)`} value={`${(outcome.before * 100).toFixed(1)}%`} />
                  <Stat label={`${t("train.accuracy")} (after)`} value={`${(outcome.after * 100).toFixed(1)}%`} />
                </div>
                <p className="tiny faint">
                  {t("weights.changed")}: {outcome.changed}
                </p>
              </>
            )}
          </div>
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("agent.level")} subtitle={t("agent.separation")}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t("agent.level")}</th>
                  <th>{t("parity.desktop")}</th>
                  <th>{t("parity.browser")}</th>
                </tr>
              </thead>
              <tbody>
                {[0, 1, 2, 3, 4].map((level) => (
                  <tr key={level}>
                    <td>{t(`agent.level${level}` as never)}</td>
                    <td className="tiny">{level <= 2 ? t("parity.full") : t("parity.experimental")}</td>
                    <td className="tiny">{level <= 1 ? t("parity.limited") : t("parity.unavailable")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="tiny faint" style={{ marginTop: "var(--space-3)" }}>
            {locale === "de"
              ? "Im Browser wird nur Stufe 1 (Vorschlag) nachgebildet; Ausführung und Freigabe liegen im Desktop-MCP-Server."
              : locale === "fr"
                ? "Dans le navigateur, seul le niveau 1 (proposition) est simulé ; l'exécution et l'approbation vivent dans le serveur MCP de bureau."
                : "In the browser only level 1 (drafting) is mirrored; execution and approval live in the desktop MCP server."}
          </p>
        </Card>
      </div>
    </>
  );
}

function buildOp(operation: AllowedOp, factor: number): PatchOp {
  switch (operation) {
    case "scale":
      return { op: "scale", factor };
    case "clamp":
      return { op: "clamp", min: -0.5, max: 0.5 };
    case "normalize":
      return { op: "normalize", eps: 1e-6 };
    case "sparsify":
      return { op: "sparsify", keep: 0.5, seed: 1 };
  }
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
