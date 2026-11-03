/** Model Studio: budget solving and architecture preview computed locally in the browser. */

import { useMemo, useState } from "react";

import { useI18n, useT } from "../i18n";
import { countTransformerParams, createMlp, serializeMlp, type TransformerPreview } from "../lib/mlp";
import { formatNumber, shortHash, weakHash } from "../lib/format";
import { useApp } from "../state";
import { Alert, Badge, Card, Field, PageHead, Stat } from "../components/ui";
import type { ModelRecord } from "../lib/types";

interface Candidate {
  config: Record<string, unknown>;
  params: number;
  error: number;
  errorPct: number;
}

function solveTransformer(target: number, policy: string): Candidate[] {
  const candidates: Candidate[] = [];
  for (let layers = 1; layers <= 8; layers += 1) {
    for (let dModel = 32; dModel <= 512; dModel += 32) {
      for (const heads of [2, 4, 8]) {
        if (dModel % heads !== 0 || (dModel / heads) % 2 !== 0) continue;
        for (const multiplier of [2.0, 2.667, 4.0]) {
          const ffnDim = Math.max(64, Math.round((multiplier * dModel) / 64) * 64);
          const config: TransformerPreview = {
            vocabSize: 64,
            dModel,
            layers,
            heads,
            ffnDim,
            tieEmbeddings: true,
            // The Python solver pairs every head count with n_heads/2 KV heads.
            nKvHeads: Math.max(1, Math.floor(heads / 2)),
          };
          const params = countTransformerParams(config);
          candidates.push({ config: { family: "transformer-lm", ...config, contextLength: 64 }, params, error: params - target, errorPct: (Math.abs(params - target) / target) * 100 });
        }
      }
    }
  }
  return rank(candidates, policy);
}

function solveMlp(target: number, policy: string): Candidate[] {
  const candidates: Candidate[] = [];
  for (let depth = 1; depth <= 3; depth += 1) {
    for (let hidden = 16; hidden <= 1024; hidden += 16) {
      const hiddenDims = Array.from({ length: depth }, () => hidden);
      const params = countMlpParamsDims(16, hiddenDims, 4);
      candidates.push({
        config: {
          family: "mlp",
          inputDim: 16,
          hiddenDims,
          outputDim: 4,
          activation: "gelu",
          seed: 0,
        },
        params,
        error: params - target,
        errorPct: (Math.abs(params - target) / target) * 100,
      });
    }
  }
  return rank(candidates, policy);
}

/** Exact count for an MLP with an arbitrary list of hidden widths (matches the Python formula). */
export function countMlpParamsDims(inputDim: number, hiddenDims: number[], outputDim: number): number {
  const widths = [inputDim, ...hiddenDims, outputDim];
  let total = 0;
  for (let i = 0; i < widths.length - 1; i += 1) total += widths[i]! * widths[i + 1]! + widths[i + 1]!;
  return total;
}

function rank(candidates: Candidate[], policy: string): Candidate[] {
  const filtered = candidates.filter((candidate) => {
    if (policy === "exact") return candidate.error === 0;
    if (policy === "closest_under") return candidate.error <= 0;
    if (policy === "at_least") return candidate.error >= 0;
    return true;
  });
  const pool = filtered.length ? filtered : candidates;
  const sorted = [...pool].sort((a, b) => Math.abs(a.error) - Math.abs(b.error));
  const seen = new Set<number>();
  const distinct: Candidate[] = [];
  for (const candidate of sorted) {
    if (seen.has(candidate.params)) continue;
    seen.add(candidate.params);
    distinct.push(candidate);
    if (distinct.length >= 5) break;
  }
  return distinct;
}

export function ModelStudioPage() {
  const t = useT();
  const { locale } = useI18n();
  const { db, refresh } = useApp();
  const [family, setFamily] = useState<"transformer-lm" | "mlp">("transformer-lm");
  const [budget, setBudget] = useState("250k");
  const [policy, setPolicy] = useState("closest");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [name, setName] = useState("Browser model");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const parsedBudget = useMemo(() => {
    const text = budget.trim().toLowerCase().replace(/[_,]/g, "");
    const match = /^(\d+(?:\.\d+)?)([kmb])?$/.exec(text);
    if (!match) return null;
    const value = Number(match[1]);
    const suffix = match[2];
    const factor = suffix === "k" ? 1e3 : suffix === "m" ? 1e6 : suffix === "b" ? 1e9 : 1;
    return Math.round(value * factor);
  }, [budget]);

  const solve = () => {
    setError(null);
    if (!parsedBudget) {
      setError(t("model.budgetHelp"));
      setCandidates([]);
      return;
    }
    const result = family === "transformer-lm" ? solveTransformer(parsedBudget, policy) : solveMlp(parsedBudget, policy);
    setCandidates(result);
    setSelected(result[0] ?? null);
  };

  const estimate = useMemo(() => {
    if (!selected) return null;
    if (family === "transformer-lm") {
      const config = selected.config as unknown as TransformerPreview & { contextLength: number };
      const bytes = selected.params * 4;
      const kv = 2 * config.layers * config.contextLength * Math.max(1, Math.floor(config.heads / 2)) * (config.dModel / config.heads) * 4;
      return {
        weights: bytes,
        gradients: bytes,
        optimizer: bytes * 3,
        activations: bytes * 0.6,
        kv,
        totalTrain: bytes * 4 + bytes * 0.6 + kv,
      };
    }
    const bytes = selected.params * 4;
    return { weights: bytes, gradients: bytes, optimizer: bytes * 3, activations: bytes * 0.3, kv: 0, totalTrain: bytes * 4.3 };
  }, [selected, family]);

  const create = async () => {
    if (!db || !selected) return;
    setError(null);
    try {
      const id = `${name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "model"}-${Date.now().toString(36)}`;
      const createdAt = new Date().toISOString();
      const architectureHash = await hashCode(JSON.stringify(selected.config));
      const mlp = family === "mlp" ? createMlp({ inputDim: 16, hidden: (selected.config.hiddenDims as number[])?.[0] ?? 32, outputDim: 4, seed: 0 }) : null;
      const model: ModelRecord = {
        id,
        name,
        family,
        config: selected.config,
        parameterCount: selected.params,
        architectureHash,
        createdAt,
        notes:
          family === "mlp"
            ? "Created in the browser: weights are real and editable."
            : "Created in the browser: architecture definition only. Weight tensors require the desktop engine.",
        source: family === "mlp" ? "browser-trained" : "definition",
        origin: "created",
        weights: mlp ? serializeMlp(mlp) : undefined,
        maturity: "stable",
      };
      await db.putModel(model);
      if (mlp) {
        await db.putCheckpoint({
          id: `ckpt_${Date.now().toString(36)}`,
          modelId: id,
          label: "init",
          sha256: await hashCode(serializeMlp(mlp)),
          sizeBytes: serializeMlp(mlp).length,
          origin: "init",
          createdAt,
          weights: serializeMlp(mlp),
        });
      }
      await db.logEvent({
        id: `evt_${Date.now().toString(36)}`,
        at: createdAt,
        level: "info",
        scope: `model:${id}`,
        message: `created ${family} with ${selected.params} parameters`,
      });
      refresh();
      setMessage(`${t("model.created")}: ${name}`);
    } catch (cause) {
      setError((cause as Error).message);
    }
  };

  return (
    <>
      <PageHead title={t("model.title")} subtitle={t("model.subtitle")} />
      <div className="split">
        <Card title={t("model.budget")}>
          <div className="grid grid-2">
            <Field label={t("model.family")} htmlFor="family">
              <select id="family" value={family} onChange={(event) => setFamily(event.target.value as typeof family)}>
                <option value="transformer-lm">{t("model.familyTransformer")}</option>
                <option value="mlp">{t("model.familyMlp")}</option>
              </select>
            </Field>
            <Field label={t("model.budget")} hint={t("model.budgetHelp")} htmlFor="budget">
              <input id="budget" value={budget} onChange={(event) => setBudget(event.target.value)} inputMode="text" />
            </Field>
            <Field label={t("model.policy")} htmlFor="policy">
              <select id="policy" value={policy} onChange={(event) => setPolicy(event.target.value)}>
                <option value="exact">{t("model.policyExact")}</option>
                <option value="closest">{t("model.policyClosest")}</option>
                <option value="closest_under">{t("model.policyUnder")}</option>
                <option value="at_least">{t("model.policyAtLeast")}</option>
              </select>
            </Field>
            <Field label={t("model.name")} htmlFor="name">
              <input id="name" value={name} onChange={(event) => setName(event.target.value)} />
            </Field>
          </div>
          <div className="row" style={{ marginTop: "var(--space-4)" }}>
            <button type="button" className="btn primary" onClick={solve}>
              {t("model.solve")}
            </button>
            <button type="button" className="btn" onClick={() => void create()} disabled={!selected || !db}>
              {t("model.create")}
            </button>
          </div>
          {error && (
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="bad">{error}</Alert>
            </div>
          )}
          {message && (
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="ok">{message}</Alert>
            </div>
          )}
          <div style={{ marginTop: "var(--space-4)" }}>
            <Alert tone="warn">{t("model.exactness")}</Alert>
          </div>
        </Card>

        <Card title={t("model.estimate")}>
          {estimate && selected ? (
            <div className="stack">
              <Stat label={t("model.actual")} value={formatNumber(selected.params, locale)} hint={`${t("model.error")}: ${selected.error >= 0 ? "+" : ""}${formatNumber(selected.error, locale)}`} />
              <table>
                <tbody>
                  <tr>
                    <td>{t("model.weights")}</td>
                    <td className="num">{formatNumber(estimate.weights / 1e6, locale, 2)} MB</td>
                  </tr>
                  <tr>
                    <td>{t("model.gradients")}</td>
                    <td className="num">{formatNumber(estimate.gradients / 1e6, locale, 2)} MB</td>
                  </tr>
                  <tr>
                    <td>{t("model.optimizer")}</td>
                    <td className="num">{formatNumber(estimate.optimizer / 1e6, locale, 2)} MB</td>
                  </tr>
                  <tr>
                    <td>{t("model.activations")}</td>
                    <td className="num">{formatNumber(estimate.activations / 1e6, locale, 2)} MB</td>
                  </tr>
                  <tr>
                    <td>{t("model.kvCache")}</td>
                    <td className="num">{formatNumber(estimate.kv / 1e6, locale, 2)} MB</td>
                  </tr>
                  <tr>
                    <td>
                      <strong>{t("model.totalTrain")}</strong>
                    </td>
                    <td className="num">
                      <strong>{formatNumber(estimate.totalTrain / 1e6, locale, 1)} MB</strong>
                    </td>
                  </tr>
                </tbody>
              </table>
              <small className="faint">
                {t("common.estimated")}: fixed multipliers, not a traced graph. Batch 8, block 32.
              </small>
            </div>
          ) : (
            <p className="muted">{t("model.noSelection")}</p>
          )}
        </Card>
      </div>

      {candidates.length > 0 && (
        <div style={{ marginTop: "var(--space-4)" }}>
          <Card title={t("model.candidates")} subtitle={t("model.parameterBreakdown")}>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("model.architecture")}</th>
                    <th className="num">{t("model.actual")}</th>
                    <th className="num">{t("model.error")}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((candidate) => (
                    <tr key={candidate.params + JSON.stringify(candidate.config)}>
                      <td className="mono tiny">{JSON.stringify(candidate.config)}</td>
                      <td className="num">{formatNumber(candidate.params, locale)}</td>
                      <td className="num">
                        {candidate.error >= 0 ? "+" : ""}
                        {formatNumber(candidate.error, locale)} ({formatNumber(candidate.errorPct, locale, 2)}%)
                      </td>
                      <td>
                        <button type="button" className="btn small" onClick={() => setSelected(candidate)} aria-pressed={selected?.params === candidate.params}>
                          {t("model.select")}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="row" style={{ marginTop: "var(--space-3)" }}>
              <Badge>{t("common.measured")}: {formatNumber(candidates.length, locale)} designs</Badge>
              {selected && <span className="tiny faint mono">hash {shortHash(shortHash(String(selected.params)))}</span>}
            </div>
          </Card>
        </div>
      )}

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("parity.title")}>
          <p className="muted">{t("parity.note")}</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t("parity.capability")}</th>
                  <th>{t("parity.desktop")}</th>
                  <th>{t("parity.browser")}</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>{t("model.title")}</td>
                  <td><Badge className="stable">{t("parity.full")}</Badge></td>
                  <td><Badge className="experimental">{t("parity.limited")}</Badge></td>
                </tr>
                <tr>
                  <td>{t("weights.title")}</td>
                  <td><Badge className="stable">{t("parity.full")}</Badge></td>
                  <td><Badge className="experimental">{t("parity.limited")}</Badge></td>
                </tr>
                <tr>
                  <td>{t("train.title")}</td>
                  <td><Badge className="stable">{t("parity.full")}</Badge></td>
                  <td><Badge className="unsupported">{t("parity.limited")}</Badge></td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}

async function hashCode(text: string): Promise<string> {
  const data = new TextEncoder().encode(text);
  if (crypto?.subtle) {
    const digest = await crypto.subtle.digest("SHA-256", data);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }
  return weakHash(text);
}
