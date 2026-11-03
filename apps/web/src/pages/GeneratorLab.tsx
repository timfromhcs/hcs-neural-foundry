/**
 * Generator Lab.
 *
 * Honesty constraint: weight-space diffusion and flow matching are NOT implemented here, and the page
 * says so instead of showing a mocked chart. What is real on this page: weight-space interpolation
 * between two checkpoints you trained in this browser, evaluated at each step.
 */

import { useEffect, useMemo, useState } from "react";

import { useI18n, useT } from "../i18n";
import { deserializeMlp, evaluate, interpolateWeights } from "../lib/mlp";
import { useApp } from "../state";
import { Alert, Card, Field, MaturityBadge, PageHead, Stat } from "../components/ui";
import type { ModelRecord } from "../lib/types";

const METHODS = [
  { id: "hyper", labelKey: "gen.methodHyper", maturity: "experimental" as const, implemented: true },
  { id: "diffusion", labelKey: "gen.methodDiffusion", maturity: "unsupported" as const, implemented: false },
  { id: "flow", labelKey: "gen.methodFlow", maturity: "unsupported" as const, implemented: false },
];

export function GeneratorLabPage() {
  const t = useT();
  const { locale } = useI18n();
  const { db, version } = useApp();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [tValue, setTValue] = useState(0.5);
  const [curve, setCurve] = useState<{ t: number; accuracy: number; loss: number }[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!db) return;
    void (async () => {
      const list = (await db.listModels()).filter((model) => Boolean(model.weights));
      setModels(list);
      if (list[0]) setLeftId((current) => current || list[0]!.id);
      if (list[1]) setRightId((current) => current || list[1]!.id);
    })();
  }, [db, version]);

  const compatible = useMemo(() => {
    const left = models.find((model) => model.id === leftId);
    const right = models.find((model) => model.id === rightId);
    if (!left || !right || !left.weights || !right.weights) return false;
    return JSON.stringify(left.config.inputDim) === JSON.stringify(right.config.inputDim) &&
      JSON.stringify(left.config.hidden) === JSON.stringify(right.config.hidden) &&
      JSON.stringify(left.config.outputDim) === JSON.stringify(right.config.outputDim);
  }, [models, leftId, rightId]);

  const interpolate = () => {
    setError(null);
    const left = models.find((model) => model.id === leftId);
    const right = models.find((model) => model.id === rightId);
    if (!left?.weights || !right?.weights) {
      setError(t("weights.noModel"));
      return;
    }
    try {
      const a = deserializeMlp(left.weights);
      const b = deserializeMlp(right.weights);
      const samples: { t: number; accuracy: number; loss: number }[] = [];
      for (const alpha of [0, 0.25, 0.5, 0.75, 1]) {
        const mixed = interpolateWeights(a, b, alpha);
        const metrics = evaluate(mixed, { batchSize: 128, seed: 424242 });
        samples.push({ t: alpha, accuracy: metrics.accuracy, loss: metrics.loss });
      }
      setCurve(samples);
    } catch (cause) {
      setCurve([]);
      setError((cause as Error).message);
    }
  };

  const current = curve.find((point) => Math.abs(point.t - tValue) < 0.01);

  return (
    <>
      <PageHead title={t("gen.title")} subtitle={t("gen.subtitle")} />

      <Card title={t("gen.method")}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("gen.method")}</th>
                <th>{t("maturity.legend")}</th>
                <th>{t("train.status")}</th>
                <th>{t("gen.why")}</th>
              </tr>
            </thead>
            <tbody>
              {METHODS.map((method) => (
                <tr key={method.id}>
                  <td>{t(method.labelKey as never)}</td>
                  <td>
                    <MaturityBadge maturity={method.maturity} />
                  </td>
                  <td className="tiny">{method.implemented ? t("gen.supported") : t("gen.notImplemented")}</td>
                  <td className="tiny muted">{method.implemented ? "—" : t("gen.whyBody")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="split" style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("gen.results")} subtitle={t("gen.holdout")}>
          <div className="grid grid-2">
            <Field label="A" htmlFor="left">
              <select id="left" value={leftId} onChange={(event) => setLeftId(event.target.value)}>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="B" htmlFor="right">
              <select id="right" value={rightId} onChange={(event) => setRightId(event.target.value)}>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <Field label={`t = ${tValue.toFixed(2)}`} htmlFor="t">
              <input id="t" type="range" min={0} max={1} step={0.25} value={tValue} onChange={(event) => setTValue(Number(event.target.value))} />
            </Field>
          </div>
          <div className="row" style={{ marginTop: "var(--space-3)" }}>
            <button type="button" className="btn primary" onClick={interpolate} disabled={models.length < 2 || !compatible}>
              {t("common.run")}
            </button>
            {!compatible && <span className="tiny faint">{t("gen.boundary")}: architectures must match.</span>}
          </div>
          {error && (
            <div style={{ marginTop: "var(--space-3)" }}>
              <Alert tone="bad">{error}</Alert>
            </div>
          )}
          {curve.length > 0 && (
            <div className="table-wrap" style={{ marginTop: "var(--space-4)" }}>
              <table>
                <thead>
                  <tr>
                    <th>t</th>
                    <th className="num">{t("train.loss")}</th>
                    <th className="num">{t("train.accuracy")}</th>
                  </tr>
                </thead>
                <tbody>
                  {curve.map((point) => (
                    <tr key={point.t} aria-current={current?.t === point.t}>
                      <td className="mono">{point.t.toFixed(2)}</td>
                      <td className="num">{point.loss.toFixed(4)}</td>
                      <td className="num">{(point.accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title={t("gen.boundary")}>
          <div className="stack">
            {current ? (
              <>
                <Stat label={`t = ${current.t.toFixed(2)}`} value={`${(current.accuracy * 100).toFixed(1)}%`} hint={t("train.accuracy")} />
                <Stat label={t("train.loss")} value={current.loss.toFixed(4)} hint={t("common.measured")} />
              </>
            ) : (
              <p className="muted">{formatHint(locale)}</p>
            )}
            <Alert tone="warn" title={t("gen.why")}>
              {t("gen.whyBody")}
            </Alert>
            <Alert>
              Interpolation is a real weight-space operation, but it is <strong>not</strong> generation: it cannot produce a
              model outside the convex hull of the two parents.
            </Alert>
            <pre className="code">{`# desktop engine: hypernetwork generator
hcs-foundry generate --model my-mlp --corpus 11,22,33,44,55,66,77,88 --epochs 200

# desktop engine: evolution with a quality-diversity archive
hcs-foundry evolve --model my-mlp --population 16 --generations 5`}</pre>
          </div>
        </Card>
      </div>
    </>
  );
}

function formatHint(locale: string): string {
  const hints: Record<string, string> = {
    en: "Train two models in the Training Lab, then interpolate between them here.",
    de: "Trainiere zwei Modelle im Trainings-Labor und interpoliere hier zwischen ihnen.",
    fr: "Entraînez deux modèles dans le labo d'entraînement, puis interpolez-les ici.",
  };
  return hints[locale] ?? hints.en!;
}
