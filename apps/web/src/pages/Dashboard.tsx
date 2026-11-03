/** Dashboard: device capability, workspace contents, shipped example, honest claim boundaries. */

import { useEffect, useState } from "react";

import { useI18n, useT } from "../i18n";
import { formatBytes, formatNumber } from "../lib/format";
import { backendStatus, runDemoModel, encodePrompt, argmax, type BackendStatus } from "../lib/onnx";
import { storageEstimate } from "../lib/db";
import { useApp } from "../state";
import { Alert, Card, MaturityBadge, PageHead, Stat, StatusPill } from "../components/ui";
import type { ModelRecord, RunRecord } from "../lib/types";

export function DashboardPage({ onNavigate }: { onNavigate: (route: string) => void }) {
  const t = useT();
  const { locale } = useI18n();
  const { db, demo, demoError, version, preferences } = useApp();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [storage, setStorage] = useState({ used: 0, quota: 0 });
  const [backend, setBackend] = useState<BackendStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [inference, setInference] = useState<{ ms: number; provider: string; next: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!db) return;
      const [modelList, runList] = await Promise.all([db.listModels(), db.listRuns()]);
      if (cancelled) return;
      setModels(modelList);
      setRuns(runList.slice(0, 6));
      setStorage(await storageEstimate());
    })();
    return () => {
      cancelled = true;
    };
  }, [db, version]);

  const probe = async () => {
    setBusy(true);
    setError(null);
    try {
      // Respect the provider chosen in Settings; "auto" is only the default, not a hard override.
      const status = await backendStatus(preferences.provider);
      setBackend(status);
      const result = await runDemoModel(encodePrompt("hcs foundry builds", 64), preferences.provider);
      setInference({ ms: result.ms, provider: result.provider, next: argmax(result.logits.slice(-64)) });
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHead
        title={t("dash.title")}
        subtitle={t("dash.subtitle")}
        actions={
          <button type="button" className="btn primary" onClick={() => onNavigate("model-studio")}>
            {t("nav.modelStudio")}
          </button>
        }
      />

      <div className="grid grid-4">
        <Card tight>
          <Stat label={t("dash.device")} value={`${navigator.hardwareConcurrency ?? 0} cores`} hint={navigator.platform || t("common.unknown")} />
        </Card>
        <Card tight>
          <Stat label={t("dash.recentModels")} value={formatNumber(models.length, locale)} hint={t("dash.workspace")} />
        </Card>
        <Card tight>
          <Stat label={t("dash.recentRuns")} value={formatNumber(runs.length, locale)} hint={t("history.title")} />
        </Card>
        <Card tight>
          <Stat
            label={t("dash.storage")}
            value={formatBytes(storage.used, locale)}
            hint={`${t("dash.storageUsed")} ${formatBytes(storage.quota, locale)}`}
          />
        </Card>
      </div>

      <div className="split" style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("dash.backend")} subtitle={t("dash.backendHint")}>
          <div className="stack">
            <div className="row">
              <button type="button" className="btn" onClick={() => void probe()} disabled={busy}>
                {busy ? t("app.loading") : t("common.run")}
              </button>
              {backend && (
                <StatusPill tone={backend.provider === "webgpu" ? "ok" : "warn"}>
                  {backend.provider.toUpperCase()}
                </StatusPill>
              )}
              <span className="tiny faint">{navigator.userAgent.includes("Chrome") ? "Chromium" : "browser"}</span>
            </div>
            {backend && (
              <div className="stack" style={{ gap: "var(--space-1)" }}>
                <span className="tiny muted">{backend.detail}</span>
                <span className="tiny faint">
                  WebGPU API present: {String(backend.webgpuSupported)} · load {formatNumber(backend.loadMs, locale, 0)} ms
                </span>
                {backend.fallbackReason && <span className="tiny faint">fallback: {backend.fallbackReason}</span>}
              </div>
            )}
            {inference && (
              <Alert tone="ok">
                {t("common.measured")}: {formatNumber(inference.ms, locale, 1)} ms on {inference.provider.toUpperCase()} ·
                argmax {inference.next}
              </Alert>
            )}
            {!backend && !error && <p className="tiny faint">{t("dash.backendHint")}</p>}
            {error && <Alert tone="bad">{error}</Alert>}
          </div>
        </Card>

        <Card
          title={demo ? demo.name : t("library.title")}
          actions={demo ? <MaturityBadge maturity={demo.maturity} /> : undefined}
        >
          {demo ? (
            <div className="stack">
              <table>
                <tbody>
                  <tr>
                    <td>{t("model.actual")}</td>
                    <td className="num">{formatNumber(demo.parameterCount, locale)}</td>
                  </tr>
                  <tr>
                    <td>{t("train.valLoss")}</td>
                    <td className="num">{demo.evaluation.metrics.find((metric) => metric.name === "val_loss")?.value.toFixed(4) ?? "—"}</td>
                  </tr>
                  <tr>
                    <td>{t("library.parity")}</td>
                    <td className="num">{demo.onnx.parity.maxAbsDiff.toExponential(2)}</td>
                  </tr>
                  <tr>
                    <td>{t("library.size")}</td>
                    <td className="num">{formatBytes(demo.onnx.bytes, locale)}</td>
                  </tr>
                </tbody>
              </table>
              <Alert tone="warn" title={t("gen.boundary")}>
                {demo.claim}
              </Alert>
            </div>
          ) : (
            <Alert tone={demoError ? "warn" : "info"}>
              {demoError ? t("common.notAvailable") : t("app.loading")}
            </Alert>
          )}
        </Card>
      </div>

      <div className="split" style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("dash.recentModels")}>
          {models.length === 0 ? (
            <p className="muted">{t("dash.noModels")}</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("model.name")}</th>
                    <th>{t("model.family")}</th>
                    <th className="num">{t("model.actual")}</th>
                    <th>{t("library.origin")}</th>
                  </tr>
                </thead>
                <tbody>
                  {models.slice(0, 6).map((model) => (
                    <tr key={model.id}>
                      <td className="truncate">{model.name}</td>
                      <td className="mono tiny">{model.family}</td>
                      <td className="num">{formatNumber(model.parameterCount, locale)}</td>
                      <td>
                        <StatusPill tone={model.weights ? "ok" : "neutral"}>{model.source}</StatusPill>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title={t("dash.quickStart")}>
          <ol className="stack" style={{ paddingLeft: "1.2rem", margin: 0 }}>
            <li>
              <button type="button" className="btn ghost small" onClick={() => onNavigate("model-studio")}>
                {t("dash.quick1")}
              </button>
            </li>
            <li>
              <button type="button" className="btn ghost small" onClick={() => onNavigate("weight-studio")}>
                {t("dash.quick2")}
              </button>
            </li>
            <li>
              <button type="button" className="btn ghost small" onClick={() => onNavigate("training-lab")}>
                {t("dash.quick3")}
              </button>
            </li>
            <li>
              <button type="button" className="btn ghost small" onClick={() => onNavigate("docs")}>
                {t("dash.quick4")}
              </button>
            </li>
          </ol>
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("dash.recentRuns")}>
          {runs.length === 0 ? (
            <p className="muted">{t("dash.noRuns")}</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("history.kind")}</th>
                    <th>{t("history.state")}</th>
                    <th>{t("history.when")}</th>
                    <th>{t("train.loss")}</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr key={run.id}>
                      <td>{run.kind}</td>
                      <td>
                        <StatusPill tone={run.state === "completed" ? "ok" : run.state === "invalid" ? "bad" : "warn"}>
                          {run.state}
                        </StatusPill>
                      </td>
                      <td className="tiny faint">{run.startedAt.slice(0, 19).replace("T", " ")}</td>
                      <td className="num">
                        {typeof run.summary.finalLoss === "number" ? (run.summary.finalLoss as number).toFixed(4) : "—"}
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
