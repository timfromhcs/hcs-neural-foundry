/** Model Library: models, checkpoints, size and export state — with honest export limits. */

import { useEffect, useState } from "react";

import { useI18n, useT } from "../i18n";
import { formatBytes, formatDate, formatNumber, shortHash } from "../lib/format";
import { serializeMlp } from "../lib/mlp";
import { useApp } from "../state";
import { Alert, Card, MaturityBadge, PageHead, StatusPill } from "../components/ui";
import type { CheckpointRecord, ModelRecord } from "../lib/types";

export function ModelLibraryPage() {
  const t = useT();
  const { locale } = useI18n();
  const { db, demo, refresh, version } = useApp();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);

  useEffect(() => {
    if (!db) return;
    void (async () => {
      setModels(await db.listModels());
      setCheckpoints(await db.listCheckpoints());
    })();
  }, [db, version]);

  const download = (model: ModelRecord) => {
    if (!model.weights) return;
    const text = serializeMlp(JSON.parse(model.weights) as never);
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${model.id}.weights.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <PageHead title={t("library.title")} subtitle={t("library.subtitle")} />

      {demo && (
        <Card title={`${demo.name} · ${t("common.measured")}`} actions={<MaturityBadge maturity={demo.maturity} />}>
          <div className="grid grid-4">
            <div>
              <span className="kpi-label">{t("model.actual")}</span>
              <div className="kpi-value">{formatNumber(demo.parameterCount, locale)}</div>
            </div>
            <div>
              <span className="kpi-label">{t("library.size")}</span>
              <div className="kpi-value">{formatBytes(demo.onnx.bytes, locale)}</div>
            </div>
            <div>
              <span className="kpi-label">{t("library.parity")}</span>
              <div className="kpi-value">{demo.onnx.parity.maxAbsDiff.toExponential(1)}</div>
            </div>
            <div>
              <span className="kpi-label">{t("history.when")}</span>
              <div className="kpi-value tiny">{demo.training.date}</div>
            </div>
          </div>
          <div style={{ marginTop: "var(--space-3)" }}>
            <Alert tone="warn">{demo.claim}</Alert>
          </div>
          <p className="tiny faint" style={{ marginTop: "var(--space-3)" }}>
            {t("library.hash")}: <span className="mono">{shortHash(demo.onnx.sha256, 24)}</span> · {t("train.steps")} {demo.training.steps} ·
            final loss {demo.training.finalLoss.toFixed(4)} · {demo.training.seconds.toFixed(1)} s ({demo.training.device})
          </p>
        </Card>
      )}

      <div style={{ marginTop: "var(--space-4)" }}>
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
                    <th>{t("library.hash")}</th>
                    <th>{t("library.created")}</th>
                    <th>{t("weights.values")}</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model) => (
                    <tr key={model.id}>
                      <td>
                        <div>{model.name}</div>
                        <small className="faint">{model.id}</small>
                      </td>
                      <td className="mono tiny">{model.family}</td>
                      <td className="num">{formatNumber(model.parameterCount, locale)}</td>
                      <td className="mono tiny">{shortHash(model.architectureHash)}</td>
                      <td className="tiny faint">{formatDate(model.createdAt, locale)}</td>
                      <td>
                        {model.weights ? (
                          <button type="button" className="btn small" onClick={() => download(model)}>
                            {t("common.export")}
                          </button>
                        ) : (
                          <StatusPill tone="neutral">{t("common.notAvailable")}</StatusPill>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("library.checkpoints")} subtitle={t("library.hash")}>
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
                    <th className="num">{t("library.size")}</th>
                    <th>{t("library.created")}</th>
                  </tr>
                </thead>
                <tbody>
                  {checkpoints.map((checkpoint) => (
                    <tr key={checkpoint.id}>
                      <td>{checkpoint.label}</td>
                      <td className="mono tiny">{shortHash(checkpoint.sha256)}</td>
                      <td>{checkpoint.origin}</td>
                      <td className="num">{formatBytes(checkpoint.sizeBytes, locale)}</td>
                      <td className="tiny faint">{formatDate(checkpoint.createdAt, locale)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("library.export")}>
          <p className="muted">{t("common.notAvailable")}: ONNX export needs the desktop engine (PyTorch + onnxruntime).</p>
          <pre className="code">{`hcs-foundry export --model <model> --format onnx --checkpoint latest`}</pre>
          <div className="row" style={{ marginTop: "var(--space-3)" }}>
            <button
              type="button"
              className="btn danger"
              onClick={async () => {
                if (!db) return;
                if (!window.confirm(t("library.deleteConfirm"))) return;
                await db.clearAll();
                refresh();
              }}
            >
              {t("library.deleteAll")}
            </button>
            <span className="tiny faint">{t("settings.privacyBody")}</span>
          </div>
        </Card>
      </div>
    </>
  );
}
