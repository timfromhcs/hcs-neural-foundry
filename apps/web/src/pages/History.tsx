/** History: runs and events, plus a full export/import bundle for portability. */

import { useEffect, useRef, useState } from "react";

import { useI18n, useT } from "../i18n";
import { formatDate } from "../lib/format";
import { isHistoryBundle } from "../lib/db";
import { useApp } from "../state";
import { Alert, Card, PageHead, StatusPill } from "../components/ui";
import type { EventRecord, RunRecord } from "../lib/types";

export function HistoryPage() {
  const t = useT();
  const { locale } = useI18n();
  const { db, refresh, version } = useApp();
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!db) return;
    void (async () => {
      setRuns(await db.listRuns());
      setEvents(await db.listEvents(50));
    })();
  }, [db, version]);

  const exportBundle = async () => {
    if (!db) return;
    const bundle = await db.exportBundle();
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `hcs-foundry-history-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setMessage(t("history.exportedFile"));
  };

  const importBundle = async (file: File) => {
    if (!db) return;
    setError(null);
    try {
      const text = await file.text();
      const parsed: unknown = JSON.parse(text);
      if (!isHistoryBundle(parsed)) throw new Error("invalid");
      await db.importBundle(parsed);
      refresh();
      setMessage(t("history.restored"));
    } catch {
      setError(t("history.invalidFile"));
    }
  };

  return (
    <>
      <PageHead
        title={t("history.title")}
        subtitle={t("history.subtitle")}
        actions={
          <>
            <button type="button" className="btn" onClick={() => void exportBundle()}>
              {t("history.export")}
            </button>
            <button type="button" className="btn" onClick={() => fileInput.current?.click()}>
              {t("history.import")}
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="application/json"
              className="sr-only"
              aria-label={t("history.import")}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void importBundle(file);
                event.target.value = "";
              }}
            />
          </>
        }
      />

      {message && <Alert tone="ok">{message}</Alert>}
      {error && (
        <div style={{ marginTop: "var(--space-3)" }}>
          <Alert tone="bad">{error}</Alert>
        </div>
      )}

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("history.runs")}>
          {runs.length === 0 ? (
            <p className="muted">{t("history.empty")}</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("history.kind")}</th>
                    <th>{t("history.state")}</th>
                    <th>{t("model.name")}</th>
                    <th className="num">{t("train.loss")}</th>
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
                      <td className="mono tiny">{run.modelId}</td>
                      <td className="num">
                        {typeof run.summary.finalLoss === "number"
                          ? (run.summary.finalLoss as number).toFixed(4)
                          : typeof run.summary.loss === "number"
                            ? (run.summary.loss as number).toFixed(4)
                            : "—"}
                      </td>
                      <td className="tiny faint">{formatDate(run.startedAt, locale)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("history.events")}>
          {events.length === 0 ? (
            <p className="muted">{t("history.empty")}</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("history.when")}</th>
                    <th>{t("history.state")}</th>
                    <th>scope</th>
                    <th>message</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event) => (
                    <tr key={event.id}>
                      <td className="tiny faint">{formatDate(event.at, locale)}</td>
                      <td>
                        <StatusPill tone={event.level === "error" ? "bad" : event.level === "warning" ? "warn" : "ok"}>{event.level}</StatusPill>
                      </td>
                      <td className="mono tiny">{event.scope}</td>
                      <td className="tiny">{event.message}</td>
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
