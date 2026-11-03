/** Settings: appearance, language, storage, privacy and the in-app capability parity matrix. */

import { useEffect, useState } from "react";

import { LOCALES, LOCALE_LABEL, type Locale } from "../locales";
import { useI18n, useT } from "../i18n";
import { formatBytes, formatNumber } from "../lib/format";
import { persistenceGranted, requestPersistence, storageEstimate } from "../lib/db";
import { backendStatus, resetSession, type BackendStatus } from "../lib/onnx";
import { useApp, type Preferences } from "../state";
import { Alert, CapabilityBadge, Card, Field, PageHead, StatusPill } from "../components/ui";
import type { Capability } from "../lib/types";

const PARITY: { key: string; desktop: Capability; browser: Capability }[] = [
  { key: "model.title", desktop: "full", browser: "full" },
  { key: "weights.title", desktop: "full", browser: "limited" },
  { key: "train.title", desktop: "full", browser: "limited" },
  { key: "gen.title", desktop: "experimental", browser: "limited" },
  { key: "agent.title", desktop: "experimental", browser: "limited" },
  { key: "eval.title", desktop: "full", browser: "limited" },
  { key: "library.title", desktop: "full", browser: "limited" },
  { key: "nav.apiMcp", desktop: "full", browser: "unavailable" },
  { key: "library.export", desktop: "experimental", browser: "unavailable" },
];

export function SettingsPage() {
  const t = useT();
  const { locale, setLocale } = useI18n();
  const { preferences, setPreferences, db, refresh } = useApp();
  const [storage, setStorage] = useState({ used: 0, quota: 0 });
  const [persisted, setPersisted] = useState(false);
  const [backend, setBackend] = useState<BackendStatus | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      setStorage(await storageEstimate());
      setPersisted(await persistenceGranted());
    })();
  }, []);

  const probe = async () => {
    setProbeError(null);
    resetSession();
    try {
      setBackend(await backendStatus(preferences.provider));
    } catch (cause) {
      setBackend(null);
      setProbeError((cause as Error).message);
    }
  };

  const update = (next: Partial<Preferences>) => {
    setPreferences(next);
    setMessage(t("settings.saved"));
    window.setTimeout(() => setMessage(null), 1500);
  };

  return (
    <>
      <PageHead title={t("settings.title")} subtitle={t("settings.subtitle")} />
      {message && <Alert tone="ok">{message}</Alert>}

      <div className="split" style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("settings.appearance")}>
          <div className="grid grid-2">
            <Field label={t("settings.theme")} htmlFor="theme">
              <select id="theme" value={preferences.theme} onChange={(event) => update({ theme: event.target.value as Preferences["theme"] })}>
                <option value="dark">{t("settings.themeDark")}</option>
                <option value="light">{t("settings.themeLight")}</option>
              </select>
            </Field>
            <Field label={t("settings.accent")} htmlFor="accent">
              <select id="accent" value={preferences.accent} onChange={(event) => update({ accent: event.target.value as Preferences["accent"] })}>
                <option value="blue">{t("settings.accentBlue")}</option>
                <option value="violet">{t("settings.accentViolet")}</option>
                <option value="emerald">{t("settings.accentEmerald")}</option>
                <option value="amber">{t("settings.accentAmber")}</option>
                <option value="rose">{t("settings.accentRose")}</option>
              </select>
            </Field>
            <Field label={t("settings.density")} htmlFor="density">
              <select id="density" value={preferences.density} onChange={(event) => update({ density: event.target.value as Preferences["density"] })}>
                <option value="comfortable">{t("settings.densityComfortable")}</option>
                <option value="compact">{t("settings.densityCompact")}</option>
              </select>
            </Field>
            <Field label={t("settings.language")} htmlFor="locale">
              <select id="locale" value={locale} onChange={(event) => setLocale(event.target.value as Locale)}>
                {LOCALES.map((code) => (
                  <option key={code} value={code}>
                    {LOCALE_LABEL[code]}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div className="row" style={{ marginTop: "var(--space-4)" }}>
            <label className="row" style={{ gap: "var(--space-2)" }}>
              <input
                type="checkbox"
                checked={preferences.contrast}
                onChange={(event) => update({ contrast: event.target.checked })}
              />
              {t("settings.contrast")}
            </label>
          </div>
        </Card>

        <Card title={t("settings.storage")}>
          <div className="stack">
            <table>
              <tbody>
                <tr>
                  <td>{t("settings.storageUsed")}</td>
                  <td className="num">{formatBytes(storage.used, locale)}</td>
                </tr>
                <tr>
                  <td>{t("settings.storageQuota")}</td>
                  <td className="num">{formatBytes(storage.quota, locale)}</td>
                </tr>
                <tr>
                  <td>{t("settings.persist")}</td>
                  <td className="num">
                    <StatusPill tone={persisted ? "ok" : "warn"}>
                      {persisted ? t("settings.persistGranted") : t("settings.persistDenied")}
                    </StatusPill>
                  </td>
                </tr>
              </tbody>
            </table>
            <div className="row">
              <button
                type="button"
                className="btn"
                onClick={async () => {
                  const granted = await requestPersistence();
                  setPersisted(granted);
                  setStorage(await storageEstimate());
                }}
              >
                {t("dash.requestPersist")}
              </button>
              <button
                type="button"
                className="btn danger"
                onClick={async () => {
                  if (!db) return;
                  if (!window.confirm(t("library.deleteConfirm"))) return;
                  await db.clearAll();
                  refresh();
                  setStorage(await storageEstimate());
                  setMessage(t("settings.clearDone"));
                }}
              >
                {t("settings.clear")}
              </button>
            </div>
            <p className="tiny faint">{formatNumber(storage.used, locale)} bytes stored locally.</p>
          </div>
        </Card>
      </div>

      <div className="split" style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("settings.backends")} subtitle={t("settings.providerNote")}>
          <div className="stack">
            <Field label={t("settings.provider")} htmlFor="provider">
              <select
                id="provider"
                value={preferences.provider}
                onChange={(event) => {
                  update({ provider: event.target.value as Preferences["provider"] });
                  resetSession();
                  setBackend(null);
                }}
              >
                <option value="auto">{t("settings.providerAuto")}</option>
                <option value="wasm">{t("settings.providerWasm")}</option>
                <option value="webgpu">{t("settings.providerWebgpu")}</option>
              </select>
            </Field>
            <div className="row">
              <button type="button" className="btn" onClick={() => void probe()}>
                {t("common.refresh")}
              </button>
              {backend && <StatusPill tone={backend.provider === "webgpu" ? "ok" : "warn"}>{backend.provider.toUpperCase()}</StatusPill>}
            </div>
            {backend && (
              <p className="tiny faint">
                {backend.detail} · load {formatNumber(backend.loadMs, locale, 0)} ms
                {backend.fallbackReason ? ` · fallback: ${backend.fallbackReason}` : ""}
              </p>
            )}
            {probeError && <Alert tone="bad">{probeError}</Alert>}
          </div>
        </Card>

        <Card title={t("settings.privacy")}>
          <Alert tone="warn">{t("settings.privacyBody")}</Alert>
          <p className="tiny faint" style={{ marginTop: "var(--space-3)" }}>
            {t("footer.localOnly")}
          </p>
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("parity.title")} subtitle={t("parity.note")}>
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
                {PARITY.map((row) => (
                  <tr key={row.key}>
                    <td>{t(row.key as never)}</td>
                    <td><CapabilityBadge value={row.desktop} /></td>
                    <td><CapabilityBadge value={row.browser} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}
