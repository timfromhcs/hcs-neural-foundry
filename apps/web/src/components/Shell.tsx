/** Application shell: persistent navigation, skip link, quick preferences, credit footer. */

import type { ReactNode } from "react";

import { LOCALES, LOCALE_LABEL, type Locale } from "../locales";
import { useI18n } from "../i18n";
import { useApp } from "../state";
import { Select } from "./ui";

export interface RouteDefinition {
  id: string;
  labelKey: Parameters<ReturnType<typeof useI18n>["t"]>[0];
  group: "work" | "research" | "system";
  icon: string;
}

export const ROUTES: RouteDefinition[] = [
  { id: "dashboard", labelKey: "nav.dashboard", group: "work", icon: "◎" },
  { id: "model-studio", labelKey: "nav.modelStudio", group: "work", icon: "◈" },
  { id: "weight-studio", labelKey: "nav.weightStudio", group: "work", icon: "▦" },
  { id: "training-lab", labelKey: "nav.trainingLab", group: "work", icon: "▷" },
  { id: "evaluation-lab", labelKey: "nav.evaluationLab", group: "work", icon: "⚖" },
  { id: "generator-lab", labelKey: "nav.generatorLab", group: "research", icon: "∿" },
  { id: "agent-lab", labelKey: "nav.agentLab", group: "research", icon: "⬡" },
  { id: "model-library", labelKey: "nav.modelLibrary", group: "research", icon: "▤" },
  { id: "history", labelKey: "nav.history", group: "system", icon: "↻" },
  { id: "api-mcp", labelKey: "nav.apiMcp", group: "system", icon: "⇄" },
  { id: "docs", labelKey: "nav.docs", group: "system", icon: "❑" },
  { id: "settings", labelKey: "nav.settings", group: "system", icon: "⚙" },
];

const GROUP_LABEL: Record<RouteDefinition["group"], "nav.group.work" | "nav.group.research" | "nav.group.system"> = {
  work: "nav.group.work",
  research: "nav.group.research",
  system: "nav.group.system",
};

export function Shell({
  route,
  onNavigate,
  children,
}: {
  route: string;
  onNavigate: (id: string) => void;
  children: ReactNode;
}) {
  const { t, locale, setLocale } = useI18n();
  const { ready } = useApp();

  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        {t("app.skipToContent")}
      </a>
      <aside className="sidebar" aria-label={t("app.name")}>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            HF
          </span>
          <div>
            <div className="brand-title">{t("app.name")}</div>
            <div className="brand-sub">
              {t("app.tagline")} · {t("app.beta")}
            </div>
          </div>
        </div>

        <nav className="nav" aria-label={t("app.name")}>
          {(["work", "research", "system"] as const).map((group) => (
            <div key={group}>
              <div className="nav-group-label">{t(GROUP_LABEL[group])}</div>
              <div className="nav">
                {ROUTES.filter((item) => item.group === group).map((item) => (
                  <a
                    key={item.id}
                    className="nav-link"
                    href={`#/${item.id}`}
                    aria-current={route === item.id ? "page" : undefined}
                    onClick={(event) => {
                      event.preventDefault();
                      onNavigate(item.id);
                    }}
                  >
                    <span className="nav-icon" aria-hidden="true">
                      {item.icon}
                    </span>
                    {t(item.labelKey)}
                  </a>
                ))}
              </div>
            </div>
          ))}
        </nav>

        <div className="spacer" />
        <div className="stack" style={{ gap: "var(--space-2)" }}>
          <Select<Locale>
            aria-label={t("settings.language")}
            value={locale}
            onChange={setLocale}
            options={LOCALES.map((code) => ({ value: code, label: LOCALE_LABEL[code] }))}
          />
          <span className="tiny faint" aria-live="polite">
            {ready ? t("footer.localOnly") : t("app.loading")}
          </span>
        </div>
      </aside>

      <main className="main" id="main" tabIndex={-1}>
        {children}
        <footer className="footer">
          <span>{t("app.credit")}</span>
          <span aria-hidden="true">·</span>
          <span>
            {t("footer.version")} 0.1.0
          </span>
          <span aria-hidden="true">·</span>
          <a href="#/docs">{t("footer.docs")}</a>
          <span aria-hidden="true">·</span>
          <a href="#/settings">{t("footer.status")}</a>
        </footer>
      </main>
    </div>
  );
}
