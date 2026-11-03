import { Component, useCallback, useEffect, useState } from "react";
import type { ErrorInfo, ReactNode } from "react";

import { Shell, ROUTES } from "./components/Shell";
import { Alert, PageHead } from "./components/ui";
import { useT } from "./i18n";
import { AppProvider } from "./state";
import { AgentLabPage } from "./pages/AgentLab";
import { ApiMcpPage } from "./pages/ApiMcp";
import { DashboardPage } from "./pages/Dashboard";
import { DocsPage } from "./pages/DocsPage";
import { EvaluationLabPage } from "./pages/EvaluationLab";
import { GeneratorLabPage } from "./pages/GeneratorLab";
import { HistoryPage } from "./pages/History";
import { ModelLibraryPage } from "./pages/ModelLibrary";
import { ModelStudioPage } from "./pages/ModelStudio";
import { SettingsPage } from "./pages/SettingsPage";
import { TrainingLabPage } from "./pages/TrainingLab";
import { WeightStudioPage } from "./pages/WeightStudio";

class PageBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console -- the only place we log, and only to the local console
    console.error("page render failed", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return <ErrorScreen error={this.state.error} onReset={() => this.setState({ error: null })} />;
    }
    return this.props.children;
  }
}

function ErrorScreen({ error, onReset }: { error: Error; onReset: () => void }) {
  const t = useT();
  return (
    <>
      <PageHead title={t("error.title")} subtitle={t("error.boundary")} />
      <Alert tone="bad" title={t("error.details")}>
        <pre className="code" style={{ marginTop: "var(--space-2)" }}>
          {error.message}
        </pre>
        <button type="button" className="btn small" style={{ marginTop: "var(--space-2)" }} onClick={onReset}>
          {t("error.retry")}
        </button>
      </Alert>
    </>
  );
}

function readRoute(): string {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return ROUTES.some((route) => route.id === hash) ? hash : "dashboard";
}

function Router() {
  const [route, setRoute] = useState<string>(() => readRoute());

  useEffect(() => {
    const onHashChange = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = useCallback((next: string) => {
    window.location.hash = `#/${next}`;
    setRoute(next);
    document.getElementById("main")?.focus();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  return (
    <Shell route={route} onNavigate={navigate}>
      <PageBoundary key={route}>
        {route === "dashboard" && <DashboardPage onNavigate={navigate} />}
        {route === "model-studio" && <ModelStudioPage />}
        {route === "weight-studio" && <WeightStudioPage onNavigate={navigate} />}
        {route === "training-lab" && <TrainingLabPage />}
        {route === "evaluation-lab" && <EvaluationLabPage />}
        {route === "generator-lab" && <GeneratorLabPage />}
        {route === "agent-lab" && <AgentLabPage />}
        {route === "model-library" && <ModelLibraryPage />}
        {route === "history" && <HistoryPage />}
        {route === "api-mcp" && <ApiMcpPage />}
        {route === "docs" && <DocsPage />}
        {route === "settings" && <SettingsPage />}
      </PageBoundary>
    </Shell>
  );
}

export function App() {
  return (
    <AppProvider>
      <Router />
    </AppProvider>
  );
}
