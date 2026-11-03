/**
 * Automated accessibility checks (axe-core) for the screens that carry the core journeys.
 *
 * Colour contrast is disabled on purpose: jsdom has no layout engine, so axe cannot compute rendered
 * colours meaningfully. Contrast is covered by the design tokens plus a manual checklist, and that
 * limitation is stated in `docs/TESTING.md` instead of being hidden behind a green check.
 */

import { render, cleanup } from "@testing-library/react";
import axe from "axe-core";
import { afterEach, describe, expect, it } from "vitest";

import { I18nProvider } from "../src/i18n";
import { AppProvider } from "../src/state";
import { Shell } from "../src/components/Shell";
import { ApiMcpPage } from "../src/pages/ApiMcp";
import { DocsPage } from "../src/pages/DocsPage";
import { SettingsPage } from "../src/pages/SettingsPage";
import { ModelStudioPage } from "../src/pages/ModelStudio";

afterEach(cleanup);

const AXE_OPTIONS: axe.RunOptions = {
  rules: {
    "color-contrast": { enabled: false },
    // Region rules need a full document landmark review; the shell provides the landmarks and is
    // covered by the structural assertions below.
    region: { enabled: false },
  },
};

async function renderWithProviders(node: React.ReactNode, locale: "en" | "de" | "fr" = "en") {
  const result = render(
    <I18nProvider initialLocale={locale}>
      <AppProvider>{node}</AppProvider>
    </I18nProvider>,
  );
  // Let the provider finish its IndexedDB open and demo-manifest fetch.
  await new Promise((resolve) => setTimeout(resolve, 0));
  return result;
}

async function violations(container: HTMLElement) {
  const results = await axe.run(container, AXE_OPTIONS);
  return results.violations.map((violation) => ({
    id: violation.id,
    impact: violation.impact,
    nodes: violation.nodes.map((node) => node.target.join(" ")),
  }));
}

describe("accessibility", () => {
  it("renders the model studio without serious or critical violations", async () => {
    const { container } = await renderWithProviders(
      <Shell route="model-studio" onNavigate={() => undefined}>
        <ModelStudioPage />
      </Shell>,
    );
    const found = await violations(container);
    expect(found.filter((item) => item.impact === "serious" || item.impact === "critical")).toEqual([]);
  });

  it("renders settings (parity matrix, forms) without serious or critical violations", async () => {
    const { container } = await renderWithProviders(
      <Shell route="settings" onNavigate={() => undefined}>
        <SettingsPage />
      </Shell>,
    );
    const found = await violations(container);
    expect(found.filter((item) => item.impact === "serious" || item.impact === "critical")).toEqual([]);
  });

  it("renders the docs curriculum without serious or critical violations", async () => {
    const { container } = await renderWithProviders(
      <Shell route="docs" onNavigate={() => undefined}>
        <DocsPage />
      </Shell>,
    );
    const found = await violations(container);
    expect(found.filter((item) => item.impact === "serious" || item.impact === "critical")).toEqual([]);
  });

  it("renders the API/MCP screen without serious or critical violations", async () => {
    const { container } = await renderWithProviders(
      <Shell route="api-mcp" onNavigate={() => undefined}>
        <ApiMcpPage />
      </Shell>,
    );
    const found = await violations(container);
    expect(found.filter((item) => item.impact === "serious" || item.impact === "critical")).toEqual([]);
  });

  it("keeps the German shell accessible", async () => {
    const { container } = await renderWithProviders(
      <Shell route="settings" onNavigate={() => undefined}>
        <SettingsPage />
      </Shell>,
      "de",
    );
    const found = await violations(container);
    expect(found.filter((item) => item.impact === "serious" || item.impact === "critical")).toEqual([]);
  });

  it("exposes navigation, a skip link and a single h1 per page", async () => {
    const { container } = await renderWithProviders(
      <Shell route="settings" onNavigate={() => undefined}>
        <SettingsPage />
      </Shell>,
    );
    expect(container.querySelectorAll("nav")).toHaveLength(1); // the sidebar is the only landmark nav
    expect(container.querySelector(".skip-link")).not.toBeNull();
    expect(container.querySelectorAll("h1")).toHaveLength(1);
    const current = container.querySelector('[aria-current="page"]');
    expect(current?.textContent).toContain("Settings");
  });
});
