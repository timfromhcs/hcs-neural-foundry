/**
 * API / MCP screen.
 *
 * The web app never needs the desktop engine. This page is an explicit, opt-in client for it: the
 * endpoint is validated before use, the token is kept in memory only, and nothing is sent automatically.
 */

import { useState } from "react";

import { useT } from "../i18n";
import { Alert, Card, Field, PageHead, StatusPill } from "../components/ui";

interface HealthResponse {
  status: string;
  version: string;
  workspace: string;
  schema_version: number;
  auth_required: boolean;
}

const MCP_TOOLS: { name: string; gated: boolean }[] = [
  { name: "foundry_get_hardware_profile", gated: false },
  { name: "foundry_capabilities", gated: false },
  { name: "foundry_list_models", gated: false },
  { name: "foundry_inspect_model", gated: false },
  { name: "foundry_list_tensors", gated: false },
  { name: "foundry_read_tensor_slice", gated: false },
  { name: "foundry_validate_weight_patch", gated: false },
  { name: "foundry_solve_parameter_budget", gated: false },
  { name: "foundry_get_run_status", gated: false },
  { name: "foundry_list_history", gated: false },
  { name: "foundry_get_evaluation_report", gated: false },
  { name: "foundry_estimate_training_resources", gated: false },
  { name: "foundry_start_evaluation", gated: true },
  { name: "foundry_start_training", gated: true },
  { name: "foundry_propose_weight_patch", gated: true },
  { name: "foundry_run_weight_patch", gated: true },
  { name: "foundry_export_model", gated: true },
  { name: "foundry_evolve", gated: true },
  { name: "foundry_generate_weights", gated: true },
];

export function ApiMcpPage() {
  const t = useT();
  const [endpoint, setEndpoint] = useState("http://127.0.0.1:8765");
  const [token, setToken] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const connect = async () => {
    setBusy(true);
    setError(null);
    setHealth(null);
    try {
      const url = new URL("/api/v1/health", endpoint.endsWith("/") ? endpoint : `${endpoint}/`);
      if (!["127.0.0.1", "localhost", "::1"].includes(url.hostname)) {
        throw new Error("Only loopback endpoints are accepted by this client.");
      }
      const response = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = (await response.json()) as HealthResponse;
      setHealth(payload);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHead title={t("api.title")} subtitle={t("api.subtitle")} />

      <div className="split">
        <Card title={t("api.connect")}>
          <div className="stack">
            <Field label={t("api.endpoint")} htmlFor="endpoint" hint="http://127.0.0.1:8765">
              <input id="endpoint" value={endpoint} onChange={(event) => setEndpoint(event.target.value)} inputMode="url" />
            </Field>
            <Field label={t("api.token")} htmlFor="token" hint="Printed by `hcs-foundry api serve`">
              <input id="token" type="password" value={token} onChange={(event) => setToken(event.target.value)} autoComplete="off" />
            </Field>
            <div className="row">
              <button type="button" className="btn primary" onClick={() => void connect()} disabled={busy}>
                {busy ? t("app.loading") : t("api.connect")}
              </button>
              <StatusPill tone={health ? "ok" : "neutral"}>{health ? t("api.connected") : t("api.notConnected")}</StatusPill>
            </div>
            {error && <Alert tone="bad">{error}</Alert>}
            {health && (
              <table>
                <tbody>
                  <tr>
                    <td>{t("api.health")}</td>
                    <td className="num">{health.status}</td>
                  </tr>
                  <tr>
                    <td>{t("footer.version")}</td>
                    <td className="num">{health.version}</td>
                  </tr>
                  <tr>
                    <td>schema</td>
                    <td className="num">{health.schema_version}</td>
                  </tr>
                  <tr>
                    <td>workspace</td>
                    <td className="mono tiny">{health.workspace}</td>
                  </tr>
                  <tr>
                    <td>auth</td>
                    <td className="num">{String(health.auth_required)}</td>
                  </tr>
                </tbody>
              </table>
            )}
            <Alert tone="warn">{t("api.noBackend")}</Alert>
            <Alert>{t("api.privacy")}</Alert>
          </div>
        </Card>

        <Card title={t("api.example")}>
          <pre className="code">{`# start the engine
hcs-foundry api serve --port 8765

# read-only request
curl -H "Authorization: Bearer $TOKEN" \\
  http://127.0.0.1:8765/api/v1/models

# MCP over stdio
hcs-foundry mcp --approve foundry_start_evaluation`}</pre>
          <div style={{ marginTop: "var(--space-3)" }}>
            <Alert>{t("api.privacy")}</Alert>
          </div>
        </Card>
      </div>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("api.tools")} subtitle={t("agent.gate")}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t("api.tools")}</th>
                  <th>{t("api.gated")}</th>
                </tr>
              </thead>
              <tbody>
                {MCP_TOOLS.map((tool) => (
                  <tr key={tool.name}>
                    <td className="mono tiny">{tool.name}</td>
                    <td>
                      <StatusPill tone={tool.gated ? "warn" : "ok"}>{tool.gated ? t("api.gated") : t("api.readOnly")}</StatusPill>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="tiny faint" style={{ marginTop: "var(--space-3)" }}>
            {t("agent.noShell")}
          </p>
        </Card>
      </div>
    </>
  );
}
