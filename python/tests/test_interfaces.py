"""Interface tests: CLI contract, HTTP API authentication/CORS, and MCP approval gating."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from hcs_foundry.api import RateLimiter, create_app
from hcs_foundry.cli.main import main as cli_main
from hcs_foundry.errors import SecurityViolation
from hcs_foundry.mcp_server import GATED_TOOLS, ApprovalGate, build_server
from hcs_foundry.schemas import MlpConfig

TOKEN = "test-token"


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def test_cli_doctor_json(home, capsys):
    code = cli_main(["doctor", "--json", "--no-probe"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["workspace"] == str(home)
    assert payload["database"]["schema_version"] >= 1
    assert any(backend["kind"] == "cpu" for backend in payload["hardware"]["backends"])


def test_cli_model_create_dry_run(home, capsys):
    code = cli_main(["model", "create", "--family", "mlp", "--dry-run", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["parameters"] > 0


def test_cli_solve_and_report(home, capsys):
    assert cli_main(["solve", "--params", "50k", "--top", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_parameters"] == 50_000
    assert payload["candidates"]


def test_cli_weight_flow(home, capsys):
    config = MlpConfig(input_dim=8, hidden_dims=(10,), output_dim=3).model_dump(mode="json")
    config_path = home / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert cli_main(["--json", "model", "create", "--name", "cli-mlp", "--config", str(config_path)]) == 0
    created = json.loads(capsys.readouterr().out)
    model_id = created["id"]
    assert cli_main(["--json", "weights", "list", "--model", model_id]) == 0
    tensors = json.loads(capsys.readouterr().out)
    assert any(tensor["name"] == "layers.0.weight" for tensor in tensors)
    assert (
        cli_main(
            [
                "--json",
                "weights",
                "patch",
                "--model",
                model_id,
                "--tensor",
                "layers.0.weight",
                "--op",
                "scale",
                "--params",
                '{"factor": 0.9}',
            ]
        )
        == 0
    )
    patched = json.loads(capsys.readouterr().out)
    assert patched["validation"]["finite"] is True


def test_cli_unknown_model_returns_error_code(home, capsys):
    code = cli_main(["eval", "--model", "does-not-exist"])
    captured = capsys.readouterr()
    assert code == 1
    assert "not found" in captured.err.lower()


# --------------------------------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def client(foundry):
    app = create_app(foundry, token=TOKEN, cors_origins=("http://localhost:5173",))
    return TestClient(app)


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_needs_no_auth(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_missing_token_is_rejected(client):
    assert client.get("/api/v1/models").status_code == 401


def test_wrong_token_is_rejected(client):
    assert client.get("/api/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 403


def test_openapi_is_available(client):
    spec = client.get("/api/v1/openapi.json").json()
    assert "/api/v1/models" in spec["paths"]
    assert "/api/v1/weight-patches" in spec["paths"]


def test_model_lifecycle_over_http(client):
    config = MlpConfig(input_dim=6, hidden_dims=(8,), output_dim=2).model_dump(mode="json")
    response = client.post("/api/v1/models", headers=auth_headers(), json={"name": "http-mlp", "config": config})
    assert response.status_code == 200, response.text
    model_id = response.json()["id"]
    assert client.get(f"/api/v1/models/{model_id}", headers=auth_headers()).status_code == 200
    parameters = client.get(f"/api/v1/models/{model_id}/parameters", headers=auth_headers()).json()
    assert parameters["formula_agrees"] is True
    tensors = client.get(f"/api/v1/models/{model_id}/tensors", headers=auth_headers()).json()
    assert tensors
    estimate = client.get(f"/api/v1/models/{model_id}/estimate", headers=auth_headers()).json()
    assert estimate["total_train_bytes"] > 0


def test_patch_validation_endpoint_rejects_unknown_operations(client, mlp_model):
    payload = {
        "model_id": mlp_model.id,
        "patch": {
            "schema_version": "1.0",
            "patch_id": "p1",
            "base_checkpoint_sha256": "a" * 64,
            "architecture_hash": "b" * 64,
            "operation": "run_arbitrary_python",
            "target_tensor": "layers.0.weight",
            "parameters": {},
            "precondition": {"shape": [1], "dtype": "float32", "base_checkpoint_sha256": "a" * 64},
            "created_by": "test",
        },
    }
    response = client.post("/api/v1/weight-patches/validate", headers=auth_headers(), json=payload)
    assert response.status_code == 422


def test_cors_allowlist_is_enforced(client):
    allowed = client.options(
        "/api/v1/models",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    blocked = client.options(
        "/api/v1/models",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert blocked.headers.get("access-control-allow-origin") is None


def test_training_job_runs_in_background(client, mlp_model):
    response = client.post(
        "/api/v1/training-runs",
        headers=auth_headers(),
        json={"model_id": mlp_model.id, "steps": 4, "batch_size": 8, "eval_every": 2, "checkpoint_every": 4},
    )
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]
    run: dict = {"state": "queued"}
    for _ in range(400):
        run = client.get(f"/api/v1/training-runs/{run_id}", headers=auth_headers()).json()
        if "state" not in run:  # pragma: no cover - transient lookup failure
            time.sleep(0.05)
            continue
        if run["state"] in {"completed", "failed", "cancelled", "invalid", "interrupted"}:
            break
        time.sleep(0.05)
    assert run["state"] == "completed", run
    assert run["summary"]["steps_completed"] == 4


def test_sse_endpoint_requires_token(client):
    assert client.get("/api/v1/events/run_missing").status_code == 401


def test_artifact_endpoint_serves_content_addressed_files(client, foundry):
    from hcs_foundry.checkpoints import write_artifact

    sha, path = write_artifact(foundry.workspace, b'{"report": "ok"}', ".json")
    response = client.get(f"/api/v1/artifacts/{sha}", headers=auth_headers())
    assert response.status_code == 200
    assert response.content == b'{"report": "ok"}'
    assert client.get("/api/v1/artifacts/not-a-hash", headers=auth_headers()).status_code == 422
    assert client.get(f"/api/v1/artifacts/{'0' * 64}", headers=auth_headers()).status_code == 404


def test_rate_limiter_blocks_burst():
    limiter = RateLimiter(limit=2, window=60)
    limiter.check("a")
    limiter.check("a")
    with pytest.raises(SecurityViolation):
        limiter.check("a")


def test_error_payload_shape(client):
    response = client.get("/api/v1/models/nope", headers=auth_headers())
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert "hint" in body["error"]


# --------------------------------------------------------------------------------------------------
# MCP
# --------------------------------------------------------------------------------------------------


def test_approval_gate_blocks_by_default(monkeypatch):
    monkeypatch.delenv("HCS_FOUNDRY_MCP_APPROVE", raising=False)
    gate = ApprovalGate()
    with pytest.raises(SecurityViolation):
        gate.require("foundry_start_training")
    gate.require("foundry_list_models")  # read-only tools are not gated


def test_approval_gate_accepts_explicit_approval(monkeypatch):
    monkeypatch.setenv("HCS_FOUNDRY_MCP_APPROVE", "foundry_start_training")
    gate = ApprovalGate()
    gate.require("foundry_start_training")


def test_mcp_server_registers_read_and_gated_tools(foundry):
    server = build_server(foundry, approved=set())
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert {"foundry_get_hardware_profile", "foundry_list_models", "foundry_inspect_model"} <= names
    assert names >= GATED_TOOLS

    # A gated tool must refuse without approval and log the failure as an error payload.
    result = asyncio.run(server.call_tool("foundry_start_training", {"model_id": "nope", "steps": 1}))
    payload = result[0] if isinstance(result, tuple) else result
    text = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
    assert "security_violation" in text or "approval" in text.lower()

    # A read-only tool works.
    read_result = asyncio.run(server.call_tool("foundry_get_hardware_profile", {}))
    read_payload = read_result[0] if isinstance(read_result, tuple) else read_result
    text = json.dumps(read_payload, default=str) if not isinstance(read_payload, str) else read_payload
    assert "backends" in text


def test_mcp_tool_calls_are_audited(foundry):
    server = build_server(foundry, approved=set())
    asyncio.run(server.call_tool("foundry_list_models", {}))
    calls = foundry.repo.list_tool_calls(limit=10)
    assert any(call["tool"] == "foundry_list_models" for call in calls)
