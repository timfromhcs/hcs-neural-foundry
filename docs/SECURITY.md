# Security, privacy and sandboxing

## 1. Threat model

| Asset | Threat | Control |
|---|---|---|
| The user's repository and files | A malicious checkpoint/archive escaping its directory | path allowlist, no archive extraction, safetensors only, `atomic_write` into the workspace |
| Local API | Any website on the machine calling `127.0.0.1` | bearer token required, strict CORS allowlist, `compare_digest`, rate limits |
| Local API | Binding to a public interface by accident | default `127.0.0.1`, explicit warning when a non-loopback host is passed |
| Benchmark integrity | An optimiser editing the test to look better | evaluator in a child process, read-only fixtures, hash recorded, `invalid` on evaluator failure |
| Weights | Silent mutation without provenance | immutable patch records with base hash, precondition, author and effect |
| Secrets | Tokens ending up in logs, prompts or commits | no secrets in schemas or logs, token file with `0600`, `.env*` ignored, secret scan in release checklist |
| Host | Arbitrary code execution via a "tensor expression" | no expression language, no `eval`/`exec`, closed operation enum (ADR-005) |
| Browser storage | Someone reading experiment data from disk | IndexedDB is origin-private; export is explicit; no telemetry |

## 2. Runtime rules, and where they are enforced

1. **No shell from model output.** `agents.py` spawns exactly one command: the verifier module with a
   JSON argument. There is no `run_shell` tool in MCP, and no endpoint accepts a command string.
   Enforced by `test_runtime_modules_avoid_dangerous_primitives`.
2. **No `eval`/`exec`/`pickle.load`.** Same test. A patch operation is an enum member; its parameters
   are validated field by field in `schemas.validate_operation_params`.
3. **safetensors only.** `checkpoints.load_state` rejects any suffix other than `.safetensors` with an
   explicit "unsupported by design" message. There is no pickle import path to disable.
4. **Path containment.** Tensor names are checked against a safe character set; workspace paths are
   built from the workspace root; imports write into the workspace only.
5. **API auth.** `Authorization: Bearer <token>` on every endpoint except `/health` and the OpenAPI
   document. The SSE endpoint accepts the token as a query parameter because `EventSource` cannot set
   headers; the token is compared with `secrets.compare_digest`.
6. **CORS.** Default allowlist is `http://localhost:5173`, `http://127.0.0.1:5173`,
   `http://localhost:4173`, overridable with `HCS_FOUNDRY_CORS_ORIGINS`. Credentials are disabled.
7. **Rate limiting.** Job creation, patching, evaluation, export, evolution and generation are
   rate-limited per client (30 requests / 60 s default).
8. **Resource caps.** Patch operations refuse tensors above 50M elements; low-rank rank is capped at
   512; training respects `max_seconds`; proposals are capped at 32 patches.
9. **Approval gate for MCP.** `GATED_TOOLS` refuse to run unless the tool name is in
   `HCS_FOUNDRY_MCP_APPROVE` or passed with `--approve`. Read-only tools never need approval.
10. **No telemetry.** Nothing in the package phones home. The web app's only outbound request is the
    loopback API the user configures, and non-loopback hosts are rejected in the client.

## 3. What an agent can and cannot do

| Level | Allowed | Blocked by default |
|---|---|---|
| 0 Explain | read models, tensors, history | everything else |
| 1 Draft (default) | create proposals | applying them |
| 2 Approved run | apply an approved proposal, run an evaluation | spending beyond `max_seconds_per_run` |
| 3 Bounded auto-run | repeat within per-task budget | leaving the workspace, network |
| 4 Campaign | finite experiment matrix | publishing, credentials, changing permissions |

`AgentPolicy` additionally allowlists operations and forbids `set_trainable` outright.

## 4. Privacy

* Prompts, datasets, checkpoints, logs and generated images are never uploaded.
* No analytics, no crash reporting, no model downloads without an explicit user action.
* The browser workspace is `IndexedDB` in the origin's private storage; the user can export a JSON
  bundle and delete everything from Settings.
* Tokens live in a file (`<workspace>/api-token`, `0600`) or an environment variable; they are never
  written to the database, logs or run records.

## 5. Reporting a vulnerability

Open a private security advisory on the repository or email the maintainers listed in `pyproject.toml`.
Please include: affected version, reproduction, impact and whether data could be exfiltrated. Do not
open a public issue for an unpatched vulnerability.

## 6. Known residual risks

1. **Loopback API is reachable by any local process** that can read the token file. Mitigation: file
   permissions; run on a single-user machine; rotate the token by deleting the file.
2. **`HCS_FOUNDRY_MCP_APPROVE=*`** approves everything. It exists for CI; do not use it interactively.
3. **Uploaded checkpoints** are validated structurally (tensor set, dtype, size caps) but a malicious
   *graph* is out of scope: this build imports weights, not ONNX graphs, into the engine.
4. **Browser storage is best effort.** Eviction can delete the workspace; the UI says so and offers
   export.
