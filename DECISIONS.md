# Architecture decision records

Format: ID, date, status, context, decision, consequences.

---

## ADR-001 — PyTorch as the single numerical engine, no hand-rolled autograd

**Status:** accepted (2026-11-03)

**Context.** The engine must train MLPs and small transformers with real gradients, export ONNX and
expose exact parameter counts. A NumPy autodiff shim would be smaller but would fragment numerics,
double the test surface and make ONNX export artificial.

**Decision.** Use PyTorch (2.5–2.x, CPU wheel) as the primary engine. Hugging Face stacks
(`transformers`, `diffusers`, `trl`) are *not* imported by core code; they remain optional adapters.

**Consequences.** ~250 MB dependency; `torch` becomes an extra (`.[torch]`) so pure inspection/CLI
paths still import without it. Every torch-touching module imports lazily and raises
`MissingDependencyError` with an install hint.

---

## ADR-002 — Python 3.12 in the venv, `requires-python = ">=3.11,<3.14"`

**Status:** accepted

**Context.** The system interpreter is 3.14.6; PyTorch does not publish cp314 wheels on the versions
we can pin.

**Decision.** Pin the venv to 3.12 via `uv venv --python 3.12`; declare the supported range in
`pyproject.toml`.

**Consequences.** Contributors on 3.14 get a clear resolution error instead of a broken install.

---

## ADR-003 — Exact parameter counts come from the instantiated module, never from formulas

**Status:** accepted

**Context.** Closed-form estimates drift from reality as soon as biases, RoPE tables, norms or tied
embeddings are added.

**Decision.** Formulas are used only to *prune* the solver search space. The number shown to the user
is always `sum(p.numel() for p in unique_parameters(model))`, with aliased/tied tensors de-duplicated
by `data_ptr`.

**Consequences.** Solver is slower (it builds candidates) but cannot lie. Tests assert equality for
every family.

---

## ADR-004 — Weight edits are immutable, hash-anchored patch objects

**Status:** accepted

**Context.** Editable tensors without provenance make reproducibility impossible and let an agent
silently change a checkpoint.

**Decision.** Every mutation is a `WeightPatch` with `base_checkpoint_sha256`, `architecture_hash`,
typed operation enum, validated parameters and a precondition block. Application is
validate → apply-to-copy → verify finite → hash → atomically persist new checkpoint + patch record.
Rollback = restore the previous checkpoint id.

**Consequences.** No free-form tensor edit path exists; a "scalar tweak" is also a patch.

---

## ADR-005 — No expression language, no `eval`, no shell from model output

**Status:** accepted

**Context.** An allowlisted tensor expression language was suggested by the spec.

**Decision.** Not implemented. Operations are a closed enum with typed parameters. `eval`, `exec`,
`pickle` and `subprocess` from agent/data input are forbidden in the codebase (enforced by a test
that greps for them in runtime modules).

**Consequences.** Less expressiveness, no arbitrary-code execution surface. A future expression mode
must run in a memory/time-bounded worker with an allowlisted AST.

---

## ADR-006 — SQLite for metadata, files for blobs, no ORM

**Status:** accepted

**Context.** Runs, patches, checkpoints and events must survive restarts without a server.

**Decision.** `sqlite3` with explicit forward-only migrations (`storage/migrations.py`), WAL mode,
`synchronous=NORMAL`. Blobs live in the artifact directory named by content hash. No ORM.

**Consequences.** Simple and portable; concurrent writers are limited (single-process by design).

---

## ADR-007 — safetensors as the only checkpoint format

**Status:** accepted

**Decision.** `.safetensors` for weights + JSON sidecar manifest. Pickle-based `.pt` loading is not
implemented for imports. Torch `state_dict` is converted to/from safetensors in memory.

**Consequences.** Importing legacy pickles is impossible by design ("unsupported", not "broken").

---

## ADR-008 — The evaluator is a separate process from the proposer

**Status:** accepted

**Decision.** `agents/` produces proposal manifests only. `evaluation/` is invoked as a subprocess
with a read-only fixture directory and its own copy of the model. Reward/score is returned as JSON;
the proposer cannot write it.

**Consequences.** Extra process spawn per evaluation; tamper resistance is structural, not textual.

---

## ADR-009 — Browser app is static, and says so

**Status:** accepted

**Decision.** Vite SPA, `output: static`, no `api/` functions. Browser inference uses
`onnxruntime-web` with `webgpu` then `wasm`; the provider is only *reported* after a smoke inference
succeeds. The demo trainer runs in a Web Worker on `Float32Array` and is labelled `Limited`.

**Consequences.** No server-side inference path can be accidentally relied upon. Some users will find
the browser version restricted — that is the honest outcome and it is visible in the parity matrix.

---

## ADR-010 — i18n from day one, keys checked by test

**Status:** accepted

**Decision.** `i18next` + `react-i18next`, three locale files, a Vitest test that asserts identical
key sets and non-empty values, and a rule that no user-visible string is hardcoded in components.

**Consequences.** Every new string costs three edits; the parity test makes the cost visible
immediately instead of at release time.

---

## ADR-011 — Capability model separates "declared" from "usable"

**Status:** accepted

**Context.** Vendor/driver probing lies regularly (e.g. a Vulkan DLL exists but no compute queue).

**Decision.** Each backend has `declared` (probe found it) and `usable` (a micro-benchmark actually
ran). UI and API expose both, plus `fallback_reason`.

**Consequences.** Slower startup (one small matmul benchmark), trustworthy claims.

---

## ADR-012 — Deterministic synthetic datasets are bundled; real datasets are manifests

**Status:** accepted

**Decision.** `datasets/builtin/*.npz` are generated locally from code with fixed seeds
(`scripts/build_datasets.py`) so CI needs no downloads and no license review. Public benchmarks are
referenced by protocol + hash only.

**Consequences.** Smoke suites measure pipeline correctness, not capability. Benchmark results are
labelled accordingly.

---

## ADR-013 — Training terminal states are explicit and exhaustive

**Status:** accepted

**Decision.** `queued|running|paused|completed|failed|cancelled|invalid|interrupted`. `invalid` is used
when the evaluator itself errors; it is never coerced to `0.0`.

**Consequences.** Aggregations must filter `invalid`; the UI shows it distinctly, not as failure.

---

## ADR-014 — MCP is a narrow typed surface over the same service layer as the HTTP API

**Status:** accepted

**Decision.** MCP tools call `services/*`, not the HTTP API, and reuse the same validators. Mutating
tools require an approval token present in the server config. No generic `run_shell` or
`read_any_file` tool exists.

**Consequences.** Two transports, one policy. Tool schemas are versioned with the package.

---

## ADR-015 — Localization scope: UI strings now, long-form docs in English

**Status:** accepted

**Context.** Full 20-lesson curriculum in three languages is a translation project, not a code task.

**Decision.** All *interface* strings ship in `en`/`de`/`fr` (parity-tested). Long-form tutorial
prose ships in English with DE/FR titles, goals and summaries. This limitation is displayed in the
Docs screen and in `docs/STATUS.md` rather than hidden.

**Consequences.** Honest partial coverage; a translation backlog item exists in `plan.md`.

---

## ADR-016 — Publishing is human-gated

**Status:** accepted

**Decision.** The agent never pushes to a remote, deploys, or publishes a package without explicit
approval per artifact. Scripts exist (`scripts/verify-release.ps1`) but are not run unattended
against credentials.

**Consequences.** Phase 14 remains open by policy, not by oversight.
