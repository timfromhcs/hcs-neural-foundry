# Implementation plan — HCS Neural Foundry

This file is the live status board. `AGENTS.md` is the contract; this file records what is real.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done and verified · `[!]` blocked/unsupported with reason

Last updated: 2026-11-03 (session 1).

---

## Phase 0 — Reconnaissance and research validation

- [x] Repository inventory: `E:\neuralfond` contained only `AGENTS.md`, no git repo, no prior work at risk.
- [x] Environment inventory: Python 3.14 system (torch unsupported range used instead → Python 3.12 venv), Node 22.23, `uv` 0.13.5, git 2.54, cargo 1.96.
- [x] Dependency reality check against the registry before choosing the engine (`torch 2.15.0`, `numpy 2.5.4`, `pydantic 2.14.2`, `fastapi 0.162.0`, `safetensors 0.8.0` resolve on cp312).
- [x] `DECISIONS.md` with ADR-001…ADR-016.
- [x] `docs/RESEARCH.md` evidence map + "claims we must not make".
- [x] Risk register (see end of this file).
- [x] License inventory: code MIT; no third-party weights or datasets bundled.

Gate: no undocumented assumptions, nothing overwritten. **PASSED.**

## Phase 1 — Foundation, shell, localization

- [x] Monorepo skeleton (`python/`, `apps/web/`, `mcp/`, `docs/`, `scripts/`).
- [x] Typed Python package `hcs_foundry` with strict schemas (pydantic v2).
- [x] SQLite metadata store with forward-only migrations + migration test.
- [x] Artifact store: temp-write → hash → atomic promote.
- [x] Web app shell: 12 routes, dark-first design tokens, responsive, keyboard accessible.
- [x] Locales `en`, `de`, `fr` with a parity test that fails on missing/extra keys.
- [x] Footer credit "Made with ❤️ by hcsmedia and timfromhcs".

Gate: clean install/build/test, no dead menu entries. **PASSED** (verified by `scripts/verify.ps1`).

## Phase 2 — Hardware detection and resource estimation

- [x] CPU / RAM / disk / OS / Python detection.
- [x] Accelerator probe: torch device list, ROCm flag, DirectML presence via `torch_directml` absence check, Vulkan via WMI/driver query — recorded as *claims*, then validated by a micro-benchmark before being marked usable.
- [x] Capability registry with `usable` vs `declared` separation and an honest fallback reason.
- [x] Resource estimator (params, grads, optimizer state, activations, KV cache, export buffers).
- [x] Profiles `safe` / `balanced` / `performance` / `custom` with hard caps.
- [x] `hcs-foundry doctor` + `doctor --json`.

Gate: fixtures simulate CPU-only, CUDA, ROCm, browser WASM/WebGPU; unsupported backends degrade with a message. **PASSED.**

## Phase 3 — Model factory and exact parameter budgets

- [x] Registry with `ModelFamilyAdapter` protocol.
- [x] `mlp` family (in/out/hidden layers/activation).
- [x] `transformer-lm` family (RoPE, GQA, SwiGLU, RMSNorm, tied embeddings optional).
- [x] Exact `numel` accounting including tied/aliased tensors (counted once).
- [x] Budget solver with `exact|closest|closest_under|at_least`, constraint validation, ranked candidates, infeasibility explanation.
- [x] Deterministic build under seed; architecture hash reproducible across processes.
- [x] Smoke test: finite forward pass, shape assertions.

Gate: displayed count == `sum(p.numel())` for every family, rebuilt config hashes identically. **PASSED.**

## Phase 4 — Weight Studio

- [x] Tensor metadata (shape, dtype, min/max/mean/std/norm, %zeros, finiteness).
- [x] Bounded slice reader + histogram (never loads a whole checkpoint into the UI).
- [x] Patch ops: `set_scalar`, `set_slice`, `init_seeded`, `add_scaled_delta`, `low_rank_delta`, `normalize`, `clamp`, `scale`, `sparsity_mask`, `interpolate`, `copy_from`.
- [x] Precondition validation (shape, dtype, base hash, parameter caps), finite-value guard, immutable patch records, rollback.
- [x] Provenance: patch id, author (`user`/`agent:<id>`), timestamps, before/after hashes.

Gate: apply → diff → evaluate → restart → rollback with provenance. **PASSED.**

## Phase 5 — Baseline training and evaluation

- [x] Unified trainer lifecycle `validate → estimate → prepare → dry_run → train → validate → checkpoint → evaluate`.
- [x] AdamW baseline, gradient clipping, deterministic seeding, checkpoint/resume, pause/cancel, terminal states.
- [x] Bundled synthetic datasets (no license friction): `tiny-shapes` (vision-ish MLP), `tiny-text` (char transformer).
- [x] Benchmark registry with protocol files, dataset hash, evaluator version, seed, hardware, timing.
- [x] Invalid-run handling: an evaluator error makes a run `invalid`, not a zero.

Gate: reproducible run, reload reproduces inference within tolerance. **PASSED.**

## Phase 6 — Weight generation research lab

- [x] Checkpoint corpus manifest with license + hash fields.
- [x] Deterministic tensor encoding (per-tensor z-score, shape-preserving).
- [x] Hypernetwork generator for a fixed tiny MLP (`hcs-foundry generate`).
- [!] D2NWG-style diffusion and DeepWeightFlow-style flow matching: **interface only** (declared `unsupported`), because reproducing them credibly needs a real checkpoint corpus and a study that this repo cannot complete honestly in one session.
- [x] Leakage guard: train/test split by checkpoint family, asserted in tests.
- [x] Independent evaluation of generated candidates.

Gate: generated candidate evaluated out-of-sample; docs avoid SOTA claims. **PASSED (narrow scope).**

## Phase 7 — Evolution and quality-diversity

- [x] Antithetic Evolution Strategies baseline with reward normalization.
- [x] MAP-Elites archive with explicit behaviour descriptors (score, latency, size).
- [x] Deterministic toy environment; comparison against gradient baseline at equal evaluation budget.
- [x] Population/Pareto export as JSON for the UI.

Gate: reproducible improvement and diversity; fair-budget comparison documented. **PASSED.**

## Phase 8 — LLM / text diffusion / image diffusion pipelines

- [~] Autoregressive transformer: implemented end-to-end at educational scale (char-level, <1M params).
- [!] Masked diffusion LM: declared `unsupported` (trainer/sampler not implemented; adapter slot exists).
- [!] Latent image diffusion / DiT: declared `unsupported`. No toy-quality claim is made.
- [x] Model→ONNX export with numeric parity check against the native run.

Gate partly met; unsupported items are labelled in `docs/STATUS.md` and in the UI. **PASSED WITH DECLARED GAPS.**

## Phase 9 — Reward / RL trainers

- [!] DPO/PPO/GRPO: declared `unsupported`. A reward interface + deterministic reward fixtures exist and are tested, so the slot is honest, but no RL trainer is claimed.

## Phase 10 — Agent Lab and MCP

- [x] Planner / proposer / verifier separation: agent proposals are immutable manifests validated by the execution service; the evaluator never runs in the proposer process.
- [x] MCP server over stdio with read-only tools and approval-gated mutating tools.
- [x] Resource caps, workspace root allowlist, no shell bridge, no arbitrary Python.
- [x] Tool-invocation log persisted to SQLite and exposed read-only.

Gate: read-only tool works with a conforming client; mutating tool refuses without approval. **PASSED.**

## Phase 11 — Browser / Vercel

- [x] Static Vite build, zero server functions, no remote inference.
- [x] ONNX Runtime Web with `webgpu` → `wasm` fallback, provider validated by a real smoke inference before use.
- [x] IndexedDB run history + export/import bundle + delete-all.
- [x] `navigator.storage.estimate()` / `persist()` surfaced in Settings.
- [x] Browser-local demo trainer (tiny MLP, Web Worker) — labelled `Limited`.
- [x] Capability parity matrix in-app.

Gate: no hidden backend calls; WASM path works; WebGPU labelled experimental. **PASSED.**

## Phase 12 — API, CLI, packaging, desktop

- [x] Versioned REST API + OpenAPI at `/api/v1/openapi.json`, bearer token for mutations, strict CORS, rate limits.
- [x] CLI covering doctor/model/weights/train/eval/evolve/generate/export/history/api/mcp.
- [x] Wheel build + clean-venv install smoke test.
- [!] npm package publish, PyPI publish, Tauri desktop, self-extracting launcher: **not published** — no credentials exercised, no signed artifacts produced. Declared.
- [x] SBOM-ish dependency inventory script + `verify-release` script.

## Phase 13 — QA gates

- [x] pytest suite (unit + integration + API security + numerical gradient checks).
- [x] vitest suite for the web app (i18n parity, storage, reducer logic).
- [x] ruff lint clean; TypeScript strict typecheck clean.
- [x] Accessibility: keyboard flow, focus visibility, labels, `prefers-reduced-motion`, no color-only status.
- [x] Locale parity test (en/de/fr) in CI script.
- [x] Performance smoke: cold start, model build, step time, export time recorded in `docs/PERFORMANCE.md`.

## Phase 14 — Publish

- [!] Not executed by design. No push, no deploy, no package publication happened in this session.
  Reasons: `AGENTS.md` requires explicit human approval for external publication, and I will not
  exercise credentials on the user's behalf.

---

## Risk register

| # | Risk | Mitigation | State |
|---|---|---|---|
| R1 | Overclaiming weight generation | maturity labels + no-SOTA rule + independent evaluator | active |
| R2 | Parameter-count drift between UI and engine | single source of truth: `count_parameters` enumerated from the live module | active |
| R3 | Unsafe checkpoint import | safetensors only, size/shape caps, no pickle path | active |
| R4 | Local API exposed beyond loopback | bind 127.0.0.1, token required, CORS allowlist, warning in Settings | active |
| R5 | Benchmark tampering by the agent | evaluator in separate process, fixtures read-only, hash recorded | active |
| R6 | torch absence on a user machine | core inspection/model/CLI paths degrade; torch-dependent commands fail with an actionable message | active |
| R7 | Determinism drift across platforms | seeds recorded, cross-platform equality documented as best-effort | active |
| R8 | Scope inflation (many half-built features) | every feature is either implemented+tested or explicitly `unsupported` in `docs/STATUS.md` | active |
