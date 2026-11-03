# Implementation plan — HCS Neural Foundry

Live status board. `AGENTS.md` is the contract; this file records what is real, what is narrow, and
what was deliberately not attempted. Anything marked `[!]` is not a TODO backlog item that was
forgotten — it is a decision, with the reason written next to it.

Legend: `[x]` done and verified · `[~]` done but narrow (declared experimental) · `[!]` not implemented,
declared unsupported · `[ ]` open backlog.

Last updated: 2026-11-03 (session 2, release 0.2.0).

---

## Phase 0 — Reconnaissance and research validation

- [x] Repository inventory: `E:\neuralfond` contained only `AGENTS.md`; no git repo; no prior work at risk.
- [x] Environment inventory: system Python 3.14.6 (no torch wheels in range) → Python 3.12.13 venv;
      Node 22.23.1; uv 0.13.5; git 2.54.0; cargo 1.96.0.
- [x] Dependency reality check against the registry *before* choosing the engine:
      `torch 2.15.0+cpu`, `numpy 2.5.4`, `pydantic 2.14.2`, `fastapi 0.162.0`, `safetensors 0.8.0`,
      `onnx 1.23.2`, `onnxruntime 1.31.1`, `mcp 1.30.0`, `onnxruntime-web 1.31.1`, `vite 8.3.5`.
- [x] 16 ADRs in `DECISIONS.md`.
- [x] `docs/RESEARCH.md` evidence map + "claims we must not make".
- [x] Risk register (end of this file).
- [x] Licence inventory: code MIT; no third-party weights or datasets bundled; the only shipped model
      was trained in-repo.

Gate: no undocumented assumptions, nothing overwritten. **PASSED.**

## Phase 1 — Foundation, shell, localization

- [x] Monorepo layout: `python/`, `apps/web/`, `docs/`, `scripts/`, `examples/`, `mcp/`.
- [x] Strict pydantic schemas for every public payload; closed enums for operations and states.
- [x] SQLite store, forward-only migrations to schema version 5, idempotency test, head pointer.
- [x] Artifact store with temp-write → hash → atomic rename.
- [x] Web shell: 12 routes, design tokens, dark-first + light theme, 5 accent colours, high contrast,
      reduced motion, skip link, keyboard navigation.
- [x] `en` / `de` / `fr` (260 keys) with a parity test that fails on drift.
- [x] Footer credit "Made with ❤️ by hcsmedia and timfromhcs" in the app, README and release docs.

Gate: `scripts/verify.ps1` green, no dead menu entries. **PASSED.**

## Phase 2 — Hardware detection and resource estimation

- [x] CPU / RAM / disk / OS / Python detection (no psutil dependency; physical cores explicitly
      labelled an estimate).
- [x] Accelerator probing: CUDA/ROCm via torch, MPS, DirectML via import check.
- [x] `declared` vs `usable` separation; a backend is `usable` only after a real matmul ran on it.
- [x] Honest fallback reason recorded per backend and surfaced by CLI/API/UI.
- [x] Resource estimator with explicit assumptions, KV-cache term and a 25% headroom rule.
- [~] Profiles `safe/balanced/performance/custom`: implemented as explicit caps in the estimator and
      trainer config; no separate GUI profile switcher. Declared in `docs/STATUS.md`.
- [~] Fixtures simulate CPU-only and validate the invariant "usable ⇒ benchmarked". No CUDA/ROCm
      hardware exists here, so those paths are unit-tested only as declared/fallback.

Gate: unsupported acceleration degrades with a message and never crashes. **PASSED (CPU-only).**

## Phase 3 — Model factory and exact parameter budgets

- [x] `ModelFamilyAdapter` protocol, registry, family maturity.
- [x] MLP family (arbitrary hidden widths, bias toggle, 3 activations).
- [x] `transformer-lm` family (RoPE, GQA, SwiGLU, RMSNorm, tied embeddings, dropout).
- [x] Exact `numel` enumeration with tied-tensor de-duplication (`shared_tensors`, `tied_to`).
- [x] Formula `predict_parameters` used for pruning only; equality with enumeration asserted in tests
      for 6 configurations including both tie modes.
- [x] Budget solver: 4 policies, constraint validation, distinct-count ranking, infeasibility text,
      verification by instantiation of the top candidates.
- [x] Deterministic build by seed; stable architecture hash across processes.

Gate: displayed count == `sum(numel)` for every family; rebuilt config hashes identically. **PASSED.**

## Phase 4 — Weight Studio

- [x] Tensor metadata: shape, dtype, min/max/mean/std/norm, zero fraction, finiteness, tie target.
- [x] Bounded slice reader (≤4096 numbers) and histogram.
- [x] 12 closed operations: `set_scalar`, `set_block`, `init_seeded`, `add_scaled_delta`,
      `low_rank_delta`, `normalize`, `clamp`, `scale`, `sparsity_mask`, `interpolate`, `copy_from`,
      `set_trainable`.
- [x] Precondition validation (shape, dtype, base hash), parameter caps, non-finite rejection.
- [x] Immutable patch records with author, timestamps, before/after norms, changed element count.
- [x] Content-addressed checkpoints, alias-aware patching (`lm_head.weight` → `tok_emb.weight`),
      head pointer and pointer-based rollback.
- [x] Browser equivalent for MLP weights: 5 operations, effect measurement, checkpoint + rollback.

Gate: apply → diff → evaluate → restart → rollback with provenance. **PASSED.**

## Phase 5 — Baseline training and evaluation

- [x] Lifecycle `validate → estimate → dry_run → train → validate → checkpoint → summarize`.
- [x] AdamW and SGD, warmup + cosine decay, gradient clipping, parameter-group decay policy.
- [x] Deterministic batching, checkpoint cadence, resume from head, pause/resume/cancel,
      `max_seconds` guard, explicit terminal states (`invalid` reserved for evaluator failure).
- [x] Telemetry: loss, val loss, lr, grad norm, tokens, elapsed, peak Python heap.
- [x] Bundled synthetic datasets with hashes and licences, plus a model-adapted MLP task.
- [x] Three evaluation suites with metric definitions, direction, baseline and provenance.
- [x] Evaluation records a run, an evaluation report and an event.

Gate: reproducible run; reload reproduces inference; invalid runs are not zeros. **PASSED.**

## Phase 6 — Weight generation research lab

- [x] Task-descriptor encoder (deterministic random projection), target-checkpoint collection from
      gradient training.
- [x] Conditional hypernetwork (numpy, Adam) for a fixed tiny MLP, with a loss curve.
- [x] Leakage guard: positional split by task seed, overlap raises.
- [x] Out-of-sample evaluation against random init and a gradient-trained baseline, plus a stored
      claim boundary.
- [~] Checkpoint corpus manifest: implemented as the generator report (`corpus` entries with split,
      steps, param norm) rather than a standalone licensed corpus file, because no external corpus is
      used.
- [x] Permutation alignment for one-hidden-layer MLPs: activation correlation + Hungarian assignment,
      with an exact functional check and a brute-force-verified solver.
- [x] Conditional flow matching in weight space over the aligned corpus, Euler sampling, and a report
      that separates reconstruction from out-of-distribution behaviour.
- [x] Measured honestly: reconstruction distance 7e-06, sampling not collapsed, OOD at chance. The
      negative OOD result is published in `docs/PERFORMANCE.md` instead of being tuned away.
- [!] D2NWG-style diffusion and DeepWeightFlow-scale results: still unsupported. They need a real
      corpus (thousands of checkpoints), dataset-conditioned encoders and cross-architecture
      alignment. A toy flow model with the papers' names on it would be a false capability claim.

Gate: generated candidate evaluated out-of-sample and *in*-distribution, alignment verified, no SOTA
language. **PASSED (narrow scope, negative OOD result published).**

## Phase 7 — Evolution and quality-diversity

- [x] Antithetic ES, rank-normalised updates, deterministic replay (identical scores and
      best-parameter hash for the same seed).
- [x] MAP-Elites archive with score/latency/size descriptors and coverage reporting.
- [x] Example-matched gradient baseline in an isolated scratch workspace.
- [x] Published result: gradient training wins at equal example budget (0.72 vs 0.39 on the demo task) —
      recorded in `docs/PERFORMANCE.md` instead of being hidden.
- [~] CMA-ES adapter: not implemented, documented as optional.

Gate: reproducible improvement and diversity with a disclosed comparison. **PASSED.**

## Phase 8 — LLM / text diffusion / image diffusion

- [x] Autoregressive character transformer: training, sampling-free evaluation, ONNX export.
- [!] Masked diffusion LM: unsupported (no trainer/sampler).
- [!] Latent image diffusion / DiT: unsupported.
- [x] ONNX export with a numerical parity gate (relative error ≤ 1e-4; measured 7.15e-07).

Gate partly met; every unsupported item is labelled in `docs/STATUS.md` **and** in the UI. **PASSED
WITH DECLARED GAPS.**

## Phase 9 — Reward / RL trainers

- [x] Reward interface with an exact verifiable reward and a **sealed fixture**: the rule is written
      read-only, hash-pinned, and re-verified before scoring.
- [x] GRPO: group-relative advantages, frozen reference policy, KL penalty, gradient clipping, and an
      exact-KL diagnostic beside the sampled estimator used in the loss.
- [x] Independent verification: the saved policy is re-scored in a separate process; a test tampers
      with the fixture and asserts the verifier refuses to score.
- [x] Measured result: 0.891 held-out accuracy on 256 unseen contexts vs 0.375 reference, 0.250
      chance, ~4 s. `hcs-foundry rl grpo` and `POST /api/v1/reinforcement-runs`.
- [!] DPO, PPO, RLHF and GRPO over token-level LLM completions: still unsupported. They need rollout
      infrastructure, a reward model or text verifier, and a compute budget this build does not have.

Gate: toy environment runs, the evaluator is independent, and the reward is checked against held-out
task success. **PASSED (toy scope, labelled).**

## Phase 10 — Agent Lab and MCP

- [x] Proposal schema, policy (allowlist, forbidden ops, size caps), approval requirement.
- [x] Isolated verifier: separate process, JSON argv, minimal environment, no DB access.
- [x] MCP server: 12 read-only tools, 7 approval-gated tools, resources, prompts, audit log.
- [x] Browser mirror of the contract (drafting level only), clearly labelled as such.
- [~] Verified through the SDK's in-process client rather than an external host application.

Gate: read-only tool works; gated tool refuses without approval; calls are audited. **PASSED.**

## Phase 11 — Browser / Vercel

- [x] Static Vite build; no server functions, no proxy, no remote inference.
- [x] ONNX Runtime Web: WebGPU attempted, WASM fallback, provider reported only after real inference.
- [x] Shipped demo model (250,432 parameters) exported with parity and a manifest recording how the
      numbers were produced.
- [x] Browser MLP trainer in a Web Worker with stop-between-steps.
- [x] Browser weight studio, weight-space interpolation, evaluation.
- [x] IndexedDB history, JSON bundle export/import, storage estimate, persistence request, delete-all.
- [x] Capability parity matrix in-app and in `docs/STATUS.md`.
- [x] `vercel.json` for a static deploy (not deployed — see Phase 14).

Gate: no hidden backend calls; WASM path verified; WebGPU labelled experimental. **PASSED.**

## Phase 12 — API, CLI, packaging, desktop

- [x] Versioned REST API, OpenAPI, bearer auth, CORS allowlist, rate limits, SSE, typed errors.
- [x] CLI covering doctor/model/solve/preset/weights/train/eval/evolve/generate/export/history/
      proposal/api/mcp, with `--json`, position-independent global flags and stable exit codes.
- [x] Wheel builds; installed into a clean venv **without torch** and `doctor` works.
- [x] Dependency inventory script (`scripts/dependency_inventory.py`) and release verifier.
- [!] npm/PyPI publication: not executed (ADR-016).
- [!] Tauri desktop shell and portable launcher: not built.

Gate: clean-venv install and CLI start verified locally. **PASSED for the wheel; publication open.**

## Phase 13 — QA gates

- [x] 99 pytest tests, 40 vitest tests, ruff clean, TypeScript strict clean.
- [x] Numerical checks: gradient reachability, parameter-count equality, ONNX parity, patch finiteness.
- [x] Security tests: auth, CORS, rate limit, approval gate, dangerous-primitive scan.
- [x] i18n parity test.
- [x] Manual keyboard + responsive checks (390 px and desktop) via a real browser; no horizontal
      overflow measured.
- [x] Automated accessibility audit: 6 axe-core tests over 4 screens in 2 locales. The first run found
      a real bug (unlabelled sidebar select), now fixed.
- [~] Colour contrast stays a manual check: jsdom has no layout engine, so axe cannot evaluate it.
- [x] End-to-end smoke workflow in `scripts/verify.ps1`.

## Phase 14 — Publish

- [!] Not executed, by policy. No push, no deploy, no package publication, no credentials exercised.
      `AGENTS.md` requires explicit human approval per artifact; `scripts/verify_release.py` exists to
      gate that step when it is approved.

---

## Session 2 — 0.2.0: audit, fix the drift, publish

The milestone contract is `docs/ROADMAP-0.2.0.md`. This section records what actually happened.

### Why the session existed

An audit of the tree against current upstream documentation found two drifts that each invalidated a
headline claim. Neither was a bug in the sense of "wrong output"; both were the software quietly
disagreeing with what we said about it.

| Drift | What the docs claimed | What the code did |
|---|---|---|
| MCP | 2026-07-28 revision (AGENTS.md §4.3) | `mcp>=1.2,<2` + `from mcp.server.fastmcp import FastMCP` → SDK v1 maintenance line |
| Browser | WebGPU execution provider | bare `onnxruntime-web` import → the JSEP backend, deprecated in ORT 1.29.0 |

### Closed

- [x] MCP migrated to SDK v2 (`mcp>=2.3,<3`, `MCPServer`). Verified end-to-end over stdio with the v2
      `Client` against a real subprocess: 20 tools, 3 resources, 3 prompts, read tool works, gated tool
      refuses without approval.
- [x] Found and fixed a second MCP defect: `hcs-foundry mcp` accepted no `--approve` and no `serve`
      positional, so the documented approval path was env-only. Now repeatable and forwarded.
- [x] Browser moved to the native WebGPU EP (`onnxruntime-web/webgpu`). Proven by the build output:
      `ort-wasm-simd-threaded.jsep-*.wasm` → `ort-wasm-simd-threaded.asyncify-*.wasm` +
      `ort.webgpu.bundle.min-*.js`, bundle size essentially unchanged.
- [x] Found and fixed a fake control: the Dashboard hard-coded `"auto"`, so the Settings provider
      choice did nothing. Now honoured, and the session is rebuilt when the preference changes.
- [x] Found and fixed an unbounded hang: switching WASM → WebGPU in one page never settled. Every
      provider attempt is now bounded at 20 s and a timeout is reported as a fallback reason.
- [x] Found and fixed console noise: ORT logged 2 node-assignment advisories per session at error
      level. Now `logSeverityLevel: 3`; measured 0 console errors.
- [x] Re-measured the browser and **changed the conclusion**: for the 250k demo, WASM warm (2.7 ms
      median) is ~6× faster than WebGPU warm (~16.8 ms), and WebGPU pays a 0.25–0.4 s cold start. The
      retired "18.2 ms on WebGPU" figure was JSEP.
- [x] Release gate hardened: `scripts/release.ps1` (fail-fast, all gates) and a rewritten
      `scripts/verify_release.py --strict` (version consistency across four files, changelog entry,
      brand credit, acceptance-matrix integrity, secrets, U+FFFD scan, demo-manifest hash, MCP pin
      agreement, built-site externals). The gate proved itself by catching two real problems during
      this session and by failing on a deliberately broken version string.
- [x] 121 Python tests, 48 web tests, ruff clean, TypeScript strict clean.
- [ ] Publishing: repository, tag, Vercel deployment — see the release section below.

### Numbers

| Deliverable | Session 1 | Session 2 |
|---|---|---|
| Python tests | 118 | **121** |
| Web tests | 40 | **48** |
| MCP SDK | v1 (`<2`) | **v2 (2.3.0)** |
| MCP protocol revision | 2025-era | **2026-07-28** |
| Browser WebGPU backend | JSEP (deprecated) | **native EP** |
| Console errors on Dashboard | 4 | **0** |
| Release gates | ad hoc | **one command, strict** |

---

## Delivered in session 1

| Deliverable | Count / result |
|---|---|
| Python modules | 34 files (engine, storage, interfaces, verifiers) |
| Python tests | 118 passed |
| Web tests | 40 passed (incl. 6 accessibility audits) |
| CLI leaf commands | 27 |
| MCP tools | 20 (13 read-only, 7 gated) |
| API endpoints | 34 |
| Weight operations | 12 (engine) + 5 (browser) |
| RL environments | 1 (rule-bandit, sealed fixture) |
| Weight generators | 2 (hypernetwork, conditional flow matching) |
| Lessons | 20 |
| ADRs | 16 |
| Bundle (gzip) | ~109 kB JS + 3.5 kB CSS + 29.5 MB ONNX runtime WASM |

## Backlog (explicitly open, not forgotten)

1. Corpus-scale weight generation: thousands of aligned checkpoints, a diffusion trainer, and
   cross-architecture alignment (Git Re-Basin beyond one hidden layer).
2. DPO/PPO/RLHF and token-level GRPO over transformer completions (the bandit covers the algorithm).
3. Masked diffusion language model trainer/sampler.
4. Tauri desktop shell + single-file launcher; npm/PyPI publication with provenance.
5. Visual regression snapshots (axe is in place; pixel diffs are not).
6. WebGPU operator coverage matrix per model family.
7. External benchmark adapters (LiveBench, BFCL) with protocol files.

## Risk register

| # | Risk | Mitigation | State |
|---|---|---|---|
| R1 | Overclaiming weight generation | maturity labels, claim boundaries in payloads, no-SOTA rule | active |
| R2 | Parameter-count drift between UI and engine | single source of truth: enumeration, tested against the formula | active |
| R3 | Unsafe checkpoint import | safetensors only, tensor-set validation, size caps | active |
| R4 | Local API exposed beyond loopback | loopback default, token required, CORS allowlist, warning | active |
| R5 | Benchmark tampering by an agent | evaluator in a separate process, read-only fixtures, hash recorded | active |
| R6 | torch missing on a user machine | lazy imports, actionable `missing_dependency`, clean-venv test | active |
| R7 | Determinism drift across platforms | seeds/versions recorded; same-stack equality tested, cross-machine not claimed | active |
| R8 | Scope inflation | every feature is implemented+tested or declared `[!]` | active |
| R9 | Browser storage eviction | estimate + persist request + export bundle + warnings | active |
