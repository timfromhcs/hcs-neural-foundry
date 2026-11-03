# Changelog

All notable changes to this project are documented here. Format follows Keep a Changelog; versioning
follows Semantic Versioning.

## [0.2.0] — 2026-11-03

First published release. Two dependency drifts were found by auditing the tree against current
upstream documentation, and both invalidated a headline claim; both are fixed here.

### Fixed — protocol and runtime drift

* **MCP server migrated to SDK v2** (`mcp>=2.3,<3`). `FastMCP` was renamed to `MCPServer` in SDK
  v2.0.0 (released 2026-07-28) and `mcp.server.fastmcp` no longer exists, so the old import silently
  pinned the server to the v1 maintenance line. The server now speaks the 2026-07-28 revision and
  still serves 2025-era clients.
* **Verified against a real client, not just in-process.** A new end-to-end probe connects over
  stdio with the SDK v2 `Client` and asserts 20 tools, 3 resources, 3 prompts, a working read tool and
  a gated tool that refuses without approval.
* **`hcs-foundry mcp serve --approve TOOL` actually works.** The CLI accepted no `--approve` and no
  positional, so approval was only possible through `HCS_FOUNDRY_MCP_APPROVE`. The documented
  flag now exists, is repeatable, and is forwarded to the server.
* **Browser runtime moved to the native WebGPU execution provider.** `onnxruntime-web` deprecated
  WebGL and JSEP in v1.29.0 (microsoft/onnxruntime#29716); the bare entry point resolved to JSEP.
  The app now imports `onnxruntime-web/webgpu`. The build proves it: the bundled wasm changed from
  `ort-wasm-simd-threaded.jsep-*.wasm` to `ort-wasm-simd-threaded.asyncify-*.wasm` plus
  `ort.webgpu.bundle.min-*.js`, at essentially unchanged bundle size (108.7 kB gzip JS).

### Fixed — behaviour that did not match the UI

* **The provider setting was inert on the Dashboard.** It hard-coded `"auto"` in both `backendStatus`
  and `runDemoModel`, so choosing WASM in Settings changed nothing. It now honours the preference, and
  the cached session is rebuilt when the preference changes.
* **A provider switch could hang the UI forever.** Moving WASM → WebGPU in one page left the runtime
  half-initialised and the probe spun indefinitely. Every provider attempt is now bounded (20 s) and a
  timeout is reported as a normal fallback reason.
* **ONNX Runtime advisory noise is no longer logged as console errors.** Two "some nodes were not
  assigned to the preferred EP" messages per session were emitted at error level; sessions now set
  `logSeverityLevel: 3`. Measured result: 0 console errors on the Dashboard.

### Changed — claims now match measurements

* WebGPU is reported only after a real inference ran on it, and only when `requestAdapter()` returned
  an adapter — API presence alone is no longer treated as support.
* The browser performance claim was re-measured on the new code path and it changed the conclusion:
  see `docs/PERFORMANCE.md`. For the shipped 250k-parameter demo, **WASM warm is ~6× faster than
  WebGPU warm**, and WebGPU pays a 0.25–0.4 s cold start. The previous "WebGPU is fast" framing is
  retired.

### Added — release engineering

* `scripts/release.ps1`: one command, all gates, fail-fast, printable table.
* `scripts/verify_release.py` rewritten: version consistency across four files, changelog entry for
  the current version, brand credit, acceptance-matrix integrity, MCP SDK major-version agreement
  with the pin, encoding scan, and a check that the built site references no external origin.
* `.github/workflows/release.yml`: build wheel + SBOM + checksums on a tag. It attaches artifacts and
  does **not** publish to a registry.
* `docs/ROADMAP-0.2.0.md`: the milestone contract, including the research that forced the two fixes.

### Added — previously unreleased work folded into this version

* **GRPO trainer** on a toy contextual bandit with an exact verifiable reward:
  `hcs_foundry/rl.py`, `hcs-foundry rl grpo`, `POST /api/v1/reinforcement-runs`.
  Group-relative advantages, frozen reference policy, KL penalty, exact-KL diagnostic.
  0.891 held-out accuracy vs 0.375 reference on 256 unseen contexts.
* **Sealed reward fixture**: written read-only, hash-pinned, and re-verified before scoring. A test
  tampers with it and asserts the isolated verifier refuses to produce a score.
* **Independent policy verification** in a child process (`hcs_foundry.rl_verify`).
* **Automated accessibility gate**: six axe-core tests over Model Studio, Settings, Docs and API/MCP in
  English and German. The first run found a real unlabelled-select bug, now fixed.
* `scripts/scan_encoding.py` to catch replacement characters in tracked text files.

### Changed

* `RunRecord.kind` accepts `reinforcement`; the web types follow.
* `Select` in the web UI now requires an accessible `label`, not an `aria-label` it silently dropped.

### Added — weight-space generation

* **Permutation alignment** (`alignment.py`): activation-correlation cost + Hungarian assignment,
  exact functional verification, corpus alignment. Solver checked against brute force.
* **Conditional flow matching** (`flow.py`): velocity field `v(x_t, t, descriptor)` over an aligned
  checkpoint corpus, per-dimension target standardisation, Euler sampling, reconstruction *and*
  out-of-distribution reporting, sampling-diversity diagnostic.
* `hcs-foundry generate --method flow` and a new `flow-mlp` preset (one hidden layer).

**Measured (and published):** reconstruction of training checkpoints at relative parameter distance
7e-06; sampling diversity 1.41 (not collapsed); out-of-distribution accuracy **at chance**, a
corpus-size limitation recorded in `docs/PERFORMANCE.md` rather than tuned away.

### Still unsupported


DPO, PPO, RLHF, token-level GRPO over LLM completions, D2NWG-scale weight diffusion, cross-architecture
alignment, diffusion language models, latent image diffusion, quantisation, desktop packaging, and
package publication.

## [0.1.0] — 2026-11-03

First vertical slice. Everything below is implemented in this tree; anything not listed is either
declared unsupported in `docs/STATUS.md` or absent.

### Added

**Engine**
* Exact parameter enumeration with tied-tensor de-duplication for MLP and decoder-transformer families.
* Parameter budget solver with `exact`, `closest`, `closest_under`, `at_least` policies; formula used
  only for pruning, every reported count verified by instantiating the module.
* Resource estimator with explicit assumptions and budget warnings.
* Hardware detection separating `declared` from `usable` backends, with a real micro-benchmark per
  accelerator probe.
* Twelve closed weight-patch operations with precondition, finiteness and size validation.
* Content-addressed safetensors checkpoints, tied-alias sidecar, atomic promotion, model head pointer
  and pointer-based rollback.
* Supervised training with AdamW/SGD, warmup + cosine decay, gradient clipping, dry-run gradient check,
  pause/resume/cancel, checkpoint cadence and explicit terminal states.
* Three evaluation suites (`text-smoke`, `shapes-smoke`, `weight-sanity`) with metric definitions,
  direction and baselines; evaluator failures produce `invalid`, never `0.0`.
* Antithetic Evolution Strategies with rank-normalised updates, MAP-Elites archive and an
  example-matched gradient baseline.
* Conditional hypernetwork weight generator with a task-level leakage guard and out-of-sample evaluation.
* ONNX export with numerical parity verification (relative error gate `1e-4`).

**Interfaces**
* CLI (`hcs-foundry`) with `--json`, `--dry-run`, stable exit codes and shell-independent flag handling.
* Versioned local HTTP API: bearer auth, CORS allowlist, rate limits, SSE progress, typed errors,
  OpenAPI, background training jobs with pause/resume/cancel.
* MCP server over stdio: 12 read-only tools, 7 approval-gated tools, resources, prompts, audit log.
* Service layer shared by all three transports so behaviour cannot drift.

**Web app**
* Static Vite/React/TypeScript client: 12 routes, dark-first token-based design system, responsive down
  to 390 px, keyboard navigation, reduced-motion and high-contrast support.
* ONNX Runtime Web inference with WebGPU → WASM fallback that only reports a provider after a real
  inference succeeded.
* Browser MLP training in a Web Worker (real SGD + momentum, stoppable between steps).
* Browser weight studio: statistics, heatmap, histogram, five patch operations, effect measurement,
  immutable checkpoints and rollback.
* Weight-space interpolation between browser checkpoints with per-`t` evaluation.
* IndexedDB persistence with export/import bundle, storage estimates and delete-all.
* English/German/French interface with a key-parity test.
* In-app capability parity matrix and maturity badges.

**Documentation**
* `docs/RESEARCH.md` evidence map and the "claims we must not make" list.
* `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `docs/SECURITY.md`, `docs/STATUS.md`,
  `docs/PERFORMANCE.md` with measured numbers.
* Twenty-lesson curriculum with a consistent contract per lesson.
* `docs/acceptance.json` machine-readable acceptance matrix.

### Declared unsupported in this release

Weight-space diffusion (D2NWG-like), flow matching (DeepWeightFlow-like), DPO/PPO/GRPO training,
diffusion language models, latent image diffusion, quantisation, desktop packaging, and package
publication to npm/PyPI. Each is listed with its reason in `docs/STATUS.md`.

### Not done by policy

No repository push, no Vercel deployment, no package publication. External publication requires explicit
human approval per artifact (ADR-016).
