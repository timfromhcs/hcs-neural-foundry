# Changelog

All notable changes to this project are documented here. Format follows Keep a Changelog; versioning
follows Semantic Versioning.

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
