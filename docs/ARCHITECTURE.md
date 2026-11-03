# Architecture

## 1. One engine, three transports, one policy layer

```text
                 ┌──────────────────────────────┐
   CLI ─────────►│                              │
   HTTP API ────►│   services.Foundry           │──► models/*      (architecture, exact counts)
   MCP server ──►│   (validation, policy,       │──► weights       (closed patch operations)
                 │    provenance, orchestration)│──► training/*    (AdamW, datasets, checkpoints)
                 └──────────────────────────────┘──► evaluation    (suites, invalid handling)
                              │                    ──► evolution     (ES + MAP-Elites)
                              │                    ──► generators    (hypernetwork)
                              ▼                    ──► exporting     (ONNX + parity)
                    storage (SQLite + content-addressed files)
```

There is exactly one place where a decision becomes an effect: `python/hcs_foundry/services.py`.
The CLI, the HTTP API and the MCP server are thin adapters that call it. This is what keeps the
three interfaces from drifting apart in behaviour or policy.

## 2. Module map

| Module | Responsibility | Notably does **not** |
|---|---|---|
| `schemas.py` | Every public payload, validated with pydantic; closed enums for operations, states, families | import torch |
| `hardware.py` | Detection, declared-vs-usable backends, resource estimation with stated assumptions | claim an accelerator without a probe |
| `models/base.py` | Handle, exact `numel` accounting, tied-tensor de-duplication | estimate counts from a formula at report time |
| `models/mlp.py`, `models/transformer_lm.py` | Real PyTorch modules + exact parameter formula for pruning | load pretrained checkpoints |
| `models/solver.py` | Budget search over discrete architecture space, policy handling, verification by instantiation | promise an exact count when the space is discrete |
| `weights.py` | Tensor statistics, bounded slices, 12 closed patch operations, effect measurement | evaluate strings, touch disk, mutate inputs |
| `checkpoints.py` | safetensors I/O, tied-alias sidecar, atomic writes, architecture-hash gate | accept pickle files |
| `training/*` | Datasets, trainer lifecycle, AdamW/SGD, resume, dry-run, telemetry | run an evaluator |
| `evaluation.py` | Suites with metric definitions, provenance, `invalid` on evaluator failure | let a caller pass a bare score |
| `evolution.py` | Antithetic ES, MAP-Elites, example-matched gradient baseline | search huge raw parameter vectors |
| `generators.py` | Conditional hypernetwork over task descriptors, leakage guard | pretend to implement diffusion/flow matching |
| `exporting.py` | ONNX export + parity check against the native model | claim parity without running the graph |
| `agents.py` | Policy, proposal construction, isolated verifier subprocess | give an agent a tensor handle |
| `storage.py` | SQLite migrations, repositories, head pointer, events, audit log | hold large blobs in the database |
| `api.py` | FastAPI app, bearer auth, CORS allowlist, rate limits, SSE | expose shell or arbitrary Python |
| `mcp_server.py` | FastMCP tools/resources/prompts, approval gate, audit log | expose generic filesystem or shell access |
| `cli/main.py` | Argument parsing, human/JSON output, exit codes | contain business logic |

## 3. Data flow for one weight edit

```text
1. CLI/API: build WeightPatch {base_checkpoint_sha256, architecture_hash, operation, parameters}
2. Foundry.apply_patch
   ├─ model lookup, architecture hash check
   ├─ base checkpoint lookup by content hash
   ├─ load_state() (verifies tensor set, restores tied aliases)
   ├─ alias resolution (e.g. lm_head.weight → tok_emb.weight)
   ├─ weights.apply_patch(state, patch) on a copy
   │    └─ rejects unknown params, shape/dtype mismatch, non-finite result
   ├─ load patched state back, re-tie aliases
   ├─ save_checkpoint() → content hash + manifest, atomic promote
   ├─ repo.add_checkpoint() → moves the model head
   ├─ repo.add_patch() → immutable record
   └─ log_event()
3. Caller receives patch id, validation (changed elements, max delta, norm change), checkpoint record
```

Rollback moves the head pointer. Checkpoints are immutable; nothing is deleted, so the audit trail is
complete and a rollback can itself be rolled back.

## 4. Storage

```
<workspace>/
├── foundry.db          # SQLite, WAL, forward-only migrations (schema version 5)
├── checkpoints/        # <sha256>.safetensors + <sha256>.json manifest (aliases, config, origin)
├── artifacts/          # content-addressed reports (sha256[:2]/sha256.json)
├── datasets/           # generated synthetic corpora (regenerable, hashed)
├── exports/            # ONNX files
└── logs/
```

`model_heads` holds the only mutable pointer per model: `save → head = new`, `rollback → head = old`.
Timestamps are deliberately *not* used for ordering (they have one-second resolution and collide).

## 5. Concurrency model

* Single process. The HTTP API runs handlers in a thread pool; the SQLite connection is guarded by a
  re-entrant lock and uses `BEGIN IMMEDIATE` for write transactions.
* Training runs either in the request thread (CLI, MCP) or in a background thread with a
  `TrainerControl` (pause/resume/cancel) managed by `api.JobManager`.
* The evaluator for agent proposals is a separate **process** by design (ADR-008).
* The browser runs its own trainer in a Web Worker; nothing is shared with the engine.

## 6. Browser application

```
apps/web/src
├── main.tsx / App.tsx        # hash router, page-level error boundary
├── state.tsx                 # IndexedDB workspace, demo manifest, preferences
├── i18n.tsx                  # typed translation function
├── locales/{en,de,fr}.ts     # flat key/value dictionaries (parity tested)
├── styles.css                # design tokens, no CSS framework
├── lib/
│   ├── mlp.ts                # tiny MLP, real backprop, patches, interpolation, count formulas
│   ├── onnx.ts               # provider probing (WebGPU → WASM), inference, prompt encoding
│   ├── db.ts                 # IndexedDB stores, export/import bundle, storage estimates
│   ├── demo.ts               # shipped manifest loader
│   ├── format.ts             # locale-aware formatting, hashing
│   └── types.ts              # wire types
├── workers/mlp.worker.ts     # incremental trainer, stoppable between chunks
├── components/               # shell, navigation, cards, charts, badges
└── pages/                    # 12 routes
```

Design constraints that shaped it:

* **No server.** `vite build` produces static assets only. There is no `api/` directory and no proxy.
* **No CDN.** ONNX Runtime's WASM binary is emitted by Vite as a local asset.
* **No faked data.** Numbers shown as "measured" come from either the shipped manifest (desktop engine,
  recorded) or the browser's own computation. Anything else is labelled with a maturity badge.

## 7. Boundary decisions

| Boundary | Rule | Enforcement |
|---|---|---|
| Agent → tensors | never direct; proposals only | `AgentProposal` schema, `AgentPolicy.check` |
| Proposer → score | separate process | `agents.verify_isolated` + `agent_verify.py` |
| UI → parameter count | one source of truth | `count_parameters` enumerated in `models/base.py` |
| Browser → network | only optional loopback API | loopback hostname check in `ApiMcp.tsx` |
| Checkpoint → disk | content hash + atomic rename | `checkpoints.save_checkpoint`, `hashing.atomic_write` |
| Patch → checkpoint | new blob, head pointer moves | `services.apply_patch` |
