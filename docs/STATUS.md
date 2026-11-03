# Capability status

Authoritative, human-readable matrix. The machine-readable version is `docs/acceptance.json`, and the
in-app parity matrix lives on the Settings screen.

Legend: **stable** (implemented, tested, documented) · **experimental** (implemented, narrow, evidence
limited) · **unsupported** (no validated implementation in this build).

Last verified: 2026-11-03, on Windows 11 (Ryzen 7 7735HS, CPU-only torch 2.15.0, Node 22, Chromium).

## Engine

| Capability | Status | Evidence |
|---|---|---|
| Hardware detection, declared vs usable backends | stable | `hardware.detect_hardware`, `doctor --json`, micro-benchmark per backend |
| Resource estimation with stated assumptions | stable | `hardware.estimate_resources`, tests for monotonicity and budget warnings |
| MLP family + exact parameter enumeration | stable | `models/mlp.py`, `test_enumerated_parameters_match_formula` |
| Decoder transformer (RoPE, GQA, SwiGLU, RMSNorm) | stable | `models/transformer_lm.py`, smoke tests, ONNX parity |
| Parameter budget solver (4 policies, verification) | stable | `models/solver.py`, 6 solver tests |
| Tensor inspection, slices, histograms | stable | `weights.summarize/slice_tensor/histogram` |
| Weight patches (12 closed operations) | stable | `weights.apply_operation`, per-operation tests |
| Patch provenance, head pointer, rollback | stable | `services.apply_patch/rollback`, storage tests |
| AdamW/SGD training, resume, pause/cancel, dry-run | stable | `training/trainer.py`, workflow tests |
| Evaluation suites with definitions + `invalid` state | stable | `evaluation.py`, invalid-run test |
| ONNX export with numerical parity gate | experimental | `exporting.py`; parity 7e-07 on the demo model |
| Evolution Strategies + MAP-Elites | experimental | `evolution.py`; tiny MLP only, example-matched baseline |
| Conditional hypernetwork generator | experimental | `generators.py`; one MLP family, 3 held-out tasks |
| Weight-space diffusion (D2NWG-like) | unsupported | interface slot only |
| Flow matching (DeepWeightFlow-like) | unsupported | interface slot only |
| LoRA fine-tuning of a pretrained base | unsupported | low-rank *edits* exist; no base model shipped |
| DPO / PPO / GRPO trainers | unsupported | reward interface only |
| Diffusion language model | unsupported | family slot reserved |
| Latent image diffusion | unsupported | not attempted |
| Quantisation | unsupported | not attempted |

## Interfaces

| Capability | Status | Evidence |
|---|---|---|
| CLI with JSON output and stable exit codes | stable | `test_interfaces.py` CLI tests |
| Local HTTP API, bearer auth, CORS, rate limits, SSE | stable | API tests including auth, CORS, background job |
| OpenAPI document | stable | `/api/v1/openapi.json`, asserted in tests |
| MCP server (stdio) with approval gate + audit log | experimental | `build_server`, gate and audit tests |
| PyPI package | unsupported | wheel builds; not published |
| npm package | unsupported | web app builds; not published |
| Desktop shell (Tauri) / single-file launcher | unsupported | not built |

## Browser

| Capability | Status | Evidence |
|---|---|---|
| Static build, no server functions | stable | `npm run build`, `dist/` only |
| ONNX Runtime Web inference | experimental | verified in Chromium: WebGPU provider, 18.2 ms, argmax returned |
| Provider fallback WebGPU → WASM | stable | probe runs a real inference before reporting |
| Browser MLP training in a Worker | experimental | 200 steps, val loss 0.62, 76.6% accuracy, 105 ms |
| Weight inspection + patching in browser | experimental | 384 changed elements, −2.000% norm, new checkpoint |
| IndexedDB history, export/import bundle, delete-all | stable | `db.test.ts` |
| Localization en/de/fr | stable (UI) | key parity test, 260 keys |
| Long-form lesson translation | unsupported | English only; stated in the UI |

## Research claims

| Claim | Allowed? | Why |
|---|---|---|
| "The pipeline runs end to end" | yes | tests and reproducible commands |
| "A 250k transformer learns the synthetic corpus" | yes, with the corpus named | val loss 2.30 vs 4.16 uniform baseline |
| "Generated weights beat gradient training" | **no** | no fair comparison in this build |
| "State of the art" | **no** | no external benchmark was run |
| "WebGPU is supported" | only as "works for this graph in this browser" | measured once, per browser |
| "The browser equals the desktop engine" | **no** | parity matrix says otherwise |

## Known limitations

1. The default workspace is `~/.hcs-foundry`; it is deleted only by you.
2. `physical_cores` is an estimate (no psutil dependency).
3. Cross-machine bit-exact reproducibility is not claimed; same-stack reproducibility is tested.
4. Browser training is one hidden layer by design.
5. The demo ONNX model was trained for 40 steps; it demonstrates plumbing, not quality.
6. WebGPU depends on the installed browser/GPU; the WASM path is the tested default.

---

**Made with ❤️ by hcsmedia and timfromhcs**
