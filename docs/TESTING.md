# Testing and quality gates

## 1. What the suites cover

### Python (`python/tests`, pytest, 85 tests)

| File | Focus |
|---|---|
| `test_core.py` | schemas, patch validation, hashing stability, hardware probing, resource estimator, migrations, repository, exact parameter counts, budget solver policies |
| `test_workflows.py` | checkpoint round-trip and tie restoration, safetensors-only enforcement, every patch operation, non-finite rejection, training convergence, resume, cancel, evaluation suites, invalid-run handling, ES reproducibility, agent policy, dangerous-primitive scan |
| `test_interfaces.py` | CLI contract and exit codes, API auth/CORS/rate-limit/error shape, background job lifecycle, MCP approval gate and tool registration/audit |
| `test_generators.py` | leakage guard, non-MLP rejection, end-to-end generator run |

### Web (`apps/web/tests`, vitest, 34 tests)

| File | Focus |
|---|---|
| `i18n.test.ts` | en/de/fr key parity, no empty or placeholder values, no HTML in strings, interpolation |
| `mlp.test.ts` | deterministic init, forward shapes, training convergence, reproducibility, stop behaviour, every patch op, serialisation, interpolation, count formulas |
| `db.test.ts` | IndexedDB round-trips, filtering, bundle export/import, foreign-JSON rejection |

## 2. Running the gates

```powershell
# Python
.\.venv\Scripts\python.exe -m pytest python/tests -q
.\.venv\Scripts\python.exe -m ruff check python

# Web
cd apps\web
npm run typecheck
npm test
npm run build

# Everything, plus the smoke workflow and docs checks
.\scripts\verify.ps1
```

## 3. Numerical correctness

* **Gradient reachability**: `SupervisedTrainer.dry_run()` proves a backward pass touches parameters
  and reports the gradient norm; the trainer refuses to start a run without it.
* **Parameter counts**: `predict_parameters` (formula) is asserted equal to enumerated `numel` for
  every family and for the documented configurations, including a tied-embedding case where the
  difference must be exactly `vocab × d_model`.
* **ONNX parity**: `exporting.export_onnx` runs the graph in ONNX Runtime and compares to the native
  output; relative error above 1e-4 aborts the export.
* **Patch effects**: every operation is checked for finiteness before persistence; the reported norm
  change is computed from the actual tensors, not predicted.
* **Fold-safe evaluation**: the generator splits tasks positionally and asserts disjointness; a leaky
  corpus raises rather than trains.

## 4. What is *not* covered (and why)

| Gap | Reason | Mitigation |
|---|---|---|
| CUDA/ROCm training lanes | No such hardware in CI | backend marked unusable without a probe; tests simulate CPU-only |
| WebGPU in CI | Browser/GPU dependent | provider is only reported after a real inference; WASM is the tested fallback |
| Real benchmark scores (LiveBench, BFCL, …) | Requires network, licences, harnesses | suites are named `*-smoke` and documented as pipeline checks |
| Full desktop packaging | Not built in this session | declared unsupported in `docs/STATUS.md` |
| Long-running training correctness | Time budget | smoke runs assert convergence direction, not final quality |

## 5. Anti-regression rules

1. A failing test is never deleted or skipped to get green CI; it is fixed or the expectation is
   corrected with a documented reason.
2. Coverage thresholds are not lowered to accommodate new code.
3. `test_runtime_modules_avoid_dangerous_primitives` greps the package for `eval(`, `exec(`,
   `pickle.load`, `os.system(` and `shell=True`; adding one fails the suite.
4. The locale parity test fails on any key drift, so a half-translated screen cannot ship.
5. Benchmark fixtures are generated from code with fixed seeds; editing them changes their hash and
   shows up in every report.

## 6. Determinism notes

* Torch determinism is requested with `use_deterministic_algorithms(warn_only=True)`; some CPU kernels
  are still order-dependent across BLAS builds, so cross-machine bit-exactness is **not** claimed.
* Within one environment, the same seed reproduces the same loss curve; the ES test asserts identical
  scores and identical best-parameter hashes.
* Latency and memory measurements are wall-clock/allocator observations and are explicitly excluded
  from reproducibility assertions.

## 7. Evidence from the last full run

Recorded in `docs/PERFORMANCE.md` (numbers, machine, date). Regenerate with `scripts/verify.ps1`.
