# Performance and measured behaviour

All numbers below were measured on this machine on **2026-11-03**. They are single-run observations,
not benchmarks. Commands are included so anyone can reproduce or contradict them.

## Machine

| Component | Value |
|---|---|
| CPU | AMD Ryzen 7 7735HS (16 logical cores) |
| RAM | 21.2 GB visible to the process |
| Disk free | 111.5 GB at measurement time |
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.13 |
| torch | 2.15.0+cpu |
| Node | 22.23.1 |
| Browser | Chromium (Playwright-driven), WebGPU available |

GPU note: the integrated Radeon 680M is not used by this build. torch reports
`cuda.is_available() == False`, so the engine runs CPU-only; the browser uses its own WebGPU/WASM path.

## Engine timings

| Operation | Model | Time |
|---|---|---|
| `model create` (solve + build + init checkpoint) | 250k transformer | ~0.6 s |
| Training, 40 steps, batch 8, block 16 | 250,432 parameters | 1.44 s (27.8 steps/s) |
| Evaluation, `text-smoke`, 4 batches | same | ~0.15 s |
| ONNX export + parity check | same | ~0.9 s, parity `7.15e-07` |
| Weight patch `low_rank_delta` (rank 2) | 64×64 tensor | ~0.05 s |
| `doctor --no-probe` | — | ~0.4 s |
| `doctor` with backend probes | — | ~0.7 s (includes 256×256 matmul benchmark) |

The demo model shipped to the browser (`scripts/build_web_demo.py`: solve → create → train 40 steps →
evaluate → export ONNX → write manifest) completes in **8.7 s** end to end.

## Test and build timings

| Gate | Result | Time |
|---|---|---|
| `pytest python/tests` | 85 passed | 25.5 s |
| `npm test` (vitest) | 34 passed | 13.2 s |
| `npm run build` (typecheck + vite) | success | 4.8 s |
| `ruff check python` | clean | < 1 s |

The slowest single test is the generator end-to-end run (training several tiny MLPs); it is marked
`heavy` so a fast loop can deselect it.

## Browser measurements (Chromium)

| Measurement | Value | Where |
|---|---|---|
| ONNX Runtime provider selected | **WebGPU** (after a real smoke inference) | Dashboard → Run |
| Single inference latency | **18.2 ms** for 8 prompt tokens, 250k model | Dashboard |
| ONNX model size | 1.15 MB (fp32) | `demo-manifest.json` |
| Runtime WASM asset | 29.5 MB (jsep, includes WebGPU + CPU) | `dist/assets` |
| Browser MLP training | 200 steps, batch 16, hidden 24 | Training Lab |
| — wall clock | **105 ms** | run summary |
| — train loss | 0.737 (last batch) | run summary |
| — validation loss | 0.625 | run summary |
| — held-out accuracy | **76.6%** (chance 25%) | run summary |
| Weight patch in browser | 384 of 384 elements changed, norm −2.000% | Weight Studio |
| Bundle size (JS + CSS, gzip) | ~109 kB + 3.5 kB | vite output |
| Mobile layout | no horizontal overflow at 390 px (scrollWidth 375 of 375) | Playwright check |

## Evolution Strategies vs backprop (equal example budget)

Measured with `examples/es-vs-backprop/run.py` on a 340-parameter MLP, 6 candidates × 2 generations,
3 validation batches per candidate:

| Method | Held-out score | Examples consumed |
|---|---|---|
| Initial random parameters | 0.281 | 0 |
| Antithetic ES (best) | 0.385 | 2,304 |
| Gradient descent (example-matched) | **0.724** | 2,304 |

This is the expected result and it is published rather than hidden: on a differentiable supervised
task, backpropagation is markedly more sample-efficient than ES. The archive found 2–3 quality cells,
which is the reason to keep ES in the toolbox — diversity, not raw sample efficiency. Wall-clock time
is not equalised in either direction.

## GRPO on the toy verifiable-reward environment

`hcs-foundry rl grpo --steps 100 --lr 0.2 --kl-beta 0.01 --seed 3`, 6 context features, 4 actions,
8 samples per context, 16 contexts per step:

| Metric | Value |
|---|---|
| Held-out accuracy (256 unseen contexts, separate process) | **0.891** |
| Reference policy at the same seed | 0.375 |
| Chance | 0.250 |
| Training reward (last step) | 0.906 |
| Exact KL to reference (last step) | 1.24 nats |
| Wall clock | ~4 s |
| Policy artifact | JSON, ~1 kB, hashed |

**Interpretation, stated plainly:** this validates the GRPO implementation and the reward-pipeline
integrity. It says nothing about language-model post-training. A large KL penalty does what it should:
at `kl_beta=10` the policy stays close to the reference (exact KL 0.33) and gives up reward (0.336).

## Weight-space alignment + flow matching (toy scale)

`hcs-foundry generate --model flow-lab --method flow --corpus 101,202,303,404,505,606`, one hidden
layer (16 → 24 → 4), 400 flow-matching steps, Euler integration with 40 steps:

| Metric | Value |
|---|---|
| Alignment functional error (permutation validity) | 0.0 (exact) |
| Reconstruction: relative parameter distance | **7e-06** |
| Reconstruction accuracy vs trained checkpoint | identical on all seen tasks |
| Sampling diversity (two draws, same condition) | 1.41 relative distance (not collapsed) |
| Out-of-distribution generated accuracy | 0.250 (= chance) |
| Random init accuracy | 0.250 |
| Gradient-trained accuracy | 0.861 |
| Wall clock | under 10 s |

**Interpretation:** the generative machinery works — it reproduces checkpoints it has seen and does not
collapse to a single output. It does **not** generalise to unseen tasks, because four training
checkpoints are not a corpus. Reporting this as a success would be the exact overstatement this
project exists to prevent; the D2NWG/DeepWeightFlow results rest on corpora three orders of magnitude
larger.

## Resource estimate vs reality

For the 250k transformer at batch 8 / block 16, the estimator predicts:

* weights 1.0 MB, gradients 1.0 MB, optimizer 3.0 MB, activations ~0.1 MB, KV cache ~0.03 MB;
* total training footprint ~5.2 MB, i.e. four orders of magnitude below available RAM.

The estimator is a heuristic with stated assumptions (`docs/ARCHITECTURE.md` §2), not a traced graph.
It is used to refuse obviously impossible runs, not to promise a fit.

## What these numbers do not show

1. No comparison with any external model. Nothing here is a benchmark score.
2. Single-run timings on one machine; expect large variance on other hardware.
3. The browser demo model was trained for 40 steps. 39% next-token accuracy on a synthetic corpus
   after 40 steps demonstrates learning, not quality.
4. WebGPU numbers depend on the installed driver and browser build.

## Reproduce

```powershell
.\scripts\verify.ps1
E:\neuralfond\.venv\Scripts\python.exe scripts\build_web_demo.py
cd apps\web; npm run build; npm run preview
```
