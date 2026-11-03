# HCS Neural Foundry

Local-first workbench for designing neural networks, inspecting and editing their weights, training
them with gradients, evolving them, benchmarking them and exporting them — honestly, reproducibly,
and without a hidden cloud backend.

**Made with ❤️ by hcsmedia and timfromhcs**

> **Status: `0.2.0`** — the pipeline is stable and published; several research features are
> deliberately narrow. Every capability below carries an explicit maturity label
> (`stable` / `experimental` / `unsupported`). The authoritative matrix is
> [docs/STATUS.md](docs/STATUS.md); the machine-readable one is [docs/acceptance.json](docs/acceptance.json).

**Browser app (static, runs entirely in your tab):** builds from `apps/web` with
`npm ci && npm run build`; deploy configuration is in `apps/web/vercel.json`.
**Source:** <https://github.com/timfromhcs/hcs-neural-foundry>

## What this actually is

A complete feedback loop, built and tested end to end:

```
design an architecture → create or generate weights → inspect and edit tensors
   → train with gradients / rewards / evolution → benchmark independently
   → compare and visualise → keep the history → export or automate via API and MCP
```

You can ask for an exact parameter budget and get the exact enumerated count back. You can open a
tensor, patch it twelve different ways, see the norm change, roll it back, and still explain a week
later which patch produced which checkpoint. You can train a 250k-parameter transformer on this
machine in a couple of seconds, export it to ONNX, and prove the browser output matches the desktop
output to within `7.15e-07`.

## What this is not

| Claim someone might expect | Reality |
|---|---|
| "An AI types weights, so the model is trained" | Generating weights is not training. The generator reproduces checkpoints it has seen and scores **at chance** out of distribution — published as such in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). |
| "Evolution beats backprop" | On a differentiable task it does not. Measured: gradient 0.724 vs ES 0.385 at an equal example budget. |
| "RL fine-tunes a language model" | A GRPO implementation is demonstrated on a **toy contextual bandit** with a sealed verifier. No token-level LLM RL. |
| "WebGPU is the fast path" | Measured the opposite for this model size: WASM warm 2.7 ms vs WebGPU warm ~16.8 ms. |
| "State of the art" | No external benchmark was run. Nothing here is a benchmark score. |

## Capability matrix

| Area | State | Where |
|---|---|---|
| Hardware detection, `declared` vs `usable` backends | stable | `hcs-foundry doctor --json` |
| Model factory: MLP + decoder Transformer, exact `numel`, tied-weight accounting | stable | `hcs-foundry model create` |
| Parameter budget solver (`exact`, `closest`, `closest_under`, `at_least`) | stable | `hcs-foundry model solve --params 1.2M` |
| Tensor inspection, histograms, bounded slices | stable | `hcs-foundry weights inspect` |
| Weight patches: 12 closed operations incl. low-rank delta, sparsity, interpolation | stable | `hcs-foundry weights patch` |
| Supervised training: AdamW/SGD, warmup+cosine, resume, cancel, telemetry | stable | `hcs-foundry train` |
| Evaluation suites with metric definitions and `invalid` (never `0.0`) on evaluator failure | stable | `hcs-foundry eval` |
| Evolution Strategies + MAP-Elites archive | experimental | `hcs-foundry evolve` |
| GRPO on a toy verifiable-reward environment with a sealed fixture | experimental | `hcs-foundry rl grpo` |
| Hypernetwork weight generator for a fixed tiny MLP | experimental | `hcs-foundry generate` |
| Permutation alignment + flow matching in weight space | experimental | `hcs-foundry generate --method flow` |
| ONNX export with a numerical parity gate | experimental | `hcs-foundry export` |
| Local HTTP API: bearer auth, CORS allowlist, rate limits, SSE, OpenAPI | stable | `hcs-foundry api serve` |
| MCP server (stdio), **SDK v2 / 2026-07-28 revision**, read-only + approval-gated tools | stable | `hcs-foundry mcp serve` |
| Static browser app: ONNX Runtime Web (native WebGPU EP + WASM), IndexedDB history, local trainer | stable | `apps/web` |
| Automated accessibility audit (axe-core, 4 screens × 2 locales) | stable | `apps/web/tests/a11y.test.tsx` |
| Weight-space diffusion (D2NWG-class), cross-architecture alignment | unsupported | roadmap |
| DPO / PPO / RLHF / token-level GRPO | unsupported | roadmap |
| Diffusion language models, latent image diffusion | unsupported | roadmap |
| Desktop shell (Tauri), PyPI/npm publication | unsupported in this tree | roadmap |

## Quick start (Windows PowerShell)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[torch,export,mcp,dev]"
.\.venv\Scripts\hcs-foundry.exe doctor
```

Four commands that prove the whole pipeline:

```powershell
.\.venv\Scripts\hcs-foundry.exe model create --family transformer-lm --params 250k --name demo
.\.venv\Scripts\hcs-foundry.exe train start --model demo --steps 200
.\.venv\Scripts\hcs-foundry.exe eval run --model demo --suite text-smoke
.\.venv\Scripts\hcs-foundry.exe export --model demo --format onnx
```

Local API + browser client:

```powershell
.\.venv\Scripts\hcs-foundry.exe api serve          # http://127.0.0.1:8765
cd apps\web; npm ci; npm run dev                 # http://localhost:5173
```

MCP host configuration (SDK v2, stdio):

```jsonc
{
  "mcpServers": {
    "hcs-neural-foundry": {
      "command": "C:\\path\\to\\.venv\\Scripts\\hcs-foundry.exe",
      "args": ["mcp", "serve", "--approve", "foundry_start_training"]
    }
  }
}
```

Gated tools refuse to run unless named after `--approve` (repeatable) or listed in
`HCS_FOUNDRY_MCP_APPROVE`. Every call is written to an audit log.

## Verify it yourself

```powershell
.\scripts\verify.ps1      # lint, tests, web build, end-to-end smoke workflow
.\scripts\release.ps1     # the same plus the strict release audit — must be green before a tag
```

The release audit checks version consistency across four files, the changelog entry, the brand
credit, the acceptance matrix, tracked secrets, replacement characters, the demo-model hash, that the
MCP pin agrees with the MCP import path, and that the built site references no external origin.

## Repository map

```
python/hcs_foundry/     engine: hardware, models, weights, training, evaluation,
                        evolution, generators, alignment, flow, rl, api, cli, mcp
python/tests/           121 pytest tests
apps/web/               static React client, 48 vitest tests incl. 8 ONNX provider tests
mcp/                    MCP server entry points and examples
docs/                   research map, architecture, testing, security, status, performance,
                        20-lesson curriculum, the 0.2.0 milestone contract
scripts/                verify, release, verify_release, build_web_demo, encoding scan, SBOM
examples/               tiny-mlp, tiny-transformer, weight-patches, es-vs-backprop, browser-onnx
```

## Scientific stance

Generating weights is **not** the same as training a capable model. This project lets you compare
agent-authored patches, weight generators, evolution and gradient training on equal footing; it does
not claim that any of them wins by default. Every result in the UI is tied to a run record with
dataset hash, seed, hardware and evaluator version. Read [docs/RESEARCH.md](docs/RESEARCH.md) before
quoting any number.

Negative results are kept and published: out-of-distribution generated weights score at chance, ES
loses to backprop at an equal example budget, and WASM beats WebGPU for a 250k model. A build that
only reports its wins is not a benchmark.

## Documentation

- [docs/ROADMAP-0.2.0.md](docs/ROADMAP-0.2.0.md) — the milestone contract, including the research that forced the 0.2.0 fixes
- [plan.md](plan.md) — implementation plan and live phase status
- [DECISIONS.md](DECISIONS.md) — architecture decision records
- [docs/RESEARCH.md](docs/RESEARCH.md) — evidence map and "claims we must not make"
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — modules, data flow, boundaries
- [docs/TESTING.md](docs/TESTING.md) — test layers, gates, how to reproduce
- [docs/SECURITY.md](docs/SECURITY.md) — threat model, sandbox, API hardening
- [docs/STATUS.md](docs/STATUS.md) — capability matrix with maturity labels
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — every measured number, including the ones that disappoint
- [docs/tutorials/](docs/tutorials) — zero-to-hero curriculum

## License

Code: MIT. Datasets, checkpoints and third-party model artifacts keep their own licenses and are
never bundled without a recorded, compatible license. The only shipped model was trained in this
repository.
