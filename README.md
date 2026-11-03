# HCS Neural Foundry

Local-first workbench for designing neural networks, inspecting and editing their weights,
training them with gradients, evolving them, benchmarking them and exporting them —
honestly, reproducibly, and without a hidden cloud backend.

**Made with ❤️ by hcsmedia and timfromhcs**

> Status: `0.1.0` alpha. Every capability below carries an explicit maturity label
> (`stable`, `experimental`, `unsupported`). See [docs/STATUS.md](docs/STATUS.md) for the
> authoritative, generated capability matrix.

## What actually works today

| Area | State | Where |
|---|---|---|
| Hardware detection + capability registry (CPU always, accelerator probed not assumed) | stable | `hcs-foundry doctor --json` |
| Model factory: MLP and decoder-only Transformer, exact `numel` counts, tied-weight accounting | stable | `hcs-foundry model create` |
| Parameter budget solver (exact / closest / closest-under / at-least) | stable | `hcs-foundry model solve --params 1.2M` |
| Tensor inspection, histograms, bounded slices | stable | `hcs-foundry weights inspect` |
| Weight patches: scalar, slice, low-rank delta, scale, clamp, normalization, sparsity, interpolation | stable | `hcs-foundry weights patch` |
| Supervised training with real backprop + AdamW, checkpoint/resume/cancel, telemetry | stable | `hcs-foundry train` |
| Evaluation on deterministic smoke suites with full provenance | stable | `hcs-foundry eval` |
| Evolution Strategies + MAP-Elites archive on a tiny model | experimental | `hcs-foundry evolve` |
| GRPO on a toy verifiable-reward environment with sealed fixture | experimental | `hcs-foundry rl grpo` |
| Hypernetwork weight generator for a fixed tiny MLP | experimental | `hcs-foundry generate` |
| Permutation alignment + flow matching in weight space | experimental | `hcs-foundry generate --method flow` |
| ONNX export + inference parity check | experimental | `hcs-foundry export` |
| Local HTTP API with token auth, strict CORS, rate limits | stable | `hcs-foundry api serve` |
| MCP server (stdio) with read-only + approval-gated tools | experimental | `hcs-foundry mcp serve` |
| Static browser app with ONNX Runtime Web, IndexedDB history, local demo trainer | experimental | `apps/web` |
| Automated accessibility audit (axe-core, 4 screens × 2 locales) | stable | `apps/web/tests/a11y.test.tsx` |
| LLM-scale RL (DPO/PPO/RLHF), diffusion text, latent image diffusion | unsupported | roadmap |
| Desktop shell (Tauri), PyPI/npm publication | unsupported in this tree | roadmap |

## Quick start (Windows PowerShell)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[torch,export,dev]"
.\.venv\Scripts\hcs-foundry.exe doctor

# build a 250k-parameter transformer, train it, evaluate, export
.\.venv\Scripts\hcs-foundry.exe model create --family transformer-lm --params 250k --name demo
.\.venv\Scripts\hcs-foundry.exe train start --model demo --steps 200
.\.venv\Scripts\hcs-foundry.exe eval run --model demo --suite text-smoke
.\.venv\Scripts\hcs-foundry.exe export --model demo --format onnx
```

Local API + web client:

```powershell
.\.venv\Scripts\hcs-foundry.exe api serve          # http://127.0.0.1:8765
cd apps\web; npm install; npm run dev            # http://127.0.0.1:5173
```

## Scientific stance

Generating weights is **not** the same as training a capable model. This project lets you
compare agent-authored patches, weight generators, evolution and gradient training on equal
footing; it does not claim that any of them wins by default. Every claim in the UI is tied to a
run record with dataset hash, seed, hardware and evaluator version. Read
[docs/RESEARCH.md](docs/RESEARCH.md) before quoting any result.

## Documentation

- [plan.md](plan.md) — implementation plan and live phase status
- [DECISIONS.md](DECISIONS.md) — architecture decision records
- [docs/RESEARCH.md](docs/RESEARCH.md) — evidence-based research map and "claims we must not make"
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — modules, data flow, boundaries
- [docs/TESTING.md](docs/TESTING.md) — test layers, gates, how to reproduce
- [docs/SECURITY.md](docs/SECURITY.md) — threat model, sandbox, API hardening
- [docs/STATUS.md](docs/STATUS.md) — capability matrix with maturity labels
- [docs/tutorials/](docs/tutorials) — zero-to-hero curriculum

## License

Code: MIT. Datasets, checkpoints and third-party model artifacts keep their own licenses and are
never bundled without a recorded, compatible license.
