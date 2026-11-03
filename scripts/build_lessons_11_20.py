"""Generate curriculum lessons 11-20 (same contract as lessons 1-10)."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "tutorials"

LESSONS = [
    (
        "11-evolution-strategies",
        "Evolution Strategies",
        "Run an antithetic ES step and read the fair-budget caveat.",
        """ES treats the model as a black box: sample perturbations, evaluate them, move toward the better
ones. Antithetic sampling evaluates `θ+σε` and `θ-σε` together, which halves the variance of the
estimate for the same number of evaluations.

It is the right tool when the objective is not differentiable. It is the wrong tool when it is:
on supervised tasks, ES needs orders of magnitude more compute than backprop.""",
        """Rank-based estimate: `g = (1/n) Σ r_i ε_i`, then `θ ← θ + lr·g`, with `r` normalised ranks.
The engine reports how many examples each method consumed so a comparison cannot be quietly unequal.""",
        """hcs-foundry model create --preset weight-lab-mlp --name "ES Lab"
hcs-foundry evolve --model es-lab --population 16 --generations 5 --sigma 0.02 --lr 0.05""",
        "Run with `--population 4` and `--population 32`. How does archive coverage change?",
        "Best score ≥ initial score, a MAP-Elites archive with several cells, and a gradient baseline labelled as example-matched.",
        "Declaring ES better because it improved the score — without comparing against backprop on the same compute budget.",
    ),
    (
        "12-reinforcement-learning",
        "Reinforcement learning and reward signals",
        "Separate DPO, PPO, GRPO and verifiable rewards.",
        """RL post-training optimises behaviour after a model has basic competence.

* **DPO** — preference optimisation on pairs; no online rollouts, no reward model at train time.
* **PPO** — policy gradient with a clipped update and a KL constraint to a reference policy.
* **GRPO** — sample several completions per prompt and use relative reward inside the group; no value
  network.
* **RLVR** — rewards from automatically checkable outcomes (maths answers, tests passing).

The failure mode that matters is reward hacking: a strong optimiser will find the cheapest way to
increase the score, including editing the test.""",
        """GRPO advantage for completion `i`: `A_i = (r_i - mean(r)) / std(r)`.
KL penalty keeps `π` close to `π_ref`; without it, the model collapses onto the reward's quirks.""",
        """# GRPO on the toy verifiable-reward environment (bandit, not an LLM)
hcs-foundry rl grpo --steps 100 --lr 0.2 --kl-beta 0.01 --seed 3
hcs-foundry history --kind reinforcement""",
        "Write the reward function you would use for a verifiable task, then write the exploit a policy would find.",
        "Proposals are immutable and the verifier runs in a separate process — the structure that makes reward tampering hard.",
        "Reporting reward improvement as capability improvement. Reward is a proxy; held-out task success is the measurement.",
    ),
    (
        "13-hypernetworks",
        "Hypernetworks and weight space",
        "Understand what a weight generator can and cannot do.",
        """A hypernetwork maps a compact conditioning vector to a large parameter vector. The compression is
the point: you cannot condition usefully on a million free floats, but you can condition on a task
descriptor, an architecture embedding or a dataset fingerprint.

What it buys you: a distribution over initialisations, cheap ensembles, and an interface for
agent-driven weight proposals. What it does not buy you: competence the training data never contained.""",
        """Generator: `g: R^z -> R^P` trained to minimise `‖g(z_i) - θ_i‖²` on a checkpoint corpus.
Leakage guard: the split is by task, and overlapping task seeds raise rather than train.""",
        """hcs-foundry model create --preset weight-lab-mlp --name "Gen Target"
hcs-foundry generate --model gen-target --corpus 11,22,33,44,55,66,77,88 --epochs 200""",
        "Move one task seed from the training list to the test side and compare generated vs random-init accuracy.",
        "The report shows generated, random-init and gradient-trained accuracy on tasks the generator never saw, plus an explicit claim boundary.",
        "Evaluating generated weights on the same checkpoints used for training and reporting the training error as quality.",
    ),
    (
        "14-weight-space-generation",
        "Diffusion and flow matching in weight space",
        "Read the D2NWG and DeepWeightFlow claims critically.",
        """Both methods learn a generative process over parameter tensors. Diffusion learns denoising across
noise levels; flow matching learns a vector field transporting noise to weights along a chosen path.

The hard part is not the generative model — it is **geometry**. Two networks with permuted hidden
units compute the same function but have unrelated parameter vectors. Averaging them is meaningless,
so serious work canonicalises first (Git Re-Basin style) and stores the permutation it applied.

This build declares both methods unsupported. The interface exists; a mocked demo does not.""",
        """Flow matching objective: regress `v_θ(x_t, t)` onto `x_1 - x_0` along an interpolation path.
Diffusion objective: predict the noise added at level `t`, then sample by iterating the reverse process.""",
        """# Not implemented on purpose. The reproducible scopes are documented instead:
Get-Content docs/RESEARCH.md
hcs-foundry generate --help""",
        "Write down what evidence would convince you that generated weights beat a trained baseline, then check whether this repo provides it.",
        "It does not — and `docs/RESEARCH.md` says so. That is the intended outcome of this lesson.",
        "Shipping an approximation under a paper's name. It looks like progress and is actually a fabricated capability.",
    ),
    (
        "15-language-model-families",
        "Autoregressive vs diffusion language models",
        "Compare decoding, cost and evaluation honestly.",
        """An autoregressive model factorises `p(x) = Π p(x_t | x_<t)` and generates left to right, one token
per step. A diffusion language model corrupts tokens with a mask and learns to denoise; generation
takes a fixed number of refinement steps and can revise earlier positions.

Consequences: AR is sequential and cannot revise; diffusion is parallel *within* a step but needs many
steps. Perplexity is not directly comparable between the two families, because the likelihood
estimators differ.""",
        """AR loss: `-Σ log p(x_t | x_<t)`. Masked diffusion loss: `-Σ_{masked} log p(x_t | x_visible)`.
Compare at equal budget: tokens generated, wall-clock, memory, steps.""",
        """hcs-foundry train --model lesson-3 --steps 100
hcs-foundry eval --model lesson-3 --suite text-smoke --batches 8""",
        "Sketch the sampler for a masked diffusion model and count how many forward passes 64 tokens need at 8 steps.",
        "8 forward passes instead of 64 — the trade-off, before any quality claim.",
        "Comparing a 1024-step diffusion sample to a 1-step AR decode and calling it a fair speed comparison.",
    ),
    (
        "16-image-diffusion",
        "Latent image diffusion",
        "Identify the components of a text-to-image pipeline and why they are versioned together.",
        """A text-to-image system is not one model. It is: tokenizer → text encoder → (optional) latent VAE
encoder → denoiser (UNet or DiT) → scheduler/solver → VAE decoder. Change the VAE and every stored
latent becomes incompatible.

Training cost lives mostly in the denoiser, but quality failures are frequently in the text encoder or
the scheduler. Reproducing a result requires the exact combination.""",
        """Forward noising: `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`.
Rectified flow: `x_t = (1-t)x_0 + t·ε`, learn `v = ε - x_0`.""",
        """# Not implemented: a toy DiT that emits noise is not a pipeline demo.
Get-Content docs/RESEARCH.md -TotalCount 60""",
        "List every artifact you would have to store to reproduce one generated image.",
        "Tokenizer, text encoder, VAE weights, denoiser weights, scheduler config, prompt, seed, steps, resolution, and the library versions.",
        "Storing only the denoiser checkpoint and being surprised that the same prompt and seed produce a different image.",
    ),
    (
        "17-agents-and-mcp",
        "Agent tool use and MCP",
        "Design a narrow tool surface with approval gates.",
        """An agent is a model plus a harness: tool schemas, permissions, retries, memory and a stopping rule.
The harness is where safety lives, not the prompt.

The Model Context Protocol gives a typed way to expose tools and resources. The design rule used here:
read-only tools are always available; anything that spends compute, writes files or mutates weights
requires an explicit approval list, and no tool is a general shell.""",
        """Tool contract: name, typed arguments, typed result, explicit failure mode.
Approval gate: `GATED_TOOLS` ∩ approved set, otherwise refuse before any side effect.""",
        """hcs-foundry mcp --approve foundry_start_evaluation
# without --approve, gated tools return a security_violation payload""",
        "Add a new read-only MCP tool and check that it appears in `server.list_tools()` without an approval entry.",
        "The audit log (`foundry_list_tool_calls`) records every invocation, approved or not.",
        "Adding a `run_shell` tool 'for flexibility'. It converts every prompt-injection risk into host compromise.",
    ),
    (
        "18-benchmarking",
        "Benchmarking and reproducibility",
        "Write a protocol someone else can rerun.",
        """A benchmark result is a tuple, not a number: task version, dataset hash and split, prompt template,
seed policy, model hash, hardware, runtime, precision, batch size, decoding parameters and evaluator
version.

Without those, two numbers are not comparable — even if they share a name.""",
        """Report distributions, not best seeds: median and interquartile range over `n` runs.
Report cost beside quality: wall-clock, memory, tokens or examples.""",
        """hcs-foundry eval --model lesson-3 --suite text-smoke --seed 1234 --batches 8 --json
hcs-foundry history --kind evaluation --json""",
        "Take one evaluation report and check whether every element of the tuple above is present.",
        "It is: metrics carry definitions, and provenance carries hashes, seeds, evaluator version and device.",
        "Averaging unrelated metrics into one 'score'. It hides every trade-off that matters.",
    ),
    (
        "19-onnx-and-browser",
        "Quantisation, ONNX and the browser",
        "Verify parity instead of assuming it.",
        """Exporting to ONNX is a graph translation, and translations lose things: dynamic shapes, custom
operators, dtype quirks. The only acceptable proof is running the exported graph on real inputs and
comparing to the native outputs within a stated tolerance.

In the browser, availability is not the same as support: `navigator.gpu` existing does not mean your
graph's operators run on WebGPU. This app compiles the model with the provider and runs a smoke
inference before it reports success.""",
        """Relative error: `max|native - onnx| / max|native|`. Tolerance here: `1e-4`.
Quantisation would add its own error budget — and is not implemented in this build.""",
        """hcs-foundry export --model lesson-3 --format onnx
# then open the web app, Dashboard -> Starten, and read the reported provider""",
        "Export, then deliberately change a weight in the checkpoint and re-run the browser demo without re-exporting.",
        "The browser keeps running the old graph — the file, not the checkpoint, is what the browser loaded.",
        "Claiming 'WebGPU supported' from a feature flag. The honest claim names the graph, the browser and the measurement.",
    ),
    (
        "20-releasing",
        "Publish a reproducible model project",
        "Ship artifacts with provenance and honest labels.",
        """A release is a set of artifacts plus the evidence that they work from a clean environment:

* code (with lockfiles) and tests;
* weights with content hashes and license;
* a model card stating what it was trained on and what it cannot do;
* benchmark reports with full provenance, including failures;
* install instructions verified in a fresh virtual environment.

Nothing in this repository is published automatically, and that is intentional.""",
        """Artifact identity: SHA-256 of the checkpoint bytes. Reproducibility: seed + config + dataset hash.
Support matrix: which backends were actually tested, and on what hardware.""",
        """\\scripts\\verify.ps1
hcs-foundry export --model lesson-3 --format onnx
hcs-foundry history --model lesson-3""",
        "Write the one-paragraph model card for the model you trained, including what it must not be used for.",
        "The card cannot claim language ability: 250k parameters on a synthetic corpus demonstrates a pipeline.",
        "Publishing a checkpoint without its config, dataset hash and evaluator version. It becomes impossible to reproduce and impossible to audit.",
    ),
]

for slug, title, goal, intuition, math, runnable, exercise, observe, mistake in LESSONS:
    body = f"""# {title}

**Goal.** {goal}

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

{intuition}

## The mathematics

```
{math}
```

## Run it

```powershell
{runnable}
```

## Exercise

{exercise}

## What you should observe

{observe}

## Common mistake

{mistake}

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
"""
    (OUT / f"{slug}.md").write_text(body, encoding="utf-8")
    print(f"wrote {slug}.md")
