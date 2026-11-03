# Research map and claim boundaries

Snapshot date: **3 November 2026**. This document is deliberately opinionated about what may and may
not be claimed. Every implementation decision in this repository traces back to one of these entries.

---

## 1. The question this project asks

> Can networks be created or improved by agents that construct parameter tensors, by weight-space
> generative models, by evolution, by gradient training, by reinforcement learning, or by combinations?

The software exists to make that an **empirical** question. It does not assume an answer.

Within this repository, the answer looks like this today:

| Method | Status here | Evidence in this repo |
|---|---|---|
| Gradient training (backprop + AdamW) | stable, works | `hcs_foundry/training/trainer.py`, loss curves in run records |
| Manual weight editing | stable, works | `hcs_foundry/weights.py` + 12 patch operations |
| Structured agent patches | experimental | `agents.py` (proposal) + isolated verifier process |
| Evolution Strategies | experimental, narrow | `evolution.py`, example-matched baseline disclosed |
| Quality-diversity (MAP-Elites) | experimental, narrow | `evolution.MapElites` |
| Conditional hypernetwork weight generation | experimental, narrow | `generators.py` (one MLP family, held-out tasks) |
| Weight-space diffusion (D2NWG-like) | **unsupported** | interface only, no implementation |
| Flow matching in weight space (DeepWeightFlow-like) | **unsupported** | interface only, no implementation |
| RL post-training (DPO/PPO/GRPO) | **unsupported** | reward fixtures only, no trainer |
| Diffusion language models | **unsupported** | model-family slot reserved |
| Latent image diffusion | **unsupported** | not attempted |

---

## 2. Weight-space generation: what the literature actually says

### 2.1 HyperNetworks (established)

Ha, Dai & Le, *HyperNetworks*, ICLR 2016.
<https://research.google/pubs/hypernetworks/>

A network generates parameters of another network, conditioned on a latent code, task identity or
architecture metadata. This is the conceptual ancestor of "an agent writes weights" — and it already
contains the key correction: the useful signal is **compact conditioning**, not a language model
typing millions of decimals.

Implementation here: `generators.py` trains a conditional hypernetwork over a *task descriptor* and
evaluates on tasks it never saw. Result is reported next to random init and gradient training, with
the caveat that the gradient baseline is step-matched, not compute-matched.

### 2.2 D2NWG (promising research)

Soro et al., *Diffusion-based Neural Network Weights Generation*, ICLR 2025.
<https://proceedings.iclr.cc/paper_files/paper/2025/hash/f74d79573d71078848973009d0e99bdb-Abstract-Conference.html>

Diffusion over pretrained weight distributions, conditioned on dataset characteristics and
architecture. The paper reports competitive in-distribution results and faster convergence when used
as an initialisation. It does **not** claim that arbitrary text can become arbitrary LLM weights.

Why not implemented here: a credible reproduction needs a corpus of real checkpoints, canonicalisation
and an independent evaluation split. Shipping a 40-line approximation with the paper's name attached
would be the exact kind of false claim `AGENTS.md` §0.2 forbids.

### 2.3 DeepWeightFlow (priority research track)

Gupta et al., *DeepWeightFlow: Re-Basined Flow Matching for Generating Neural Network Weights*,
ICLR 2026. <https://proceedings.iclr.cc/paper_files/paper/2026/hash/e2f7407b62f152b7fe533fbc077fddb7-Abstract-Conference.html>

Flow matching in weight space, with Git Re-Basin style canonicalisation to handle permutation
symmetry, reporting complete networks that need no fine-tuning in its evaluated settings.

The important engineering lesson is not "flow matching wins". It is that **alignment is mandatory**:
interpolating or modelling unaligned checkpoints averages functionally unrelated solutions.

What this repository implements (toy scale, `flow.py` + `alignment.py`): activation-correlation +
Hungarian assignment to align one-hidden-layer MLPs exactly, then a conditional flow-matching model
`v(x_t, t, c)` trained on the aligned corpus and integrated with Euler steps. On four to six synthetic
tasks it reconstructs training checkpoints almost exactly (relative parameter distance ~7e-06) and
scores at chance on held-out tasks. That second number is the honest one: a handful of checkpoints is
not a corpus, and the papers' results rest on thousands.

What is **not** implemented: the papers' diffusion/flow architectures at their scale, dataset-conditioned
encoders, and cross-architecture alignment. Calling the toy pipeline a reproduction would be false.

### 2.4 Weight space learning survey (context)

Han et al., *A Survey of Weight Space Learning: Understanding, Representation, and Generation*,
arXiv:2603.10090. <https://arxiv.org/abs/2603.10090>

Three separable problems — understanding, representation, generation. This repository mirrors that
split: `weights.py` (understanding/editing), encoding utilities (representation), `generators.py`
(generation). Conflating them is how projects end up claiming "checkpoint generation" while shipping
checkpoint *search*.

---

## 3. Optimisation: three different tools


### 3.1 Backpropagation

The default for differentiable objectives. Implemented with AdamW, gradient clipping, cosine decay,
warmup, dry-run gradient checks and checkpoint/resume. Nothing here is novel; the novelty would be a
claim that it isn't needed.

### 3.2 Evolution Strategies

OpenAI, *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*.
<https://openai.com/index/evolution-strategies/>

That work explicitly notes ES can be dramatically slower than backprop on supervised tasks (their
MNIST preliminary result: up to ~1000× slowdown). This repo implements antithetic ES because it is a
useful *black-box* tool, and states in the result payload that the comparison is example-matched, not
compute-matched.

### 3.3 Quality-Diversity

Qin et al., *A survey on Quality-Diversity optimization*, Swarm and Evolutionary Computation, 2026.
<https://www.sciencedirect.com/science/article/pii/S2210650225003979>

MAP-Elites keeps one elite per behaviour cell. Here the descriptors are score, latency and parameter
count — the things a user choosing a small local model actually cares about.

### 3.4 Reinforcement learning (toy GRPO implemented; LLM-scale RL unsupported)

Reference points for the roadmap:
* Wu, *A Comprehensive Survey on Learning from Rewards for LLMs*, ACL Findings EMNLP 2025.
* TRL GRPO trainer docs: <https://huggingface.co/docs/trl/grpo_trainer>
* Reward hacking in agentic systems, 2026: <https://link.springer.com/article/10.1007/s44163-026-01980-z>

What is implemented: GRPO — group-relative advantages, a frozen reference policy and a KL penalty —
on a **toy contextual bandit with an exact verifiable reward**. In 100 steps it reaches 0.891 held-out
accuracy against a 0.375 reference and 0.25 chance, and the saved policy is re-scored by a separate
process against a sealed fixture. A test also tampers with the fixture and asserts the verifier refuses.

What is deliberately not implemented: DPO, PPO, RLHF, or GRPO over token-level completions of the
transformer. Those need rollouts, a reward model or a verifier over text, and a compute budget this
build does not have. Reporting a 250k-parameter character model as "RLHF" would be the exact
overstatement this project exists to avoid.

The 2026 reward-hacking survey is the reason the evaluator runs in a separate process with a read-only
fixture directory: a stronger optimiser will exploit a weak reward, including by editing the test.

---

## 4. Model families: scope statements

### 4.1 Autoregressive transformer (implemented, educational scale)

RoPE, grouped-query attention, SwiGLU, RMSNorm, tied embeddings, character-level training on a
synthetic corpus. Parameter count is enumerated, not estimated. This is a teaching and
infrastructure artefact; it is not a capable language model and no benchmark in this repo suggests
otherwise.

### 4.2 Diffusion language models (unsupported)

LLaDA (NeurIPS 2025) and FS-DFM (ICLR 2026) require a different training objective (masked-token
diffusion) and a different sampler. Applying next-token cross-entropy and calling it diffusion would
be wrong, so the family is declared unsupported rather than approximated.

### 4.3 Image generation (unsupported)

Latent diffusion / rectified flow need a VAE, a text encoder, a scheduler and a denoiser versioned
together. A "toy DiT" that produces noise does not demonstrate the pipeline. Not implemented.

### 4.4 "JEv" models

No unambiguous public architecture definition was available at implementation time. The registry keeps
an extension point; nothing was invented.

---

## 5. Claims we must not make

1. "Generated weights are better than RL." No experiment here supports that; no fair comparison exists.
2. "State of the art" for anything. There is no benchmark comparison against external models.
3. "Unlimited model size." Usable size is bounded by memory, compute, data and runtime support.
4. "WebGPU supported." Only "WebGPU worked for this exact graph, in this browser, after a real
   inference". Fallback is always reported.
5. "Lossless quantisation." No quantisation is implemented.
6. "The browser version is equivalent to the desktop engine." The parity matrix says otherwise.
7. "Results are reproducible" without stating platform. Floating-point results are reproducible on
   the same stack; seeds and versions are recorded so the claim is checkable, not assumed.
8. Any benchmark score without dataset hash, evaluator version, seed and hardware.

---

## 6. Evaluation references (for the roadmap)

* LiveBench 2.0 — <https://livebench.ai/v1/>
* BFCL v4 — <https://gorilla.cs.berkeley.edu/leaderboard>
* Terminal-Bench 2.0 — <https://www.tbench.ai/?version=2.0>
* GAIA — <https://hal.cs.princeton.edu/gaia>
* SWE-bench Pro V2 — <https://labs.scale.com/leaderboard/swe_bench_pro_public_v2>
* T2I-CompBench — <https://arxiv.org/abs/2307.06350>
* ImageReward — <https://arxiv.org/abs/2304.05977>

None of these are executed by this build. The bundled suites are deliberately named `*-smoke` so they
cannot be mistaken for them.
