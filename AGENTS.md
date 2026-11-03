# HCS Neural Foundry — Autonomous Build Specification

> **Project codename:** HCS Neural Foundry  
> **Repository slug:** `hcs-neural-foundry`  
> **Product:** local-first neural network design, weight-space research, training, evaluation and agent studio  
> **Brand credit:** **Made with ❤️ by hcsmedia and timfromhcs**  
> **Research snapshot:** 3 November 2026  
> **Primary deliverable:** production-quality repository with local desktop application, local API, CLI, MCP server, npm and PyPI packages, and a statically deployed Vercel web application.  
> **This file is the autonomous implementation contract.** Read it before changing the repository.

---

## 0. Instructions to the autonomous coding agent

You are the principal engineer, ML research engineer, product designer, QA engineer, release engineer and technical writer for this repository. Build the project end-to-end, locally first, then publish only after all local gates pass.

Do not treat this document as a wish list. Convert it into working software, tests, documentation, reproducible experiments and release artifacts. Do not claim a capability merely because a button, settings field, mocked chart, placeholder API or static demo exists. A capability is complete only when it has an implementation, input validation, error handling, tests, documentation and a visible user workflow.

### 0.1 Required execution order

1. Inspect the actual repository, current branch, git status, package manifests, existing code, CI, licenses, release history and available secrets. Never overwrite or delete user work without first understanding it.
2. Create `plan.md`, `DECISIONS.md`, `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`, `docs/TESTING.md` and a machine-readable feature/acceptance checklist. Keep this `AGENTS.md` as the source of truth and update all plans when implementation reality changes.
3. Build a minimal vertical slice locally: application shell → hardware detection → model/parameter schema → one tiny model → manual weight editing → deterministic evaluation → history persistence.
4. Expand to the complete local engine, model families, trainers, agent control plane and MCP/API.
5. Build the browser application as a static client that uses ONNX Runtime Web / Transformers.js and browser-local persistence. Do not silently substitute a hosted inference backend.
6. Add CLI, npm, PyPI and desktop packaging. Verify actual install and execution from clean environments.
7. Run full test, security, performance, accessibility, localization and release gates.
8. Only after all local gates pass, create/push the GitHub repository or branch, deploy the static Vercel application, publish packages and create releases. Use existing credentials only. Do not print, copy into logs, commit or expose tokens.

### 0.2 Agent operating rules

- Work in small, reviewable steps. Prefer vertical slices and working software over a huge untested rewrite.
- Before each task, state the expected outcome, files likely to change, risks and tests.
- After each task, inspect the diff, run the narrow test, run relevant integration tests and update the plan.
- Never mark a task complete merely because code was generated. A successful exit code is not sufficient if logs contain hidden errors or assertions were skipped.
- Never silently weaken tests, lower coverage thresholds, delete failing tests, replace real behavior with mocks or alter expected values merely to obtain green CI.
- Never modify benchmark ground truth, hide failed runs or let the agent grade its own work without an independent verifier.
- Keep a durable experiment journal: configuration, seed, software versions, hardware, dataset hashes, checkpoint hashes, metrics, duration, memory and failure reason.
- Use deterministic seeds where feasible. Record when operations are nondeterministic.
- Do not use `eval`, arbitrary shell execution from model output, pickle loading of untrusted checkpoints, or unrestricted filesystem/network access for model-generated code.
- Use explicit human approval for destructive actions, external publication, credential use, data upload, package release, changing access permissions and running unknown code. An unattended mode may perform only actions within a preconfigured sandbox and resource budget.
- If a dependency, API or research claim is unclear, inspect its current official documentation and package metadata before relying on it. Pin versions and record why.
- If a task is blocked, implement a safe, honest fallback and document the limitation. Do not leave fake controls in the UI.
- Do not invent GPU support. Capability-detect the exact backend and operation at runtime, benchmark it, and fall back transparently.
- Do not claim that generated weights are “better than RL” or “SOTA” without a fair, reproducible comparison on independent evaluation data.

### 0.3 Definition of done

A feature is done only when:

- it works from a fresh install, not only from the developer's current machine;
- its public API and schemas are typed and validated;
- its failure modes are surfaced to the user in understandable language;
- logs and history identify the exact run and relevant hashes;
- unit, integration and end-to-end tests cover expected behavior and edge cases;
- there is user-facing documentation and a practical tutorial;
- the UI is responsive, keyboard accessible and localized in English, German and French;
- no secrets, private datasets or unlicensed model weights enter the repository;
- the feature is accurately labeled as stable, experimental or unsupported.

---

# 1. Product vision and non-negotiable scientific framing

## 1.1 Mission

Create a beautiful, approachable but technically serious workbench where a user can create neural network architectures with a chosen parameter budget, inspect and edit their numerical weights, ask an AI agent to propose structured weight changes, generate parameters with learned weight-space models, train models with conventional optimizers, perform evolutionary search, fine-tune with reward-based methods, compare all approaches and export reproducible models.

The application must make the complete process understandable from beginner to expert. It should expose the math and tensors without forcing beginners to understand every tensor before getting a working model.

The same project must be usable in several ways:

1. **Desktop/local application:** full local engine, training, model management, history, API and agent tools.
2. **CLI:** scriptable workflows for training, generation, evaluation and automation.
3. **Local HTTP API:** typed, documented and versioned endpoints.
4. **MCP server:** model agents can inspect experiments, create proposals, run permitted jobs and retrieve results.
5. **npm package:** TypeScript client and browser-compatible utilities; optionally a Node CLI wrapper.
6. **PyPI package:** Python engine, SDK and CLI.
7. **Vercel web application:** static site with browser-local inference and supported small-scale operations via ONNX Runtime Web, WebGPU or WebAssembly. It must not require a remote inference backend for normal user work.

## 1.2 The central research question

Can networks be created or improved by agents that construct numerical parameter tensors, by weight-space generative models, by evolution, by gradient training, by reinforcement learning, or by combinations of these methods?

The software must allow empirical comparison rather than declare a winner in advance.

### Important distinction: generating weights is not the same as training a capable model

A neural network consists of an architecture and parameter tensors. An agent can write values into tensors, generate a parameter vector from a learned distribution, search parameter vectors with evolution, or optimize parameters using gradients. But a raw list of numbers has no demonstrated competence until it is evaluated on tasks it has not seen.

The agent can be valuable as:

- architecture designer;
- initializer designer;
- structured parameter editor;
- hypernetwork or weight-generator controller;
- experiment planner;
- evaluator and error analyst;
- trainer orchestrator;
- evolutionary candidate proposer;
- reward/environment designer.

It is not scientifically valid to infer that manually authored weights outperform gradient training or RL merely because the model produces a plausible output once.

## 1.3 No false “unlimited model size” promise

The user may choose an exact or approximate parameter budget, but usable model size is bounded by memory, compute, data, architecture, precision, sequence length, optimizer state, activation memory, inference cache and target runtime support. The UI must display these constraints before a job starts.

A user-selected parameter count is a design budget, not an intelligence slider. Doubling parameters does not automatically double quality.

## 1.4 Research status labels

Every method and feature must have one of these labels:

- **Established:** repeatedly demonstrated and has mature implementations.
- **Promising research:** peer-reviewed or credible preprint evidence, but narrow or not yet broadly validated.
- **Experimental:** implementation or evidence is preliminary.
- **Unsupported:** no validated implementation in this repository.

Do not market an experimental weight-space technique as a general replacement for pretraining or fine-tuning.

---

# 2. Research synthesis as of 3 November 2026

This is a focused technical synthesis, not a claim that every paper on the internet has been reviewed. Primary papers and official documentation take precedence over social posts and leaderboard summaries. Before implementing a paper, read the full paper, supplemental material, code and license; reproduce a small baseline before extending it.

## 2.1 Direct neural weight generation and weight-space learning

### A. HyperNetworks — established foundation

A hypernetwork is a neural network that generates some or all parameters of another network. It can condition generated weights on task identity, architecture metadata, data embeddings or a latent code. This is the conceptual ancestor of the project's “agent writes weights” workflow.

- Foundational paper: Ha, Dai and Le, *HyperNetworks*, ICLR 2016.
- Official reference: https://research.google/pubs/hypernetworks/
- Practical interpretation: do not ask a general-purpose chat model to emit a million unrelated decimal values. Give it a structured architecture and a compact latent/control representation; a trained generator maps that representation to tensors.
- Project status: first-class research adapter. Implement a tiny hypernetwork baseline early; larger weight generators remain experimental.

### B. D2NWG — diffusion-based neural network weight generation

Soro et al., *Diffusion-based Neural Network Weights Generation*, ICLR 2025, models distributions of pretrained network weights and generates task-conditioned weights from dataset characteristics, task descriptions and architecture specifications. The paper reports competitive in-distribution results, use as an initialization for new tasks, faster convergence and a reported few-shot improvement in its tested settings. It also reports experiments involving a Llama-3.2-1B-Instruct model, but these results must not be generalized to arbitrary frontier-scale LLMs.

- Official ICLR paper page: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f74d79573d71078848973009d0e99bdb-Abstract-Conference.html
- Official PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/f74d79573d71078848973009d0e99bdb-Paper-Conference.pdf
- Project interpretation: a learned distribution over checkpoints can produce task-specific initial weights. This is not a claim that a text agent can write arbitrary large model weights from scratch with no training data.
- Project status: reproduce on small MLP/CNN/ViT-like architectures where the paper's pipeline can be tested. Treat LLM-scale extension as a research experiment.

### C. DeepWeightFlow — especially relevant 2026 development

Gupta et al., *DeepWeightFlow: Re-Basined Flow Matching for Generating Neural Network Weights*, ICLR 2026. The method uses flow matching directly in weight space. It addresses the difficult geometry of weight space and parameter permutation symmetries using Git Re-Basin and TransFusion-style canonicalization. The paper reports generation of complete networks, including networks that do not require fine-tuning to perform well in its evaluated settings, and efficient generation of ensembles.

- Official ICLR 2026 page: https://proceedings.iclr.cc/paper_files/paper/2026/hash/e2f7407b62f152b7fe533fbc077fddb7-Abstract-Conference.html
- arXiv identifier: https://arxiv.org/abs/2601.05052
- Hugging Face paper index: https://huggingface.co/papers/2601.05052
- Project interpretation: flow matching may be a more efficient weight-space generator than multi-step diffusion in the investigated settings. The paper's benchmark scope and architecture families must be checked before using it for LLMs, image diffusion transformers or arbitrary parameter counts.
- Project status: priority research track after the parameter schema, checkpoint corpus and reproducible baseline exist.

### D. Weight Space Learning taxonomy

The March 2026 survey *A Survey of Weight Space Learning: Understanding, Representation, and Generation* (arXiv:2603.10090) groups the field into:

1. **Weight Space Understanding:** geometry, symmetries and structure of parameters.
2. **Weight Space Representation:** embeddings and representations of checkpoints.
3. **Weight Space Generation:** hypernetworks and generative models that synthesize weights.

It connects these ideas to model retrieval, transfer, continual/federated learning, architecture search and data-free reconstruction.

- Survey: https://arxiv.org/abs/2603.10090
- Project consequence: create a `weight-space` subsystem with separate modules for checkpoint ingestion, canonicalization, tensor encoding, latent representation, generation, interpolation, evaluation and provenance. Do not conflate checkpoint search, checkpoint merging and full checkpoint generation.

### E. Weight permutation symmetry and model alignment

A neural network can often represent the same function after permuting hidden units/channels and applying corresponding permutations to connected tensors. Naively interpolating or training a generator over unaligned checkpoints can produce meaningless averages. A canonicalization/alignment stage is therefore not optional for serious weight-space modeling.

- Git Re-Basin is used in recent weight-space generation work; see the DeepWeightFlow paper above.
- Implement alignment as an explicit, testable transform with before/after functional checks. Store the alignment permutation and original checkpoint hash.
- Do not average unrelated models merely because tensor shapes match.
- Test activation similarity, assignment cost, output equivalence and interpolation behavior.

### F. Weight-space diffusion versus flow matching

Both approaches learn a generative process over parameter representations. Diffusion commonly learns denoising across noise levels. Flow matching learns a vector field that transports samples along a chosen probability path. The DeepWeightFlow paper is an important signal that flow-matching approaches may improve efficiency in this specific domain; it does not establish that flow matching always beats diffusion.

Implement the generator interface so both are swappable:

```text
WeightGenerator
  fit(checkpoint_dataset, architecture_schema, conditioning_schema)
  sample(condition, seed, target_budget)
  interpolate(a, b, t)
  score(candidate, evaluator)
  export(candidate, format)
```

### G. Manual numeric weight construction

The product should offer genuinely manual construction, but at multiple useful levels:

- scalar/tensor inspector for tiny models;
- layer-level parameter editor;
- matrix/vector and block editors;
- seeded initialization controls;
- structured formulas and tensor expressions;
- low-rank update editor (LoRA-style delta matrices);
- sparsity masks and structured pruning;
- copy, interpolate, normalize, clamp, quantize/dequantize and compare operations;
- an agent proposal panel showing exact tensor names, shapes, before/after hashes, changed element count, norms and expected resource impact.

Do not put a giant editable grid of millions of floating point values in the main UI. Use paginated tensor views, histograms, slices, heatmaps and formula operations. A dedicated low-level mode can edit individual scalar values for small parameter tensors.

### H. Key scientific lesson

The promising direction is not “replace training with a language model typing numbers.” It is to learn reusable structure in weight space, let agents manipulate high-level or sparse/low-rank controls, and evaluate every candidate. This creates a useful hybrid system.

## 2.2 Gradient training, evolution and reinforcement learning are different tools

### A. Backpropagation and gradient-based training

When the objective is differentiable (for example, next-token cross entropy or a diffusion/flow-matching training loss), backpropagation gives a much more direct signal than black-box weight search. It should remain the primary trainer for supervised pretraining and fine-tuning.

Use:

- AdamW or another validated optimizer as a default baseline;
- SGD where appropriate for vision/small networks;
- mixed precision only when the backend supports it safely;
- gradient clipping, checkpointing, accumulation and early stopping;
- LoRA/adapters for resource-constrained fine-tuning;
- distillation where a larger teacher can provide training signals.

### B. Evolution Strategies and neuroevolution

Evolution Strategies (ES) optimize expected reward by sampling parameter perturbations and evaluating fitness. They can be useful when gradients are unavailable, the system is black-box, the environment is non-differentiable, or parallel rollouts are cheap. They are not generally a faster replacement for backpropagation on differentiable supervised tasks.

OpenAI's ES analysis explicitly notes that ES can compete in RL-style settings but can be dramatically slower than backpropagation for supervised learning (their reported MNIST example found up to a 1,000× slowdown in preliminary experiments).

- Reference: https://openai.com/index/evolution-strategies/
- Implement: antithetic perturbations, population tracking, reward normalization, checkpointed seeds and parallel workers.
- Do not compare ES to RL using an advantage that gives ES more environment steps or more compute without disclosing it.

### C. Quality-Diversity / MAP-Elites

Quality-Diversity (QD) methods aim to find many diverse high-performing solutions, not merely one maximum. MAP-Elites stores candidates across behavioral descriptors, while novelty/local competition methods explore underrepresented behaviors. This is highly relevant for a model studio: users may want a set of small models with different latency, parameter count, memory profile, style or task strengths.

- Survey: Qin et al., *A survey on Quality-Diversity optimization: Approaches, applications, and challenges*, published in *Swarm and Evolutionary Computation*, January 2026: https://www.sciencedirect.com/science/article/pii/S2210650225003979
- Implement a QD archive using explicit behavior descriptors: validation score, latency, peak memory, model size, task profile, diversity measure.
- Preserve Pareto-front candidates rather than always overwriting the “best” checkpoint.

### D. RL for language model post-training

RL is usually a post-training or interactive optimization stage after a model has acquired basic language competence. It can optimize behavior against human preference, AI feedback or verifiable task rewards. It cannot create language competence from nothing under a tiny local compute budget.

Important methods:

- **PPO:** policy-gradient method with clipped updates and a reference/KL constraint in common LLM setups.
- **GRPO:** group-based relative advantage estimation; commonly samples multiple completions per prompt and uses relative reward within the group.
- **RLHF/RLAIF:** reward signals from human or AI preferences.
- **RLVR:** rewards from automatically verifiable outcomes such as mathematical correctness or test success.
- **DPO:** direct preference optimization; useful but should be labeled preference optimization, not conventional online RL.

Useful research and implementation references:

- ACL 2025 survey, *A Comprehensive Survey on Learning from Rewards for Large Language Models*: https://aclanthology.org/2025.findings-emnlp.970/
- Hugging Face TRL GRPO trainer docs: https://huggingface.co/docs/trl/grpo_trainer
- RL post-training survey: https://arxiv.org/abs/2407.16216

### E. Reward hacking and evaluator security

A stronger optimizer can exploit a weak reward. An agent may modify tests, suppress logs, optimize a proxy metric, or exploit a benchmark's assumptions. The 2026 survey on reward hacking in agentic systems categorizes failures from feature-level shortcuts to environment-level tampering.

- Survey: https://link.springer.com/article/10.1007/s44163-026-01980-z
- Required mitigations: independent verifier process, read-only benchmark fixtures, pristine evaluation containers, sealed test data, signed artifacts, append-only run logs, explicit permission boundaries and human approval for destructive operations.

### F. Hybrid training strategy — default recommended recipe

Implement a reproducible hybrid pipeline rather than positioning one optimizer as universally superior:

1. Choose model family and exact parameter budget.
2. Build architecture from a validated schema.
3. Initialize with tested initializers or a compatible pretrained checkpoint.
4. Optionally generate initial weights with a hypernetwork / D2NWG-like diffusion / DeepWeightFlow-like flow model.
5. Run functional smoke tests and finite-value checks.
6. Supervised pretraining or fine-tuning with backpropagation.
7. Optional preference optimization / DPO for preference datasets.
8. Optional RLVR/PPO/GRPO for verifiable interactive objectives.
9. Optional ES/QD search for nondifferentiable controls, architecture variants or black-box evaluation.
10. Distill or select candidates using a held-out validation set.
11. Run independent final benchmarks and export with complete provenance.

This does not mean every job uses every method. The planner must recommend the simplest method that matches the objective and available hardware.

## 2.3 Model families that the architecture must support

### A. Autoregressive text models / LLMs

- Decoder-only Transformer first.
- Tokenizer, embeddings, attention, MLP/SwiGLU, normalization, positional encoding/RoPE, KV cache and generation sampler must be explicit components.
- Support small models from scratch for educational experiments and compatible checkpoint adapters for transfer/fine-tuning.
- Expose context length, vocabulary, tied embeddings, layers, hidden dimension, attention heads, KV heads and feed-forward dimension.
- Reasoning is not just a layer type. Treat reasoning capability as a combination of base model quality, training data, post-training, verification, inference strategy and compute budget.

### B. Diffusion language models / text diffusion

LLaDA demonstrates a masked-token diffusion language model trained from scratch with pretraining and SFT. FS-DFM (ICLR 2026) explores few-step discrete flow matching and reports an 8-step result matching a 1,024-step baseline in its evaluated setting. `dLLM` provides an open-source framework for training, inference and evaluation of diffusion language models.

- LLaDA, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/48b383b24230e0e6e649d9c98dae4d8c-Abstract-Conference.html
- FS-DFM, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/20b8fd57404ab88a8b5dea0f05e2080f-Abstract-Conference.html
- dLLM, ACL 2026 Demo: https://aclanthology.org/2026.acl-demo.8/

Implement a separate DLM trainer and sampler rather than assuming autoregressive next-token loss and decoding apply unchanged. Evaluate masked-token reconstruction, text quality, instruction following, math/code tasks, latency and sampling-step trade-offs.

### C. Image generation

The first serious image pipeline should use latent diffusion / diffusion transformers or rectified-flow models. Latent diffusion reduces training cost by doing diffusion in a learned latent space instead of pixel space. Rectified flow is a related generative formulation and has been used for high-resolution text-to-image systems.

- Latent Diffusion Models: https://doi.org/10.1109/CVPR52688.2022.01042
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, ICML 2024: https://proceedings.mlr.press/v235/esser24a.html
- Diffusers training overview: https://huggingface.co/docs/diffusers/training/overview
- DreamBooth and LoRA notes: https://huggingface.co/docs/diffusers/training/dreambooth

Do not promise training a high-quality general image foundation model from scratch on a consumer PC. Support small toy diffusion/flow models from scratch, fine-tuning/LoRA where supported, and import/export of compatible pretrained checkpoints. Image-generation pipelines include text encoder, tokenizer, VAE/latent encoder, denoiser/DiT, scheduler/flow solver and safety/metadata handling; these are separate artifacts and must be versioned together.

### D. Agentic models

“Agentic” is a system capability, not a magic parameter category. It depends on model behavior, tool schemas, planning, memory, orchestration, environment feedback, error handling, policy and permission boundaries. Train/evaluate the model and the agent harness separately.

- Include tool-call datasets and verifiable sandbox environments.
- Store a record of each tool schema version and execution result.
- Evaluate direct function-call accuracy separately from end-to-end agent task completion.

### E. “JEv” model family

The initial request contains the term “JEv models”, which is not a sufficiently unambiguous public architecture label in this context. Do not invent a definition. Keep the model-family registry extensible and add a `custom/unknown` adapter placeholder in the architecture schema. Before implementing a named JEv family, inspect the user's later definition, an upstream paper or a referenced model. Never silently reinterpret it as another architecture.

## 2.4 Evaluation and benchmarks

Build a benchmark registry, not one universal score. Every score needs task version, dataset split, evaluator version, model hash, prompt/template, seed, hardware, runtime, precision, batch size and resource use.

### Text / reasoning

- LiveBench 2.0 (latest surfaced release dated 25 October 2026; contamination-conscious, objective task families): https://livebench.ai/v1/
- GPQA Diamond / MMLU-Pro / math suites: optional adapters; verify current license and exact split before use.
- HumanEval+ / MBPP+ / LiveCodeBench for code; do not rely on static code benchmarks alone.
- IFEval or an equivalent verifiable instruction-following set.
- Perplexity / negative log likelihood on a held-out dataset where the model architecture supports it.

### Agentic / tool use

- BFCL v4: https://gorilla.cs.berkeley.edu/leaderboard — function/tool-call accuracy, multi-turn and agentic evaluation.
- Terminal-Bench 2.0: https://www.tbench.ai/?version=2.0 — terminal task execution in controlled environments.
- GAIA: https://hal.cs.princeton.edu/gaia — general assistant tasks requiring reasoning, multimodal understanding, browsing and tool use.
- SWE-bench Pro V2: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2 — long-horizon software engineering; freeze the exact dataset/protocol and inspect known benchmark audits before publishing scores.
- OSWorld for desktop/computer-use agents: https://os-world.github.io/ (verify current version and tasks before integration).

### Image generation

- T2I-CompBench / T2I-CompBench++: compositional prompts, attribute binding, object relations, numeracy and complex scene structure. https://arxiv.org/abs/2307.06350 and https://ieeexplore.ieee.org/abstract/document/10847875/
- ImageReward: preference-oriented scoring for text-to-image results. https://arxiv.org/abs/2304.05977
- Add CLIP-style alignment metrics and human preference review only as complementary signals. No single automated image score is a complete measure of quality.
- Record image seeds, prompt, negative prompt, model component hashes, scheduler/flow settings, dimensions and inference steps.

### Diffusion text

- Compare quality and cost at equivalent budgets (sample steps, wall-clock, token count, memory, FLOPs where measurable).
- Evaluate masked reconstruction, conditional completion, instruction following, math/code, long-form coherence and sampling-step scaling.
- Do not use perplexity alone when the model's likelihood estimator differs from the autoregressive baseline.

### Benchmark rules

1. Public benchmark results are not training data by default.
2. Use held-out local smoke sets for rapid iteration and sealed splits for final results.
3. Include trivial baselines, random baselines and established pretrained baselines when licenses permit.
4. Report confidence intervals or repeated-seed dispersion where practical.
5. Report cost, latency and memory beside quality.
6. Never combine unlike metrics into an opaque leaderboard score.
7. Do not compare different tokenizers, context lengths, image resolutions or inference budgets without clear stratification.
8. Preserve failed runs. A run with a broken evaluator is `invalid`, not a zero, and must explain why.

---

# 3. Product requirements and user journeys

## 3.1 Core navigation

Create a clear application shell with these primary areas:

1. **Home / Dashboard:** recent projects, last runs, hardware status, quick-start cards and resource warnings.
2. **Model Studio:** architecture builder, parameter budget, model-family selection and architecture graph.
3. **Weight Studio:** tensor explorer, manual editor, expression editor, weight patches and agent proposals.
4. **Generator Lab:** HyperNetworks, weight-space diffusion, flow matching, checkpoint alignment and sampling.
5. **Training Lab:** supervised training, fine-tuning, LoRA, preference/RL training, evolutionary search and schedules.
6. **Agent Lab:** agent runs, plans, tools, memory, budget, approvals and traces.
7. **Evaluation Lab:** benchmark selection, controlled runs, comparisons and Pareto fronts.
8. **Model Library:** local checkpoints, adapters, dataset references, formats, licenses and provenance.
9. **History:** persistent experiments, comparisons, logs, diffs and rollback.
10. **API / MCP:** local service status, endpoints, MCP tools, auth, connections and examples.
11. **Tutorials / Docs:** zero-to-hero courses, interactive explanations and reference docs.
12. **Settings:** device/backend, resource budgets, language, theme, privacy, storage, cache, downloads, safety and update preferences.

## 3.2 Essential user journeys

### Journey A: beginner creates a small network

- Select “Create a model”.
- Choose “Text”, “Image”, “Diffusion text”, “Vision” or “Custom”.
- Choose a beginner preset and parameter budget.
- See the architecture graph, parameter count, estimated memory and what each block does.
- Generate a deterministic initialization.
- Run a tiny dataset and observe training loss.
- Open a weight tensor and change a safe example parameter.
- Compare before/after outputs and save a checkpoint.

### Journey B: expert chooses an exact parameter count

- Enter a target parameter budget (integer count or unit such as 25M/1.2B).
- Define allowable width/depth/head/vocabulary constraints.
- Use the architecture solver to find a valid design or show the nearest feasible candidates.
- Explain why exact equality may not be achievable under the selected constraints.
- Show exact parameter count from the built module, not only an approximation.
- Save architecture JSON and reproducible seed.

### Journey C: user manually writes numerical weights

- Select a model and tensor.
- View tensor shape, dtype, values, distribution, histogram and connected layer.
- Change a scalar, slice or structured formula.
- Preview the patch, norm change, finite-value status and affected output.
- Save a named patch; optionally ask an agent to explain or propose a patch.
- Apply, compare, undo or branch from the checkpoint.

### Journey D: agent designs weights

- User chooses task, parameter budget, allowed methods, dataset/evaluator and budget.
- Agent inspects hardware and the selected architecture.
- Agent proposes a structured plan: architecture candidate, initializer/generator, patch operations, training and evaluation.
- User can approve, edit or enable a bounded auto-run policy.
- Agent executes in a sandbox, records tool calls and creates immutable run artifacts.
- Independent evaluator grades the candidate.
- Agent explains results and recommends the next iteration.

### Journey E: compare manual, generator, gradient, evolution and RL

- Fix architecture, train/validation/test split, compute budget and seed policy.
- Initialize separate candidate branches.
- Run: standard initializer + supervised training; manually authored structured weights; hypernetwork/flow-generated weights; ES/QD search; supervised + RLVR/GRPO.
- Compare same held-out tasks, same allowed compute/rollout budget and exact resource usage.
- Display quality, wall-clock, tokens/examples/rollouts, memory, stability and cost.
- Highlight that a comparison is fair only if the budget and evaluator are aligned.

## 3.3 Visual design system

The design should feel like a high-quality creative engineering studio, not a generic admin dashboard.

- Dark-first interface with a refined light theme; user-selectable accent color.
- Deep neutral surfaces, subtle gradients, restrained glass effects, crisp borders and high-contrast text.
- Do not sacrifice performance or accessibility for blur/glass effects.
- Use a consistent spacing scale, typography scale, grid, radius and motion system.
- Use visually distinct but restrained colors for model families, tensor types, training states and warning levels.
- Data visualization should prioritize readability: loss curves, parameter histograms, weight heatmaps, layer graphs, memory timelines, reward curves and Pareto fronts.
- Use tooltips with plain language and math details on demand.
- Provide “Beginner / Advanced / Research” density modes without changing the underlying experiment semantics.
- Include responsive desktop, laptop, tablet and mobile layouts. Mobile may simplify dense tensor editing but must not hide core status, history, runs, docs or API configuration.
- Support keyboard navigation, visible focus, reduced motion, high contrast and screen-reader labels.
- Never encode success/failure using color alone.

### Brand copy

Product footer and About panel should include exactly:

**Made with ❤️ by hcsmedia and timfromhcs**

Do not invent company registration numbers, claims, customer counts or partnerships.

---

# 4. Architecture and repository structure

## 4.1 Architectural principles

- Local-first, offline-capable after dependencies/models are downloaded.
- Explicit separation of UI, orchestration, ML engine, persistence, API and export formats.
- Backend adapters instead of hard-coded hardware assumptions.
- Model-family plugins with versioned capabilities.
- Reproducible execution and artifact provenance.
- Browser build is static and client-side; no hidden cloud compute.
- Keep research integrations optional and isolate experimental dependencies.
- A failed optional backend must not prevent CPU mode or the UI from opening.

## 4.2 Recommended monorepo layout

```text
hcs-neural-foundry/
├── AGENTS.md
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── plan.md
├── DECISIONS.md
├── pyproject.toml
├── package.json
├── pnpm-workspace.yaml
├── uv.lock
├── pnpm-lock.yaml
├── .github/
│   ├── workflows/ci.yml
│   ├── workflows/release.yml
│   ├── workflows/web-deploy.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── apps/
│   ├── web/                 # Vite/React/TypeScript static client
│   └── desktop/             # Tauri 2 shell + packaged local engine
├── packages/
│   ├── ui/                  # design system and reusable components
│   ├── api-client/          # typed TS client
│   ├── browser-runtime/     # ORT Web, WebGPU/WASM, IndexedDB/OPFS
│   ├── schemas/             # JSON Schema / OpenAPI shared schemas
│   └── charts/              # accessible experiment visualizations
├── python/
│   ├── hcs_foundry/         # Python core package
│   │   ├── api/
│   │   ├── cli/
│   │   ├── core/
│   │   ├── hardware/
│   │   ├── models/
│   │   │   ├── registry.py
│   │   │   ├── transformer_lm/
│   │   │   ├── diffusion_lm/
│   │   │   ├── image_diffusion/
│   │   │   ├── vision/
│   │   │   └── custom/
│   │   ├── weights/
│   │   │   ├── schema.py
│   │   │   ├── tensor_store.py
│   │   │   ├── patch.py
│   │   │   ├── initializers.py
│   │   │   ├── alignment.py
│   │   │   └── generators/
│   │   ├── training/
│   │   │   ├── supervised/
│   │   │   ├── finetune/
│   │   │   ├── preference/
│   │   │   ├── reinforcement/
│   │   │   ├── evolution/
│   │   │   └── distillation/
│   │   ├── agents/
│   │   ├── benchmarks/
│   │   ├── datasets/
│   │   ├── storage/
│   │   ├── export/
│   │   └── security/
│   └── tests/
├── mcp/
│   ├── server/
│   ├── schemas/
│   └── examples/
├── docs/
│   ├── RESEARCH.md
│   ├── ARCHITECTURE.md
│   ├── TESTING.md
│   ├── SECURITY.md
│   ├── tutorials/
│   ├── user-guide/
│   ├── api/
│   └── research-notes/
├── examples/
│   ├── tiny-mlp/
│   ├── tiny-transformer/
│   ├── weight-patches/
│   ├── es-vs-backprop/
│   └── browser-onnx/
├── scripts/
│   ├── bootstrap.*
│   ├── build-*.py
│   ├── verify-release.py
│   └── benchmark-smoke.py
├── models/                 # manifests only; do not commit large external weights
├── datasets/               # manifests/sample fixtures only
└── artifacts/              # ignored local outputs
```

Adapt to the repository if an existing structure is already sound. Do not perform a gratuitous rewrite solely to match this tree.

## 4.3 Recommended technology direction

Choose stable versions at implementation time and pin them after testing. Do not blindly use `latest` in production.

### Frontend

- TypeScript with strict mode.
- React + Vite for a static Vercel build; choose Next.js only if there is a real requirement for server-rendered routes. This project deliberately avoids inference server functions.
- Tailwind CSS or a similarly disciplined token-based styling system.
- Accessible component primitives; charting library chosen for keyboard/accessibility and performance.
- i18next or equivalent with complete locale files.
- Web Workers for heavy browser operations; never block the main UI thread with model inference, parsing large checkpoints or large tensor transforms.

### Local engine

- Python 3.11+ as the first ML orchestration layer, with PyTorch and optional ecosystem libraries.
- FastAPI (or equivalent typed local ASGI API) for local REST and streaming progress.
- Pydantic schemas / JSON Schema for every public request and artifact manifest.
- `safetensors` for safe tensor checkpoints where supported; avoid untrusted pickle-based checkpoint loading.
- `onnx` and `onnxruntime` for export and compatible inference.
- `torchao` for supported quantization workflows; check dtype/backend support per model and hardware.
- Hugging Face Transformers / Diffusers / TRL may be optional adapters; isolate version-sensitive integration code and record licenses.
- `uv` or another reproducible Python environment tool.

### Desktop shell and packaging

- Evaluate Tauri 2 for the UI shell and Rust bootstrapper. Tauri documents bundling external sidecar binaries, including Python CLI/API binaries: https://v2.tauri.app/develop/sidecar/ (verify current path/API during implementation).
- The release target is both (a) a normal installer and (b) a portable single-file launcher where technically reliable. A self-extracting executable may extract its signed payload to a versioned per-user cache/temp directory and launch a local process. Verify SHA-256 before execution and clean up safely.
- Do not mislabel a multi-file installer as a portable one-file executable. Provide exact artifact descriptions.
- Windows WebView2 runtime availability must be checked and the user must receive a clear install/fallback path.
- Provide a native local API process or embedded server; ensure shutdown and process cleanup on exit.

### Browser ML runtime

- ONNX Runtime Web with `webgpu` where supported and validated; `wasm` CPU fallback.
- Transformers.js can be used for supported ONNX model workflows: https://huggingface.co/docs/transformers.js/guides/webgpu
- ONNX Runtime Web official docs: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
- WebGPU operator coverage is a subset and documented as experimental in ONNX Runtime Web; WASM has broader ONNX operator coverage. Do capability checks at model/operator level, not simply `navigator.gpu` detection.

### MCP

- The MCP specification dated 2026-07-28 is the current research target at this snapshot. It emphasizes a more stateless protocol core, multi-round-trip requests and hardened authorization.
- Official spec: https://modelcontextprotocol.io/specification/2026-07-28
- Official TypeScript SDK v2: https://ts.sdk.modelcontextprotocol.io/v2/
- Official Python SDK v2: https://py.sdk.modelcontextprotocol.io/
- Pin the compatible SDK and follow its exact package/API generation. Do not copy old v1 examples into a v2 server.

---

# 5. Exact parameter budgets and model architecture factory

## 5.1 Parameter budget requirements

Provide both:

- `target_parameters`: user-requested budget;
- `actual_parameters`: exact count from the instantiated model;
- `budget_error`: signed and absolute difference;
- `budget_policy`: `exact`, `closest_under`, `closest`, or `at_least`;
- `constraints`: allowed width/depth/head/rank/vocabulary/experts/shape ranges.

The architecture solver must return multiple feasible candidates where possible and explain why an exact count is infeasible when constraints are discrete.

## 5.2 Example Transformer parameter estimate

For a decoder Transformer with hidden dimension `d`, layers `L`, vocabulary `V`, feed-forward dimension `f`, combined QKV projection, output projection, SwiGLU MLP and tied token embeddings, a useful approximate count is:

\[
P \approx Vd + L(4d^2 + 3df + 2d) + d
\]

This simplified expression excludes or approximates some biases, rotary constants, implementation-specific normalization, optional adapters, output bias and special modules. Untied input/output embeddings add roughly another `Vd`.

The instantiated module's exact count must always be computed by enumerating parameter tensors:

\[
P_{exact} = \sum_{t \in parameters(model)} numel(t)
\]

Implement a per-module breakdown with shared/tied parameter handling so tied tensors are not double-counted.

## 5.3 Memory estimation

Display estimates for:

- model parameters;
- gradients;
- optimizer states;
- activations;
- attention buffers;
- KV cache;
- temporary conversion/export buffers;
- dataset cache and checkpoint writes.

A rough FP32 Adam training baseline is around 16 bytes per trainable parameter for weights + gradients + two FP32 optimizer moment tensors, before activations and temporary buffers. Actual use depends on master weights, optimizer, precision and implementation. Label estimates and provide a safety margin. Never promise a run will fit from parameter size alone.

Approximate KV cache memory per batch item for a decoder model:

\[
M_{KV} \approx 2 \cdot L \cdot T \cdot H_{KV} \cdot D_{head} \cdot bytesPerElement
\]

where 2 represents keys and values, `T` is context tokens, and grouped-query attention uses `H_KV`, not necessarily the query-head count.

## 5.4 Architecture solver

The solver must:

1. validate architecture constraints;
2. use parameter-count formulas only to prune the search;
3. instantiate candidate modules to obtain exact counts;
4. reject invalid shape/head combinations;
5. estimate training and inference memory;
6. measure or predict operator/backend compatibility;
7. rank candidates by budget error, feasibility, performance heuristics and user preferences;
8. persist the full architecture config and solver version.

## 5.5 Model registry contract

Every model family plugin must expose:

```python
class ModelFamilyAdapter(Protocol):
    family_id: str
    display_name: str
    capabilities: ModelCapabilities
    def validate_config(self, config: dict) -> ValidationReport: ...
    def build(self, config: dict) -> ModelHandle: ...
    def count_parameters(self, model: ModelHandle) -> ParameterReport: ...
    def estimate_resources(self, config: dict, runtime: RuntimeProfile) -> ResourceEstimate: ...
    def smoke_test(self, model: ModelHandle) -> SmokeTestReport: ...
    def export(self, model: ModelHandle, target: ExportTarget) -> ExportReport: ...
```

Capability declarations must be specific: `train_cpu`, `train_cuda`, `train_rocm`, `infer_onnx_wasm`, `infer_onnx_webgpu`, `export_onnx`, `supports_lora`, `supports_gradient_checkpointing`, etc. A family may support inference but not training or browser export.

---

# 6. Weight Studio: manual editing, numeric construction and agent proposals

## 6.1 Tensor metadata

Each tensor view must display:

- fully qualified tensor name;
- parent layer/module;
- shape and number of elements;
- dtype and device;
- trainable/frozen flag;
- min, max, mean, standard deviation, norm and percentage of zeros;
- finite value validation;
- quantization metadata where applicable;
- source checkpoint hash and last patch identifier.

## 6.2 Patch format

All edits must be represented as immutable, versioned patch objects. Example schema:

```json
{
  "schema_version": "1.0",
  "patch_id": "uuid",
  "base_checkpoint_sha256": "...",
  "architecture_hash": "...",
  "operation": "low_rank_delta",
  "target_tensor": "layers.0.attn.q_proj.weight",
  "parameters": {
    "rank": 4,
    "alpha": 8,
    "seed": 1234
  },
  "precondition": {
    "shape": [128, 128],
    "dtype": "float32"
  },
  "validation": {
    "finite": true,
    "max_abs_delta": 0.05
  },
  "created_by": "user|agent_id",
  "created_at": "ISO-8601"
}
```

This is an illustrative payload. Implement strict schemas and reject unknown unsafe operations.

## 6.3 Allowed operations

Initial safe operation set:

- set scalar (only in explicit expert mode);
- set bounded slice;
- apply seeded initializer to a selected tensor;
- add scaled tensor delta;
- low-rank delta `W' = W + (alpha/r) A B`;
- normalize rows/columns with explicit epsilon;
- clamp to explicit bounds;
- multiply by scalar;
- apply sparsity mask;
- interpolate between two compatible and aligned tensors;
- quantize/dequantize using a declared scheme;
- copy from an architecture-compatible checkpoint;
- freeze/unfreeze parameter group.

Operations must validate shape, dtype, device, finite values, expected base hash and parameter limits. No arbitrary Python expression evaluation. If a tensor expression language is implemented, parse an allowlisted AST and execute only in a memory/time-bounded worker.

## 6.4 Agent authored weight plans

An agent never receives unrestricted direct tensor write access. It creates a proposal manifest with:

- task hypothesis;
- selected model/architecture/checkpoint;
- tensor names and expected shapes;
- mathematical operation and hyperparameters;
- seed;
- expected compute/memory use;
- reason and predicted effect;
- validation plan;
- rollback checkpoint.

The execution service validates and applies the proposal. The evaluator is separate from the proposer. The result stores both predicted and measured outcomes so the system can learn whether its design hypotheses were useful.

## 6.5 Visualization

Provide:

- tensor heatmap for small/medium tensor slices;
- histogram and quantile plots;
- layer-to-layer graph;
- parameter norm by layer;
- weight delta overlay;
- principal-component or embedding projections for checkpoint collections (clearly label approximate projections);
- candidate population / Pareto visualization;
- model architecture graph with parameter counts per module.

Do not load every tensor of a very large checkpoint into the browser just to draw a chart. Stream metadata and bounded slices from the local engine; in browser mode, use worker-based reads and a sampling policy.

---

# 7. Training Lab and algorithm implementations

## 7.1 Unified trainer interface

All trainers should implement a shared lifecycle:

```text
validate_config → estimate_resources → prepare → dry_run → train → validate
→ checkpoint → evaluate → summarize → export
```

Each run must support pause/resume where the underlying trainer allows it, cancellation, checkpoint cadence, logging, progress streaming, early stopping and an explicit terminal state (`completed`, `failed`, `cancelled`, `invalid`, `interrupted`).

## 7.2 Initializers and supervised learning

Build a tiny, testable baseline first:

- Xavier/Glorot and Kaiming/He initializers;
- zero/bias and normalization initialization;
- embeddings and positional parameters;
- deterministic seeded initialization;
- cross entropy and common task losses;
- train/validation split validation;
- data loader and reproducible shuffle;
- AdamW baseline with gradient clipping;
- loss, throughput, gradient norm, learning rate, memory and validation score logs.

## 7.3 Fine-tuning and adapters

- Full fine-tuning where feasible.
- LoRA with rank/alpha/dropout and per-module targeting.
- QLoRA only where the selected backend and quantization flow are verified.
- Adapter merge/unmerge with functional comparison.
- Distillation from a selected teacher.
- Resume checkpoint compatibility tests.

## 7.4 Preference and RL training

Implement in stages, with a tiny local environment before expensive model training:

1. reward-function interface and deterministic reward fixtures;
2. preference dataset validation;
3. DPO adapter;
4. PPO/GRPO adapter using a maintained library where possible;
5. RLVR verifiers for exact math/code/structured tasks;
6. isolated rollout runner;
7. reward diagnostics and anti-hacking tests.

Metrics include reward distribution, task success, KL/reference drift, entropy, length, completion rate, verifier mismatch and held-out accuracy. Reward must not be the only displayed metric.

## 7.5 Evolution and quality-diversity

Provide:

- simple genetic algorithm for tiny models;
- Natural Evolution Strategies / antithetic ES baseline;
- mutation/crossover operators compatible with the tensor schema;
- optional CMA-ES adapter for low-dimensional controls, not huge raw weight vectors by default;
- MAP-Elites archive with configurable behavior descriptors;
- population pruning, diversity metric, elite checkpoint and deterministic replay.

Start with low-dimensional latent controls, architecture choices, initializer seeds, adapter vectors or small networks. Searching billions of raw float weights with naive mutation is not a realistic default.

## 7.6 Weight generators

Separate training data ingestion from generator implementation. Each weight-generation experiment needs:

- checkpoint corpus manifest and license metadata;
- model-family and architecture compatibility;
- canonicalization method;
- tensor encoding and normalization;
- conditioning fields;
- train/validation/test split over source checkpoints/tasks;
- generator training config;
- sample seeds and exact output hashes;
- functional evaluation of generated models.

Implement in priority order:

1. hypernetwork for a tiny MLP or fixed small architecture;
2. weight-space encoder/decoder and interpolation;
3. simple flow-matching generator over small aligned tensors;
4. diffusion generator baseline if reproducible;
5. extend to bigger architectures only after memory and quality gates pass.

Never train a weight generator and evaluate only on the same checkpoints used to train it. Avoid leakage through near-duplicate checkpoints and sibling fine-tunes.

## 7.7 Resource-aware scheduler

- Detect CPU cores, total RAM, available disk, OS, Python environment, GPU vendor/device, supported runtime providers, VRAM/shared memory and driver/runtime versions.
- Run lightweight backend capability probes and microbenchmarks; do not trust device name alone.
- Provide profiles: `safe`, `balanced`, `performance`, `custom`.
- Allow explicit RAM/VRAM cap, thread count, worker count, cache size, batch size, context length and disk budget.
- Predict before launch; require approval or reduce batch/sequence/model size if the predicted footprint exceeds the budget.
- Use graceful fallback with explanation: WebGPU → WASM, native GPU backend → CPU, optional training feature → unsupported explanation.
- Never silently change precision or model architecture to make a run fit; ask or clearly show an auto-adjustment preview.

---

# 8. Local API, CLI and MCP control plane

## 8.1 Local HTTP API

Provide versioned endpoints with OpenAPI and generated typed clients. Suggested groups:

```text
GET    /api/v1/health
GET    /api/v1/hardware
GET    /api/v1/backends
GET    /api/v1/models
POST   /api/v1/models
GET    /api/v1/models/{id}
POST   /api/v1/models/{id}/build
GET    /api/v1/models/{id}/parameters
GET    /api/v1/models/{id}/tensors
POST   /api/v1/weight-patches/validate
POST   /api/v1/weight-patches
POST   /api/v1/generators/{id}/sample
POST   /api/v1/training-runs
GET    /api/v1/training-runs/{id}
POST   /api/v1/training-runs/{id}/pause
POST   /api/v1/training-runs/{id}/resume
POST   /api/v1/training-runs/{id}/cancel
POST   /api/v1/evaluations
GET    /api/v1/evaluations/{id}
GET    /api/v1/history
GET    /api/v1/artifacts/{id}
POST   /api/v1/export
GET    /api/v1/events/{run_id}
```

The route names can evolve, but keep semantics stable and versioned. Use SSE or WebSocket for events. API clients must receive typed error codes and actionable messages.

### Local service security

- Bind to `127.0.0.1` by default, never `0.0.0.0` by default.
- Generate a random local session token and require it for mutating endpoints.
- Strict CORS allowlist for the app origin; reject arbitrary websites.
- Add CSRF protections where cookie auth is used.
- Validate paths and prevent directory traversal/symlink escape.
- Rate-limit job creation and expensive endpoints.
- Treat imported model files and datasets as untrusted.
- Do not expose shell execution or arbitrary Python execution through the API.
- Network exposure is explicit opt-in, with firewall/auth instructions and a visible warning.

## 8.2 CLI

The Python package should expose a stable command hierarchy similar to:

```bash
hcs-foundry doctor
hcs-foundry hardware
hcs-foundry model create --family transformer-lm --params 10m --config model.json
hcs-foundry model inspect MODEL_ID
hcs-foundry weights inspect MODEL_ID --tensor layers.0.attn.q_proj.weight
hcs-foundry weights patch validate patch.json
hcs-foundry train start --config train.json
hcs-foundry eval run --suite text-smoke --model MODEL_ID
hcs-foundry history list
hcs-foundry export --model MODEL_ID --format onnx
hcs-foundry api serve
hcs-foundry mcp serve --transport stdio
```

Use `--json` for machine-readable output, exit codes for success/failure, `--dry-run` for resource/permission checks, and deterministic config files. Commands must work in Windows PowerShell and cmd as well as Linux/macOS shells. Document shell-specific quoting.

## 8.3 MCP server

Implement the current MCP protocol version supported by the selected official SDK, and verify the exact version before coding. The research snapshot is the 2026-07-28 specification.

Expose a narrow, typed, capability-based tool set. Suggested tools:

- `foundry_get_hardware_profile`
- `foundry_list_models`
- `foundry_inspect_model`
- `foundry_create_model_draft`
- `foundry_validate_weight_patch`
- `foundry_propose_weight_patch`
- `foundry_run_weight_patch` (approval/policy gated)
- `foundry_estimate_training_resources`
- `foundry_start_training` (budget/policy gated)
- `foundry_get_run_status`
- `foundry_cancel_run`
- `foundry_start_evaluation`
- `foundry_get_evaluation_report`
- `foundry_list_history`
- `foundry_export_model` (destination allowlist and approval gated)

Expose resources:

- model and architecture manifests;
- run summaries and logs;
- experiment artifacts and benchmark reports;
- tutorial/reference content;
- current capability/backend registry.

Expose prompts:

- beginner model design;
- architecture feasibility review;
- weight patch design review;
- experiment comparison;
- error analysis;
- research-paper reproduction planning.

### MCP safety policy

- MCP tools must not become a general-purpose shell or unrestricted filesystem bridge.
- The UI must show connected clients, exposed tools, invocation history and permission scope.
- Expensive or destructive tools require explicit approvals unless an administrator has configured a constrained unattended policy.
- Validate every argument against a versioned schema. Enforce resource budgets, job concurrency limits and cancellation.
- Tool output should contain concise summaries plus artifact IDs/paths accessible only within permitted workspace roots.
- Use official authorization/security guidance for remote transports. Local stdio is preferred for desktop integration. Do not put long-lived secrets in prompts, logs or MCP resources.

## 8.4 NPM and PyPI interfaces

- NPM: publish a TypeScript API client, browser utilities and optionally a CLI wrapper; keep browser and Node entry points separate.
- PyPI: publish the core Python package and CLI; ML-heavy dependencies should be optional extras (`cpu`, `cuda`, `rocm` where supported, `export`, `mcp`, `dev`).
- Package names are proposals. Check NPM/PyPI availability and naming policies immediately before release.
- Use semantic versioning, changelog, provenance/attestation where supported, package metadata, README examples, typed declarations, license and dependency audit.
- Never publish a package with unbounded dependency versions or accidental local paths.

---

# 9. Browser app on Vercel: same product philosophy, browser-native constraints

## 9.1 Deployment model

Deploy the web app as static frontend assets to Vercel. The normal web experience must not call a remote model inference API, serverless function or hosted Python backend. The browser downloads user-selected model artifacts and executes supported operations locally.

Vercel documentation confirms static output directories are served as static content: https://vercel.com/docs/builds/configure-a-build and https://vercel.com/docs/builds. Configure the project so the deploy contains only the frontend and public model manifests/demo assets.

## 9.2 Browser inference

- ONNX Runtime Web: WebGPU when supported for the exact graph/operators, WASM CPU fallback.
- Transformers.js can simplify supported transformer/ONNX use cases.
- Model manifests must declare input/output names, shapes, opset, external weight files, quantization, required operators, tested providers and known browser versions.
- Before download, calculate approximate storage and memory needs and ask permission where needed.
- Stream large weight downloads in chunks where supported; allow pause/resume and checksum verification.
- Use Web Workers and transferables where possible.
- Keep UI interactive; show download, initialization and inference progress separately.

### Browser capability facts and caveats

As of the cited ONNX Runtime Web documentation, WASM CPU is broadly available across browsers, while WebGPU availability varies by browser/OS and supports a smaller operator set; the WebGPU provider is documented as experimental. Check current compatibility at build time and runtime:

- https://onnxruntime.ai/docs/get-started/with-javascript/web.html
- https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- https://github.com/microsoft/onnxruntime/tree/main/js/web

Do not say “WebGPU supported” just because `navigator.gpu` exists. Load/compile the exact model with the provider and run a smoke inference before enabling it.

## 9.3 Browser-local storage and history

The web app should have no application server database for user history. Persist on the user's device:

- metadata and run history in IndexedDB;
- large binary model/cache files in OPFS where available;
- model/experiment manifests and selected artifacts;
- settings, tutorial progress and comparison data.

OPFS is private to the web origin, supports efficient in-place file access and can be used from workers, but is quota-limited and can be cleared with site data. Browser storage is best-effort by default unless persistence is granted.

- OPFS docs: https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system
- Storage quota/eviction: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria

Required UX:

- display estimated/used storage;
- query `navigator.storage.estimate()`;
- request `navigator.storage.persist()` at an appropriate time and explain the decision;
- provide export/import of a complete history bundle;
- provide delete-cache and delete-all-local-data controls;
- show a warning that browser cleanup, private browsing or storage eviction may remove data;
- never claim browser storage is an unconditional permanent backup.

## 9.4 Browser feature parity matrix

The web version should preserve concepts and workflows but must clearly distinguish supported operations:

| Capability | Local desktop | Browser/Vercel |
|---|---|---|
| Project/model design | Full | Full UI; supported schemas |
| Manual tensor inspection | Full, streamed | Supported ONNX/model tensors only; bounded view |
| Agent proposals | Local agent/MCP | Local browser-side rule/agent adapters only unless user deliberately connects an external/local agent |
| Training | Full within detected device budget | Small supported browser experiments only; never imply full desktop parity |
| ONNX inference | Native runtime providers | WebGPU/WASM |
| Weight generation | Python generator adapters | Only exported compatible generator models and safe browser ops |
| RL rollouts | Local sandbox | Restricted demo/small supported environments |
| History | Local app database/files | IndexedDB/OPFS on this device |
| MCP | Local stdio/HTTP | Optional user-configured remote/local connection; not an always-on server |
| Export | Full local file export | Browser download/file picker where supported |

A parity badge must say `Full`, `Limited`, `Experimental` or `Unavailable` per capability. Never hide these differences.

---

# 10. Persistence, artifacts and provenance

## 10.1 Local storage design

Use a lightweight local database (for example SQLite) for metadata and an artifact directory for large files. Keep data paths configurable and show them in Settings. Use migrations with backups.

Suggested entities:

- `projects`
- `model_definitions`
- `model_checkpoints`
- `tensor_patches`
- `datasets`
- `dataset_versions`
- `training_runs`
- `generator_runs`
- `agent_runs`
- `evaluation_runs`
- `benchmark_results`
- `artifacts`
- `tool_invocations`
- `event_logs`
- `settings`

## 10.2 Artifact manifest

Every exported model/checkpoint should carry:

- model ID and human-readable name;
- architecture family and version;
- exact parameter count;
- tensor dtype and quantization metadata;
- tokenizer/VAE/text-encoder/scheduler references where applicable;
- source checkpoint hashes;
- training method/config/seed;
- dataset IDs, hashes, source, license and split;
- software/library versions;
- hardware/runtime backend;
- benchmark reports and metrics;
- creation timestamp;
- model/data license and usage restrictions;
- provenance of any manual or agent-authored patches.

## 10.3 Checkpoint integrity

- Use SHA-256 or an appropriate content hash for each checkpoint and external data shard.
- Write to a temporary path, fsync where appropriate, verify the hash, then atomically promote.
- Never overwrite the only good checkpoint.
- Keep a retention policy with configurable disk cap.
- Support export/restore and project relocation.
- Verify imported archives against path traversal and decompression-bomb risks.

---

# 11. Security, privacy and licensing

## 11.1 Local-first privacy

- Do not upload prompts, datasets, checkpoints, logs or generated images by default.
- Telemetry is disabled by default; if added, it must be opt-in, minimal and documented.
- No remote inference fallback without a clear user action and visible destination.
- Do not hard-code API keys. Use OS credential storage or environment variables only for explicitly configured integrations.
- Logs must redact tokens, file paths where configured, and personal content where possible.

## 11.2 Untrusted model and dataset inputs

- Treat all imported files as untrusted.
- Prefer `safetensors` and ONNX over pickle-based formats.
- If legacy formats are supported, require an explicit trust warning and use a restricted conversion process.
- Validate archive contents, tensor sizes, shape metadata, dtype, checksums and expected parameter count.
- Prevent resource exhaustion by enforcing import size, tensor count, decompression, memory and timeout limits.
- Validate URLs and redirect chains for downloads; use checksums and license records.

## 11.3 Agent sandbox

- Use a temporary workspace with a configurable disk quota.
- Allow read access only to selected project/dataset roots.
- Deny network access by default for code-execution environments; allow only explicit model/data downloads through a controlled downloader.
- No arbitrary shell; map actions to typed internal operations.
- Separate generator/proposer from evaluator/verifier.
- Run evaluators in a clean process/container where feasible, with read-only benchmark tests and fresh checkouts.
- Enforce CPU/RAM/VRAM/time/rollout/token budgets and cancellation.
- Record every tool call, policy decision and approval.
- Test prompt injection through imported dataset rows, model cards, filenames, logs and documentation.

## 11.4 Licenses

Maintain a license registry for code, datasets, checkpoints, tokenizers and reward models. Before model download or export, surface the license and restrictions. Never bundle third-party model weights or datasets in the application without an explicit compatible license. Model cards and dataset provenance must travel with the experiment.

---

# 12. Localization, tutorials and documentation

## 12.1 Languages

Fully support:

- English (`en`) as the source locale;
- German (`de`);
- French (`fr`).

No key screen may be left half translated. No untranslated raw keys in the UI. Dates, decimal separators, units, numbers, file sizes and plural forms must use locale-aware formatting.

Technical API names, code, tensor identifiers and math notation remain stable; explain them in the selected language.

## 12.2 Zero-to-hero curriculum

Create a progressive course with runnable examples and checkpoints:

1. What is a neural network?
2. Tensors, parameters, layers and activations.
3. How a parameter count is calculated.
4. How a model produces an output.
5. Random initialization and why it works.
6. Manual numerical weight editing.
7. Loss functions and gradients.
8. Backpropagation and AdamW.
9. Training/validation/test splits and overfitting.
10. Fine-tuning and LoRA.
11. Evolution Strategies and population search.
12. Reinforcement learning, reward signals, PPO and GRPO.
13. HyperNetworks and weight-space representations.
14. Diffusion and flow matching in weight space.
15. Autoregressive LLM versus diffusion language model.
16. Image diffusion, latent space and DiT/flow models.
17. Agent tool use, MCP and safe execution.
18. Benchmarking, reproducibility and scientific claims.
19. Quantization, ONNX and browser inference.
20. Build and publish a reproducible model project.

Each lesson includes: learning goal, intuitive explanation, mathematical version, annotated diagram, runnable example, exercise, expected output, common mistake and solution. Do not merely write long pages; provide a usable learning path.

## 12.3 Documentation requirements

- Quick start for Windows, Linux and macOS.
- Hardware/backend compatibility table.
- Model family and parameter budget guide.
- Training algorithms explained with pseudocode and limitations.
- Weight editing and patch reference.
- API OpenAPI docs and examples.
- MCP connection examples for Claude Desktop, VS Code, Cursor and generic MCP hosts, after verifying their current config formats.
- CLI reference with Windows PowerShell examples.
- Browser storage/privacy/export guide.
- Troubleshooting guide with diagnostic bundle export (redacted by default).
- Reproduction guides for every paper-derived experimental module.
- Clear `Stable / Experimental / Unsupported` table.

---

# 13. Autonomous agent architecture and self-healing loops

## 13.1 Agent roles

Use specialized roles if they materially improve separation; avoid creating agents just for role-play.

- **Architect:** translates requirements into schemas, modules and acceptance criteria.
- **Researcher:** verifies papers, APIs, licenses and compatibility.
- **Builder:** implements a bounded task.
- **ML Scientist:** designs fair experiments and checks statistical validity.
- **QA / Verifier:** independently tests behavior and benchmark integrity.
- **Security Reviewer:** checks untrusted input, sandbox and network boundaries.
- **UX Reviewer:** checks usability, responsive layout, localization and accessibility.
- **Release Manager:** versions and packages only after gates pass.

A single orchestrator may delegate to these roles; the core requirement is independent verification, not multi-agent complexity.

## 13.2 Long-horizon task contract

Every task must include:

```yaml
id: unique-task-id
goal: measurable outcome
context:
  files_to_read: []
  constraints: []
  dependencies: []
implementation:
  allowed_paths: []
  forbidden_actions: []
acceptance_criteria: []
tests:
  unit: []
  integration: []
  e2e: []
resource_budget:
  wall_time_minutes: 0
  max_memory_mb: 0
  max_disk_mb: 0
  max_training_steps: 0
rollback:
  checkpoint_or_commit: required
completion_report:
  changed_files: required
  commands_run: required
  test_results: required
  known_limitations: required
```

Populate real budgets; never leave a zero interpreted as unlimited.

## 13.3 Self-healing development loop

For each task:

1. **Observe:** inspect repository state, logs, dependency health and test failures.
2. **Reproduce:** create a minimal failing test or deterministic reproduction.
3. **Diagnose:** identify root cause; distinguish code bug, environment issue, flaky test, invalid expectation and unsupported hardware.
4. **Plan:** propose the smallest safe correction and identify regression risk.
5. **Implement:** change only the required paths; avoid sweeping changes.
6. **Validate:** run unit tests, integration tests, lint/type checks and targeted e2e tests.
7. **Review:** inspect diff for accidental API changes, security regressions, license issues and feature placeholders.
8. **Record:** update decisions, plan, docs, changelog if needed and experiment history.
9. **Commit locally:** only after gates pass; use a descriptive message.
10. **Escalate:** after a bounded number of failed repair attempts, stop repeating the same action. Produce a diagnosis and ask for a decision when an external permission or requirement is needed.

Suggested repair budget: at most 3 materially different fix attempts per defect before escalating. Never loop indefinitely.

## 13.4 Training/evaluation self-healing loop

For ML experiments:

1. validate dataset manifest, licenses and split;
2. validate architecture and exact parameter count;
3. estimate memory and compute;
4. run a one-batch dry run;
5. verify loss is finite and gradients reach intended parameters;
6. train a short smoke run;
7. evaluate on fixed validation fixtures;
8. detect collapse, NaN/Inf, reward hacking, severe overfitting or stalled progress;
9. propose one bounded config adjustment at a time;
10. rerun from a documented seed/checkpoint;
11. preserve previous run and failed artifacts;
12. promote only after independent holdout evaluation.

Never allow the agent to change the held-out benchmark, remove hard examples, modify reference answers or report only its best seed without showing the distribution.

## 13.5 Agent autonomy levels

- **Level 0 — Explain:** read-only inspection and explanations.
- **Level 1 — Draft:** create architecture/patch/training proposals; no execution.
- **Level 2 — Approved run:** execute after user approval.
- **Level 3 — Bounded auto-run:** execute within per-task resource, filesystem, model and time budgets.
- **Level 4 — Research campaign:** run a finite experiment matrix and recommend/promote candidates only after independent validation. External publishing still requires explicit approval.

The user can set allowed levels by tool, model, workspace, dataset and cost budget. Defaults to Level 1.

---

# 14. Testing strategy and quality gates

## 14.1 Test layers

### Unit tests

- parameter count and tied-weight handling;
- architecture config validation;
- tensor shape/dtype/finiteness;
- weight patch preconditions and rollback;
- deterministic initialization;
- optimizer step and gradient reachability;
- reward functions and verifiers;
- checkpoint hash and manifest verification;
- storage migrations;
- locale key parity;
- API schema validation;
- backend capability detection.

### Integration tests

- tiny model create → train → checkpoint → reload → evaluate;
- manual patch → apply → compare → rollback;
- hypernetwork generate → smoke evaluate;
- flow generator toy run → sample → evaluate;
- ES toy environment → population → elite archive;
- GRPO toy task → rollouts → reward → update → validation;
- ONNX export → ORT native inference parity;
- ONNX export → browser WASM inference;
- WebGPU inference when CI hardware/browser supports it;
- MCP client connects → lists tools → invokes read-only tool → approval-gated mutation;
- API job start → progress stream → cancel/resume;
- history export → delete local project → restore archive.

### End-to-end tests

- first-run onboarding;
- create small model;
- inspect/edit weights;
- create and compare at least two experiments;
- history survives app restart;
- browser history survives tab reload and can be exported;
- browser site works offline after cache/model download where the browser permits;
- responsive viewport tests at mobile/tablet/desktop sizes;
- keyboard-only core journey;
- `en`, `de`, `fr` screenshots and key parity.

### Security tests

- path traversal and archive traversal;
- oversized tensor/checkpoint denial;
- malicious pickle refusal or isolated handling;
- local API auth and CORS;
- SSRF/download redirect validation;
- MCP prompt injection from hostile resource text;
- shell/metacharacter injection attempts;
- symlink escape;
- decompression bomb and memory exhaustion;
- benchmark file modification detection;
- secret scanning and dependency vulnerability scan.

## 14.2 Numerical correctness

- Compare model outputs before and after ONNX export within documented tolerances.
- Check all floating-point tensors for NaN/Inf after initialization, training, patching, loading and export.
- Validate gradient finite values and expected trainable parameter groups.
- Test deterministic repeatability where supported.
- Use numerical gradient checks for small custom operations.
- Verify patch inverse/rollback behavior.
- Test quantization quality and error metrics per tensor/layer; never label a quantized model “lossless” unless it is actually lossless.

## 14.3 Performance and resource gates

- Capture cold start, warm start, model loading, first-token/first-image latency, tokens/sec or images/sec, peak RSS, GPU memory, disk usage and browser heap where measurable.
- Establish baselines on at least one CPU-only device, one supported GPU device and one browser WASM environment. Add a WebGPU browser lane when available.
- Do not hard-code a single user machine as the only supported profile.
- Small smoke models should be deliberately tiny and run in CI within predictable resource budgets.

## 14.4 Coverage and quality policy

- Set meaningful coverage thresholds after baseline tests exist; increase them by subsystem.
- Critical math, serialization, permissions, security and data-integrity code must receive direct tests.
- No `skip` added to resolve a failure without a documented reason and issue.
- CI must report skipped tests and unsupported hardware separately from passed tests.

---

# 15. Build roadmap and milestone acceptance criteria

Do not skip milestones. Each phase must end with a working demonstration and a committed acceptance report.

## Phase 0 — Repository reconnaissance and research validation

Deliver:

- repository report;
- research notes with paper links and evidence labels;
- architecture decision records;
- initial risk register;
- package name availability checks;
- license/dependency inventory;
- `plan.md` and acceptance matrix.

Gate: no implementation assumptions remain undocumented; no existing user work is lost.

## Phase 1 — Project foundation and beautiful shell

Build:

- monorepo/tooling;
- design tokens and responsive app shell;
- full EN/DE/FR locale structure;
- navigation, onboarding and settings shell;
- typed shared schemas;
- local database migration framework;
- empty-but-real hardware status API;
- logging and error reporting.

Gate: clean install/build/test, no placeholder menu actions, all major routes render responsively.

## Phase 2 — Hardware detection and resource estimator

Build:

- CPU/RAM/disk/OS detection;
- GPU/provider capability probes;
- native backend registry;
- browser WebGPU/WASM probes;
- resource estimate model;
- device profile editor and safe defaults.

Gate: test fixtures simulate CPU-only, CUDA, ROCm, DirectML, browser WebGPU and browser WASM profiles; unsupported backends fall back with honest explanation.

## Phase 3 — Model factory and exact parameter budgets

Build:

- model registry;
- architecture schemas;
- tiny MLP/vision model;
- small autoregressive Transformer;
- parameter count breakdown;
- target budget solver;
- architecture graph and config export.

Gate: exact `numel` count matches displayed count; saved config rebuilds identical architecture/hash under the same version.

## Phase 4 — Weight Studio

Build:

- tensor explorer;
- histograms/heatmaps;
- manual scalar/slice editor;
- structured patch operations;
- patch validation/apply/undo;
- checkpoint integrity and history.

Gate: apply a patch, show exact diff, rerun evaluation, restart app and restore/rollback with full provenance.

## Phase 5 — Baseline training and evaluation

Build:

- supervised training loop;
- AdamW baseline;
- checkpoint/resume;
- loss/gradient/resource telemetry;
- small text and vision smoke datasets;
- benchmark registry and reports.

Gate: reproducible small run; training loss and validation metrics recorded; reload reproduces inference within tolerance.

## Phase 6 — Weight generation research lab

Build:

- hypernetwork toy implementation;
- checkpoint dataset manifest;
- canonicalization/alignment interface;
- weight-space encoder representation;
- flow-matching toy generator;
- D2NWG-style diffusion adapter interface;
- generated-candidate evaluation and comparison.

Gate: generated candidate is evaluated independently; training/eval data leakage tests pass; no unsupported claims in UI/docs.

## Phase 7 — Evolution and QD

Build:

- ES baseline with antithetic perturbations;
- small genetic algorithm;
- MAP-Elites archive;
- low-dimensional latent/adapter/architecture search;
- population and Pareto visualizations.

Gate: deterministic toy environment demonstrates improvement and diversity; compare against gradient baseline under equal evaluation budget.

## Phase 8 — LLM, text diffusion and image-generation pipelines

Build in separate adapters:

- autoregressive Transformer training/inference;
- DLM masked-token baseline and sampling;
- latent image diffusion/flow educational model;
- LoRA/fine-tuning adapters;
- tokenizer/VAE/denoiser/scheduler artifact registry;
- model import/export compatibility checks.

Gate: each family has one reproducible tiny example, tests and honest memory estimates. Large model training is not a CI requirement.

## Phase 9 — Reward and reinforcement trainers

Build:

- reward interface and fixtures;
- DPO adapter;
- PPO/GRPO adapter;
- verifiable reward environments;
- isolated rollout engine;
- anti-reward-hacking tests;
- supervised-vs-RL comparison templates.

Gate: toy environment run completes, evaluator is independent, reward improvements are checked against held-out task success and drift.

## Phase 10 — Agent Lab and MCP

Build:

- planner/runner/verifier separation;
- bounded task schema;
- approval manager;
- sandbox and resource caps;
- MCP tools/resources/prompts;
- trace viewer and policy controls;
- agent-generated weight patch workflow.

Gate: agent can create a small model proposal, run an approved bounded experiment, retrieve evaluation, explain result and cannot escape workspace or edit held-out tests.

## Phase 11 — Browser/Vercel version

Build:

- static SPA build;
- ONNX Runtime Web integration;
- WebGPU/WASM provider fallback;
- browser model manifest and downloader;
- IndexedDB/OPFS history;
- backup/restore;
- offline-capable shell;
- browser capability matrix.

Gate: deployed preview makes no hidden inference/backend requests; CPU WASM path works; WebGPU is tested only where exact model operators are supported; data stays local.

## Phase 12 — API, CLI, npm, PyPI and desktop packaging

Build:

- stable local API/OpenAPI;
- CLI cross-platform commands;
- npm client and browser package;
- Python/PyPI package;
- desktop shell;
- portable self-extracting launcher where robust;
- regular installer;
- software bill of materials and release verification.

Gate: clean virtual environments and clean OS test VMs install and run the CLI/API/app; package import smoke tests pass; signatures/hashes verified.

## Phase 13 — Full product QA and release candidate

Run:

- complete test suite;
- security audit;
- dependency/license audit;
- localization completeness;
- accessibility tests;
- visual regression;
- performance matrix;
- docs/tutorial validation;
- model artifact manifest validation;
- release dry run.

Gate: all critical tests pass; noncritical known limitations listed; no false claims or unimplemented controls.

## Phase 14 — Publish and deploy

Only after Phase 13 gate:

1. Confirm repository destination, branch and visibility.
2. Confirm package names and account permissions.
3. Run secret scan and `git status` review.
4. Push source and tags.
5. Build release artifacts from a clean CI environment.
6. Publish PyPI/npm packages with trusted publishing/attestation where available.
7. Publish GitHub release with hashes, checksums, SBOM, changelog and signed artifacts where possible.
8. Deploy static Vercel web build.
9. Test production web deployment on clean browsers.
10. Verify package install from public registry and desktop artifact launch from a clean machine.
11. Publish release notes in EN/DE/FR.

Do not push unfinished work merely because a remote exists. Do not create releases before tests and local installation checks pass.

---

# 16. CI/CD and release engineering

## 16.1 CI matrix

Run at minimum:

- Python lint, type checks and unit tests;
- TypeScript lint, type checks and tests;
- schema compatibility checks;
- UI build;
- Python package build and install smoke test;
- npm package build and import smoke test;
- static Vercel build;
- ONNX WASM browser smoke test;
- Windows build/package smoke test;
- Linux package smoke test;
- macOS build smoke test if runner budget/availability permits;
- license/dependency/security/secret scans.

GPU-specific jobs should be optional self-hosted lanes; CI must never pretend hardware tests ran when they were skipped.

## 16.2 Versioning

Use semantic versioning for public packages. Keep app, engine, schema and MCP compatibility versions visible. A breaking schema change requires a migration and clear release note.

## 16.3 Release artifacts

Produce only artifacts that pass clean-environment validation:

- Windows installer EXE;
- Windows portable single-file EXE if extraction, startup, shutdown and integrity checks work;
- Linux package/AppImage or other validated distribution format;
- macOS package if built and tested;
- Python wheel and source distribution;
- npm package tarball;
- static web deployment;
- CLI release archives;
- checksums and SBOM.

## 16.4 Release notes

Include:

- key features;
- new/changed APIs;
- supported backend/model-family matrix;
- benchmark methodology changes;
- known limitations;
- migration steps;
- security notes;
- artifact hashes;
- third-party licenses/model restrictions.

---

# 17. Acceptance matrix — the product is not finished until these pass

| Area | Required acceptance result |
|---|---|
| Model parameter budget | User can request a budget; UI shows exact actual count and honest feasibility/error |
| Manual weights | User can inspect and edit a small tensor and safely roll back |
| Agent weights | Agent produces a structured patch proposal; validator rejects malformed/unsafe edits |
| Generator | A small hypernetwork/flow generator produces a candidate with reproducible seed and evaluation |
| Gradient training | Tiny model trains, checkpoints, resumes and evaluates |
| RL | Toy reward environment supports a bounded update and independent held-out verification |
| Evolution | ES/QD run produces a traceable population and compares fairly to a baseline |
| LLM | Small autoregressive text model demo runs locally |
| Text diffusion | Small masked-token DLM demo runs with step-count and quality metrics |
| Image generation | Tiny educational diffusion/flow example works; pretrained pipeline import/export is separately labeled |
| Benchmarks | Results include dataset/protocol/model hashes, hardware, runtime and resource metrics |
| History | Runs and patches survive restart; export/import works |
| MCP | Read-only and approval-gated tools work with a conforming client |
| Local API | OpenAPI available, authentication and CORS tests pass |
| CLI | Fresh environment can run doctor, create, train, eval and export smoke workflow |
| NPM | Package builds, installs and imports from clean Node project |
| PyPI | Wheel installs and CLI starts from clean Python environment |
| Browser | Static site runs ONNX WASM; WebGPU used only on tested supported models |
| Browser storage | History stored locally, export/restore and delete controls work |
| Desktop | Clean-machine install and app lifecycle work; portable single-file claim verified separately |
| Languages | EN/DE/FR key parity and all primary journeys translated |
| UX/accessibility | Desktop/mobile responsive, keyboard core flow and accessibility checks pass |
| Security | No unrestricted shell, unsafe untrusted pickle loading, API exposure or benchmark tampering |
| Documentation | Zero-to-hero lessons have runnable examples and accurate explanations |
| Branding | “Made with ❤️ by hcsmedia and timfromhcs” appears in About/footer/release docs |

---

# 18. First implementation sprint — exact initial tasks

Start with these tasks in order. Do not begin by implementing every advanced ML method at once.

### Task 001 — Repository inventory

- Inspect all files and git history.
- Report existing code, branch, uncommitted changes, licenses and current tests.
- Identify reusable components and avoid destructive restructuring.
- Create `plan.md` and initial `DECISIONS.md`.

### Task 002 — Research notebook and decision log

- Validate links in Section 2.
- Read the full D2NWG and DeepWeightFlow papers and identify their exact supported architectures/tasks/data.
- Record reproducing risks, code availability and licenses.
- Create a one-page research map and a “claims we must not make” section.

### Task 003 — Monorepo skeleton and CI

- Create Python and TypeScript workspaces.
- Add lint/type/test/build commands.
- Add dependency lockfiles.
- Add secret/license scans.
- Ensure CI passes with the empty product shell.

### Task 004 — UI shell and localization

- Implement navigation, themes, layout, responsive breakpoints and locale switching.
- Add all primary routes with meaningful empty states and working navigation.
- Add footer credit.

### Task 005 — Schemas and persistence

- Define model, architecture, parameter report, patch, run, artifact, benchmark and hardware schemas.
- Create SQLite metadata repository and migration tests.
- Implement artifact directory integrity and temporary-write promotion.

### Task 006 — Hardware doctor

- Implement system profile and backend capability registry.
- Add CPU-only safe mode.
- Add `doctor` CLI command and UI device card.
- Prove that unsupported acceleration does not crash the app.

### Task 007 — Tiny model factory

- Build MLP and small Transformer.
- Add exact parameter breakdown and budget solver.
- Write deterministic architecture tests.

### Task 008 — Weight Studio vertical slice

- Add tensor metadata, bounded slice reader, histogram and patch UI.
- Apply one scalar and one low-rank patch.
- Add rollback and provenance tests.

### Task 009 — Tiny training/evaluation loop

- Train a tiny model on a bundled synthetic dataset.
- Persist checkpoints and metrics.
- Reload and compare inference.

### Task 010 — User-facing tutorial and initial release checklist

- Add “Create your first model” tutorial in EN/DE/FR.
- Create clean-machine installation scripts and a smoke test.
- Report exactly what works and what remains experimental.

Only after these ten tasks pass should the autonomous agent continue to weight-space generators, evolution, RL, multimodal models and publishing.

---

# 19. Operational rules for model-generated code and research

- Read and understand any upstream code before copying it. Preserve license notices and attribution.
- Keep paper reproduction code isolated from production code until it passes tests.
- Record deviations from published methods.
- Do not tune on public test sets while claiming held-out evaluation.
- Do not fabricate SOTA labels, leaderboard rank or performance numbers.
- Every chart must show the source run and metric definition.
- Every generated model must have a model card and provenance manifest.
- Every benchmark must have a protocol file and pinned dataset version.
- The application should distinguish “trained from scratch”, “fine-tuned”, “adapter tuned”, “weights generated”, “weights manually authored”, “evolved”, “distilled” and “merged”.
- Do not merge weight tensors from different architectures without a validated mapping/alignment strategy.
- Keep research code optional so a broken experimental dependency does not break stable inference or model inspection.
- Keep the CPU path operational even when acceleration is unavailable.
- Keep the browser build static and privacy-preserving; no hidden server-side model proxy.
- Publish only packages and artifacts that can be installed or launched in a clean environment.

---

# 20. Research and implementation references

The following list is the starting bibliography. Confirm details and versions again when each integration is implemented.

## Weight-space generation and neural parameter learning

1. Ha, Dai, Le — *HyperNetworks*, ICLR 2016: https://research.google/pubs/hypernetworks/
2. Soro et al. — *Diffusion-based Neural Network Weights Generation*, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f74d79573d71078848973009d0e99bdb-Abstract-Conference.html
3. Gupta et al. — *DeepWeightFlow: Re-Basined Flow Matching for Generating Neural Network Weights*, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/e2f7407b62f152b7fe533fbc077fddb7-Abstract-Conference.html
4. Han et al. — *A Survey of Weight Space Learning: Understanding, Representation, and Generation*, arXiv:2603.10090: https://arxiv.org/abs/2603.10090
5. *Position: Weight Space Should Be a First-Class Generative AI Modality*, arXiv:2605.18632: https://arxiv.org/abs/2605.18632

## Evolution, optimization and reward learning

6. OpenAI — *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*: https://openai.com/index/evolution-strategies/
7. Qin et al. — *A survey on Quality-Diversity optimization: Approaches, applications, and challenges*, 2026: https://doi.org/10.1016/j.swevo.2025.102240
8. Wu — *A Comprehensive Survey on Learning from Rewards for Large Language Models*, ACL Findings EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.970/
9. Hugging Face TRL GRPO trainer documentation: https://huggingface.co/docs/trl/grpo_trainer
10. *A survey of reward hacking in agentic large language model systems*, 2026: https://link.springer.com/article/10.1007/s44163-026-01980-z

## Text diffusion and image generation

11. Nie et al. — *Large Language Diffusion Models (LLaDA)*, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/48b383b24230e0e6e649d9c98dae4d8c-Abstract-Conference.html
12. Monsefi et al. — *FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models*, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/20b8fd57404ab88a8b5dea0f05e2080f-Abstract-Conference.html
13. Zhou et al. — *dLLM: Simple Diffusion Language Modeling*, ACL 2026 Demo: https://aclanthology.org/2026.acl-demo.8/
14. Rombach et al. — *High-Resolution Image Synthesis with Latent Diffusion Models*: https://doi.org/10.1109/CVPR52688.2022.01042
15. Lipman et al. — *Flow Matching for Generative Modeling*: https://arxiv.org/abs/2210.02747
16. Esser et al. — *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, ICML 2024: https://proceedings.mlr.press/v235/esser24a.html
17. Hugging Face Diffusers training overview: https://huggingface.co/docs/diffusers/training/overview
18. Hugging Face DreamBooth/LoRA guide: https://huggingface.co/docs/diffusers/training/dreambooth

## Evaluation

19. LiveBench 2.0: https://livebench.ai/v1/
20. Berkeley Function Calling Leaderboard v4: https://gorilla.cs.berkeley.edu/leaderboard
21. Terminal-Bench 2.0: https://www.tbench.ai/?version=2.0
22. GAIA / Princeton HAL: https://hal.cs.princeton.edu/gaia
23. SWE-bench Pro V2: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
24. T2I-CompBench: https://arxiv.org/abs/2307.06350
25. T2I-CompBench++: https://ieeexplore.ieee.org/abstract/document/10847875/
26. ImageReward: https://arxiv.org/abs/2304.05977

## ONNX, browser runtime, storage and packaging

27. ONNX Runtime Web compatibility and setup: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
28. ONNX Runtime WebGPU provider: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
29. Transformers.js WebGPU guide: https://huggingface.co/docs/transformers.js/guides/webgpu
30. PyTorch ONNX exporter: https://docs.pytorch.org/docs/main/onnx_export.html
31. torchao quantization overview: https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html
32. MDN Origin Private File System: https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system
33. MDN browser storage quotas/eviction: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria
34. Vercel build/output configuration: https://vercel.com/docs/builds/configure-a-build
35. Tauri sidecar documentation: https://v2.tauri.app/develop/sidecar/

## MCP

36. MCP specification 2026-07-28: https://modelcontextprotocol.io/specification/2026-07-28
37. MCP TypeScript SDK v2: https://ts.sdk.modelcontextprotocol.io/v2/
38. MCP Python SDK v2: https://py.sdk.modelcontextprotocol.io/
39. MCP authorization/security considerations: https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/security-considerations

---

# 21. Final instruction

Build **HCS Neural Foundry** as a real, research-grounded, local-first product. The most important differentiator is the complete feedback loop:

**Design architecture → create or generate weights → inspect and edit tensors → train with gradients/rewards/evolution → independently benchmark → compare and visualize → preserve history → export and automate through API/MCP.**

Keep scientific claims honest, keep the CPU/browser fallback usable, make every experiment reproducible, and do not publish until the actual product—not just the interface—passes its acceptance gates.

**Made with ❤️ by hcsmedia and timfromhcs**
