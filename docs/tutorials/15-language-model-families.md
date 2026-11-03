# Autoregressive vs diffusion language models

**Goal.** Compare decoding, cost and evaluation honestly.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

An autoregressive model factorises `p(x) = Π p(x_t | x_<t)` and generates left to right, one token
per step. A diffusion language model corrupts tokens with a mask and learns to denoise; generation
takes a fixed number of refinement steps and can revise earlier positions.

Consequences: AR is sequential and cannot revise; diffusion is parallel *within* a step but needs many
steps. Perplexity is not directly comparable between the two families, because the likelihood
estimators differ.

## The mathematics

```
AR loss: `-Σ log p(x_t | x_<t)`. Masked diffusion loss: `-Σ_{masked} log p(x_t | x_visible)`.
Compare at equal budget: tokens generated, wall-clock, memory, steps.
```

## Run it

```powershell
hcs-foundry train --model lesson-3 --steps 100
hcs-foundry eval --model lesson-3 --suite text-smoke --batches 8
```

## Exercise

Sketch the sampler for a masked diffusion model and count how many forward passes 64 tokens need at 8 steps.

## What you should observe

8 forward passes instead of 64 — the trade-off, before any quality claim.

## Common mistake

Comparing a 1024-step diffusion sample to a 1-step AR decode and calling it a fair speed comparison.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
