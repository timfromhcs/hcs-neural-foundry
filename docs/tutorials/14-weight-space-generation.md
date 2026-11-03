# Diffusion and flow matching in weight space

**Goal.** Read the D2NWG and DeepWeightFlow claims critically.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

Both methods learn a generative process over parameter tensors. Diffusion learns denoising across
noise levels; flow matching learns a vector field transporting noise to weights along a chosen path.

The hard part is not the generative model — it is **geometry**. Two networks with permuted hidden
units compute the same function but have unrelated parameter vectors. Averaging them is meaningless,
so serious work canonicalises first (Git Re-Basin style) and stores the permutation it applied.

This build declares both methods unsupported. The interface exists; a mocked demo does not.

## The mathematics

```
Flow matching objective: regress `v_θ(x_t, t)` onto `x_1 - x_0` along an interpolation path.
Diffusion objective: predict the noise added at level `t`, then sample by iterating the reverse process.
```

## Run it

```powershell
# Not implemented on purpose. The reproducible scopes are documented instead:
Get-Content docs/RESEARCH.md
hcs-foundry generate --help
```

## Exercise

Write down what evidence would convince you that generated weights beat a trained baseline, then check whether this repo provides it.

## What you should observe

It does not — and `docs/RESEARCH.md` says so. That is the intended outcome of this lesson.

## Common mistake

Shipping an approximation under a paper's name. It looks like progress and is actually a fabricated capability.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
