# Latent image diffusion

**Goal.** Identify the components of a text-to-image pipeline and why they are versioned together.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A text-to-image system is not one model. It is: tokenizer → text encoder → (optional) latent VAE
encoder → denoiser (UNet or DiT) → scheduler/solver → VAE decoder. Change the VAE and every stored
latent becomes incompatible.

Training cost lives mostly in the denoiser, but quality failures are frequently in the text encoder or
the scheduler. Reproducing a result requires the exact combination.

## The mathematics

```
Forward noising: `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`.
Rectified flow: `x_t = (1-t)x_0 + t·ε`, learn `v = ε - x_0`.
```

## Run it

```powershell
# Not implemented: a toy DiT that emits noise is not a pipeline demo.
Get-Content docs/RESEARCH.md -TotalCount 60
```

## Exercise

List every artifact you would have to store to reproduce one generated image.

## What you should observe

Tokenizer, text encoder, VAE weights, denoiser weights, scheduler config, prompt, seed, steps, resolution, and the library versions.

## Common mistake

Storing only the denoiser checkpoint and being surprised that the same prompt and seed produce a different image.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
