# Random initialisation and why it works

**Goal.** Explain fan-in scaling and reproduce a seed.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

Random weights are not a placeholder; they decide whether gradients flow at all.

Too small: activations collapse toward zero and every layer learns the same thing.
Too large: activations explode and softmax saturates.

The usual fix is variance scaling: for ReLU-like units, `std = sqrt(2/fan_in)` (He); for symmetric
activations, `std = sqrt(2/(fan_in+fan_out))` (Xavier). This engine uses normal(0, 0.02) for
embeddings and projections plus a depth-scaled init on residual output projections, so variance stays
stable as layers are added.

## The mathematics

```
He: `std² = 2/fan_in`. Xavier: `std² = 2/(fan_in+fan_out)`.
Residual scaling here: `std = 0.02/√(2L)` for `o_proj` and `w2`.
```

## Run it

```powershell
hcs-foundry model create --preset tiny-transformer --name "Seed A"
hcs-foundry model create --preset tiny-transformer --name "Seed B"
hcs-foundry weights list --model seed-a --filter tok_emb
```

## Exercise

Change the seed in a config JSON, rebuild, and compare `std` of `tok_emb.weight`.

## What you should observe

Same seed reproduces identical statistics and identical values; a different seed changes them while keeping `std` near the configured scale.

## Common mistake

Assuming a different init is a different architecture. It is the same architecture with a different point in parameter space — which is why seeds belong in provenance.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
