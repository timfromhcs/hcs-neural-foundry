# Backpropagation and AdamW

**Goal.** Follow one optimiser step end to end.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

Backpropagation applies the chain rule backwards through the graph. AdamW adds two refinements:
per-parameter adaptive step sizes (`m`, `v`) and **decoupled** weight decay, which shrinks weights
directly instead of leaking into the gradient.

This engine uses AdamW for matrices, no decay for 1-D parameters (norms, biases), gradient clipping,
linear warmup and cosine decay.

## The mathematics

```
`m = β₁m + (1-β₁)g`, `v = β₂v + (1-β₂)g²`, `θ ← θ - lr·m̂/(√v̂+ε) - lr·wd·θ`.
```

## Run it

```powershell
hcs-foundry train --model lesson-3 --steps 200 --lr 0.003 --eval-every 50 --checkpoint-every 100
hcs-foundry history --model lesson-3 --limit 5
```

## Exercise

Run the same config twice and compare the final loss. Then change only the seed.

## What you should observe

Same seed → identical curve. Different seed → different trajectory, similar scale.

## Common mistake

Applying weight decay to norm scales. It shrinks exactly the parameters that keep activations stable.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
