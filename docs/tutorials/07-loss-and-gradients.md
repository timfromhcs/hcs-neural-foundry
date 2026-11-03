# Loss functions and gradients

**Goal.** Compute cross-entropy and verify a gradient numerically.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

For next-token prediction the loss is mean cross-entropy:

`L = -(1/N) Σ log softmax(logits)[target]`

with `dL/dlogits = (softmax(logits) - onehot(target)) / N`. That single expression is why
classification training is cheap to implement and hard to get subtly wrong.

Gradients are only trustworthy if gradients *reach* parameters. The trainer's dry run checks exactly
that before a run starts.

## The mathematics

```
`softmax(z)_i = exp(z_i)/Σ exp(z_j)`.
Numerical gradient check: `(L(θ+ε) - L(θ-ε)) / (2ε)` for a single scalar parameter.
```

## Run it

```powershell
hcs-foundry train --model lesson-3 --steps 1 --dry-run
hcs-foundry train --model lesson-3 --steps 4 --eval-every 2 --checkpoint-every 4
```

## Exercise

Predict the sign of the first few losses when `lr` is 10× too large, then test it.

## What you should observe

The dry run reports a finite loss, a positive gradient norm and the number of parameters that received gradients.

## Common mistake

Reporting a loss decrease without checking that gradients exist. A frozen model can also show 'improving' val loss by chance.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
