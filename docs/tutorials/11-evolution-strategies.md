# Evolution Strategies

**Goal.** Run an antithetic ES step and read the fair-budget caveat.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

ES treats the model as a black box: sample perturbations, evaluate them, move toward the better
ones. Antithetic sampling evaluates `θ+σε` and `θ-σε` together, which halves the variance of the
estimate for the same number of evaluations.

It is the right tool when the objective is not differentiable. It is the wrong tool when it is:
on supervised tasks, ES needs orders of magnitude more compute than backprop.

## The mathematics

```
Rank-based estimate: `g = (1/n) Σ r_i ε_i`, then `θ ← θ + lr·g`, with `r` normalised ranks.
The engine reports how many examples each method consumed so a comparison cannot be quietly unequal.
```

## Run it

```powershell
hcs-foundry model create --preset weight-lab-mlp --name "ES Lab"
hcs-foundry evolve --model es-lab --population 16 --generations 5 --sigma 0.02 --lr 0.05
```

## Exercise

Run with `--population 4` and `--population 32`. How does archive coverage change?

## What you should observe

Best score ≥ initial score, a MAP-Elites archive with several cells, and a gradient baseline labelled as example-matched.

## Common mistake

Declaring ES better because it improved the score — without comparing against backprop on the same compute budget.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
