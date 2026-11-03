# Hypernetworks and weight space

**Goal.** Understand what a weight generator can and cannot do.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A hypernetwork maps a compact conditioning vector to a large parameter vector. The compression is
the point: you cannot condition usefully on a million free floats, but you can condition on a task
descriptor, an architecture embedding or a dataset fingerprint.

What it buys you: a distribution over initialisations, cheap ensembles, and an interface for
agent-driven weight proposals. What it does not buy you: competence the training data never contained.

## The mathematics

```
Generator: `g: R^z -> R^P` trained to minimise `‖g(z_i) - θ_i‖²` on a checkpoint corpus.
Leakage guard: the split is by task, and overlapping task seeds raise rather than train.
```

## Run it

```powershell
hcs-foundry model create --preset weight-lab-mlp --name "Gen Target"
hcs-foundry generate --model gen-target --corpus 11,22,33,44,55,66,77,88 --epochs 200
```

## Exercise

Move one task seed from the training list to the test side and compare generated vs random-init accuracy.

## What you should observe

The report shows generated, random-init and gradient-trained accuracy on tasks the generator never saw, plus an explicit claim boundary.

## Common mistake

Evaluating generated weights on the same checkpoints used for training and reporting the training error as quality.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
