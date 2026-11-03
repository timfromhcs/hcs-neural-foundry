# What is a neural network?

**Goal.** Separate architecture, parameters and activations in your head.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A network is three things wearing one name:

1. **Architecture** — which operations are wired together (layers, attention, norms).
2. **Parameters** — the numbers that get learned (weights, biases, norm scales).
3. **Activations** — the temporary numbers produced during one forward pass.

Only (2) is saved in a checkpoint. (1) is a config file. (3) is computed and discarded.

## The mathematics

```
An MLP layer computes `y = xW + b` with `W` of shape `(fan_in, fan_out)`.
Parameter count for that layer: `fan_in x fan_out + fan_out`.

A tiny model: `16 -> 32 -> 4` has  `16*32 + 32 + 32*4 + 4 = 676` parameters.
```

## Run it

```powershell
hcs-foundry preset list
hcs-foundry model create --preset beginner-mlp --name "Lesson 1"
hcs-foundry model show --model lesson-1
```

## Exercise

Change `--preset beginner-mlp` to `weight-lab-mlp` and predict the new count before running it.

## What you should observe

The reported count matches your hand calculation exactly (676 vs the new value).

## Common mistake

Assuming parameters and activations are the same thing, then being surprised that a checkpoint is small while a long forward pass uses a lot of memory.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
