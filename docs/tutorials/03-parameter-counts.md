# How a parameter count is calculated

**Goal.** Reproduce the exact count for an MLP and a decoder-only transformer.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

Closed-form estimates are fine for pruning a search space and dangerous for reporting.

For this transformer — vocabulary `V`, width `d`, layers `L`, SwiGLU width `f`, `kv` key/value heads,
head dimension `d/H`:

```
per block = 2d² + 2·d·kv·(d/H) + 3df + 2d
total     = V·d + L·per_block + d + (V·d if not tied)
```

Every term corresponds to a real tensor: q/o projections `2d²`, k/v projections
`2·d·kv·(d/H)`, three MLP matrices `3df`, two RMSNorm scales `2d`.

## The mathematics

```
For `V=64, d=64, L=2, H=4, kv=4, f=192`:
`per_block = 2·4096 + 2·64·4·16 + 3·64·192 + 128 = 53376`, total `= 4096 + 2·53376 + 64 = 110912`.
```

## Run it

```powershell
hcs-foundry solve --params 110912 --policy exact
hcs-foundry model create --preset tiny-transformer --name "Lesson 3"
hcs-foundry model show --model lesson-3 --json
```

## Exercise

Compute the count for `tie_embeddings: false` and confirm the difference is exactly `V·d`.

## What you should observe

With ties off, the model grows by `64 * 64 = 4096` parameters.

## Common mistake

Reporting the formula's number instead of the module's. The engine enumerates `numel` and treats the formula as a hint (ADR-003).

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
