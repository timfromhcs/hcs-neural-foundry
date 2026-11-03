# Manual numerical weight editing

**Goal.** Apply a patch, read its effect, and roll back safely.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

Editing weights only makes sense if the edit is reproducible. Hence patches:

* a patch names its **base checkpoint** by content hash;
* it declares the operation and parameters from a closed set;
* it validates shape and dtype before touching anything;
* it refuses to produce non-finite values;
* it writes a **new** checkpoint and moves a pointer — it does not overwrite bytes.

That makes "what did I change and can I undo it?" answerable.

## The mathematics

```
Low-rank delta: `W' = W + (alpha/r)·A·B` with `A: (out, r)`, `B: (r, in)`.
Norm change: `(‖W'‖ − ‖W‖)/‖W‖`, computed from the actual tensors.
```

## Run it

```powershell
hcs-foundry weights list --model lesson-3 --filter q_proj
hcs-foundry weights patch --model lesson-3 --tensor blocks.0.attn.q_proj.weight --op low_rank_delta --params '{"rank":2,"alpha":4,"seed":7,"std":0.01}'
hcs-foundry eval --model lesson-3 --suite weight-sanity
```

## Exercise

Apply `scale` with factor 0.9, then apply the inverse (factor 1/0.9) and compare the norm.

## What you should observe

The norm returns close to the original but the checkpoint hash differs — the path is part of the history.

## Common mistake

Editing a tensor in place in a notebook, then being unable to reproduce the result or explain the regression.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
