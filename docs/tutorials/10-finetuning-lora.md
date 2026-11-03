# Fine-tuning and LoRA

**Goal.** Understand low-rank adapters and their honest limits.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

Full fine-tuning updates every parameter. LoRA freezes the base and learns a low-rank update
`ΔW = (alpha/r)·A·B` per target matrix, which is cheap to store and cheap to merge.

Two things LoRA does not do: it does not add knowledge the base model never had, and it does not make
a tiny model capable. It changes *style and task alignment* within the base distribution.

This build implements LoRA-style **edits** (the `low_rank_delta` patch, `lora_rank` in training
config) but does not ship a pretrained base model, so there is nothing meaningful to fine-tune here.

## The mathematics

```
Parameter cost per adapted matrix: `r·(in + out)` instead of `in·out`.
Merge: `W ← W + (alpha/r)·A·B`.
```

## Run it

```powershell
hcs-foundry weights patch --model lesson-3 --tensor blocks.1.attn.o_proj.weight --op low_rank_delta --params '{"rank":4,"alpha":8,"seed":1,"std":0.02}'
hcs-foundry eval --model lesson-3 --suite weight-sanity
```

## Exercise

Compare `r=2` and `r=16` on the same tensor: how does the norm change scale?

## What you should observe

The patch reports changed elements and relative norm change; a higher rank touches more of the matrix.

## Common mistake

Calling a low-rank edit 'fine-tuning' and implying it teaches the model a new capability.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
