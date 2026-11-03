# Tensors, parameters, layers

**Goal.** Read a tensor shape and predict its element count without running code.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A tensor is an array with a shape. Shape rules are the whole grammar of a network:

* matrix multiply needs the inner dimensions to agree: `(n, k) @ (k, m) -> (n, m)`;
* embeddings map integers to vectors: `(batch, seq) -> (batch, seq, d)`;
* attention makes a score matrix: `(batch, heads, seq, seq)` — this is why context length is expensive.

The number of elements is the product of the shape. That is the only budget number that matters for
memory.

## The mathematics

```
For a weight `W: (d_in, d_out)`, `numel = d_in * d_out`.
Attention scores for one layer: `batch * heads * seq * seq` floats, times 2 for softmax buffers.
```

## Run it

```powershell
hcs-foundry weights list --model lesson-1
hcs-foundry weights slice --model lesson-1 --tensor layers.0.weight --shape 2,4
```

## Exercise

Pick the largest tensor by `numel` and explain why it is the largest, using the shape rules above.

## What you should observe

The largest tensor is a weight matrix; the slice command returns exactly `2*4` numbers.

## Common mistake

Reading `(64, 64)` as 64 parameters. It is 4096, and attention buffers are usually bigger still.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
