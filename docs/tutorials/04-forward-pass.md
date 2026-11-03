# How a model produces an output

**Goal.** Trace one forward pass through attention and the MLP.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A token id becomes a vector (embedding). Each block does:

1. **RMSNorm** — rescale the vector to unit RMS.
2. **Attention** — every position looks at earlier positions; RoPE rotates query/key pairs so the
   model can represent relative distance; softmax turns scores into weights; the weighted sum of
   values is projected back.
3. **Residual add** — the block output is added to its input.
4. **SwiGLU MLP** — `W2(silu(W1 x) * W3 x)`.
5. **Residual add** again, then a final norm and a linear map to vocabulary logits.

The causal mask is what makes it a language model: position `t` may only see `<= t`.

## The mathematics

```
Attention: `softmax(QKᵀ/√d_head + mask) · V`. Mask entries above the diagonal are `-inf`.
Logits shape: `(batch, seq, vocab)`. Cross-entropy compares each position with the next token.
```

## Run it

```powershell
hcs-foundry train --model lesson-3 --steps 5 --block-size 16 --eval-every 5 --checkpoint-every 5
hcs-foundry eval --model lesson-3 --suite text-smoke --batches 2
```

## Exercise

Set `block_size` larger than `context_length` and observe what the trainer warns about.

## What you should observe

The trainer clamps to the context length and the estimator explicitly warns instead of silently truncating.

## Common mistake

Believing the model sees the whole sequence when no mask is applied. Without a causal mask, training loss becomes implausibly good and generation is useless.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
