"""Generate the zero-to-hero curriculum (lessons 1-10).

Keeping the generator in scripts/ makes the structure auditable: every lesson must have a goal, an
intuition, the mathematics, a runnable command, an exercise, an expected observation and a common
mistake. A lesson missing one of those is a bug, not a style choice.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
OUT.mkdir(parents=True, exist_ok=True)

LESSONS = [
    (
        "01-what-is-a-neural-network",
        "What is a neural network?",
        "Separate architecture, parameters and activations in your head.",
        """A network is three things wearing one name:

1. **Architecture** — which operations are wired together (layers, attention, norms).
2. **Parameters** — the numbers that get learned (weights, biases, norm scales).
3. **Activations** — the temporary numbers produced during one forward pass.

Only (2) is saved in a checkpoint. (1) is a config file. (3) is computed and discarded.""",
        """An MLP layer computes `y = xW + b` with `W` of shape `(fan_in, fan_out)`.
Parameter count for that layer: `fan_in x fan_out + fan_out`.

A tiny model: `16 -> 32 -> 4` has  `16*32 + 32 + 32*4 + 4 = 676` parameters.""",
        """hcs-foundry preset list
hcs-foundry model create --preset beginner-mlp --name "Lesson 1"
hcs-foundry model show --model lesson-1""",
        "Change `--preset beginner-mlp` to `weight-lab-mlp` and predict the new count before running it.",
        "The reported count matches your hand calculation exactly (676 vs the new value).",
        "Assuming parameters and activations are the same thing, then being surprised that a checkpoint is small while a long forward pass uses a lot of memory.",
    ),
    (
        "02-tensors-and-layers",
        "Tensors, parameters, layers",
        "Read a tensor shape and predict its element count without running code.",
        """A tensor is an array with a shape. Shape rules are the whole grammar of a network:

* matrix multiply needs the inner dimensions to agree: `(n, k) @ (k, m) -> (n, m)`;
* embeddings map integers to vectors: `(batch, seq) -> (batch, seq, d)`;
* attention makes a score matrix: `(batch, heads, seq, seq)` — this is why context length is expensive.

The number of elements is the product of the shape. That is the only budget number that matters for
memory.""",
        """For a weight `W: (d_in, d_out)`, `numel = d_in * d_out`.
Attention scores for one layer: `batch * heads * seq * seq` floats, times 2 for softmax buffers.""",
        """hcs-foundry weights list --model lesson-1
hcs-foundry weights slice --model lesson-1 --tensor layers.0.weight --shape 2,4""",
        "Pick the largest tensor by `numel` and explain why it is the largest, using the shape rules above.",
        "The largest tensor is a weight matrix; the slice command returns exactly `2*4` numbers.",
        "Reading `(64, 64)` as 64 parameters. It is 4096, and attention buffers are usually bigger still.",
    ),
    (
        "03-parameter-counts",
        "How a parameter count is calculated",
        "Reproduce the exact count for an MLP and a decoder-only transformer.",
        """Closed-form estimates are fine for pruning a search space and dangerous for reporting.

For this transformer — vocabulary `V`, width `d`, layers `L`, SwiGLU width `f`, `kv` key/value heads,
head dimension `d/H`:

```
per block = 2d² + 2·d·kv·(d/H) + 3df + 2d
total     = V·d + L·per_block + d + (V·d if not tied)
```

Every term corresponds to a real tensor: q/o projections `2d²`, k/v projections
`2·d·kv·(d/H)`, three MLP matrices `3df`, two RMSNorm scales `2d`.""",
        """For `V=64, d=64, L=2, H=4, kv=4, f=192`:
`per_block = 2·4096 + 2·64·4·16 + 3·64·192 + 128 = 53376`, total `= 4096 + 2·53376 + 64 = 110912`.""",
        """hcs-foundry solve --params 110912 --policy exact
hcs-foundry model create --preset tiny-transformer --name "Lesson 3"
hcs-foundry model show --model lesson-3 --json""",
        "Compute the count for `tie_embeddings: false` and confirm the difference is exactly `V·d`.",
        "With ties off, the model grows by `64 * 64 = 4096` parameters.",
        "Reporting the formula's number instead of the module's. The engine enumerates `numel` and treats the formula as a hint (ADR-003).",
    ),
    (
        "04-forward-pass",
        "How a model produces an output",
        "Trace one forward pass through attention and the MLP.",
        """A token id becomes a vector (embedding). Each block does:

1. **RMSNorm** — rescale the vector to unit RMS.
2. **Attention** — every position looks at earlier positions; RoPE rotates query/key pairs so the
   model can represent relative distance; softmax turns scores into weights; the weighted sum of
   values is projected back.
3. **Residual add** — the block output is added to its input.
4. **SwiGLU MLP** — `W2(silu(W1 x) * W3 x)`.
5. **Residual add** again, then a final norm and a linear map to vocabulary logits.

The causal mask is what makes it a language model: position `t` may only see `<= t`.""",
        """Attention: `softmax(QKᵀ/√d_head + mask) · V`. Mask entries above the diagonal are `-inf`.
Logits shape: `(batch, seq, vocab)`. Cross-entropy compares each position with the next token.""",
        """hcs-foundry train --model lesson-3 --steps 5 --block-size 16 --eval-every 5 --checkpoint-every 5
hcs-foundry eval --model lesson-3 --suite text-smoke --batches 2""",
        "Set `block_size` larger than `context_length` and observe what the trainer warns about.",
        "The trainer clamps to the context length and the estimator explicitly warns instead of silently truncating.",
        "Believing the model sees the whole sequence when no mask is applied. Without a causal mask, training loss becomes implausibly good and generation is useless.",
    ),
    (
        "05-initialisation",
        "Random initialisation and why it works",
        "Explain fan-in scaling and reproduce a seed.",
        """Random weights are not a placeholder; they decide whether gradients flow at all.

Too small: activations collapse toward zero and every layer learns the same thing.
Too large: activations explode and softmax saturates.

The usual fix is variance scaling: for ReLU-like units, `std = sqrt(2/fan_in)` (He); for symmetric
activations, `std = sqrt(2/(fan_in+fan_out))` (Xavier). This engine uses normal(0, 0.02) for
embeddings and projections plus a depth-scaled init on residual output projections, so variance stays
stable as layers are added.""",
        """He: `std² = 2/fan_in`. Xavier: `std² = 2/(fan_in+fan_out)`.
Residual scaling here: `std = 0.02/√(2L)` for `o_proj` and `w2`.""",
        """hcs-foundry model create --preset tiny-transformer --name "Seed A"
hcs-foundry model create --preset tiny-transformer --name "Seed B"
hcs-foundry weights list --model seed-a --filter tok_emb""",
        "Change the seed in a config JSON, rebuild, and compare `std` of `tok_emb.weight`.",
        "Same seed reproduces identical statistics and identical values; a different seed changes them while keeping `std` near the configured scale.",
        "Assuming a different init is a different architecture. It is the same architecture with a different point in parameter space — which is why seeds belong in provenance.",
    ),
    (
        "06-weight-editing",
        "Manual numerical weight editing",
        "Apply a patch, read its effect, and roll back safely.",
        """Editing weights only makes sense if the edit is reproducible. Hence patches:

* a patch names its **base checkpoint** by content hash;
* it declares the operation and parameters from a closed set;
* it validates shape and dtype before touching anything;
* it refuses to produce non-finite values;
* it writes a **new** checkpoint and moves a pointer — it does not overwrite bytes.

That makes "what did I change and can I undo it?" answerable.""",
        """Low-rank delta: `W' = W + (alpha/r)·A·B` with `A: (out, r)`, `B: (r, in)`.
Norm change: `(‖W'‖ − ‖W‖)/‖W‖`, computed from the actual tensors.""",
        """hcs-foundry weights list --model lesson-3 --filter q_proj
hcs-foundry weights patch --model lesson-3 --tensor blocks.0.attn.q_proj.weight --op low_rank_delta --params '{"rank":2,"alpha":4,"seed":7,"std":0.01}'
hcs-foundry eval --model lesson-3 --suite weight-sanity""",
        "Apply `scale` with factor 0.9, then apply the inverse (factor 1/0.9) and compare the norm.",
        "The norm returns close to the original but the checkpoint hash differs — the path is part of the history.",
        "Editing a tensor in place in a notebook, then being unable to reproduce the result or explain the regression.",
    ),
    (
        "07-loss-and-gradients",
        "Loss functions and gradients",
        "Compute cross-entropy and verify a gradient numerically.",
        """For next-token prediction the loss is mean cross-entropy:

`L = -(1/N) Σ log softmax(logits)[target]`

with `dL/dlogits = (softmax(logits) - onehot(target)) / N`. That single expression is why
classification training is cheap to implement and hard to get subtly wrong.

Gradients are only trustworthy if gradients *reach* parameters. The trainer's dry run checks exactly
that before a run starts.""",
        """`softmax(z)_i = exp(z_i)/Σ exp(z_j)`.
Numerical gradient check: `(L(θ+ε) - L(θ-ε)) / (2ε)` for a single scalar parameter.""",
        """hcs-foundry train --model lesson-3 --steps 1 --dry-run
hcs-foundry train --model lesson-3 --steps 4 --eval-every 2 --checkpoint-every 4""",
        "Predict the sign of the first few losses when `lr` is 10× too large, then test it.",
        "The dry run reports a finite loss, a positive gradient norm and the number of parameters that received gradients.",
        "Reporting a loss decrease without checking that gradients exist. A frozen model can also show 'improving' val loss by chance.",
    ),
    (
        "08-backprop-adamw",
        "Backpropagation and AdamW",
        "Follow one optimiser step end to end.",
        """Backpropagation applies the chain rule backwards through the graph. AdamW adds two refinements:
per-parameter adaptive step sizes (`m`, `v`) and **decoupled** weight decay, which shrinks weights
directly instead of leaking into the gradient.

This engine uses AdamW for matrices, no decay for 1-D parameters (norms, biases), gradient clipping,
linear warmup and cosine decay.""",
        """`m = β₁m + (1-β₁)g`, `v = β₂v + (1-β₂)g²`, `θ ← θ - lr·m̂/(√v̂+ε) - lr·wd·θ`.""",
        """hcs-foundry train --model lesson-3 --steps 200 --lr 0.003 --eval-every 50 --checkpoint-every 100
hcs-foundry history --model lesson-3 --limit 5""",
        "Run the same config twice and compare the final loss. Then change only the seed.",
        "Same seed → identical curve. Different seed → different trajectory, similar scale.",
        "Applying weight decay to norm scales. It shrinks exactly the parameters that keep activations stable.",
    ),
    (
        "09-splits-overfitting",
        "Splits, validation and overfitting",
        "Detect overfitting instead of celebrating a low training loss.",
        """A model that memorises its training set is not learning a rule. The only defence is data the
model never trained on, plus the discipline to look at it.

The bundled corpus is split chronologically: the first 90% of windows train, the last 10% validate.
That is a weak split (a synthetic corpus has no distribution shift), which is exactly why the reports
say what they measure.""",
        """Report both: `train_loss` (last batch) and `val_loss` (mean over held-out batches).
A gap that widens while train loss falls is overfitting, not progress.""",
        """hcs-foundry train --model lesson-3 --steps 400 --eval-every 50 --checkpoint-every 200
hcs-foundry eval --model lesson-3 --suite text-smoke --batches 8""",
        "Train a model with a very small `--batch-size` for many steps and watch the train/val gap.",
        "Run records contain both losses, and the evaluator uses its own seeds rather than the training seeds.",
        "Tuning on the validation set and calling the final number a held-out result.",
    ),
    (
        "10-finetuning-lora",
        "Fine-tuning and LoRA",
        "Understand low-rank adapters and their honest limits.",
        """Full fine-tuning updates every parameter. LoRA freezes the base and learns a low-rank update
`ΔW = (alpha/r)·A·B` per target matrix, which is cheap to store and cheap to merge.

Two things LoRA does not do: it does not add knowledge the base model never had, and it does not make
a tiny model capable. It changes *style and task alignment* within the base distribution.

This build implements LoRA-style **edits** (the `low_rank_delta` patch, `lora_rank` in training
config) but does not ship a pretrained base model, so there is nothing meaningful to fine-tune here.""",
        """Parameter cost per adapted matrix: `r·(in + out)` instead of `in·out`.
Merge: `W ← W + (alpha/r)·A·B`.""",
        """hcs-foundry weights patch --model lesson-3 --tensor blocks.1.attn.o_proj.weight --op low_rank_delta --params '{"rank":4,"alpha":8,"seed":1,"std":0.02}'
hcs-foundry eval --model lesson-3 --suite weight-sanity""",
        "Compare `r=2` and `r=16` on the same tensor: how does the norm change scale?",
        "The patch reports changed elements and relative norm change; a higher rank touches more of the matrix.",
        "Calling a low-rank edit 'fine-tuning' and implying it teaches the model a new capability.",
    ),
]

for slug, title, goal, intuition, math, runnable, exercise, observe, mistake in LESSONS:
    body = f"""# {title}

**Goal.** {goal}

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

{intuition}

## The mathematics

```
{math}
```

## Run it

```powershell
{runnable}
```

## Exercise

{exercise}

## What you should observe

{observe}

## Common mistake

{mistake}

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
"""
    (OUT / f"{slug}.md").write_text(body, encoding="utf-8")
    print(f"wrote {slug}.md")
