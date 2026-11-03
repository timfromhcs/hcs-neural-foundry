# Splits, validation and overfitting

**Goal.** Detect overfitting instead of celebrating a low training loss.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A model that memorises its training set is not learning a rule. The only defence is data the
model never trained on, plus the discipline to look at it.

The bundled corpus is split chronologically: the first 90% of windows train, the last 10% validate.
That is a weak split (a synthetic corpus has no distribution shift), which is exactly why the reports
say what they measure.

## The mathematics

```
Report both: `train_loss` (last batch) and `val_loss` (mean over held-out batches).
A gap that widens while train loss falls is overfitting, not progress.
```

## Run it

```powershell
hcs-foundry train --model lesson-3 --steps 400 --eval-every 50 --checkpoint-every 200
hcs-foundry eval --model lesson-3 --suite text-smoke --batches 8
```

## Exercise

Train a model with a very small `--batch-size` for many steps and watch the train/val gap.

## What you should observe

Run records contain both losses, and the evaluator uses its own seeds rather than the training seeds.

## Common mistake

Tuning on the validation set and calling the final number a held-out result.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
