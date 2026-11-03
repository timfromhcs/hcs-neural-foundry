# Benchmarking and reproducibility

**Goal.** Write a protocol someone else can rerun.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A benchmark result is a tuple, not a number: task version, dataset hash and split, prompt template,
seed policy, model hash, hardware, runtime, precision, batch size, decoding parameters and evaluator
version.

Without those, two numbers are not comparable — even if they share a name.

## The mathematics

```
Report distributions, not best seeds: median and interquartile range over `n` runs.
Report cost beside quality: wall-clock, memory, tokens or examples.
```

## Run it

```powershell
hcs-foundry eval --model lesson-3 --suite text-smoke --seed 1234 --batches 8 --json
hcs-foundry history --kind evaluation --json
```

## Exercise

Take one evaluation report and check whether every element of the tuple above is present.

## What you should observe

It is: metrics carry definitions, and provenance carries hashes, seeds, evaluator version and device.

## Common mistake

Averaging unrelated metrics into one 'score'. It hides every trade-off that matters.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
