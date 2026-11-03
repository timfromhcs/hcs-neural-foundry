# Publish a reproducible model project

**Goal.** Ship artifacts with provenance and honest labels.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

A release is a set of artifacts plus the evidence that they work from a clean environment:

* code (with lockfiles) and tests;
* weights with content hashes and license;
* a model card stating what it was trained on and what it cannot do;
* benchmark reports with full provenance, including failures;
* install instructions verified in a fresh virtual environment.

Nothing in this repository is published automatically, and that is intentional.

## The mathematics

```
Artifact identity: SHA-256 of the checkpoint bytes. Reproducibility: seed + config + dataset hash.
Support matrix: which backends were actually tested, and on what hardware.
```

## Run it

```powershell
\scripts\verify.ps1
hcs-foundry export --model lesson-3 --format onnx
hcs-foundry history --model lesson-3
```

## Exercise

Write the one-paragraph model card for the model you trained, including what it must not be used for.

## What you should observe

The card cannot claim language ability: 250k parameters on a synthetic corpus demonstrates a pipeline.

## Common mistake

Publishing a checkpoint without its config, dataset hash and evaluator version. It becomes impossible to reproduce and impossible to audit.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
