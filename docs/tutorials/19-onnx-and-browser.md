# Quantisation, ONNX and the browser

**Goal.** Verify parity instead of assuming it.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

Exporting to ONNX is a graph translation, and translations lose things: dynamic shapes, custom
operators, dtype quirks. The only acceptable proof is running the exported graph on real inputs and
comparing to the native outputs within a stated tolerance.

In the browser, availability is not the same as support: `navigator.gpu` existing does not mean your
graph's operators run on WebGPU. This app compiles the model with the provider and runs a smoke
inference before it reports success.

## The mathematics

```
Relative error: `max|native - onnx| / max|native|`. Tolerance here: `1e-4`.
Quantisation would add its own error budget — and is not implemented in this build.
```

## Run it

```powershell
hcs-foundry export --model lesson-3 --format onnx
# then open the web app, Dashboard -> Starten, and read the reported provider
```

## Exercise

Export, then deliberately change a weight in the checkpoint and re-run the browser demo without re-exporting.

## What you should observe

The browser keeps running the old graph — the file, not the checkpoint, is what the browser loaded.

## Common mistake

Claiming 'WebGPU supported' from a feature flag. The honest claim names the graph, the browser and the measurement.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
