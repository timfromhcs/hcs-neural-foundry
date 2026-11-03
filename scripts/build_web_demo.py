"""Build the shipped browser demo model.

This script runs the real engine (solve budget → build → train → evaluate → export ONNX) and writes:

  apps/web/public/models/demo-char-lm.onnx
  apps/web/public/models/demo-manifest.json

The manifest records exactly how the numbers were produced, so the browser can display them as
*desktop-engine measurements* instead of inventing local results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from hcs_foundry.hashing import hash_file  # noqa: E402
from hcs_foundry.schemas import TrainingConfig  # noqa: E402
from hcs_foundry.services import Foundry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--budget", default="250k")
    parser.add_argument("--out", default=str(ROOT / "apps" / "web" / "public" / "models"))
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="foundry-web-demo-"))
    try:
        foundry = Foundry(workspace, probe_backends=False)
        target = _parse_budget(args.budget)
        solution = foundry.solve(target, policy="closest", family="transformer-lm", top_k=1)
        config = solution["candidates"][0]["config"]
        model = foundry.create_model("Demo Char LM", config, notes="Shipped browser example, trained with the desktop engine.")

        run = foundry.train(
            TrainingConfig(
                model_id=model.id,
                dataset_id="tiny-text",
                steps=args.steps,
                batch_size=8,
                block_size=16,
                eval_every=max(10, args.steps // 2),
                checkpoint_every=args.steps,
                max_seconds=600,
                seed=0,
            )
        )
        report = foundry.evaluate(model.id, suite="text-smoke", seed=1234, batches=4)
        export = foundry.export(model.id, fmt="onnx", checkpoint="latest")

        target_onnx = out / "demo-char-lm.onnx"
        shutil.copyfile(export["path"], target_onnx)
        sha = hash_file(target_onnx)

        dataset = foundry.repo.get_dataset("tiny-text")
        manifest = {
            "id": "demo-char-lm",
            "name": "Demo Char LM (desktop-engine example)",
            "family": "transformer-lm",
            "maturity": "experimental",
            "config": model.config.model_dump(mode="json"),
            "parameterCount": model.parameter_count,
            "architectureHash": model.architecture_hash,
            "onnx": {
                "path": "/models/demo-char-lm.onnx",
                "sha256": sha,
                "bytes": target_onnx.stat().st_size,
                "opset": export["opset"],
                "parity": {
                    "maxAbsDiff": export["max_abs_diff"],
                    "tolerance": export["tolerance"],
                    "provider": export["provider"],
                },
            },
            "training": {
                "steps": run.summary.get("steps_completed", 0),
                "batchSize": 8,
                "blockSize": 16,
                "finalLoss": run.summary.get("final_loss"),
                "valLoss": run.summary.get("val_loss"),
                "seconds": run.summary.get("seconds", 0.0),
                "device": run.summary.get("device", "cpu"),
                "seed": 0,
                "date": run.finished_at or run.started_at,
            },
            "evaluation": {
                "suite": report.suite_id,
                "metrics": [
                    {"name": metric.name, "value": metric.value, "unit": metric.unit, "definition": metric.definition}
                    for metric in report.metrics
                ],
            },
            "corpus": {"sha256": dataset.sha256, "chars": dataset.train_rows + dataset.val_rows},
            "claim": (
                "Trained for "
                f"{run.summary.get('steps_completed', 0)} steps on a synthetic character corpus and evaluated on held-out "
                "windows. It demonstrates the pipeline, not language ability. Do not cite these numbers as model quality."
            ),
        }
        (out / "demo-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"model           : {model.id} ({model.parameter_count:,} parameters)")
        print(f"train           : {run.summary.get('steps_completed')} steps, final loss {run.summary.get('final_loss'):.4f}")
        print(f"eval            : val_loss {report.metric('val_loss'):.4f}, accuracy {report.metric('next_token_accuracy'):.4f}")
        print(f"onnx            : {target_onnx} ({target_onnx.stat().st_size:,} bytes, parity {export['max_abs_diff']:.2e})")
        print(f"manifest        : {out / 'demo-manifest.json'}")
        foundry.close()
        return 0
    finally:
        if not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def _parse_budget(value: str) -> int:
    from hcs_foundry.models.solver import parse_parameter_budget

    return parse_parameter_budget(value)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
