"""Standalone verifier process.

Invoked as ``python -m hcs_foundry.agent_verify '<json spec>'`` by :func:`hcs_foundry.agents.verify_isolated`.
It has no database access and no network; it loads a checkpoint, runs one suite and prints JSON.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .checkpoints import load_state
from .evaluation import evaluate
from .models.registry import build_model


def run(spec: dict[str, Any]) -> dict[str, Any]:
    handle = build_model(spec["model_config"])
    load_state(handle, spec["checkpoint_path"])
    report = evaluate(
        handle,
        suite_id=spec["suite_id"],
        checkpoint_sha256=spec.get("checkpoint_sha256", ""),
        seed=int(spec.get("seed", 424242)),
    )
    return {
        "state": report.state.value,
        "suite_id": report.suite_id,
        "metrics": [metric.model_dump() for metric in report.metrics],
        "notes": report.notes,
        "evaluator_version": report.provenance.evaluator_version,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(json.dumps({"error": "expected exactly one JSON argument"}))
        return 2
    try:
        spec = json.loads(args[0])
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid JSON: {exc}"}))
        return 2
    try:
        result = run(spec)
    except Exception as exc:  # pragma: no cover - process boundary
        print(json.dumps({"state": "invalid", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
