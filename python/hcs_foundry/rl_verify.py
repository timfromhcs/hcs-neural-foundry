"""Standalone policy verifier for reinforcement runs.

Invoked as ``python -m hcs_foundry.rl_verify '<json spec>'`` by
:func:`hcs_foundry.rl.verify_policy_isolated`. It has no access to the training objects and refuses to
score if the sealed reward fixture changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .rl import RuleBandit, evaluate_policy, load_policy


def run(spec: dict[str, Any]) -> dict[str, Any]:
    fixture = Path(spec["fixture_path"])
    policy = load_policy(Path(spec["policy_path"]))
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    env = RuleBandit.load(fixture)
    recorded = payload.get("rule_sha256")
    if recorded and recorded != env.rule_sha256:
        return {"state": "invalid", "error": "fixture contents do not match the recorded rule hash"}

    contexts = int(spec.get("contexts", 256))
    seed = int(spec.get("seed", 0))
    metrics = evaluate_policy(env, policy, seed=seed, contexts=contexts)
    return {
        "policy_accuracy": metrics["accuracy"],
        "policy_nll": metrics["nll"],
        "policy_entropy": metrics["entropy"],
        "contexts": metrics["contexts"],
        "rule_sha256": env.rule_sha256,
        "chance_accuracy": round(1.0 / env.n_actions, 6),
        "note": "scored in a separate process against the sealed fixture",
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
