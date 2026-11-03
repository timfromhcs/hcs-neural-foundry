"""Evolution Strategies vs gradient training on an equal *example* budget.

This example exists to show the comparison honestly: the report states what is matched (examples) and
what is not (wall clock, gradient cost). ES is not claimed to be faster.

    python examples/es-vs-backprop/run.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from hcs_foundry.schemas import EvolutionConfig, MlpConfig  # noqa: E402
from hcs_foundry.services import Foundry  # noqa: E402


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="foundry-example-es-"))
    foundry = Foundry(home, probe_backends=False)
    print(f"workspace: {home}")

    config = MlpConfig(input_dim=16, hidden_dims=(16,), output_dim=4, seed=0)
    model = foundry.create_model("ES Lab", config.model_dump(mode="json"))
    print(f"model     : {model.id} with {model.parameter_count:,} parameters")

    result = foundry.evolve(
        EvolutionConfig(model_id=model.id, population=12, generations=4, sigma=0.03, lr=0.05, seed=3)
    )

    print("\nmethod                         score")
    print(f"initial parameters            {result['initial']['score']:.3f}")
    print(f"ES best                       {result['best']['score']:.3f}")
    baseline = result.get("baseline") or {}
    if "score" in baseline:
        print(f"gradient ({baseline.get('steps')} steps)        {baseline['score']:.3f}")
    print(f"\nexamples consumed by ES       {result['examples_seen']:,}")
    print(f"archive cells                 {len(result['archive'])}")
    print(f"claim boundary                {result['claim_boundary']}")

    foundry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
