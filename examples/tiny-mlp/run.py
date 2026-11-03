"""Create, train, evaluate and export a tiny MLP.

Run from the repository root:

    python examples/tiny-mlp/run.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from hcs_foundry.schemas import MlpConfig, TrainingConfig  # noqa: E402
from hcs_foundry.services import Foundry  # noqa: E402


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="foundry-example-mlp-"))
    foundry = Foundry(home, probe_backends=False)
    print(f"workspace: {home}")

    config = MlpConfig(input_dim=16, hidden_dims=(32,), output_dim=4, seed=0)
    model = foundry.create_model("Example MLP", config.model_dump(mode="json"))
    print(f"model     : {model.id} with {model.parameter_count:,} parameters")
    print(f"expected  : 16*32+32+32*4+4 = {16 * 32 + 32 + 32 * 4 + 4}")

    report = foundry.parameter_report(model.id)
    print(f"formula agrees with enumeration: {report['formula_agrees']}")

    run = foundry.train(
        TrainingConfig(
            model_id=model.id,
            dataset_id="tiny-shapes",
            steps=120,
            batch_size=16,
            eval_every=60,
            checkpoint_every=120,
            max_seconds=120,
        )
    )
    print(f"training  : {run.state.value}, final loss {run.summary['final_loss']:.4f}, val {run.summary['val_loss']:.4f}")

    evaluation = foundry.evaluate(model.id, suite="shapes-smoke")
    accuracy = evaluation.metric("val_accuracy")
    print(f"evaluation: accuracy {accuracy:.3f} (chance {1 / config.output_dim:.3f})")

    export = foundry.export(model.id, fmt="onnx")
    print(f"export    : {export['bytes']:,} bytes, parity {export['max_abs_diff']:.2e}")

    foundry.close()
    print("\nNext: hcs-foundry --home", home, "history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
