"""Apply a structured weight patch, measure its effect, then roll back.

    python examples/weight-patch/run.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from hcs_foundry.agents import build_patch  # noqa: E402
from hcs_foundry.checkpoints import flattened_state  # noqa: E402
from hcs_foundry.schemas import PatchOperation, TransformerConfig  # noqa: E402
from hcs_foundry.services import Foundry  # noqa: E402


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="foundry-example-patch-"))
    foundry = Foundry(home, probe_backends=False)
    print(f"workspace: {home}")

    config = TransformerConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, context_length=32, seed=1)
    model = foundry.create_model("Patch Lab", config.model_dump(mode="json"))
    handle, checkpoint = foundry.handle_for(model.id, "latest")
    assert checkpoint is not None

    before = foundry.evaluate(model.id, suite="text-smoke", batches=2)
    print(f"before    : val_loss {before.metric('val_loss'):.4f}")

    state = flattened_state(handle.module)["tensors"]
    tensor = state["blocks.0.attn.q_proj.weight"]
    patch = build_patch(
        model_id=model.id,
        architecture_hash=handle.architecture_hash,
        base_checkpoint_sha256=checkpoint.sha256,
        tensor_name="blocks.0.attn.q_proj.weight",
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        operation=PatchOperation.LOW_RANK_DELTA,
        parameters={"rank": 2, "alpha": 8.0, "seed": 5, "std": 0.05},
        created_by="user",
        note="example: structured perturbation",
    )
    applied = foundry.apply_patch(model.id, patch)
    validation = applied["validation"]
    print(f"patch     : {validation['changed_elements']} elements, max|d| {validation['max_abs_delta']:.5f}, "
          f"norm change {validation['relative_norm_change']:+.3%}")

    after = foundry.evaluate(model.id, suite="text-smoke", batches=2)
    print(f"after     : val_loss {after.metric('val_loss'):.4f}")

    init = next(record for record in foundry.repo.list_checkpoints(model.id) if record.origin == "init")
    rolled = foundry.rollback(model.id, init.id)
    print(f"rollback  : head moved to {rolled['sha256'][:12]}…")

    foundry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
