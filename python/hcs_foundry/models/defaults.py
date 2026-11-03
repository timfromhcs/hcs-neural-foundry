"""Built-in architecture presets.

Presets are starting points, not claims: each one is small enough to train on a laptop CPU in
seconds to minutes, and every one is labelled with what it is actually for.
"""

from __future__ import annotations

from typing import Any

from ..schemas import MlpConfig, TransformerConfig


def presets() -> dict[str, dict[str, Any]]:
    return {
        "tiny-transformer": {
            "description": "2-layer character transformer, ~110k parameters, trains in seconds",
            "config": TransformerConfig(
                vocab_size=64, d_model=64, n_layers=2, n_heads=4, context_length=64, seed=0
            ).model_dump(mode="json"),
        },
        "char-lm-small": {
            "description": "4-layer character transformer, ~250k parameters, context 128",
            "config": TransformerConfig(
                vocab_size=64, d_model=96, n_layers=4, n_heads=4, ffn_dim=256, context_length=128, seed=0
            ).model_dump(mode="json"),
        },
        "beginner-mlp": {
            "description": "16 -> 32 -> 32 -> 4 MLP for the synthetic classification task",
            "config": MlpConfig(input_dim=16, hidden_dims=(32, 32), output_dim=4, activation="gelu").model_dump(
                mode="json"
            ),
        },
        "weight-lab-mlp": {
            "description": "small MLP used by the weight studio, evolution and generator demos",
            "config": MlpConfig(input_dim=16, hidden_dims=(24, 24), output_dim=4, activation="gelu").model_dump(
                mode="json"
            ),
        },
        "flow-mlp": {
            "description": "single hidden layer, required by permutation alignment and flow matching",
            "config": MlpConfig(input_dim=16, hidden_dims=(24,), output_dim=4, activation="gelu").model_dump(
                mode="json"
            ),
        },
    }
