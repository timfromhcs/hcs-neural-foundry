"""Weight-generator tests: leakage guard and a real (small) end-to-end run."""

from __future__ import annotations

import pytest

from hcs_foundry.errors import ValidationFailure
from hcs_foundry.schemas import GeneratorConfig


def test_generator_rejects_leaky_corpus(foundry, mlp_model):
    config = GeneratorConfig(
        target_model_id=mlp_model.id,
        corpus=[5, 6, 7, 5],  # seed 5 appears on both sides of the chronological split
        epochs=2,
    )
    with pytest.raises(ValidationFailure) as excinfo:
        foundry.generate(config)
    assert "leak" in str(excinfo.value).lower()


def test_generator_rejects_non_mlp_target(foundry, transformer_model):
    config = GeneratorConfig(target_model_id=transformer_model.id, corpus=[1, 2, 3, 4])
    with pytest.raises(ValidationFailure):
        foundry.generate(config)


@pytest.mark.heavy
def test_generator_runs_end_to_end(foundry, mlp_model):
    config = GeneratorConfig(
        target_model_id=mlp_model.id,
        corpus=[101, 202, 303, 404, 505],
        epochs=25,
        latent_dim=6,
        seed=3,
    )
    result = foundry.generate(config)
    assert result["run_id"]
    assert 0.0 <= result["generated_mean_accuracy"] <= 1.0
    assert 0.0 <= result["random_init_mean_accuracy"] <= 1.0
    assert result["checkpoint"] is not None
    assert result["report_path"]

    run = foundry.repo.get_run(result["run_id"])
    assert run.summary["claim_boundary"]
    assert "not evidence" in run.summary["claim_boundary"]

    # The generated candidate is stored as a normal checkpoint with provenance.
    record = foundry.repo.find_checkpoint_by_sha(result["checkpoint"]["sha256"])
    assert record is not None and record.origin == "generated"
