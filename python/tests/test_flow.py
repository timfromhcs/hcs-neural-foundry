"""Tests for weight-space alignment + conditional flow matching.

The quality assertions are deliberately about *mechanics* (alignment is exact, the loss goes down, the
pipeline reports honestly), not about generated models beating gradient training. A negative result is
recorded in `docs/PERFORMANCE.md`, not asserted away.
"""

from __future__ import annotations

import numpy as np
import pytest

from hcs_foundry.errors import ValidationFailure
from hcs_foundry.flow import (
    FlowModel,
    from_torch_flat,
    run_flow_experiment,
    sample,
    to_torch_flat,
    train_flow,
)
from hcs_foundry.schemas import GeneratorConfig, MlpConfig


def _toy_targets(count: int = 6, dim: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    targets = rng.standard_normal((count, dim)).astype(np.float32)
    descriptors = rng.standard_normal((count, 3)).astype(np.float32)
    return targets, descriptors


# --------------------------------------------------------------------------------------------------
# Flow model mechanics
# --------------------------------------------------------------------------------------------------


def test_training_reduces_the_flow_loss_and_is_deterministic():
    targets, descriptors = _toy_targets()
    first, losses_a = train_flow(targets, descriptors, hidden=32, steps=120, seed=3)
    second, losses_b = train_flow(targets, descriptors, hidden=32, steps=120, seed=3)
    assert losses_a[-1] < losses_a[0]
    assert losses_a == losses_b
    assert np.allclose(first.weights[0], second.weights[0])


def test_sampling_produces_finite_vectors_of_the_right_size():
    targets, descriptors = _toy_targets(dim=8)
    model, _ = train_flow(targets, descriptors, hidden=16, steps=30, seed=1)
    vector = sample(model, descriptors[0], steps=8, seed=2)
    assert vector.shape == (8,)
    assert np.isfinite(vector).all()


def test_sampling_rejects_zero_steps():
    model, _ = train_flow(*_toy_targets(), hidden=8, steps=5, seed=0)
    with pytest.raises(ValidationFailure):
        sample(model, np.zeros(3, dtype=np.float32), steps=0)


def test_model_serialisation_round_trip():
    model, _ = train_flow(*_toy_targets(), hidden=12, steps=10, seed=0)
    restored = FlowModel.from_dict(model.to_dict())
    probe = np.zeros((1, model.sizes[0]), dtype=np.float32)
    assert np.allclose(model.forward(probe), restored.forward(probe))


def test_train_flow_validates_its_inputs():
    targets, descriptors = _toy_targets()
    with pytest.raises(ValidationFailure):
        train_flow(targets[:1], descriptors[:1], hidden=8, steps=1)
    with pytest.raises(ValidationFailure):
        train_flow(targets, descriptors[:-1], hidden=8, steps=1)


# --------------------------------------------------------------------------------------------------
# Layout conversions
# --------------------------------------------------------------------------------------------------


def test_torch_flat_round_trip():
    rng = np.random.default_rng(0)
    weights = {
        "w1": rng.standard_normal((4, 6)).astype(np.float32),  # (in, hidden)
        "b1": rng.standard_normal(6).astype(np.float32),
        "w2": rng.standard_normal((6, 2)).astype(np.float32),  # (hidden, out)
        "b2": rng.standard_normal(2).astype(np.float32),
    }
    shapes = {key: value.shape for key, value in weights.items()}
    restored = from_torch_flat(to_torch_flat(weights), shapes)
    for key in weights:
        assert np.allclose(restored[key], weights[key])


# --------------------------------------------------------------------------------------------------
# End-to-end experiment
# --------------------------------------------------------------------------------------------------


@pytest.mark.heavy
def test_flow_experiment_reports_alignment_reconstruction_and_boundaries(foundry):
    config = MlpConfig(input_dim=8, hidden_dims=(10,), output_dim=3, seed=0)
    model = foundry.create_model("Flow target", config.model_dump(mode="json"))
    report = run_flow_experiment(
        config,
        GeneratorConfig(target_model_id=model.id, corpus=[11, 22, 33, 44], latent_dim=4, seed=2),
        hidden=32,
        steps=150,
        integration_steps=15,
    )

    # Alignment must be exact (a permutation cannot change the function) and reported.
    assert report["alignment"]["all_verified"] is True
    assert all(item["verified"] for item in report["alignment"]["reports"])
    assert max(item["functional_max_abs_diff"] for item in report["alignment"]["reports"]) < 1e-6

    # The generator must fit the corpus it was trained on.
    assert report["generator"]["loss_curve"][-1] < report["generator"]["loss_curve"][0]
    assert report["in_distribution"], "expected reconstruction entries for training tasks"
    closest = min(item["relative_parameter_distance"] for item in report["in_distribution"])
    assert closest < 0.05

    # Out-of-distribution results and the honest framing must be present.
    assert set(report["results"]) >= {
        "generated_mean_accuracy",
        "random_init_mean_accuracy",
        "gradient_trained_mean_accuracy",
        "samples",
    }
    assert "sampling_diversity" in report
    assert "not a reproduction" in report["claim_boundary"]
    assert report["corpus"]["train_tasks"]
    assert report["corpus"]["test_tasks"]


@pytest.mark.heavy
def test_flow_experiment_rejects_too_small_a_corpus(foundry):
    config = MlpConfig(input_dim=8, hidden_dims=(10,), output_dim=3, seed=0)
    model = foundry.create_model("Flow target", config.model_dump(mode="json"))
    with pytest.raises(ValidationFailure):
        run_flow_experiment(
            config,
            GeneratorConfig(target_model_id=model.id, corpus=[1, 2], latent_dim=2),
            hidden=8,
            steps=2,
        )


@pytest.mark.heavy
def test_flow_experiment_rejects_leaky_corpus(foundry):
    config = MlpConfig(input_dim=8, hidden_dims=(10,), output_dim=3, seed=0)
    model = foundry.create_model("Flow target", config.model_dump(mode="json"))
    with pytest.raises(ValidationFailure):
        run_flow_experiment(
            config,
            GeneratorConfig(target_model_id=model.id, corpus=[5, 6, 7, 5], latent_dim=2),
            hidden=8,
            steps=2,
        )
