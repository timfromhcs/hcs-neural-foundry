"""Tests for schemas, hashing, hardware probing, storage, model factory and the budget solver."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hcs_foundry.errors import NotFoundError, ValidationFailure
from hcs_foundry.hardware import detect_hardware, estimate_resources
from hcs_foundry.hashing import canonical_json, hash_json, sha256_text
from hcs_foundry.models import build_model, list_families, parse_config, solve_budget
from hcs_foundry.models.solver import (
    SolverConstraints,
    human_parameters,
    parse_parameter_budget,
)
from hcs_foundry.models.transformer_lm import predict_parameters as predict_transformer
from hcs_foundry.schemas import (
    BudgetPolicy,
    MlpConfig,
    PatchOperation,
    TransformerConfig,
    WeightPatch,
    architecture_hash,
    validate_operation_params,
)
from hcs_foundry.storage import MIGRATIONS, Database, Repository, Workspace

# --------------------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------------------


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert hash_json({"a": [1, 2]}) == hash_json({"a": [1, 2]})
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_architecture_hash_is_stable_and_sensitive():
    first = TransformerConfig(d_model=64, n_heads=4)
    second = TransformerConfig(d_model=64, n_heads=4)
    different = TransformerConfig(d_model=64, n_heads=8)
    assert architecture_hash(first) == architecture_hash(second)
    assert architecture_hash(first) != architecture_hash(different)


def test_transformer_rejects_invalid_head_geometry():
    with pytest.raises(ValidationError):
        TransformerConfig(d_model=64, n_heads=3)
    with pytest.raises(ValidationError):
        TransformerConfig(d_model=64, n_heads=4, n_kv_heads=3)
    with pytest.raises(ValidationError):
        TransformerConfig(d_model=64, n_heads=64)  # head_dim 1 -> odd, unusable for RoPE


def test_mlp_rejects_absurd_hidden_layers():
    with pytest.raises(ValidationError):
        MlpConfig(hidden_dims=tuple([16] * 17))


def test_parse_parameter_budget_accepts_units():
    assert parse_parameter_budget("250k") == 250_000
    assert parse_parameter_budget("1.2M") == 1_200_000
    assert parse_parameter_budget("10m") == 10_000_000
    assert parse_parameter_budget(1234) == 1234
    with pytest.raises(ValidationFailure):
        parse_parameter_budget("many")


def test_human_parameters_formats():
    assert "k" in human_parameters(1_500)
    assert "M" in human_parameters(2_400_000)


def _patch(operation: PatchOperation, parameters: dict, shape=(4, 4)) -> WeightPatch:
    return WeightPatch(
        patch_id="patch_test",
        base_checkpoint_sha256="a" * 64,
        architecture_hash="b" * 64,
        operation=operation,
        target_tensor="layers.0.weight",
        parameters=parameters,
        precondition={"shape": shape, "dtype": "float32", "base_checkpoint_sha256": "a" * 64},
        created_by="test",
    )


def test_patch_schema_rejects_unknown_parameters():
    with pytest.raises(ValidationFailure):
        _patch(PatchOperation.SCALE, {"factor": 2.0, "surprise": 1})


def test_patch_schema_rejects_bad_ranges():
    with pytest.raises(ValidationFailure):
        _patch(PatchOperation.SPARSITY_MASK, {"keep_fraction": 0.0, "seed": 1})
    with pytest.raises(ValidationFailure):
        _patch(PatchOperation.LOW_RANK_DELTA, {"rank": 0, "alpha": 1.0, "seed": 1})
    with pytest.raises(ValidationFailure):
        _patch(PatchOperation.CLAMP, {})


def test_patch_schema_requires_matching_precondition_hash():
    with pytest.raises(ValidationError):
        WeightPatch(
            patch_id="p",
            base_checkpoint_sha256="a" * 64,
            architecture_hash="b" * 64,
            operation="scale",
            target_tensor="t",
            parameters={"factor": 1.0},
            precondition={"shape": [2], "dtype": "float32", "base_checkpoint_sha256": "c" * 64},
            created_by="test",
        )


def test_validate_operation_params_allows_optional_keys():
    params = validate_operation_params(PatchOperation.CLAMP, {"min": -1.0})
    assert params["min"] == -1.0


# --------------------------------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------------------------------


def test_hardware_profile_never_claims_untested_acceleration():
    profile = detect_hardware(probe_backends=True)
    assert profile.os
    assert profile.cpu.logical_cores >= 1
    for backend in profile.backends:
        if backend.usable:
            # CPU is benchmarked; accelerators must have run a probe too.
            assert backend.benchmark or backend.kind.value in {"mps", "directml"}
    cpu = profile.backend(__import__("hcs_foundry.schemas", fromlist=["BackendKind"]).BackendKind.CPU)
    assert cpu is not None and cpu.usable
    assert cpu.benchmark["gflops"] > 0


def test_resource_estimator_is_monotonic_and_flags_budget():
    small = TransformerConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2)
    large = TransformerConfig(vocab_size=256, d_model=256, n_layers=4, n_heads=8)
    from hcs_foundry.models.transformer_lm import predict_parameters

    small_estimate = estimate_resources(small, parameter_count=predict_parameters(small), batch_size=8)
    large_estimate = estimate_resources(large, parameter_count=predict_parameters(large), batch_size=8)
    assert large_estimate.total_train_bytes > small_estimate.total_train_bytes
    assert large_estimate.kv_cache_bytes > small_estimate.kv_cache_bytes
    tight = estimate_resources(
        large, parameter_count=predict_parameters(large), batch_size=8, available_bytes=1_000_000
    )
    assert tight.fits_budget is False
    assert tight.warnings


def test_estimator_warns_when_block_exceeds_context():
    config = TransformerConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, context_length=16)
    estimate = estimate_resources(config, parameter_count=predict_transformer(config), block_size=64)
    assert any("block_size" in warning for warning in estimate.warnings)


# --------------------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------------------


def test_migrations_are_idempotent(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    first = Database(workspace)
    assert first.version == MIGRATIONS[-1][0]
    first.close()
    second = Database(workspace)
    assert second.version == MIGRATIONS[-1][0]
    second.close()


def test_repository_roundtrip(foundry, mlp_model):
    repo = foundry.repo
    model = repo.get_model(mlp_model.id)
    assert model.parameter_count == mlp_model.parameter_count
    assert repo.list_models()[0].id == mlp_model.id
    with pytest.raises(NotFoundError):
        repo.get_model("nope")


def test_settings_and_events(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    repo = Repository(workspace)
    repo.set_setting("theme", "dark")
    assert repo.get_setting("theme") == "dark"
    repo.log_event("info", "test", "hello", {"a": 1})
    assert repo.list_events()[0]["message"] == "hello"


# --------------------------------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        MlpConfig(input_dim=7, hidden_dims=(5,), output_dim=2),
        MlpConfig(input_dim=3, hidden_dims=(), output_dim=1, bias=False),
        MlpConfig(input_dim=16, hidden_dims=(32, 32, 32), output_dim=4),
        TransformerConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, context_length=16),
        TransformerConfig(
            vocab_size=50, d_model=48, n_layers=3, n_heads=6, n_kv_heads=2, context_length=24
        ),
        TransformerConfig(
            vocab_size=64, d_model=64, n_layers=2, n_heads=4, tie_embeddings=False, context_length=32
        ),
    ],
)
def test_enumerated_parameters_match_formula(config):
    handle = build_model(config.model_dump(mode="json"))
    family = __import__("hcs_foundry.models.registry", fromlist=["get_family"]).get_family(config.family)
    predicted = family.predict_parameters(config)
    assert handle.parameter_count == predicted
    assert handle.report.total == sum(entry.numel for entry in handle.report.entries if not entry.tied_to)


def test_tied_embeddings_are_counted_once():
    tied = build_model(
        TransformerConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, tie_embeddings=True).model_dump(
            mode="json"
        )
    )
    untied = build_model(
        TransformerConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, tie_embeddings=False).model_dump(
            mode="json"
        )
    )
    assert untied.parameter_count - tied.parameter_count == 32 * 32
    assert tied.report.shared_tensors == 1
    tied_entry = next(entry for entry in tied.report.entries if entry.name == "lm_head.weight")
    assert tied_entry.tied_to == "tok_emb.weight"


def test_initialisation_is_deterministic_for_a_seed():
    config = TransformerConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, seed=11)
    first = build_model(config.model_dump(mode="json"))
    second = build_model(config.model_dump(mode="json"))
    a = first.module.state_dict()["tok_emb.weight"]
    b = second.module.state_dict()["tok_emb.weight"]
    assert a.equal(b)


def test_unknown_family_is_actionable():
    with pytest.raises(NotFoundError):
        parse_config({"family": "gpt-9"})
    assert {family["id"] for family in list_families()} == {"mlp", "transformer-lm"}


def test_transformer_rejects_sequence_longer_than_context():
    handle = build_model(
        TransformerConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2, context_length=8).model_dump(
            mode="json"
        )
    )
    import torch

    with pytest.raises(ValueError):
        handle.module(torch.zeros(1, 9, dtype=torch.long))


# --------------------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------------------


def test_solver_returns_verified_counts_and_explains_infeasibility():
    solution = solve_budget(250_000, policy=BudgetPolicy.CLOSEST, top_k=3)
    assert solution.candidates
    assert solution.candidates[0].selected
    for candidate in solution.candidates:
        assert candidate.verified
        assert candidate.actual_parameters is not None
        assert candidate.actual_parameters == candidate.predicted_parameters
    if not solution.exact_match:
        assert "exact" in solution.explanation.lower()


def test_solver_exact_policy_reports_no_match():
    solution = solve_budget(1_000_003, policy=BudgetPolicy.EXACT, top_k=2)
    assert solution.exact_match is False
    assert all(not candidate.satisfies_policy for candidate in solution.candidates)


def test_solver_respects_closest_under_policy():
    target = 200_000
    solution = solve_budget(target, policy=BudgetPolicy.CLOSEST_UNDER, top_k=3)
    assert all(candidate.actual_parameters <= target for candidate in solution.candidates)


def test_solver_respects_at_least_policy():
    target = 200_000
    solution = solve_budget(target, policy=BudgetPolicy.AT_LEAST, top_k=3)
    assert all(candidate.actual_parameters >= target for candidate in solution.candidates)


def test_solver_mlp_family():
    constraints = SolverConstraints(family="mlp", hidden_max=256, max_hidden_layers=3)
    solution = solve_budget(5_000, policy=BudgetPolicy.CLOSEST, constraints=constraints, top_k=3)
    assert solution.candidates
    assert all(candidate.config["family"] == "mlp" for candidate in solution.candidates)


def test_solver_rejects_bad_constraints():
    with pytest.raises(ValidationError):
        SolverConstraints(d_model_min=100, d_model_max=50)


def test_solver_candidates_are_distinct_parameter_counts():
    solution = solve_budget(250_000, policy=BudgetPolicy.CLOSEST, top_k=4)
    counts = [candidate.actual_parameters for candidate in solution.candidates]
    assert len(counts) == len(set(counts))
