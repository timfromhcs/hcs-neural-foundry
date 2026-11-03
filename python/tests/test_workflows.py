"""Workflow tests: checkpoints, weight patches, training, evaluation, evolution and agent policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hcs_foundry.agents import AgentPolicy, build_patch
from hcs_foundry.checkpoints import flattened_state, load_state
from hcs_foundry.errors import NotFoundError, SecurityViolation, ValidationFailure
from hcs_foundry.evaluation import evaluate, get_suite, list_suites
from hcs_foundry.schemas import (
    EvolutionConfig,
    PatchOperation,
    PatchPrecondition,
    TrainingConfig,
    WeightPatch,
)
from hcs_foundry.training.trainer import SupervisedTrainer, TrainerControl
from hcs_foundry.weights import get_tensor, histogram, slice_tensor, summarize

# --------------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------------


def test_checkpoint_roundtrip_restores_values_and_tying(foundry, transformer_model):
    handle, record = foundry.handle_for(transformer_model.id, "latest")
    before = handle.module.state_dict()["tok_emb.weight"].detach().clone()
    handle.module.state_dict()["tok_emb.weight"].add_(1.0)
    load_state(handle, record.path)
    after = handle.module.state_dict()["tok_emb.weight"].detach().clone()
    assert before.allclose(after)
    # Tied weights stay physically shared after loading.
    assert handle.module.lm_head.weight.data_ptr() == handle.module.tok_emb.weight.data_ptr()


def test_checkpoint_rejects_architecture_mismatch(foundry, transformer_model, mlp_model):
    _, record = foundry.handle_for(transformer_model.id, "latest")
    other_handle, _ = foundry.handle_for(mlp_model.id, None)
    with pytest.raises(ValidationFailure):
        load_state(other_handle, record.path)


def test_checkpoint_refuses_non_safetensors(tmp_path, foundry, mlp_model):
    handle, _ = foundry.handle_for(mlp_model.id, None)
    bogus = tmp_path / "evil.pt"
    bogus.write_bytes(b"pickle")
    with pytest.raises(ValidationFailure):
        load_state(handle, bogus)


def test_safetensors_file_has_no_duplicate_storage(foundry, transformer_model):
    handle, record = foundry.handle_for(transformer_model.id, "latest")
    packed = flattened_state(handle.module)
    assert "lm_head.weight" in packed["aliases"]
    assert packed["aliases"]["lm_head.weight"] == "tok_emb.weight"
    assert len(packed["tensors"]) == len(set(map(id, packed["tensors"].values())))


# --------------------------------------------------------------------------------------------------
# Weight studio
# --------------------------------------------------------------------------------------------------


def _tensor_patch(foundry, model_id, tensor_name, operation, parameters):
    handle, record = foundry.handle_for(model_id, "latest")
    state = dict(handle.module.state_dict())
    tensor = state[tensor_name]
    import uuid

    return WeightPatch(
        patch_id=f"patch_{uuid.uuid4().hex[:10]}",
        base_checkpoint_sha256=record.sha256,
        architecture_hash=handle.architecture_hash,
        operation=operation,
        target_tensor=tensor_name,
        parameters=parameters,
        precondition=PatchPrecondition(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            base_checkpoint_sha256=record.sha256,
        ),
        created_by="test",
    )


def test_scale_patch_changes_values_and_reports_delta(foundry, mlp_model):
    patch = _tensor_patch(foundry, mlp_model.id, "layers.0.weight", PatchOperation.SCALE, {"factor": 0.5})
    result = foundry.apply_patch(mlp_model.id, patch)
    validation = result["validation"]
    assert validation["changed_elements"] > 0
    assert validation["norm_after"] < validation["norm_before"]
    assert validation["relative_norm_change"] < 0


def test_patch_on_tied_tensor_resolves_alias(foundry, transformer_model):
    patch = _tensor_patch(
        foundry, transformer_model.id, "lm_head.weight", PatchOperation.SCALE, {"factor": 0.9}
    )
    result = foundry.apply_patch(transformer_model.id, patch)
    assert result["validation"]["changed_elements"] > 0
    handle, _ = foundry.handle_for(transformer_model.id, "latest")
    # The tied embedding must have moved with it.
    assert handle.module.lm_head.weight.data_ptr() == handle.module.tok_emb.weight.data_ptr()


def test_low_rank_delta_is_bounded_by_alpha(foundry, mlp_model):
    patch = _tensor_patch(
        foundry,
        mlp_model.id,
        "layers.0.weight",
        PatchOperation.LOW_RANK_DELTA,
        {"rank": 2, "alpha": 1.0, "seed": 4, "std": 0.01},
    )
    result = foundry.apply_patch(mlp_model.id, patch)
    assert 0 < result["validation"]["max_abs_delta"] < 0.2


def test_patch_rejects_non_finite_result(foundry, mlp_model):
    patch = _tensor_patch(
        foundry, mlp_model.id, "layers.0.weight", PatchOperation.SCALE, {"factor": 1e39}
    )
    with pytest.raises(ValidationFailure) as excinfo:
        foundry.apply_patch(mlp_model.id, patch)
    assert "non-finite" in str(excinfo.value)


def test_patch_rejects_wrong_shape_precondition(foundry, mlp_model):
    patch = _tensor_patch(foundry, mlp_model.id, "layers.0.weight", PatchOperation.SCALE, {"factor": 1.0})
    broken = patch.model_copy(update={"precondition": patch.precondition.model_copy(update={"shape": (3, 3)})})
    with pytest.raises(ValidationFailure):
        foundry.apply_patch(mlp_model.id, broken)


def test_patch_rejects_unknown_base_checkpoint(foundry, mlp_model):
    patch = _tensor_patch(foundry, mlp_model.id, "layers.0.weight", PatchOperation.SCALE, {"factor": 1.0})
    broken = patch.model_copy(update={"base_checkpoint_sha256": "f" * 64})
    with pytest.raises(NotFoundError):
        foundry.apply_patch(mlp_model.id, broken)


def test_all_documented_operations_execute(foundry, mlp_model):
    operations = [
        (PatchOperation.SET_SCALAR, {"index": [0, 0], "value": 0.25}),
        (PatchOperation.SET_BLOCK, {"start": [0, 0], "block_shape": [2, 2], "mode": "zeros"}),
        (PatchOperation.INIT_SEEDED, {"scheme": "xavier", "seed": 1}),
        (PatchOperation.ADD_SCALED_DELTA, {"scale": 0.1, "std": 0.01, "seed": 2}),
        (PatchOperation.LOW_RANK_DELTA, {"rank": 1, "alpha": 2.0, "seed": 3}),
        (PatchOperation.NORMALIZE, {"axis": 1, "eps": 1e-6}),
        (PatchOperation.CLAMP, {"min": -0.5, "max": 0.5}),
        (PatchOperation.SCALE, {"factor": 1.01}),
        (PatchOperation.SPARSITY_MASK, {"keep_fraction": 0.5, "seed": 4}),
    ]
    for operation, parameters in operations:
        patch = _tensor_patch(foundry, mlp_model.id, "layers.0.weight", operation, parameters)
        result = foundry.apply_patch(mlp_model.id, patch)
        assert result["validation"]["finite"] is True


def test_bias_tensor_supports_scalar_but_not_low_rank(foundry, mlp_model):
    scalar = _tensor_patch(foundry, mlp_model.id, "layers.0.bias", PatchOperation.SET_SCALAR, {"index": [0], "value": 0.3})
    foundry.apply_patch(mlp_model.id, scalar)
    low_rank = _tensor_patch(
        foundry, mlp_model.id, "layers.0.bias", PatchOperation.LOW_RANK_DELTA, {"rank": 1, "alpha": 1.0, "seed": 1}
    )
    with pytest.raises(ValidationFailure):
        foundry.apply_patch(mlp_model.id, low_rank)


def test_tensor_stats_slice_and_histogram(foundry, mlp_model):
    handle, _ = foundry.handle_for(mlp_model.id, "latest")
    stats = {entry.name: entry for entry in summarize(handle.module)}
    assert stats["layers.0.weight"].finite is True
    assert stats["layers.0.weight"].numel == 12 * 8
    block = slice_tensor(handle.module, "layers.0.weight", offset=(0, 0), shape=(2, 2))
    assert len(block.values) == 4
    wide = slice_tensor(handle.module, "layers.0.weight", shape=(1000, 1000), max_elements=64)
    assert wide.truncated and len(wide.values) <= 64
    hist = histogram(handle.module, "layers.0.weight", bins=8)
    assert len(hist.counts) == 8
    with pytest.raises(ValidationFailure):
        slice_tensor(handle.module, "layers.0.weight", offset=(999, 0))
    with pytest.raises(ValidationFailure):
        get_tensor(handle.module, "../etc/passwd")


def test_rollback_moves_head_to_older_checkpoint(foundry, mlp_model):
    init = foundry.repo.list_checkpoints(mlp_model.id)[0]
    from hcs_foundry.schemas import TrainingConfig

    foundry.train(
        TrainingConfig(
            model_id=mlp_model.id, dataset_id="tiny-shapes", steps=3, batch_size=8,
            eval_every=3, checkpoint_every=3, max_seconds=60,
        )
    )
    newest = foundry.repo.latest_checkpoint(mlp_model.id)
    assert newest.id != init.id
    rolled = foundry.rollback(mlp_model.id, init.id)
    assert rolled["head"] == init.sha256
    assert foundry.repo.latest_checkpoint(mlp_model.id).id == init.id


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------


def test_training_reduces_loss(foundry, transformer_model):
    config = TrainingConfig(
        model_id=transformer_model.id,
        dataset_id="tiny-text",
        steps=24,
        batch_size=8,
        block_size=16,
        eval_every=12,
        checkpoint_every=24,
        max_seconds=120,
    )
    run = foundry.train(config)
    assert run.state.value == "completed"
    losses = [point.loss for point in run.metrics if point.val_loss is None]
    assert losses[-1] < losses[0]
    assert run.summary["tokens"] > 0
    assert run.provenance.dataset_sha256


def test_training_writes_checkpoint_and_can_resume(foundry, mlp_model):
    config = TrainingConfig(
        model_id=mlp_model.id,
        dataset_id="tiny-shapes",
        steps=6,
        batch_size=8,
        eval_every=6,
        checkpoint_every=6,
        max_seconds=60,
    )
    first = foundry.train(config)
    assert first.state.value == "completed"
    checkpoints = foundry.repo.list_checkpoints(mlp_model.id)
    assert len(checkpoints) >= 2
    second = foundry.train(config)
    assert second.provenance.base_checkpoint_sha256 is not None


def test_training_rejects_wrong_dataset(foundry, mlp_model):
    config = TrainingConfig(model_id=mlp_model.id, dataset_id="tiny-text", steps=2)
    with pytest.raises(ValidationFailure):
        foundry.train(config)


def test_dry_run_produces_gradients(foundry, mlp_model):
    from hcs_foundry.training import datasets as dataset_module

    handle, _ = foundry.handle_for(mlp_model.id, None)
    spec = foundry.repo.get_dataset("tiny-shapes")
    trainer = SupervisedTrainer(
        foundry.workspace,
        foundry.repo,
        handle,
        TrainingConfig(model_id=mlp_model.id, dataset_id="tiny-shapes", steps=1),
        dataset=spec,
    )
    loss = trainer.dry_run()
    assert np.isfinite(loss)
    assert trainer.dry_run_report["parameters_with_grad"] > 0
    assert trainer.dry_run_report["grad_norm"] > 0
    # dry_run must clean up after itself
    assert all(parameter.grad is None for parameter in handle.module.parameters())
    assert dataset_module.dataset_for_model("mlp") == "tiny-shapes"


def test_cancel_stops_training(foundry, mlp_model):
    control = TrainerControl()
    control.request_cancel()
    config = TrainingConfig(model_id=mlp_model.id, dataset_id="tiny-shapes", steps=10, max_seconds=60)

    handle, _ = foundry.handle_for(mlp_model.id, None)
    trainer = SupervisedTrainer(
        foundry.workspace,
        foundry.repo,
        handle,
        config,
        dataset=foundry.repo.get_dataset("tiny-shapes"),
        control=control,
    )
    run = trainer.run()
    assert run.state.value == "cancelled"


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------


def test_text_suite_reports_definitions_and_baselines(foundry, transformer_model):
    report = foundry.evaluate(transformer_model.id, suite="text-smoke")
    assert report.state.value == "completed"
    names = {metric.name for metric in report.metrics}
    assert {"val_loss", "perplexity", "next_token_accuracy"} <= names
    for metric in report.metrics:
        assert metric.definition
    assert report.provenance.evaluator_version == "1"
    assert report.checkpoint_sha256


def test_shapes_suite_on_mlp(foundry, mlp_model):
    report = foundry.evaluate(mlp_model.id, suite="shapes-smoke")
    assert report.state.value == "completed"
    accuracy = report.metric("val_accuracy")
    assert accuracy is not None and 0.0 <= accuracy <= 1.0


def test_suite_rejects_wrong_family(foundry, mlp_model):
    with pytest.raises(NotFoundError):
        foundry.evaluate(mlp_model.id, suite="text-smoke")


def test_evaluator_error_marks_run_invalid(monkeypatch, foundry, mlp_model):
    suite = get_suite("shapes-smoke")

    def boom(*args, **kwargs):
        raise RuntimeError("evaluator exploded")

    monkeypatch.setattr(suite, "run", boom)
    report = evaluate(
        foundry.handle_for(mlp_model.id, "latest")[0],
        suite_id="shapes-smoke",
        checkpoint_sha256="c" * 64,
    )
    assert report.state.value == "invalid"
    assert report.metrics == []
    assert any("evaluator error" in note for note in report.notes)


def test_weight_sanity_suite_available_for_all_families():
    assert {"mlp", "transformer-lm"} <= set(get_suite("weight-sanity").families)
    assert len(list_suites()) >= 3


# --------------------------------------------------------------------------------------------------
# Evolution
# --------------------------------------------------------------------------------------------------


def test_evolution_improves_and_builds_archive(foundry, mlp_model):
    config = EvolutionConfig(
        model_id=mlp_model.id, population=8, generations=3, sigma=0.05, lr=0.05, seed=7
    )
    result = foundry.evolve(config)
    assert result["best"]["fitness"] >= result["initial"]["fitness"]
    assert result["archive"]
    assert 0.0 < result["archive_coverage"] <= 1.0
    assert result["candidates_evaluated"] == 8 * 3
    assert "compute-matched" in result["claim_boundary"]
    assert result["checkpoint"]["sha256"]
    assert foundry.repo.list_archives(mlp_model.id)


def test_evolution_is_reproducible(foundry, mlp_model):
    config = EvolutionConfig(model_id=mlp_model.id, population=4, generations=1, seed=99)
    first = foundry.evolve(config)
    second = foundry.evolve(config)
    # Scores and parameters are reproducible; latency is a wall-clock measurement and is not.
    assert first["best"]["score"] == second["best"]["score"]
    assert first["best_parameters_sha256"] == second["best_parameters_sha256"]


# --------------------------------------------------------------------------------------------------
# Agent policy and isolation
# --------------------------------------------------------------------------------------------------


def test_policy_blocks_forbidden_operations(foundry, mlp_model):
    handle, record = foundry.handle_for(mlp_model.id, "latest")
    tensor = flattened_state(handle.module)["tensors"]["layers.0.weight"]
    patch = build_patch(
        model_id=mlp_model.id,
        architecture_hash=handle.architecture_hash,
        base_checkpoint_sha256=record.sha256,
        tensor_name="layers.0.weight",
        shape=tuple(tensor.shape),
        dtype="float32",
        operation=PatchOperation.SET_TRAINABLE,
        parameters={"trainable": False},
        created_by="agent:test",
    )
    from hcs_foundry.agents import build_proposal

    proposal = build_proposal(
        agent_id="agent:test",
        goal="freeze",
        hypothesis="freezing stabilises",
        model_id=mlp_model.id,
        base_checkpoint_sha256=record.sha256,
        patches=[patch],
    )
    with pytest.raises(SecurityViolation):
        AgentPolicy().check(proposal)


def test_proposal_flow_requires_approval_then_verifies(foundry, mlp_model):
    proposal = foundry.propose_patch(
        mlp_model.id,
        agent_id="agent:test",
        goal="tiny perturbation",
        hypothesis="small low-rank change should not destroy accuracy",
        tensor="layers.0.weight",
        operation=PatchOperation.LOW_RANK_DELTA,
        parameters={"rank": 1, "alpha": 1.0, "seed": 3, "std": 0.001},
    )
    assert proposal["approval"] == "pending"
    applied = foundry.approve_proposal(proposal["proposal_id"])
    assert applied["applied"][0]["validation"]["finite"] is True
    verification = foundry.verify_proposal(proposal["proposal_id"], suite="weight-sanity")
    assert verification["state"] == "completed"


def test_runtime_modules_avoid_dangerous_primitives():
    """No eval/exec/pickle loading and no shell bridge in the runtime package."""
    import re

    package_root = Path(__file__).resolve().parents[1] / "hcs_foundry"
    patterns = {
        "bare eval(": re.compile(r"(?<![\w.])eval\("),
        "bare exec(": re.compile(r"(?<![\w.])exec\("),
        "pickle.load": re.compile(r"pickle\.load"),
        "os.system(": re.compile(r"os\.system\("),
        "shell=True": re.compile(r"shell=True"),
        "subprocess with shell": re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True"),
    }
    offenders = []
    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Strip strings/comments naively; the goal is catching real calls, not perfect AST analysis.
        for name, pattern in patterns.items():
            if pattern.search(text):
                offenders.append(f"{path.name}: {name}")
    assert offenders == []
