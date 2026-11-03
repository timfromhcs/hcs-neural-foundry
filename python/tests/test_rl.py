"""Tests for the reinforcement-learning slice: reward integrity, GRPO maths, isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hcs_foundry.errors import SecurityViolation
from hcs_foundry.rl import (
    LinearSoftmaxPolicy,
    RuleBandit,
    default_fixture_path,
    ensure_environment,
    evaluate_policy,
    group_advantages,
    load_policy,
    run_grpo,
    verify_policy_isolated,
)
from hcs_foundry.schemas import RlConfig

# --------------------------------------------------------------------------------------------------
# Environment and reward
# --------------------------------------------------------------------------------------------------


def test_bandit_is_deterministic_and_reward_is_exact():
    env = RuleBandit.create(seed=5)
    contexts = env.contexts(32, seed=1)
    first = env.correct_actions(contexts)
    second = env.correct_actions(contexts)
    assert np.array_equal(first, second)

    rewards = env.rewards(contexts, first)
    assert rewards.min() == 1.0
    wrong = (first + 1) % env.n_actions
    assert env.rewards(contexts, wrong).max() == 0.0


def test_bandit_description_matches_reality():
    env = RuleBandit.create(context_dim=5, n_actions=3, seed=2)
    described = env.describe()
    assert described["chance_accuracy"] == pytest.approx(1 / 3, rel=1e-6)
    # A policy that knows the rule scores 1.0; a uniform one scores near chance.
    oracle = LinearSoftmaxPolicy(env.weights * 10.0, env.context_dim, env.n_actions)
    assert evaluate_policy(env, oracle, seed=0, contexts=500)["accuracy"] == 1.0


def test_fixture_is_sealed_and_tampering_is_detected(tmp_path):
    env = RuleBandit.create(seed=1)
    path = tmp_path / "rule-bandit.json"
    sha = env.save(path)
    assert path.exists()

    # Re-loading with the expected hash works.
    reloaded = RuleBandit.load(path, expected_sha256=sha)
    assert reloaded.rule_sha256 == env.rule_sha256

    # Editing the file is refused when the hash is pinned.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weights"][0][0] += 5.0
    path.chmod(0o644)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SecurityViolation):
        RuleBandit.load(path, expected_sha256=sha)
    # And the internal rule hash no longer matches its own record.
    with pytest.raises(SecurityViolation):
        RuleBandit.load(path)


def test_ensure_environment_reuses_the_existing_fixture(foundry):
    first, path = ensure_environment(foundry.workspace, seed=9)
    second, path2 = ensure_environment(foundry.workspace, seed=9)
    assert path == path2
    assert first.rule_sha256 == second.rule_sha256
    assert path.exists()
    # A different seed would otherwise rewrite it; the sealed file must win.
    third, _ = ensure_environment(foundry.workspace, seed=123)
    assert third.rule_sha256 == first.rule_sha256


# --------------------------------------------------------------------------------------------------
# Policy and GRPO maths
# --------------------------------------------------------------------------------------------------


def test_policy_round_trip_and_normalisation():
    policy = LinearSoftmaxPolicy.initialize(4, 3, seed=2)
    probabilities = policy.probabilities(np.ones((5, 4), dtype=np.float32))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    restored = LinearSoftmaxPolicy.from_dict(policy.to_dict())
    assert np.allclose(restored.weights, policy.weights)


def test_group_advantages_are_standardised():
    rewards = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    advantages = group_advantages(rewards)
    assert advantages.shape == rewards.shape
    assert np.allclose(advantages.mean(axis=1), 0.0, atol=1e-6)
    # A group with identical rewards carries no signal: advantages must be exactly zero.
    assert np.all(advantages[1] == 0.0)


def test_sampling_is_reproducible_for_a_seed():
    policy = LinearSoftmaxPolicy.initialize(6, 4, seed=3)
    contexts = np.ones((8, 6), dtype=np.float32)
    first = policy.sample(contexts, 5, seed=17)
    second = policy.sample(contexts, 5, seed=17)
    assert np.array_equal(first, second)
    assert first.shape == (8, 5)


@pytest.mark.heavy
def test_grpo_improves_held_out_accuracy():
    env = RuleBandit.create(seed=0)
    result = run_grpo(
        env,
        config=RlConfig(steps=80, group_size=8, batch_contexts=16, lr=0.2, kl_beta=0.01, seed=3),
    )
    held_out = result.report["held_out"]
    assert held_out["policy_accuracy"] > held_out["reference_accuracy"] + 0.2
    assert held_out["policy_accuracy"] > held_out["chance_accuracy"]
    assert result.report["improved"] is True
    assert result.report["history"][-1]["kl"] > 0
    assert "not language-model post-training" in result.report["claim_boundary"]


@pytest.mark.heavy
def test_kl_penalty_trades_reward_for_proximity():
    """A large KL penalty must keep the policy closer to the reference, and cost reward.

    This asserts the *mechanism* (the penalty works) rather than a quality claim. The reported
    ``kl_exact`` is computed in closed form so the assertion is not swamped by the single-sample
    estimator used inside the loss.
    """
    env = RuleBandit.create(seed=1)
    options = {"steps": 40, "group_size": 8, "batch_contexts": 16, "lr": 0.05, "seed": 4}
    free = run_grpo(env, config=RlConfig(kl_beta=0.02, **options))
    tethered = run_grpo(env, config=RlConfig(kl_beta=10.0, **options))

    assert tethered.report["history"][-1]["kl_exact"] < free.report["history"][-1]["kl_exact"]
    assert tethered.report["history"][-1]["mean_reward"] < free.report["history"][-1]["mean_reward"]


# --------------------------------------------------------------------------------------------------
# Orchestration and isolation
# --------------------------------------------------------------------------------------------------


@pytest.mark.heavy
def test_rl_run_records_provenance_and_isolated_verification(foundry):
    run = foundry.run_rl(RlConfig(steps=60, group_size=8, lr=0.2, kl_beta=0.01, seed=3))
    assert run.state.value == "completed"
    assert run.kind == "reinforcement"
    assert run.summary["env"]["env_id"] == "rule-bandit"
    assert run.summary["isolated_verification"]["state"] == "completed"
    assert run.summary["isolated_verification"]["policy_accuracy"] > 0.5
    assert run.provenance.dataset_sha256 == run.summary["env"]["rule_sha256"]

    policy_path = Path(run.summary["policy_path"])
    assert policy_path.exists()
    policy = load_policy(policy_path)
    assert policy.n_actions == 4

    stored = foundry.repo.get_run(run.id)
    assert stored.summary["policy_sha256"] == run.summary["policy_sha256"]


def test_verifier_rejects_a_tampered_fixture(foundry, tmp_path):
    env, path = ensure_environment(foundry.workspace, seed=2)
    policy = LinearSoftmaxPolicy.initialize(env.context_dim, env.n_actions, seed=1)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema": "hcs-foundry.policy/1",
                "run_id": "test",
                "env_id": "rule-bandit",
                "rule_sha256": env.rule_sha256,
                "policy": policy.to_dict(),
            }
        ),
        encoding="utf-8",
    )

    clean = verify_policy_isolated(
        workspace_root=foundry.workspace.root,
        policy_path=policy_path,
        fixture_path=path,
        contexts=64,
        seed=0,
    )
    assert clean["state"] == "completed"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weights"][0][0] += 1.0
    path.chmod(0o644)
    path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = verify_policy_isolated(
        workspace_root=foundry.workspace.root,
        policy_path=policy_path,
        fixture_path=path,
        contexts=64,
        seed=0,
    )
    assert tampered["state"] == "invalid"


def test_default_fixture_path_is_inside_the_workspace(foundry):
    path = default_fixture_path(foundry.workspace)
    assert foundry.workspace.root in path.parents
