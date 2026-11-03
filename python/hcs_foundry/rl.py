"""Reinforcement learning with verifiable rewards — deliberately small and honest.

What this module is
-------------------
A working GRPO implementation (group-relative advantages, KL to a frozen reference, clipped ratio)
driving a **toy verifiable-reward environment**: a contextual bandit whose reward comes from a sealed
rule that the policy cannot read.

What this module is not
-----------------------
An LLM post-training pipeline. Applying this to the transformer needs token-level rollouts, a reward
model or verifier over completions, and a much larger compute budget. That is declared unsupported in
``docs/STATUS.md``; nothing here is labelled as language-model RL.

Why the environment is a bandit
-------------------------------
Because it makes the algorithm falsifiable: the optimal policy is computable, the reward is exact, and
a short run either improves held-out success or it does not. A language-model demo would hide the same
question behind sampling noise and a huge compute bill.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .errors import NotFoundError, SecurityViolation, ValidationFailure
from .hashing import atomic_write, hash_file, sha256_hex
from .schemas import Provenance, RlConfig, RunRecord, RunState
from .storage import Repository, Workspace

# --------------------------------------------------------------------------------------------------
# Verifiable reward environment
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleBandit:
    """Contextual bandit with a sealed deterministic rule.

    ``weights`` maps context features to logits over actions; the correct action is the argmax. The
    policy never receives ``weights`` through the environment API — only contexts and scalar rewards.
    """

    weights: np.ndarray  # (context_dim, n_actions)
    context_dim: int
    n_actions: int
    seed: int

    @property
    def rule_sha256(self) -> str:
        return sha256_hex(self.weights.astype("<f4").tobytes())

    @classmethod
    def create(cls, *, context_dim: int = 6, n_actions: int = 4, seed: int = 0) -> RuleBandit:
        if context_dim < 2 or n_actions < 2:
            raise ValidationFailure("the bandit needs at least 2 context features and 2 actions")
        rng = np.random.default_rng(seed * 7919 + 13)
        weights = rng.standard_normal((context_dim, n_actions)).astype(np.float32)
        return cls(weights=weights, context_dim=context_dim, n_actions=n_actions, seed=seed)

    # -- fixture persistence -----------------------------------------------------------------
    def save(self, path: Path) -> str:
        payload = {
            "schema": "hcs-foundry.rule-bandit/1",
            "context_dim": self.context_dim,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "weights": self.weights.tolist(),
            "rule_sha256": self.rule_sha256,
        }
        with atomic_write(path) as handle:
            handle.write(json.dumps(payload, indent=2).encode("utf-8"))
        # Make the sealed rule read-only for the current user where the platform allows it.
        with contextlib.suppress(OSError):
            path.chmod(0o444)
        return hash_file(path)

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str | None = None) -> RuleBandit:
        if not Path(path).exists():
            raise NotFoundError(f"reward fixture {path} does not exist", hint="create the environment first")
        if expected_sha256 and hash_file(path) != expected_sha256:
            raise SecurityViolation(
                "the sealed reward fixture changed since the run started",
                hint="recreate the environment; a modified fixture invalidates every score",
                details={"path": str(path)},
            )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = np.asarray(payload["weights"], dtype=np.float32)
        env = cls(
            weights=weights,
            context_dim=int(payload["context_dim"]),
            n_actions=int(payload["n_actions"]),
            seed=int(payload["seed"]),
        )
        if payload.get("rule_sha256") and payload["rule_sha256"] != env.rule_sha256:
            raise SecurityViolation("fixture contents do not match its recorded rule hash")
        return env

    # -- sampling and reward -----------------------------------------------------------------
    def contexts(self, count: int, *, split: str = "train", seed: int = 0) -> np.ndarray:
        offset = 0 if split == "train" else 10_000_019
        rng = np.random.default_rng(self.seed * 104_729 + seed + offset)
        return rng.standard_normal((count, self.context_dim)).astype(np.float32)

    def correct_actions(self, contexts: np.ndarray) -> np.ndarray:
        return np.argmax(contexts @ self.weights, axis=1).astype(np.int64)

    def rewards(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """1.0 for the rule-consistent action, 0.0 otherwise. No partial credit, no shaping."""
        return (actions == self.correct_actions(contexts)).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "env_id": "rule-bandit",
            "context_dim": self.context_dim,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "rule_sha256": self.rule_sha256,
            "reward": "1 if action == argmax(context @ sealed_weights) else 0",
            "optimal_accuracy": 1.0,
            "chance_accuracy": round(1.0 / self.n_actions, 6),
        }


# --------------------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------------------


def torch_module():  # noqa: ANN201
    from .models.base import torch_module as load

    return load()


@dataclass
class LinearSoftmaxPolicy:
    """Linear softmax policy over context features — small enough to inspect, real enough to learn."""

    weights: np.ndarray  # (context_dim, n_actions)
    context_dim: int
    n_actions: int

    @classmethod
    def initialize(cls, context_dim: int, n_actions: int, *, seed: int = 0) -> LinearSoftmaxPolicy:
        rng = np.random.default_rng(seed + 991)
        weights = (rng.standard_normal((context_dim, n_actions)) * 0.01).astype(np.float32)
        return cls(weights=weights, context_dim=context_dim, n_actions=n_actions)

    def logits(self, contexts: np.ndarray) -> np.ndarray:
        return contexts @ self.weights

    def probabilities(self, contexts: np.ndarray) -> np.ndarray:
        logits = self.logits(contexts)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def sample(self, contexts: np.ndarray, group_size: int, *, seed: int) -> np.ndarray:
        """Return actions shaped ``(batch, group_size)``."""
        rng = np.random.default_rng(seed)
        probabilities = self.probabilities(contexts)
        actions = np.empty((contexts.shape[0], group_size), dtype=np.int64)
        for row, probs in enumerate(probabilities):
            actions[row] = rng.choice(self.n_actions, size=group_size, p=probs)
        return actions

    def greedy(self, contexts: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(contexts), axis=1).astype(np.int64)

    def accuracy(self, contexts: np.ndarray, correct: np.ndarray) -> float:
        return float((self.greedy(contexts) == correct).mean()) if len(correct) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear-softmax",
            "context_dim": self.context_dim,
            "n_actions": self.n_actions,
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinearSoftmaxPolicy:
        weights = np.asarray(payload["weights"], dtype=np.float32)
        return cls(weights=weights, context_dim=int(payload["context_dim"]), n_actions=int(payload["n_actions"]))

    def copy(self) -> LinearSoftmaxPolicy:
        return LinearSoftmaxPolicy(self.weights.copy(), self.context_dim, self.n_actions)


class Policy(Protocol):
    """Anything that can score actions in context. Kept as a protocol to leave room for a real LLM."""

    n_actions: int

    def probabilities(self, contexts: np.ndarray) -> np.ndarray: ...

    def sample(self, contexts: np.ndarray, group_size: int, *, seed: int) -> np.ndarray: ...


# --------------------------------------------------------------------------------------------------
# GRPO
# --------------------------------------------------------------------------------------------------


def group_advantages(rewards: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Group-relative advantages: zero-mean, unit-scale within each context's sample group."""
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / np.maximum(std, eps)


def _policy_logprob_tensor(policy_weights, contexts, actions):  # noqa: ANN001, ANN202
    torch = torch_module()
    logits = contexts @ policy_weights
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)


def exact_kl(policy: LinearSoftmaxPolicy, reference: LinearSoftmaxPolicy, contexts: np.ndarray) -> float:
    """Closed-form KL(pi || pi_ref) in nats, averaged over contexts.

    The training loss uses the usual single-sample estimator (that is what GRPO does); this function
    exists so the *reported* divergence is not dominated by sampling noise.
    """
    p = policy.probabilities(contexts)
    q = reference.probabilities(contexts)
    return float(
        np.mean(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12))), axis=1))
    )


@dataclass
class GrpoResult:
    policy: LinearSoftmaxPolicy
    report: dict[str, Any] = field(default_factory=dict)


def run_grpo(
    env: RuleBandit,
    *,
    config: RlConfig,
    progress: Any | None = None,
) -> GrpoResult:
    """Run GRPO on the toy environment. Returns the trained policy and a JSON-safe report."""
    torch = torch_module()
    torch.manual_seed(config.seed)

    reference = LinearSoftmaxPolicy.initialize(env.context_dim, env.n_actions, seed=config.seed)
    policy = reference.copy()

    weights = torch.tensor(policy.weights, requires_grad=True, dtype=torch.float32)
    ref_weights = torch.tensor(reference.weights, dtype=torch.float32)
    optimizer = torch.optim.Adam([weights], lr=config.lr)

    history: list[dict[str, Any]] = []
    examples = 0
    for step in range(config.steps):
        contexts_np = env.contexts(config.batch_contexts, split="train", seed=config.seed * 10_007 + step)
        actions_np = policy.sample(contexts_np, config.group_size, seed=config.seed + step)
        rewards = np.stack(
            [env.rewards(contexts_np, actions_np[:, k]) for k in range(config.group_size)], axis=1
        )
        advantages = group_advantages(rewards)

        # Flatten (batch, group) into one flat batch of sampled (context, action) pairs.
        flat_contexts = np.repeat(contexts_np, config.group_size, axis=0)
        flat_actions = actions_np.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        contexts = torch.from_numpy(flat_contexts)
        actions = torch.from_numpy(flat_actions)
        advantage_tensor = torch.from_numpy(flat_advantages)

        log_probs = _policy_logprob_tensor(weights, contexts, actions)
        with torch.no_grad():
            ref_log_probs = _policy_logprob_tensor(ref_weights, contexts, actions)
        # Single-sample KL estimator: E[log π(a|s) - log π_ref(a|s)] over actions drawn from π.
        kl = (log_probs - ref_log_probs).mean()
        loss = -(advantage_tensor * log_probs).mean() + config.kl_beta * kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weights], 1.0)
        optimizer.step()

        examples += config.batch_contexts
        with torch.no_grad():
            policy.weights = weights.detach().numpy().astype(np.float32)
        if step % max(1, config.steps // 10) == 0 or step == config.steps - 1:
            entry = {
                "step": step,
                "loss": round(float(loss.detach()), 6),
                "kl": round(float(kl.detach()), 6),
                "kl_exact": round(exact_kl(policy, reference, contexts_np), 6),
                "mean_reward": round(float(rewards.mean()), 6),
                "advantage_abs_mean": round(float(np.abs(advantages).mean()), 6),
                "examples": examples,
            }
            history.append(entry)
            if progress:
                progress(entry)
        if examples >= config.max_contexts:
            break

    evaluation = evaluate_policy(env, policy, seed=config.seed + 1, contexts=config.eval_contexts)
    reference_evaluation = evaluate_policy(env, reference, seed=config.seed + 1, contexts=config.eval_contexts)
    report = {
        "algorithm": "grpo",
        "env": env.describe(),
        "config": config.model_dump(mode="json"),
        "history": history,
        "examples": examples,
        "held_out": {
            "policy_accuracy": evaluation["accuracy"],
            "reference_accuracy": reference_evaluation["accuracy"],
            "chance_accuracy": round(1.0 / env.n_actions, 6),
            "contexts": config.eval_contexts,
            "seed": config.seed + 1,
        },
        "improved": evaluation["accuracy"] > reference_evaluation["accuracy"],
        "claim_boundary": (
            "Toy contextual bandit with an exact verifiable reward. This validates the GRPO "
            "implementation, not language-model post-training. No DPO/PPO/RLHF claim is made."
        ),
    }
    return GrpoResult(policy=policy, report=report)


def evaluate_policy(env: RuleBandit, policy: Policy, *, seed: int, contexts: int) -> dict[str, Any]:
    batch = env.contexts(contexts, split="val", seed=seed)
    correct = env.correct_actions(batch)
    probabilities = policy.probabilities(batch)
    actions = np.argmax(probabilities, axis=1)
    accuracy = float((actions == correct).mean()) if contexts else 0.0
    nll = float(-np.log(np.maximum(probabilities[np.arange(len(correct)), correct], 1e-9)).mean()) if contexts else float("nan")
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-9))).sum(axis=1).mean()) if contexts else 0.0
    return {
        "accuracy": accuracy,
        "nll": nll,
        "entropy": entropy,
        "contexts": contexts,
    }


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------


def default_fixture_path(workspace: Workspace, env_id: str = "rule-bandit") -> Path:
    return workspace.datasets / f"{env_id}.json"


def ensure_environment(
    workspace: Workspace,
    *,
    context_dim: int = 6,
    n_actions: int = 4,
    seed: int = 0,
) -> tuple[RuleBandit, Path]:
    """Create the sealed fixture once; reuse it afterwards and never overwrite it silently."""
    path = default_fixture_path(workspace)
    if path.exists():
        return RuleBandit.load(path), path
    env = RuleBandit.create(context_dim=context_dim, n_actions=n_actions, seed=seed)
    env.save(path)
    return env, path


def policy_to_run(
    *,
    workspace: Workspace,
    repository: Repository,
    env: RuleBandit,
    config: RlConfig,
    run_id: str,
    progress: Any | None = None,
) -> RunRecord:
    """Train, persist the policy artifact, verify it in a separate process and record the run."""
    from .hashing import utcnow

    run = RunRecord(
        id=run_id,
        kind="reinforcement",
        model_id=f"env:{config.env_id}",
        state=RunState.RUNNING,
        config=config.model_dump(mode="json"),
        provenance=Provenance(
            engine_version="0.1.0",
            seeds={"rl": config.seed},
            dataset_sha256=env.rule_sha256,
            notes=[f"env={config.env_id}", "toy verifiable reward; not language-model RL"],
        ),
        started_at=utcnow(),
    )
    repository.add_run(run)

    result = run_grpo(env, config=config, progress=progress)
    policy_path = workspace.artifacts / f"{env.rule_sha256[:2]}" / f"policy-{run_id}.json"
    payload = {
        "schema": "hcs-foundry.policy/1",
        "run_id": run_id,
        "env_id": config.env_id,
        "rule_sha256": env.rule_sha256,
        "policy": result.policy.to_dict(),
    }
    with atomic_write(policy_path) as handle:
        handle.write(json.dumps(payload, indent=2).encode("utf-8"))
    policy_sha = hash_file(policy_path)

    verification = verify_policy_isolated(
        workspace_root=workspace.root,
        policy_path=policy_path,
        fixture_path=default_fixture_path(workspace),
        contexts=config.eval_contexts,
        seed=config.seed + 1,
    )

    summary = {
        **result.report,
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha,
        "isolated_verification": verification,
        "artifact": policy_sha,
    }
    repository.update_run(run.id, state=RunState.COMPLETED, summary=summary, finished=True)
    repository.log_event(
        "info",
        f"rl:{run.id}",
        f"grpo finished: held-out accuracy {result.report['held_out']['policy_accuracy']:.3f}",
        {"env": config.env_id, "policy_sha256": policy_sha},
    )
    return repository.get_run(run.id)


def verify_policy_isolated(
    *,
    workspace_root: Path,
    policy_path: Path,
    fixture_path: Path,
    contexts: int,
    seed: int,
    timeout: int = 120,
) -> dict[str, Any]:
    """Score a saved policy in a child process that cannot see the training objects."""
    import os
    import subprocess
    import sys

    spec = {
        "policy_path": str(policy_path),
        "fixture_path": str(fixture_path),
        "contexts": contexts,
        "seed": seed,
    }
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "HCS_FOUNDRY_HOME": str(workspace_root),
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "hcs_foundry.rl_verify", json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
            cwd=str(workspace_root),
        )
    except subprocess.TimeoutExpired:
        return {"state": "invalid", "error": f"verifier exceeded {timeout}s"}
    if completed.returncode != 0:
        return {
            "state": "invalid",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:],
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"state": "invalid", "error": "unparsable verifier output"}
    payload["state"] = "completed"
    return payload


def load_policy(path: Path) -> LinearSoftmaxPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "hcs-foundry.policy/1":
        raise ValidationFailure("not a Foundry policy artifact")
    return LinearSoftmaxPolicy.from_dict(payload["policy"])
