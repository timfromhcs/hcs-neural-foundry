"""Agent control plane: proposals, policy and isolated verification.

The separation that matters (ADR-008): an agent can only *propose* patches. The service validates and
applies them, and the evaluator runs in a separate process with its own model copy. The proposer never
touches the score it is judged by.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from .errors import SecurityViolation, ValidationFailure
from .hashing import new_id
from .schemas import AgentProposal, PatchOperation, PatchPrecondition, WeightPatch, validate_operation_params


class AutonomyLevel(IntEnum):
    EXPLAIN = 0
    DRAFT = 1
    APPROVED_RUN = 2
    BOUNDED_AUTO = 3
    CAMPAIGN = 4


DEFAULT_LEVEL = AutonomyLevel.DRAFT


@dataclass
class AgentPolicy:
    level: AutonomyLevel = DEFAULT_LEVEL
    max_patches_per_proposal: int = 8
    max_elements_per_patch: int = 2_000_000
    allowed_operations: frozenset[PatchOperation] = field(
        default_factory=lambda: frozenset(
            {
                PatchOperation.SCALE,
                PatchOperation.CLAMP,
                PatchOperation.LOW_RANK_DELTA,
                PatchOperation.SET_BLOCK,
                PatchOperation.NORMALIZE,
                PatchOperation.SPARSITY_MASK,
                PatchOperation.INIT_SEEDED,
            }
        )
    )
    forbidden_operations: frozenset[PatchOperation] = field(
        default_factory=lambda: frozenset({PatchOperation.SET_TRAINABLE})
    )
    allow_network: bool = False
    require_approval_above: AutonomyLevel = AutonomyLevel.APPROVED_RUN
    max_seconds_per_run: int = 300
    max_export_bytes: int = 512 * 1024 * 1024

    def check(self, proposal: AgentProposal) -> None:
        if self.level is AutonomyLevel.EXPLAIN:
            raise SecurityViolation(
                "agent autonomy level 0 (explain) forbids submitting proposals",
                hint="raise the level explicitly in configuration",
            )
        if len(proposal.patches) > self.max_patches_per_proposal:
            raise SecurityViolation(
                f"proposal has {len(proposal.patches)} patches, limit is {self.max_patches_per_proposal}"
            )
        for patch in proposal.patches:
            if patch.operation in self.forbidden_operations:
                raise SecurityViolation(f"operation '{patch.operation.value}' is forbidden by policy")
            if patch.operation not in self.allowed_operations:
                raise SecurityViolation(f"operation '{patch.operation.value}' is not in the allowlist")
            elements = 1
            for dim in patch.precondition.shape:
                elements *= max(1, dim)
            if elements > self.max_elements_per_patch:
                raise SecurityViolation(
                    f"patch targets {elements:,} elements, limit is {self.max_elements_per_patch:,}"
                )

    def describe(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "level_name": self.level.name.lower(),
            "allowed_operations": sorted(op.value for op in self.allowed_operations),
            "forbidden_operations": sorted(op.value for op in self.forbidden_operations),
            "max_patches_per_proposal": self.max_patches_per_proposal,
            "max_elements_per_patch": self.max_elements_per_patch,
            "allow_network": self.allow_network,
            "requires_approval": self.level >= self.require_approval_above,
            "max_seconds_per_run": self.max_seconds_per_run,
        }


def build_patch(
    *,
    model_id: str,
    architecture_hash: str,
    base_checkpoint_sha256: str,
    tensor_name: str,
    shape: tuple[int, ...],
    dtype: str,
    operation: PatchOperation,
    parameters: dict[str, Any],
    created_by: str,
    note: str = "",
) -> WeightPatch:
    """Construct a validated patch. Raises on anything malformed — before a tensor is touched."""
    params = validate_operation_params(operation, parameters)
    return WeightPatch(
        patch_id=new_id("patch"),
        base_checkpoint_sha256=base_checkpoint_sha256,
        architecture_hash=architecture_hash,
        operation=operation,
        target_tensor=tensor_name,
        parameters=params,
        precondition=PatchPrecondition(
            shape=tuple(shape), dtype=dtype, base_checkpoint_sha256=base_checkpoint_sha256
        ),
        created_by=created_by,
        note=note,
    )


def build_proposal(
    *,
    agent_id: str,
    goal: str,
    hypothesis: str,
    model_id: str,
    base_checkpoint_sha256: str,
    patches: list[WeightPatch],
    validation_plan: list[str] | None = None,
    expected_effect: str = "",
) -> AgentProposal:
    return AgentProposal(
        proposal_id=new_id("prop"),
        agent_id=agent_id,
        goal=goal,
        hypothesis=hypothesis,
        model_id=model_id,
        base_checkpoint_sha256=base_checkpoint_sha256,
        patches=patches,
        validation_plan=validation_plan
        or ["weight-sanity suite", "task suite before/after comparison", "independent evaluator"],
        expected_effect=expected_effect,
        estimated_cost={"patches": len(patches)},
    )


def verify_isolated(
    *,
    workspace_root: str | Path,
    model_config: dict[str, Any],
    checkpoint_path: str | Path,
    suite_id: str,
    seed: int = 424242,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run the evaluator in a child process with no access to the caller's objects.

    The child gets a JSON spec on argv, prints a JSON report on stdout, and cannot modify the parent's
    database. Network access is not granted to the child.
    """
    payload = {
        "model_config": model_config,
        "checkpoint_path": str(checkpoint_path),
        "suite_id": suite_id,
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
            [sys.executable, "-m", "hcs_foundry.agent_verify", json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
            cwd=str(workspace_root),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationFailure(
            f"isolated verification exceeded {timeout}s", hint="reduce batches or raise the timeout"
        ) from exc

    if completed.returncode != 0:
        return {
            "state": "invalid",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
            "stdout": completed.stdout[-2000:],
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValidationFailure(f"verifier returned unparsable output: {exc}") from exc
