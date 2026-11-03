"""Shared model plumbing: the handle, exact parameter counting and the family protocol.

The counting rule is deliberate (ADR-003): the number shown to the user is always enumerated from
the live module, never from a formula. ``predict_parameters`` exists so the budget solver can prune
a large search space cheaply, and tests assert that both agree for every family.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..errors import MissingDependencyError
from ..schemas import (
    Maturity,
    ParameterEntry,
    ParameterReport,
    ResourceEstimate,
)


def torch_module():  # noqa: ANN201 - typed as Any on purpose to keep schemas torch-free
    """Import torch lazily and fail with an actionable message if the extra is missing."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError(
            "PyTorch is required for this operation but is not installed",
            hint='install the engine extra: uv pip install -e ".[torch]"',
            details={"import_error": str(exc)},
        ) from exc
    return torch


@dataclass
class ModelHandle:
    """A built model: config, live torch module, exact report and architecture hash."""

    config: Any
    module: Any
    report: ParameterReport
    architecture_hash: str
    family: str
    notes: list[str] = field(default_factory=list)

    @property
    def parameter_count(self) -> int:
        return self.report.total

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state, strict=True)

    def device(self) -> str:
        return str(next(self.module.parameters()).device)

    def trainable_parameters(self) -> list[Any]:
        return [p for p in self.module.parameters() if p.requires_grad]


def count_parameters(module: Any, *, architecture_hash_value: str = "") -> ParameterReport:
    """Enumerate parameters, de-duplicating aliased (tied) tensors by storage pointer."""
    seen: dict[int, str] = {}
    entries: list[ParameterEntry] = []
    total = 0
    trainable = 0
    shared = 0

    for name, tensor in module.named_parameters(remove_duplicate=False):
        key = int(tensor.data_ptr())
        if key in seen and tensor.numel() > 0:
            shared += 1
            entries.append(
                ParameterEntry(
                    name=name,
                    shape=tuple(tensor.shape),
                    numel=int(tensor.numel()),
                    trainable=bool(tensor.requires_grad),
                    tied_to=seen[key],
                    dtype=str(tensor.dtype).replace("torch.", ""),
                )
            )
            continue
        seen[key] = name
        numel = int(tensor.numel())
        total += numel
        if tensor.requires_grad:
            trainable += numel
        entries.append(
            ParameterEntry(
                name=name,
                shape=tuple(tensor.shape),
                numel=numel,
                trainable=bool(tensor.requires_grad),
                dtype=str(tensor.dtype).replace("torch.", ""),
            )
        )

    return ParameterReport(
        total=total,
        trainable=trainable,
        unique_tensors=len(seen),
        shared_tensors=shared,
        entries=entries,
        architecture_hash=architecture_hash_value,
        exact=True,
    )


def seed_everything(seed: int) -> None:
    torch = torch_module()
    torch.manual_seed(seed)
    with contextlib.suppress(Exception):  # pragma: no cover - older torch
        torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: str = "error"


@dataclass
class SmokeTestReport:
    passed: bool
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


@runtime_checkable
class ModelFamilyAdapter(Protocol):
    """Contract every model family must satisfy (AGENTS.md §5.5)."""

    family_id: str
    display_name: str
    maturity: Maturity

    def validate_config(self, config: dict[str, Any]) -> list[ValidationIssue]: ...

    def build(self, config: Any) -> ModelHandle: ...

    def predict_parameters(self, config: Any) -> int: ...

    def estimate_resources(self, config: Any, **kwargs: Any) -> ResourceEstimate: ...

    def smoke_test(self, handle: ModelHandle) -> SmokeTestReport: ...

    def sample_batch(self, handle: ModelHandle, *, batch_size: int, block_size: int) -> dict[str, Any]: ...
