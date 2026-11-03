"""Parameter budget solver.

Formulas are used to *prune* the discrete search space; the final reports are confirmed by building
the module and enumerating ``numel`` (ADR-003). When an exact count is impossible the solver says so
and returns the closest feasible designs instead of pretending.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import ValidationFailure
from ..hardware import estimate_resources
from ..schemas import (
    BudgetPolicy,
    FamilyId,
    MlpConfig,
    ResourceEstimate,
    TransformerConfig,
)
from .mlp import predict_parameters as predict_mlp
from .transformer_lm import predict_parameters as predict_transformer

MAX_CANDIDATES = 4000


class SolverConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: FamilyId = FamilyId.TRANSFORMER_LM
    # transformer knobs
    d_model_min: int = Field(32, ge=8, le=8192)
    d_model_max: int = Field(512, ge=8, le=8192)
    d_model_step: int = Field(32, ge=1, le=1024)
    n_layers_min: int = Field(1, ge=1, le=128)
    n_layers_max: int = Field(8, ge=1, le=128)
    n_heads_options: tuple[int, ...] = (2, 4, 8)
    vocab_size: int = Field(64, ge=2, le=1_000_000)
    context_length: int = Field(64, ge=8, le=131072)
    tie_embeddings: bool = True
    ffn_multipliers: tuple[float, ...] = (2.0, 2.667, 4.0)
    # mlp knobs
    input_dim: int = Field(16, ge=1, le=1_000_000)
    output_dim: int = Field(4, ge=1, le=1_000_000)
    hidden_max: int = Field(1024, ge=1, le=1_000_000)
    hidden_step: int = Field(16, ge=1, le=1024)
    max_hidden_layers: int = Field(3, ge=0, le=16)
    seed: int = Field(0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _ranges(self) -> SolverConstraints:
        if self.d_model_min > self.d_model_max:
            raise ValueError("d_model_min must be <= d_model_max")
        if self.n_layers_min > self.n_layers_max:
            raise ValueError("n_layers_min must be <= n_layers_max")
        if not self.n_heads_options:
            raise ValueError("n_heads_options must not be empty")
        return self


class BudgetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    predicted_parameters: int
    actual_parameters: int | None = None
    budget_error: int = 0
    budget_error_pct: float = 0.0
    satisfies_policy: bool = True
    verified: bool = False
    selected: bool = False
    estimate: ResourceEstimate | None = None
    notes: list[str] = Field(default_factory=list)


class BudgetSolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_parameters: int
    policy: BudgetPolicy
    constraints: dict[str, Any]
    candidates: list[BudgetCandidate]
    exact_match: bool
    explanation: str
    search_space_size: int

    @property
    def best(self) -> BudgetCandidate | None:
        return next((c for c in self.candidates if c.selected), None)


def _round_to(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _transformer_space(constraints: SolverConstraints, target: int) -> list[tuple[int, TransformerConfig]]:
    space: list[tuple[int, TransformerConfig]] = []
    for layers in range(constraints.n_layers_min, constraints.n_layers_max + 1):
        for d_model in range(constraints.d_model_min, constraints.d_model_max + 1, constraints.d_model_step):
            for heads in constraints.n_heads_options:
                if d_model % heads:
                    continue
                if (d_model // heads) % 2:
                    continue
                for multiplier in constraints.ffn_multipliers:
                    ffn_dim = _round_to(multiplier * d_model, 64)
                    if ffn_dim < d_model // 2:
                        continue
                    try:
                        config = TransformerConfig(
                            vocab_size=constraints.vocab_size,
                            d_model=d_model,
                            n_layers=layers,
                            n_heads=heads,
                            n_kv_heads=min(heads, max(1, heads // 2)) if heads >= 2 else heads,
                            ffn_dim=ffn_dim,
                            context_length=constraints.context_length,
                            tie_embeddings=constraints.tie_embeddings,
                            seed=constraints.seed,
                        )
                    except Exception:
                        continue
                    predicted = predict_transformer(config)
                    space.append((predicted, config))
                    if len(space) >= MAX_CANDIDATES:
                        return space
    return space


def _mlp_space(constraints: SolverConstraints) -> list[tuple[int, MlpConfig]]:
    space: list[tuple[int, MlpConfig]] = []
    for depth in range(0, constraints.max_hidden_layers + 1):
        for width in range(constraints.hidden_step, constraints.hidden_max + 1, constraints.hidden_step):
            hidden = tuple([width] * depth)
            config = MlpConfig(
                input_dim=constraints.input_dim,
                hidden_dims=hidden,
                output_dim=constraints.output_dim,
                seed=constraints.seed,
            )
            space.append((predict_mlp(config), config))
    return space


def _score(predicted: int, target: int, policy: BudgetPolicy) -> tuple[bool, int, float]:
    error = predicted - target
    pct = (abs(error) / target * 100.0) if target else 0.0
    if policy is BudgetPolicy.EXACT:
        return error == 0, error, pct
    if policy is BudgetPolicy.CLOSEST:
        return True, error, pct
    if policy is BudgetPolicy.CLOSEST_UNDER:
        return predicted <= target, error, pct
    return predicted >= target, error, pct


def solve_budget(
    target_parameters: int,
    *,
    policy: BudgetPolicy = BudgetPolicy.CLOSEST,
    constraints: SolverConstraints | None = None,
    top_k: int = 5,
    verify: bool = True,
) -> BudgetSolution:
    """Find architecture candidates for a parameter budget."""
    if target_parameters < 1:
        raise ValidationFailure("target_parameters must be >= 1")
    constraints = constraints or SolverConstraints()

    if constraints.family is FamilyId.MLP:
        space = [item for item in _mlp_space(constraints)]
    else:
        space = _transformer_space(constraints, target_parameters)

    scored: list[tuple[bool, int, float, int, Any]] = []
    for predicted, config in space:
        satisfies, error, pct = _score(predicted, target_parameters, policy)
        scored.append((satisfies, error, pct, predicted, config))

    satisfying = [item for item in scored if item[0]]
    if policy is BudgetPolicy.CLOSEST_UNDER:
        satisfying.sort(key=lambda item: (-item[3], item[1]))
    elif policy is BudgetPolicy.AT_LEAST:
        satisfying.sort(key=lambda item: (item[3], item[1]))
    else:
        satisfying.sort(key=lambda item: (item[2], abs(item[1])))

    nearest_any = sorted(scored, key=lambda item: (item[2], abs(item[1])))

    def distinct(items: list[tuple[bool, int, float, int, Any]], limit: int) -> list[tuple[bool, int, float, int, Any]]:
        """Keep one design per exact parameter count so the list is not three clones of one answer."""
        seen: set[int] = set()
        out: list[tuple[bool, int, float, int, Any]] = []
        for item in items:
            if item[3] in seen:
                continue
            seen.add(item[3])
            out.append(item)
            if len(out) >= limit:
                break
        return out

    chosen = distinct(satisfying, top_k) if satisfying else distinct(nearest_any, top_k)
    exact = any(item[3] == target_parameters for item in scored)

    candidates: list[BudgetCandidate] = []
    for index, (satisfies, error, pct, predicted, config) in enumerate(chosen):
        candidate = BudgetCandidate(
            config=config.model_dump(mode="json"),
            predicted_parameters=predicted,
            budget_error=error,
            budget_error_pct=round(pct, 4),
            satisfies_policy=satisfies,
            selected=index == 0,
        )
        if verify:
            from .registry import build_model

            handle = build_model(candidate.config)
            candidate.actual_parameters = handle.parameter_count
            candidate.verified = True
            candidate.budget_error = handle.parameter_count - target_parameters
            candidate.budget_error_pct = round(
                abs(candidate.budget_error) / target_parameters * 100.0, 4
            ) if target_parameters else 0.0
            candidate.estimate = estimate_resources(
                handle.config,
                parameter_count=handle.parameter_count,
                train=True,
                batch_size=8,
                block_size=min(32, getattr(handle.config, "context_length", 32)),
            )
            if predicted != handle.parameter_count:
                candidate.notes.append(
                    f"formula predicted {predicted}, enumeration found {handle.parameter_count}"
                )
        candidates.append(candidate)

    if exact:
        explanation = f"Exact budget of {target_parameters:,} parameters is reachable within the constraints."
    elif not satisfying:
        explanation = (
            f"No architecture within the constraints satisfies policy '{policy.value}' for "
            f"{target_parameters:,} parameters. Exact counts are discrete: the shown candidates are the "
            "closest feasible designs in the searched space."
        )
    else:
        explanation = (
            f"No exact match for {target_parameters:,} parameters exists in the searched discrete space. "
            f"The closest design under policy '{policy.value}' is shown first."
        )

    return BudgetSolution(
        target_parameters=target_parameters,
        policy=policy,
        constraints=constraints.model_dump(mode="json"),
        candidates=candidates,
        exact_match=exact,
        explanation=explanation,
        search_space_size=len(space),
    )


def parse_parameter_budget(value: str | int) -> int:
    """Parse ``250k``, ``1.2M``, ``10m``, ``1B`` or a plain integer."""
    if isinstance(value, int):
        return value
    text = value.strip().lower().replace("_", "").replace(",", "")
    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}
    if text and text[-1] in suffixes:
        number = float(text[:-1])
        return int(round(number * suffixes[text[-1]]))
    if not text.isdigit():
        raise ValidationFailure(
            f"cannot parse parameter budget '{value}'",
            hint="use an integer or a suffix such as 250k, 1.2M, 1B",
        )
    return int(text)


def human_parameters(count: int) -> str:
    for limit, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if count >= limit:
            value = count / limit
            return f"{value:.2f}{suffix}".rstrip("0").rstrip(".") + f" ({count:,})"
    return f"{count:,}"


__all__ = [
    "BudgetCandidate",
    "BudgetSolution",
    "SolverConstraints",
    "human_parameters",
    "math",
    "parse_parameter_budget",
    "solve_budget",
]
