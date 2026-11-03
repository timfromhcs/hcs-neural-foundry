"""Model factory: families, exact parameter accounting and the budget solver."""

from __future__ import annotations

from .base import (
    ModelFamilyAdapter,
    ModelHandle,
    SmokeTestReport,
    ValidationIssue,
    count_parameters,
    seed_everything,
    torch_module,
)
from .registry import (
    build_model,
    default_config,
    get_family,
    list_families,
    parse_config,
    validate_config,
)
from .solver import (
    BudgetCandidate,
    BudgetSolution,
    SolverConstraints,
    human_parameters,
    parse_parameter_budget,
    solve_budget,
)

__all__ = [
    "BudgetCandidate",
    "BudgetSolution",
    "ModelFamilyAdapter",
    "ModelHandle",
    "SmokeTestReport",
    "SolverConstraints",
    "ValidationIssue",
    "build_model",
    "count_parameters",
    "default_config",
    "get_family",
    "human_parameters",
    "list_families",
    "parse_config",
    "parse_parameter_budget",
    "seed_everything",
    "solve_budget",
    "torch_module",
    "validate_config",
]
