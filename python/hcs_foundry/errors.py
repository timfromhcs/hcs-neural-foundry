"""Typed, actionable errors for HCS Neural Foundry.

Every error carries a machine-readable ``code`` so the CLI, the HTTP API and the MCP server can
return the same failure semantics, plus a ``hint`` a human can act on.
"""

from __future__ import annotations

from typing import Any


class FoundryError(Exception):
    """Base class for all expected, user-facing failures."""

    code = "foundry_error"
    http_status = 400

    def __init__(self, message: str, *, hint: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


class NotFoundError(FoundryError):
    code = "not_found"
    http_status = 404


class ConflictError(FoundryError):
    code = "conflict"
    http_status = 409


class ValidationFailure(FoundryError):
    code = "validation_failed"
    http_status = 422


class BudgetExceeded(FoundryError):
    code = "budget_exceeded"
    http_status = 409


class SecurityViolation(FoundryError):
    code = "security_violation"
    http_status = 403


class MissingDependencyError(FoundryError):
    code = "missing_dependency"
    http_status = 501


class UnsupportedCapability(FoundryError):
    code = "unsupported_capability"
    http_status = 501


class RunInvalid(FoundryError):
    """The evaluator itself failed. The run is ``invalid``, never silently scored as zero."""

    code = "run_invalid"
    http_status = 422


def require(condition: bool, message: str, *, hint: str | None = None) -> None:
    """Small assertion helper that raises a typed error instead of ``AssertionError``."""
    if not condition:
        raise ValidationFailure(message, hint=hint)
