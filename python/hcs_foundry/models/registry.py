"""Model family registry.

Adding a family means adding a module that exposes ``FAMILY`` and registering it here. The rest of
the application (CLI, API, MCP, web) only ever talks to these adapters.
"""

from __future__ import annotations

from typing import Any

from ..errors import NotFoundError
from ..schemas import FamilyId, Maturity, MlpConfig, TransformerConfig
from .base import ModelFamilyAdapter, ModelHandle, ValidationIssue
from .mlp import FAMILY as MLP_FAMILY
from .transformer_lm import FAMILY as TRANSFORMER_FAMILY

_FAMILIES: dict[str, ModelFamilyAdapter] = {
    MLP_FAMILY.family_id: MLP_FAMILY,
    TRANSFORMER_FAMILY.family_id: TRANSFORMER_FAMILY,
}


def list_families() -> list[dict[str, Any]]:
    return [
        {
            "id": adapter.family_id,
            "display_name": adapter.display_name,
            "maturity": adapter.maturity.value,
        }
        for adapter in _FAMILIES.values()
    ]


def get_family(family_id: str) -> ModelFamilyAdapter:
    adapter = _FAMILIES.get(family_id)
    if adapter is None:
        raise NotFoundError(
            f"unknown model family '{family_id}'",
            hint=f"available families: {sorted(_FAMILIES)}",
        )
    return adapter


def parse_config(payload: dict[str, Any]) -> MlpConfig | TransformerConfig:
    """Validate a raw dict into a concrete, frozen architecture config."""
    family = payload.get("family")
    if family == FamilyId.MLP.value:
        return MlpConfig.model_validate(payload)
    if family == FamilyId.TRANSFORMER_LM.value:
        return TransformerConfig.model_validate(payload)
    raise NotFoundError(
        f"unknown or missing family '{family}'",
        hint=f"available families: {sorted(_FAMILIES)}",
    )


def validate_config(payload: dict[str, Any]) -> list[ValidationIssue]:
    try:
        config = parse_config(payload)
    except Exception as exc:
        return [ValidationIssue(field="config", message=str(exc))]
    return get_family(config.family).validate_config(payload)


def build_model(payload: dict[str, Any]) -> ModelHandle:
    config = parse_config(payload)
    return get_family(config.family).build(config)


def default_config(family: FamilyId = FamilyId.TRANSFORMER_LM) -> dict[str, Any]:
    if family is FamilyId.MLP:
        return MlpConfig().model_dump(mode="json")
    return TransformerConfig().model_dump(mode="json")


def adapter_maturity(family_id: str) -> Maturity:
    return get_family(family_id).maturity
