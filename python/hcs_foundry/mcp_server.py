"""MCP server (stdio).

Narrow, typed tools over the same service layer as the HTTP API (ADR-014). There is deliberately no
``run_shell``, no ``read_any_file`` and no ``eval`` tool. Mutating tools require an explicit approval
list, because an MCP client is by definition an autonomous caller.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
from pathlib import Path
from typing import Any

from .agents import AgentPolicy, AutonomyLevel
from .errors import FoundryError, SecurityViolation
from .schemas import EvalReport, PatchOperation, TrainingConfig, WeightPatch
from .services import Foundry

# Tools that change state or spend compute. They refuse to run unless explicitly approved.
GATED_TOOLS = {
    "foundry_run_weight_patch",
    "foundry_start_training",
    "foundry_start_evaluation",
    "foundry_export_model",
    "foundry_evolve",
    "foundry_generate_weights",
}


class ApprovalGate:
    def __init__(self, approved: set[str] | None = None):
        env = os.environ.get("HCS_FOUNDRY_MCP_APPROVE", "")
        from_env = {item.strip() for item in env.split(",") if item.strip()}
        self.approved = (approved or set()) | from_env

    def require(self, tool: str) -> None:
        if tool not in GATED_TOOLS:
            return
        if tool in self.approved or "*" in self.approved:
            return
        raise SecurityViolation(
            f"tool '{tool}' is approval-gated",
            hint=(
                "restart the MCP server with --approve "
                + tool
                + " (or set HCS_FOUNDRY_MCP_APPROVE) after reviewing the call"
            ),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "gated_tools": sorted(GATED_TOOLS),
            "approved_tools": sorted(self.approved),
            "default_level": int(AutonomyLevel.DRAFT),
        }


def build_server(foundry: Foundry | None = None, *, approved: set[str] | None = None):
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - optional extra
        raise FoundryError(
            "the MCP extra is not installed",
            hint='install: uv pip install -e ".[mcp]"',
            details={"import_error": str(exc)},
        ) from exc

    foundry = foundry or Foundry()
    gate = ApprovalGate(approved)
    policy = AgentPolicy()
    server = FastMCP(
        "hcs-neural-foundry",
        instructions=(
            "Local-first neural network workbench. Read-only tools are always available. Tools that "
            "mutate weights, spend compute or export files are approval-gated. Every result is tied to "
            "a run with dataset hash, seed and evaluator version. Never describe generated or edited "
            "weights as state of the art."
        ),
    )

    def _log(tool: str, args: dict[str, Any], result: Any, approved: bool) -> None:
        # Logging must never break a tool call.
        with contextlib.suppress(Exception):
            foundry.repo.log_tool_call(tool, args, {"ok": True, "result": _truncate(result)}, approved=approved)

    def _truncate(payload: Any, limit: int = 4000) -> Any:
        text = json.dumps(payload, default=str)
        return payload if len(text) <= limit else {"truncated": True, "preview": text[:limit]}

    def read_tool(name: str):
        def decorator(func):
            @server.tool(name=name)
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any):
                try:
                    result = func(*args, **kwargs)
                    _log(name, kwargs, result, approved=False)
                    return result
                except FoundryError as exc:
                    return {"error": exc.to_dict()}

            return wrapper

        return decorator

    def gated_tool(name: str):
        def decorator(func):
            @server.tool(name=name)
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any):
                try:
                    gate.require(name)
                    result = func(*args, **kwargs)
                    _log(name, kwargs, result, approved=True)
                    return result
                except FoundryError as exc:
                    return {"error": exc.to_dict()}

            return wrapper

        return decorator

    # -- read-only -------------------------------------------------------------------------
    @read_tool("foundry_get_hardware_profile")
    def get_hardware_profile() -> dict[str, Any]:
        """Hardware profile with declared-vs-usable backend capabilities."""
        return foundry.hardware.model_dump(mode="json")

    @read_tool("foundry_capabilities")
    def capabilities() -> dict[str, Any]:
        """Model families, evaluation suites and agent policy."""
        from .evaluation import list_suites
        from .models import list_families

        return {
            "families": list_families(),
            "suites": list_suites(),
            "policy": policy.describe(),
            "gate": gate.describe(),
        }

    @read_tool("foundry_list_models")
    def list_models() -> list[dict[str, Any]]:
        """List models in this workspace."""
        return [model.model_dump(mode="json") for model in foundry.list_models()]

    @read_tool("foundry_inspect_model")
    def inspect_model(model_id: str) -> dict[str, Any]:
        """Model definition, exact parameter breakdown and checkpoint list."""
        model = foundry.get_model(model_id)
        return {
            "model": model.model_dump(mode="json"),
            "parameters": foundry.parameter_report(model_id),
            "checkpoints": [
                record.model_dump(mode="json") for record in foundry.repo.list_checkpoints(model_id)
            ],
        }

    @read_tool("foundry_list_tensors")
    def list_tensors(model_id: str) -> list[dict[str, Any]]:
        """Tensor metadata: shapes, statistics, finiteness, tie information."""
        return [tensor.model_dump(mode="json") for tensor in foundry.list_tensors(model_id)]

    @read_tool("foundry_read_tensor_slice")
    def read_tensor_slice(model_id: str, tensor: str, offset: str = "", shape: str = "") -> dict[str, Any]:
        """Bounded tensor read (max 4096 numbers)."""
        parsed_offset = tuple(int(v) for v in offset.split(",")) if offset else None
        parsed_shape = tuple(int(v) for v in shape.split(",")) if shape else None
        return foundry.tensor_slice(model_id, tensor, offset=parsed_offset, shape=parsed_shape)

    @read_tool("foundry_validate_weight_patch")
    def validate_weight_patch(model_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Schema-level validation of a patch without touching any tensor."""
        foundry.get_model(model_id)
        parsed = WeightPatch.model_validate(patch)
        return {"valid": True, "patch": parsed.model_dump(mode="json")}

    @read_tool("foundry_solve_parameter_budget")
    def solve_parameter_budget(target_parameters: int, family: str = "transformer-lm", policy: str = "closest") -> dict[str, Any]:
        """Find architectures for a parameter budget, with exact enumerated counts."""
        return foundry.solve(target_parameters, policy=policy, family=family, top_k=5)

    @read_tool("foundry_get_run_status")
    def get_run_status(run_id: str) -> dict[str, Any]:
        """State, metrics and summary of a run."""
        return foundry.repo.get_run(run_id).model_dump(mode="json")

    @read_tool("foundry_list_history")
    def list_history(model_id: str = "", kind: str = "", limit: int = 25) -> list[dict[str, Any]]:
        """Recent runs with provenance."""
        return foundry.history(model_id=model_id or None, kind=kind or None, limit=limit)

    @read_tool("foundry_get_evaluation_report")
    def get_evaluation_report(evaluation_id: str) -> dict[str, Any]:
        """Stored evaluation report with metric definitions and provenance."""
        return foundry.repo.get_evaluation(evaluation_id).model_dump(mode="json")

    @read_tool("foundry_estimate_training_resources")
    def estimate_training_resources(model_id: str, batch_size: int = 8, block_size: int = 32) -> dict[str, Any]:
        """Memory/compute estimate with explicit assumptions and warnings."""
        return foundry.estimate(model_id, batch_size=batch_size, block_size=block_size)

    @read_tool("foundry_list_tool_calls")
    def list_tool_calls(limit: int = 25) -> list[dict[str, Any]]:
        """Audit log of MCP tool invocations."""
        return foundry.repo.list_tool_calls(limit=limit)

    # -- gated -----------------------------------------------------------------------------
    @gated_tool("foundry_start_evaluation")
    def start_evaluation(model_id: str, suite: str = "auto", seed: int = 1234) -> dict[str, Any]:
        """Run an evaluation suite. Approval-gated because it spends compute."""
        report: EvalReport = foundry.evaluate(model_id, suite=suite, seed=seed)
        return report.model_dump(mode="json")

    @gated_tool("foundry_start_training")
    def start_training(model_id: str, steps: int = 100, batch_size: int = 16, lr: float = 0.003) -> dict[str, Any]:
        """Run a short supervised training job synchronously. Approval-gated."""
        model = foundry.get_model(model_id)
        dataset = "tiny-text" if model.config.family == "transformer-lm" else "tiny-shapes"
        config = TrainingConfig(
            model_id=model_id,
            dataset_id=dataset,
            steps=steps,
            batch_size=batch_size,
            eval_every=max(1, steps // 2),
            checkpoint_every=max(1, steps),
            max_seconds=300,
        )
        config.optimizer.lr = lr
        run = foundry.train(config)
        return {"run_id": run.id, "state": run.state.value, "summary": run.summary}

    @gated_tool("foundry_propose_weight_patch")
    def propose_weight_patch(
        model_id: str,
        tensor: str,
        operation: str,
        parameters: dict[str, Any],
        goal: str = "",
        hypothesis: str = "",
    ) -> dict[str, Any]:
        """Create a proposal (not an application) for a weight patch."""
        return foundry.propose_patch(
            model_id,
            agent_id="agent:mcp",
            goal=goal or "weight patch",
            hypothesis=hypothesis,
            tensor=tensor,
            operation=PatchOperation(operation),
            parameters=parameters,
            policy=policy,
        )

    @gated_tool("foundry_run_weight_patch")
    def run_weight_patch(proposal_id: str) -> dict[str, Any]:
        """Apply an approved proposal. Approval-gated and audited."""
        return foundry.approve_proposal(proposal_id, approver="agent:mcp")

    @gated_tool("foundry_export_model")
    def export_model(model_id: str, format: str = "onnx") -> dict[str, Any]:
        """Export to ONNX with a parity check. Approval-gated (writes files)."""
        return foundry.export(model_id, fmt=format)

    @gated_tool("foundry_evolve")
    def evolve(model_id: str, generations: int = 3, population: int = 12) -> dict[str, Any]:
        """Run antithetic ES with a MAP-Elites archive. Approval-gated."""
        from .schemas import EvolutionConfig

        return foundry.evolve(
            EvolutionConfig(model_id=model_id, generations=generations, population=population)
        )

    @gated_tool("foundry_generate_weights")
    def generate_weights(model_id: str, epochs: int = 150) -> dict[str, Any]:
        """Train the conditional hypernetwork and generate weights. Approval-gated."""
        from .schemas import GeneratorConfig

        return foundry.generate(GeneratorConfig(target_model_id=model_id, epochs=epochs))

    # -- resources -------------------------------------------------------------------------
    @server.resource("foundry://capabilities", mime_type="application/json")
    def capabilities_resource() -> str:
        from .evaluation import list_suites
        from .models import list_families

        return json.dumps(
            {
                "families": list_families(),
                "suites": list_suites(),
                "gate": gate.describe(),
                "policy": policy.describe(),
            },
            indent=2,
        )

    @server.resource("foundry://models", mime_type="application/json")
    def models_resource() -> str:
        return json.dumps([model.model_dump(mode="json") for model in foundry.list_models()], indent=2)

    @server.resource("foundry://runs", mime_type="application/json")
    def runs_resource() -> str:
        return json.dumps(foundry.history(limit=25), indent=2, default=str)

    # -- prompts ---------------------------------------------------------------------------
    @server.prompt("beginner_model_design")
    def beginner_model_design(goal: str = "classify a small tabular dataset") -> str:
        return (
            f"Design a small model for: {goal}.\n"
            "1. Call foundry_solve_parameter_budget with a budget under 500k parameters.\n"
            "2. Explain the parameter breakdown per module, in plain language.\n"
            "3. State the memory estimate and its assumptions.\n"
            "4. Propose one training config and one evaluation suite.\n"
            "Do not claim the model will perform well before an evaluation run exists."
        )

    @server.prompt("weight_patch_review")
    def weight_patch_review(tensor: str = "layers.0.weight") -> str:
        return (
            f"Review a proposed patch on {tensor}.\n"
            "Check: base checkpoint hash, shape/dtype preconditions, expected norm change, "
            "finite-value risk, and whether the hypothesis is falsifiable. "
            "Then define the exact evaluation that would falsify it."
        )

    @server.prompt("experiment_comparison")
    def experiment_comparison() -> str:
        return (
            "Compare the available runs on equal footing: same dataset hash, same evaluator version, "
            "same seed policy, and disclose unequal compute. Never merge metrics into one opaque score."
        )

    return server


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="hcs-foundry mcp", description="MCP server over stdio")
    parser.add_argument("--home", default=None)
    parser.add_argument("--approve", action="append", default=[], help="approve a gated tool")
    args = parser.parse_args(argv)
    foundry = Foundry(Path(args.home) if args.home else None)
    server = build_server(foundry, approved=set(args.approve))
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
