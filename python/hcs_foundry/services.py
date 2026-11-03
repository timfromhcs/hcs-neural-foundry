"""Service layer: the single orchestration surface for CLI, HTTP API and MCP server.

Putting orchestration here (instead of in each transport) is what keeps the three interfaces
behaviourally identical: same validators, same policies, same provenance.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .agents import AgentPolicy, build_proposal, verify_isolated
from .checkpoints import (
    flattened_state,
    load_state,
    read_manifest,
    restore_aliases,
    save_checkpoint,
    write_artifact,
)
from .errors import ConflictError, NotFoundError, ValidationFailure
from .evaluation import evaluate as evaluate_handle
from .evaluation import list_suites
from .evolution import run_evolution
from .exporting import export_onnx, inspect_onnx
from .generators import train_hypernetwork
from .hardware import ENGINE_VERSION, detect_hardware
from .hashing import model_slug, new_id, utcnow
from .models import build_model, get_family, list_families, parse_config, solve_budget
from .models.base import ModelHandle, torch_module
from .models.solver import SolverConstraints, human_parameters
from .schemas import (
    BudgetPolicy,
    EvalReport,
    EvolutionConfig,
    GeneratorConfig,
    MetricPoint,
    ModelDefinition,
    PatchOperation,
    PatchPrecondition,
    Provenance,
    RunRecord,
    RunState,
    TensorSummary,
    TrainingConfig,
    WeightPatch,
    architecture_hash,
)
from .storage import Repository, Workspace
from .training import datasets as dataset_module
from .training.trainer import SupervisedTrainer, TrainerControl
from .weights import apply_patch as apply_patch_to_state
from .weights import histogram as tensor_histogram
from .weights import slice_tensor, summarize


class Foundry:
    """One workspace: database, artifacts, models and services."""

    def __init__(self, home: str | Path | None = None, *, probe_backends: bool = False):
        self.workspace = Workspace(home)
        self.repo = Repository(self.workspace)
        self._hardware = None
        self._probe = probe_backends
        self.ensure_datasets()

    # -- environment -------------------------------------------------------------------------
    @property
    def hardware(self):
        if self._hardware is None:
            self._hardware = detect_hardware(probe_backends=self._probe)
        return self._hardware

    def ensure_datasets(self) -> None:
        for spec in dataset_module.ensure_builtin_datasets(self.workspace.root):
            existing = None
            try:
                existing = self.repo.get_dataset(spec.id)
            except NotFoundError:
                existing = None
            if existing is None or existing.sha256 != spec.sha256:
                self.repo.upsert_dataset(spec)

    def doctor(self) -> dict[str, Any]:
        profile = self.hardware
        return {
            "engine_version": __version__,
            "workspace": str(self.workspace.root),
            "database": {"path": str(self.workspace.db_path), "schema_version": self.repo.db.version},
            "hardware": profile.model_dump(mode="json"),
            "families": list_families(),
            "suites": list_suites(),
            "datasets": [spec.model_dump(mode="json") for spec in self.repo.list_datasets()],
            "notes": [
                "a backend is only 'usable' after a probe actually ran on it",
                "synthetic datasets measure pipeline correctness, not model capability",
            ],
        }

    # -- models ------------------------------------------------------------------------------
    def create_model(
        self,
        name: str,
        config: dict[str, Any],
        *,
        notes: str = "",
        source: str = "created",
        checkpoint: bool = True,
    ) -> ModelDefinition:
        parsed = parse_config(config)
        issues = [issue for issue in get_family(parsed.family).validate_config(config) if issue.severity == "error"]
        if issues:
            raise ValidationFailure("; ".join(issue.message for issue in issues))
        digest = architecture_hash(parsed)
        base = model_slug(name)
        model_id = base
        suffix = 1
        while True:
            try:
                self.repo.get_model(model_id)
                suffix += 1
                model_id = f"{base}-{suffix}"
            except NotFoundError:
                break
        handle = build_model(parsed.model_dump(mode="json"))
        model = ModelDefinition(
            id=model_id,
            name=name,
            config=parsed,
            architecture_hash=digest,
            parameter_count=handle.parameter_count,
            notes=notes,
            source=source,  # type: ignore[arg-type]
        )
        if model.notes == "" and handle.notes:
            model.notes = " | ".join(handle.notes)
        self.repo.add_model(model)
        if checkpoint:
            record = save_checkpoint(
                self.workspace,
                model_id=model.id,
                module=handle.module,
                architecture_hash=digest,
                config_payload=parsed.model_dump(mode="json"),
                label="init",
                origin="init",
            )
            self.repo.add_checkpoint(record, digest)
        self.repo.log_event(
            "info",
            f"model:{model.id}",
            f"created {parsed.family} with {handle.parameter_count:,} parameters",
            {"architecture_hash": digest},
        )
        return model

    def list_models(self):
        return self.repo.list_models()

    def get_model(self, model_id: str) -> ModelDefinition:
        return self.repo.get_model(model_id)

    def handle_for(
        self, model_id: str, checkpoint: str | None = None
    ) -> tuple[ModelHandle, Any | None]:
        """Build a model. ``checkpoint=None`` means the deterministic initialisation."""
        model = self.repo.get_model(model_id)
        handle = build_model(model.config.model_dump(mode="json"))
        if checkpoint is None:
            return handle, None
        if checkpoint in {"latest", ""}:
            record = self.repo.latest_checkpoint(model_id)
        elif checkpoint == "init":
            records = self.repo.list_checkpoints(model_id)
            record = next((r for r in records if r.origin == "init"), records[0] if records else None)
        elif len(checkpoint) == 64:
            record = self.repo.find_checkpoint_by_sha(checkpoint)
            if record is None:
                raise NotFoundError(f"no checkpoint with sha256 {checkpoint[:12]}… in this workspace")
        else:
            record = self.repo.get_checkpoint(checkpoint)
        if record is None:
            return handle, None
        if record.model_id != model_id:
            raise ConflictError(
                f"checkpoint {record.id} belongs to model {record.model_id}, not {model_id}"
            )
        load_state(handle, record.path)
        return handle, record

    def parameter_report(self, model_id: str) -> dict[str, Any]:
        handle, _ = self.handle_for(model_id, None)
        report = handle.report
        payload = report.model_dump(mode="json")
        payload["by_module"] = report.by_module(limit=32)
        payload["human"] = human_parameters(report.total)
        payload["predictions_agree"] = True
        family = get_family(handle.family)
        payload["formula_parameters"] = family.predict_parameters(handle.config)
        payload["formula_agrees"] = payload["formula_parameters"] == report.total
        return payload

    def estimate(self, model_id: str, *, batch_size: int = 8, block_size: int = 32) -> dict[str, Any]:
        handle, _ = self.handle_for(model_id, None)
        estimate = get_family(handle.family).estimate_resources(
            handle.config,
            parameter_count=handle.parameter_count,
            train=True,
            batch_size=batch_size,
            block_size=min(block_size, getattr(handle.config, "context_length", block_size)),
        )
        return estimate.model_dump(mode="json")

    def solve(
        self,
        target: int,
        *,
        policy: str = "closest",
        family: str = "transformer-lm",
        constraints: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        payload = dict(constraints or {})
        payload.setdefault("family", family)
        parsed_constraints = SolverConstraints.model_validate(payload)
        solution = solve_budget(
            target,
            policy=BudgetPolicy(policy),
            constraints=parsed_constraints,
            top_k=top_k,
        )
        return solution.model_dump(mode="json")

    # -- weights -----------------------------------------------------------------------------
    def list_tensors(self, model_id: str, checkpoint: str | None = None) -> list[TensorSummary]:
        handle, _ = self.handle_for(model_id, checkpoint if checkpoint is not None else "latest")
        return summarize(handle.module)

    def tensor_slice(
        self,
        model_id: str,
        name: str,
        *,
        offset: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        handle, _ = self.handle_for(model_id, checkpoint if checkpoint is not None else "latest")
        return slice_tensor(handle.module, name, offset=offset, shape=shape).model_dump(mode="json")

    def tensor_histogram(self, model_id: str, name: str, *, bins: int = 32) -> dict[str, Any]:
        handle, _ = self.handle_for(model_id, "latest")
        return tensor_histogram(handle.module, name, bins=bins).model_dump(mode="json")

    def apply_patch(self, model_id: str, patch: WeightPatch | dict[str, Any], *, require_base: bool = True) -> dict[str, Any]:
        """Validate and apply one patch, producing a new checkpoint with provenance."""
        if not isinstance(patch, WeightPatch):
            patch = WeightPatch.model_validate(patch)
        model = self.repo.get_model(model_id)
        if patch.architecture_hash != model.architecture_hash:
            raise ValidationFailure(
                "patch architecture hash does not match the model",
                hint="rebuild the patch against the current architecture",
            )
        base = self.repo.find_checkpoint_by_sha(patch.base_checkpoint_sha256)
        if base is None:
            raise NotFoundError(
                f"base checkpoint {patch.base_checkpoint_sha256[:12]}… is not in this workspace"
            )
        if base.model_id != model_id:
            raise ConflictError("base checkpoint belongs to a different model")

        handle = build_model(model.config.model_dump(mode="json"))
        load_state(handle, base.path)
        manifest = read_manifest(base.path)
        aliases: dict[str, str] = manifest.get("aliases", {})
        canonical = aliases.get(patch.target_tensor, patch.target_tensor)

        state = flattened_state(handle.module)["tensors"]
        other_state = None
        if patch.operation in (PatchOperation.INTERPOLATE, PatchOperation.COPY_FROM):
            key = patch.parameters.get(
                "other_checkpoint_sha256", patch.parameters.get("source_checkpoint_sha256")
            )
            other = self.repo.find_checkpoint_by_sha(key) if key else None
            if other is None:
                raise NotFoundError(f"referenced checkpoint {str(key)[:12]}… is not in this workspace")
            other_handle = build_model(model.config.model_dump(mode="json"))
            load_state(other_handle, other.path)
            other_state = flattened_state(other_handle.module)["tensors"]

        canonical_patch = patch if canonical == patch.target_tensor else patch.model_copy(
            update={"target_tensor": canonical}
        )
        new_state, validation = apply_patch_to_state(state, canonical_patch, other_state=other_state)

        handle.module.load_state_dict({**new_state}, strict=False)
        if aliases:
            restore_aliases(handle.module, aliases)

        if patch.operation is PatchOperation.SET_TRAINABLE:
            params = dict(handle.module.named_parameters(remove_duplicate=False))
            target = params.get(canonical)
            if target is None:
                raise ValidationFailure(f"parameter '{canonical}' not found")
            target.requires_grad = bool(patch.parameters["trainable"])

        record = save_checkpoint(
            self.workspace,
            model_id=model_id,
            module=handle.module,
            architecture_hash=handle.architecture_hash,
            config_payload=model.config.model_dump(mode="json"),
            label=f"patch:{patch.operation.value}",
            origin="patch",
            parent_sha256=base.sha256,
        )
        self.repo.add_checkpoint(record, handle.architecture_hash)
        applied = patch.model_copy(update={"status": "applied"})
        self.repo.add_patch(model_id, applied, checkpoint_id=record.id)
        self.repo.log_event(
            "info",
            f"patch:{applied.patch_id}",
            f"{applied.operation.value} on {applied.target_tensor}",
            {"checkpoint": record.sha256, "by": applied.created_by},
        )
        return {
            "patch": applied.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
            "checkpoint": record.model_dump(mode="json"),
        }

    def rollback(self, model_id: str, checkpoint_id: str) -> dict[str, Any]:
        """Move the model head back to an existing checkpoint.

        Checkpoints are content-addressed and immutable, so a rollback changes a pointer instead of
        duplicating bytes. The event log keeps the audit trail.
        """
        record = self.repo.get_checkpoint(checkpoint_id)
        if record.model_id != model_id:
            raise ConflictError("checkpoint belongs to a different model")
        previous = self.repo.get_head(model_id)
        self.repo.set_head(model_id, record.id)
        self.repo.log_event(
            "warning",
            f"model:{model_id}",
            f"head moved to {record.sha256[:12]}",
            {
                "from": previous.sha256 if previous else None,
                "to": record.sha256,
                "label": record.label,
            },
        )
        payload = record.model_dump(mode="json")
        payload["head"] = record.sha256
        payload["previous_head"] = previous.sha256 if previous else None
        return payload

    # -- proposals ---------------------------------------------------------------------------
    def propose_patch(
        self,
        model_id: str,
        *,
        agent_id: str,
        goal: str,
        hypothesis: str,
        tensor: str,
        operation: PatchOperation,
        parameters: dict[str, Any],
        base_checkpoint: str = "latest",
        policy: AgentPolicy | None = None,
    ) -> dict[str, Any]:
        handle, record = self.handle_for(model_id, base_checkpoint)
        if record is None:
            raise NotFoundError("no checkpoint to patch")
        state = flattened_state(handle.module)["tensors"]
        tensor_value = state.get(tensor)
        if tensor_value is None:
            raise NotFoundError(f"tensor '{tensor}' not found", hint="list tensors first")
        patch = WeightPatch(
            patch_id=new_id("patch"),
            base_checkpoint_sha256=record.sha256,
            architecture_hash=handle.architecture_hash,
            operation=operation,
            target_tensor=tensor,
            parameters=parameters,
            precondition=PatchPrecondition(
                shape=tuple(tensor_value.shape),
                dtype=str(tensor_value.dtype).replace("torch.", ""),
                base_checkpoint_sha256=record.sha256,
            ),
            created_by=agent_id,
            note=goal,
        )
        proposal = build_proposal(
            agent_id=agent_id,
            goal=goal,
            hypothesis=hypothesis,
            model_id=model_id,
            base_checkpoint_sha256=record.sha256,
            patches=[patch],
        )
        (policy or AgentPolicy()).check(proposal)
        self.repo.add_proposal(model_id, proposal.model_dump(mode="json"), approval="pending")
        return proposal.model_dump(mode="json")

    def approve_proposal(self, proposal_id: str, *, approver: str = "user") -> dict[str, Any]:
        payload = self.repo.get_proposal(proposal_id)
        if payload.get("approval") == "approved":
            raise ConflictError("proposal already approved")
        applied = []
        for patch_payload in payload["patches"]:
            patch = WeightPatch.model_validate(patch_payload)
            applied.append(self.apply_patch(payload["model_id"], patch))
        payload["approval"] = "approved"
        payload["approved_by"] = approver
        payload["applied"] = applied
        self.repo.update_proposal(proposal_id, approval="approved", payload=payload)
        # Re-apply outcome metadata onto the stored proposal for auditability.
        self.repo.update_proposal(proposal_id, approval="approved", payload={**payload, "approval": "approved"})
        return {"proposal_id": proposal_id, "applied": applied}

    def verify_proposal(self, proposal_id: str, *, suite: str = "weight-sanity") -> dict[str, Any]:
        payload = self.repo.get_proposal(proposal_id)
        model = self.repo.get_model(payload["model_id"])
        applied = payload.get("applied") or []
        if not applied:
            raise ValidationFailure("proposal has not been applied yet")
        checkpoint = applied[-1]["checkpoint"]
        return verify_isolated(
            workspace_root=self.workspace.root,
            model_config=model.config.model_dump(mode="json"),
            checkpoint_path=checkpoint["path"],
            suite_id=suite,
        )

    # -- training / evaluation ---------------------------------------------------------------
    def train(
        self,
        config: TrainingConfig,
        *,
        control: TrainerControl | None = None,
        progress: Callable[[MetricPoint], None] | None = None,
        resume: bool = True,
        run_id: str | None = None,
    ) -> RunRecord:
        model = self.repo.get_model(config.model_id)
        dataset = self.repo.get_dataset(config.dataset_id)
        expected = dataset_module.dataset_for_model(model.config.family)
        if config.dataset_id != expected:
            raise ValidationFailure(
                f"dataset '{config.dataset_id}' is not valid for family '{model.config.family}'",
                hint=f"use '{expected}'",
            )
        handle = build_model(model.config.model_dump(mode="json"))
        base = self.repo.latest_checkpoint(model.id) if resume else None
        if base and resume:
            records = self.repo.list_checkpoints(model.id)
            base = records[-1] if records else None
        trainer = SupervisedTrainer(
            self.workspace,
            self.repo,
            handle,
            config,
            dataset=dataset,
            base_checkpoint=base,
            control=control,
            progress=progress,
            profile=self.hardware,
            run_id=run_id,
        )
        return trainer.run()

    def evaluate(
        self,
        model_id: str,
        *,
        suite: str = "auto",
        checkpoint: str | None = None,
        seed: int = 1234,
        batches: int = 4,
    ) -> EvalReport:
        model = self.repo.get_model(model_id)
        handle, record = self.handle_for(model_id, checkpoint or "latest")
        if suite == "auto":
            suite = "text-smoke" if model.config.family == "transformer-lm" else "shapes-smoke"
        sha = record.sha256 if record else ""
        report = evaluate_handle(handle, suite_id=suite, checkpoint_sha256=sha, seed=seed, batches=batches)
        report.model_id = model_id
        run = RunRecord(
            id=report.run_id,
            kind="evaluation",
            model_id=model_id,
            state=report.state,
            config={"suite": suite, "seed": seed, "batches": batches},
            summary={
                "metrics": {metric.name: metric.value for metric in report.metrics},
                "checkpoint_sha256": sha,
            },
            provenance=report.provenance,
            finished_at=utcnow(),
        )
        self.repo.add_run(run)
        self.repo.add_evaluation(report.run_id, report)
        self.repo.log_event(
            "info" if report.state is RunState.COMPLETED else "warning",
            f"evaluation:{report.run_id}",
            f"{suite} finished with state {report.state.value}",
            {"metrics": {metric.name: metric.value for metric in report.metrics}},
        )
        return report

    # -- research features -------------------------------------------------------------------
    def evolve(self, config: EvolutionConfig) -> dict[str, Any]:
        model = self.repo.get_model(config.model_id)
        handle, record = self.handle_for(config.model_id, "latest")
        progress_points: list[dict[str, Any]] = []

        def progress(entry: dict[str, Any]) -> None:
            progress_points.append(entry)

        result = run_evolution(handle, config, progress=progress)
        run = RunRecord(
            id=new_id("run"),
            kind="evolution",
            model_id=config.model_id,
            state=RunState.COMPLETED,
            config=config.model_dump(mode="json"),
            summary={k: v for k, v in result.items() if k != "best_parameters"},
            provenance=Provenance(
                engine_version=ENGINE_VERSION,
                python=platform.python_version(),
                torch=torch_module().__version__,
                platform=platform.platform(),
                seeds={"evolution": config.seed},
                architecture_hash=model.architecture_hash,
                base_checkpoint_sha256=record.sha256 if record else None,
            ),
            finished_at=utcnow(),
        )
        self.repo.add_run(run)

        from .evolution import assign_parameters

        assign_parameters(handle.module, np.asarray(result["best_parameters"], dtype=np.float32))
        checkpoint = save_checkpoint(
            self.workspace,
            model_id=config.model_id,
            module=handle.module,
            architecture_hash=handle.architecture_hash,
            config_payload=model.config.model_dump(mode="json"),
            label="es-best",
            origin="evolved",
            parent_sha256=record.sha256 if record else None,
        )
        self.repo.add_checkpoint(checkpoint, handle.architecture_hash)
        archive_id = self.repo.add_archive(config.model_id, {"history": result["history"], "archive": result["archive"]})
        self.repo.update_run(
            run.id,
            summary={**run.summary, "checkpoint_sha256": checkpoint.sha256, "archive_id": archive_id},
        )
        return {
            "run_id": run.id,
            "checkpoint": checkpoint.model_dump(mode="json"),
            "archive_id": archive_id,
            **{k: v for k, v in result.items() if k != "best_parameters"},
            "best_parameters_sha256": result["best_parameters_sha256"],
        }

    def generate(self, config: GeneratorConfig) -> dict[str, Any]:
        model = self.repo.get_model(config.target_model_id)
        if model.config.family != "mlp":
            raise ValidationFailure(
                "the hypernetwork generator currently targets MLP models only",
                hint="this limitation is intentional and documented in docs/STATUS.md",
            )
        handle, record = self.handle_for(config.target_model_id, None)
        report = train_hypernetwork(model.config, config)
        sha, path = write_artifact(
            self.workspace, json.dumps(report, indent=2, default=str).encode("utf-8"), ".json"
        )
        run = RunRecord(
            id=new_id("run"),
            kind="generation",
            model_id=config.target_model_id,
            state=RunState.COMPLETED,
            config=config.model_dump(mode="json"),
            summary={
                "results": report["results"],
                "generator": report["generator"],
                "claim_boundary": report["claim_boundary"],
                "report_sha256": sha,
            },
            provenance=Provenance(
                engine_version=ENGINE_VERSION,
                python=platform.python_version(),
                torch=torch_module().__version__,
                platform=platform.platform(),
                seeds={"generator": config.seed},
                architecture_hash=model.architecture_hash,
            ),
            finished_at=utcnow(),
        )
        self.repo.add_run(run)

        checkpoint_payload = None
        predictions = report.get("predictions", {})
        if predictions:
            first_key = sorted(predictions)[0]
            vector = np.asarray(predictions[first_key], dtype=np.float32)
            from .evolution import assign_parameters

            assign_parameters(handle.module, vector)
            record_out = save_checkpoint(
                self.workspace,
                model_id=config.target_model_id,
                module=handle.module,
                architecture_hash=handle.architecture_hash,
                config_payload=model.config.model_dump(mode="json"),
                label=f"generated:task{first_key}",
                origin="generated",
            )
            self.repo.add_checkpoint(record_out, handle.architecture_hash)
            checkpoint_payload = record_out.model_dump(mode="json")

        return {"run_id": run.id, "report_sha256": sha, "report_path": str(path), "checkpoint": checkpoint_payload, **report["results"]}

    # -- export ------------------------------------------------------------------------------
    def export(self, model_id: str, *, fmt: str = "onnx", checkpoint: str = "latest") -> dict[str, Any]:
        handle, record = self.handle_for(model_id, checkpoint)
        if fmt != "onnx":
            raise ValidationFailure(f"unsupported export format '{fmt}'", hint="supported: onnx")
        target = self.workspace.exports / f"{model_id}-{record.sha256[:8] if record else 'init'}.onnx"
        info = export_onnx(handle, target)
        info["sha256"] = __import__("hashlib").sha256(Path(target).read_bytes()).hexdigest()
        info["graph"] = inspect_onnx(target)
        run = RunRecord(
            id=new_id("run"),
            kind="export",
            model_id=model_id,
            state=RunState.COMPLETED,
            config={"format": fmt, "checkpoint": record.sha256 if record else None},
            summary=info,
            provenance=Provenance(
                engine_version=ENGINE_VERSION,
                python=platform.python_version(),
                torch=torch_module().__version__,
                platform=platform.platform(),
                architecture_hash=handle.architecture_hash,
                base_checkpoint_sha256=record.sha256 if record else None,
            ),
            finished_at=utcnow(),
        )
        self.repo.add_run(run)
        return {"run_id": run.id, **info}

    # -- history -----------------------------------------------------------------------------
    def history(self, *, model_id: str | None = None, kind: str | None = None, limit: int = 50):
        return [run.model_dump(mode="json") for run in self.repo.list_runs(model_id=model_id, kind=kind, limit=limit)]

    def timeline(self, limit: int = 100) -> list[dict[str, Any]]:
        events = self.repo.list_events(limit=limit)
        runs = [run.model_dump(mode="json") for run in self.repo.list_runs(limit=limit)]
        return [{"type": "event", **event} for event in events] + [{"type": "run", **run} for run in runs]

    def summary(self) -> dict[str, Any]:
        models = self.repo.list_models()
        runs = self.repo.list_runs(limit=500)
        evaluations = [run for run in runs if run.kind == "evaluation"]
        return {
            "engine_version": __version__,
            "workspace": str(self.workspace.root),
            "models": len(models),
            "parameters_total": sum(model.parameter_count for model in models),
            "runs": len(runs),
            "evaluations": len(evaluations),
            "completed_runs": sum(1 for run in runs if run.state is RunState.COMPLETED),
            "invalid_runs": sum(1 for run in runs if run.state is RunState.INVALID),
            "recent_models": [model.model_dump(mode="json") for model in models[:5]],
            "hardware": {
                "os": self.hardware.os,
                "cpu": self.hardware.cpu.brand,
                "logical_cores": self.hardware.cpu.logical_cores,
                "ram_gb": round(self.hardware.memory.total_bytes / 1e9, 1),
                "train_backend": self.hardware.preferred_train_backend.name,
                "train_backend_usable": self.hardware.preferred_train_backend.usable,
            },
        }

    def close(self) -> None:
        self.repo.close()
