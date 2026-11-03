"""Supervised training: real backpropagation, AdamW/SGD, checkpoint/resume, honest telemetry.

Lifecycle (AGENTS.md §7.1)::

    validate → estimate → prepare → dry_run → train → validate → checkpoint → summarize

Terminal states are exhaustive and never coerced: a diverged run is ``failed``, a cancelled run is
``cancelled``, and a run whose evaluator crashed is ``invalid`` (see evaluation).
"""

from __future__ import annotations

import math
import platform
import threading
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np

from ..checkpoints import load_state, save_checkpoint
from ..errors import BudgetExceeded, ValidationFailure
from ..hardware import ENGINE_VERSION
from ..hashing import new_id
from ..models.base import ModelHandle, torch_module
from ..schemas import (
    MetricPoint,
    MlpConfig,
    Provenance,
    ResourceEstimate,
    RunRecord,
    RunState,
    TrainingConfig,
    TransformerConfig,
)
from ..storage import Repository, Workspace
from . import datasets

ProgressHook = Callable[[MetricPoint], None]


class TrainerControl:
    """Cooperative pause/cancel used by the CLI, API and MCP server."""

    def __init__(self) -> None:
        self._pause = threading.Event()
        self._cancel = threading.Event()

    def request_pause(self) -> None:
        self._pause.set()

    def request_resume(self) -> None:
        self._pause.clear()

    def request_cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def wait_if_paused(self, deadline: float) -> bool:
        """Block while paused. Returns False if the run must stop."""
        while self.paused and not self.cancel_requested:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return not self.cancel_requested


def _lr_at(step: int, total: int, base_lr: float, warmup_fraction: float = 0.1) -> float:
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def _param_groups(module: Any, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not name.endswith("embedding") or param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


class SupervisedTrainer:
    def __init__(
        self,
        workspace: Workspace,
        repository: Repository,
        handle: ModelHandle,
        config: TrainingConfig,
        *,
        dataset: Any,
        base_checkpoint: Any | None = None,
        control: TrainerControl | None = None,
        progress: ProgressHook | None = None,
        profile: Any | None = None,
        run_id: str | None = None,
    ):
        self.workspace = workspace
        self.repo = repository
        self.handle = handle
        self.config = config
        self.dataset = dataset
        self.base_checkpoint = base_checkpoint
        self.control = control or TrainerControl()
        self.progress = progress
        self.device_name = "cpu"
        self.profile = profile
        self.run_id = run_id
        self.dry_run_report: dict[str, Any] = {}

    # -- lifecycle steps ---------------------------------------------------------------------
    def estimate(self) -> ResourceEstimate:
        family_adapter = self._family()
        return family_adapter.estimate_resources(
            self.handle.config,
            parameter_count=self.handle.parameter_count,
            train=True,
            batch_size=self.config.batch_size,
            block_size=min(self.config.block_size, getattr(self.handle.config, "context_length", self.config.block_size)),
            optimizer=self.config.optimizer.name,
        )

    def validate(self) -> None:
        if self.config.steps < 1:
            raise ValidationFailure("steps must be >= 1")
        estimate = self.estimate()
        available = None
        if self.profile is not None and getattr(self.profile, "memory", None):
            available = self.profile.memory.available_bytes
        if available and not estimate.fits_budget:
            raise BudgetExceeded(
                "the estimated training footprint does not fit the available memory",
                hint="reduce batch_size or model size, or raise the memory budget deliberately",
                details={"estimate_bytes": estimate.total_train_bytes, "available_bytes": available},
            )
        if isinstance(self.handle.config, MlpConfig) and self.dataset.id != "tiny-shapes":
            raise ValidationFailure("MLP models train on 'tiny-shapes'")
        if isinstance(self.handle.config, TransformerConfig) and self.dataset.id != "tiny-text":
            raise ValidationFailure("transformer models train on 'tiny-text'")

    def dry_run(self) -> float:
        """One batch forward/backward to prove the graph works before committing to a run."""
        module = self.handle.module
        module.train()
        batch = self._batch(step=0, split="train")
        loss = self._loss(batch)
        loss.backward()
        grad_count = sum(1 for p in module.parameters() if p.grad is not None)
        grad_norm = float(
            sum(float(p.grad.detach().pow(2).sum()) for p in module.parameters() if p.grad is not None) ** 0.5
        )
        module.zero_grad(set_to_none=True)
        loss_value = float(loss.detach())
        self.dry_run_report = {
            "loss": loss_value,
            "parameters_with_grad": grad_count,
            "parameters_total": sum(1 for _ in module.parameters()),
            "grad_norm": grad_norm,
            "finite": math.isfinite(loss_value),
        }
        if not math.isfinite(loss_value):
            raise ValidationFailure("dry-run loss is not finite; the model or data is broken")
        if grad_count == 0:
            raise ValidationFailure("no gradients reached any parameter")
        return loss_value

    # -- batching ----------------------------------------------------------------------------
    def _batch(self, *, step: int, split: str) -> dict[str, Any]:
        seed = self.config.seed * 1_000_003 + step
        if isinstance(self.handle.config, TransformerConfig):
            xs, ys = datasets.text_batch(
                self.handle.config,
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                seed=seed,
                split=split,
            )
            torch = torch_module()
            return {
                "x": torch.from_numpy(xs).long(),
                "y": torch.from_numpy(ys).long(),
                "tokens": int(xs.size),
            }
        xs, ys = datasets.mlp_batch(
            self.handle.config, batch_size=self.config.batch_size, seed=seed, split=split
        )
        torch = torch_module()
        return {
            "x": torch.from_numpy(xs),
            "y": torch.from_numpy(ys).long(),
            "tokens": 0,
        }

    def _loss(self, batch: dict[str, Any]) -> Any:
        torch = torch_module()
        module = self.handle.module
        if isinstance(self.handle.config, TransformerConfig):
            _, loss = module(batch["x"], batch["y"])
            return loss
        logits = module(batch["x"])
        return torch.nn.functional.cross_entropy(logits, batch["y"])

    def _evaluate(self, *, batches: int = 4, seed: int = 999_999) -> float:
        torch = torch_module()
        module = self.handle.module
        was_training = module.training
        module.eval()
        total = 0.0
        with torch.no_grad():
            for index in range(batches):
                batch = self._batch(step=seed + index, split="val")
                total += float(self._loss(batch))
        module.train(was_training)
        return total / max(1, batches)

    def _family(self) -> Any:
        from ..models.registry import get_family

        return get_family(self.handle.family)

    # -- main loop ---------------------------------------------------------------------------
    def run(self) -> RunRecord:
        torch = torch_module()
        run = RunRecord(
            id=self.run_id or new_id("run"),
            kind="training",
            model_id=self.config.model_id,
            state=RunState.QUEUED,
            config=self.config.model_dump(mode="json"),
            provenance=Provenance(
                engine_version=ENGINE_VERSION,
                python=platform.python_version(),
                torch=torch.__version__,
                platform=platform.platform(),
                device=self.device_name,
                seeds={"training": self.config.seed, "model": int(self.handle.config.seed)},
                dataset_sha256=self.dataset.sha256,
                architecture_hash=self.handle.architecture_hash,
                base_checkpoint_sha256=self.base_checkpoint.sha256 if self.base_checkpoint else None,
            ),
        )
        self.validate()
        self.dry_run()
        run.summary["dry_run"] = dict(self.dry_run_report)
        self.repo.add_run(run)

        if self.base_checkpoint is not None:
            load_state(self.handle, self.base_checkpoint.path)

        module = self.handle.module
        try:
            module.to(self.device_name)
        except Exception as exc:  # pragma: no cover - device dependent
            self.repo.update_run(run.id, state=RunState.FAILED, error={"message": str(exc)}, finished=True)
            raise

        optimizer_cls = torch.optim.AdamW if self.config.optimizer.name == "adamw" else torch.optim.SGD
        kwargs: dict[str, Any] = {"lr": self.config.optimizer.lr}
        if self.config.optimizer.name == "adamw":
            kwargs.update(
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                eps=1e-8,
            )
        else:
            kwargs.update(momentum=self.config.optimizer.momentum)
        optimizer = optimizer_cls(_param_groups(module, self.config.optimizer.weight_decay), **kwargs)

        self.repo.update_run(run.id, state=RunState.RUNNING)
        module.train()
        tracemalloc.start()
        start = time.monotonic()
        deadline = start + self.config.max_seconds
        tokens_seen = 0
        losses: list[float] = []
        checkpoints: list[str] = []
        state = RunState.COMPLETED
        error: dict[str, Any] | None = None
        metric_interval = max(1, self.config.steps // 400)

        try:
            for step in range(self.config.steps):
                if not self.control.wait_if_paused(deadline):
                    state = RunState.CANCELLED if self.control.cancel_requested else RunState.INTERRUPTED
                    break
                if time.monotonic() > deadline:
                    state = RunState.INTERRUPTED
                    run.summary["stop_reason"] = "max_seconds exceeded"
                    break

                lr = _lr_at(step, self.config.steps, self.config.optimizer.lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                batch = self._batch(step=step, split="train")
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch)
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    state = RunState.FAILED
                    error = {"message": "non-finite loss", "step": step, "loss": loss_value}
                    break
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in module.parameters() if p.grad is not None],
                    self.config.optimizer.grad_clip,
                )
                optimizer.step()
                losses.append(loss_value)
                tokens_seen += int(batch["tokens"])

                if step % metric_interval == 0 or step == self.config.steps - 1:
                    elapsed = time.monotonic() - start
                    point = MetricPoint(
                        step=step,
                        loss=loss_value,
                        lr=lr,
                        grad_norm=float(grad_norm),
                        tokens=tokens_seen,
                        elapsed_s=round(elapsed, 4),
                        peak_memory_bytes=tracemalloc.get_traced_memory()[1],
                    )
                    self.repo.add_metric(run.id, step, point.model_dump())
                    run.metrics.append(point)
                    if self.progress:
                        self.progress(point)

                if (step + 1) % self.config.eval_every == 0 or step == self.config.steps - 1:
                    val_loss = self._evaluate()
                    point = MetricPoint(
                        step=step,
                        loss=loss_value,
                        lr=lr,
                        grad_norm=float(grad_norm),
                        tokens=tokens_seen,
                        elapsed_s=round(time.monotonic() - start, 4),
                        peak_memory_bytes=tracemalloc.get_traced_memory()[1],
                        val_loss=val_loss,
                    )
                    self.repo.add_metric(run.id, step + 1_000_000, point.model_dump())
                    run.metrics.append(point)
                    module.train()

                if (step + 1) % self.config.checkpoint_every == 0:
                    record = self._checkpoint(label=f"step-{step + 1}", run=run)
                    checkpoints.append(record.id)
                    self.repo.add_checkpoint(record, self.handle.architecture_hash)

        except Exception as exc:
            state = RunState.FAILED
            error = {"message": str(exc), "type": type(exc).__name__}

        elapsed = time.monotonic() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        final_checkpoint = None
        if state in (RunState.COMPLETED, RunState.INTERRUPTED):
            final_checkpoint = self._checkpoint(label="final", run=run)
            checkpoints.append(final_checkpoint.id)
            self.repo.add_checkpoint(final_checkpoint, self.handle.architecture_hash)

        val_loss = None
        if state is RunState.COMPLETED:
            val_loss = self._evaluate()

        summary = {
            **run.summary,
            "final_loss": losses[-1] if losses else None,
            "mean_loss_last_10": float(np.mean(losses[-10:])) if losses else None,
            "val_loss": val_loss,
            "steps_completed": len(losses),
            "tokens": tokens_seen,
            "seconds": round(elapsed, 3),
            "steps_per_second": round(len(losses) / elapsed, 3) if elapsed > 0 else 0.0,
            "peak_python_heap_bytes": int(peak),
            "parameters": self.handle.parameter_count,
            "checkpoint_ids": checkpoints,
            "final_checkpoint_id": final_checkpoint.id if final_checkpoint else None,
            "device": self.device_name,
            "note": "synthetic task: measures pipeline correctness, not capability",
        }
        self.repo.update_run(run.id, state=state, summary=summary, error=error, finished=True)
        finished = self.repo.get_run(run.id)
        self.repo.log_event(
            "info" if state is RunState.COMPLETED else "warning",
            f"training:{run.id}",
            f"training run {state.value}",
            {"steps": len(losses), "final_loss": summary["final_loss"]},
        )
        return finished

    def _checkpoint(self, *, label: str, run: RunRecord) -> Any:
        current = self.repo.latest_checkpoint(self.config.model_id)
        return save_checkpoint(
            self.workspace,
            model_id=self.config.model_id,
            module=self.handle.module,
            architecture_hash=self.handle.architecture_hash,
            config_payload=self.handle.config.model_dump(mode="json"),
            label=label,
            origin="training",
            parent_sha256=current.sha256 if current else None,
        )
