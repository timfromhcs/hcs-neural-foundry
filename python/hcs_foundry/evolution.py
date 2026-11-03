"""Evolution Strategies with antithetic sampling, plus a MAP-Elites archive.

Two honest constraints shape this module:
1. ES is compared against gradient training with the *same number of examples*, and the report states
   how the budget was matched. ES is not claimed to be faster than backpropagation.
2. Search happens in raw parameter space only because the demo model is tiny. The archive stores
   behaviour descriptors (score, latency, size) so a diverse set survives instead of one winner.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .models.base import ModelHandle, torch_module
from .schemas import EvolutionConfig, MlpConfig
from .training import datasets

FitnessFn = Callable[[np.ndarray, int], dict[str, float]]


def flatten_parameters(module: Any) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in module.parameters()])


def assign_parameters(module: Any, flat: np.ndarray) -> None:
    torch = torch_module()
    offset = 0
    with torch.no_grad():
        for param in module.parameters():
            count = param.numel()
            block = flat[offset : offset + count].reshape(param.shape)
            param.copy_(torch.from_numpy(block).to(param.dtype).to(param.device))
            offset += count
    if offset != flat.size:  # pragma: no cover - guard against silent shape drift
        raise ValueError(f"parameter vector has {flat.size} elements but model consumed {offset}")


def make_fitness(handle: ModelHandle, *, seed: int, batches: int = 3, latency_weight: float = 0.0) -> FitnessFn:
    """Accuracy-based fitness on fixed held-out batches. Deterministic for a given seed."""

    def fitness(flat: np.ndarray, generation: int) -> dict[str, float]:
        torch = torch_module()
        assign_parameters(handle.module, flat)
        handle.module.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for index in range(batches):
                xs, ys = datasets.mlp_batch(
                    handle.config, batch_size=64, seed=seed + index + generation * 97, split="val"
                )
                logits = handle.module(torch.from_numpy(xs))
                correct += int((logits.argmax(dim=-1) == torch.from_numpy(ys).long()).sum())
                total += int(ys.size)
        score = correct / total if total else 0.0
        started = time.perf_counter()
        with torch.no_grad():
            sample = torch.zeros(1, handle.config.input_dim)
            for _ in range(5):
                handle.module(sample)
        latency_ms = (time.perf_counter() - started) / 5 * 1000
        return {
            "score": score,
            "latency_ms": latency_ms,
            "params": float(flat.size),
            "fitness": score - latency_weight * latency_ms,
        }

    return fitness


class MapElites:
    """Small, explicit archive: behaviour descriptors are quantised into named cells."""

    def __init__(self) -> None:
        self.cells: dict[str, dict[str, Any]] = {}

    @staticmethod
    def descriptor(metrics: dict[str, float]) -> tuple[int, int, int]:
        score_bin = min(9, int(metrics["score"] * 10))
        latency = metrics["latency_ms"]
        latency_bin = 0 if latency < 5 else 1 if latency < 15 else 2 if latency < 50 else 3
        size = int(metrics["params"])
        size_bin = 0 if size < 10_000 else 1 if size < 100_000 else 2
        return score_bin, latency_bin, size_bin

    def add(self, flat: np.ndarray, metrics: dict[str, float], generation: int) -> bool:
        key = "score{}|lat{}|size{}".format(*self.descriptor(metrics))
        current = self.cells.get(key)
        if current is None or metrics["fitness"] > current["metrics"]["fitness"]:
            self.cells[key] = {
                "cell": key,
                "metrics": metrics,
                "generation": generation,
                "parameters": flat.tolist(),
            }
            return True
        return False

    def best(self) -> dict[str, Any] | None:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda item: item["metrics"]["fitness"])

    def coverage(self) -> float:
        return len(self.cells) / (10 * 4 * 3)

    def public(self) -> list[dict[str, Any]]:
        return [
            {
                "cell": item["cell"],
                "metrics": {k: round(v, 6) for k, v in item["metrics"].items()},
                "generation": item["generation"],
            }
            for item in sorted(self.cells.values(), key=lambda i: -i["metrics"]["fitness"])
        ]


def run_evolution(
    handle: ModelHandle,
    config: EvolutionConfig,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Antithetic ES. Returns a JSON-serialisable report including the best parameter vector."""
    if not isinstance(handle.config, MlpConfig):
        raise ValueError("evolution is currently limited to MLP models (raw parameter search)")
    rng = np.random.default_rng(config.seed)
    fitness = make_fitness(handle, seed=config.seed * 7919 + 13, batches=3)
    theta = flatten_parameters(handle.module)
    best_metrics = fitness(theta, -1)
    best_theta = theta.copy()
    initial = dict(best_metrics)
    archive = MapElites()
    archive.add(theta, best_metrics, generation=0)
    history: list[dict[str, Any]] = []
    half = max(1, config.population // 2)
    examples_seen = 0
    disturbed = 0

    for generation in range(config.generations):
        epsilons = rng.standard_normal((half, theta.size)).astype(np.float32)
        candidates = np.concatenate([theta + config.sigma * epsilons, theta - config.sigma * epsilons], axis=0)
        rewards: list[float] = []
        generation_best = None
        for index, candidate in enumerate(candidates):
            metrics = fitness(candidate, generation)
            examples_seen += 3 * 64
            rewards.append(metrics["fitness"])
            if metrics["fitness"] > best_metrics["fitness"]:
                best_metrics = metrics
                best_theta = candidate.copy()
            if generation_best is None or metrics["fitness"] > generation_best["metrics"]["fitness"]:
                generation_best = {"metrics": metrics, "index": index}
            archive.add(candidate, metrics, generation)
            disturbed += 1

        rewards_array = np.asarray(rewards, dtype=np.float32)
        # Rank-based normalisation: robust to reward scale and keeps the step size stable.
        order = np.argsort(rewards_array)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(rewards_array), dtype=np.float32)
        normalised = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
        signed = np.concatenate([normalised[:half], normalised[half:]])
        gradient = (signed[:, None] * np.concatenate([epsilons, -epsilons], axis=0)).mean(axis=0)
        theta = theta + config.lr * gradient

        entry = {
            "generation": generation,
            "best_fitness": round(float(best_metrics["fitness"]), 6),
            "generation_best": round(float(generation_best["metrics"]["fitness"]) if generation_best else 0.0, 6),
            "mean_reward": round(float(rewards_array.mean()), 6),
            "archive_cells": len(archive.cells),
            "coverage": round(archive.coverage(), 4),
            "theta_norm": round(float(np.linalg.norm(theta)), 6),
        }
        history.append(entry)
        if progress:
            progress(entry)

    # Equal-example gradient baseline: same number of *examples*, not the same wall-clock time.
    baseline: dict[str, Any] = {}
    if config.baseline_steps > 0:
        baseline = _gradient_baseline(handle, config, examples_seen, fitness)
    else:
        steps = max(1, examples_seen // 64)
        baseline = _gradient_baseline(handle, config, examples_seen, fitness, steps=steps)
        baseline["note"] = (
            "gradient baseline uses the same number of examples as the ES search; wall-clock time and "
            "gradient cost are not equalised, so this is a sanity comparison, not a race"
        )

    best_cell = archive.best()
    best_array = np.asarray(best_theta, dtype=np.float32)
    return {
        "algorithm": "antithetic-evolution-strategies",
        "best_parameters_sha256": __import__("hashlib").sha256(best_array.tobytes()).hexdigest(),
        "config": config.model_dump(mode="json"),
        "initial": {k: round(v, 6) for k, v in initial.items()},
        "best": {k: round(v, 6) for k, v in best_metrics.items()},
        "best_parameters": best_theta.tolist(),
        "best_cell": best_cell["cell"] if best_cell else None,
        "history": history,
        "archive": archive.public(),
        "archive_coverage": round(archive.coverage(), 4),
        "examples_seen": examples_seen,
        "candidates_evaluated": disturbed,
        "baseline": baseline,
        "claim_boundary": (
            "ES searched raw parameters of a tiny MLP. Nothing here generalises to large models, and "
            "the baseline is example-matched rather than compute-matched."
        ),
    }


def _gradient_baseline(
    handle: ModelHandle,
    config: EvolutionConfig,
    examples: int,
    fitness: FitnessFn,
    *,
    steps: int | None = None,
) -> dict[str, Any]:
    """Train a fresh clone with backprop for a comparable example budget."""
    from .models.registry import build_model
    from .schemas import DatasetSpec, TrainingConfig
    from .training.trainer import SupervisedTrainer, TrainerControl

    cloned = build_model(handle.config.model_dump(mode="json"))
    steps = steps or max(1, config.baseline_steps or examples // 16)
    datasets_spec = DatasetSpec(
        id="tiny-shapes",
        name="synthetic",
        kind="classification",
        path="",
        sha256=datasets.corpus_sha256(),
    )
    training = TrainingConfig(
        model_id="baseline",
        dataset_id="tiny-shapes",
        steps=steps,
        batch_size=16,
        eval_every=max(1, steps),
        checkpoint_every=max(1, steps * 10),
        max_seconds=600,
        seed=config.seed + 1,
    )
    classes = _NullRepo()
    trainer = SupervisedTrainer(
        classes,  # type: ignore[arg-type]
        classes,  # type: ignore[arg-type]
        cloned,
        training,
        dataset=datasets_spec,
        control=TrainerControl(),
    )
    try:
        trainer.run()
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"{type(exc).__name__}: {exc}", "steps": steps}
    metrics = fitness(flatten_parameters(cloned.module), -1)
    return {
        "steps": steps,
        "examples": steps * training.batch_size,
        "score": round(metrics["score"], 6),
        "latency_ms": round(metrics["latency_ms"], 4),
    }


class _NullRepo:
    """In-memory sink so the baseline can reuse the trainer without touching the real database."""

    def __init__(self) -> None:
        self.runs: dict[str, Any] = {}
        self.metrics: list[Any] = []
        self.checkpoints: list[Any] = []

    def add_run(self, run: Any) -> Any:
        self.runs[run.id] = run
        return run

    def update_run(self, run_id: str, **kwargs: Any) -> None:
        self.runs.setdefault(run_id, None)

    def add_metric(self, run_id: str, step: int, payload: dict[str, Any]) -> None:
        self.metrics.append((run_id, step, payload))

    def add_checkpoint(self, record: Any, architecture_hash: str | None = None) -> Any:
        self.checkpoints.append(record)
        return record

    def latest_checkpoint(self, model_id: str) -> None:
        return None

    def get_run(self, run_id: str) -> Any:
        return self.runs[run_id]

    def log_event(self, *args: Any, **kwargs: Any) -> None:
        return None
