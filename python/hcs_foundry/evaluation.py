"""Evaluation suites.

Every metric carries its own definition string, its direction and an optional baseline, so the UI
can never present a bare number without meaning. A suite that raises produces a report with state
``invalid`` â€” never a score of zero (ADR-013).
"""

from __future__ import annotations

import math
import platform
import time
from typing import Any, Protocol

import numpy as np

from .errors import NotFoundError
from .hashing import new_id
from .models.base import ModelHandle, torch_module
from .schemas import EvalMetric, EvalReport, MlpConfig, Provenance, RunState, TransformerConfig
from .training import datasets

EVALUATOR_VERSION = "1"


class Suite(Protocol):
    id: str
    version: str
    description: str
    families: tuple[str, ...]

    def run(self, handle: ModelHandle, *, seed: int, batches: int) -> tuple[list[EvalMetric], list[dict[str, Any]], list[str]]: ...


class TextSmokeSuite:
    id = "text-smoke"
    version = "1"
    description = (
        "Held-out next-token loss, perplexity and accuracy on the bundled character corpus. "
        "Measures learnability of this synthetic corpus only."
    )
    families = ("transformer-lm",)

    def run(self, handle: ModelHandle, *, seed: int, batches: int) -> tuple[list[EvalMetric], list[dict[str, Any]], list[str]]:
        torch = torch_module()
        config: TransformerConfig = handle.config
        module = handle.module
        module.eval()
        losses: list[float] = []
        correct = 0
        total = 0
        per_example: list[dict[str, Any]] = []
        started = time.perf_counter()
        with torch.no_grad():
            for index in range(batches):
                xs, ys = datasets.text_batch(
                    config,
                    batch_size=8,
                    block_size=min(16, config.context_length),
                    seed=seed + index,
                    split="val",
                )
                x = torch.from_numpy(xs).long()
                y = torch.from_numpy(ys).long()
                logits, loss = module(x, y)
                losses.append(float(loss))
                predictions = logits.argmax(dim=-1)
                correct += int((predictions == y).sum())
                total += int(y.numel())
                per_example.append(
                    {
                        "batch": index,
                        "loss": round(float(loss), 5),
                        "accuracy": round(float((predictions == y).float().mean()), 5),
                    }
                )
        elapsed = time.perf_counter() - started
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        accuracy = correct / total if total else 0.0
        metrics = [
            EvalMetric(
                name="val_loss",
                value=mean_loss,
                higher_is_better=False,
                definition="mean cross-entropy over held-out character windows (synthetic corpus)",
                baseline=math.log(config.vocab_size),
            ),
            EvalMetric(
                name="perplexity",
                value=float(math.exp(min(mean_loss, 50.0))),
                higher_is_better=False,
                definition="exp(val_loss), clamped for display",
                baseline=float(config.vocab_size),
            ),
            EvalMetric(
                name="next_token_accuracy",
                value=accuracy,
                higher_is_better=True,
                unit="fraction",
                definition="argmax accuracy on the same held-out windows",
                baseline=1.0 / config.vocab_size,
            ),
            EvalMetric(
                name="tokens_evaluated",
                value=float(total),
                higher_is_better=True,
                unit="tokens",
                definition="number of predicted tokens in the evaluation",
            ),
            EvalMetric(
                name="seconds",
                value=round(elapsed, 4),
                higher_is_better=False,
                unit="s",
                definition="wall-clock evaluation time on this machine",
            ),
        ]
        notes = [
            "baseline is a uniform predictor over the vocabulary, not a strong reference model",
            "synthetic corpus: results say nothing about general language ability",
        ]
        return metrics, per_example, notes


class ShapesSmokeSuite:
    id = "shapes-smoke"
    version = "1"
    description = "Held-out accuracy on the synthetic classification task used for MLP training."
    families = ("mlp",)

    def run(self, handle: ModelHandle, *, seed: int, batches: int) -> tuple[list[EvalMetric], list[dict[str, Any]], list[str]]:
        torch = torch_module()
        config: MlpConfig = handle.config
        module = handle.module
        module.eval()
        correct = 0
        total = 0
        losses: list[float] = []
        per_example: list[dict[str, Any]] = []
        started = time.perf_counter()
        with torch.no_grad():
            for index in range(batches):
                xs, ys = datasets.mlp_batch(config, batch_size=64, seed=seed + index, split="val")
                x = torch.from_numpy(xs)
                y = torch.from_numpy(ys).long()
                logits = module(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                losses.append(float(loss))
                predictions = logits.argmax(dim=-1)
                batch_correct = int((predictions == y).sum())
                correct += batch_correct
                total += int(y.numel())
                per_example.append(
                    {"batch": index, "loss": round(float(loss), 5), "correct": batch_correct, "n": int(y.numel())}
                )
        elapsed = time.perf_counter() - started
        accuracy = correct / total if total else 0.0
        chance = 1.0 / config.output_dim
        metrics = [
            EvalMetric(
                name="val_accuracy",
                value=accuracy,
                definition="held-out accuracy on the synthetic task (10% label noise)",
                baseline=chance,
            ),
            EvalMetric(
                name="val_loss",
                value=float(np.mean(losses)) if losses else float("nan"),
                higher_is_better=False,
                definition="mean cross-entropy on held-out batches",
                baseline=math.log(config.output_dim),
            ),
            EvalMetric(
                name="examples_evaluated",
                value=float(total),
                unit="examples",
                definition="number of held-out examples",
            ),
            EvalMetric(
                name="seconds",
                value=round(elapsed, 4),
                higher_is_better=False,
                unit="s",
                definition="wall-clock evaluation time on this machine",
            ),
        ]
        notes = [
            "label noise caps reachable accuracy below 1.0 by construction",
            "synthetic task: a high score does not imply generalisation to real data",
        ]
        return metrics, per_example, notes


class WeightSanitySuite:
    id = "weight-sanity"
    version = "1"
    description = "Finite-value, magnitude and norm checks. Available for every family."
    families = ("mlp", "transformer-lm")

    def run(self, handle: ModelHandle, *, seed: int, batches: int) -> tuple[list[EvalMetric], list[dict[str, Any]], list[str]]:
        torch = torch_module()
        state = handle.module.state_dict()
        finite = 0
        total = 0
        max_abs = 0.0
        norms: dict[str, float] = {}
        for name, tensor in state.items():
            flat = tensor.detach().float().reshape(-1)
            total += flat.numel()
            finite += int(torch.isfinite(flat).sum())
            if flat.numel():
                max_abs = max(max_abs, float(flat.abs().max()))
                norms[name] = float(torch.linalg.vector_norm(flat))
        metrics = [
            EvalMetric(
                name="finite_fraction",
                value=finite / total if total else 1.0,
                definition="fraction of finite values across all tensors",
                baseline=1.0,
            ),
            EvalMetric(
                name="max_abs",
                value=max_abs,
                higher_is_better=False,
                definition="largest absolute value in any tensor",
            ),
            EvalMetric(
                name="total_norm",
                value=float(math.sqrt(sum(v * v for v in norms.values()))),
                higher_is_better=False,
                definition="L2 norm across all tensors",
            ),
            EvalMetric(
                name="tensor_count",
                value=float(len(state)),
                unit="tensors",
                definition="number of tensors in the checkpoint",
            ),
        ]
        per_example = [{"tensor": name, "norm": round(value, 6)} for name, value in sorted(norms.items())]
        return metrics, per_example, ["run this suite after any patch or import"]


SUITES: dict[str, Suite] = {
    TextSmokeSuite.id: TextSmokeSuite(),
    ShapesSmokeSuite.id: ShapesSmokeSuite(),
    WeightSanitySuite.id: WeightSanitySuite(),
}


def list_suites() -> list[dict[str, Any]]:
    return [
        {
            "id": suite.id,
            "version": suite.version,
            "description": suite.description,
            "families": list(suite.families),
        }
        for suite in SUITES.values()
    ]


def get_suite(suite_id: str) -> Suite:
    suite = SUITES.get(suite_id)
    if suite is None:
        raise NotFoundError(
            f"unknown evaluation suite '{suite_id}'",
            hint=f"available suites: {sorted(SUITES)}",
        )
    return suite


def evaluate(
    handle: ModelHandle,
    *,
    suite_id: str,
    checkpoint_sha256: str,
    seed: int = 1234,
    batches: int = 4,
) -> EvalReport:
    """Run one suite. Exceptions are converted into an ``invalid`` report with the reason."""
    suite = get_suite(suite_id)
    if handle.family not in suite.families:
        raise NotFoundError(
            f"suite '{suite_id}' does not support family '{handle.family}'",
            hint=f"supported families: {list(suite.families)}",
        )
    provenance = Provenance(
        python=platform.python_version(),
        torch=getattr(torch_module(), "__version__", None),
        platform=platform.platform(),
        device=handle.device(),
        seeds={"evaluation": seed},
        architecture_hash=handle.architecture_hash,
        base_checkpoint_sha256=checkpoint_sha256,
        evaluator_version=EVALUATOR_VERSION,
    )
    try:
        metrics, per_example, notes = suite.run(handle, seed=seed, batches=batches)
        state = RunState.COMPLETED
    except Exception as exc:
        metrics, per_example, notes = [], [], [f"evaluator error: {type(exc).__name__}: {exc}"]
        state = RunState.INVALID
    return EvalReport(
        run_id=new_id("eval"),
        suite_id=suite_id,
        suite_version=suite.version,
        model_id="",
        checkpoint_sha256=checkpoint_sha256,
        state=state,
        metrics=metrics,
        per_example=per_example[:200],
        provenance=provenance,
        notes=notes,
    )

