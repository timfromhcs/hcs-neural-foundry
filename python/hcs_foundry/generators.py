"""Hypernetwork weight generation for a fixed tiny MLP — an honest, narrow experiment.

What is actually implemented
----------------------------
A conditional hypernetwork maps a compact *task descriptor* to the full parameter vector of a fixed
architecture. It is trained on checkpoints produced by gradient training on a set of synthetic tasks,
then evaluated on **held-out tasks it never saw** (split by task seed, asserted in code).

What is deliberately *not* implemented
--------------------------------------
D2NWG-style diffusion and DeepWeightFlow-style flow matching. Reproducing them credibly requires a
real checkpoint corpus, canonicalisation/alignment (Git Re-Basin), and evaluation that this build
cannot support. They are declared ``unsupported`` rather than faked (see docs/STATUS.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ValidationFailure
from .training import datasets

# --------------------------------------------------------------------------------------------------
# Tiny numpy MLP used as the generator (no torch dependency, easy to serialise)
# --------------------------------------------------------------------------------------------------


class NumpyMlp:
    def __init__(self, sizes: list[int], seed: int):
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for index in range(len(sizes) - 1):
            scale = 1.0 / math.sqrt(sizes[index])
            self.weights.append((rng.standard_normal((sizes[index], sizes[index + 1])) * scale).astype(np.float32))
            self.biases.append(np.zeros(sizes[index + 1], dtype=np.float32))

    def forward(self, x: np.ndarray, *, cache: bool = False) -> Any:
        activations = [x]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ weight + bias
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
            activations.append(x)
        return (x, activations) if cache else x

    def parameters(self) -> list[np.ndarray]:
        return [*self.weights, *self.biases]


def _train_numpy_mlp(
    net: NumpyMlp,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    params = net.parameters()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    last_loss = float("nan")
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            xb, yb = x[batch], y[batch]
            out, activations = net.forward(xb, cache=True)
            # MSE with a linear output layer as a regression proxy for the parameter vector.
            error = out - yb
            last_loss = float((error**2).mean())
            grad = 2.0 * error / error.size
            for layer in range(len(net.weights) - 1, -1, -1):
                if layer < len(net.weights) - 1:
                    grad = grad * (activations[layer + 1] > 0)
                dw = activations[layer].T @ grad
                db = grad.sum(axis=0)
                step += 1
                gw, gb = dw, db
                for tensor, gradient, slot in (
                    (net.weights[layer], gw, layer),
                    (net.biases[layer], gb, layer + len(net.weights)),
                ):
                    m[slot] = beta1 * m[slot] + (1 - beta1) * gradient
                    v[slot] = beta2 * v[slot] + (1 - beta2) * gradient**2
                    m_hat = m[slot] / (1 - beta1**step)
                    v_hat = v[slot] / (1 - beta2**step)
                    tensor -= lr * m_hat / (np.sqrt(v_hat) + eps)
                grad = grad @ net.weights[layer].T
    return last_loss


# --------------------------------------------------------------------------------------------------
# Target model helpers (torch)
# --------------------------------------------------------------------------------------------------


def _train_target(
    config: Any, *, task_seed: int, steps: int, seed: int, lr: float = 5e-3
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gradient-train a fresh MLP on one synthetic task and return its flat parameters."""
    from .models.registry import build_model

    torch = __import__("torch")
    handle = build_model(config.model_dump(mode="json"))
    module = handle.module
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    module.train()
    final_loss = float("nan")
    for _ in range(steps):
        xs, ys = datasets.mlp_batch(
            config,
            batch_size=32,
            seed=int(rng.integers(0, 2**31 - 1)),
            task_seed=task_seed,
        )
        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys).long()
        optimizer.zero_grad(set_to_none=True)
        logits = module(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    flat = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in module.parameters()]).astype(np.float32)
    return flat, {"task_seed": task_seed, "steps": steps, "final_batch_loss": final_loss}


def _descriptor(config: Any, task_seed: int, latent_dim: int) -> np.ndarray:
    """Compact, deterministic descriptor of a task: random projection of the teacher matrix."""
    weights, bias = datasets._task_matrix(config.input_dim, config.output_dim, task_seed)  # noqa: SLF001
    raw = np.concatenate([weights.reshape(-1), bias])
    rng = np.random.default_rng(424242)
    projection = rng.standard_normal((raw.size, latent_dim)).astype(np.float32) / math.sqrt(raw.size)
    return np.tanh(raw @ projection)


@dataclass
class GeneratorReport:
    payload: dict[str, Any]
    parameters: np.ndarray
    task_seed: int
    samples: list[dict[str, Any]] = field(default_factory=list)


def train_hypernetwork(
    config: Any,
    generator_config: Any,
    *,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Train a conditional hypernetwork and evaluate it on held-out tasks."""
    if len(generator_config.corpus) < 3:
        raise ValidationFailure("generator corpus needs at least 3 tasks", hint="corpus = list of task seeds")
    task_seeds = [int(value) for value in generator_config.corpus]
    split = max(2, int(len(task_seeds) * 0.75))
    train_seeds, test_seeds = task_seeds[:split], task_seeds[split:]
    if set(train_seeds) & set(test_seeds):
        raise ValidationFailure("generator corpus leaks test tasks into the training split")

    latent = generator_config.latent_dim
    descriptors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    corpus_manifest: list[dict[str, Any]] = []
    for task_seed in train_seeds:
        flat, stats = _train_target(config, task_seed=task_seed, steps=120, seed=task_seed + 7)
        descriptors.append(_descriptor(config, task_seed, latent))
        targets.append(flat)
        corpus_manifest.append({"split": "train", **stats, "param_norm": round(float(np.linalg.norm(flat)), 5)})

    matrix = np.stack(targets)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-6
    normalized = (matrix - mean) / std

    net = NumpyMlp([latent, generator_config.hidden, generator_config.hidden, matrix.shape[1]], generator_config.seed)
    x = np.stack(descriptors)
    losses: list[float] = []
    for epoch in range(generator_config.epochs):
        loss = _train_numpy_mlp(
            net,
            x,
            normalized,
            epochs=1,
            lr=generator_config.lr,
            batch_size=min(8, len(x)),
            seed=generator_config.seed + epoch,
        )
        losses.append(round(loss, 6))
        if progress and (epoch % max(1, generator_config.epochs // 10) == 0):
            progress({"epoch": epoch, "loss": round(loss, 6)})

    train_predictions = net.forward(x) * std + mean
    train_mse = float(((train_predictions - matrix) ** 2).mean())

    from .models.registry import build_model

    samples: list[dict[str, Any]] = []
    random_scores: list[float] = []
    trained_scores: list[float] = []
    generated_scores: list[float] = []
    predictions: dict[str, list[float]] = {}
    for task_seed in test_seeds:
        descriptor = _descriptor(config, task_seed, latent)
        prediction = (net.forward(descriptor[None, :]) * std + mean)[0]
        handle = build_model(config.model_dump(mode="json"))
        generated_metrics = _score_flat(handle, prediction, config, task_seed)
        generated_scores.append(generated_metrics["accuracy"])

        random_handle = build_model(config.model_dump(mode="json"))
        random_metrics = _score_flat(
            random_handle,
            np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in random_handle.module.parameters()]),
            config,
            task_seed,
        )
        random_scores.append(random_metrics["accuracy"])

        trained_flat, trained_stats = _train_target(config, task_seed=task_seed, steps=120, seed=task_seed + 11)
        trained_handle = build_model(config.model_dump(mode="json"))
        trained_metrics = _score_flat(trained_handle, trained_flat, config, task_seed)
        trained_scores.append(trained_metrics["accuracy"])
        corpus_manifest.append({"split": "test", **trained_stats})

        samples.append(
            {
                "task_seed": task_seed,
                "generated_accuracy": round(generated_metrics["accuracy"], 6),
                "random_init_accuracy": round(random_metrics["accuracy"], 6),
                "gradient_trained_accuracy": round(trained_metrics["accuracy"], 6),
                "latency_ms": round(generated_metrics["latency_ms"], 4),
            }
        )
        predictions[str(task_seed)] = [round(float(v), 6) for v in prediction.tolist()]

    payload = {
        "method": "conditional-hypernetwork (MLP)",
        "status": "experimental",
        "target_config": config.model_dump(mode="json"),
        "generator": {
            "latent_dim": latent,
            "hidden": generator_config.hidden,
            "epochs": generator_config.epochs,
            "train_tasks": train_seeds,
            "test_tasks": test_seeds,
            "train_loss_curve": losses,
            "train_parameter_mse": round(train_mse, 6),
        },
        "corpus": corpus_manifest,
        "results": {
            "generated_mean_accuracy": round(float(np.mean(generated_scores)), 6),
            "random_init_mean_accuracy": round(float(np.mean(random_scores)), 6),
            "gradient_trained_mean_accuracy": round(float(np.mean(trained_scores)), 6),
            "samples": samples,
        },
        "leakage_control": "test tasks are disjoint from training tasks and asserted in code",
        "predictions": predictions,        "claim_boundary": (
            "One tiny MLP family, one synthetic task distribution, 3 held-out tasks. This is a "
            "feasibility demonstration of a weight-space generator, not evidence that generated weights "
            "beat gradient training. The gradient baseline uses the same number of steps, not the same "
            "compute."
        ),
    }
    return payload


def _score_flat(handle: Any, flat: np.ndarray, config: Any, task_seed: int) -> dict[str, float]:
    import time

    torch = __import__("torch")

    offset = 0
    with torch.no_grad():
        for param in handle.module.parameters():
            count = param.numel()
            block = flat[offset : offset + count].reshape(param.shape)
            param.copy_(torch.from_numpy(np.asarray(block, dtype=np.float32)))
            offset += count
    handle.module.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index in range(4):
            xs, ys = datasets.mlp_batch(
                config, batch_size=64, seed=555_000 + index, split="val", task_seed=task_seed
            )
            logits = handle.module(torch.from_numpy(xs))
            correct += int((logits.argmax(dim=-1) == torch.from_numpy(ys).long()).sum())
            total += int(ys.size)
    started = time.perf_counter()
    with torch.no_grad():
        sample = torch.zeros(1, config.input_dim)
        for _ in range(5):
            handle.module(sample)
    latency = (time.perf_counter() - started) / 5 * 1000
    return {"accuracy": correct / total if total else 0.0, "latency_ms": latency}


def serialize_generator(report: dict[str, Any], net: NumpyMlp | None = None) -> dict[str, Any]:
    """Return a JSON-safe summary (the trained numpy weights are saved separately as .npz)."""
    return {"report": report, "has_weights": net is not None}
