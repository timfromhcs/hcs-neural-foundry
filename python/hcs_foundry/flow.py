"""Conditional flow matching in weight space.

The pipeline, in order:

1. train a small corpus of MLPs on different tasks (gradient training, not generation);
2. **align** every checkpoint to the anchor, because permuted hidden units compute the same function
   but live in unrelated parameter locations;
3. train a conditional flow-matching model ``v(x_t, t, c)`` that transports Gaussian noise to a
   checkpoint's parameter vector, conditioned on a compact task descriptor;
4. sample on held-out tasks and score generated weights against random initialisation and gradient
   training.

Every number is reported with its corpus, and the claim boundary states what a 3-5 task corpus can and
cannot support. This is a reproduction *of the idea* at toy scale, not of the papers' benchmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .alignment import align_corpus
from .errors import ValidationFailure
from .generators import _descriptor, _score_flat, _train_target
from .training import datasets


def _weights_from_handle(handle: Any) -> dict[str, np.ndarray]:
    """Extract the one-hidden-layer MLP in ``(in, out)`` layout, which alignment expects.

    PyTorch stores ``layers.0.weight`` as ``(hidden, input)``; :mod:`hcs_foundry.alignment` works with
    ``(input, hidden)`` so that hidden units are columns.
    """
    state = handle.module.state_dict()
    required = ("layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias")
    extra_layers = [name for name in state if name.startswith("layers.2.")]
    missing = [name for name in required if name not in state or extra_layers]
    if missing:
        raise ValidationFailure(
            "flow matching needs exactly one hidden layer",
            hint="use hidden_dims=(width,) - exactly one hidden layer, e.g. the beginner-mlp preset",
            details={"missing": missing},
        )
    return {
        "w1": state["layers.0.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b1": state["layers.0.bias"].detach().cpu().numpy().astype(np.float32),
        "w2": state["layers.1.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b2": state["layers.1.bias"].detach().cpu().numpy().astype(np.float32),
    }


def to_torch_flat(weights: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten in PyTorch parameter order, which is what the engine's scorers consume."""
    return np.concatenate(
        [
            weights["w1"].T.reshape(-1),
            weights["b1"].reshape(-1),
            weights["w2"].T.reshape(-1),
            weights["b2"].reshape(-1),
        ]
    ).astype(np.float32)


def from_torch_flat(vector: np.ndarray, shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Inverse of :func:`to_torch_flat` for the same architecture."""
    offset = 0

    def take(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        size = int(np.prod(shape))
        block = vector[offset : offset + size].reshape(shape)
        offset += size
        return block.astype(np.float32)

    w1 = take((shapes["w1"][1], shapes["w1"][0])).T  # stored (hidden, in) -> (in, hidden)
    b1 = take((shapes["b1"][0],))
    w2 = take((shapes["w2"][1], shapes["w2"][0])).T
    b2 = take((shapes["b2"][0],))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


@dataclass
class FlowModel:
    """Small numpy MLP regressing a velocity field. Deterministic and serialisable."""

    sizes: tuple[int, ...]
    weights: list[np.ndarray] = field(default_factory=list)
    biases: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def create(cls, input_dim: int, hidden: int, output_dim: int, *, seed: int) -> FlowModel:
        rng = np.random.default_rng(seed)
        sizes = (input_dim, hidden, hidden, output_dim)
        weights = [
            (rng.standard_normal((sizes[i], sizes[i + 1])) / math.sqrt(sizes[i])).astype(np.float32)
            for i in range(len(sizes) - 1)
        ]
        biases = [np.zeros(sizes[i + 1], dtype=np.float32) for i in range(len(sizes) - 1)]
        return cls(sizes=tuple(sizes), weights=weights, biases=biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ weight + bias
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x

    def parameters(self) -> list[np.ndarray]:
        return [*self.weights, *self.biases]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sizes": list(self.sizes),
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FlowModel:
        return cls(
            sizes=tuple(payload["sizes"]),
            weights=[np.asarray(w, dtype=np.float32) for w in payload["weights"]],
            biases=[np.asarray(b, dtype=np.float32) for b in payload["biases"]],
        )


def _adam_step(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    state: dict[str, Any],
    *,
    lr: float,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
        m = state["m"][index] = beta1 * state["m"][index] + (1 - beta1) * grad
        v = state["v"][index] = beta2 * state["v"][index] + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_flow(
    targets: np.ndarray,
    descriptors: np.ndarray,
    *,
    hidden: int = 128,
    steps: int = 400,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    progress: Any | None = None,
) -> tuple[FlowModel, list[float]]:
    """Train the velocity field. Returns the model and the loss curve."""
    if targets.ndim != 2 or descriptors.ndim != 2:
        raise ValidationFailure("targets and descriptors must be 2-D")
    if len(targets) != len(descriptors):
        raise ValidationFailure("every checkpoint needs exactly one descriptor")
    if len(targets) < 2:
        raise ValidationFailure("flow matching needs at least 2 aligned checkpoints")

    rng = np.random.default_rng(seed)
    param_dim = targets.shape[1]
    cond_dim = descriptors.shape[1]
    model = FlowModel.create(param_dim + 1 + cond_dim, hidden, param_dim, seed=seed)
    state = {
        "m": [np.zeros_like(p) for p in model.parameters()],
        "v": [np.zeros_like(p) for p in model.parameters()],
    }
    losses: list[float] = []

    for step in range(1, steps + 1):
        index = rng.integers(0, len(targets), size=batch_size)
        x1 = targets[index]
        x0 = rng.standard_normal(x1.shape).astype(np.float32)
        t = rng.random((batch_size, 1)).astype(np.float32)
        x_t = (1 - t) * x0 + t * x1
        target_velocity = (x1 - x0).astype(np.float32)

        inputs = np.concatenate([x_t, t, descriptors[index]], axis=1)
        out = model.forward(inputs)
        error = out - target_velocity
        loss = float((error**2).mean())

        # Backward pass through the small MLP.
        grads_w: list[np.ndarray] = [np.zeros_like(w) for w in model.weights]
        grads_b: list[np.ndarray] = [np.zeros_like(b) for b in model.biases]
        grad = 2.0 * error / error.size
        activations = [inputs]
        current = inputs
        for layer, (weight, bias) in enumerate(zip(model.weights, model.biases, strict=True)):
            current = current @ weight + bias
            if layer < len(model.weights) - 1:
                current = np.maximum(current, 0.0)
            activations.append(current)
        for layer in range(len(model.weights) - 1, -1, -1):
            if layer < len(model.weights) - 1:
                grad = grad * (activations[layer + 1] > 0)
            grads_w[layer] = activations[layer].T @ grad
            grads_b[layer] = grad.sum(axis=0)
            grad = grad @ model.weights[layer].T

        _adam_step(model.parameters(), [*grads_w, *grads_b], state, lr=lr, step=step)
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            losses.append(round(loss, 6))
            if progress:
                progress({"step": step, "loss": round(loss, 6)})
    return model, losses


def sample(
    model: FlowModel,
    descriptor: np.ndarray,
    *,
    steps: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """Integrate the learned field from noise to a parameter vector (Euler)."""
    if steps < 1:
        raise ValidationFailure("integration steps must be >= 1")
    rng = np.random.default_rng(seed)
    param_dim = model.sizes[-1]
    x = rng.standard_normal((1, param_dim)).astype(np.float32)
    dt = 1.0 / steps
    for step in range(steps):
        t = np.full((1, 1), step * dt, dtype=np.float32)
        velocity = model.forward(np.concatenate([x, t, descriptor[None, :]], axis=1))
        x = (x + velocity * dt).astype(np.float32)
    return x[0]


def run_flow_experiment(
    config: Any,
    generator_config: Any,
    *,
    hidden: int = 128,
    steps: int = 400,
    integration_steps: int = 40,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Build the aligned corpus, train the flow model, sample and evaluate on held-out tasks."""
    task_seeds = [int(value) for value in generator_config.corpus]
    if len(task_seeds) < 4:
        raise ValidationFailure(
            "flow matching needs at least 4 corpus tasks (3 to train, 1 held out)",
            hint="pass --corpus with at least four distinct task seeds",
        )
    split = max(3, int(len(task_seeds) * 0.75))
    train_seeds, test_seeds = task_seeds[:split], task_seeds[split:]
    if set(train_seeds) & set(test_seeds):
        raise ValidationFailure("corpus leaks test tasks into the training split")

    from .models.registry import build_model

    raw_weights: list[dict[str, np.ndarray]] = []
    descriptors: list[np.ndarray] = []
    corpus_manifest: list[dict[str, Any]] = []
    for task_seed in train_seeds:
        flat, stats = _train_target(config, task_seed=task_seed, steps=120, seed=task_seed + 7)
        handle = build_model(config.model_dump(mode="json"))
        raw_weights.append(_weights_from_handle(handle))
        descriptors.append(_descriptor(config, task_seed, generator_config.latent_dim))
        corpus_manifest.append({"split": "train", **stats, "param_norm": round(float(np.linalg.norm(flat)), 5)})

    shapes = {key: value.shape for key, value in raw_weights[0].items()}
    probe = datasets.mlp_batch(config, batch_size=64, seed=4242, split="val")[0]
    aligned_weights, alignment_reports = align_corpus(raw_weights, probe=probe)
    if not all(report.get("verified") for report in alignment_reports):
        raise ValidationFailure(
            "alignment failed its functional check; refusing to train a generator on misaligned weights",
            details={"reports": alignment_reports},
        )

    targets = np.stack([to_torch_flat(weights) for weights in aligned_weights])
    descriptor_matrix = np.stack(descriptors)

    # Per-dimension standardisation: parameter scales differ by orders of magnitude across tensors, and
    # an unnormalised regression target makes the velocity field learn bias instead of structure.
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0) + 1e-6
    normalised = ((targets - target_mean) / target_std).astype(np.float32)

    model, losses = train_flow(
        normalised,
        descriptor_matrix,
        hidden=hidden,
        steps=steps,
        seed=generator_config.seed,
        progress=progress,
    )

    # In-distribution check: the generator should reproduce checkpoints it was trained on. This is a
    # different question from out-of-distribution generalisation, and both are reported.
    in_distribution: list[dict[str, Any]] = []
    for position, task_seed in enumerate(train_seeds):
        descriptor = _descriptor(config, task_seed, generator_config.latent_dim)
        vector = sample(
            model, descriptor, steps=integration_steps, seed=generator_config.seed + 1000 + task_seed
        )
        vector = vector * target_std + target_mean
        weights = from_torch_flat(vector, shapes)
        flat = to_torch_flat(weights)
        handle = build_model(config.model_dump(mode="json"))
        generated_metrics = _score_flat(handle, flat, config, task_seed)
        trained_handle = build_model(config.model_dump(mode="json"))
        trained_metrics = _score_flat(trained_handle, targets[position], config, task_seed)
        distance = float(np.linalg.norm(flat - targets[position]) / max(1e-9, np.linalg.norm(targets[position])))
        in_distribution.append(
            {
                "task_seed": task_seed,
                "generated_accuracy": round(generated_metrics["accuracy"], 6),
                "trained_accuracy": round(trained_metrics["accuracy"], 6),
                "relative_parameter_distance": round(distance, 6),
            }
        )

    # Sampling diversity: two draws from the same condition should not be identical. If they are, the
    # flow has collapsed to a deterministic autoencoder — a real failure mode at small corpus sizes,
    # so it is measured instead of assumed away.
    descriptor = _descriptor(config, train_seeds[0], generator_config.latent_dim)
    first = sample(model, descriptor, steps=integration_steps, seed=generator_config.seed + 7)
    second = sample(model, descriptor, steps=integration_steps, seed=generator_config.seed + 8)
    diversity = float(
        np.linalg.norm(first - second) / max(1e-9, 0.5 * (np.linalg.norm(first) + np.linalg.norm(second)))
    )

    generated_scores: list[float] = []
    random_scores: list[float] = []
    trained_scores: list[float] = []
    samples: list[dict[str, Any]] = []
    for task_seed in test_seeds:
        descriptor = _descriptor(config, task_seed, generator_config.latent_dim)
        vector = sample(
            model, descriptor, steps=integration_steps, seed=generator_config.seed + task_seed
        )
        vector = vector * target_std + target_mean
        weights = from_torch_flat(vector, shapes)
        flat = to_torch_flat(weights)

        generated_handle = build_model(config.model_dump(mode="json"))
        generated = _score_flat(generated_handle, flat, config, task_seed)
        generated_scores.append(generated["accuracy"])

        random_handle = build_model(config.model_dump(mode="json"))
        random_flat = np.concatenate(
            [p.detach().cpu().numpy().reshape(-1) for p in random_handle.module.parameters()]
        ).astype(np.float32)
        random_metrics = _score_flat(random_handle, random_flat, config, task_seed)
        random_scores.append(random_metrics["accuracy"])

        trained_flat, trained_stats = _train_target(config, task_seed=task_seed, steps=120, seed=task_seed + 11)
        trained_handle = build_model(config.model_dump(mode="json"))
        trained_metrics = _score_flat(trained_handle, trained_flat, config, task_seed)
        trained_scores.append(trained_metrics["accuracy"])
        corpus_manifest.append({"split": "test", **trained_stats})

        samples.append(
            {
                "task_seed": task_seed,
                "generated_accuracy": round(generated["accuracy"], 6),
                "random_init_accuracy": round(random_metrics["accuracy"], 6),
                "gradient_trained_accuracy": round(trained_metrics["accuracy"], 6),
            }
        )

    return {
        "method": "conditional-flow-matching",
        "status": "experimental",
        "target_config": config.model_dump(mode="json"),
        "corpus": {
            "train_tasks": train_seeds,
            "test_tasks": test_seeds,
            "entries": corpus_manifest,
        },
        "alignment": {
            "method": "activation-correlation + Hungarian assignment",
            "reports": alignment_reports,
            "all_verified": True,
        },
        "generator": {
            "architecture": list(model.sizes),
            "hidden": hidden,
            "train_steps": steps,
            "integration_steps": integration_steps,
            "loss_curve": losses,
            "parameters": int(sum(p.size for p in model.parameters())),
            "target_normalisation": {
                "mean_abs": float(np.abs(target_mean).mean()),
                "std_mean": float(target_std.mean()),
            },
        },
        "in_distribution": in_distribution,
        "sampling_diversity": {
            "relative_distance": round(diversity, 8),
            "collapsed": diversity < 1e-3,
            "note": "two draws from the same condition; near-zero means the generator ignores the noise",
        },
        "results": {
            "generated_mean_accuracy": round(float(np.mean(generated_scores)), 6),
            "random_init_mean_accuracy": round(float(np.mean(random_scores)), 6),
            "gradient_trained_mean_accuracy": round(float(np.mean(trained_scores)), 6),
            "samples": samples,
        },
        "leakage_control": "test tasks are disjoint from training tasks and asserted in code",
        "claim_boundary": (
            "Toy scale: one tiny MLP family, a handful of synthetic tasks, and a few hundred flow-matching "
            "steps. Alignment is exact for one hidden layer only. This demonstrates the pipeline and the "
            "alignment requirement; it is not a reproduction of D2NWG or DeepWeightFlow, and it makes no "
            "claim that generated weights beat gradient training."
        ),
        "model": model.to_dict(),
    }
