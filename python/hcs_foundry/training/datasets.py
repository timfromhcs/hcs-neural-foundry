"""Bundled synthetic datasets.

These exist so that CI and first-run experiences never depend on a download or a licence review
(ADR-012). They measure *pipeline correctness*, not model capability, and every report says so.

``tiny-text`` is a deterministic character corpus with learnable local structure.
``tiny-shapes`` is a deterministic linearly-separable-plus-noise classification problem whose
dimensionality adapts to the target MLP so any architecture can be trained honestly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import NotFoundError
from ..schemas import DatasetSpec, MlpConfig, TransformerConfig

CORPUS_SEED = 20261103


def _grammar_corpus(repeats: int = 220) -> str:
    """Deterministic pseudo-text with strong local structure (learnable by a tiny model)."""
    rng = np.random.default_rng(CORPUS_SEED)
    subjects = ["hcs", "foundry", "tensor", "weight", "model", "agent", "kernel", "lambda"]
    verbs = ["builds", "measures", "trains", "edits", "checks", "exports", "compares"]
    objects = ["a network", "the loss", "one patch", "a budget", "the report", "two seeds"]
    lines: list[str] = []
    for index in range(repeats):
        subject = subjects[int(rng.integers(0, len(subjects)))]
        verb = verbs[int(rng.integers(0, len(verbs)))]
        obj = objects[int(rng.integers(0, len(objects)))]
        lines.append(f"{index:04d} {subject} {verb} {obj}. ")
    return "".join(lines)


CORPUS = _grammar_corpus()
CHARS = sorted(set(CORPUS))
CHAR_TO_ID = {char: index for index, char in enumerate(CHARS)}


def corpus_sha256() -> str:
    return hashlib.sha256(CORPUS.encode("utf-8")).hexdigest()


def encode_text(text: str, vocab_size: int) -> list[int]:
    """Map characters into ``[0, vocab_size)``. Unknown characters fall back to a stable bucket."""
    ids = []
    for char in text:
        value = CHAR_TO_ID.get(char)
        if value is None:
            value = (ord(char) + 7) % vocab_size
        ids.append(value % vocab_size)
    return ids


def text_batch(
    config: TransformerConfig,
    *,
    batch_size: int,
    block_size: int,
    seed: int,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` int arrays shaped ``(batch, block)`` for next-token prediction."""
    block_size = min(block_size, config.context_length)
    ids = np.asarray(encode_text(CORPUS, config.vocab_size), dtype=np.int64)
    usable = len(ids) - block_size - 1
    if usable < block_size:
        raise NotFoundError("corpus is too small for the requested block size")
    cutoff = int(usable * 0.9)
    if split == "train":
        hi = cutoff
        lo = 0
    else:
        lo = cutoff
        hi = usable
    rng = np.random.default_rng(seed)
    starts = rng.integers(lo, hi, size=batch_size)
    xs = np.stack([ids[s : s + block_size] for s in starts])
    ys = np.stack([ids[s + 1 : s + 1 + block_size] for s in starts])
    return xs, ys


def text_split_sizes(config: TransformerConfig) -> tuple[int, int]:
    ids = encode_text(CORPUS, config.vocab_size)
    usable = max(0, len(ids) - config.context_length - 1)
    cutoff = int(usable * 0.9)
    return cutoff, usable - cutoff


def _task_matrix(input_dim: int, output_dim: int, task_seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fixed, seeded teacher used by the MLP task so the problem is identical across runs."""
    base = CORPUS_SEED if task_seed is None else int(task_seed)
    rng = np.random.default_rng(base + input_dim * 31 + output_dim)
    weights = rng.standard_normal((input_dim, output_dim)).astype(np.float32) / np.sqrt(input_dim)
    bias = rng.standard_normal(output_dim).astype(np.float32) * 0.1
    return weights, bias


def mlp_batch(
    config: MlpConfig,
    *,
    batch_size: int,
    seed: int,
    split: str = "train",
    task_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic classification batch: logits = x @ W + b with label-noise at generation time."""
    offset = 0 if split == "train" else 10_000_000
    rng = np.random.default_rng(seed + offset)
    x = rng.standard_normal((batch_size, config.input_dim)).astype(np.float32)
    weights, bias = _task_matrix(config.input_dim, config.output_dim, task_seed)
    logits = x @ weights + bias
    # 10% label noise keeps the task from being trivially memorisable and makes accuracy meaningful.
    flip = rng.random(batch_size) < 0.1
    labels = logits.argmax(axis=1).astype(np.int64)
    if flip.any():
        replacement = rng.integers(0, config.output_dim, size=int(flip.sum()))
        labels[flip] = replacement
    return x, labels


def ensure_builtin_datasets(workspace: Path) -> list[DatasetSpec]:
    """Materialise dataset manifests on disk and return their specs."""
    specs: list[DatasetSpec] = []
    directory = Path(workspace) / "datasets"
    directory.mkdir(parents=True, exist_ok=True)

    text_path = directory / "tiny-text.txt"
    if not text_path.exists():
        text_path.write_text(CORPUS, encoding="utf-8")
    corpus_bytes = CORPUS.encode("utf-8")
    text_sha = hashlib.sha256(corpus_bytes).hexdigest()
    train_chars, val_chars = len(CORPUS) - int(len(CORPUS) * 0.1), int(len(CORPUS) * 0.1)
    specs.append(
        DatasetSpec(
            id="tiny-text",
            name="Tiny grammar corpus (character level)",
            kind="text",
            path=str(text_path),
            sha256=text_sha,
            train_rows=train_chars,
            val_rows=val_chars,
            license="synthetic, generated in-repo (MIT)",
            notes=f"vocab={len(CHARS)} distinct characters; deterministic seed {CORPUS_SEED}",
        )
    )

    shapes_path = directory / "tiny-shapes.npz"
    if not shapes_path.exists():
        weights, bias = _task_matrix(16, 4)
        np.savez(shapes_path, weights=weights, bias=bias)
    shapes_sha = hashlib.sha256(shapes_path.read_bytes()).hexdigest()
    specs.append(
        DatasetSpec(
            id="tiny-shapes",
            name="Synthetic classification task (16 -> 4)",
            kind="classification",
            path=str(shapes_path),
            sha256=shapes_sha,
            train_rows=100_000,
            val_rows=25_000,
            license="synthetic, generated in-repo (MIT)",
            notes="fixed teacher matrix; 10% label noise; dimensionality adapts to the target MLP",
        )
    )
    return specs


def dataset_for_model(family: str) -> str:
    return "tiny-text" if family == "transformer-lm" else "tiny-shapes"


def describe() -> dict[str, Any]:
    return {
        "corpus_chars": len(CORPUS),
        "distinct_chars": len(CHARS),
        "corpus_sha256": corpus_sha256(),
        "seed": CORPUS_SEED,
    }

