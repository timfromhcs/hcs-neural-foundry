"""Tests for permutation alignment: solver correctness, exactness, and corpus alignment."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hcs_foundry.alignment import (
    activation,
    align_corpus,
    align_to,
    linear_sum_assignment,
    permute_hidden,
)
from hcs_foundry.errors import ValidationFailure


def _random_mlp(hidden: int = 6, input_dim: int = 5, output_dim: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.standard_normal((input_dim, hidden)).astype(np.float32),
        "b1": rng.standard_normal(hidden).astype(np.float32) * 0.1,
        "w2": rng.standard_normal((hidden, output_dim)).astype(np.float32),
        "b2": rng.standard_normal(output_dim).astype(np.float32) * 0.1,
    }


# --------------------------------------------------------------------------------------------------
# Assignment solver
# --------------------------------------------------------------------------------------------------


def test_assignment_matches_brute_force():
    rng = np.random.default_rng(7)
    for size in (2, 3, 4, 5):
        for _ in range(4):
            cost = rng.standard_normal((size, size))
            assignment = linear_sum_assignment(cost)
            best = min(
                (cost[np.arange(size), perm].sum(), perm)
                for perm in itertools.permutations(range(size))
            )
            assert cost[np.arange(size), assignment].sum() == pytest.approx(best[0], abs=1e-9)


def test_assignment_handles_rectangular_matrices():
    cost = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])  # 2 rows, 3 columns
    assignment = linear_sum_assignment(cost)
    assert assignment.tolist() == [0, 1]
    tall = cost.T  # 3 rows, 2 columns
    assignment_tall = linear_sum_assignment(tall)
    # Brute force over all valid (non-unique) column assignments for a tall matrix.
    best = min(
        tall[np.arange(3), combo].sum()
        for combo in ([0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0])
    )
    assert tall[np.arange(3), assignment_tall].sum() == pytest.approx(best, abs=1e-9)


def test_assignment_rejects_non_matrix():
    with pytest.raises(ValidationFailure):
        linear_sum_assignment(np.zeros(3))


# --------------------------------------------------------------------------------------------------
# Permutation semantics
# --------------------------------------------------------------------------------------------------


def test_permutation_preserves_the_function_exactly():
    weights = _random_mlp(seed=1)
    order = np.array([2, 0, 3, 1, 5, 4])
    permuted = permute_hidden(weights, order)
    probe = np.random.default_rng(3).standard_normal((16, 5)).astype(np.float32)
    before = activation(weights, probe) @ weights["w2"] + weights["b2"]
    after = activation(permuted, probe) @ permuted["w2"] + permuted["b2"]
    assert np.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------------------


def test_alignment_recovers_a_known_permutation():
    target = _random_mlp(seed=2)
    order = np.array([3, 0, 5, 1, 4, 2])
    source = permute_hidden(target, order)
    probe = np.random.default_rng(5).standard_normal((64, 5)).astype(np.float32)

    result = align_to(source, target, probe=probe)
    assert result["verified"] is True
    assert result["functional_max_abs_diff"] < 1e-5
    # The recovered order must undo the shuffle: applying it to source reproduces target's layout.
    recovered = permute_hidden(source, np.asarray(result["order"]))
    assert np.allclose(recovered["w1"], target["w1"], atol=1e-5)
    assert np.allclose(recovered["w2"], target["w2"], atol=1e-5)


def test_alignment_beats_the_unpermuted_baseline_on_correlation():
    target = _random_mlp(seed=3)
    source = permute_hidden(target, np.array([4, 2, 0, 5, 1, 3]))
    probe = np.random.default_rng(9).standard_normal((64, 5)).astype(np.float32)
    result = align_to(source, target, probe=probe)
    # Perfectly aligned units are identical, so mean correlation is ~1.
    assert result["mean_correlation"] > 0.99


def test_alignment_rejects_mismatched_architectures():
    with pytest.raises(ValidationFailure):
        align_to(_random_mlp(hidden=6), _random_mlp(hidden=7), probe=np.zeros((4, 5), dtype=np.float32))


def test_activation_supports_documented_functions():
    weights = _random_mlp(seed=4)
    probe = np.random.default_rng(1).standard_normal((8, 5)).astype(np.float32)
    for name in ("relu", "gelu", "tanh"):
        values = activation(weights, probe, name)
        assert values.shape == (8, 6)
        assert np.isfinite(values).all()
    with pytest.raises(ValidationFailure):
        activation(weights, probe, "swish")


# --------------------------------------------------------------------------------------------------
# Corpus alignment
# --------------------------------------------------------------------------------------------------


def test_corpus_alignment_reports_every_item():
    anchor = _random_mlp(seed=10)
    second = permute_hidden(_random_mlp(seed=11), np.array([1, 4, 0, 5, 3, 2]))
    third = _random_mlp(seed=12)
    probe = np.random.default_rng(2).standard_normal((64, 5)).astype(np.float32)

    aligned, reports = align_corpus([anchor, second, third], probe=probe)
    assert len(aligned) == 3 and len(reports) == 3
    assert reports[0]["index"] == 0
    assert all(report["verified"] for report in reports)
    # The first checkpoint is the anchor and must not move.
    assert np.allclose(aligned[0]["w1"], anchor["w1"])


def test_corpus_alignment_of_empty_input():
    aligned, reports = align_corpus([], probe=np.zeros((1, 2), dtype=np.float32))
    assert aligned == [] and reports == []
