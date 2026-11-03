"""Permutation alignment for weight-space models.

Two networks with permuted hidden units compute the *same function* but have unrelated parameter
vectors. Any generator, interpolator or corpus statistic that ignores this produces meaningless
averages — which is why alignment is a prerequisite, not a refinement.

Scope: one-hidden-layer MLPs with a ReLU/GELU-style activation. That is the architecture this build can
align exactly and verify functionally. Bigger models need a real Git Re-Basin implementation; that is
declared unsupported rather than approximated.
"""

from __future__ import annotations

import numpy as np

from .errors import ValidationFailure

ACTIVATION_NAMES = ("relu", "gelu", "tanh")


def linear_sum_assignment(cost: np.ndarray) -> np.ndarray:
    """Hungarian algorithm on a square (or wide) cost matrix.

    Returns ``assignment`` where ``assignment[row] = column``. Rows are the smaller dimension.
    Implemented directly (e-maxx shortest augmenting path form) to avoid a SciPy dependency for ~50
    lines of arithmetic; correctness is checked against brute force in the tests.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValidationFailure("cost must be a 2-D matrix")
    rows, cols = matrix.shape
    if rows > cols:
        # Transpose so rows <= cols; the algorithm requires it. ``inner[i] = j`` means transposed row
        # i (an original column) is matched to transposed column j (an original row).
        inner = linear_sum_assignment(matrix.T)
        assignment = np.full(rows, -1, dtype=np.int64)
        for original_col, original_row in enumerate(inner):
            assignment[original_row] = original_col
        return assignment

    inf = float("inf")
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(cols + 1, inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = np.full(rows, -1, dtype=np.int64)
    for j in range(1, cols + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def activation(weights: dict[str, np.ndarray], probe: np.ndarray, activation_name: str = "relu") -> np.ndarray:
    """Hidden activations for a one-hidden-layer MLP: shape ``(n_probes, hidden)``."""
    hidden = probe @ weights["w1"] + weights["b1"]
    if activation_name == "relu":
        return np.maximum(hidden, 0.0)
    if activation_name == "gelu":
        # tanh approximation is fine here: alignment only needs a monotone, differentiable response.
        return 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)))
    if activation_name == "tanh":
        return np.tanh(hidden)
    raise ValidationFailure(f"unsupported activation '{activation_name}'", hint=f"use one of {ACTIVATION_NAMES}")


def _unit_correlations(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Correlation between every source unit and every target unit, robust to dead units."""
    a = source - source.mean(axis=0, keepdims=True)
    b = target - target.mean(axis=0, keepdims=True)
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    numerator = a.T @ b
    denominator = np.outer(a_norm, b_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denominator > 1e-12, numerator / denominator, 0.0)
    return np.nan_to_num(corr, nan=0.0)


def permute_hidden(weights: dict[str, np.ndarray], order: np.ndarray) -> dict[str, np.ndarray]:
    """Relabel hidden units. ``order[new] = old``."""
    order = np.asarray(order, dtype=np.int64)
    out = {
        "w1": weights["w1"][:, order],
        "b1": weights["b1"][order],
        "w2": weights["w2"][order, :],
        "b2": weights["b2"].copy(),
    }
    return out


def align_to(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    probe: np.ndarray,
    activation_name: str = "relu",
) -> dict[str, object]:
    """Find the hidden-unit permutation that best matches ``source`` to ``target``.

    Returns the permuted source, the permutation, the assignment cost and a functional check. The
    functional check is the point: if permuted outputs disagree with the original, the alignment is
    wrong and the caller must not use it.
    """
    for key in ("w1", "b1", "w2", "b2"):
        if key not in source or key not in target:
            raise ValidationFailure(f"missing tensor '{key}'")
    if source["w1"].shape != target["w1"].shape:
        raise ValidationFailure(
            "alignment requires identical architectures",
            details={"source": list(source["w1"].shape), "target": list(target["w1"].shape)},
        )

    source_act = activation(source, probe, activation_name)
    target_act = activation(target, probe, activation_name)
    corr = _unit_correlations(source_act, target_act)
    cost = -corr  # maximise correlation == minimise negative correlation
    assignment = linear_sum_assignment(cost)
    if (assignment < 0).any():  # pragma: no cover - guard against a partial assignment
        raise ValidationFailure("alignment produced an incomplete assignment")
    # ``assignment[source_unit] = target_unit``; ``permute_hidden`` needs order[new] = old, i.e. the
    # inverse: place source unit i into slot assignment[i].
    order = np.empty_like(assignment)
    order[assignment] = np.arange(len(assignment), dtype=np.int64)

    permuted = permute_hidden(source, order)
    probe_outputs_before = probe @ source["w1"] + source["b1"]
    probe_outputs_after = probe @ permuted["w1"] + permuted["b1"]
    # Reordering rows of w1 and b1 changes the layout; compare function, not layout.
    function_before = activation(source, probe, activation_name) @ source["w2"] + source["b2"]
    function_after = activation(permuted, probe, activation_name) @ permuted["w2"] + permuted["b2"]
    max_abs_diff = float(np.max(np.abs(function_before - function_after))) if function_before.size else 0.0
    reconstruction = float(np.max(np.abs(probe_outputs_before[:, order] - probe_outputs_after)))

    return {
        "weights": permuted,
        "order": order,
        "cost": float(cost[np.arange(len(assignment)), assignment].sum()),
        "mean_correlation": float(corr[np.arange(len(assignment)), assignment].mean()),
        "functional_max_abs_diff": max_abs_diff,
        "layout_max_abs_diff": reconstruction,
        "verified": max_abs_diff < 1e-5,
    }


def alignment_report(result: dict[str, object]) -> dict[str, object]:
    """JSON-safe subset for run records (the permutation as a list, no arrays)."""
    return {
        "order": np.asarray(result["order"]).tolist(),
        "cost": round(float(result["cost"]), 6),
        "mean_correlation": round(float(result["mean_correlation"]), 6),
        "functional_max_abs_diff": float(result["functional_max_abs_diff"]),
        "verified": bool(result["verified"]),
    }


def align_corpus(
    corpus: list[dict[str, np.ndarray]],
    *,
    probe: np.ndarray,
    activation_name: str = "relu",
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
    """Align every checkpoint to the first one. Returns the aligned corpus and per-item reports."""
    if not corpus:
        return [], []
    anchor = corpus[0]
    aligned = [anchor]
    reports = [
        {
            "index": 0,
            "order": list(range(anchor["w1"].shape[1])),
            "verified": True,
            "mean_correlation": 1.0,
            "cost": 0.0,
            "functional_max_abs_diff": 0.0,
        }
    ]
    for index, item in enumerate(corpus[1:], start=1):
        result = align_to(item, anchor, probe=probe, activation_name=activation_name)
        aligned.append(result["weights"])  # type: ignore[arg-type]
        report = alignment_report(result)
        report["index"] = index
        reports.append(report)
    return aligned, reports
