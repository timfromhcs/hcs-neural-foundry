"""Weight Studio: tensor statistics, bounded slicing and immutable patch operations.

Nothing here writes to a checkpoint. A patch produces a *new* state dictionary, which the service
layer verifies and persists. Operations are a closed enum (ADR-005) — there is no expression
language and no ``eval`` path.
"""

from __future__ import annotations

from typing import Any

from .errors import ValidationFailure
from .models.base import torch_module
from .schemas import (
    MAX_PATCH_ELEMENTS,
    PatchOperation,
    PatchValidation,
    TensorHistogram,
    TensorSlice,
    TensorSummary,
    WeightPatch,
)

SAFE_TENSOR_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def tensor_state(module: Any, *, dedupe: bool = False) -> dict[str, Any]:
    """Mapping of tensor name -> tensor. Aliased entries are included unless ``dedupe``."""
    if dedupe:
        from .checkpoints import flattened_state

        return flattened_state(module)["tensors"]
    return dict(module.state_dict())


def get_tensor(module: Any, name: str) -> Any:
    if not name or not set(name) <= SAFE_TENSOR_NAME_CHARS:
        raise ValidationFailure(f"invalid tensor name '{name}'")
    state = tensor_state(module)
    tensor = state.get(name)
    if tensor is None:
        candidates = [n for n in state if name.lower() in n.lower()][:8]
        raise ValidationFailure(
            f"tensor '{name}' does not exist",
            hint=f"did you mean: {candidates}" if candidates else "inspect the model tensor list first",
        )
    return tensor


def summarize(module: Any) -> list[TensorSummary]:
    """Cheap statistics for every tensor. Uses a bounded sample for very large tensors."""
    torch = torch_module()
    out: list[TensorSummary] = []
    state = tensor_state(module)
    seen: dict[int, str] = {}
    for name, tensor in state.items():
        tied_to = None
        key = int(tensor.data_ptr())
        if key in seen and tensor.numel() > 0:
            tied_to = seen[key]
        else:
            seen[key] = name
        flat = tensor.detach().reshape(-1).float()
        if flat.numel() == 0:
            out.append(
                TensorSummary(
                    name=name,
                    shape=tuple(tensor.shape),
                    numel=0,
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    trainable=bool(tensor.requires_grad),
                    min=0.0,
                    max=0.0,
                    mean=0.0,
                    std=0.0,
                    norm=0.0,
                    zeros_fraction=0.0,
                    finite=True,
                    tied_to=tied_to,
                )
            )
            continue
        sample = flat if flat.numel() <= 262_144 else flat[:: max(1, flat.numel() // 262_144)]
        finite = bool(torch.isfinite(sample).all())
        safe = sample[torch.isfinite(sample)] if not finite else sample
        out.append(
            TensorSummary(
                name=name,
                shape=tuple(tensor.shape),
                numel=int(tensor.numel()),
                dtype=str(tensor.dtype).replace("torch.", ""),
                trainable=bool(tensor.requires_grad),
                min=float(safe.min()) if safe.numel() else 0.0,
                max=float(safe.max()) if safe.numel() else 0.0,
                mean=float(safe.mean()) if safe.numel() else 0.0,
                std=float(safe.std(unbiased=False)) if safe.numel() > 1 else 0.0,
                norm=float(torch.linalg.vector_norm(tensor.detach().float())),
                zeros_fraction=float((sample == 0).float().mean()),
                finite=finite,
                tied_to=tied_to,
            )
        )
    return out


def slice_tensor(
    module: Any,
    name: str,
    *,
    offset: tuple[int, ...] | None = None,
    shape: tuple[int, ...] | None = None,
    max_elements: int = 4096,
) -> TensorSlice:
    """Bounded read: never returns more than ``max_elements`` numbers."""
    tensor = get_tensor(module, name)
    if tensor.dim() == 0:
        return TensorSlice(name=name, offset=(), shape=(), values=[float(tensor)], truncated=False)
    offset = tuple(offset or (0,) * tensor.dim())
    if len(offset) != tensor.dim():
        raise ValidationFailure(f"offset must have {tensor.dim()} entries for tensor '{name}'")
    for axis, (start, size) in enumerate(zip(offset, tensor.shape, strict=False)):
        if start < 0 or start >= size:
            raise ValidationFailure(f"offset[{axis}]={start} is outside dimension of size {size}")
    shape = tuple(shape or tensor.shape)
    if len(shape) != tensor.dim():
        raise ValidationFailure(f"shape must have {tensor.dim()} entries for tensor '{name}'")
    requested = 1
    for value in shape:
        requested *= max(1, value)
    truncated = requested > max_elements
    if truncated:
        # Halve the innermost dimension until the request fits.
        shape_list = list(shape)
        while requested > max_elements and shape_list:
            index = next((i for i in range(len(shape_list) - 1, -1, -1) if shape_list[i] > 1), None)
            if index is None:
                break
            shape_list[index] = max(1, shape_list[index] // 2)
            requested = 1
            for value in shape_list:
                requested *= value
        shape = tuple(shape_list)

    slices = tuple(slice(offset[i], min(offset[i] + shape[i], tensor.shape[i])) for i in range(tensor.dim()))
    block = tensor.detach()[slices].float().cpu()
    return TensorSlice(
        name=name,
        offset=offset,
        shape=tuple(block.shape),
        values=[float(v) for v in block.reshape(-1).tolist()],
        truncated=truncated,
    )


def histogram(module: Any, name: str, *, bins: int = 32, max_samples: int = 200_000) -> TensorHistogram:
    tensor = get_tensor(module, name).detach().float().reshape(-1)
    if tensor.numel() == 0:
        return TensorHistogram(name=name, bins=[], counts=[], sampled=0)
    if tensor.numel() > max_samples:
        step = max(1, tensor.numel() // max_samples)
        tensor = tensor[::step]
    counts = torch_module().histogram(tensor, bins=bins)
    return TensorHistogram(
        name=name,
        bins=[float(v) for v in counts.bin_edges.tolist()],
        counts=[int(v) for v in counts.hist.tolist()],
        sampled=int(tensor.numel()),
    )


# --------------------------------------------------------------------------------------------------
# Patch operations
# --------------------------------------------------------------------------------------------------


def _validate_shape_and_dtype(tensor: Any, patch: WeightPatch) -> None:
    if tuple(tensor.shape) != tuple(patch.precondition.shape):
        raise ValidationFailure(
            f"precondition shape {tuple(patch.precondition.shape)} != tensor shape {tuple(tensor.shape)}",
            hint="the patch was written against a different architecture or checkpoint",
        )
    dtype = str(tensor.dtype).replace("torch.", "")
    if dtype != patch.precondition.dtype:
        raise ValidationFailure(f"precondition dtype {patch.precondition.dtype} != tensor dtype {dtype}")


def apply_operation(tensor: Any, patch: WeightPatch, *, other: Any | None = None) -> Any:
    """Return a new tensor with the patch applied. Pure: the input tensor is never mutated."""
    torch = torch_module()
    params = patch.parameters
    out = tensor.detach().clone()
    op = patch.operation

    if op is PatchOperation.SET_SCALAR:
        index = tuple(int(i) for i in params["index"])
        if len(index) != out.dim():
            raise ValidationFailure(f"index has {len(index)} entries but tensor has {out.dim()} dimensions")
        for axis, position in enumerate(index):
            if position >= out.shape[axis]:
                raise ValidationFailure(
                    f"index[{axis}]={position} out of bounds for dimension of size {out.shape[axis]}"
                )
        out[index] = float(params["value"])
        return out

    if op is PatchOperation.SET_BLOCK:
        start = tuple(int(v) for v in params["start"])
        block_shape = tuple(int(v) for v in params["block_shape"])
        if len(start) != out.dim() or len(block_shape) != out.dim():
            raise ValidationFailure("start and block_shape must match the tensor rank")
        slices = tuple(slice(start[i], start[i] + block_shape[i]) for i in range(out.dim()))
        for axis in range(out.dim()):
            if start[axis] + block_shape[axis] > out.shape[axis]:
                raise ValidationFailure(
                    f"block exceeds dimension {axis}: {start[axis]}+{block_shape[axis]} > {out.shape[axis]}"
                )
        mode = params["mode"]
        if mode == "zeros":
            out[slices] = 0.0
        elif mode == "ones":
            out[slices] = 1.0
        elif mode == "value":
            out[slices] = float(params["value"])
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
            noise = torch.randn(block_shape, generator=generator, dtype=out.dtype) * float(params["std"])
            out[slices] = noise.to(out.device)
        return out

    if op is PatchOperation.INIT_SEEDED:
        generator = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
        scheme = params["scheme"]
        if scheme == "zeros":
            return torch.zeros_like(out)
        if scheme == "ones":
            return torch.ones_like(out)
        if scheme == "normal":
            std = float(params.get("std", 0.02))
            return (torch.randn(out.shape, generator=generator, dtype=out.dtype) * std).to(out.device)
        if scheme == "uniform":
            bound = float(params.get("std", 0.05)) * 3**0.5
            return (torch.rand(out.shape, generator=generator, dtype=out.dtype) * 2 - 1) * bound
        if out.dim() < 2:
            raise ValidationFailure(f"{scheme} initialization needs a tensor with at least 2 dimensions")
        fan_in = out.shape[1] * (out.shape[2] if out.dim() > 2 else 1)
        fan_out = out.shape[0] * (out.shape[2] if out.dim() > 2 else 1)
        if scheme == "xavier":
            std = (2.0 / (fan_in + fan_out)) ** 0.5
        else:  # kaiming
            fan = fan_in if params.get("fan_mode", "fan_in") == "fan_in" else fan_out
            std = (2.0 / max(1, fan)) ** 0.5
        return (torch.randn(out.shape, generator=generator, dtype=out.dtype) * std).to(out.device)

    if op is PatchOperation.ADD_SCALED_DELTA:
        generator = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
        noise = torch.randn(out.shape, generator=generator, dtype=out.dtype) * float(params["std"])
        return out + noise.to(out.device) * float(params["scale"])

    if op is PatchOperation.LOW_RANK_DELTA:
        if out.dim() != 2:
            raise ValidationFailure("low_rank_delta requires a 2-D tensor (matrix)")
        rank = int(params["rank"])
        alpha = float(params["alpha"])
        std = float(params.get("std", 0.02))
        generator = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
        left = torch.randn(out.shape[0], rank, generator=generator, dtype=torch.float32) * std
        right = torch.randn(rank, out.shape[1], generator=generator, dtype=torch.float32) * std
        delta = (alpha / rank) * (left @ right)
        return (out.float() + delta).to(out.dtype)

    if op is PatchOperation.NORMALIZE:
        axis = int(params["axis"])
        eps = float(params["eps"])
        norm = torch.linalg.vector_norm(out.float(), dim=axis, keepdim=True).clamp_min(eps)
        return (out.float() / norm).to(out.dtype)

    if op is PatchOperation.CLAMP:
        low = params.get("min")
        high = params.get("max")
        return torch.clamp(out, min=low, max=high)

    if op is PatchOperation.SCALE:
        return out * float(params["factor"])

    if op is PatchOperation.SPARSITY_MASK:
        keep = float(params["keep_fraction"])
        structured = params.get("structured", "none")
        if structured in {"rows", "cols"}:
            if out.dim() != 2:
                raise ValidationFailure("structured sparsity requires a 2-D tensor")
            axis = 0 if structured == "rows" else 1
            size = out.shape[axis]
            generator = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
            keep_count = max(1, int(round(size * keep)))
            keep_idx = torch.randperm(size, generator=generator)[:keep_count].to(out.device)
            if structured == "rows":
                masked = torch.zeros_like(out)
                masked[keep_idx, :] = out[keep_idx, :]
            else:
                masked = torch.zeros_like(out)
                masked[:, keep_idx] = out[:, keep_idx]
            return masked
        generator = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
        mask = torch.rand(out.shape, generator=generator) < keep
        return torch.where(mask.to(out.device), out, torch.zeros_like(out))

    if op is PatchOperation.INTERPOLATE:
        if other is None:
            raise ValidationFailure("interpolate requires a second checkpoint")
        if tuple(other.shape) != tuple(out.shape):
            raise ValidationFailure("interpolate requires identically shaped tensors")
        t = float(params["t"])
        return ((1.0 - t) * out.float() + t * other.float()).to(out.dtype)

    if op is PatchOperation.COPY_FROM:
        if other is None:
            raise ValidationFailure("copy_from requires the source checkpoint")
        if tuple(other.shape) != tuple(out.shape):
            raise ValidationFailure(
                f"copy_from shape mismatch: {tuple(other.shape)} != {tuple(out.shape)}"
            )
        return other.detach().clone().to(out.dtype)

    if op is PatchOperation.SET_TRAINABLE:
        # Handled by the service (module-level flag); the tensor values do not change.
        return out

    raise ValidationFailure(f"unsupported operation '{op.value}'")  # pragma: no cover - enum is closed


def apply_patch(
    state: dict[str, Any],
    patch: WeightPatch,
    *,
    other_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], PatchValidation]:
    """Apply one patch to a copy of ``state`` and report exactly what changed."""
    torch = torch_module()
    tensor = state.get(patch.target_tensor)
    if tensor is None:
        raise ValidationFailure(
            f"target tensor '{patch.target_tensor}' is missing from the checkpoint",
            hint="tensor names come from the architecture; run `weights list`",
        )
    _validate_shape_and_dtype(tensor, patch)
    if tensor.numel() > MAX_PATCH_ELEMENTS:
        raise ValidationFailure(
            f"tensor has {tensor.numel():,} elements, above the {MAX_PATCH_ELEMENTS:,} patch guard rail",
            hint="split the edit or raise the limit explicitly in configuration",
        )

    other = None
    if patch.operation is PatchOperation.INTERPOLATE:
        key = patch.parameters["other_checkpoint_sha256"]
        if other_state is None:
            raise ValidationFailure(f"checkpoint {key[:12]}… was not supplied for interpolation")
        other = other_state.get(patch.target_tensor)
    elif patch.operation is PatchOperation.COPY_FROM:
        key = patch.parameters["source_checkpoint_sha256"]
        if other_state is None:
            raise ValidationFailure(f"checkpoint {key[:12]}… was not supplied for copy_from")
        source_name = patch.parameters["source_tensor"]
        other = other_state.get(source_name)
        if other is None:
            raise ValidationFailure(f"source tensor '{source_name}' missing in the source checkpoint")

    updated = apply_operation(tensor, patch, other=other)
    new_state = dict(state)
    new_state[patch.target_tensor] = updated

    before = tensor.detach().float()
    after = updated.detach().float()
    finite = bool(torch.isfinite(after).all()) if after.numel() else True
    if not finite:
        raise ValidationFailure(
            "patch produced non-finite values; the edit was rejected",
            hint="lower the scale/std or clamp the tensor afterwards",
        )
    delta = (after - before).abs()
    changed = int((after != before).sum())
    norm_before = float(torch.linalg.vector_norm(before))
    norm_after = float(torch.linalg.vector_norm(after))
    rel = (norm_after - norm_before) / norm_before if norm_before else 0.0
    return new_state, PatchValidation(
        finite=finite,
        max_abs_delta=float(delta.max()) if delta.numel() else 0.0,
        changed_elements=changed,
        norm_before=norm_before,
        norm_after=norm_after,
        relative_norm_change=rel,
    )
