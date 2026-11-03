"""Checkpoint I/O: safetensors only, with explicit handling of tied tensors.

PyTorch ``state_dict()`` returns the *same* storage twice for tied weights (e.g. token embedding and
an untied-looking ``lm_head``). safetensors refuses shared storage, so we record aliases in a JSON
sidecar, save only canonical tensors, and re-tie after loading. Getting this wrong silently doubles
a checkpoint's size and breaks patching, so it is covered by tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ValidationFailure
from .hashing import hash_file, new_id, utcnow, write_json_atomic
from .schemas import CheckpointRecord
from .storage import Workspace


def flattened_state(module: Any) -> dict[str, Any]:
    """state_dict with no shared storage: duplicated tensors removed, aliases recorded separately."""
    raw = module.state_dict()
    pointers: dict[int, str] = {}
    tensors: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for name, tensor in raw.items():
        key = int(tensor.data_ptr())
        if key in pointers and tensor.shape == raw[pointers[key]].shape:
            aliases[name] = pointers[key]
            continue
        pointers[key] = name
        tensors[name] = tensor.detach().cpu().contiguous().clone()
    return {"tensors": tensors, "aliases": aliases}


def _set_module_attr(module: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = module
    for part in parts[:-1]:
        current = getattr(current, part) if not part.isdigit() else current[int(part)]
    setattr(current, parts[-1], value)


def restore_aliases(module: Any, aliases: dict[str, str]) -> None:
    """Re-establish tied parameters after a load."""
    for alias, canonical in aliases.items():
        target = dict(module.named_parameters(remove_duplicate=False)).get(canonical)
        if target is None:
            # Fall back to a module attribute (buffers or non-parameter entries)
            raise ValidationFailure(
                f"cannot restore alias {alias} -> {canonical}: canonical tensor missing",
                hint="the checkpoint was produced by an incompatible architecture",
            )
        _set_module_attr(module, alias, target)


def save_checkpoint(
    workspace: Workspace,
    *,
    model_id: str,
    module: Any,
    architecture_hash: str,
    config_payload: dict[str, Any],
    label: str = "checkpoint",
    origin: str = "init",
    parent_sha256: str | None = None,
) -> CheckpointRecord:
    """Persist a module as ``<workspace>/checkpoints/<sha256>.safetensors`` + manifest."""
    from safetensors.torch import save_file

    packed = flattened_state(module)
    checkpoint_id = new_id("ckpt")
    tmp = workspace.checkpoints / f".tmp-{checkpoint_id}.safetensors"
    try:
        save_file(packed["tensors"], str(tmp), metadata={"format": "pt", "model_id": model_id})
        sha = hash_file(tmp)
        final = workspace.checkpoints / f"{sha}.safetensors"
        if final.exists():
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(final)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    manifest = {
        "schema_version": "1.0",
        "checkpoint_id": checkpoint_id,
        "model_id": model_id,
        "sha256": sha,
        "architecture_hash": architecture_hash,
        "config": config_payload,
        "aliases": packed["aliases"],
        "tensor_count": len(packed["tensors"]),
        "label": label,
        "origin": origin,
        "parent_sha256": parent_sha256,
        "created_at": utcnow(),
        "format": "safetensors",
    }
    write_json_atomic(workspace.checkpoints / f"{sha}.json", manifest)

    return CheckpointRecord(
        id=checkpoint_id,
        model_id=model_id,
        sha256=sha,
        path=str(final),
        bytes=final.stat().st_size,
        label=label,
        parent_sha256=parent_sha256,
        origin=origin,  # type: ignore[arg-type]
    )


def read_manifest(path: str | Path) -> dict[str, Any]:
    import json

    manifest_path = Path(path).with_suffix(".json")
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_state(handle: Any, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load safetensors into a model handle, restoring tied aliases afterwards."""
    from safetensors.torch import load_file

    path = Path(checkpoint_path)
    if path.suffix != ".safetensors":
        raise ValidationFailure(
            f"refusing to load '{path.suffix}' checkpoints",
            hint="only safetensors are supported; pickle-based checkpoints are unsupported by design",
        )
    manifest = read_manifest(path)
    if manifest.get("architecture_hash") and manifest["architecture_hash"] != handle.architecture_hash:
        raise ValidationFailure(
            "checkpoint architecture hash does not match the model definition",
            hint="create the model from the manifest config or use a different checkpoint",
            details={
                "checkpoint_hash": manifest["architecture_hash"],
                "model_hash": handle.architecture_hash,
            },
        )
    tensors = load_file(str(path), device="cpu")
    aliases = manifest.get("aliases", {})
    expected = set(flattened_state(handle.module)["tensors"])
    missing = expected - set(tensors)
    extra = set(tensors) - expected
    if missing or extra:
        raise ValidationFailure(
            "checkpoint tensors do not match the model",
            details={"missing": sorted(missing)[:20], "unexpected": sorted(extra)[:20]},
            hint="the architecture config and the checkpoint must come from the same build",
        )
    handle.module.load_state_dict(tensors, strict=False)
    if aliases:
        restore_aliases(handle.module, aliases)
    handle.module.eval()
    return tensors


def write_artifact(workspace: Workspace, data: bytes, suffix: str = "") -> tuple[str, Path]:
    """Content-addressed blob write used for reports and exports."""
    from .hashing import atomic_write, sha256_hex

    sha = sha256_hex(data)
    target = workspace.artifact_path(sha, suffix)
    if target.exists():
        return sha, target
    with atomic_write(target) as handle:
        handle.write(data)
    return sha, target
