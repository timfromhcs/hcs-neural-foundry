"""ONNX export with a real numerical parity check.

The export is only considered successful if ONNX Runtime reproduces the native outputs within a
documented tolerance. A silent export is not a deliverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .errors import MissingDependencyError, ValidationFailure
from .models.base import ModelHandle, torch_module
from .schemas import TransformerConfig

DEFAULT_OPSET = 17


def _torch_export(module: Any, args: tuple[Any, ...], path: Path, input_names: list[str], output_names: list[str], dynamic_axes: dict[str, Any]) -> None:
    torch = torch_module()
    kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": DEFAULT_OPSET,
        "do_constant_folding": True,
    }
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kwargs)
    except TypeError:
        # Older/newer signatures without the dynamo flag.
        torch.onnx.export(module, args, str(path), **kwargs)


def export_onnx(handle: ModelHandle, output_path: str | Path) -> dict[str, Any]:
    """Export to ONNX and return parity information. Raises if the export is unusable."""
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional extra
        raise MissingDependencyError(
            "ONNX export needs the export extra",
            hint='install: uv pip install -e ".[export]"',
            details={"import_error": str(exc)},
        ) from exc

    torch = torch_module()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = handle.module
    module.eval()

    if isinstance(handle.config, TransformerConfig):
        seq = min(8, handle.config.context_length)
        sample = torch.zeros(1, seq, dtype=torch.long)
        input_names = ["token_ids"]
        dynamic_axes = {"token_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}}
        numpy_input = {"token_ids": np.zeros((2, seq), dtype=np.int64)}
    else:
        sample = torch.zeros(1, handle.config.input_dim, dtype=torch.float32)
        input_names = ["features"]
        dynamic_axes = {"features": {0: "batch"}, "logits": {0: "batch"}}
        numpy_input = {"features": np.zeros((2, handle.config.input_dim), dtype=np.float32)}

    with torch.no_grad():
        _torch_export(module, (sample,), path, input_names, ["logits"], dynamic_axes)

    if not path.exists() or path.stat().st_size == 0:
        raise ValidationFailure("ONNX export produced no file")

    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    with torch.no_grad():
        if isinstance(handle.config, TransformerConfig):
            native = module(torch.from_numpy(numpy_input["token_ids"]).long())
        else:
            native = module(torch.from_numpy(numpy_input["features"]))
        native_array = native.detach().cpu().numpy()

    onnx_output = session.run(None, numpy_input)[0]
    if native_array.shape != onnx_output.shape:
        raise ValidationFailure(
            f"ONNX output shape {onnx_output.shape} != native shape {native_array.shape}"
        )
    max_abs = float(np.max(np.abs(native_array - onnx_output)))
    denom = float(np.max(np.abs(native_array))) or 1.0
    relative = max_abs / denom
    tolerance = 1e-4
    if relative > tolerance:
        raise ValidationFailure(
            f"ONNX export diverges from the native model (relative error {relative:.2e})",
            hint="tighten the model or raise the tolerance deliberately; do not ship a mismatched graph",
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "opset": DEFAULT_OPSET,
        "provider": session.get_providers()[0] if session.get_providers() else "unknown",
        "max_abs_diff": max_abs,
        "relative_diff": relative,
        "tolerance": tolerance,
        "input_names": input_names,
        "output_names": ["logits"],
        "parity_ok": True,
        "note": "parity measured on CPU with ONNX Runtime; browser providers must be re-validated in situ",
    }


def inspect_onnx(path: str | Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    return {
        "ir_version": model.ir_version,
        "opset": [{"domain": op.domain or "ai.onnx", "version": op.version} for op in model.opset_import],
        "inputs": [{"name": i.name, "type": i.type.tensor_type.elem_type} for i in model.graph.input],
        "outputs": [{"name": o.name, "type": o.type.tensor_type.elem_type} for o in model.graph.output],
        "node_count": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }
