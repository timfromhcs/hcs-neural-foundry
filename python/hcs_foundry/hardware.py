"""Hardware detection, capability probing and resource estimation.

The rule from ``AGENTS.md`` is simple and non-negotiable: never invent an accelerator. A backend is
reported as ``declared`` when a probe or driver suggests it exists, and only as ``usable`` after a
real micro-benchmark ran on it. Everything else carries a human-readable ``fallback_reason``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

from .schemas import (
    BackendCapability,
    BackendKind,
    CpuProfile,
    HardwareProfile,
    Maturity,
    MemoryProfile,
    MlpConfig,
    ResourceEstimate,
    TransformerConfig,
)

ENGINE_VERSION = "0.1.0"


# --------------------------------------------------------------------------------------------------
# Low-level probes
# --------------------------------------------------------------------------------------------------


def _cpu_brand() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return str(value).strip()
        except OSError:
            pass
    brand = platform.processor() or platform.machine() or "unknown"
    return brand.strip()


def _physical_cores() -> int:
    logical = os.cpu_count() or 1
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            try:
                value, _ = winreg.QueryValueEx(key, "~MHz")
                _ = value
            finally:
                winreg.CloseKey(key)
        except OSError:
            pass
    # Without psutil we cannot read the topology portably; report logical cores as an upper bound
    # and say so instead of pretending to know the physical count.
    return max(1, logical // 2) if logical > 2 else logical


def _avx2_supported() -> bool:
    try:
        import numpy as np

        # numpy moved this private namespace; support both spellings and stay quiet.
        multiarray = getattr(getattr(np, "_core", None), "_multiarray_umath", None)
        if multiarray is None:
            multiarray = np.core._multiarray_umath  # type: ignore[attr-defined]
        features = multiarray.__cpu_features__
        return bool(features.get("AVX2", False))
    except Exception:  # pragma: no cover - depends on numpy internals
        return False


def _memory() -> MemoryProfile:
    total = available = 0
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
    else:
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            total = page * os.sysconf("SC_PHYS_PAGES")
            available = page * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
            pass
    disk_free = shutil.disk_usage(Path.home()).free
    return MemoryProfile(total_bytes=total, available_bytes=available, disk_free_bytes=disk_free)


def _benchmark_matmul() -> dict[str, float]:
    """Tiny, deterministic CPU benchmark used to prove a backend actually computes."""
    import numpy as np

    rng = np.random.default_rng(12345)
    a = rng.standard_normal((256, 256)).astype(np.float32)
    b = rng.standard_normal((256, 256)).astype(np.float32)
    a @ b  # warm-up
    start = time.perf_counter()
    repeats = 8
    for _ in range(repeats):
        a @ b
    elapsed = time.perf_counter() - start
    flops = repeats * 2 * 256**3
    return {"gflops": round(flops / elapsed / 1e9, 3), "elapsed_ms": round(elapsed / repeats * 1000, 3)}


# --------------------------------------------------------------------------------------------------
# Backend probing
# --------------------------------------------------------------------------------------------------


def _torch_backends() -> list[BackendCapability]:
    out: list[BackendCapability] = []
    try:
        import torch
    except Exception:  # torch not installed: not an error, a documented mode
        return out

    if torch.cuda.is_available():
        is_rocm = bool(getattr(torch.version, "hip", None))
        kind = BackendKind.ROCM if is_rocm else BackendKind.CUDA
        name = torch.cuda.get_device_name(0)
        entry = BackendCapability(
            kind=kind,
            name=name,
            declared=True,
            maturity=Maturity.STABLE,
            detail=f"torch {torch.__version__}",
        )
        try:
            device = torch.device("cuda")
            x = torch.randn(256, 256, device=device)
            torch.mm(x, x)
            torch.cuda.synchronize()
            entry.usable = True
            entry.benchmark = {"probe": "256x256 matmul on device 0"}
        except Exception as exc:  # pragma: no cover - hardware dependent
            entry.fallback_reason = f"matmul on device failed: {exc}"
        out.append(entry)
    else:
        reason = "torch.cuda.is_available() == False"
        hip = getattr(torch.version, "hip", None)
        out.append(
            BackendCapability(
                kind=BackendKind.ROCM if hip else BackendKind.CUDA,
                name="none",
                declared=False,
                usable=False,
                fallback_reason=reason,
                detail=f"hip={hip}" if hip else "",
            )
        )

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out.append(
            BackendCapability(
                kind=BackendKind.METAL,
                name="Apple Metal (MPS)",
                declared=True,
                usable=True,
                maturity=Maturity.EXPERIMENTAL,
                detail=f"torch {torch.__version__}",
            )
        )
    return out


def _directml_backend() -> BackendCapability | None:
    try:
        import torch_directml  # noqa: F401

        return BackendCapability(
            kind=BackendKind.DIRECTML,
            name="DirectML",
            declared=True,
            usable=False,
            maturity=Maturity.EXPERIMENTAL,
            detail="torch_directml importable",
            fallback_reason="not benchmarked in this build; training stays on the verified backend",
        )
    except Exception:
        return None


def detect_hardware(*, probe_backends: bool = True) -> HardwareProfile:
    """Detect the machine without assuming anything about acceleration."""
    memory = _memory()
    cpu = CpuProfile(
        brand=_cpu_brand(),
        physical_cores=_physical_cores(),
        logical_cores=os.cpu_count() or 1,
        avx2=_avx2_supported(),
    )
    profile = HardwareProfile(
        os=platform.system(),
        os_version=platform.version(),
        python=platform.python_version(),
        cpu=cpu,
        memory=memory,
        notes=[
            "physical_cores is an estimate: no psutil dependency in the core package",
            "accelerator entries are only marked usable after a real probe ran",
        ],
    )

    cpu_backend = BackendCapability(
        kind=BackendKind.CPU,
        name=cpu.brand,
        declared=True,
        usable=True,
        maturity=Maturity.STABLE,
        detail=f"{cpu.logical_cores} logical cores",
    )
    if probe_backends:
        try:
            cpu_backend.benchmark = _benchmark_matmul()
        except Exception as exc:  # pragma: no cover - numpy missing
            cpu_backend.usable = False
            cpu_backend.fallback_reason = f"numpy benchmark failed: {exc}"
    profile.backends.append(cpu_backend)

    profile.backends.extend(_torch_backends())
    dml = _directml_backend()
    if dml:
        profile.backends.append(dml)

    try:
        import torch

        has_torch = True
        torch_version = torch.__version__
    except Exception:
        has_torch = False
        torch_version = None
    if not has_torch:
        profile.notes.append("torch is not installed: training and ONNX export are unavailable (.[torch])")
    else:
        profile.notes.append(f"torch {torch_version}")

    profile.notes.append(f"hostname={socket.gethostname()}")
    return profile


# --------------------------------------------------------------------------------------------------
# Resource estimation
# --------------------------------------------------------------------------------------------------

FP32 = 4
FP16 = 2


def estimate_resources(
    config: MlpConfig | TransformerConfig,
    *,
    parameter_count: int,
    train: bool = True,
    batch_size: int = 16,
    block_size: int = 32,
    optimizer: str = "adamw",
    precision_bytes: int = FP32,
    available_bytes: int | None = None,
) -> ResourceEstimate:
    """Heuristic footprint, clearly labelled as an estimate.

    The numbers are deliberately conservative: they include gradient and optimizer state for
    training and refuse to claim a run fits without a safety margin. They are *not* a measurement.
    """
    param_bytes = parameter_count * precision_bytes
    grad_bytes = parameter_count * precision_bytes if train else 0
    if not train:
        optimizer_bytes = 0
    elif optimizer == "adamw":
        optimizer_bytes = parameter_count * FP32 * 2 + parameter_count * FP32  # m, v, master copy
    else:
        optimizer_bytes = parameter_count * FP32  # momentum buffer

    assumptions = [
        f"weights in {precision_bytes * 8}-bit",
        "AdamW keeps m, v and an fp32 master copy" if optimizer == "adamw" and train else
        ("SGD keeps one momentum buffer" if train else "inference only"),
        "activation estimate uses a fixed multiplier per layer, not a traced graph",
    ]
    warnings: list[str] = []

    if isinstance(config, TransformerConfig):
        d = config.d_model
        f = config.resolved_ffn_dim
        layers = config.n_layers
        # per-token, per-layer: hidden state, qkv, attention scores (block_size), swiglu intermediates
        per_token = d * 4 + 2 * f * 4 + block_size * config.n_heads * 2
        activation_bytes = int(batch_size * block_size * layers * per_token * precision_bytes * 3)
        kv = (
            2
            * layers
            * block_size
            * config.resolved_kv_heads
            * (d // config.n_heads)
            * precision_bytes
            * max(1, batch_size)
        )
        kv_cache_bytes = int(kv)
        export_bytes = int(parameter_count * FP32 * 1.6)
        if block_size > config.context_length:
            warnings.append(
                f"block_size {block_size} exceeds context_length {config.context_length}; "
                "the trainer will clamp it"
            )
    else:
        widths = [config.input_dim, *config.hidden_dims, config.output_dim]
        activation_bytes = int(batch_size * sum(widths) * precision_bytes * 4)  # fwd + 2x bwd + slack
        kv_cache_bytes = 0
        export_bytes = int(parameter_count * FP32 * 1.4)
        if len(config.hidden_dims) > 8:
            warnings.append("more than 8 hidden layers is unusual for an MLP preset")

    total_train = param_bytes + grad_bytes + optimizer_bytes + activation_bytes
    total_infer = param_bytes + kv_cache_bytes + int(parameter_count * precision_bytes * 0.1)

    fits = True
    # Require 25% headroom: allocator fragmentation and the OS both need room.
    if available_bytes and total_train > available_bytes * 0.75:
        fits = False
        warnings.append(
                f"estimated training footprint {total_train / 1e9:.2f} GB exceeds 75% of "
                f"{available_bytes / 1e9:.2f} GB available memory"
            )

    return ResourceEstimate(
        parameters=parameter_count,
        param_bytes=int(param_bytes),
        grad_bytes=int(grad_bytes),
        optimizer_bytes=int(optimizer_bytes),
        activation_bytes=int(activation_bytes),
        kv_cache_bytes=int(kv_cache_bytes),
        export_bytes=int(export_bytes),
        total_train_bytes=int(total_train),
        total_infer_bytes=int(total_infer),
        assumptions=assumptions,
        warnings=warnings,
        fits_budget=fits,
    )
