"""Pydantic schemas: the single source of truth for every public payload.

Design rules
------------
1. Every field that a user or an agent can influence is validated here, not in the callers.
2. Nothing in this module imports torch. Schemas must be importable on a machine without ML
   dependencies so that inspection, validation and the API can degrade gracefully.
3. Enum values are stable wire formats. Renaming one is a breaking schema change.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ValidationFailure
from .hashing import hash_json, utcnow

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------------------


class Maturity(StrEnum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class FamilyId(StrEnum):
    MLP = "mlp"
    TRANSFORMER_LM = "transformer-lm"


class BudgetPolicy(StrEnum):
    EXACT = "exact"
    CLOSEST = "closest"
    CLOSEST_UNDER = "closest_under"
    AT_LEAST = "at_least"


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INVALID = "invalid"
    INTERRUPTED = "interrupted"


TERMINAL_STATES = frozenset(
    {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED, RunState.INVALID, RunState.INTERRUPTED}
)


class PatchOperation(StrEnum):
    SET_SCALAR = "set_scalar"
    SET_BLOCK = "set_block"
    INIT_SEEDED = "init_seeded"
    ADD_SCALED_DELTA = "add_scaled_delta"
    LOW_RANK_DELTA = "low_rank_delta"
    NORMALIZE = "normalize"
    CLAMP = "clamp"
    SCALE = "scale"
    SPARSITY_MASK = "sparsity_mask"
    INTERPOLATE = "interpolate"
    COPY_FROM = "copy_from"
    SET_TRAINABLE = "set_trainable"


class BackendKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    DIRECTML = "directml"
    METAL = "metal"
    WEBGPU = "webgpu"
    WASM = "wasm"


# --------------------------------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------------------------------


class MlpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["mlp"] = "mlp"
    input_dim: int = Field(16, ge=1, le=1_000_000)
    hidden_dims: tuple[int, ...] = (32,)
    output_dim: int = Field(4, ge=1, le=1_000_000)
    activation: Literal["relu", "gelu", "tanh"] = "gelu"
    bias: bool = True
    seed: int = Field(0, ge=0, le=2**31 - 1)

    @field_validator("hidden_dims")
    @classmethod
    def _valid_hidden(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) > 16:
            raise ValueError("at most 16 hidden layers are allowed")
        if any(dim < 1 or dim > 1_000_000 for dim in value):
            raise ValueError("hidden dimensions must be between 1 and 1,000,000")
        return value


class TransformerConfig(BaseModel):
    """Decoder-only transformer with RoPE, GQA and SwiGLU.

    ``ffn_dim=None`` resolves to the conventional ``round_to_multiple(8/3 * d_model, 64)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["transformer-lm"] = "transformer-lm"
    vocab_size: int = Field(64, ge=2, le=1_000_000)
    d_model: int = Field(64, ge=8, le=8192)
    n_layers: int = Field(2, ge=1, le=128)
    n_heads: int = Field(4, ge=1, le=128)
    n_kv_heads: int | None = Field(None, ge=1, le=128)
    ffn_dim: int | None = Field(None, ge=1, le=65536)
    context_length: int = Field(64, ge=8, le=131072)
    tie_embeddings: bool = True
    rope_theta: float = Field(10000.0, gt=1.0)
    norm_eps: float = Field(1e-6, gt=0, lt=1e-2)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    seed: int = Field(0, ge=0, le=2**31 - 1)

    @field_validator("n_kv_heads")
    @classmethod
    def _kv_default(cls, value: int | None) -> int | None:
        return value

    @model_validator(mode="after")
    def _check_shapes(self) -> TransformerConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        if self.n_heads % kv != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({kv})")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        return self

    @property
    def resolved_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def resolved_ffn_dim(self) -> int:
        if self.ffn_dim is not None:
            return self.ffn_dim
        raw = int(8 * self.d_model / 3)
        return max(64, ((raw + 63) // 64) * 64)


ArchitectureConfig = Annotated[MlpConfig | TransformerConfig, Field(discriminator="family")]


def architecture_hash(config: MlpConfig | TransformerConfig) -> str:
    """Stable hash of an architecture. Two configs produce the same hash iff they are identical."""
    return hash_json({"schema_version": SCHEMA_VERSION, "config": config.model_dump(mode="json")})


# --------------------------------------------------------------------------------------------------
# Parameter and resource reports
# --------------------------------------------------------------------------------------------------


class ParameterEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    shape: tuple[int, ...]
    numel: int
    trainable: bool = True
    tied_to: str | None = None
    dtype: str = "float32"


class ParameterReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    trainable: int
    unique_tensors: int
    shared_tensors: int
    entries: list[ParameterEntry] = Field(default_factory=list)
    architecture_hash: str = ""
    exact: bool = True

    def by_module(self, limit: int = 64) -> dict[str, int]:
        """Aggregate numel by the first two dotted components of the tensor name."""
        out: dict[str, int] = {}
        for entry in self.entries:
            if entry.tied_to:
                continue
            key = ".".join(entry.name.split(".")[:2])
            out[key] = out.get(key, 0) + entry.numel
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:limit])


class ResourceEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: int
    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    kv_cache_bytes: int
    export_bytes: int
    total_train_bytes: int
    total_infer_bytes: int
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fits_budget: bool = True


# --------------------------------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------------------------------


class CpuProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str = "unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    max_clock_mhz: int | None = None
    avx2: bool = False


class MemoryProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_bytes: int = 0
    available_bytes: int = 0
    disk_free_bytes: int = 0


class BackendCapability(BaseModel):
    """A backend is only ``usable`` after a micro-benchmark actually ran on it."""

    model_config = ConfigDict(extra="forbid")

    kind: BackendKind
    name: str
    declared: bool = False
    usable: bool = False
    detail: str = ""
    fallback_reason: str | None = None
    benchmark: dict[str, Any] = Field(default_factory=dict)
    maturity: Maturity = Maturity.EXPERIMENTAL


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    os: str = ""
    os_version: str = ""
    python: str = ""
    cpu: CpuProfile = Field(default_factory=CpuProfile)
    memory: MemoryProfile = Field(default_factory=MemoryProfile)
    backends: list[BackendCapability] = Field(default_factory=list)
    detected_at: str = Field(default_factory=utcnow)
    notes: list[str] = Field(default_factory=list)

    def backend(self, kind: BackendKind) -> BackendCapability | None:
        return next((b for b in self.backends if b.kind == kind), None)

    @property
    def preferred_train_backend(self) -> BackendCapability:
        order = (BackendKind.CUDA, BackendKind.ROCM, BackendKind.DIRECTML, BackendKind.MPS, BackendKind.CPU)
        for kind in order:
            backend = self.backend(kind)
            if backend and backend.usable:
                return backend
        return self.backend(BackendKind.CPU) or BackendCapability(
            kind=BackendKind.CPU, name="cpu", declared=True, usable=True
        )


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------


class ModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str
    name: str
    config: ArchitectureConfig
    architecture_hash: str
    parameter_count: int
    created_at: str = Field(default_factory=utcnow)
    notes: str = ""
    maturity: Maturity = Maturity.STABLE
    source: Literal["created", "imported", "generated", "evolved", "patched"] = "created"


class ModelSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    family: str
    parameter_count: int
    architecture_hash: str
    created_at: str
    checkpoints: int = 0
    best_score: float | None = None


class CheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model_id: str
    sha256: str
    path: str
    bytes: int
    created_at: str = Field(default_factory=utcnow)
    label: str = "checkpoint"
    parent_sha256: str | None = None
    patches: list[str] = Field(default_factory=list)
    origin: Literal["init", "training", "patch", "generated", "evolved", "import"] = "init"


class TensorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str
    trainable: bool
    min: float
    max: float
    mean: float
    std: float
    norm: float
    zeros_fraction: float
    finite: bool
    tied_to: str | None = None


class TensorSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    offset: tuple[int, ...]
    shape: tuple[int, ...]
    values: list[float]
    truncated: bool = False


class TensorHistogram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    bins: list[float]
    counts: list[int]
    sampled: int


# --------------------------------------------------------------------------------------------------
# Weight patches
# --------------------------------------------------------------------------------------------------

_OP_PARAMS: dict[PatchOperation, tuple[frozenset[str], frozenset[str]]] = {
    PatchOperation.SET_SCALAR: (frozenset({"index", "value"}), frozenset()),
    PatchOperation.SET_BLOCK: (frozenset({"start", "block_shape", "mode"}), frozenset({"value", "seed", "std"})),
    PatchOperation.INIT_SEEDED: (frozenset({"scheme", "seed"}), frozenset({"std", "fan_mode"})),
    PatchOperation.ADD_SCALED_DELTA: (frozenset({"scale", "std", "seed"}), frozenset()),
    PatchOperation.LOW_RANK_DELTA: (frozenset({"rank", "alpha", "seed"}), frozenset({"std"})),
    PatchOperation.NORMALIZE: (frozenset({"axis", "eps"}), frozenset()),
    PatchOperation.CLAMP: (frozenset(), frozenset({"min", "max"})),
    PatchOperation.SCALE: (frozenset({"factor"}), frozenset()),
    PatchOperation.SPARSITY_MASK: (frozenset({"keep_fraction", "seed"}), frozenset({"structured"})),
    PatchOperation.INTERPOLATE: (frozenset({"other_checkpoint_sha256", "t"}), frozenset()),
    PatchOperation.COPY_FROM: (frozenset({"source_checkpoint_sha256", "source_tensor"}), frozenset()),
    PatchOperation.SET_TRAINABLE: (frozenset({"trainable"}), frozenset()),
}

# Guard rails against a single patch requesting absurd work.
MAX_PATCH_ELEMENTS = 50_000_000
MAX_LOW_RANK = 512


def validate_operation_params(operation: PatchOperation, params: dict[str, Any]) -> dict[str, Any]:
    """Reject unknown/extra keys and out-of-range values *before* any tensor is touched."""
    required, optional = _OP_PARAMS[operation]
    keys = set(params)
    unknown = keys - required - optional
    if unknown:
        raise ValidationFailure(
            f"unknown parameters for {operation.value}: {sorted(unknown)}",
            hint=f"allowed: {sorted(required | optional)}",
        )
    missing = required - keys
    if missing:
        raise ValidationFailure(f"missing parameters for {operation.value}: {sorted(missing)}")

    def finite(name: str) -> float:
        value = params[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationFailure(f"{name} must be a number")
        if not math.isfinite(float(value)):
            raise ValidationFailure(f"{name} must be finite")
        return float(value)

    def integer(name: str, lo: int, hi: int) -> int:
        value = params[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationFailure(f"{name} must be an integer")
        if not lo <= value <= hi:
            raise ValidationFailure(f"{name} must be between {lo} and {hi}")
        return value

    if operation is PatchOperation.SET_SCALAR:
        index = params["index"]
        if not isinstance(index, (list, tuple)) or not index:
            raise ValidationFailure("index must be a non-empty list of integers")
        for pos in index:
            integer_check = isinstance(pos, int) and not isinstance(pos, bool) and pos >= 0
            if not integer_check:
                raise ValidationFailure("index entries must be non-negative integers")
        finite("value")
    elif operation is PatchOperation.SET_BLOCK:
        if params["mode"] not in {"zeros", "ones", "value", "normal"}:
            raise ValidationFailure("mode must be one of zeros|ones|value|normal")
        if params["mode"] == "value":
            finite("value")
        if params["mode"] == "normal":
            if "std" not in params:
                raise ValidationFailure("std is required when mode=normal")
            finite("std")
            integer("seed", 0, 2**31 - 1)
    elif operation is PatchOperation.INIT_SEEDED:
        if params["scheme"] not in {"xavier", "kaiming", "normal", "zeros", "ones", "uniform"}:
            raise ValidationFailure("scheme must be one of xavier|kaiming|normal|zeros|ones|uniform")
        integer("seed", 0, 2**31 - 1)
        if "std" in params and finite("std") <= 0:
            raise ValidationFailure("std must be > 0")
        if "fan_mode" in params and params["fan_mode"] not in {"fan_in", "fan_out"}:
            raise ValidationFailure("fan_mode must be fan_in or fan_out")
    elif operation is PatchOperation.ADD_SCALED_DELTA:
        finite("scale")
        if finite("std") <= 0:
            raise ValidationFailure("std must be > 0")
        integer("seed", 0, 2**31 - 1)
    elif operation is PatchOperation.LOW_RANK_DELTA:
        integer("rank", 1, MAX_LOW_RANK)
        if finite("alpha") <= 0:
            raise ValidationFailure("alpha must be > 0")
        integer("seed", 0, 2**31 - 1)
        if "std" in params and finite("std") <= 0:
            raise ValidationFailure("std must be > 0")
    elif operation is PatchOperation.NORMALIZE:
        if params["axis"] not in (-1, 0, 1):
            raise ValidationFailure("axis must be -1, 0 or 1")
        if finite("eps") <= 0:
            raise ValidationFailure("eps must be > 0")
    elif operation is PatchOperation.CLAMP:
        if "min" not in params and "max" not in params:
            raise ValidationFailure("clamp requires at least one of min|max")
        low = finite("min") if "min" in params else None
        high = finite("max") if "max" in params else None
        if low is not None and high is not None and low >= high:
            raise ValidationFailure("min must be < max")
    elif operation is PatchOperation.SCALE:
        if not math.isfinite(finite("factor")):
            raise ValidationFailure("factor must be finite")
    elif operation is PatchOperation.SPARSITY_MASK:
        keep = finite("keep_fraction")
        if not 0.0 < keep <= 1.0:
            raise ValidationFailure("keep_fraction must be in (0, 1]")
        integer("seed", 0, 2**31 - 1)
        if "structured" in params and params["structured"] not in {"none", "rows", "cols"}:
            raise ValidationFailure("structured must be none|rows|cols")
    elif operation is PatchOperation.INTERPOLATE:
        other = params["other_checkpoint_sha256"]
        if not isinstance(other, str) or len(other) != 64:
            raise ValidationFailure("other_checkpoint_sha256 must be a 64-char sha256")
        t = finite("t")
        if not 0.0 <= t <= 1.0:
            raise ValidationFailure("t must be in [0, 1]")
    elif operation is PatchOperation.COPY_FROM:
        other = params["source_checkpoint_sha256"]
        if not isinstance(other, str) or len(other) != 64:
            raise ValidationFailure("source_checkpoint_sha256 must be a 64-char sha256")
        if not isinstance(params["source_tensor"], str) or not params["source_tensor"]:
            raise ValidationFailure("source_tensor must be a non-empty string")
    elif operation is PatchOperation.SET_TRAINABLE:
        if not isinstance(params["trainable"], bool):
            raise ValidationFailure("trainable must be a boolean")
    return params


class PatchPrecondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, ...]
    dtype: str = "float32"
    base_checkpoint_sha256: str


class PatchValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finite: bool = True
    max_abs_delta: float = 0.0
    changed_elements: int = 0
    norm_before: float = 0.0
    norm_after: float = 0.0
    relative_norm_change: float = 0.0


class WeightPatch(BaseModel):
    """Immutable description of one weight edit. Application is a separate, verified step."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    patch_id: str
    base_checkpoint_sha256: str
    architecture_hash: str
    operation: PatchOperation
    target_tensor: str
    parameters: dict[str, Any]
    precondition: PatchPrecondition
    created_by: str = "user"
    created_at: str = Field(default_factory=utcnow)
    note: str = ""
    status: Literal["proposed", "applied", "rejected"] = "proposed"

    @model_validator(mode="after")
    def _validate(self) -> WeightPatch:
        if len(self.base_checkpoint_sha256) != 64:
            raise ValueError("base_checkpoint_sha256 must be a 64-char sha256")
        if len(self.architecture_hash) != 64:
            raise ValueError("architecture_hash must be a 64-char sha256")
        if not self.target_tensor or ".." in self.target_tensor:
            raise ValueError("target_tensor must be a valid tensor name")
        if not self.created_by or len(self.created_by) > 120:
            raise ValueError("created_by must be 1..120 characters")
        object.__setattr__(self, "parameters", validate_operation_params(self.operation, self.parameters))
        if self.precondition.base_checkpoint_sha256 != self.base_checkpoint_sha256:
            raise ValueError("precondition.base_checkpoint_sha256 must match base_checkpoint_sha256")
        return self


class PatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: WeightPatch
    validation: PatchValidation
    checkpoint: CheckpointRecord


# --------------------------------------------------------------------------------------------------
# Datasets, training, evaluation
# --------------------------------------------------------------------------------------------------


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: Literal["text", "classification", "regression"]
    path: str
    sha256: str
    train_rows: int = 0
    val_rows: int = 0
    license: str = "synthetic (generated in-repo)"
    notes: str = ""


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["adamw", "sgd"] = "adamw"
    lr: float = Field(3e-3, gt=0, le=10)
    weight_decay: float = Field(0.01, ge=0, le=1)
    beta1: float = Field(0.9, ge=0, lt=1)
    beta2: float = Field(0.95, ge=0, lt=1)
    momentum: float = Field(0.9, ge=0, lt=1)
    grad_clip: float = Field(1.0, ge=0, le=100)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    dataset_id: str = "tiny-text"
    steps: int = Field(200, ge=1, le=200_000)
    batch_size: int = Field(16, ge=1, le=4096)
    block_size: int = Field(32, ge=1, le=8192)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    device: str = "auto"
    eval_every: int = Field(50, ge=1, le=100_000)
    checkpoint_every: int = Field(100, ge=1, le=100_000)
    max_seconds: int = Field(600, ge=1, le=86_400)
    lora_rank: int | None = Field(None, ge=1, le=512)


class MetricPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    loss: float
    lr: float
    grad_norm: float = 0.0
    tokens: int = 0
    elapsed_s: float = 0.0
    peak_memory_bytes: int = 0
    val_loss: float | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str = "0.1.0"
    python: str = ""
    torch: str | None = None
    platform: str = ""
    device: str = "cpu"
    seeds: dict[str, int] = Field(default_factory=dict)
    dataset_sha256: str | None = None
    architecture_hash: str | None = None
    base_checkpoint_sha256: str | None = None
    evaluator_version: str = "1"
    notes: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str
    kind: Literal["training", "evaluation", "evolution", "generation", "export", "reinforcement"]
    model_id: str
    state: RunState = RunState.QUEUED
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: list[MetricPoint] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    artifacts: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None
    error: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class EvalMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    higher_is_better: bool = True
    unit: str = ""
    baseline: float | None = None
    definition: str = ""


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    suite_id: str
    suite_version: str = "1"
    model_id: str
    checkpoint_sha256: str
    state: RunState = RunState.COMPLETED
    metrics: list[EvalMetric] = Field(default_factory=list)
    per_example: list[dict[str, Any]] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)

    def metric(self, name: str) -> float | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric.value
        return None


# --------------------------------------------------------------------------------------------------
# Agent proposals
# --------------------------------------------------------------------------------------------------


class AgentProposal(BaseModel):
    """What an agent is allowed to produce. It never writes tensors directly."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    proposal_id: str
    agent_id: str
    goal: str
    hypothesis: str
    model_id: str
    base_checkpoint_sha256: str
    patches: list[WeightPatch] = Field(default_factory=list)
    expected_effect: str = ""
    validation_plan: list[str] = Field(default_factory=list)
    estimated_cost: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)
    approval: Literal["pending", "approved", "rejected"] = "pending"
    approved_by: str | None = None
    outcome: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _budget(self) -> AgentProposal:
        if len(self.patches) > 32:
            raise ValueError("a proposal may contain at most 32 patches")
        for patch in self.patches:
            if patch.base_checkpoint_sha256 != self.base_checkpoint_sha256:
                raise ValueError("all patches must target the proposal base checkpoint")
        return self


class EvolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    population: int = Field(16, ge=2, le=512)
    generations: int = Field(5, ge=1, le=200)
    sigma: float = Field(0.02, gt=0, le=1)
    lr: float = Field(0.05, gt=0, le=10)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    evaluations_per_candidate: int = Field(1, ge=1, le=16)
    baseline_steps: int = Field(0, ge=0, le=100_000)
    dataset_id: str = "tiny-text"


class RlConfig(BaseModel):
    """GRPO on a toy verifiable-reward environment (not language-model post-training)."""

    model_config = ConfigDict(extra="forbid")

    env_id: str = "rule-bandit"
    steps: int = Field(200, ge=1, le=20_000)
    group_size: int = Field(8, ge=2, le=64)
    batch_contexts: int = Field(16, ge=1, le=1024)
    lr: float = Field(0.1, gt=0, le=10)
    kl_beta: float = Field(0.02, ge=0, le=10)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    eval_contexts: int = Field(256, ge=8, le=100_000)
    max_contexts: int = Field(20_000, ge=1, le=10_000_000)
    context_dim: int = Field(6, ge=2, le=1024)
    n_actions: int = Field(4, ge=2, le=1024)


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_model_id: str
    corpus: list[int] = Field(default_factory=list)
    epochs: int = Field(200, ge=1, le=100_000)
    latent_dim: int = Field(8, ge=1, le=256)
    hidden: int = Field(64, ge=1, le=4096)
    lr: float = Field(1e-3, gt=0, le=1)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    samples: int = Field(4, ge=1, le=64)
