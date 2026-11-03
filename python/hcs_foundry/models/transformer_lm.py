"""Decoder-only transformer language model: RoPE, grouped-query attention, SwiGLU, RMSNorm.

Scope, stated plainly: this is an educational-scale implementation that trains from scratch on
character-level text. It is not an adapter for pretrained checkpoints, and nothing here should be
read as an LLM capability claim.
"""

from __future__ import annotations

import math
from typing import Any

from ..hardware import estimate_resources
from ..schemas import Maturity, ResourceEstimate, TransformerConfig, architecture_hash
from .base import (
    ModelHandle,
    SmokeTestReport,
    ValidationIssue,
    count_parameters,
    seed_everything,
    torch_module,
)


def predict_parameters(config: TransformerConfig) -> int:
    """Exact-by-construction count; a test asserts equality with the instantiated module."""
    d = config.d_model
    f = config.resolved_ffn_dim
    kv = config.resolved_kv_heads
    hd = d // config.n_heads
    per_block = 2 * d * d + 2 * d * kv * hd + 3 * d * f + 2 * d
    total = config.vocab_size * d + config.n_layers * per_block + d
    if not config.tie_embeddings:
        total += config.vocab_size * d
    return int(total)


def _build_module(config: TransformerConfig):  # noqa: ANN202
    torch = torch_module()
    nn = torch.nn
    F = torch.nn.functional

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):  # noqa: ANN001, ANN201
            var = x.float().pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(var + self.eps)).type_as(x) * self.weight

    def rope_tables(context: int, head_dim: int, theta: float, dtype):  # noqa: ANN001, ANN202
        # One frequency per coordinate pair, then interleaved so the table lines up with the
        # (x0, x1, x2, x3, ...) layout used by ``apply_rope``.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(context, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq).repeat_interleave(2, dim=-1)
        return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)

    def apply_rope(x, cos, sin):  # noqa: ANN001, ANN201
        # x: (B, H, T, hd) -> rotate pairs
        t = x.shape[2]
        cos_t = cos[:t].unsqueeze(0).unsqueeze(0)
        sin_t = sin[:t].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos_t) + (rotated * sin_t)

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_heads = config.n_heads
            self.n_kv_heads = config.resolved_kv_heads
            self.head_dim = config.d_model // config.n_heads
            self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)
            self.dropout = config.dropout

        def forward(self, x, cos, sin):  # noqa: ANN001, ANN201
            batch, seq, _ = x.shape
            q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            if self.n_kv_heads != self.n_heads:
                repeat = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            if self.dropout and self.training:
                weights = F.dropout(weights, p=self.dropout)
            out = weights @ v
            out = out.transpose(1, 2).contiguous().view(batch, seq, self.n_heads * self.head_dim)
            return self.o_proj(out)

    class SwiGlu(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ffn = config.resolved_ffn_dim
            self.w1 = nn.Linear(config.d_model, ffn, bias=False)
            self.w3 = nn.Linear(config.d_model, ffn, bias=False)
            self.w2 = nn.Linear(ffn, config.d_model, bias=False)

        def forward(self, x):  # noqa: ANN001, ANN201
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = RMSNorm(config.d_model, config.norm_eps)
            self.attn = Attention()
            self.norm2 = RMSNorm(config.d_model, config.norm_eps)
            self.mlp = SwiGlu()

        def forward(self, x, cos, sin):  # noqa: ANN001, ANN201
            x = x + self.attn(self.norm1(x), cos, sin)
            x = x + self.mlp(self.norm2(x))
            return x

    class TransformerLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(config.n_layers)])
            self.norm_f = RMSNorm(config.d_model, config.norm_eps)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.tok_emb.weight
            cos, sin = rope_tables(
                config.context_length,
                config.d_model // config.n_heads,
                config.rope_theta,
                torch.float32,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.apply(self._init)
            # scaled init for the residual output projections keeps variance stable with depth
            for name, param in self.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        @staticmethod
        def _init(module) -> None:  # noqa: ANN001
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):  # noqa: ANN001, ANN201
            batch, seq = idx.shape
            if not torch.jit.is_tracing() and seq > config.context_length:
                raise ValueError(f"sequence length {seq} exceeds context_length {config.context_length}")
            x = self.tok_emb(idx)
            for block in self.blocks:
                x = block(x, self.rope_cos, self.rope_sin)
            x = self.norm_f(x)
            logits = self.lm_head(x)
            if targets is None:
                return logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            return logits, loss

    return TransformerLM()


class TransformerFamily:
    family_id = "transformer-lm"
    display_name = "Decoder-only transformer (character or token level)"
    maturity = Maturity.STABLE

    def validate_config(self, config: dict[str, Any]) -> list[ValidationIssue]:
        try:
            parsed = TransformerConfig.model_validate(config)
        except Exception as exc:
            return [ValidationIssue(field="config", message=str(exc))]
        issues: list[ValidationIssue] = []
        if predict_parameters(parsed) > 2_000_000_000:
            issues.append(
                ValidationIssue("d_model", "model would exceed the 2B parameter guard rail of this build")
            )
        if parsed.context_length > 4096:
            issues.append(
                ValidationIssue(
                    "context_length",
                    "context lengths beyond 4096 are untested in this build (memory and attention cost)",
                    severity="warning",
                )
            )
        return issues

    def build(self, config: TransformerConfig) -> ModelHandle:
        torch_module()
        seed_everything(config.seed)
        module = _build_module(config)
        digest = architecture_hash(config)
        report = count_parameters(module, architecture_hash_value=digest)
        notes = []
        predicted = predict_parameters(config)
        if predicted != report.total:
            notes.append(
                f"parameter formula ({predicted}) disagreed with enumerated count ({report.total}); "
                "the enumerated count is authoritative"
            )
        return ModelHandle(
            config=config,
            module=module,
            report=report,
            architecture_hash=digest,
            family=self.family_id,
            notes=notes,
        )

    def predict_parameters(self, config: TransformerConfig) -> int:
        return predict_parameters(config)

    def estimate_resources(self, config: TransformerConfig, **kwargs: Any) -> ResourceEstimate:
        kwargs.pop("parameter_count", None)
        return estimate_resources(config, parameter_count=predict_parameters(config), **kwargs)

    def smoke_test(self, handle: ModelHandle) -> SmokeTestReport:
        torch = torch_module()
        checks: list[str] = []
        failures: list[str] = []
        handle.module.eval()
        with torch.no_grad():
            idx = torch.zeros(2, min(8, handle.config.context_length), dtype=torch.long)
            logits = handle.module(idx)
            checks.append("forward pass executes")
            expected = (2, idx.shape[1], handle.config.vocab_size)
            if tuple(logits.shape) != expected:
                failures.append(f"logit shape {tuple(logits.shape)} != {expected}")
            else:
                checks.append("logit shape matches config")
            if not torch.isfinite(logits).all():
                failures.append("logits contain non-finite values")
            else:
                checks.append("logits are finite")
            idx2 = torch.randint(0, handle.config.vocab_size, (2, 8))
            _, loss = handle.module(idx2, idx2)
            if not torch.isfinite(loss):
                failures.append("loss is not finite")
            else:
                checks.append(f"loss computes (initial {loss.item():.4f})")
        return SmokeTestReport(passed=not failures, checks=checks, failures=failures)


FAMILY = TransformerFamily()
