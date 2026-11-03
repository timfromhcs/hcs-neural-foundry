"""A small, fully connected MLP family.

Kept deliberately simple so it can serve as the reference target for weight patches, evolution and
the hypernetwork generator: the parameter space is small enough to be followed by a human.
"""

from __future__ import annotations

from typing import Any

from ..hardware import estimate_resources
from ..schemas import Maturity, MlpConfig, ResourceEstimate, architecture_hash
from .base import (
    ModelHandle,
    SmokeTestReport,
    ValidationIssue,
    count_parameters,
    seed_everything,
    torch_module,
)

ACTIVATIONS = {"relu": "ReLU", "gelu": "GELU", "tanh": "Tanh"}


def _build_module(config: MlpConfig):  # noqa: ANN202
    torch = torch_module()
    nn = torch.nn

    class Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths = [config.input_dim, *config.hidden_dims, config.output_dim]
            self.layers = nn.ModuleList(
                [nn.Linear(widths[i], widths[i + 1], bias=config.bias) for i in range(len(widths) - 1)]
            )
            self.activation_name = config.activation

        def forward(self, x):  # noqa: ANN001, ANN201
            for index, layer in enumerate(self.layers):
                x = layer(x)
                if index < len(self.layers) - 1:
                    if self.activation_name == "relu":
                        x = torch.relu(x)
                    elif self.activation_name == "gelu":
                        x = torch.nn.functional.gelu(x)
                    else:
                        x = torch.tanh(x)
            return x

        def describe(self) -> dict[str, Any]:
            return {"widths": [config.input_dim, *config.hidden_dims, config.output_dim]}

    return Mlp()


def predict_parameters(config: MlpConfig) -> int:
    widths = [config.input_dim, *config.hidden_dims, config.output_dim]
    bias = 1 if config.bias else 0
    return sum(widths[i] * widths[i + 1] + bias * widths[i + 1] for i in range(len(widths) - 1))


class MlpFamily:
    family_id = "mlp"
    display_name = "Multi-layer perceptron"
    maturity = Maturity.STABLE

    def validate_config(self, config: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        try:
            parsed = MlpConfig.model_validate(config)
        except Exception as exc:
            return [ValidationIssue(field="config", message=str(exc))]
        if predict_parameters(parsed) > 500_000_000:
            issues.append(ValidationIssue("hidden_dims", "model would exceed the 500M parameter guard rail"))
        return issues

    def build(self, config: MlpConfig) -> ModelHandle:
        torch = torch_module()  # noqa: F841 - ensures a clear error if torch is missing
        seed_everything(config.seed)
        module = _build_module(config)
        digest = architecture_hash(config)
        report = count_parameters(module, architecture_hash_value=digest)
        return ModelHandle(
            config=config, module=module, report=report, architecture_hash=digest, family=self.family_id
        )

    def predict_parameters(self, config: MlpConfig) -> int:
        return predict_parameters(config)

    def estimate_resources(self, config: MlpConfig, **kwargs: Any) -> ResourceEstimate:
        kwargs.pop("parameter_count", None)
        return estimate_resources(config, parameter_count=predict_parameters(config), **kwargs)

    def smoke_test(self, handle: ModelHandle) -> SmokeTestReport:
        torch = torch_module()
        checks: list[str] = []
        failures: list[str] = []
        handle.module.eval()
        with torch.no_grad():
            x = torch.zeros(2, handle.config.input_dim)
            y = handle.module(x)
            checks.append("forward pass executes")
            if tuple(y.shape) != (2, handle.config.output_dim):
                failures.append(f"output shape {tuple(y.shape)} != (2, {handle.config.output_dim})")
            else:
                checks.append("output shape matches config")
            if not torch.isfinite(y).all():
                failures.append("output contains non-finite values")
            else:
                checks.append("output is finite")
        return SmokeTestReport(passed=not failures, checks=checks, failures=failures)

    def sample_batch(self, handle: ModelHandle, *, batch_size: int, block_size: int) -> dict[str, Any]:
        """Deterministic synthetic regression batch, seeded by step index upstream."""
        from ..training.datasets import make_mlp_batch

        return make_mlp_batch(handle.config, batch_size=batch_size, seed=handle.config.seed)


FAMILY = MlpFamily()
