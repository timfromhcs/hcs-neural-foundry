"""Shared pytest fixtures.

Every test writes into its own temporary workspace so nothing leaks into the developer's real
``~/.hcs-foundry`` and tests can run in parallel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "workspace"
    monkeypatch.setenv("HCS_FOUNDRY_HOME", str(path))
    return path


@pytest.fixture()
def foundry(home):
    from hcs_foundry.services import Foundry

    instance = Foundry(home, probe_backends=False)
    yield instance
    instance.close()


@pytest.fixture(scope="session")
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


@pytest.fixture()
def transformer_model(foundry):
    from hcs_foundry.schemas import TransformerConfig

    config = TransformerConfig(
        vocab_size=32, d_model=32, n_layers=1, n_heads=2, context_length=32, seed=3
    )
    return foundry.create_model("Test LM", config.model_dump(mode="json"))


@pytest.fixture()
def mlp_model(foundry):
    from hcs_foundry.schemas import MlpConfig

    config = MlpConfig(input_dim=8, hidden_dims=(12, 12), output_dim=3, seed=5)
    return foundry.create_model("Test MLP", config.model_dump(mode="json"))


def pytest_configure(config):
    if os.environ.get("HCS_FOUNDRY_SKIP_HEAVY"):  # pragma: no cover - manual switch
        config.addinivalue_line("markers", "heavy: skipped by HCS_FOUNDRY_SKIP_HEAVY")
