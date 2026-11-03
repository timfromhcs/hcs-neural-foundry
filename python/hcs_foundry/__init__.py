"""HCS Neural Foundry — local-first neural network design, weight-space research and agent studio.

Submodules are imported lazily on purpose: ``import hcs_foundry`` must stay cheap and must not
require torch. Callers that need the engine import ``hcs_foundry.engine``.

Made with ❤️ by hcsmedia and timfromhcs
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = [
    "__version__",
    "SCHOLARLY_NOTE",
]

SCHOLARLY_NOTE = (
    "Generating, editing or evolving weights is not the same as training a capable model. "
    "Every result in this project is tied to a run with a dataset hash, seed, evaluator version "
    "and hardware record; none of it is evidence of state-of-the-art performance."
)
