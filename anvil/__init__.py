"""Anvil — open-source post-training toolkit (SFT/RL, LoRA-first).

Phase 0: typed client contract + fake backend for golden tests.
"""

from anvil.client import SamplingClient, ServiceClient, TrainingClient
from anvil.protocol import (
    AdamParams,
    Datum,
    Example,
    ExportFormat,
    LossFn,
    Message,
    ModelInput,
    SamplingParams,
    TextPart,
)

__version__ = "0.0.2"

__all__ = [
    "AdamParams",
    "Datum",
    "Example",
    "ExportFormat",
    "LossFn",
    "Message",
    "ModelInput",
    "SamplingClient",
    "SamplingParams",
    "ServiceClient",
    "TextPart",
    "TrainingClient",
    "__version__",
]
