"""Chat / multimodal renderers (train/sample must share one)."""

from anvil.render.base import Renderer
from anvil.render.text import ToyTextRenderer

__all__ = ["Renderer", "ToyTextRenderer"]
