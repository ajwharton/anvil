"""Named loss plugins."""

from anvil.losses.registry import LossSpec, get_loss, list_losses, register_loss

__all__ = ["LossSpec", "get_loss", "list_losses", "register_loss"]
