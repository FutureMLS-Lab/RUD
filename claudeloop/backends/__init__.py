"""Backend factory for claudeloop."""

from __future__ import annotations

from .base import Backend, BackendResponse, StreamEvent
from .cli_backend import CLIBackend
from .sdk_backend import SDKBackend

__all__ = ["Backend", "BackendResponse", "StreamEvent", "CLIBackend", "SDKBackend", "create_backend"]


def create_backend(name: str, **kwargs) -> Backend:
    """Create a backend by name ('cli' or 'sdk')."""
    if name == "cli":
        return CLIBackend(**kwargs)
    if name == "sdk":
        return SDKBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r}. Choose 'cli' or 'sdk'.")
