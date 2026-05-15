"""Abstract backend interface for Claude invocations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BackendResponse:
    """Response from a Claude invocation."""

    text: str
    is_error: bool
    cost_usd: float
    duration_secs: float
    cancelled: bool = False
    raw: dict | None = field(default=None, repr=False)


@dataclass
class StreamEvent:
    """A single streaming event from Claude."""

    type: str  # "text", "tool_use", "tool_result", "result", "error"
    message: str  # Human-readable summary
    data: dict = field(default_factory=dict)


class Backend(ABC):
    """Abstract base class for Claude backends."""

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        model: str,
        dangerously_skip_permissions: bool = False,
        allowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> BackendResponse:
        """Send a prompt to Claude and return the response.

        If *on_event* is provided, streaming events are delivered to the
        callback in real-time as the agent works.
        """
        ...
