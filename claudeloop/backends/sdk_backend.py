"""Anthropic SDK backend for direct API calls."""

from __future__ import annotations

import time

from collections.abc import Callable

from .base import Backend, BackendResponse, StreamEvent

# Token pricing per million tokens (approximate, USD)
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-opus-4-5-20250514": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


class SDKBackend(Backend):
    """Backend using the ``anthropic`` Python SDK directly.

    Note: This backend has no tool-use capabilities. It sends a prompt
    and returns the text response. Suitable for the evaluator agent but
    limited for the worker (which needs Claude Code's built-in tools).
    """

    def __init__(self, api_key: str | None = None, max_tokens: int = 16384):
        try:
            import anthropic, os
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the SDK backend. "
                "Install with: pip install anthropic"
            )
        token = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if token.startswith("sk-ant-oat"):
            self.client = anthropic.Anthropic(auth_token=token, api_key=None)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def invoke(
        self,
        prompt: str,
        model: str,
        dangerously_skip_permissions: bool = False,  # ignored for SDK
        allowed_tools: list[str] | None = None,  # ignored for SDK
        system_prompt: str | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,  # ignored for SDK
    ) -> BackendResponse:
        kwargs: dict = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        start = time.time()
        response = self.client.messages.create(**kwargs)
        duration = time.time() - start

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        cost = self._estimate_cost(response.usage, model)

        return BackendResponse(
            text=text,
            is_error=(response.stop_reason == "error"),
            cost_usd=cost,
            duration_secs=duration,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    @staticmethod
    def _estimate_cost(usage, model: str) -> float:
        rates = _PRICING.get(model, _DEFAULT_PRICING)
        input_cost = (usage.input_tokens / 1_000_000) * rates["input"]
        output_cost = (usage.output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost
