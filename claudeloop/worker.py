"""Worker agent — builds prompt and invokes Claude Code to do work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from claudeloop.backends.base import Backend, BackendResponse, StreamEvent
from claudeloop.prompts import WORKER_PROMPT


@dataclass
class WorkerResult:
    """Result of a single worker iteration."""

    output: str
    cost_usd: float
    duration_secs: float
    is_error: bool
    cancelled: bool = False


class Worker:
    """Worker agent that invokes Claude Code to perform development tasks."""

    def __init__(
        self,
        prompt_path: Path,
        plan_path: Path,
        model: str,
        backend: Backend,
        additional_prompt: str | None = None,
        dangerously_skip_permissions: bool = False,
        allowed_tools: list[str] | None = None,
    ):
        self.prompt_path = prompt_path
        self.plan_path = plan_path
        self.model = model
        self.backend = backend
        self.additional_prompt = additional_prompt
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.allowed_tools = allowed_tools

    def build_prompt(
        self,
        iteration: int,
        max_iters: int = 0,
        total_cost: float = 0.0,
        max_cost: float = 0.0,
    ) -> str:
        """Construct the full prompt for the worker agent."""
        task_prompt = self.prompt_path.read_text()
        if self.additional_prompt:
            task_prompt += f"\n\n## Additional Instructions\n{self.additional_prompt}\n"
        plan = self.plan_path.read_text() if self.plan_path.exists() else "(No plan yet — create one in PLAN.md)"
        return WORKER_PROMPT.format(
            iteration=iteration,
            max_iters=max_iters,
            total_cost=total_cost,
            max_cost=max_cost,
            task_prompt=task_prompt,
            plan=plan,
        )

    def execute(
        self,
        iteration: int,
        on_event: Callable[[StreamEvent], None] | None = None,
        max_iters: int = 0,
        total_cost: float = 0.0,
        max_cost: float = 0.0,
    ) -> WorkerResult:
        """Run one worker iteration."""
        prompt = self.build_prompt(
            iteration,
            max_iters=max_iters,
            total_cost=total_cost,
            max_cost=max_cost,
        )
        resp: BackendResponse = self.backend.invoke(
            prompt=prompt,
            model=self.model,
            dangerously_skip_permissions=self.dangerously_skip_permissions,
            allowed_tools=self.allowed_tools,
            on_event=on_event,
        )
        return WorkerResult(
            output=resp.text,
            cost_usd=resp.cost_usd,
            duration_secs=resp.duration_secs,
            is_error=resp.is_error,
            cancelled=resp.cancelled,
        )
