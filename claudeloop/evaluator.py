"""Evaluator agent — Claude runs tests and judges success directly."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from typing import Callable, Optional

from claudeloop.backends.base import Backend, StreamEvent
from claudeloop.prompts import EVALUATOR_PROMPT, EVALUATOR_SYSTEM_PROMPT
from claudeloop.utils import parse_json_response


@dataclass
class EvaluatorJudgment:
    """Claude's judgment on whether success conditions are met."""

    success: bool
    reason: str
    suggestions: str
    raw_response: str = field(default="", repr=False)


@dataclass
class EvalResult:
    """Full evaluation result."""

    judgment: EvaluatorJudgment
    cost_usd: float
    duration_secs: float
    cancelled: bool = False


class Evaluator:
    """Evaluator agent — Claude runs tests and judges success directly."""

    def __init__(
        self,
        success_path: Path,
        plan_path: Path,
        model: str,
        backend: Backend,
        dangerously_skip_permissions: bool = False,
        allowed_tools: list[str] | None = None,
    ):
        self.success_path = success_path
        self.plan_path = plan_path
        self.model = model
        self.backend = backend
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.allowed_tools = allowed_tools

    def _build_prompt(self) -> str:
        plan = self.plan_path.read_text() if self.plan_path.exists() else "(No plan)"
        return EVALUATOR_PROMPT.format(
            success_condition=self.success_path.read_text(),
            plan=plan,
        )

    def _parse_judgment(self, text: str) -> EvaluatorJudgment:
        data = parse_json_response(text)
        if data and isinstance(data.get("success"), bool):
            return EvaluatorJudgment(
                success=data["success"],
                reason=data.get("reason", ""),
                suggestions=data.get("suggestions", ""),
                raw_response=text,
            )
        # Fallback: heuristic
        lower = text.lower()
        success = '"success": true' in lower or '"success":true' in lower
        return EvaluatorJudgment(
            success=success,
            reason=text[:500],
            suggestions="" if success else "Could not parse evaluator response.",
            raw_response=text,
        )

    def _append_feedback_to_plan(self, suggestions: str, iteration: int) -> None:
        """Append evaluator feedback to PLAN.md for the next worker."""
        current = self.plan_path.read_text() if self.plan_path.exists() else ""
        addition = (
            f"\n\n## Evaluator Feedback (Iteration {iteration})\n{suggestions}\n"
        )
        self.plan_path.write_text(current + addition)

    def evaluate(
        self,
        iteration: int,
        on_event: Optional[Callable[[StreamEvent], None]] = None,
    ) -> EvalResult:
        """Run evaluation: Claude runs tests and judges success directly."""
        eval_start = time.time()

        prompt = self._build_prompt()
        resp = self.backend.invoke(
            prompt=prompt,
            model=self.model,
            dangerously_skip_permissions=self.dangerously_skip_permissions,
            allowed_tools=self.allowed_tools,
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            on_event=on_event,
        )

        if resp.cancelled:
            eval_duration = time.time() - eval_start
            return EvalResult(
                judgment=EvaluatorJudgment(
                    success=False,
                    reason="Cancelled by user",
                    suggestions="",
                ),
                cost_usd=resp.cost_usd,
                duration_secs=eval_duration,
                cancelled=True,
            )

        # Parse judgment from Claude's response
        judgment = self._parse_judgment(resp.text)

        # Feed suggestions back into PLAN.md
        if not judgment.success and judgment.suggestions:
            self._append_feedback_to_plan(judgment.suggestions, iteration)

        eval_duration = time.time() - eval_start
        return EvalResult(
            judgment=judgment,
            cost_usd=resp.cost_usd,
            duration_secs=eval_duration,
        )
