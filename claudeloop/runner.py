"""Main loop orchestration — worker -> commit -> evaluate -> repeat."""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from claudeloop import git_utils
from claudeloop.backends import create_backend
from claudeloop.backends.base import StreamEvent
from claudeloop.evaluator import EvalResult, Evaluator
from claudeloop.key_listener import KeyListener
from claudeloop.utils import setup_logging
from claudeloop.worker import Worker

# All Claude Code tools — used as default for --allowedTools.
# Interactive tools (EnterPlanMode, ExitPlanMode, AskUserQuestion) are
# excluded so the loop runs fully unattended without blocking on user input.
DEFAULT_ALLOWED_TOOLS: list[str] = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch", "NotebookEdit", "Agent",
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
]

# Style map for streaming event types
_EVENT_STYLES: dict[str, tuple[str, str]] = {
    # type -> (rich style, prefix)
    "tool_use":    ("cyan",   "  tool"),
    "tool_result": ("dim",    "     "),
    "text":        ("dim",    "     "),
    "system":      ("dim",    "  sys "),
    "error":       ("red",    "  err "),
    "result":      ("green",  "     "),
}


@dataclass
class RunConfig:
    """Configuration for a claudeloop run."""

    prompt_path: Path
    success_path: Path
    plan_path: Path
    max_iters: int
    model: str
    backend_name: str
    log_dir: Path
    max_cost: float
    auto_commit: bool
    verbose: bool
    additional_prompt: str | None = None
    effort_level: str | None = None
    fast_mode: bool = False
    resume: bool = False
    dangerously_skip_permissions: bool = False
    allowed_tools: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_TOOLS))


# ---------------------------------------------------------------------------
# Session state — persisted to {log_dir}/session.json
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Record for a single completed iteration."""

    iteration: int
    worker_cost: float
    worker_duration: float
    worker_error: bool
    commit_hash: str | None
    eval_cost: float
    eval_duration: float
    eval_success: bool
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "worker_cost": self.worker_cost,
            "worker_duration": self.worker_duration,
            "worker_error": self.worker_error,
            "commit_hash": self.commit_hash,
            "eval_cost": self.eval_cost,
            "eval_duration": self.eval_duration,
            "eval_success": self.eval_success,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionState:
    """Persistent session state."""

    session_id: str
    status: str  # "running", "success", "failed", "paused", "error"
    started_at: str
    updated_at: str
    completed_iteration: int
    total_cost: float
    last_commit_hash: str | None
    iterations: list[IterationRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_iteration": self.completed_iteration,
            "total_cost": self.total_cost,
            "last_commit_hash": self.last_commit_hash,
            "iterations": [it.to_dict() for it in self.iterations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        iterations = [
            IterationRecord(**rec) for rec in data.get("iterations", [])
        ]
        return cls(
            session_id=data["session_id"],
            status=data["status"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            completed_iteration=data["completed_iteration"],
            total_cost=data["total_cost"],
            last_commit_hash=data.get("last_commit_hash"),
            iterations=iterations,
        )

    @classmethod
    def new(cls) -> SessionState:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            session_id=uuid.uuid4().hex[:12],
            status="running",
            started_at=now,
            updated_at=now,
            completed_iteration=0,
            total_cost=0.0,
            last_commit_hash=None,
        )


class Runner:
    """Orchestrates the worker-evaluator loop."""

    def __init__(self, config: RunConfig, key_listener: KeyListener | None = None):
        self.config = config
        self.key_listener = key_listener
        self.console = Console(width=240, soft_wrap=True)
        self.logger = setup_logging(config.log_dir, config.verbose)
        self._session_path = config.log_dir / "session.json"

        # Worker always uses CLI backend (needs Claude Code tools)
        cli_kwargs: dict = {}
        if config.effort_level:
            cli_kwargs["effort_level"] = config.effort_level
        if config.fast_mode:
            cli_kwargs["fast_mode"] = True
        worker_backend = create_backend("cli", **cli_kwargs)
        # Evaluator uses user-chosen backend
        eval_backend = create_backend(config.backend_name, **cli_kwargs) if config.backend_name == "cli" else create_backend(config.backend_name)

        self.worker = Worker(
            prompt_path=config.prompt_path,
            plan_path=config.plan_path,
            model=config.model,
            backend=worker_backend,
            additional_prompt=config.additional_prompt,
            dangerously_skip_permissions=config.dangerously_skip_permissions,
            allowed_tools=config.allowed_tools,
        )
        self.evaluator = Evaluator(
            success_path=config.success_path,
            plan_path=config.plan_path,
            model=config.model,
            backend=eval_backend,
            dangerously_skip_permissions=config.dangerously_skip_permissions,
            allowed_tools=config.allowed_tools,
        )

        # Session state
        if config.resume:
            self.session = self._load_session()
            self.total_cost = self.session.total_cost
            self.console.print(
                f"[yellow]Resuming session {self.session.session_id} "
                f"from iteration {self.session.completed_iteration + 1} "
                f"(prior cost: ${self.session.total_cost:.4f})[/yellow]"
            )
        else:
            self.session = SessionState.new()
            self.total_cost: float = 0.0

    # --- Session persistence ---

    def _save_session(self) -> None:
        self.session.updated_at = datetime.now(timezone.utc).isoformat()
        self.session.total_cost = self.total_cost
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._session_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.session.to_dict(), indent=2))
        tmp.replace(self._session_path)

    def _load_session(self) -> SessionState:
        if not self._session_path.exists():
            self.console.print(
                f"[red]Error:[/red] No session file found at {self._session_path}"
            )
            raise SystemExit(1)
        data = json.loads(self._session_path.read_text())
        session = SessionState.from_dict(data)
        if session.status == "success":
            self.console.print(
                "[yellow]Warning:[/yellow] Previous session already succeeded. "
                "Starting fresh session instead."
            )
            return SessionState.new()
        return session

    # --- Main loop ---

    def run_loop(self) -> bool:
        """Run the agentic loop. Returns True if success condition is met."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._print_banner()

        start_iter = self.session.completed_iteration + 1

        try:
            return self._run_iterations(start_iter)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user.[/yellow]")
            self.session.status = "error"
            self._save_session()
            self.console.print(
                f"[dim]Session saved. Resume with: claudeloop run --resume[/dim]"
            )
            return False
        except Exception:
            self.console.print(f"[red]Unexpected error:[/red]")
            self.console.print(traceback.format_exc())
            self.session.status = "error"
            self._save_session()
            self.console.print(
                f"[dim]Session saved. Resume with: claudeloop run --resume[/dim]"
            )
            raise

    def _run_iterations(self, start_iter: int) -> bool:
        """Core iteration loop, separated for clean error handling."""
        self.session.status = "running"
        self._save_session()

        iteration = start_iter
        while iteration <= self.config.max_iters:
            self.console.rule(f"[bold blue]Iteration {iteration}/{self.config.max_iters}")

            iter_start = time.time()

            # --- Pause check: before worker ---
            if not self._check_pause(iteration):
                return False

            # --- Worker Phase ---
            self.console.print("[yellow]WORKER | Agent running...[/yellow]")
            self.logger.info("Iteration %d: worker starting", iteration)

            backend = self.worker.backend
            if hasattr(backend, "reset_cancel"):
                backend.reset_cancel()

            worker_result = self.worker.execute(
                iteration=iteration,
                on_event=self._on_worker_event,
                max_iters=self.config.max_iters,
                total_cost=self.total_cost,
                max_cost=self.config.max_cost,
            )
            self.total_cost += worker_result.cost_usd

            # Handle mid-worker cancellation
            if worker_result.cancelled:
                self.console.print(
                    f"[yellow]WORKER | Cancelled by user "
                    f"(${worker_result.cost_usd:.4f})[/yellow]"
                )
                if not self._handle_pause(iteration):
                    return False
                # Restart the same iteration after pause
                continue

            if worker_result.is_error:
                self.console.print(f"[red]WORKER | Error:[/red] {worker_result.output[:500]}")
                self.logger.error("Worker error: %s", worker_result.output[:1000])
            else:
                self.console.print(
                    f"[green]WORKER | Finished[/green] "
                    f"(${worker_result.cost_usd:.4f}, {worker_result.duration_secs:.0f}s)"
                )

            # --- Git Commit Phase ---
            commit_hash = None
            if self.config.auto_commit:
                commit_hash = git_utils.commit_all(
                    message=f"claudeloop: iteration {iteration}"
                )
                if commit_hash:
                    self.console.print(f"[green]Committed:[/green] {commit_hash}")
                else:
                    self.console.print("[dim]No changes to commit[/dim]")

            # --- Log Phase ---
            self._write_log(iteration, commit_hash, worker_result)

            # --- Pause check: before evaluator ---
            if not self._check_pause(iteration):
                return False

            # --- Evaluator Phase ---
            self.console.print("[yellow]EVALUATOR | Agent running...[/yellow]")
            self.logger.info("Iteration %d: evaluator starting", iteration)

            eval_backend = self.evaluator.backend
            if hasattr(eval_backend, "reset_cancel"):
                eval_backend.reset_cancel()

            eval_result = self.evaluator.evaluate(
                iteration=iteration,
                on_event=self._on_evaluator_event,
            )
            self.total_cost += eval_result.cost_usd

            # Handle mid-evaluator cancellation
            if eval_result.cancelled:
                self.console.print(
                    f"[yellow]EVALUATOR | Cancelled by user "
                    f"(${eval_result.cost_usd:.4f})[/yellow]"
                )
                if not self._handle_pause(iteration):
                    return False
                # Restart the same iteration after pause
                continue

            # --- Display Results ---
            iter_duration = time.time() - iter_start
            self._print_eval_results(iteration, eval_result, iter_duration)
            self.logger.info(
                "Iteration %d: success=%s reason=%s",
                iteration,
                eval_result.judgment.success,
                eval_result.judgment.reason,
            )

            # --- Save session state ---
            self.session.completed_iteration = iteration
            self.session.last_commit_hash = commit_hash
            self.session.iterations.append(
                IterationRecord(
                    iteration=iteration,
                    worker_cost=worker_result.cost_usd,
                    worker_duration=worker_result.duration_secs,
                    worker_error=worker_result.is_error,
                    commit_hash=commit_hash,
                    eval_cost=eval_result.cost_usd,
                    eval_duration=eval_result.duration_secs,
                    eval_success=eval_result.judgment.success,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

            # --- Check Success ---
            if eval_result.judgment.success:
                self.session.status = "success"
                self._save_session()
                self._print_success(iteration)
                return True

            self._save_session()
            self.console.print(f"[dim]Total cost so far: ${self.total_cost:.4f}[/dim]\n")

            # --- Pause check: after evaluator, before next iteration ---
            if not self._check_pause(iteration):
                return False

            # --- Cost Cap Check ---
            if self.total_cost >= self.config.max_cost:
                self.session.status = "failed"
                self._save_session()
                self._print_cost_exceeded()
                return False

            iteration += 1

        self.session.status = "failed"
        self._save_session()
        self._print_max_iters()
        return False

    def _on_worker_event(self, event: StreamEvent) -> None:
        """Handle a streaming event from the worker agent."""
        style, prefix = _EVENT_STYLES.get(event.type, ("dim", "     "))
        ts = datetime.now().strftime("%H:%M:%S")
        cost_tag = ""
        if event.type == "result":
            invocation_cost = event.data.get("total_cost_usd", 0.0)
            session_total = self.total_cost + invocation_cost
            cost_tag = f" [dim](session: ${session_total:.2f})[/dim]"
        self.console.print(f"[{style}]{ts} WORKER | {prefix} | {event.message}{cost_tag}[/{style}]")
        self._log_event("worker", event)

        # Check if ESC was pressed — cancel the worker subprocess
        if (
            self.key_listener is not None
            and self.key_listener.pause_requested.is_set()
        ):
            backend = self.worker.backend
            if hasattr(backend, "cancel"):
                backend.cancel()

    def _on_evaluator_event(self, event: StreamEvent) -> None:
        """Handle a streaming event from the evaluator agent."""
        style, prefix = _EVENT_STYLES.get(event.type, ("dim", "     "))
        ts = datetime.now().strftime("%H:%M:%S")
        cost_tag = ""
        if event.type == "result":
            invocation_cost = event.data.get("total_cost_usd", 0.0)
            session_total = self.total_cost + invocation_cost
            cost_tag = f" [dim](session: ${session_total:.2f})[/dim]"
        self.console.print(f"[{style}]{ts} EVALUATOR | {prefix} | {event.message}{cost_tag}[/{style}]")
        self._log_event("evaluator", event)

        # Check if ESC was pressed — cancel the evaluator subprocess
        if (
            self.key_listener is not None
            and self.key_listener.pause_requested.is_set()
        ):
            eval_backend = self.evaluator.backend
            if hasattr(eval_backend, "cancel"):
                eval_backend.cancel()

    def _log_event(self, source: str, event: StreamEvent) -> None:
        """Log event details to the log file.

        At INFO level: type + message (one-liner).
        At DEBUG level: also dump raw event data (tool inputs, outputs, etc.).
        """
        self.logger.info("%s %s: %s", source, event.type, event.message)
        if event.data and self.logger.isEnabledFor(logging.DEBUG):
            try:
                raw = json.dumps(event.data, indent=2, default=str)
                # Truncate very large payloads (e.g. tool results)
                if len(raw) > 5000:
                    raw = raw[:5000] + "\n... (truncated)"
                self.logger.debug("%s %s data:\n%s", source, event.type, raw)
            except (TypeError, ValueError):
                self.logger.debug("%s %s data: %r", source, event.type, event.data)

    # --- Pause handling ---

    def _check_pause(self, iteration: int) -> bool:
        """Check if ESC was pressed and handle the pause.

        Returns ``True`` to continue the loop, ``False`` to quit.
        """
        if self.key_listener is None or not self.key_listener.pause_requested.is_set():
            return True
        return self._handle_pause(iteration)

    def _handle_pause(self, iteration: int) -> bool:
        """Show interactive pause prompt.

        Returns ``True`` to continue, ``False`` to quit.
        """
        if self.key_listener:
            self.key_listener.pause_listening()

        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold yellow]PAUSED[/bold yellow] (ESC pressed)\n\n"
                "Options:\n"
                "  - Type feedback for the next iteration, then press Enter on an empty line\n"
                "  - Press Enter immediately to continue without feedback\n"
                "  - Type [bold]q[/bold] and Enter to quit (session saved for --resume)",
                title="Paused",
                border_style="yellow",
            )
        )

        feedback_lines: list[str] = []
        should_continue = True

        try:
            while True:
                try:
                    line = input("> ")
                except EOFError:
                    break

                if line.strip().lower() == "q":
                    should_continue = False
                    break

                if line == "" and not feedback_lines:
                    break

                if line == "" and feedback_lines:
                    break

                feedback_lines.append(line)
        except KeyboardInterrupt:
            should_continue = False

        if feedback_lines:
            feedback_text = "\n".join(feedback_lines)
            self._append_user_feedback(feedback_text, iteration)
            self.console.print("[green]Feedback appended to PLAN.md[/green]")

        if should_continue:
            self.console.print("[green]Resuming...[/green]\n")
        else:
            self.console.print("[yellow]Quitting... session saved.[/yellow]")
            self.session.status = "paused"
            self._save_session()

        if self.key_listener:
            self.key_listener.reset()
            if should_continue:
                self.key_listener.resume_listening()

        return should_continue

    def _append_user_feedback(self, feedback: str, iteration: int) -> None:
        """Append user feedback to PLAN.md."""
        plan_path = self.config.plan_path
        current = plan_path.read_text() if plan_path.exists() else ""
        addition = f"\n\n## User Feedback (Iteration {iteration})\n{feedback}\n"
        plan_path.write_text(current + addition)

    # --- Display helpers ---

    def _print_banner(self) -> None:
        resume_line = ""
        if self.config.resume:
            resume_line = f"\nResuming session: {self.session.session_id}"
        esc_hint = ""
        if self.key_listener is not None:
            esc_hint = "\n[dim]Press ESC at any time to pause[/dim]"
        self.console.print(
            Panel.fit(
                f"[bold]claudeloop[/bold] - Self-improving Agentic Loop\n"
                f"Model: {self.config.model}\n"
                f"Max iterations: {self.config.max_iters}\n"
                f"Max cost: ${self.config.max_cost:.2f}\n"
                f"Backend (evaluator): {self.config.backend_name}\n"
                f"Effort level: {self.config.effort_level or 'default'}\n"
                f"Fast mode: {'on' if self.config.fast_mode else 'off'}"
                f"{resume_line}{esc_hint}",
                title="Configuration",
            )
        )

    def _print_eval_results(
        self, iteration: int, eval_result: EvalResult, duration: float
    ) -> None:
        j = eval_result.judgment
        color = "green" if j.success else "red"
        body = (
            f"[{color}]{'SUCCESS' if j.success else 'NOT YET'}[/{color}]\n"
            f"Reason: {j.reason}\n"
            f"Cost: ${eval_result.cost_usd:.4f} | Duration: {duration:.0f}s"
        )
        if j.suggestions:
            body += f"\nSuggestions: {j.suggestions}"
        self.console.print(
            Panel(body, title=f"Evaluator Judgment (Iteration {iteration})", border_style=color)
        )

    def _write_log(self, iteration: int, commit_hash: str | None, worker_result) -> None:
        hash_part = commit_hash or "nocommit"
        log_path = self.config.log_dir / f"agent_iter_{iteration}_{hash_part}.log"
        content = (
            f"=== Iteration {iteration} ===\n"
            f"Commit: {commit_hash}\n"
            f"Cost: ${worker_result.cost_usd:.4f}\n"
            f"Duration: {worker_result.duration_secs:.1f}s\n"
            f"Error: {worker_result.is_error}\n"
            f"\n--- Worker Output ---\n"
            f"{worker_result.output}\n"
        )
        log_path.write_text(content)
        self.logger.debug("Log written to %s", log_path)

    def _print_success(self, iteration: int) -> None:
        self.console.print(
            Panel.fit(
                f"[bold green]SUCCESS after {iteration} iteration(s)![/bold green]\n"
                f"Total cost: ${self.total_cost:.4f}",
                title="Complete",
                border_style="green",
            )
        )

    def _print_cost_exceeded(self) -> None:
        self.console.print(
            Panel.fit(
                f"[bold red]Cost cap exceeded "
                f"(${self.total_cost:.4f} >= ${self.config.max_cost:.2f}).[/bold red]",
                title="Budget Exceeded",
                border_style="red",
            )
        )

    def _print_max_iters(self) -> None:
        self.console.print(
            Panel.fit(
                f"[bold yellow]Max iterations ({self.config.max_iters}) reached "
                f"without success.[/bold yellow]\n"
                f"Total cost: ${self.total_cost:.4f}",
                title="Incomplete",
                border_style="yellow",
            )
        )
