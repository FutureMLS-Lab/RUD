"""Tmux-based controller mode for claudeloop.

Launches an interactive Claude Code session in a tmux pane, monitors it via
polling, and uses a separate controller agent to evaluate progress and send
feedback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
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
from claudeloop.key_listener import KeyListener
from claudeloop.prompts import (
    TMUX_CONTROLLER_PROMPT,
    TMUX_CONTROLLER_SYSTEM_PROMPT,
    TMUX_WORKER_FOLLOWUP,
    TMUX_WORKER_INITIAL_PROMPT,
)
from claudeloop.runner import DEFAULT_ALLOWED_TOOLS, IterationRecord, SessionState
from claudeloop.tmux_util import tmux_subprocess_env
from claudeloop.utils import parse_json_response, setup_logging


@dataclass
class TmuxConfig:
    """Configuration for a tmux controller run."""

    prompt_path: Path
    success_path: Path
    plan_path: Path
    max_rounds: int
    model: str
    log_dir: Path
    max_cost: float
    auto_commit: bool
    verbose: bool
    poll_interval: float = 10.0
    idle_threshold: int = 3
    session_name: str | None = None
    additional_prompt: str | None = None
    effort_level: str | None = None
    fast_mode: bool = False
    resume: bool = False
    dangerously_skip_permissions: bool = False
    allowed_tools: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_TOOLS))


class TmuxSession:
    """Low-level tmux wrapper for creating and interacting with a tmux session.

    All send/capture methods accept an optional *target* override so
    different windows/panes within the same session can be addressed.
    When *target* is ``None`` the default ``self.target`` (window 0,
    pane 0) is used.
    """

    def __init__(self, session_name: str):
        self.session_name = session_name
        self.target = f"{session_name}:0.0"
        tmux_path = shutil.which("tmux")
        if tmux_path is None:
            raise FileNotFoundError(
                "tmux not found on PATH. Install tmux to use tmux controller mode."
            )
        self.tmux_path = tmux_path

    def create(self, cwd: Path | None = None) -> None:
        """Create a new detached tmux session."""
        cmd = [
            self.tmux_path, "new-session",
            "-d", "-s", self.session_name,
            "-x", "220", "-y", "50",
        ]
        subprocess.run(
            cmd, check=True,
            cwd=cwd or Path.cwd(),
            capture_output=True, text=True,
            env=tmux_subprocess_env(),
        )
        self.resize_window()

    def resize_window(self, columns: int = 240, rows: int = 64) -> None:
        """Best-effort resize for readable terminal output in web captures."""
        try:
            subprocess.run(
                [
                    self.tmux_path,
                    "resize-window",
                    "-t",
                    f"{self.session_name}:0",
                    "-x",
                    str(columns),
                    "-y",
                    str(rows),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=tmux_subprocess_env(),
            )
        except OSError:
            pass

    def new_window(self, name: str | None = None, cwd: Path | None = None) -> str:
        """Create a new window in the session. Returns the target (e.g. 'session:1.0')."""
        cmd = [self.tmux_path, "new-window", "-t", self.session_name, "-P", "-F", "#{window_index}"]
        if name:
            cmd.extend(["-n", name])
        result = subprocess.run(
            cmd, check=True,
            cwd=cwd or Path.cwd(),
            capture_output=True, text=True,
            env=tmux_subprocess_env(),
        )
        window_index = result.stdout.strip()
        return f"{self.session_name}:{window_index}.0"

    def split_window(self, horizontal: bool = True, cwd: Path | None = None, target: str | None = None) -> str:
        """Split the current window into two panes. Returns the target of the new pane.

        *horizontal* = True creates a left/right split (-h), False creates top/bottom (-v).
        """
        t = target or self.target
        cmd = [
            self.tmux_path, "split-window",
            "-t", t,
            "-P", "-F", "#{pane_index}",
        ]
        if horizontal:
            cmd.append("-h")
        result = subprocess.run(
            cmd, check=True,
            cwd=cwd or Path.cwd(),
            capture_output=True, text=True,
            env=tmux_subprocess_env(),
        )
        pane_index = result.stdout.strip()
        # target format: session:window.pane
        window_part = t.rsplit(".", 1)[0]  # e.g. "session:0"
        return f"{window_part}.{pane_index}"

    def send_keys(self, text: str, enter: bool = True, target: str | None = None) -> None:
        """Send keystrokes to a tmux pane."""
        t = target or self.target
        cmd = [self.tmux_path, "send-keys", "-t", t, text]
        if enter:
            cmd.append("Enter")
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=tmux_subprocess_env())

    def send_text_via_buffer(self, text: str, submit: bool = True, target: str | None = None) -> None:
        """Send long text via tmux load-buffer / paste-buffer to avoid escaping issues.

        If *submit* is True (default), sends two Enter keys after pasting so that
        interactive programs like Claude Code treat the input as submitted
        (Claude Code requires an empty-line Enter to submit multi-line input).
        """
        t = target or self.target
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmpfile = f.name
        try:
            subprocess.run(
                [self.tmux_path, "load-buffer", tmpfile],
                check=True, capture_output=True, text=True,
                env=tmux_subprocess_env(),
            )
            subprocess.run(
                [self.tmux_path, "paste-buffer", "-t", t],
                check=True, capture_output=True, text=True,
                env=tmux_subprocess_env(),
            )
            if submit:
                # Small delay to let the paste complete before sending Enter
                time.sleep(0.3)
                # Send Enter twice — Claude Code needs an empty line to submit
                # multi-line input
                subprocess.run(
                    [self.tmux_path, "send-keys", "-t", t, "Enter"],
                    check=True, capture_output=True, text=True,
                    env=tmux_subprocess_env(),
                )
                time.sleep(0.1)
                subprocess.run(
                    [self.tmux_path, "send-keys", "-t", t, "Enter"],
                    check=True, capture_output=True, text=True,
                    env=tmux_subprocess_env(),
                )
        finally:
            os.unlink(tmpfile)

    def capture_pane(self, lines: int = 500, target: str | None = None) -> str:
        """Capture the current content of a tmux pane.

        Trailing blank lines are stripped so that idle-detection hashing
        is not thrown off by the cursor position on an empty line.
        """
        t = target or self.target
        result = subprocess.run(
            [
                self.tmux_path, "capture-pane",
                "-t", t,
                "-p",  # print to stdout
                "-S", f"-{lines}",
            ],
            capture_output=True, text=True,
            env=tmux_subprocess_env(),
        )
        return result.stdout.rstrip("\n")

    def kill(self) -> None:
        """Kill the tmux session."""
        subprocess.run(
            [self.tmux_path, "kill-session", "-t", self.session_name],
            capture_output=True, text=True,
            env=tmux_subprocess_env(),
        )

    def exists(self) -> bool:
        """Check if the tmux session exists."""
        result = subprocess.run(
            [self.tmux_path, "has-session", "-t", self.session_name],
            capture_output=True, text=True,
            env=tmux_subprocess_env(),
        )
        return result.returncode == 0

    def is_alive(self) -> bool:
        """Check if the tmux session is alive and has running processes."""
        return self.exists()


class TmuxController:
    """Main orchestrator for the tmux-based controller mode."""

    def __init__(self, config: TmuxConfig, key_listener: KeyListener | None = None):
        self.config = config
        self.key_listener = key_listener
        self.console = Console(width=240, soft_wrap=True)
        self.logger = setup_logging(config.log_dir, config.verbose)

        # Generate session name if not provided
        self.session_id = uuid.uuid4().hex[:8]
        session_name = config.session_name or f"claudeloop-{self.session_id}"
        self.tmux = TmuxSession(session_name)
        self.evaluator_target: str | None = None  # set during _setup_tmux

        # Rolling memory of recent summaries per pane (to avoid repetition)
        self._summary_history: dict[str, list[str]] = {}
        self._summary_history_max = 5

        # Rate-limit and dedup for pane analysis
        self._last_analysis_time: dict[str, float] = {}
        self._last_displayed_summary: dict[str, str] = {}
        self._analysis_min_interval = 30.0  # min seconds between API calls per pane

        # Anthropic client for lightweight summarization (haiku)
        try:
            import anthropic
            import os
            token = os.environ.get("ANTHROPIC_API_KEY", "")
            if token.startswith("sk-ant-oat"):
                self._anthropic = anthropic.Anthropic(auth_token=token, api_key=None)
            else:
                self._anthropic = anthropic.Anthropic()
        except Exception:
            self._anthropic = None
            self.logger.info("Anthropic SDK not available, using regex-based summarization")

        # Session state
        self._session_path = config.log_dir / "session.json"
        self._resumed_tmux = False  # True if we reconnected to an existing tmux session
        if config.resume:
            self.session = self._load_session()
            self.session_id = self.session.session_id

            session_data = json.loads(self._session_path.read_text())

            # Try to reconnect to previous tmux session
            prev_tmux_name = session_data.get("tmux_session_name")
            prev_eval_target = session_data.get("tmux_evaluator_target")
            if prev_tmux_name:
                prev_tmux = TmuxSession(prev_tmux_name)
                if prev_tmux.exists():
                    self.tmux = prev_tmux
                    self.evaluator_target = prev_eval_target
                    self._resumed_tmux = True
                    self.console.print(
                        f"[yellow]Resuming session {self.session.session_id} "
                        f"(tmux: {prev_tmux_name}) "
                        f"from round {self.session.completed_iteration + 1}[/yellow]"
                    )
                else:
                    self.console.print(
                        f"[yellow]Previous tmux session '{prev_tmux_name}' not found, "
                        f"creating new session. Resuming from round "
                        f"{self.session.completed_iteration + 1}[/yellow]"
                    )
            else:
                self.console.print(
                    f"[yellow]Resuming session {self.session.session_id} "
                    f"from round {self.session.completed_iteration + 1}[/yellow]"
                )
        else:
            self.session = SessionState.new()
            self.session.session_id = self.session_id

    def run(self) -> bool:
        """Entry point — run the tmux controller loop. Returns True on success."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._print_banner()

        try:
            if self._resumed_tmux:
                self.console.print(
                    f"[green]Reconnected to tmux session: {self.tmux.session_name}[/green]"
                )
            else:
                self._setup_tmux()
                self._send_initial_prompt()
            return self._controller_loop()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user.[/yellow]")
            self.console.print(
                f"[dim]Tmux session kept alive: {self.tmux.session_name}[/dim]"
            )
            self.console.print(
                f"[dim]Attach with: tmux attach -t {self.tmux.session_name}[/dim]"
            )
            self.session.status = "paused"
            self._save_session()
            return False
        except Exception:
            self.console.print(f"[red]Unexpected error:[/red]")
            self.console.print(traceback.format_exc())
            self.console.print(
                f"[dim]Tmux session kept alive: {self.tmux.session_name}[/dim]"
            )
            self.session.status = "error"
            self._save_session()
            raise

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_tmux(self) -> None:
        """Create tmux session with full-width worker and evaluator panes."""
        self.console.print(f"[cyan]Creating tmux session: {self.tmux.session_name}[/cyan]")
        self.logger.info("Creating tmux session: %s", self.tmux.session_name)
        self.tmux.create(cwd=Path.cwd())
        time.sleep(3)  # Wait for bash to initialize in the new pane

        # --- Worker pane (left, pane 0) ---
        claude_cmd = self._build_claude_cmd()
        worker_shell_cmd = shlex.join(claude_cmd)
        self.console.print(f"[dim]Worker   [pane 0] | Launching: {worker_shell_cmd}[/dim]")
        self.logger.info("Launching worker claude: %s", worker_shell_cmd)
        self.tmux.send_keys(worker_shell_cmd)

        # --- Evaluator pane (bottom, pane 1) — keep full width for readable logs ---
        self.evaluator_target = self.tmux.split_window(horizontal=False, cwd=Path.cwd())
        self.tmux.resize_window()
        time.sleep(3)  # Wait for bash to initialize in the split pane
        eval_cmd = self._build_claude_cmd()
        eval_shell_cmd = shlex.join(eval_cmd)
        self.console.print(f"[dim]Evaluator[pane 1] | Launching: {eval_shell_cmd}[/dim]")
        self.logger.info("Launching evaluator claude: %s", eval_shell_cmd)
        self.tmux.send_keys(eval_shell_cmd, target=self.evaluator_target)

        # Wait for both to be ready
        self.console.print("[dim]Waiting for Claude to start...[/dim]")
        self._wait_for_prompt(timeout=120)
        self._wait_for_prompt(timeout=120, target=self.evaluator_target)
        self.console.print("[green]Worker and evaluator are ready.[/green]")

    def _build_claude_cmd(self) -> list[str]:
        """Build the claude CLI command for interactive mode."""
        cmd = ["claude", "--model", self.config.model]
        if self.config.dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        elif self.config.allowed_tools:
            for tool in self.config.allowed_tools:
                cmd.extend(["--allowedTools", tool])
        if self.config.effort_level:
            cmd.extend(["--effort", self.config.effort_level])
        return cmd

    def _wait_for_prompt(self, timeout: int = 120, target: str | None = None) -> None:
        """Wait until Claude's interactive prompt ('>') appears."""
        start = time.time()
        while time.time() - start < timeout:
            content = self.tmux.capture_pane(lines=50, target=target)
            # Look for Claude Code's actual prompt indicators (not bash prompts)
            if "❯" in content or "╭" in content or "tips:" in content.lower() or "/help" in content:
                time.sleep(10)  # Give Claude CLI time to fully initialize
                return
            time.sleep(3)
        self.logger.warning("Timed out waiting for Claude prompt, proceeding anyway")

    def _send_initial_prompt(self) -> None:
        """Send the initial task prompt to the interactive Claude session."""
        task_prompt = self.config.prompt_path.read_text()
        plan = self.config.plan_path.read_text() if self.config.plan_path.exists() else "No plan yet."

        if self.config.additional_prompt:
            task_prompt += f"\n\n{self.config.additional_prompt}"

        prompt = TMUX_WORKER_INITIAL_PROMPT.format(
            task_prompt=task_prompt,
            plan=plan,
        )

        self.console.print("[cyan]Sending initial prompt to worker...[/cyan]")
        self.logger.info("Sending initial prompt (%d chars)", len(prompt))
        self.tmux.send_text_via_buffer(prompt)

    # ------------------------------------------------------------------
    # Controller loop
    # ------------------------------------------------------------------

    def _controller_loop(self) -> bool:
        """Core poll/decide/act loop."""
        self.session.status = "running"
        self._save_session()

        round_num = 0
        prev_hash: str | None = None
        stable_count = 0
        poll_interval = self.config.poll_interval  # dynamic — updated by agent

        while round_num < self.config.max_rounds:
            # Check for ESC pause
            if self.key_listener and self.key_listener.pause_requested.is_set():
                if not self._handle_pause(round_num):
                    return False

            # Check if tmux session is still alive
            if not self.tmux.is_alive():
                self.console.print("[red]Tmux session died unexpectedly.[/red]")
                self.session.status = "error"
                self._save_session()
                return False

            # Poll the pane
            time.sleep(poll_interval)
            pane_content = self.tmux.capture_pane(lines=500)

            # Check for permission prompts and auto-approve
            if self._detect_permission_prompt(pane_content):
                self.console.print("[yellow]Detected permission prompt, sending Y[/yellow]")
                self.logger.info("Auto-approving permission prompt")
                self.tmux.send_keys("Y")
                stable_count = 0
                prev_hash = None
                poll_interval = 5.0  # check soon after permission approval
                continue

            # Idle detection via content hashing
            content_hash = hashlib.md5(pane_content.encode()).hexdigest()
            if content_hash == prev_hash:
                stable_count += 1
            else:
                stable_count = 0
                prev_hash = content_hash

            if stable_count < self.config.idle_threshold:
                if stable_count == 0:
                    ts = datetime.now().strftime("%H:%M:%S")

                    # Rate-limit analysis API calls.
                    now = time.time()
                    last_call = self._last_analysis_time.get(self.tmux.target, 0.0)
                    if now - last_call >= self._analysis_min_interval:
                        analysis = self._analyze_pane(pane_content, self.tmux.target)
                        self._last_analysis_time[self.tmux.target] = now
                        summary = analysis.get("summary", "working...")
                        poll_interval = analysis.get("next_poll_interval", self.config.poll_interval)

                        # Only display if summary changed
                        if summary != self._last_displayed_summary.get(self.tmux.target):
                            self._last_displayed_summary[self.tmux.target] = summary
                            self.console.print(
                                f"[dim]{ts} Worker  [pane 0] | "
                                f"poll: {poll_interval:.0f}s[/dim]"
                            )
                            self.console.print(f"[dim]  {summary}[/dim]")

                        if self._act_on_analysis(analysis, self.tmux.target):
                            stable_count = 0
                            prev_hash = None
                            poll_interval = 5.0
                continue

            # Worker idle — invoke evaluator
            round_num += 1
            stable_count = 0
            prev_hash = None

            self.console.rule(f"[bold blue]Round {round_num}/{self.config.max_rounds}")
            self.console.print("[yellow]Worker idle — invoking controller agent...[/yellow]")
            self.logger.info("Round %d: worker idle, invoking controller", round_num)

            # Git commit before evaluation
            commit_hash = None
            if self.config.auto_commit:
                commit_hash = git_utils.commit_all(
                    message=f"claudeloop-tmux: round {round_num}"
                )
                if commit_hash:
                    self.console.print(f"[green]Committed:[/green] {commit_hash}")
                else:
                    self.console.print("[dim]No changes to commit[/dim]")

            # Invoke the controller agent
            decision = self._invoke_controller_agent()
            action = decision.get("action", "feedback")
            message = decision.get("message", "")
            controller_duration = decision.get("_duration", 0.0)

            # Record iteration
            self.session.completed_iteration = round_num
            self.session.last_commit_hash = commit_hash
            self.session.iterations.append(
                IterationRecord(
                    iteration=round_num,
                    worker_cost=0.0,
                    worker_duration=0.0,
                    worker_error=False,
                    commit_hash=commit_hash,
                    eval_cost=0.0,
                    eval_duration=controller_duration,
                    eval_success=(action == "success"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

            # Act on decision
            if action == "success":
                self.console.print(
                    Panel.fit(
                        f"[bold green]SUCCESS after {round_num} round(s)![/bold green]\n"
                        f"Reason: {message}",
                        title="Complete",
                        border_style="green",
                    )
                )
                self.session.status = "success"
                self._save_session()
                return True

            # Send feedback to worker
            self.console.print(
                Panel(
                    f"[yellow]Feedback:[/yellow] {message}",
                    title=f"Round {round_num} — Continuing",
                    border_style="yellow",
                )
            )
            self.logger.info("Round %d: sending feedback to worker: %s", round_num, message[:500])

            feedback_prompt = TMUX_WORKER_FOLLOWUP.format(
                round_num=round_num,
                feedback=message,
            )
            self.tmux.send_text_via_buffer(feedback_prompt)
            self._save_session()

        # Max rounds reached
        self.console.print(
            Panel.fit(
                f"[bold yellow]Max rounds ({self.config.max_rounds}) reached "
                f"without success.[/bold yellow]",
                title="Incomplete",
                border_style="yellow",
            )
        )
        self.session.status = "failed"
        self._save_session()
        return False

    # ------------------------------------------------------------------
    # Controller agent invocation
    # ------------------------------------------------------------------

    def _invoke_controller_agent(self) -> dict[str, Any]:
        """Send the evaluation prompt to the evaluator tmux pane and wait for its response."""
        assert self.evaluator_target is not None

        success_condition = self.config.success_path.read_text()

        prompt = TMUX_CONTROLLER_PROMPT.format(
            worker_target=self.tmux.target,
            success_condition=success_condition,
        )

        self.logger.info("Sending prompt to evaluator pane (%d chars)", len(prompt))
        self.tmux.send_text_via_buffer(prompt, target=self.evaluator_target)

        # Poll the evaluator pane until it becomes idle
        eval_start = time.time()
        prev_hash: str | None = None
        stable_count = 0
        eval_nudged = False
        eval_poll_interval = self.config.poll_interval  # dynamic

        while True:
            time.sleep(eval_poll_interval)
            eval_content = self.tmux.capture_pane(lines=500, target=self.evaluator_target)

            # Auto-approve permission prompts in evaluator pane (fast path)
            if self._detect_permission_prompt(eval_content):
                self.console.print("[yellow]Evaluator permission prompt, sending Y[/yellow]")
                self.tmux.send_keys("Y", target=self.evaluator_target)
                stable_count = 0
                prev_hash = None
                eval_nudged = False
                eval_poll_interval = 5.0
                continue

            content_hash = hashlib.md5(eval_content.encode()).hexdigest()
            if content_hash == prev_hash:
                stable_count += 1
            else:
                stable_count = 0
                prev_hash = content_hash
                eval_nudged = False

            if stable_count < self.config.idle_threshold:
                if stable_count == 0:
                    ts = datetime.now().strftime("%H:%M:%S")
                    now = time.time()
                    last_call = self._last_analysis_time.get(self.evaluator_target, 0.0)
                    if now - last_call >= self._analysis_min_interval:
                        analysis = self._analyze_pane(eval_content, self.evaluator_target)
                        self._last_analysis_time[self.evaluator_target] = now
                        summary = analysis.get("summary", "evaluating...")
                        eval_poll_interval = analysis.get("next_poll_interval", self.config.poll_interval)

                        if summary != self._last_displayed_summary.get(self.evaluator_target):
                            self._last_displayed_summary[self.evaluator_target] = summary
                            self.console.print(
                                f"[dim]{ts} Evaluator [pane 1] | "
                                f"poll: {eval_poll_interval:.0f}s[/dim]"
                            )
                            self.console.print(f"[dim]  {summary}[/dim]")

                        if self._act_on_analysis(analysis, self.evaluator_target):
                            stable_count = 0
                            prev_hash = None
                            eval_nudged = False
                            eval_poll_interval = 5.0
                continue

            # Evaluator appears idle — use agent to check if stuck
            if not eval_nudged:
                self.console.print("[dim]Evaluator appears idle, analyzing pane...[/dim]")
                analysis = self._analyze_pane(eval_content, self.evaluator_target)
                eval_poll_interval = analysis.get("next_poll_interval", self.config.poll_interval)
                eval_nudged = True
                if self._act_on_analysis(analysis, self.evaluator_target):
                    stable_count = 0
                    prev_hash = None
                    eval_poll_interval = 5.0
                    continue

            # Evaluator is genuinely idle — capture its output
            break

        eval_duration = time.time() - eval_start
        self.logger.info("Evaluator finished in %.0fs", eval_duration)

        # Use controller agent to interpret the evaluator's output and decide
        cleaned_eval = self._clean_pane_content(eval_content)
        decision = self._decide_from_eval(cleaned_eval)

        decision["_duration"] = eval_duration
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_controller_agent(
        self, prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 500,
    ) -> str | None:
        """Call controller agent via SDK or CLI fallback. Returns response text or None on failure."""
        if self._anthropic is not None:
            try:
                response = self._anthropic.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                self.logger.info("Controller agent call (%s): ok", model)
                return text
            except Exception as exc:
                self.logger.warning("SDK call failed (%s): %s — falling back to CLI", model, exc)

        return self._call_controller_agent_cli(prompt, model, max_tokens)

    def _call_controller_agent_cli(
        self, prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 500,
    ) -> str | None:
        """Fallback: call controller agent via claude CLI (uses OAuth auth)."""
        import subprocess, tempfile
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            result = subprocess.run(
                ["claude", "--model", model,
                 "--output-format", "text", "-p", prompt],
                capture_output=True, text=True, timeout=120,
            )
            import os
            os.unlink(prompt_file)
            if result.returncode == 0 and result.stdout.strip():
                text = result.stdout.strip()
                self.logger.info("Controller agent CLI call (%s): ok", model)
                return text
            else:
                self.logger.warning("Controller CLI failed: rc=%d, stderr=%s", result.returncode, result.stderr[:200])
                return None
        except Exception as exc:
            self.console.print(f"[red]Controller agent CLI call failed ({model}): {exc}[/red]")
            self.logger.warning("Controller agent CLI call failed (%s): %s", model, exc)
            return None

    def _decide_from_eval(self, eval_content: str) -> dict[str, Any]:
        """Use Opus to interpret the evaluator's output and decide next action."""
        text = self._call_controller_agent(
            prompt=(
                "You are a controller agent. An evaluator agent just finished checking "
                "whether a software project meets its success conditions. "
                "Read the evaluator's output below and decide:\n\n"
                "- If ALL tests passed and the evaluator concluded success, "
                'return: {"action": "success", "message": "brief summary of what passed"}\n'
                "- If tests failed or the evaluator found issues, "
                'return: {"action": "feedback", "message": "specific actionable feedback for the worker"}\n\n'
                "Return ONLY the JSON object, nothing else.\n\n"
                f"## Evaluator Output\n```\n{eval_content[:8000]}\n```"
            ),
            model=self.config.model,
            max_tokens=1000,
        )
        if text is not None:
            decision = parse_json_response(text)
            if decision is not None:
                return decision
            self.logger.warning("Controller decision unparseable: %s", text[:500])

        # Fallback: try regex JSON parse on raw evaluator output
        decision = parse_json_response(eval_content)
        if decision is not None:
            return decision

        return {
            "action": "feedback",
            "message": eval_content[-2000:] if len(eval_content) > 2000 else eval_content,
        }

    # Lines that are UI chrome / noise — skip these when analyzing
    _NOISE_PATTERNS = [
        r"bypass permissions",
        r"shift\+tab to cycle",
        r"esc to interrupt",
        r"ctrl\+t to hide",
        r"^[>\s❯]*$",              # empty prompt lines
        r"^\s*$",                   # blank
        r"^─+$",                    # horizontal rules
        r"^Press .* to ",           # key hints
    ]

    def _clean_pane_content(self, pane_content: str) -> str:
        """Strip ANSI codes and noise lines, return tail of meaningful content (up to 3000 chars)."""
        lines = pane_content.strip().splitlines()
        cleaned = [re.sub(r"\x1b\[[0-9;]*m", "", l).strip() for l in lines[-60:]]
        meaningful = [
            l for l in cleaned
            if l and not any(re.search(p, l, re.IGNORECASE) for p in self._NOISE_PATTERNS)
        ]
        if not meaningful:
            return ""
        # Keep the tail (most recent activity) rather than the head
        result = "\n".join(meaningful)
        if len(result) > 1000:
            result = result[-1000:]
        return result

    def _analyze_pane(self, pane_content: str, target: str) -> dict[str, Any]:
        """Analyze pane content using Sonnet 4.6: summarize + decide if keys are needed.

        Returns a dict with:
          - summary: str — what the agent is doing
          - needs_keys: bool — whether keys should be sent
          - keys: str — the keys to send (if needs_keys)
          - enter: bool — whether to press Enter after keys
          - reason: str — why keys are needed
          - next_poll_interval: float — suggested seconds until next poll
        """
        default_interval = self.config.poll_interval
        content = self._clean_pane_content(pane_content)
        if not content:
            return {
                "summary": "waiting...", "needs_keys": False,
                "keys": "", "enter": False, "reason": "",
                "next_poll_interval": default_interval,
            }

        pane_label = "worker" if target == self.tmux.target else "evaluator"

        # Build history context so the agent doesn't repeat itself
        history = self._summary_history.get(target, [])
        history_block = ""
        if history:
            past = "\n".join(f"- {s}" for s in history)
            history_block = (
                "\n## Previous Summaries (already reported — do NOT repeat this info)\n"
                f"{past}\n\n"
                "Focus your summary on what is NEW or CHANGED since the last summary. "
                "Do not restate information already covered above.\n"
            )

        text = self._call_controller_agent(
            prompt=(
                f"You are monitoring a Claude Code agent ({pane_label}) running in a tmux pane. "
                "Analyze its current output and provide THREE things:\n\n"
                "1. A concise summary (under 200 words) of what the agent is currently doing. "
                "Only report NEW progress — do not repeat previously reported information.\n"
                "2. Whether the agent needs keyboard input to continue.\n"
                "3. A suggested poll interval (seconds) for how soon to check again.\n\n"
                "Poll interval guidelines:\n"
                "- Agent is running a long build/test/install → 15-30s\n"
                "- Agent is actively editing files or reading code → 8-15s\n"
                "- Agent just started working on something → 10-15s\n"
                "- Agent seems close to finishing a step → 5-8s\n"
                "- Agent is waiting for input / stuck → 3-5s\n"
                f"- Default: {default_interval}s\n\n"
                "Common situations requiring input:\n"
                '- Permission prompt "Allow? [Y/n]" or similar → send "Y"\n'
                '- Plan mode selection (numbered options like "1.", "2.") → send the best option number\n'
                '- Confirmation prompt "Continue? [y/N]" → send "y"\n'
                '- Waiting for Enter / "press any key" → send Enter\n'
                '- Question asking for a choice → send the appropriate answer\n'
                "- Agent is actively working (spinner, running command) → no keys needed\n"
                "- Agent finished and is at the prompt → no keys needed\n\n"
                "Return a JSON object:\n"
                "```\n"
                "{\n"
                '  "summary": "concise summary of NEW progress only",\n'
                '  "needs_keys": true/false,\n'
                '  "keys": "text to send (empty string if not needed or just Enter)",\n'
                '  "enter": true/false,\n'
                '  "reason": "brief explanation of why keys are needed",\n'
                f'  "next_poll_interval": {default_interval}\n'
                "}\n"
                "```\n\n"
                "Return ONLY the JSON object.\n"
                f"{history_block}\n"
                f"## Pane Content\n```\n{content}\n```"
            ),
            model="claude-sonnet-4-6",
            max_tokens=500,
        )

        if text is not None:
            result = parse_json_response(text)
            if result is not None:
                # Clamp poll interval to sane range [3, 60]
                raw_interval = result.get("next_poll_interval", default_interval)
                try:
                    result["next_poll_interval"] = max(3.0, min(60.0, float(raw_interval)))
                except (ValueError, TypeError):
                    result["next_poll_interval"] = default_interval
                # Record summary in history
                summary = result.get("summary", "")
                if summary:
                    hist = self._summary_history.setdefault(target, [])
                    hist.append(summary)
                    if len(hist) > self._summary_history_max:
                        hist.pop(0)
                return result
            self.logger.debug("Pane analysis unparseable: %s", text[:300])

        # Fallback: return tail of raw content as summary, no keys
        return {
            "summary": content[-500:] if len(content) > 500 else content,
            "needs_keys": False,
            "keys": "",
            "enter": False,
            "reason": "",
            "next_poll_interval": default_interval,
        }

    def _act_on_analysis(self, analysis: dict[str, Any], target: str) -> bool:
        """Send keys based on pane analysis. Returns True if keys were sent."""
        if not analysis.get("needs_keys", False):
            return False

        keys = analysis.get("keys", "")
        enter = analysis.get("enter", True)
        reason = analysis.get("reason", "")
        pane_label = "worker" if target == self.tmux.target else "evaluator"

        self.console.print(
            f"[yellow]Sending keys to {pane_label}: {keys!r} (enter={enter}) — {reason}[/yellow]"
        )
        self.logger.info("Sending keys to %s: keys=%r enter=%s reason=%s", pane_label, keys, enter, reason)
        self.tmux.send_keys(keys, enter=enter, target=target)
        return True

    def _detect_permission_prompt(self, pane_content: str) -> bool:
        """Detect if the pane shows a permission prompt like 'Allow? [Y/n]'."""
        # Look for common permission prompt patterns in the last few lines
        lines = pane_content.strip().splitlines()
        tail = "\n".join(lines[-10:]) if len(lines) > 10 else pane_content
        patterns = [
            r"Allow\?.*\[Y/n\]",
            r"allow\?.*\[y/n\]",
            r"Do you want to proceed\?",
        ]
        for pattern in patterns:
            if re.search(pattern, tail, re.IGNORECASE):
                return True
        return False

    def _handle_pause(self, round_num: int) -> bool:
        """Handle ESC pause. Returns True to continue, False to quit."""
        if self.key_listener:
            self.key_listener.pause_listening()

        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold yellow]PAUSED[/bold yellow] (ESC pressed)\n\n"
                "Options:\n"
                "  - Type feedback to send to the worker, then Enter on empty line\n"
                "  - Press Enter immediately to continue\n"
                "  - Type [bold]q[/bold] and Enter to quit",
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
                except (EOFError, OSError):
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
            feedback_prompt = TMUX_WORKER_FOLLOWUP.format(
                round_num=round_num,
                feedback=feedback_text,
            )
            self.tmux.send_text_via_buffer(feedback_prompt)
            self.console.print("[green]Feedback sent to worker.[/green]")

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

    def _load_session(self) -> SessionState:
        """Load session state from disk."""
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

    def _save_session(self) -> None:
        """Save session state to disk, including tmux session name for resume."""
        self.session.updated_at = datetime.now(timezone.utc).isoformat()
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.session.to_dict()
        # Save tmux-specific info for resume
        data["tmux_session_name"] = self.tmux.session_name
        data["tmux_evaluator_target"] = self.evaluator_target
        data["max_rounds"] = self.config.max_rounds
        data["current_round"] = (
            min(self.session.completed_iteration + 1, self.config.max_rounds)
            if self.session.status == "running"
            else self.session.completed_iteration
        )
        tmp = self._session_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._session_path)

    def _cleanup(self) -> None:
        """Kill the tmux session on exit."""
        if self.tmux.exists():
            self.console.print(f"[dim]Cleaning up tmux session: {self.tmux.session_name}[/dim]")
            self.tmux.kill()

    def _print_banner(self) -> None:
        """Print startup banner."""
        esc_hint = ""
        if self.key_listener is not None:
            esc_hint = "\n[dim]Press ESC at any time to pause[/dim]"
        self.console.print(
            Panel.fit(
                f"[bold]claudeloop[/bold] - Tmux Controller Mode\n"
                f"Session: {self.tmux.session_name}\n"
                f"Model: {self.config.model}\n"
                f"Max rounds: {self.config.max_rounds}\n"
                f"Poll interval: {self.config.poll_interval}s\n"
                f"Idle threshold: {self.config.idle_threshold} stable polls\n"
                f"Effort level: {self.config.effort_level or 'default'}\n"
                f"Fast mode: {'on' if self.config.fast_mode else 'off'}"
                f"{esc_hint}\n\n"
                f"[dim]Attach: tmux attach -t {self.tmux.session_name}[/dim]\n"
                f"[dim]Switch panes: Ctrl-b + arrow keys[/dim]",
                title="Tmux Controller",
            )
        )
