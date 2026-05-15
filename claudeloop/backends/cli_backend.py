"""Claude CLI subprocess backend."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from .base import Backend, BackendResponse, StreamEvent


class CLIBackend(Backend):
    """Backend that invokes the ``claude`` CLI as a subprocess."""

    def __init__(
        self,
        claude_path: str | None = None,
        timeout: int = 1800,
        effort_level: str | None = None,
        fast_mode: bool = False,
    ):
        resolved = claude_path or shutil.which("claude")
        if resolved is None:
            raise FileNotFoundError(
                "claude CLI not found on PATH. "
                "Install it from https://docs.anthropic.com/en/docs/claude-code"
            )
        self.claude_path = resolved
        self.timeout = timeout
        self.effort_level = effort_level  # "low", "medium", or "high"
        self.fast_mode = fast_mode
        self._current_proc: subprocess.Popen | None = None
        self._cancelled: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(
        self,
        prompt: str,
        model: str,
        dangerously_skip_permissions: bool = False,
        allowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> BackendResponse:
        if on_event is not None:
            resp = self._invoke_streaming(
                prompt, model, dangerously_skip_permissions, allowed_tools, system_prompt, on_event,
            )
            # If streaming failed (e.g. unsupported format), fall back to quiet
            if resp.is_error and "output-format" in resp.text.lower():
                on_event(StreamEvent(
                    type="system",
                    message="Streaming not supported, falling back to quiet mode",
                ))
                return self._invoke_quiet(
                    prompt, model, dangerously_skip_permissions, allowed_tools, system_prompt,
                )
            return resp
        return self._invoke_quiet(
            prompt, model, dangerously_skip_permissions, allowed_tools, system_prompt,
        )

    def cancel(self) -> None:
        """Cancel the currently running subprocess, if any."""
        self._cancelled = True
        proc = self._current_proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            except Exception:
                pass

    def reset_cancel(self) -> None:
        """Reset the cancelled flag for the next invocation."""
        self._cancelled = False

    # ------------------------------------------------------------------
    # Quiet mode  (--output-format json, captures everything)
    # ------------------------------------------------------------------

    def _invoke_quiet(
        self,
        prompt: str,
        model: str,
        dangerously_skip_permissions: bool,
        allowed_tools: list[str] | None,
        system_prompt: str | None,
    ) -> BackendResponse:
        cmd = self._build_cmd(model, dangerously_skip_permissions, allowed_tools, system_prompt, "json")

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                input=prompt,
                timeout=self.timeout, cwd=Path.cwd(),
            )
        except subprocess.TimeoutExpired:
            return BackendResponse(
                text="ERROR: Claude CLI timed out",
                is_error=True, cost_usd=0.0,
                duration_secs=float(self.timeout),
            )

        try:
            data = json.loads(proc.stdout)
            return BackendResponse(
                text=data.get("result", proc.stdout),
                is_error=data.get("is_error", False),
                cost_usd=data.get("total_cost_usd", 0.0),
                duration_secs=data.get("duration_ms", 0) / 1000.0,
                raw=data,
            )
        except (json.JSONDecodeError, TypeError):
            return BackendResponse(
                text=proc.stdout or proc.stderr,
                is_error=(proc.returncode != 0),
                cost_usd=0.0, duration_secs=0.0, raw=None,
            )

    # ------------------------------------------------------------------
    # Streaming mode  (--output-format stream-json, real-time events)
    # ------------------------------------------------------------------

    def _invoke_streaming(
        self,
        prompt: str,
        model: str,
        dangerously_skip_permissions: bool,
        allowed_tools: list[str] | None,
        system_prompt: str | None,
        on_event: Callable[[StreamEvent], None],
    ) -> BackendResponse:
        cmd = self._build_cmd(
            model, dangerously_skip_permissions, allowed_tools, system_prompt, "stream-json",
        )

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=Path.cwd(),
            )
            # Send prompt via stdin, then close to signal EOF
            proc.stdin.write(prompt)
            proc.stdin.close()
            self._current_proc = proc
        except Exception as exc:
            return BackendResponse(
                text=f"ERROR: Failed to start claude CLI: {exc}",
                is_error=True, cost_usd=0.0, duration_secs=0.0,
            )

        # Read stderr in a background thread to avoid deadlocks and
        # to capture diagnostics if something goes wrong.
        stderr_lines: list[str] = []

        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        result_text = ""
        cost_usd = 0.0
        duration_secs = 0.0
        is_error = False
        raw: dict | None = None
        got_any_event = False

        assert proc.stdout is not None
        for line in proc.stdout:
            if self._cancelled:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON output — could be an error message from the CLI
                stderr_lines.append(line)
                continue

            got_any_event = True
            ev_type = event.get("type", "")

            if ev_type == "assistant":
                self._handle_assistant_event(event, on_event)

            elif ev_type == "tool_use":
                tool = event.get("tool", "unknown")
                on_event(StreamEvent(
                    type="tool_use",
                    message=f"Using tool: {tool}",
                    data=event,
                ))

            elif ev_type == "tool_result":
                on_event(StreamEvent(
                    type="tool_result",
                    message="Tool finished",
                    data=event,
                ))

            elif ev_type == "result":
                result_text = event.get("result", "")
                cost_usd = event.get("total_cost_usd", 0.0)
                duration_secs = event.get("duration_ms", 0) / 1000.0
                is_error = event.get("is_error", False)
                raw = event
                on_event(StreamEvent(
                    type="result",
                    message=f"Done (${cost_usd:.4f}, {duration_secs:.0f}s)",
                    data=event,
                ))

            elif ev_type == "error":
                is_error = True
                on_event(StreamEvent(
                    type="error",
                    message=event.get("error", {}).get("message", str(event)),
                    data=event,
                ))

            elif ev_type == "system":
                msg = event.get("message", "") or event.get("subtype", "")
                if msg:
                    on_event(StreamEvent(type="system", message=msg, data=event))

        self._current_proc = None

        if self._cancelled:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            stderr_thread.join(timeout=5)
            return BackendResponse(
                text="Cancelled by user",
                is_error=False,
                cost_usd=cost_usd,
                duration_secs=duration_secs,
                cancelled=True,
                raw=raw,
            )

        proc.wait(timeout=60)
        stderr_thread.join(timeout=5)
        stderr_output = "".join(stderr_lines).strip()

        # If we never got a valid event, the process likely failed immediately
        if not got_any_event and (proc.returncode or stderr_output):
            error_msg = stderr_output or f"claude CLI exited with code {proc.returncode}"
            on_event(StreamEvent(type="error", message=error_msg))
            return BackendResponse(
                text=error_msg,
                is_error=True, cost_usd=0.0, duration_secs=0.0,
            )

        if proc.returncode and not is_error:
            is_error = True
            if stderr_output and not result_text:
                result_text = stderr_output

        return BackendResponse(
            text=result_text,
            is_error=is_error,
            cost_usd=cost_usd,
            duration_secs=duration_secs,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_assistant_event(
        event: dict, on_event: Callable[[StreamEvent], None],
    ) -> None:
        """Parse an assistant message event and forward relevant info."""
        content_parts = event.get("message", {}).get("content", [])
        if isinstance(content_parts, str):
            content_parts = [{"type": "text", "text": content_parts}]

        for part in content_parts:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "tool_use":
                name = part.get("name", "unknown")
                inp = part.get("input", {})
                summary = _tool_summary(name, inp)
                on_event(StreamEvent(
                    type="tool_use",
                    message=summary,
                    data=part,
                ))
            elif ptype == "text":
                text = part.get("text", "").strip()
                if text:
                    preview = text[:1000] + ("..." if len(text) > 1000 else "")
                    on_event(StreamEvent(
                        type="text",
                        message=preview,
                        data=part,
                    ))

    def _build_cmd(
        self,
        model: str,
        dangerously_skip_permissions: bool,
        allowed_tools: list[str] | None,
        system_prompt: str | None,
        output_format: str,
    ) -> list[str]:
        cmd: list[str] = [
            self.claude_path, "-p", "-",
            "--model", model,
            "--output-format", output_format,
        ]
        # stream-json requires --verbose in print mode
        if output_format == "stream-json":
            cmd.append("--verbose")
        if dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        elif allowed_tools:
            for tool in allowed_tools:
                cmd.extend(["--allowedTools", tool])
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if self.effort_level:
            cmd.extend(["--effort", self.effort_level])
        return cmd


def _tool_summary(name: str, inp: dict) -> str:
    """Build a concise one-line summary of a tool call."""
    if name == "Edit":
        fp = inp.get("file_path", "?")
        return f"Edit: {fp}"
    if name == "Write":
        fp = inp.get("file_path", "?")
        return f"Write: {fp}"
    if name == "Read":
        fp = inp.get("file_path", "?")
        return f"Read: {fp}"
    if name == "Bash":
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        label = desc or (cmd[:80] + ("..." if len(cmd) > 80 else ""))
        return f"Bash: {label}"
    if name in ("Glob", "Grep"):
        pattern = inp.get("pattern", "")
        return f"{name}: {pattern}"
    return f"{name}"
