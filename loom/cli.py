"""CLI entry point for Loom.

Only two commands now that the agent-loop machinery is gone:
- ``loom init`` writes the minimal PLAN.md / NOTES.md templates
  into the current directory.
- ``loom web`` runs the local web UI for browsing / editing tasks
  and launching the deep-interview pane.

(``loom`` remains as a legacy alias for ``loom``.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from loom.openclaw import build_openclaw_config, openclaw_status
from loom.paths import bundled_skills_path

app = typer.Typer(
    name="loom",
    help="Loom - a lightweight task console for Claude Code / Codex (deep interview + PLAN.md + worktrees + diffs + notes).",
    add_completion=False,
)
console = Console()

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Fallback templates used when /templates is missing (e.g. installed wheel
# without source dir).  Kept small on purpose - the real content comes from
# the deep-interview pane and the user's own edits.
_INLINE_TEMPLATES = {
    "PLAN.md": """\
# Plan

## Status
Not started

## Goal
<!-- Filled in from the task goal; the deep-interview pane usually rewrites
     this whole section.  Run `/goal` in Claude Code to act on it. -->

## Next steps
- [ ] TODO
""",
    "NOTES.md": """\
# Notes

Free-form scratch space for future work, ideas, things to come back to.
""",
}


@app.command()
def init() -> None:
    """Create template PLAN.md and NOTES.md in the current directory."""
    created = 0
    for name in ("PLAN.md", "NOTES.md"):
        dest = Path.cwd() / name
        if dest.exists():
            console.print(f"[yellow]Skipped:[/yellow] {name} already exists")
            continue
        src = _TEMPLATES_DIR / name
        if src.is_file():
            shutil.copy2(src, dest)
        else:
            dest.write_text(_INLINE_TEMPLATES[name], encoding="utf-8")
        console.print(f"[green]Created:[/green] {name}")
        created += 1
    if created:
        console.print("\nEdit PLAN.md to describe your goal, NOTES.md for future ideas.")
    else:
        console.print("\nAll template files already exist.")


@app.command("web")
def web_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8765, "--port", help="HTTP port"),
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Project root (git checkout); defaults to current directory",
    ),
    skills: Path = typer.Option(
        bundled_skills_path(),
        "--skills",
        help="Default skills markdown for new tasks (package default: loom/skills/charlie_skills.md)",
    ),
    daemon: bool = typer.Option(
        False,
        "--daemon",
        "--nohup",
        help="Start the web server in the background and exit",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Daemon log file; defaults to <project>/.RUD/web.log",
    ),
    auth_token: str | None = typer.Option(
        None,
        "--auth-token",
        help="Require HTTP auth for the web UI/API; username can be anything, password is this token",
    ),
    openclaw: bool = typer.Option(
        False,
        "--openclaw",
        help="Enable direct Loom -> OpenClaw gateway events",
    ),
    openclaw_url: str | None = typer.Option(
        None,
        "--openclaw-url",
        help="OpenClaw gateway URL to POST Loom events to",
    ),
    openclaw_token: str | None = typer.Option(
        None,
        "--openclaw-token",
        help="OpenClaw hooks token; sent as Authorization: Bearer <token>",
    ),
    openclaw_header: list[str] | None = typer.Option(
        None,
        "--openclaw-header",
        help="Header for OpenClaw requests, repeatable. Use 'Name: value' or 'Name=value'",
    ),
    openclaw_config: Path | None = typer.Option(
        None,
        "--openclaw-config",
        help="Loom OpenClaw JSON config with url, headers, timeout, enabled",
    ),
    openclaw_timeout_ms: int = typer.Option(
        10000,
        "--openclaw-timeout-ms",
        help="OpenClaw request timeout in milliseconds",
    ),
    openclaw_hook: str | None = typer.Option(
        None,
        "--openclaw-hook",
        help="OpenClaw HTTP hook payload type: wake or agent; inferred from URL if omitted",
    ),
    openclaw_wake_mode: str = typer.Option(
        "now",
        "--openclaw-wake-mode",
        help="OpenClaw wake mode: now or next-heartbeat",
    ),
    openclaw_agent_name: str | None = typer.Option(
        None,
        "--openclaw-agent-name",
        help="Name field for /hooks/agent payloads",
    ),
    openclaw_agent_id: str | None = typer.Option(
        None,
        "--openclaw-agent-id",
        help="Optional agentId for /hooks/agent payloads",
    ),
    openclaw_channel: str | None = typer.Option(
        None,
        "--openclaw-channel",
        help="Optional channel for /hooks/agent delivery, such as slack",
    ),
    openclaw_to: str | None = typer.Option(
        None,
        "--openclaw-to",
        help="Optional delivery target for /hooks/agent, such as channel:C123",
    ),
    openclaw_debug: bool = typer.Option(
        False,
        "--openclaw-debug",
        help="Enable Loom OpenClaw debug logging",
    ),
    projects: bool = typer.Option(
        False,
        "--projects",
        help=(
            "Multi-project workspace: launch directory is a container for several git repos; "
            "drop a redundant registry row for the launch path when child repos are registered. "
            "Omit this if the launch directory itself is a normal single project root."
        ),
    ),
) -> None:
    """Start local web UI for `.RUD` tasks (interview, PLAN.md, NOTES.md)."""
    from loom.web import serve

    root = (project or Path.cwd()).resolve()
    web_auth_token = (auth_token or os.environ.get("LOOM_WEB_AUTH_TOKEN", "")).strip()
    openclaw_cfg = build_openclaw_config(
        enabled=openclaw,
        url=openclaw_url,
        token=openclaw_token,
        headers=openclaw_header,
        config_path=openclaw_config,
        timeout_ms=openclaw_timeout_ms,
        hook=openclaw_hook,
        wake_mode=openclaw_wake_mode,
        agent_name=openclaw_agent_name,
        agent_id=openclaw_agent_id,
        channel=openclaw_channel,
        to=openclaw_to,
        debug=openclaw_debug,
    )
    if daemon:
        log_path = (log_file.expanduser().resolve() if log_file else root / ".RUD" / "web.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "loom",
            "web",
            "--host",
            host,
            "--port",
            str(port),
            "--project",
            str(root),
            "--skills",
            str(skills.resolve()),
        ]
        child_env = os.environ.copy()
        if web_auth_token:
            child_env["LOOM_WEB_AUTH_TOKEN"] = web_auth_token
        if openclaw_cfg.enabled:
            cmd.append("--openclaw")
        if openclaw_url:
            cmd.extend(["--openclaw-url", openclaw_url])
        if openclaw_token:
            cmd.extend(["--openclaw-token", openclaw_token])
        if openclaw_config:
            cmd.extend(["--openclaw-config", str(openclaw_config.expanduser().resolve())])
        if openclaw_timeout_ms != 10000:
            cmd.extend(["--openclaw-timeout-ms", str(openclaw_timeout_ms)])
        if openclaw_hook:
            cmd.extend(["--openclaw-hook", openclaw_hook])
        if openclaw_wake_mode != "now":
            cmd.extend(["--openclaw-wake-mode", openclaw_wake_mode])
        if openclaw_agent_name:
            cmd.extend(["--openclaw-agent-name", openclaw_agent_name])
        if openclaw_agent_id:
            cmd.extend(["--openclaw-agent-id", openclaw_agent_id])
        if openclaw_channel:
            cmd.extend(["--openclaw-channel", openclaw_channel])
        if openclaw_to:
            cmd.extend(["--openclaw-to", openclaw_to])
        if openclaw_debug:
            cmd.append("--openclaw-debug")
        if projects:
            cmd.append("--projects")
        for h in openclaw_header or []:
            cmd.extend(["--openclaw-header", h])
        with open(log_path, "ab", buffering=0) as out:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=str(root),
                env=child_env,
                start_new_session=True,
                close_fds=True,
            )
        console.print(f"[green]Loom started in background[/green] pid={proc.pid}")
        console.print(f"[dim]URL:[/dim] http://{host}:{port}/")
        console.print(f"[dim]Log:[/dim] {log_path}")
        if openclaw_cfg.enabled:
            console.print(f"[dim]OpenClaw:[/dim] {openclaw_status(openclaw_cfg)}")
        if web_auth_token:
            console.print("[dim]Auth:[/dim] enabled")
        return
    serve(
        host,
        port,
        root,
        skills.resolve(),
        openclaw_config=openclaw_cfg,
        auth_token=web_auth_token,
        multi_project_workspace=projects,
    )
