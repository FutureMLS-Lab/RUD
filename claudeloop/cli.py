"""CLI entry point for claudeloop."""

from __future__ import annotations

import shutil
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from claudeloop.key_listener import KeyListener
from claudeloop.openclaw import (
    build_openclaw_config,
    openclaw_status,
)
from claudeloop.paths import bundled_skills_path
from claudeloop.runner import DEFAULT_ALLOWED_TOOLS, RunConfig, Runner
from claudeloop.tmux_controller import TmuxConfig, TmuxController

app = typer.Typer(
    name="claudeloop",
    help="Self-improving agentic loop using Claude Code.",
    add_completion=False,
)
console = Console()


class BackendChoice(str, Enum):
    cli = "cli"
    sdk = "sdk"


# --- Templates shipped with the package ---
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


@app.command()
def run(
    prompt: Path = typer.Option(
        Path("TASK_PROMPT.md"), "--prompt", "-p", help="Path to task prompt file"
    ),
    success: Path = typer.Option(
        Path("SUCCESS_CONDITION.md"), "--success", "-s", help="Path to success condition file"
    ),
    plan: Path = typer.Option(
        Path("PLAN.md"), "--plan", help="Path to plan file"
    ),
    max_iters: int = typer.Option(
        50, "--max-iters", "-n", help="Maximum iterations"
    ),
    model: str = typer.Option(
        "claude-opus-4-6", "--model", "-m", help="Claude model to use"
    ),
    backend: BackendChoice = typer.Option(
        BackendChoice.cli,
        "--backend",
        "-b",
        help="Backend for evaluator (worker always uses CLI)",
    ),
    log_dir: Path = typer.Option(
        Path("agent_logs"), "--log-dir", help="Directory for logs"
    ),
    no_commit: bool = typer.Option(
        False, "--no-commit", help="Don't auto-commit after each iteration"
    ),
    max_cost: float = typer.Option(
        1000.0, "--max-cost", help="Deprecated; ignored", hidden=True
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    effort_level: str = typer.Option(
        None, "--effort", "-e", help="Thinking effort level: low, medium, high, or max"
    ),
    fast_mode: bool = typer.Option(
        False, "--fast", help="Enable fast mode"
    ),
    additional_prompt: str = typer.Option(
        None, "--additional-prompt", "--ap", help="Additional prompt text appended to the task prompt"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from previous session (reads session.json from log dir)"
    ),
    dangerously_skip_permissions: bool = typer.Option(
        False, "--dangerously-skip-permissions", help="Skip all permission prompts in Claude Code (use with caution)"
    ),
    allowed_tools: str = typer.Option(
        None,
        "--allowed-tools",
        help="Comma-separated list of tools to auto-approve (default: all tools). "
        "Ignored when --dangerously-skip-permissions is set.",
    ),
) -> None:
    """Run the self-improving agentic loop."""
    # Validate required files
    if not prompt.exists():
        console.print(f"[red]Error:[/red] Prompt file not found: {prompt}")
        console.print("Run [bold]claudeloop init[/bold] to create template files.")
        raise typer.Exit(1)
    if not success.exists():
        console.print(f"[red]Error:[/red] Success condition file not found: {success}")
        console.print("Run [bold]claudeloop init[/bold] to create template files.")
        raise typer.Exit(1)

    # Parse allowed tools: use user-provided list, or default to all tools
    if allowed_tools is not None:
        tools_list = [t.strip() for t in allowed_tools.split(",") if t.strip()]
    else:
        tools_list = list(DEFAULT_ALLOWED_TOOLS)

    config = RunConfig(
        prompt_path=prompt.resolve(),
        success_path=success.resolve(),
        plan_path=plan.resolve(),
        max_iters=max_iters,
        model=model,
        backend_name=backend.value,
        log_dir=log_dir.resolve(),
        max_cost=max_cost,
        auto_commit=not no_commit,
        verbose=verbose,
        additional_prompt=additional_prompt,
        effort_level=effort_level,
        fast_mode=fast_mode,
        resume=resume,
        dangerously_skip_permissions=dangerously_skip_permissions,
        allowed_tools=tools_list,
    )

    with KeyListener() as listener:
        runner = Runner(config, key_listener=listener)
        succeeded = runner.run_loop()
    raise typer.Exit(0 if succeeded else 1)


@app.command()
def tmux(
    prompt: Path = typer.Option(
        Path("TASK_PROMPT.md"), "--prompt", "-p", help="Path to task prompt file"
    ),
    success: Path = typer.Option(
        Path("SUCCESS_CONDITION.md"), "--success", "-s", help="Path to success condition file"
    ),
    plan: Path = typer.Option(
        Path("PLAN.md"), "--plan", help="Path to plan file"
    ),
    max_iters: int = typer.Option(
        50, "--max-iters", "--max-rounds", "-n", help="Maximum controller rounds"
    ),
    model: str = typer.Option(
        "claude-opus-4-6", "--model", "-m", help="Claude model to use"
    ),
    log_dir: Path = typer.Option(
        Path("agent_logs"), "--log-dir", help="Directory for logs"
    ),
    no_commit: bool = typer.Option(
        False, "--no-commit", help="Don't auto-commit after each round"
    ),
    max_cost: float = typer.Option(
        1000.0, "--max-cost", help="Deprecated; ignored", hidden=True
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    effort_level: str = typer.Option(
        None, "--effort", "-e", help="Thinking effort level: low, medium, high, or max"
    ),
    fast_mode: bool = typer.Option(
        False, "--fast", help="Enable fast mode"
    ),
    additional_prompt: str = typer.Option(
        None, "--additional-prompt", "--ap", help="Additional prompt text appended to the task prompt"
    ),
    poll_interval: float = typer.Option(
        10.0, "--poll-interval", help="Seconds between idle-detection polls"
    ),
    idle_threshold: int = typer.Option(
        3, "--idle-threshold", help="Consecutive stable captures before considering idle"
    ),
    session_name: str = typer.Option(
        None, "--session-name", help="Tmux session name (auto-generated if not set)"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from previous session (reads session.json from log dir)"
    ),
    dangerously_skip_permissions: bool = typer.Option(
        False, "--dangerously-skip-permissions", help="Skip all permission prompts in Claude Code (use with caution)"
    ),
    allowed_tools: str = typer.Option(
        None,
        "--allowed-tools",
        help="Comma-separated list of tools to auto-approve (default: all tools). "
        "Ignored when --dangerously-skip-permissions is set.",
    ),
) -> None:
    """Run the tmux-based controller mode.

    Launches an interactive Claude Code session in a tmux pane, monitors
    it via polling, and uses a separate controller agent to evaluate
    progress and send feedback.
    """
    # Validate required files
    if not prompt.exists():
        console.print(f"[red]Error:[/red] Prompt file not found: {prompt}")
        console.print("Run [bold]claudeloop init[/bold] to create template files.")
        raise typer.Exit(1)
    if not success.exists():
        console.print(f"[red]Error:[/red] Success condition file not found: {success}")
        console.print("Run [bold]claudeloop init[/bold] to create template files.")
        raise typer.Exit(1)

    # Parse allowed tools
    if allowed_tools is not None:
        tools_list = [t.strip() for t in allowed_tools.split(",") if t.strip()]
    else:
        tools_list = list(DEFAULT_ALLOWED_TOOLS)

    config = TmuxConfig(
        prompt_path=prompt.resolve(),
        success_path=success.resolve(),
        plan_path=plan.resolve(),
        max_rounds=max_iters,
        model=model,
        log_dir=log_dir.resolve(),
        max_cost=max_cost,
        auto_commit=not no_commit,
        verbose=verbose,
        poll_interval=poll_interval,
        idle_threshold=idle_threshold,
        session_name=session_name,
        additional_prompt=additional_prompt,
        effort_level=effort_level,
        fast_mode=fast_mode,
        resume=resume,
        dangerously_skip_permissions=dangerously_skip_permissions,
        allowed_tools=tools_list,
    )

    with KeyListener() as listener:
        controller = TmuxController(config, key_listener=listener)
        succeeded = controller.run()
    raise typer.Exit(0 if succeeded else 1)


@app.command()
def init() -> None:
    """Create template TASK_PROMPT.md, SUCCESS_CONDITION.md, and PLAN.md files."""
    templates = {
        "TASK_PROMPT.md": _TEMPLATES_DIR / "TASK_PROMPT.md",
        "SUCCESS_CONDITION.md": _TEMPLATES_DIR / "SUCCESS_CONDITION.md",
        "PLAN.md": _TEMPLATES_DIR / "PLAN.md",
    }

    created = 0
    for name, src in templates.items():
        dest = Path.cwd() / name
        if dest.exists():
            console.print(f"[yellow]Skipped:[/yellow] {name} already exists")
            continue
        if src.exists():
            shutil.copy2(src, dest)
        else:
            # Fallback: write inline defaults
            dest.write_text(_INLINE_TEMPLATES[name])
        console.print(f"[green]Created:[/green] {name}")
        created += 1

    if created:
        console.print(
            "\nEdit these files to define your task, then run [bold]claudeloop run[/bold]."
        )
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
        help="Default skills markdown for new tasks (package default: claudeloop/skills/AK_skills.md)",
    ),
    work_dir: list[Path] | None = typer.Option(
        None,
        "--work-dir",
        help="Default git repo path(s) for new tasks (repeatable)",
    ),
    interview_backend: str = typer.Option(
        "cli",
        "--interview-backend",
        help="Interview: cli (claude CLI) or sdk (Anthropic API)",
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
        help="Enable direct claudeloop -> OpenClaw gateway events",
    ),
    openclaw_url: str | None = typer.Option(
        None,
        "--openclaw-url",
        help="OpenClaw gateway URL to POST claudeloop events to",
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
        help="claudeloop OpenClaw JSON config with url, headers, timeout, enabled",
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
    openclaw_deliver: bool = typer.Option(
        False,
        "--openclaw-deliver",
        help="Set deliver=true for /hooks/agent payloads",
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
        help="Enable claudeloop OpenClaw debug logging",
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
    """Start local web UI for `.RUD` tasks (interview, templates, workers)."""
    from claudeloop.web import serve

    root = (project or Path.cwd()).resolve()
    wd = [p.expanduser().resolve() for p in (work_dir or [])]
    ib = interview_backend.lower() if interview_backend.lower() in ("cli", "sdk") else "cli"
    web_auth_token = (auth_token or os.environ.get("CLAUDELOOP_WEB_AUTH_TOKEN", "")).strip()
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
        deliver=openclaw_deliver,
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
            "claudeloop",
            "web",
            "--host",
            host,
            "--port",
            str(port),
            "--project",
            str(root),
            "--skills",
            str(skills.resolve()),
            "--interview-backend",
            ib,
        ]
        child_env = os.environ.copy()
        if web_auth_token:
            child_env["CLAUDELOOP_WEB_AUTH_TOKEN"] = web_auth_token
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
        if openclaw_deliver:
            cmd.append("--openclaw-deliver")
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
        for p in wd:
            cmd.extend(["--work-dir", str(p)])
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
        console.print(f"[green]claudeloop web started in background[/green] pid={proc.pid}")
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
        wd,
        interview_backend_default=ib,
        openclaw_config=openclaw_cfg,
        auth_token=web_auth_token,
        multi_project_workspace=projects,
    )


# Inline fallback templates (used if templates/ dir is missing)
_INLINE_TEMPLATES = {
    "TASK_PROMPT.md": """\
# Task Prompt

## Role
You are a software development agent. Your goal is to implement the project
described below through iterative development.

## Project Description
<!-- Describe what you want to build here -->
TODO: Describe your project

## Constraints
- Write clean, well-documented code
- Include tests for new functionality
- Follow the existing code style and conventions
- Keep task status, decisions, next steps, and progress logs in the task directory's PLAN.md
- Do not create scattered TODO/NOTES/PROGRESS/status files in the repo or worktree

## Tools Available
You have access to Claude Code tools: Bash, Edit, Write, Read, Glob, Grep.
Use them to explore the codebase, write code, and run tests.
""",
    "SUCCESS_CONDITION.md": """\
# Success Conditions

## Criteria
<!-- Define what "done" means for your project -->
1. TODO: Define success criterion 1
2. TODO: Define success criterion 2

## Test Commands
The following commands must all pass (exit code 0):

```bash
# TODO: Add your test commands here
# Example:
# pytest tests/ -v
# python -c "import mymodule; print('import ok')"
```

## Notes
- All test commands must exit with code 0 for success
- The evaluator will also check qualitative criteria above
""",
    "PLAN.md": """\
# Plan

## Status
Not started

## Tasks
- [ ] TODO: First task

## Next Steps
1. Start with the first task above

## Progress Log
<!-- The agent will update this section after each iteration -->
""",
}
