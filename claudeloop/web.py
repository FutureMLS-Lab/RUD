"""Lightweight local web UI for `.RUD` tasks (interview, templates, workers)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from claudeloop.openclaw import OpenClawClient, OpenClawConfig, openclaw_status
from claudeloop.paths import bundled_skills_path, web_static_dir
from claudeloop.web_projects import WebProjectRegistry
from claudeloop.rud_task import (
    PLAN,
    SUCCESS_CONDITION,
    TASK_PROMPT,
    WORK_ROOT_REPO_KEY,
    create_task,
    delete_task,
    list_tasks,
    list_work_repo_keys,
    prepare_all_worktrees,
    read_interview,
    read_meta,
    read_template,
    reorder_tasks,
    runs_dir_for_repo,
    task_root,
    update_meta,
    validate_repo_key,
    work_path_for_repo_key,
    write_template,
    direct_child_git_repos,
    git_toplevel,
    prepare_selected_worktrees,
)
from claudeloop.tmux_util import (
    capture_pane,
    list_tmux_panes,
    list_tmux_sessions,
    send_pane_key,
    send_pane_text,
    tmux_available,
    tmux_subprocess_env,
    validate_tmux_target,
)

_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_STATIC_MIME: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}

_WORKER_REPO_DEFAULT = WORK_ROOT_REPO_KEY


def _tmux_id_fragment(project_id: str) -> str:
    frag = re.sub(r"[^A-Za-z0-9]+", "", (project_id or "x"))[:8]
    return frag or "proj"


def _safe_tmux_session_name(project_id: str, slug: str, repo: str) -> str:
    tid = _tmux_id_fragment(project_id)
    raw = f"claudeloop-{tid}-{slug}-{repo.replace('/', '-')}"
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "-", raw).strip("-")
    return safe[:90] or "claudeloop-task"


def _safe_interview_session_name(project_id: str, slug: str) -> str:
    tid = _tmux_id_fragment(project_id)
    raw = f"claudeloop-interview-{tid}-{slug}"
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "-", raw).strip("-")
    return safe[:90] or "claudeloop-interview"


def _safe_ask_session_name(project_id: str, slug: str) -> str:
    tid = _tmux_id_fragment(project_id)
    raw = f"claudeloop-ask-{tid}-{slug}"
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "-", raw).strip("-")
    return safe[:90] or "claudeloop-ask"


def _tmux_session_belongs_to_project_fragment(session_name: str, tid: str) -> bool:
    """True if session name was created for this project id fragment (worker / interview / ask)."""
    if not session_name or not tid:
        return False
    if session_name.startswith("claudeloop-interview-"):
        return session_name.startswith(f"claudeloop-interview-{tid}-")
    if session_name.startswith("claudeloop-ask-"):
        return session_name.startswith(f"claudeloop-ask-{tid}-")
    if session_name.startswith("claudeloop-"):
        if session_name.startswith("claudeloop-interview-") or session_name.startswith("claudeloop-ask-"):
            return False
        return session_name.startswith(f"claudeloop-{tid}-")
    return False


def _session_name_from_tmux_target(target: str) -> str:
    """``session:0.0`` → ``session`` (session names we generate never contain ``:``)."""
    t = (target or "").strip()
    if not t:
        return ""
    if ":" in t:
        return t.split(":", 1)[0].strip()
    return t


def _task_meta_tmux_session_names(project_root: Path) -> set[str]:
    """Tmux session base names referenced by tasks under this filesystem project root."""
    out: set[str] = set()
    try:
        root = project_root.resolve()
    except OSError:
        return out
    if not root.is_dir():
        return out
    for meta in list_tasks(root):
        for attr in ("tmux_runner_target", "tmux_evaluator_target", "tmux_interview_target", "tmux_ask_target"):
            n = _session_name_from_tmux_target(getattr(meta, attr, "") or "")
            if n:
                out.add(n)
    return out


def _filter_tmux_sessions_for_project(
    sessions: list[dict[str, str]],
    project_id: str,
    project_root: Path | None,
) -> list[dict[str, str]]:
    """Match tmux sessions by (1) web project id fragment in the session name, or (2) task.json pane targets.

    (2) covers switching the UI project after starting a worker: session names embed the *old* project id,
    but ``tmux_runner_target`` / … under ``project_root`` still point at the live session.
    """
    tid = _tmux_id_fragment(project_id)
    picked: dict[str, dict[str, str]] = {}
    for s in sessions:
        name = str(s.get("name", ""))
        if not name:
            continue
        if tid and _tmux_session_belongs_to_project_fragment(name, tid):
            picked[name] = s
    if project_root is not None:
        for nm in _task_meta_tmux_session_names(project_root):
            for s in sessions:
                if str(s.get("name", "")) == nm:
                    picked[nm] = s
                    break
    return sorted(picked.values(), key=lambda x: str(x.get("name", "")).lower())


def _launch_root_child_dirs(launch_root: Path, *, limit: int = 200) -> list[dict[str, str]]:
    """Immediate non-hidden subdirectories of the server launch directory (for Add-project quick pick)."""
    out: list[dict[str, str]] = []
    try:
        root = launch_root.resolve()
    except OSError:
        return out
    if not root.is_dir():
        return out
    try:
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if len(out) >= limit:
                break
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            if child.name.startswith("."):
                continue
            try:
                out.append({"name": child.name, "path": str(child.resolve())})
            except OSError:
                continue
    except OSError:
        return out
    return out


def _git_run(args: list[str], cwd: Path, timeout: int = 30) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = "\n".join(x for x in (result.stdout.strip(), result.stderr.strip()) if x)
    return result.returncode == 0, output


def _push_worktree_branch(project_root: Path, slug: str, repo: str, project_id: str) -> dict[str, Any]:
    td = task_root(project_root, slug)
    wt = work_path_for_repo_key(td, repo)
    if wt is None or not wt.is_dir():
        return {"ok": False, "error": f"Worktree missing or invalid repo key: {repo!r}"}

    branch = _safe_tmux_session_name(project_id, slug, repo)
    ok, out = _git_run(["rev-parse", "--is-inside-work-tree"], wt)
    if not ok:
        return {"ok": False, "error": out or "Not a git worktree", "branch": branch}
    ok, remotes = _git_run(["remote"], wt)
    if not ok or "origin" not in remotes.splitlines():
        return {"ok": False, "error": "No git remote named origin", "branch": branch}

    ok, current = _git_run(["branch", "--show-current"], wt)
    if not ok:
        return {"ok": False, "error": current, "branch": branch}
    if current.strip() != branch:
        ok, exists = _git_run(["rev-parse", "--verify", "--quiet", branch], wt)
        switch_args = ["switch", branch] if ok else ["switch", "-c", branch]
        ok, switched = _git_run(switch_args, wt)
        if not ok:
            return {"ok": False, "error": switched, "branch": branch}

    ok, status = _git_run(["status", "--porcelain"], wt)
    if not ok:
        return {"ok": False, "error": status, "branch": branch}
    committed = False
    commit_output = "No local changes to commit."
    if status.strip():
        ok, add_out = _git_run(["add", "-A"], wt)
        if not ok:
            return {"ok": False, "error": add_out, "branch": branch}
        message = f"Complete claudeloop task {slug}"
        ok, commit_output = _git_run(["commit", "-m", message], wt, timeout=60)
        if not ok:
            return {"ok": False, "error": commit_output, "branch": branch}
        committed = True

    ok, push_output = _git_run(["push", "-u", "origin", branch], wt, timeout=120)
    if not ok:
        return {"ok": False, "error": push_output, "branch": branch, "committed": committed}
    return {
        "ok": True,
        "branch": branch,
        "worktree": str(wt),
        "committed": committed,
        "commit_output": commit_output,
        "push_output": push_output,
    }


def _push_all_worktree_branches(project_root: Path, slug: str, project_id: str) -> dict[str, Any]:
    td = task_root(project_root, slug)
    repos = list_work_repo_keys(td)
    results = []
    for repo in repos:
        row = _push_worktree_branch(project_root, slug, repo, project_id)
        row["repo"] = repo
        results.append(row)
    return {
        "ok": bool(results) and all(bool(r.get("ok")) for r in results),
        "count": len(results),
        "results": results,
    }


def _task_state_prompt(task_dir: Path, worktree_dir: Path | None = None) -> str:
    plan_path = (task_dir / PLAN).resolve()
    worktree_note = (
        f"\n- Code edits should happen in this worktree: {worktree_dir.resolve()}"
        if worktree_dir
        else ""
    )
    return f"""## Task State Discipline
- The authoritative task directory is: {task_dir.resolve()}
- The authoritative plan/progress file is: {plan_path}
- Treat every instruction that mentions PLAN.md as referring to that exact file.
- Keep a `Progress Log` section in {plan_path}; append concise dated entries there as work progresses.
- Do not create or update scattered task-management files in the repo/worktree, such as TODO.md, NOTES.md, PROGRESS.md, status logs, scratch plans, or duplicate PLAN.md files.
- Only source-code and implementation artifacts belong in the repo/worktree. Task status, notes, decisions, and next steps belong in {plan_path}.{worktree_note}
"""


def _build_interview_prompt(project_root: Path, slug: str) -> str:
    meta = read_meta(project_root, slug)
    if not meta:
        return ""
    td = task_root(project_root, slug)
    skills = ""
    if meta.skills_path:
        sp = Path(meta.skills_path)
        if sp.is_file():
            skills = sp.read_text(encoding="utf-8", errors="replace")[:12000]
    return f"""You are running claudeloop deep-interview for this task.

You are in the task directory:
{td}

General goal:
{meta.general_goal}

Default skills:
---
{skills or "(none)"}
---

Your job:
1. Interview the user interactively in this Claude Code pane.
2. Ask exactly one high-leverage question at a time.
3. Focus on missing scope, constraints, success criteria, tests, non-goals, and repo/worktree details.
4. Keep notes in {td / "INTERVIEW.md"}.
5. When the task is clear enough, write or overwrite exactly these three files:
   - {td / TASK_PROMPT}
   - {td / SUCCESS_CONDITION}
   - {td / PLAN}

File requirements:
- TASK_PROMPT.md: full autonomous worker prompt with context, constraints, and concrete implementation goal.
- SUCCESS_CONDITION.md: concrete success criteria and bash test commands in fenced ```bash blocks.
- PLAN.md: concise implementation checklist plus a `Progress Log` section.
- The generated TASK_PROMPT.md must instruct workers to keep all task status, notes, decisions, and progress updates in {td / PLAN}; do not let them create scattered TODO/NOTES/PROGRESS/status files in the repo.

Important constraints:
- Work only in this task directory while interviewing.
- Do not modify source code repositories during interview.
- Do not start the worker. The web UI will start claudeloop tmux after the three files are ready.
- If information is missing, ask the user one question and wait.

{_task_state_prompt(td)}

Begin by reading the current files, then ask the first question or, if already clear, write the three files."""


def _build_ask_prompt(project_root: Path, slug: str, repo: str = "") -> str:
    meta = read_meta(project_root, slug)
    if not meta:
        return ""
    td = task_root(project_root, slug)
    selected_worktree = work_path_for_repo_key(td, repo) if repo and validate_repo_key(repo) else None
    worktree_lines = []
    for key in list_work_repo_keys(td):
        path = work_path_for_repo_key(td, key)
        if path:
            worktree_lines.append(f"- {key}: {path}")
    selected_note = f"\nSelected repo/worktree: {repo} -> {selected_worktree}" if selected_worktree else ""
    return f"""You are a claudeloop Ask assistant for this task.

You are running in the task directory:
{td}

General goal:
{meta.general_goal}

You may use all task-local context to answer questions. Inspect broadly when needed:
- PLAN.md: current task state, progress, blockers, and next steps
- TASK_PROMPT.md: worker prompt
- SUCCESS_CONDITION.md: evaluator contract
- INTERVIEW.md: deep-interview notes, if any
- runs/: worker controller logs, process metadata, session state, and run outputs
- work/: all task worktrees and all code/files under them
- any other files under this task directory that help explain what happened

Known worktrees:
{chr(10).join(worktree_lines) if worktree_lines else "(none yet)"}{selected_note}

Behavior:
- Answer the user's questions about what changed, what is running, and what the worker/evaluator did.
- Prefer reading PLAN.md, runs/, work/, and relevant source files before answering.
- You can inspect the full worktree/code contents to explain changes, current behavior, errors, and next steps.
- Do not modify source code, task files, tmux sessions, or git state unless the user explicitly asks.
- If the user asks for a task-state update, write it only to PLAN.md in this task directory.
- Keep answers concise and grounded in files/logs you inspected.
"""


def _json_bytes(obj: Any, status: int = 200) -> tuple[int, bytes, list[tuple[str, str]]]:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    return status, body, headers


def _text_bytes(
    text: str | bytes,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> tuple[int, bytes, list[tuple[str, str]]]:
    body = text if isinstance(text, bytes) else text.encode("utf-8")
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
    ]
    return status, body, headers


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    n = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(n) if n > 0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


def _safe_static_path(static_root: Path, url_path: str) -> Path | None:
    if not url_path.startswith("/static/"):
        return None
    rel = unquote(url_path[len("/static/") :])
    if not rel or ".." in rel.split("/"):
        return None
    candidate = (static_root / rel).resolve()
    try:
        candidate.relative_to(static_root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


class RunRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, subprocess.Popen[Any]] = {}

    def _key(self, project_id: str, slug: str, repo: str) -> str:
        return f"{project_id}::{slug}::{repo}"

    def start(
        self,
        project_root: Path,
        project_id: str,
        slug: str,
        repo: str,
        *,
        mode: str,
        max_iters: int,
        model: str,
    ) -> tuple[bool, str]:
        key = self._key(project_id, slug, repo)
        with self._lock:
            if key in self._runs:
                p = self._runs[key]
                if p.poll() is None:
                    return False, "Worker already running for this repo"
                del self._runs[key]

        td = task_root(project_root, slug)
        wt_path = work_path_for_repo_key(td, repo)
        if wt_path is None or not wt_path.is_dir():
            return False, f"Worktree missing or invalid repo key: {repo!r}"
        wt = wt_path
        for name in (TASK_PROMPT, SUCCESS_CONDITION, PLAN):
            if not (td / name).is_file():
                return False, f"Missing template {name}"

        runs = runs_dir_for_repo(td, repo)
        if runs is None:
            return False, "invalid repo key"
        log_dir = runs / "agent_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = runs / "worker.log"
        meta = read_meta(project_root, slug)
        extra_parts = [_task_state_prompt(td, wt)]
        if meta and meta.skills_path:
            sp = Path(meta.skills_path)
            if sp.is_file():
                extra_parts.append(
                    "## Default skills\n"
                    + sp.read_text(encoding="utf-8", errors="replace")[:12000]
                )
        extra = "\n\n".join(part.strip() for part in extra_parts if part.strip())

        is_tmux = mode == "tmux"
        session_name = _safe_tmux_session_name(project_id, slug, repo) if is_tmux else ""
        cmd: list[str] = [
            sys.executable,
            "-m",
            "claudeloop",
            "tmux" if is_tmux else "run",
            "--prompt",
            str((td / TASK_PROMPT).resolve()),
            "--success",
            str((td / SUCCESS_CONDITION).resolve()),
            "--plan",
            str((td / PLAN).resolve()),
            "--log-dir",
            str(log_dir.resolve()),
            "--max-rounds" if is_tmux else "--max-iters",
            str(max_iters),
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--effort",
            "max",
            "--no-commit",
        ]
        if is_tmux:
            cmd += ["--session-name", session_name]
        if extra:
            cmd += ["--additional-prompt", extra]

        f = open(log_path, "ab", buffering=0)  # noqa: SIM115
        env = tmux_subprocess_env()
        env.update(
            {
                "COLUMNS": "240",
                "LINES": "64",
                "RICH_WIDTH": "240",
            }
        )
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(wt),
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except OSError as e:
            f.close()
            return False, str(e)

        with self._lock:
            self._runs[key] = proc
        meta_update: dict[str, str] = {}
        if is_tmux and session_name:
            meta_update = {
                "tmux_runner_target": f"{session_name}:0.0",
                "tmux_evaluator_target": f"{session_name}:0.1",
            }
            update_meta(project_root, slug, **meta_update)
        (runs / "process.json").write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "mode": mode,
                    "repo": repo,
                    "cwd": str(wt),
                    "cmd": cmd,
                    **meta_update,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        threading.Thread(target=lambda: self._wait_and_close(key, proc, f), daemon=True).start()
        return True, str(log_path)

    def stop(self, project_root: Path, project_id: str, slug: str, repo: str) -> dict[str, Any]:
        key = self._key(project_id, slug, repo)
        stopped_proc = False
        proc_msg = "not running in registry"
        with self._lock:
            proc = self._runs.pop(key, None)
        if proc is not None:
            stopped_proc, proc_msg = self._stop_process(proc)

        if proc is None:
            from_file, from_file_msg = self._stop_process_from_file(project_root, slug, repo)
            stopped_proc = stopped_proc or from_file
            proc_msg = from_file_msg

        session_name = _safe_tmux_session_name(project_id, slug, repo)
        stopped_tmux, tmux_msg = self._kill_tmux_session(session_name)
        update_meta(project_root, slug, tmux_runner_target="", tmux_evaluator_target="")
        return {
            "ok": True,
            "process_stopped": stopped_proc,
            "process_message": proc_msg,
            "tmux_stopped": stopped_tmux,
            "tmux_message": tmux_msg,
            "tmux_session": session_name,
        }

    def _stop_process(self, proc: subprocess.Popen[Any]) -> tuple[bool, str]:
        if proc.poll() is not None:
            return False, f"already exited with {proc.returncode}"
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False, "process already gone"
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=8)
            return True, "terminated"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                proc.kill()
            return True, "killed"

    def _stop_process_from_file(self, project_root: Path, slug: str, repo: str) -> tuple[bool, str]:
        td = task_root(project_root, slug)
        rd = runs_dir_for_repo(td, repo)
        proc_file = (rd / "process.json") if rd else None
        if proc_file is None or not proc_file.is_file():
            return False, "no process file"
        try:
            data = json.loads(proc_file.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False, "invalid process file"
        if pid <= 1:
            return False, "invalid pid"
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            return False, "process already gone"
        ok, reason = self._process_file_matches_task(data, pid, cmdline, td, rd)
        if not ok:
            return False, reason
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False, "process already gone"
        except OSError as exc:
            return False, str(exc)
        return True, "terminated from process file"

    def _process_file_matches_task(
        self,
        data: dict[str, Any],
        pid: int,
        cmdline: str,
        task_dir: Path,
        runs_dir: Path | None,
    ) -> tuple[bool, str]:
        if pid == os.getpid():
            return False, "process file points at web server; skipped"
        if "claudeloop tmux" not in cmdline and "claudeloop run" not in cmdline:
            return False, "pid is not a claudeloop worker; skipped"
        expected_cmd = data.get("cmd", [])
        if not isinstance(expected_cmd, list):
            return False, "invalid process command"
        expected_prompt = str((task_dir / TASK_PROMPT).resolve())
        expected_log_dir = str((runs_dir / "agent_logs").resolve()) if runs_dir else ""
        if expected_prompt not in cmdline:
            return False, "pid belongs to different task; skipped"
        if expected_log_dir and expected_log_dir not in cmdline:
            return False, "pid belongs to different run dir; skipped"
        return True, "matched"

    def _kill_tmux_session(self, session_name: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                text=True,
                env=tmux_subprocess_env(),
                timeout=8,
            )
        except FileNotFoundError:
            return False, "tmux not on PATH"
        except subprocess.TimeoutExpired:
            return False, "tmux kill timed out"
        if result.returncode == 0:
            return True, "tmux session killed"
        msg = (result.stderr or result.stdout or "tmux session not found").strip()
        return False, msg

    def _wait_and_close(self, key: str, proc: subprocess.Popen[Any], logf) -> None:
        try:
            proc.wait()
        finally:
            try:
                logf.close()
            except Exception:
                pass
            with self._lock:
                if self._runs.get(key) is proc:
                    del self._runs[key]

    def status(self, project_root: Path, project_id: str, slug: str, repo: str) -> dict[str, Any]:
        key = self._key(project_id, slug, repo)
        with self._lock:
            p = self._runs.get(key)
        rc = p.poll() if p is not None else None
        running = bool(p is not None and rc is None)
        status: dict[str, Any] = {
            "running": running,
            "pid": p.pid if p is not None else None,
            "returncode": rc,
            "session": self._read_session_status(project_root, slug, repo),
        }
        if p is None:
            status.update(self._read_process_status(project_root, slug, repo))
        return status

    def _read_session_status(self, project_root: Path, slug: str, repo: str) -> dict[str, Any]:
        rd = runs_dir_for_repo(task_root(project_root, slug), repo)
        session_path = rd / "agent_logs" / "session.json" if rd else None
        if session_path is None or not session_path.is_file():
            return {}
        try:
            data = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        completed = int(data.get("completed_iteration", 0) or 0)
        max_rounds = int(data.get("max_rounds", 0) or 0)
        if not max_rounds and rd:
            max_rounds = self._read_max_rounds_from_process_file(rd)
        state = str(data.get("status", "unknown"))
        current = int(data.get("current_round", 0) or 0)
        if not current:
            current = min(completed + 1, max_rounds) if state == "running" and max_rounds else completed
        return {
            "status": state,
            "completed_iteration": completed,
            "current_round": current,
            "max_rounds": max_rounds,
            "updated_at": data.get("updated_at", ""),
            "tmux_session_name": data.get("tmux_session_name", ""),
            "tmux_evaluator_target": data.get("tmux_evaluator_target", ""),
        }

    def _read_max_rounds_from_process_file(self, runs_dir: Path) -> int:
        proc_file = runs_dir / "process.json"
        if not proc_file.is_file():
            return 0
        try:
            data = json.loads(proc_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        cmd = data.get("cmd", [])
        if not isinstance(cmd, list):
            return 0
        for flag in ("--max-rounds", "--max-iters"):
            if flag in cmd:
                idx = cmd.index(flag)
                try:
                    return int(cmd[idx + 1])
                except (IndexError, TypeError, ValueError):
                    return 0
        return 0

    def _read_process_status(self, project_root: Path, slug: str, repo: str) -> dict[str, Any]:
        rd = runs_dir_for_repo(task_root(project_root, slug), repo)
        proc_file = rd / "process.json" if rd else None
        if proc_file is None or not proc_file.is_file():
            return {}
        try:
            data = json.loads(proc_file.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0) or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}
        if pid <= 1:
            return {}
        proc_path = Path(f"/proc/{pid}")
        if not proc_path.exists():
            return {"pid": pid, "running": False, "returncode": None}
        try:
            cmdline = (proc_path / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace",
            )
        except OSError:
            return {"pid": pid, "running": False, "returncode": None}
        ok, _ = self._process_file_matches_task(data, pid, cmdline, task_root(project_root, slug), rd)
        if not ok:
            return {"pid": pid, "running": False, "returncode": None}
        return {"pid": pid, "running": True, "returncode": None}

    def read_log_tail(self, project_root: Path, slug: str, repo: str, lines: int = 80) -> str:
        td = task_root(project_root, slug)
        rd = runs_dir_for_repo(td, repo)
        path = (rd / "worker.log") if rd else None
        if path is None or not path.is_file():
            return ""
        try:
            data = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(data.splitlines()[-lines:])


class InterviewRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, subprocess.Popen[Any]] = {}

    @staticmethod
    def _registry_key(project_id: str, slug: str) -> str:
        return f"{project_id}::{slug}"

    def start(self, project_root: Path, project_id: str, slug: str) -> dict[str, Any]:
        meta = read_meta(project_root, slug)
        if not meta:
            return {"ok": False, "error": "Task not found"}
        td = task_root(project_root, slug)
        if not td.is_dir():
            return {"ok": False, "error": "Task directory missing"}

        session_name = _safe_interview_session_name(project_id, slug)
        target = f"{session_name}:0.0"
        if self._tmux_session_exists(session_name):
            update_meta(project_root, slug, tmux_interview_target=target)
            return {"ok": True, "target": target, "session": session_name, "already_running": True}

        cmd = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "-x",
            "240",
            "-y",
            "64",
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(td),
                capture_output=True,
                text=True,
                env=tmux_subprocess_env(),
                check=True,
                timeout=8,
            )
        except FileNotFoundError:
            return {"ok": False, "error": "tmux not on PATH"}
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "error": str(e)}

        claude_cmd = [
            "claude",
            "--model",
            meta.interview_model or "claude-sonnet-4-6",
            "--dangerously-skip-permissions",
            "--effort",
            "max",
        ]
        try:
            proc = subprocess.Popen(
                ["tmux", "send-keys", "-t", target, shlex.join(claude_cmd), "Enter"],
                cwd=str(td),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=tmux_subprocess_env(),
                start_new_session=True,
            )
        except OSError as e:
            return {"ok": False, "error": str(e)}
        with self._lock:
            self._runs[self._registry_key(project_id, slug)] = proc

        update_meta(project_root, slug, tmux_interview_target=target)
        threading.Thread(
            target=self._paste_prompt_after_startup,
            args=(project_root, slug, target),
            daemon=True,
        ).start()
        return {
            "ok": True,
            "target": target,
            "session": session_name,
            "already_running": False,
            "prompt_pending": True,
        }

    def stop(self, project_root: Path, project_id: str, slug: str) -> dict[str, Any]:
        session_name = _safe_interview_session_name(project_id, slug)
        stopped, msg = self._kill_tmux_session(session_name)
        update_meta(project_root, slug, tmux_interview_target="")
        with self._lock:
            self._runs.pop(self._registry_key(project_id, slug), None)
        return {
            "ok": True,
            "tmux_stopped": stopped,
            "tmux_message": msg,
            "tmux_session": session_name,
        }

    def _tmux_session_exists(self, session_name: str) -> bool:
        try:
            r = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True,
                text=True,
                env=tmux_subprocess_env(),
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0

    def _kill_tmux_session(self, session_name: str) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                text=True,
                env=tmux_subprocess_env(),
                timeout=8,
            )
        except FileNotFoundError:
            return False, "tmux not on PATH"
        except subprocess.TimeoutExpired:
            return False, "tmux kill timed out"
        if r.returncode == 0:
            return True, "tmux session killed"
        return False, (r.stderr or r.stdout or "tmux session not found").strip()

    def _wait_for_claude_ready(self, target: str, timeout: float = 45.0) -> None:
        import time

        deadline = time.time() + timeout
        markers = ("❯", "╭", "tips:", "/help")
        while time.time() < deadline:
            ok, text = capture_pane(target, 80)
            if ok and any(m in text.lower() for m in markers):
                time.sleep(2)
                return
            time.sleep(2)

    def _paste_prompt_after_startup(self, project_root: Path, slug: str, target: str) -> None:
        import time

        time.sleep(5)
        self._wait_for_claude_ready(target, timeout=90.0)
        prompt = _build_interview_prompt(project_root, slug)
        if not prompt:
            return
        ok, _ = send_pane_text(target, prompt, submit=False)
        if not ok:
            return
        # Claude Code often needs an empty-line submit after bracketed paste.
        time.sleep(0.3)
        send_pane_key(target, "Enter")
        time.sleep(0.1)
        send_pane_key(target, "Enter")


class AskRegistry(InterviewRegistry):
    def start(self, project_root: Path, project_id: str, slug: str, repo: str = "") -> dict[str, Any]:
        meta = read_meta(project_root, slug)
        if not meta:
            return {"ok": False, "error": "Task not found"}
        repo = repo or _WORKER_REPO_DEFAULT
        if repo and not validate_repo_key(repo):
            return {"ok": False, "error": "invalid repo key"}
        td = task_root(project_root, slug)
        if not td.is_dir():
            return {"ok": False, "error": "Task directory missing"}
        ask_cwd = work_path_for_repo_key(td, repo) or td
        ask_cwd.mkdir(parents=True, exist_ok=True)

        session_name = _safe_ask_session_name(project_id, slug)
        target = f"{session_name}:0.0"
        if self._tmux_session_exists(session_name):
            update_meta(project_root, slug, tmux_ask_target=target)
            return {"ok": True, "target": target, "session": session_name, "already_running": True}

        cmd = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "-x",
            "240",
            "-y",
            "64",
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(ask_cwd),
                capture_output=True,
                text=True,
                env=tmux_subprocess_env(),
                check=True,
                timeout=8,
            )
        except FileNotFoundError:
            return {"ok": False, "error": "tmux not on PATH"}
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "error": str(e)}

        claude_cmd = [
            "claude",
            "--model",
            "claude-opus-4-7",
            "--dangerously-skip-permissions",
            "--effort",
            "max",
        ]
        try:
            proc = subprocess.Popen(
                ["tmux", "send-keys", "-t", target, shlex.join(claude_cmd), "Enter"],
                cwd=str(ask_cwd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=tmux_subprocess_env(),
                start_new_session=True,
            )
        except OSError as e:
            return {"ok": False, "error": str(e)}
        with self._lock:
            self._runs[self._registry_key(project_id, slug)] = proc

        update_meta(project_root, slug, tmux_ask_target=target)
        threading.Thread(
            target=self._paste_ask_prompt_after_startup,
            args=(project_root, slug, repo, target),
            daemon=True,
        ).start()
        return {
            "ok": True,
            "target": target,
            "session": session_name,
            "already_running": False,
            "prompt_pending": True,
        }

    def stop(self, project_root: Path, project_id: str, slug: str) -> dict[str, Any]:
        session_name = _safe_ask_session_name(project_id, slug)
        stopped, msg = self._kill_tmux_session(session_name)
        update_meta(project_root, slug, tmux_ask_target="")
        with self._lock:
            self._runs.pop(self._registry_key(project_id, slug), None)
        return {
            "ok": True,
            "tmux_stopped": stopped,
            "tmux_message": msg,
            "tmux_session": session_name,
        }

    def _paste_ask_prompt_after_startup(self, project_root: Path, slug: str, repo: str, target: str) -> None:
        import time

        time.sleep(5)
        self._wait_for_claude_ready(target, timeout=90.0)
        prompt = _build_ask_prompt(project_root, slug, repo)
        if not prompt:
            return
        ok, _ = send_pane_text(target, prompt, submit=False)
        if not ok:
            return
        time.sleep(0.3)
        send_pane_key(target, "Enter")
        time.sleep(0.1)
        send_pane_key(target, "Enter")


def make_handler(
    project_registry: WebProjectRegistry,
    launch_root: Path,
    default_skills: Path,
    default_work_dirs: list[Path],
    registry: RunRegistry,
    interview_registry: InterviewRegistry,
    ask_registry: AskRegistry,
    interview_backend_default: str,
    openclaw_client: OpenClawClient,
    auth_token: str = "",
    *,
    multi_project_workspace: bool = False,
) -> type[BaseHTTPRequestHandler]:
    static_root = web_static_dir().resolve()
    required_token = auth_token.strip()
    pr = project_registry
    launch_root_resolved = launch_root.resolve()
    multi_ws = multi_project_workspace

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} - {fmt % args}", flush=True)

        def _send(self, status: int, body: bytes, headers: list[tuple[str, str]]) -> None:
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _resolve_scope(self, parsed) -> tuple[Path | None, str | None]:
            qs = parse_qs(parsed.query or "")
            qpid = (qs.get("project") or [""])[0].strip()
            hp = (self.headers.get("X-ClaudeLoop-Project") or "").strip()
            pid = qpid or hp or pr.default_project_id
            if not pid:
                return None, None
            pth = pr.get_path(pid)
            if pth is None:
                return None, None
            return pth, pid

        def _work_dirs_for(self, root: Path, project_id: str) -> list[Path]:
            custom = pr.get_default_work_dirs(project_id)
            if custom:
                return custom
            if default_work_dirs:
                return list(default_work_dirs)
            return [root]

        def _effective_work_dirs_for_task(self, root: Path, project_id: str, meta: Any) -> list[Path]:
            if meta.work_dirs:
                out: list[Path] = []
                for item in meta.work_dirs:
                    try:
                        out.append(Path(str(item)).expanduser().resolve())
                    except OSError:
                        continue
                if out:
                    return out
            return self._work_dirs_for(root, project_id)

        def _worktree_candidates(self, work_dirs: list[Path]) -> dict[str, Any]:
            groups: list[dict[str, Any]] = []
            for raw in work_dirs:
                try:
                    work_dir = raw.expanduser().resolve()
                except OSError:
                    groups.append({"workDir": str(raw), "kind": "invalid", "repos": [], "reason": "invalid path"})
                    continue
                top = git_toplevel(work_dir)
                if top:
                    groups.append(
                        {
                            "workDir": str(work_dir),
                            "kind": "repo",
                            "repos": [{"name": top.name, "repoKey": top.name, "path": str(top.resolve())}],
                        }
                    )
                    continue
                repos = [
                    {"name": p.name, "repoKey": p.name, "path": str(p.resolve())}
                    for p in direct_child_git_repos(work_dir)
                ]
                groups.append(
                    {
                        "workDir": str(work_dir),
                        "kind": "container",
                        "repos": repos,
                        "reason": "" if repos else "no direct child git repositories",
                    }
                )
            needs_selection = any(g.get("kind") == "container" and len(g.get("repos") or []) > 1 for g in groups)
            auto_work_dirs: list[str] = []
            for g in groups:
                repos = g.get("repos") or []
                if g.get("kind") == "repo" and repos:
                    auto_work_dirs.append(str(repos[0]["path"]))
                elif g.get("kind") == "container" and len(repos) == 1:
                    auto_work_dirs.append(str(repos[0]["path"]))
            return {"groups": groups, "needsSelection": needs_selection, "autoWorkDirs": auto_work_dirs}

        def _bad_project(self) -> None:
            st, b, h = _json_bytes(
                {"error": "unknown or invalid project; pass ?project=<id> or header X-ClaudeLoop-Project"},
                400,
            )
            self._send(st, b, h)

        def _is_authorized(self) -> bool:
            if not required_token:
                return True
            raw = self.headers.get("Authorization", "").strip()
            if raw.lower().startswith("bearer "):
                token = raw[7:].strip()
                return hmac.compare_digest(token, required_token)
            if raw.lower().startswith("basic "):
                encoded = raw[6:].strip()
                try:
                    decoded = base64.b64decode(encoded).decode("utf-8")
                except (binascii.Error, ValueError, UnicodeDecodeError):
                    return False
                _, _, password = decoded.partition(":")
                return hmac.compare_digest(password, required_token)
            return False

        def _require_auth(self) -> bool:
            if self._is_authorized():
                return True
            body = b"authentication required\n"
            self.send_response(401)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("WWW-Authenticate", 'Basic realm="claudeloop"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                idx = static_root / "index.html"
                if not idx.is_file():
                    st, b, h = _text_bytes("missing index.html", 500)
                    self._send(st, b, h)
                    return
                st, b, h = _text_bytes(idx.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
                self._send(st, b, h)
                return

            if path.startswith("/static/"):
                sp = _safe_static_path(static_root, path)
                if sp is None:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                mime = _STATIC_MIME.get(sp.suffix) or mimetypes.guess_type(str(sp))[0] or "application/octet-stream"
                st, b, h = _text_bytes(sp.read_bytes(), content_type=mime)
                self._send(st, b, h)
                return

            if path == "/api/project":
                root, pid = self._resolve_scope(parsed)
                if root is None or pid is None:
                    self._bad_project()
                    return
                sk = default_skills.resolve()
                st, b, h = _json_bytes(
                    {
                        "projectRoot": str(root),
                        "projectId": pid,
                        "skillsPath": str(sk),
                        "skillsBundledRelative": "claudeloop/skills/AK_skills.md",
                    }
                )
                self._send(st, b, h)
                return

            if path == "/api/projects":
                if multi_ws:
                    pr.prune_redundant_parent_projects(launch_root_resolved)
                cur_id = (parse_qs(parsed.query or "").get("project") or [""])[0].strip()
                hdr = (self.headers.get("X-ClaudeLoop-Project") or "").strip()
                resolved = cur_id or hdr or pr.default_project_id
                cur_path = pr.get_path(resolved) if resolved else None
                current = resolved if (resolved and cur_path) else ""
                st, b, h = _json_bytes(
                    {
                        "projects": pr.list_projects(),
                        "defaultProjectId": pr.default_project_id,
                        "currentProjectId": current,
                        "launchRoot": str(launch_root_resolved),
                        "launchRootChildren": _launch_root_child_dirs(launch_root_resolved),
                        "multiProjectWorkspace": multi_ws,
                    }
                )
                self._send(st, b, h)
                return

            if path == "/api/tmux/sessions":
                qs = parse_qs(parsed.query or "")
                proj = (qs.get("project") or [""])[0].strip()
                all_sessions = list_tmux_sessions()
                if proj:
                    p_root = pr.get_path(proj)
                    sessions = _filter_tmux_sessions_for_project(all_sessions, proj, p_root)
                else:
                    sessions = all_sessions
                st, b, h = _json_bytes({"tmux": tmux_available(), "sessions": sessions})
                self._send(st, b, h)
                return

            if path == "/api/tmux/panes":
                qs = parse_qs(parsed.query or "")
                sess = (qs.get("session") or [""])[0].strip()
                if not sess:
                    st, b, h = _json_bytes({"error": "session required"}, 400)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes({"panes": list_tmux_panes(sess)})
                self._send(st, b, h)
                return

            if path == "/api/tmux/capture":
                qs = parse_qs(parsed.query or "")
                target = (qs.get("target") or [""])[0].strip()
                lines = int((qs.get("lines") or ["80"])[0] or 80)
                if not validate_tmux_target(target):
                    st, b, h = _json_bytes({"ok": False, "error": "invalid target", "text": ""}, 400)
                    self._send(st, b, h)
                    return
                ok, text = capture_pane(target, lines)
                st, b, h = _json_bytes({"ok": ok, "text": text if ok else "", "error": "" if ok else text})
                self._send(st, b, h)
                return

            if path == "/api/tasks":
                root, pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                st, b, h = _json_bytes({"tasks": [m.to_dict() for m in list_tasks(root)]})
                self._send(st, b, h)
                return

            m_candidates = re.match(r"^/api/tasks/([^/]+)/worktree-candidates$", path)
            if m_candidates:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_candidates.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                candidates = self._worktree_candidates(self._effective_work_dirs_for_task(root, project_id, meta))
                st, b, h = _json_bytes(candidates)
                self._send(st, b, h)
                return

            m = re.match(r"^/api/tasks/([^/]+)$", path)
            if m:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                templates = {
                    TASK_PROMPT: read_template(root, slug, TASK_PROMPT) or "",
                    SUCCESS_CONDITION: read_template(root, slug, SUCCESS_CONDITION) or "",
                    PLAN: read_template(root, slug, PLAN) or "",
                }
                td_get = task_root(root, slug)
                st, b, h = _json_bytes(
                    {
                        "meta": meta.to_dict(),
                        "templates": templates,
                        "interview": read_interview(root, slug),
                        "work_repos": list_work_repo_keys(td_get),
                    }
                )
                self._send(st, b, h)
                return

            m2 = re.match(r"^/api/tasks/([^/]+)/worker/log$", path)
            if m2:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m2.group(1)
                qs = parse_qs(parsed.query or "")
                repo = (qs.get("repo") or [_WORKER_REPO_DEFAULT])[0] or _WORKER_REPO_DEFAULT
                if not validate_repo_key(repo) or not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "bad request"}, 400)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes(
                    {
                        "tail": registry.read_log_tail(root, slug, repo),
                        "status": registry.status(root, project_id, slug, repo),
                    }
                )
                self._send(st, b, h)
                return

            st, b, h = _json_bytes({"error": "not found"}, 404)
            self._send(st, b, h)

        def do_POST(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path
            body = _read_json(self)

            if path == "/api/tasks/reorder":
                root, _project_id = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                raw_slugs = body.get("slugs", [])
                if not isinstance(raw_slugs, list):
                    st, b, h = _json_bytes({"error": "slugs must be a list"}, 400)
                    self._send(st, b, h)
                    return
                ok_order, err_order = reorder_tasks(root, [str(x) for x in raw_slugs])
                if not ok_order:
                    st, b, h = _json_bytes({"error": err_order}, 400)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes({"ok": True, "tasks": [m.to_dict() for m in list_tasks(root)]})
                self._send(st, b, h)
                return

            if path == "/api/tasks":
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                title = str(body.get("title", "")).strip()
                general_goal = str(body.get("general_goal", "")).strip()
                if not title or not general_goal:
                    st, b, h = _json_bytes({"error": "title and general_goal required"}, 400)
                    self._send(st, b, h)
                    return
                skills_path = bundled_skills_path().resolve()
                raw_sp = body.get("skills_path")
                if raw_sp and str(raw_sp).strip():
                    cand = Path(str(raw_sp)).expanduser().resolve()
                    if cand.is_file():
                        skills_path = cand
                raw_ib = str(body.get("interview_backend", interview_backend_default)).lower()
                task_ib = raw_ib if raw_ib in ("cli", "sdk") else interview_backend_default
                wd = self._work_dirs_for(root, project_id)
                meta = create_task(
                    root,
                    title,
                    general_goal,
                    skills_path=skills_path,
                    interview_model=str(body.get("interview_model", "claude-sonnet-4-6")),
                    interview_backend=task_ib,
                    work_dirs=wd,
                )
                print(
                    f"[web] created task slug={meta.slug} dir={task_root(root, meta.slug)}",
                    flush=True,
                )
                openclaw_client.emit(
                    "task-created",
                    instruction=f"claudeloop task created: {meta.slug}",
                    project_root=root,
                    task_slug=meta.slug,
                    data={
                        "title": meta.title,
                        "taskDir": str(task_root(root, meta.slug)),
                        "projectId": project_id,
                    },
                )
                st, b, h = _json_bytes({"meta": meta.to_dict()}, 201)
                self._send(st, b, h)
                return

            if path == "/api/tmux/send-text":
                target = str(body.get("target", "")).strip()
                text = body.get("text", "")
                submit = bool(body.get("submit", False))
                if not isinstance(text, str):
                    st, b, h = _json_bytes({"ok": False, "error": "text must be string"}, 400)
                    self._send(st, b, h)
                    return
                ok, msg = send_pane_text(target, text, submit=submit)
                st, b, h = (
                    _json_bytes({"ok": True})
                    if ok
                    else _json_bytes({"ok": False, "error": msg}, 400)
                )
                self._send(st, b, h)
                return

            if path == "/api/tmux/send-key":
                target = str(body.get("target", "")).strip()
                key = str(body.get("key", "")).strip()
                ok, msg = send_pane_key(target, key)
                st, b, h = (
                    _json_bytes({"ok": True})
                    if ok
                    else _json_bytes({"ok": False, "error": msg}, 400)
                )
                self._send(st, b, h)
                return

            if path == "/api/projects/reorder":
                raw_ids = body.get("ids", [])
                if not isinstance(raw_ids, list):
                    st, b, h = _json_bytes({"error": "ids must be a list"}, 400)
                    self._send(st, b, h)
                    return
                ok_order, err_order = pr.reorder([str(x) for x in raw_ids])
                if not ok_order:
                    st, b, h = _json_bytes({"error": err_order}, 400)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes(
                    {
                        "ok": True,
                        "projects": pr.list_projects(),
                        "defaultProjectId": pr.default_project_id,
                    }
                )
                self._send(st, b, h)
                return

            if path == "/api/projects":
                raw_path = str(body.get("path", "")).strip()
                if not raw_path:
                    st, b, h = _json_bytes({"error": "path required"}, 400)
                    self._send(st, b, h)
                    return
                new_id, err = pr.add_by_path(raw_path)
                if err or not new_id:
                    st, b, h = _json_bytes({"error": err or "failed"}, 400)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes(
                    {"id": new_id, "defaultProjectId": pr.default_project_id, "projects": pr.list_projects()},
                    201,
                )
                self._send(st, b, h)
                return

            m_move = re.match(r"^/api/projects/([^/]+)/move$", path)
            if m_move:
                pid_move = m_move.group(1)
                direction = str(body.get("direction", "")).strip().lower()
                ok_move, err_move = pr.move(pid_move, direction)
                if not ok_move:
                    status = 404 if err_move == "project not found" else 400
                    st, b, h = _json_bytes({"error": err_move}, status)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes(
                    {
                        "ok": True,
                        "projects": pr.list_projects(),
                        "defaultProjectId": pr.default_project_id,
                    }
                )
                self._send(st, b, h)
                return

            m_activate = re.match(r"^/api/projects/([^/]+)/activate$", path)
            if m_activate:
                pid_act = m_activate.group(1)
                if not pr.set_default(pid_act):
                    st, b, h = _json_bytes({"error": "project not found"}, 404)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes({"ok": True, "defaultProjectId": pid_act})
                self._send(st, b, h)
                return

            m_start_interview = re.match(r"^/api/tasks/([^/]+)/interview/start$", path)
            if m_start_interview:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_start_interview.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                result = interview_registry.start(root, project_id, slug)
                print(
                    f"[web] start interview slug={slug} ok={bool(result.get('ok'))} "
                    f"session={result.get('session', '')} target={result.get('target', '')}",
                    flush=True,
                )
                openclaw_client.emit(
                    "interview-start",
                    instruction=f"claudeloop deep-interview started for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data=result,
                )
                st, b, h = (
                    _json_bytes(result)
                    if result.get("ok")
                    else _json_bytes(result, 400)
                )
                self._send(st, b, h)
                return

            m_stop_interview = re.match(r"^/api/tasks/([^/]+)/interview/stop$", path)
            if m_stop_interview:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_stop_interview.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                result = interview_registry.stop(root, project_id, slug)
                openclaw_client.emit(
                    "interview-stop",
                    instruction=f"claudeloop deep-interview stopped for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data=result,
                )
                st, b, h = _json_bytes(result)
                self._send(st, b, h)
                return

            m_start_ask = re.match(r"^/api/tasks/([^/]+)/ask/start$", path)
            if m_start_ask:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_start_ask.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                repo = str(body.get("repo", "")).strip()
                result = ask_registry.start(root, project_id, slug, repo=repo)
                print(
                    f"[web] start ask slug={slug} repo={repo} ok={bool(result.get('ok'))} "
                    f"session={result.get('session', '')} target={result.get('target', '')}",
                    flush=True,
                )
                openclaw_client.emit(
                    "ask-start",
                    instruction=f"claudeloop ask pane started for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    repo=repo or None,
                    data=result,
                )
                st, b, h = (
                    _json_bytes(result)
                    if result.get("ok")
                    else _json_bytes(result, 400)
                )
                self._send(st, b, h)
                return

            m_stop_ask = re.match(r"^/api/tasks/([^/]+)/ask/stop$", path)
            if m_stop_ask:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_stop_ask.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                result = ask_registry.stop(root, project_id, slug)
                openclaw_client.emit(
                    "ask-stop",
                    instruction=f"claudeloop ask pane stopped for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data=result,
                )
                st, b, h = _json_bytes(result)
                self._send(st, b, h)
                return

            m2 = re.match(r"^/api/tasks/([^/]+)/worktrees$", path)
            if m2:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m2.group(1)
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                if not meta.work_dirs:
                    source_dirs = self._work_dirs_for(root, project_id)
                    meta = update_meta(root, slug, work_dirs=[str(p) for p in source_dirs])
                    if not meta:
                        st, b, h = _json_bytes({"error": "not found"}, 404)
                        self._send(st, b, h)
                        return
                selected_work_dirs: list[Path] | None = None
                raw_work_dirs = body.get("work_dirs")
                if raw_work_dirs is not None:
                    if not isinstance(raw_work_dirs, list):
                        st, b, h = _json_bytes({"error": "work_dirs must be a list"}, 400)
                        self._send(st, b, h)
                        return
                    candidates = self._worktree_candidates(self._effective_work_dirs_for_task(root, project_id, meta))
                    allowed = {
                        str(Path(str(repo.get("path", ""))).expanduser().resolve())
                        for group in candidates.get("groups", [])
                        for repo in (group.get("repos") or [])
                        if str(repo.get("path", "")).strip()
                    }
                    selected_work_dirs = []
                    for item in raw_work_dirs:
                        try:
                            candidate = Path(str(item)).expanduser().resolve()
                        except OSError:
                            st, b, h = _json_bytes({"error": f"invalid work_dir: {item}"}, 400)
                            self._send(st, b, h)
                            return
                        if str(candidate) not in allowed:
                            st, b, h = _json_bytes({"error": f"work_dir is not an allowed candidate: {candidate}"}, 400)
                            self._send(st, b, h)
                            return
                        if candidate not in selected_work_dirs:
                            selected_work_dirs.append(candidate)
                    if not selected_work_dirs:
                        st, b, h = _json_bytes({"error": "select at least one work_dir"}, 400)
                        self._send(st, b, h)
                        return
                results = (
                    prepare_selected_worktrees(root, slug, selected_work_dirs)
                    if selected_work_dirs is not None
                    else prepare_all_worktrees(root, slug)
                )
                print(
                    f"[web] create worktree slug={slug} "
                    f"ok={sum(1 for r in results if r.get('ok'))}/{len(results)}",
                    flush=True,
                )
                for row in results:
                    if row.get("ok"):
                        continue
                    print(
                        "[web] create worktree failed "
                        f"slug={slug} repo={row.get('repo_key') or row.get('work_dir') or '?'} "
                        f"reason={row.get('reason') or 'unknown'}",
                        flush=True,
                    )
                openclaw_client.emit(
                    "worktree-created",
                    instruction=f"claudeloop worktree created for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data={"results": results},
                )
                st, b, h = _json_bytes({"results": results})
                self._send(st, b, h)
                return

            m3 = re.match(r"^/api/tasks/([^/]+)/worker/start$", path)
            if m3:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m3.group(1)
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                repo = str(body.get("repo", _WORKER_REPO_DEFAULT)).strip() or _WORKER_REPO_DEFAULT
                if not validate_repo_key(repo):
                    st, b, h = _json_bytes({"error": "invalid repo key"}, 400)
                    self._send(st, b, h)
                    return
                mode = "tmux"
                ok, msg = registry.start(
                    root,
                    project_id,
                    slug,
                    repo,
                    mode=mode,
                    max_iters=int(body.get("max_iters", 200) or 200),
                    model=str(body.get("model", "claude-opus-4-6")),
                )
                print(
                    f"[web] start worker slug={slug} repo={repo} ok={ok} detail={msg}",
                    flush=True,
                )
                openclaw_client.emit(
                    "worker-start",
                    instruction=f"claudeloop worker started for task {slug} repo {repo}",
                    project_root=root,
                    task_slug=slug,
                    repo=repo,
                    data={"ok": ok, "detail": msg},
                )
                st, b, h = (
                    _json_bytes({"ok": True, "log_path": msg})
                    if ok
                    else _json_bytes({"ok": False, "error": msg}, 400)
                )
                self._send(st, b, h)
                return

            m4 = re.match(r"^/api/tasks/([^/]+)/worker/stop$", path)
            if m4:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m4.group(1)
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                repo = str(body.get("repo", _WORKER_REPO_DEFAULT)).strip() or _WORKER_REPO_DEFAULT
                if not validate_repo_key(repo):
                    st, b, h = _json_bytes({"error": "invalid repo key"}, 400)
                    self._send(st, b, h)
                    return
                result = registry.stop(root, project_id, slug, repo)
                print(f"[web] stop worker slug={slug} repo={repo}", flush=True)
                openclaw_client.emit(
                    "worker-stop",
                    instruction=f"claudeloop worker stopped for task {slug} repo {repo}",
                    project_root=root,
                    task_slug=slug,
                    repo=repo,
                    data=result,
                )
                st, b, h = _json_bytes(result)
                self._send(st, b, h)
                return

            m_push = re.match(r"^/api/tasks/([^/]+)/worker/push$", path)
            if m_push:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_push.group(1)
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"ok": False, "error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                repo = str(body.get("repo", "")).strip()
                if repo:
                    if not validate_repo_key(repo) or repo == _WORKER_REPO_DEFAULT:
                        st, b, h = _json_bytes({"ok": False, "error": "invalid repo key"}, 400)
                        self._send(st, b, h)
                        return
                    result = _push_worktree_branch(root, slug, repo, project_id)
                    result["repo"] = repo
                else:
                    result = _push_all_worktree_branches(root, slug, project_id)
                print(
                    f"[web] push branch slug={slug} repo={repo or '*'} ok={result.get('ok')} "
                    f"branch={result.get('branch', '')} count={result.get('count', '')}",
                    flush=True,
                )
                openclaw_client.emit(
                    "worker-push",
                    instruction=f"claudeloop pushed task {slug} worktree branches",
                    project_root=root,
                    task_slug=slug,
                    repo=repo or None,
                    data=result,
                )
                st, b, h = _json_bytes(result, 200 if result.get("ok") else 400)
                self._send(st, b, h)
                return

            st, b, h = _json_bytes({"error": "not found"}, 404)
            self._send(st, b, h)

        def do_PUT(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path
            body = _read_json(self)

            m0 = re.match(r"^/api/tasks/([^/]+)/tmux$", path)
            if m0:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m0.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                r_t = str(body.get("tmux_runner_target", "")).strip()
                e_t = str(body.get("tmux_evaluator_target", "")).strip()
                if not validate_tmux_target(r_t) or not validate_tmux_target(e_t):
                    st, b, h = _json_bytes({"error": "invalid pane target"}, 400)
                    self._send(st, b, h)
                    return
                updated = update_meta(
                    root,
                    slug,
                    tmux_runner_target=r_t,
                    tmux_evaluator_target=e_t,
                )
                if not updated:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes({"ok": True, "meta": updated.to_dict()})
                self._send(st, b, h)
                return

            m = re.match(r"^/api/tasks/([^/]+)/template$", path)
            if not m:
                st, b, h = _json_bytes({"error": "not found"}, 404)
                self._send(st, b, h)
                return
            root, _pid = self._resolve_scope(parsed)
            if root is None:
                self._bad_project()
                return
            slug = m.group(1)
            if not _SLUG_RE.match(slug):
                st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                self._send(st, b, h)
                return
            name = str(body.get("name", ""))
            content = body.get("content", "")
            if not isinstance(content, str):
                st, b, h = _json_bytes({"error": "content must be string"}, 400)
                self._send(st, b, h)
                return
            if not write_template(root, slug, name, content):
                st, b, h = _json_bytes({"error": "invalid template"}, 400)
                self._send(st, b, h)
                return
            st, b, h = _json_bytes({"ok": True})
            self._send(st, b, h)

        def do_DELETE(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path

            m_task_del = re.match(r"^/api/tasks/([^/]+)$", path)
            if m_task_del:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m_task_del.group(1)
                ok_task, err_task = delete_task(root, slug)
                if not ok_task:
                    status = 404 if err_task == "task not found" else 400
                    st, b, h = _json_bytes({"error": err_task}, status)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes({"ok": True, "slug": slug})
                self._send(st, b, h)
                return

            m_del = re.match(r"^/api/projects/([^/]+)$", path)
            if not m_del:
                st, b, h = _json_bytes({"error": "not found"}, 404)
                self._send(st, b, h)
                return
            pid_del = m_del.group(1)
            ok_del, err_msg = pr.remove(pid_del)
            if not ok_del:
                st, b, h = _json_bytes({"error": err_msg}, 400)
                self._send(st, b, h)
                return
            st, b, h = _json_bytes(
                {
                    "ok": True,
                    "projects": pr.list_projects(),
                    "defaultProjectId": pr.default_project_id,
                }
            )
            self._send(st, b, h)

    return Handler


def serve(
    host: str,
    port: int,
    project_root: Path,
    default_skills: Path,
    default_work_dirs: list[Path],
    interview_backend_default: str = "cli",
    openclaw_config: OpenClawConfig | None = None,
    auth_token: str = "",
    *,
    multi_project_workspace: bool = False,
) -> None:
    project_root = project_root.resolve()
    os.environ["CLAUDELOOP_PROJECT_ROOT"] = str(project_root)
    web_project_registry = WebProjectRegistry()
    if multi_project_workspace:
        web_project_registry.prune_redundant_parent_projects(project_root)
    registry = RunRegistry()
    interview_registry = InterviewRegistry()
    ask_registry = AskRegistry()
    openclaw_client = OpenClawClient(openclaw_config)
    ib = interview_backend_default if interview_backend_default in ("cli", "sdk") else "cli"
    sk = default_skills if default_skills.is_file() else bundled_skills_path().resolve()
    handler = make_handler(
        web_project_registry,
        project_root,
        sk,
        default_work_dirs,
        registry,
        interview_registry,
        ask_registry,
        ib,
        openclaw_client,
        auth_token,
        multi_project_workspace=multi_project_workspace,
    )
    server = HTTPServer((host, port), handler)
    rud_root = project_root / ".RUD"
    work_sources = ", ".join(str(p) for p in default_work_dirs) if default_work_dirs else str(project_root)
    print("", flush=True)
    print("claudeloop web", flush=True)
    print(f"  URL:              http://{host}:{port}/", flush=True)
    print(
        f"  Server cwd:       {project_root}  (--project / launch directory; not auto-registered)"
        f"{'  [multi-project workspace: --projects]' if multi_project_workspace else ''}",
        flush=True,
    )
    print(f"  Project registry: {web_project_registry.persist_path}", flush=True)
    print(f"  Task root:        {rud_root}", flush=True)
    print(f"  Static assets:    {web_static_dir().resolve()}", flush=True)
    print(f"  Default skills:   {sk}", flush=True)
    print(f"  Interview:        {ib} backend, Claude Code tmux pane on demand", flush=True)
    print(f"  Worktree source:  {work_sources}", flush=True)
    print("  Worker mode:      claudeloop tmux, effort=max, no auto-commit", flush=True)
    print(f"  Auth:             {'enabled' if auth_token.strip() else 'disabled'}", flush=True)
    print(f"  OpenClaw:         {openclaw_status(openclaw_client.config)}", flush=True)
    print("  Stop behavior:    stops controller process and kills the task tmux session", flush=True)
    print("  Logs:             this terminal; worker logs under .RUD/<task>/runs/<repo>/worker.log", flush=True)
    print("", flush=True)
    openclaw_client.emit(
        "web-start",
        instruction=f"claudeloop web started for project {project_root}",
        project_root=project_root,
        data={"url": f"http://{host}:{port}/", "taskRoot": str(rud_root)},
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
