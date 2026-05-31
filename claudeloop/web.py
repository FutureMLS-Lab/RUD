"""Lightweight local web UI for `.RUD` tasks.

Three concerns after the agent-loop rewrite:

1. **Task CRUD** - list / create / delete tasks (``<project>/.RUD/<slug>/``).
   Each new task auto-creates a git worktree at
   ``<task>/work/<repo>`` on branch ``zhongzhu/<slug>`` (best-effort -
   non-git project roots just skip the worktree step).
2. **Project notes** - one ``<project>/.RUD/NOTES.md`` per project,
   served by ``GET/PUT /api/notes``.
3. **Claude pane** - launch a tmux + ``claude`` CLI in the task's
   worktree, automatically capture the Claude Code session UUID from
   ``~/.claude/projects/<encoded>/``, and let the UI resume any
   previously-captured session even after tmux is killed.

The only per-task editable template is ``PLAN.md``.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import mimetypes
import os
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from claudeloop.openclaw import OpenClawClient, OpenClawConfig, openclaw_status
from claudeloop.paths import bundled_skills_path, web_static_dir
from claudeloop.rud_task import (
    AGENT_CLAUDE,
    AGENT_CODEX,
    PLAN,
    SUPPORTED_AGENTS,
    add_claude_session,
    agent_default_model,
    agent_label,
    build_agent_command,
    create_task,
    delete_task,
    detect_and_persist_worktree,
    list_session_files,
    list_task_markdown_files,
    list_task_worktree_statuses,
    list_task_worktrees,
    list_tasks,
    list_worktree_candidates,
    normalize_agent,
    prepare_task_worktree_from,
    push_worktree_branch,
    read_meta,
    read_project_notes,
    read_task_markdown_file,
    read_template,
    remove_task_worktree,
    reorder_tasks,
    rename_task_meta,
    session_id_from_path,
    task_root,
    task_worktree_path,
    update_meta,
    worktree_status,
    write_project_notes,
    write_template,
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
from claudeloop.web_projects import WebProjectRegistry

_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{1,80}$")
_STATIC_MIME: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}


# --- naming / filtering helpers --------------------------------------------


def _tmux_id_fragment(project_id: str) -> str:
    frag = re.sub(r"[^A-Za-z0-9]+", "", (project_id or "x"))[:8]
    return frag or "proj"


def _safe_claude_session_name(project_id: str, slug: str, agent: str = AGENT_CLAUDE) -> str:
    """Tmux session name for a task's agent pane.

    The agent name is part of the prefix so a claude pane and a codex
    pane for the same project never share a tmux session if the user
    ever changes agent.  Backwards-compatible aliases handled in
    ``_filter_tmux_sessions_for_project``.
    """
    tid = _tmux_id_fragment(project_id)
    agent = normalize_agent(agent)
    raw = f"claudeloop-{agent}-{tid}-{slug}"
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "-", raw).strip("-")
    return safe[:90] or f"claudeloop-{agent}"


def _session_name_from_tmux_target(target: str) -> str:
    """``session:0.0`` -> ``session`` (we never put ``:`` in session names)."""
    t = (target or "").strip()
    if ":" in t:
        return t.split(":", 1)[0].strip()
    return t


def _task_meta_tmux_session_names(project_root: Path) -> set[str]:
    out: set[str] = set()
    try:
        root = project_root.resolve()
    except OSError:
        return out
    if not root.is_dir():
        return out
    for meta in list_tasks(root):
        n = _session_name_from_tmux_target(getattr(meta, "tmux_interview_target", "") or "")
        if n:
            out.add(n)
    return out


def _filter_tmux_sessions_for_project(
    sessions: list[dict[str, str]],
    project_id: str,
    project_root: Path | None,
) -> list[dict[str, str]]:
    tid = _tmux_id_fragment(project_id)
    picked: dict[str, dict[str, str]] = {}
    # We accept session-name prefixes for every supported agent plus the
    # legacy "claudeloop-interview-<tid>-..." used before the rename.
    prefixes = tuple(
        f"claudeloop-{name}-{tid}-"
        for name in (*SUPPORTED_AGENTS, "interview")
    )
    for s in sessions:
        name = str(s.get("name", ""))
        if name and tid and any(name.startswith(p) for p in prefixes):
            picked[name] = s
    if project_root is not None:
        for nm in _task_meta_tmux_session_names(project_root):
            for s in sessions:
                if str(s.get("name", "")) == nm:
                    picked[nm] = s
                    break
    return sorted(picked.values(), key=lambda x: str(x.get("name", "")).lower())


def _launch_root_child_dirs(launch_root: Path, *, limit: int = 200) -> list[dict[str, str]]:
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


# --- HTTP response helpers --------------------------------------------------


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


# --- Claude prompt builder --------------------------------------------------


def _build_claude_prompt(project_root: Path, slug: str) -> str:
    meta = read_meta(project_root, slug)
    if not meta:
        return ""
    td = task_root(project_root, slug)
    wt = task_worktree_path(project_root, slug)
    wt_line = f"Worktree (branch {meta.branch or '(unset)'}): {wt}" if wt else "Worktree: (none)"
    skills = ""
    if meta.skills_path:
        sp = Path(meta.skills_path)
        if sp.is_file():
            skills = sp.read_text(encoding="utf-8", errors="replace")[:12000]
    plan_path = td / PLAN
    return f"""You are running claudeloop's {agent_label(meta.agent)} pane for this task.

You are in the task directory:
{td}

General goal:
{meta.general_goal}

{wt_line}

Default skills:
---
{skills or "(none)"}
---

How you should help the user:
1. If {plan_path} is empty or vague, run a short deep-interview - ask one
   high-leverage question at a time about scope, constraints, acceptance,
   tests, non-goals. Capture all the decisions DIRECTLY in {plan_path};
   do not create a separate notes / transcript / INTERVIEW file.
2. Once the goal is clear, write or overwrite {plan_path} with:
   - Goal, Constraints / non-goals, Acceptance, Next steps (checkbox list),
     Progress Log section the user updates as they work.
3. After PLAN.md is solid, the user will typically continue this same
   session and drive the work with ``/goal`` against PLAN.md.

Behavioural constraints:
- PLAN.md is the ONLY task-state file. Do not create INTERVIEW.md,
  TODO.md, PROGRESS.md, NOTES.md, or any other scattered status files in
  the task directory or the repo.
- Project-scoped scratch lives in the project's NOTES.md at .RUD/NOTES.md
  (handled by the user via the web UI), not inside the worktree.

Begin by reading {plan_path}, then either ask the first interview
question or, if PLAN.md is already detailed enough, acknowledge that and
wait for ``/goal``.
"""


# --- Claude tmux registry ---------------------------------------------------


class ClaudeRegistry:
    """Manage tmux + claude CLI panes per (project, task).

    A pane's lifecycle:
    1. ``start`` opens a tmux session, launches ``claude``, sends the
       deep-interview prompt as a bracketed paste, and kicks off a
       background watcher.
    2. The watcher polls ``~/.claude/projects/<encoded-cwd>/`` for a new
       ``<uuid>.jsonl`` and, when one appears, records the UUID via
       ``add_claude_session``.  This is what lets us offer a Resume button.
    3. ``stop`` kills the tmux session but leaves the session UUIDs in
       metadata so they remain resumable from the CLI.
    4. ``resume`` re-launches ``claude --resume <uuid>`` in a fresh tmux
       pane.  Useful when the original tmux was killed but the session
       transcript on disk is still good.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, subprocess.Popen[Any]] = {}

    @staticmethod
    def _registry_key(project_id: str, slug: str) -> str:
        return f"{project_id}::{slug}"

    def start(
        self,
        project_root: Path,
        project_id: str,
        slug: str,
        *,
        resume_session_id: str = "",
    ) -> dict[str, Any]:
        meta = read_meta(project_root, slug)
        if not meta:
            return {"ok": False, "error": "Task not found"}
        td = task_root(project_root, slug)
        if not td.is_dir():
            return {"ok": False, "error": "Task directory missing"}

        # Run the agent inside the worktree when we have one - that's where
        # the user will eventually want /goal (or codex's equivalent) to act.
        worktree = task_worktree_path(project_root, slug)
        cwd = worktree if worktree is not None else td

        agent = normalize_agent(meta.agent)
        session_name = _safe_claude_session_name(project_id, slug, agent)
        target = f"{session_name}:0.0"
        if self._tmux_session_exists(session_name):
            if resume_session_id:
                return {
                    "ok": False,
                    "error": "Stop the running tmux pane before resuming another session.",
                    "target": target,
                    "session": session_name,
                }
            update_meta(project_root, slug, tmux_interview_target=target)
            return {
                "ok": True,
                "target": target,
                "session": session_name,
                "cwd": str(cwd),
                "agent": agent,
                "already_running": True,
            }

        # Snapshot existing session files so the watcher can spot the new one.
        existing_files = {p.name for p in list_session_files(cwd, agent)}

        try:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name, "-x", "240", "-y", "64"],
                cwd=str(cwd),
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

        agent_cmd = build_agent_command(
            agent,
            model=meta.interview_model or agent_default_model(agent),
            resume_session_id=resume_session_id,
        )
        try:
            proc = subprocess.Popen(
                ["tmux", "send-keys", "-t", target, shlex.join(agent_cmd), "Enter"],
                cwd=str(cwd),
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
        if resume_session_id:
            add_claude_session(project_root, slug, resume_session_id)
            threading.Thread(
                target=self._watch_for_session_id,
                args=(project_root, slug, cwd, agent, existing_files),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._paste_prompt_and_watch_session,
                args=(project_root, slug, target, cwd, agent, existing_files),
                daemon=True,
            ).start()
        return {
            "ok": True,
            "target": target,
            "session": session_name,
            "cwd": str(cwd),
            "agent": agent,
            "resumed_session_id": resume_session_id or None,
            "already_running": False,
            "prompt_pending": not bool(resume_session_id),
        }

    def stop(self, project_root: Path, project_id: str, slug: str) -> dict[str, Any]:
        meta = read_meta(project_root, slug)
        agent = normalize_agent(meta.agent) if meta else AGENT_CLAUDE
        session_name = _safe_claude_session_name(project_id, slug, agent)
        stopped, msg = self._kill_tmux_session(session_name)
        # Also clean up the legacy interview session name and the *other*
        # agent's session in case the user flipped agents.
        legacy_aliases = {
            re.sub(r"^claudeloop-[A-Za-z0-9]+-", "claudeloop-interview-", session_name),
            *[
                _safe_claude_session_name(project_id, slug, other)
                for other in SUPPORTED_AGENTS
                if other != agent
            ],
        }
        legacy_aliases.discard(session_name)
        for alias in legacy_aliases:
            self._kill_tmux_session(alias)
        update_meta(project_root, slug, tmux_interview_target="")
        with self._lock:
            self._runs.pop(self._registry_key(project_id, slug), None)
        return {
            "ok": True,
            "tmux_stopped": stopped,
            "tmux_message": msg,
            "tmux_session": session_name,
        }

    # --- helpers ---

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

    def session_status(self, project_id: str, slug: str, agent: str = AGENT_CLAUDE) -> dict[str, Any]:
        session_name = _safe_claude_session_name(project_id, slug, agent)
        return {
            "session": session_name,
            "target": f"{session_name}:0.0",
            "tmux_alive": self._tmux_session_exists(session_name),
            "agent": normalize_agent(agent),
        }

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
        deadline = time.time() + timeout
        markers = ("\u276f", "\u256d", "tips:", "/help")
        while time.time() < deadline:
            ok, text = capture_pane(target, 80)
            if ok and any(m in text.lower() for m in markers):
                time.sleep(2)
                return
            time.sleep(2)

    def _watch_for_session_id(
        self,
        project_root: Path,
        slug: str,
        cwd: Path,
        agent: str,
        existing_filenames: set[str],
    ) -> None:
        """Poll the agent's session dir for a freshly-written session file."""
        deadline = time.time() + 90.0
        while time.time() < deadline:
            for p in list_session_files(cwd, agent):
                if p.name not in existing_filenames:
                    sid = session_id_from_path(p, agent)
                    if sid:
                        add_claude_session(project_root, slug, sid)
                        return
            time.sleep(2)

    def _paste_prompt_and_watch_session(
        self,
        project_root: Path,
        slug: str,
        target: str,
        cwd: Path,
        agent: str,
        existing_filenames: set[str],
    ) -> None:
        time.sleep(5)
        self._wait_for_claude_ready(target, timeout=90.0)
        prompt = _build_claude_prompt(project_root, slug)
        if prompt:
            ok, _ = send_pane_text(target, prompt, submit=False)
            if ok:
                # Both Claude Code and Codex need an Enter to submit after
                # the bracketed paste; the extra Enter is a no-op for both.
                time.sleep(0.3)
                send_pane_key(target, "Enter")
                time.sleep(0.1)
                send_pane_key(target, "Enter")
        self._watch_for_session_id(project_root, slug, cwd, agent, existing_filenames)


# --- HTTP handler factory ---------------------------------------------------


def make_handler(
    project_registry: WebProjectRegistry,
    launch_root: Path,
    default_skills: Path,
    claude_registry: ClaudeRegistry,
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

        def _claude_session_summary(self, project_id: str, slug: str, meta) -> dict[str, Any]:
            agent = normalize_agent(meta.agent)
            cwd_str = (meta.worktree_path or "").strip() or str(task_root(pr.get_path(project_id), slug))
            try:
                cwd = Path(cwd_str)
            except OSError:
                cwd = Path(cwd_str)
            files_by_id: dict[str, dict[str, Any]] = {}
            for p in list_session_files(cwd, agent):
                sid = session_id_from_path(p, agent)
                if not sid:
                    continue
                try:
                    stat = p.stat()
                except OSError:
                    continue
                files_by_id[sid] = {
                    "id": sid,
                    "path": str(p),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
            # Preserve task-meta order (history of who-was-spawned-when)
            # but enrich with on-disk info.
            ordered = []
            seen: set[str] = set()
            for sid in meta.claude_session_ids:
                if sid in files_by_id:
                    ordered.append(files_by_id[sid])
                else:
                    ordered.append({"id": sid, "path": "", "mtime": 0.0, "size": 0})
                seen.add(sid)
            for sid, info in files_by_id.items():
                if sid not in seen:
                    ordered.append(info)
            ordered.sort(key=lambda x: x.get("mtime", 0.0), reverse=True)
            live = claude_registry.session_status(project_id, slug, agent)
            return {
                "agent": agent,
                "agent_label": agent_label(agent),
                "tracked": [sid for sid in meta.claude_session_ids],
                "sessions": ordered,
                "tmux_alive": live["tmux_alive"],
                "tmux_session": live["session"],
                "tmux_target": meta.tmux_interview_target or "",
                "claude_cwd": str(cwd),
            }

        # ===== GET =====

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
                st, b, h = _text_bytes(
                    idx.read_text(encoding="utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                self._send(st, b, h)
                return

            if path.startswith("/static/"):
                sp = _safe_static_path(static_root, path)
                if sp is None:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                mime = (
                    _STATIC_MIME.get(sp.suffix)
                    or mimetypes.guess_type(str(sp))[0]
                    or "application/octet-stream"
                )
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

            if path == "/api/notes":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                st, b, h = _json_bytes({"content": read_project_notes(root)})
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
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                st, b, h = _json_bytes({"tasks": [m.to_dict() for m in list_tasks(root)]})
                self._send(st, b, h)
                return

            m_wt_cand = re.match(r"^/api/tasks/([^/]+)/worktree-candidates$", path)
            if m_wt_cand:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m_wt_cand.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                candidates = list_worktree_candidates(root)
                # Annotate each candidate with the destination path + a
                # flag the UI uses to disable rows that are already wired
                # in.  "Already created" means the dest dir is a registered
                # git worktree (so picking again would be a no-op).
                dest_parent = task_root(root, slug) / "work"
                existing_paths = {str(p) for p in list_task_worktrees(root, slug)}
                for c in candidates:
                    dest = dest_parent / Path(c["path"]).name
                    c["destination"] = str(dest)
                    c["already_created"] = str(dest.resolve()) in existing_paths
                st, b, h = _json_bytes(
                    {
                        "projectRoot": str(root),
                        "candidates": candidates,
                        "worktrees": list(meta.worktrees),
                    }
                )
                self._send(st, b, h)
                return

            m_sessions = re.match(r"^/api/tasks/([^/]+)/claude-sessions$", path)
            if m_sessions:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_sessions.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                # Same back-fill so the live Claude info card always sees
                # the disk truth.
                meta = detect_and_persist_worktree(root, slug) or meta
                st, b, h = _json_bytes(self._claude_session_summary(project_id, slug, meta))
                self._send(st, b, h)
                return

            m = re.match(r"^/api/tasks/([^/]+)$", path)
            if m:
                root, project_id = self._resolve_scope(parsed)
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
                # Back-fill worktree_path / branch on tasks that pre-date the
                # auto-worktree feature, or tasks where the user manually
                # added a worktree under work/ later on.
                meta = detect_and_persist_worktree(root, slug) or meta
                # The expensive bits are git-status-per-worktree and the
                # Claude session enrichment (tmux subprocess + filesystem
                # scan). Fan both out so they overlap with the synchronous
                # markdown reads below - on a typical task this brings the
                # endpoint from ~600-1500ms down to ~150-400ms.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    statuses_fut = pool.submit(list_task_worktree_statuses, root, slug)
                    summary_fut = (
                        pool.submit(self._claude_session_summary, project_id, slug, meta)
                        if project_id
                        else None
                    )
                    # Surface every top-level *.md file in the task directory so
                    # the Claude tab's embedded picker can switch between them.
                    # PLAN.md is always present even if empty so the dedicated
                    # PLAN.md editor has something to render.
                    md_names = list_task_markdown_files(root, slug)
                    templates: dict[str, str] = {}
                    for md_name in md_names:
                        content = read_task_markdown_file(root, slug, md_name)
                        if content is not None:
                            templates[md_name] = content
                    if PLAN not in templates:
                        templates[PLAN] = read_template(root, slug, PLAN) or ""
                        if PLAN not in md_names:
                            md_names = [PLAN, *md_names]
                    statuses = statuses_fut.result()
                    summary = summary_fut.result() if summary_fut is not None else None
                st, b, h = _json_bytes(
                    {
                        "meta": meta.to_dict(),
                        "templates": templates,
                        "task_markdown_files": md_names,
                        "claude": summary or {},
                        "worktree_statuses": statuses,
                    }
                )
                self._send(st, b, h)
                return

            st, b, h = _json_bytes({"error": "not found"}, 404)
            self._send(st, b, h)

        # ===== POST =====

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
                raw_agent = str(body.get("agent", AGENT_CLAUDE)).strip().lower()
                if raw_agent and raw_agent not in SUPPORTED_AGENTS:
                    st, b, h = _json_bytes(
                        {"error": f"agent must be one of {sorted(SUPPORTED_AGENTS)}"},
                        400,
                    )
                    self._send(st, b, h)
                    return
                meta = create_task(
                    root,
                    title,
                    general_goal,
                    skills_path=skills_path,
                    interview_model=str(body.get("interview_model", "")),
                    agent=raw_agent or AGENT_CLAUDE,
                )
                cands = list_worktree_candidates(root)
                hint = ""
                if not meta.worktree_path:
                    if not cands:
                        hint = (
                            f" (no git repo at project root {root} or its direct"
                            " children; nothing to worktree)"
                        )
                    else:
                        hint = (
                            f" (auto-skip: {len(cands)} candidate(s) "
                            f"available - pick one via the Claude tab)"
                        )
                print(
                    f"[web] created task slug={meta.slug} dir={task_root(root, meta.slug)} "
                    f"worktree={meta.worktree_path or '(none)'} "
                    f"branch={meta.branch or '(none)'}{hint}",
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
                        "worktree": meta.worktree_path or "",
                        "branch": meta.branch or "",
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

            # Claude pane lifecycle - the same two route prefixes were
            # called /interview/{start,stop} before the rename, accept both.
            m_start = re.match(r"^/api/tasks/([^/]+)/(?:claude|interview)/start$", path)
            if m_start:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_start.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                result = claude_registry.start(root, project_id, slug)
                print(
                    f"[web] start claude slug={slug} ok={bool(result.get('ok'))} "
                    f"session={result.get('session', '')} target={result.get('target', '')}",
                    flush=True,
                )
                openclaw_client.emit(
                    "claude-start",
                    instruction=f"claudeloop Claude pane started for task {slug}",
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

            m_stop = re.match(r"^/api/tasks/([^/]+)/(?:claude|interview)/stop$", path)
            if m_stop:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_stop.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                result = claude_registry.stop(root, project_id, slug)
                openclaw_client.emit(
                    "claude-stop",
                    instruction=f"claudeloop Claude pane stopped for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data=result,
                )
                st, b, h = _json_bytes(result)
                self._send(st, b, h)
                return

            m_wt_create = re.match(r"^/api/tasks/([^/]+)/worktree$", path)
            if m_wt_create:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_wt_create.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                raw_src = str(body.get("source_repo", "")).strip()
                if not raw_src:
                    st, b, h = _json_bytes({"error": "source_repo required"}, 400)
                    self._send(st, b, h)
                    return
                # Whitelist against the project's candidate list so a
                # poisoned request can't make us run `git worktree add`
                # against an arbitrary path on disk.
                allowed = {
                    str(Path(c["path"]).resolve())
                    for c in list_worktree_candidates(root)
                }
                try:
                    src_resolved = str(Path(raw_src).expanduser().resolve())
                except OSError as exc:
                    st, b, h = _json_bytes({"error": f"invalid path: {exc}"}, 400)
                    self._send(st, b, h)
                    return
                if src_resolved not in allowed:
                    st, b, h = _json_bytes(
                        {
                            "error": "source_repo is not in the project's candidate list",
                            "allowed": sorted(allowed),
                        },
                        400,
                    )
                    self._send(st, b, h)
                    return
                wt, branch, msg = prepare_task_worktree_from(
                    root, slug, Path(src_resolved)
                )
                print(
                    f"[web] manual worktree slug={slug} src={src_resolved} "
                    f"ok={wt is not None} msg={msg}",
                    flush=True,
                )
                if wt is None:
                    st, b, h = _json_bytes(
                        {"ok": False, "error": msg, "branch": branch}, 400
                    )
                    self._send(st, b, h)
                    return
                # Append (or refresh) the worktree list from disk.  Don't
                # call update_meta directly so order / branches stay in
                # sync across the existing entries.
                updated = detect_and_persist_worktree(root, slug) or read_meta(root, slug)
                openclaw_client.emit(
                    "worktree-created",
                    instruction=f"claudeloop worktree created for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data={"source_repo": src_resolved, "worktree": str(wt), "branch": branch},
                )
                st, b, h = _json_bytes(
                    {
                        "ok": True,
                        "worktree_path": str(wt),
                        "branch": branch,
                        "message": msg,
                        "meta": updated.to_dict() if updated else None,
                    }
                )
                self._send(st, b, h)
                return

            m_wt_push = re.match(r"^/api/tasks/([^/]+)/worktree/push$", path)
            if m_wt_push:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_wt_push.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                raw_path = str(body.get("path", "")).strip()
                if not raw_path:
                    st, b, h = _json_bytes({"error": "path required"}, 400)
                    self._send(st, b, h)
                    return
                try:
                    wt = Path(raw_path).expanduser().resolve()
                except OSError as exc:
                    st, b, h = _json_bytes({"error": f"invalid path: {exc}"}, 400)
                    self._send(st, b, h)
                    return
                if str(wt) not in meta.worktrees:
                    st, b, h = _json_bytes(
                        {"error": "worktree is not registered with this task"},
                        400,
                    )
                    self._send(st, b, h)
                    return
                result = push_worktree_branch(wt)
                # Refresh status snapshot so the UI can update ahead/behind.
                result["status"] = worktree_status(wt)
                print(
                    f"[web] push worktree slug={slug} path={wt} "
                    f"ok={result.get('ok')} branch={result.get('branch')}",
                    flush=True,
                )
                st, b, h = _json_bytes(result, 200 if result.get("ok") else 400)
                self._send(st, b, h)
                return

            m_wt_push_all = re.match(r"^/api/tasks/([^/]+)/worktrees/push-all$", path)
            if m_wt_push_all:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_wt_push_all.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                results: list[dict[str, Any]] = []
                for p_str in meta.worktrees:
                    wt = Path(p_str)
                    row = push_worktree_branch(wt)
                    row["path"] = p_str
                    row["status"] = worktree_status(wt)
                    results.append(row)
                ok_all = bool(results) and all(r.get("ok") for r in results)
                print(
                    f"[web] push-all slug={slug} ok={sum(1 for r in results if r.get('ok'))}/{len(results)}",
                    flush=True,
                )
                openclaw_client.emit(
                    "worktrees-pushed",
                    instruction=f"claudeloop pushed worktree branches for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data={"results": results},
                )
                st, b, h = _json_bytes(
                    {"ok": ok_all, "count": len(results), "results": results}
                )
                self._send(st, b, h)
                return

            m_resume = re.match(r"^/api/tasks/([^/]+)/claude/resume$", path)
            if m_resume:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_resume.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                sid = str(body.get("session_id", "")).strip()
                if not _SESSION_ID_RE.match(sid):
                    st, b, h = _json_bytes({"error": "invalid session_id"}, 400)
                    self._send(st, b, h)
                    return
                result = claude_registry.start(root, project_id, slug, resume_session_id=sid)
                print(
                    f"[web] resume claude slug={slug} session={sid} ok={bool(result.get('ok'))} "
                    f"target={result.get('target', '')}",
                    flush=True,
                )
                openclaw_client.emit(
                    "claude-resume",
                    instruction=f"claudeloop Claude pane resumed for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data={**result, "session_id": sid},
                )
                st, b, h = (
                    _json_bytes(result)
                    if result.get("ok")
                    else _json_bytes(result, 400)
                )
                self._send(st, b, h)
                return

            st, b, h = _json_bytes({"error": "not found"}, 404)
            self._send(st, b, h)

        # ===== PUT =====

        def do_PUT(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path
            body = _read_json(self)

            if path == "/api/notes":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                content = body.get("content", "")
                if not isinstance(content, str):
                    st, b, h = _json_bytes({"error": "content must be string"}, 400)
                    self._send(st, b, h)
                    return
                if not write_project_notes(root, content):
                    st, b, h = _json_bytes({"error": "failed to write NOTES.md"}, 500)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes({"ok": True})
                self._send(st, b, h)
                return

            m_meta = re.match(r"^/api/tasks/([^/]+)/meta$", path)
            if m_meta:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m_meta.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                title = body.get("title")
                goal = body.get("general_goal")
                agent_in = body.get("agent")
                if title is None and goal is None and agent_in is None:
                    st, b, h = _json_bytes(
                        {"error": "supply title and/or general_goal and/or agent"},
                        400,
                    )
                    self._send(st, b, h)
                    return
                if agent_in is not None:
                    raw = str(agent_in).strip().lower()
                    if raw not in SUPPORTED_AGENTS:
                        st, b, h = _json_bytes(
                            {"error": f"agent must be one of {sorted(SUPPORTED_AGENTS)}"},
                            400,
                        )
                        self._send(st, b, h)
                        return
                    update_meta(root, slug, agent=raw)
                updated = rename_task_meta(
                    root,
                    slug,
                    title=str(title) if title is not None else None,
                    general_goal=str(goal) if goal is not None else None,
                ) or read_meta(root, slug)
                if updated is None:
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

        # ===== DELETE =====

        def do_DELETE(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path

            m_wt_del = re.match(r"^/api/tasks/([^/]+)/worktree$", path)
            if m_wt_del:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_wt_del.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                qs = parse_qs(parsed.query or "")
                raw_path = (qs.get("path") or [""])[0].strip()
                if not raw_path:
                    st, b, h = _json_bytes({"error": "path query param required"}, 400)
                    self._send(st, b, h)
                    return
                try:
                    wt_target = Path(raw_path).expanduser().resolve()
                except OSError as exc:
                    st, b, h = _json_bytes({"error": f"invalid path: {exc}"}, 400)
                    self._send(st, b, h)
                    return
                ok_rm, msg_rm = remove_task_worktree(root, slug, wt_target)
                print(
                    f"[web] remove worktree slug={slug} path={wt_target} "
                    f"ok={ok_rm} msg={msg_rm}",
                    flush=True,
                )
                if not ok_rm:
                    st, b, h = _json_bytes({"ok": False, "error": msg_rm}, 400)
                    self._send(st, b, h)
                    return
                openclaw_client.emit(
                    "worktree-removed",
                    instruction=f"claudeloop worktree removed for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data={"worktree": str(wt_target)},
                )
                updated = read_meta(root, slug)
                st, b, h = _json_bytes(
                    {
                        "ok": True,
                        "message": msg_rm,
                        "meta": updated.to_dict() if updated else None,
                    }
                )
                self._send(st, b, h)
                return

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


# --- Bootstrap --------------------------------------------------------------


def serve(
    host: str,
    port: int,
    project_root: Path,
    default_skills: Path,
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
    claude_registry = ClaudeRegistry()
    openclaw_client = OpenClawClient(openclaw_config)
    sk = default_skills if default_skills.is_file() else bundled_skills_path().resolve()
    handler = make_handler(
        web_project_registry,
        project_root,
        sk,
        claude_registry,
        openclaw_client,
        auth_token,
        multi_project_workspace=multi_project_workspace,
    )
    server = HTTPServer((host, port), handler)
    rud_root = project_root / ".RUD"
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
    print(f"  Project notes:    {rud_root}/NOTES.md", flush=True)
    print(f"  Static assets:    {web_static_dir().resolve()}", flush=True)
    print(f"  Default skills:   {sk}", flush=True)
    print("  Tabs:             Claude, PLAN.md (per task) + Notes button (per project)", flush=True)
    print(f"  Auth:             {'enabled' if auth_token.strip() else 'disabled'}", flush=True)
    print(f"  OpenClaw:         {openclaw_status(openclaw_client.config)}", flush=True)
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
