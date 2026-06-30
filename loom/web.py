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
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from loom.openclaw import OpenClawClient, OpenClawConfig, openclaw_status
from loom.paths import bundled_skills_path, kernel_hub_dir, web_static_dir
from loom.rud_task import (
    AGENT_CLAUDE,
    AGENT_CODEX,
    DEFAULT_MONITOR_PATTERN,
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
    merge_worktree_to_base,
    normalize_agent,
    prepare_task_worktree_from,
    push_worktree_branch,
    read_kernel_interview,
    read_meta,
    read_project_notes,
    read_task_markdown_file,
    read_task_monitor,
    read_template,
    remove_task_worktree,
    reorder_tasks,
    rename_task_meta,
    session_id_from_path,
    task_root,
    task_worktree_diffs,
    task_worktree_path,
    update_meta,
    worktree_diff,
    worktree_status,
    write_kernel_interview,
    write_project_notes,
    write_task_monitor,
    write_template,
)
from loom.tmux_util import (
    capture_pane,
    list_tmux_panes,
    list_tmux_sessions,
    open_pane_attach,
    scroll_pane,
    send_pane_key,
    send_pane_literal,
    send_pane_text,
    tmux_available,
    tmux_subprocess_env,
    validate_tmux_target,
)
from loom.web_projects import WebProjectRegistry

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


# Current brand prefix for new tmux session names. "claudeloop" is the legacy
# prefix from before the rename and is still recognized for reuse/cleanup so
# panes started by older builds keep working.
_SESSION_BRAND = "loom"
_SESSION_BRANDS = ("loom", "claudeloop")


def _sanitize_session_name(raw: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "-", raw).strip("-")
    return safe[:90] or fallback


def _safe_claude_session_name(project_id: str, slug: str, agent: str = AGENT_CLAUDE) -> str:
    """Tmux session name for a task's agent pane (current ``loom-`` brand).

    The agent name is part of the prefix so a claude pane and a codex pane for
    the same project never share a tmux session if the user ever changes agent.
    The legacy ``claudeloop-`` prefix and old ``interview`` pane name are handled
    via ``_legacy_claude_session_name`` / ``_session_name_aliases`` and
    ``_filter_tmux_sessions_for_project``.
    """
    tid = _tmux_id_fragment(project_id)
    agent = normalize_agent(agent)
    return _sanitize_session_name(f"{_SESSION_BRAND}-{agent}-{tid}-{slug}", f"{_SESSION_BRAND}-{agent}")


def _legacy_claude_session_name(project_id: str, slug: str, agent: str = AGENT_CLAUDE) -> str:
    """Pre-rename ``claudeloop-`` session name for the same task/agent."""
    tid = _tmux_id_fragment(project_id)
    agent = normalize_agent(agent)
    return _sanitize_session_name(f"claudeloop-{agent}-{tid}-{slug}", f"claudeloop-{agent}")


def _session_name_aliases(project_id: str, slug: str) -> set[str]:
    """Every session name this task could use across brands + agents (plus the
    old ``interview`` pane name) - so we can find/clean up its pane regardless of
    which build started it."""
    tid = _tmux_id_fragment(project_id)
    names: set[str] = set()
    for brand in _SESSION_BRANDS:
        for ag in (*SUPPORTED_AGENTS, "interview"):
            names.add(_sanitize_session_name(f"{brand}-{ag}-{tid}-{slug}", f"{brand}-{ag}"))
    return names


def _path_within(child: Path, parent: Path) -> bool:
    """True if *child* resolves to *parent* or a path inside it (works for paths
    that don't exist yet)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _git_clone(repo_url: str, dest: Path, timeout: int = 900) -> tuple[bool, str]:
    """``git clone`` *repo_url* into *dest*. Returns ``(ok, error)``."""
    import shutil

    if not shutil.which("git"):
        return False, "git is not installed on the server"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, str(exc)
    try:
        # ``--`` so a repo URL can never be misread as a git option.
        r = subprocess.run(
            ["git", "clone", "--", repo_url, str(dest)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"git clone timed out after {timeout}s"
    except OSError as exc:
        return False, str(exc)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "git clone failed").strip()[-2000:]
    return True, ""


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
        f"{brand}-{name}-{tid}-"
        for brand in _SESSION_BRANDS
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


def _available_skill_options(
    default_skills: Path,
    project_root: Path | None = None,  # kept for call-site compatibility; unused
    *,
    limit: int = 500,
) -> list[dict[str, str]]:
    """Return selectable skill markdown files for the web UI.

    Scope is intentionally limited to the bundled ``loom/skills``
    directory (plus the configured default skills file). We do **not** scan
    the user's project tree, so unrelated README/PLAN/etc. markdown never
    shows up in the Skills picker - only real skills files are selectable.
    """
    del project_root  # skills come only from the skills directory
    seen: set[Path] = set()
    options: list[dict[str, str]] = []
    skills_root = bundled_skills_path().parent

    def add(path: Path) -> None:
        try:
            p = path.expanduser().resolve()
        except OSError:
            return
        if (
            len(options) >= limit
            or not p.is_file()
            or p.suffix.lower() != ".md"
            or p in seen
        ):
            return
        seen.add(p)
        label = p.name
        try:
            label = str(p.relative_to(skills_root))
        except ValueError:
            pass
        options.append({"label": label, "path": str(p)})

    add(default_skills)
    if skills_root.is_dir():
        for p in sorted(skills_root.rglob("*.md"), key=lambda x: str(x).lower()):
            add(p)
    return options


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


# --- Kernel Lab (vendored kernel hub) ---------------------------------------
# Loom drives kernel-optimization runs by shelling out to the bundled
# kernel_hub/scaffold/agent_runner/rud_kernel.py helper (JSON in/out). Run
# records are stored per task under <root>/.RUD/kernel-runs/<id>.json.

_KERNEL_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _kernel_runs_dir(root: Path) -> Path:
    return root / ".RUD" / "kernel-runs"


def _kernel_write_record(root: Path, rec: dict[str, Any]) -> None:
    d = _kernel_runs_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{rec['id']}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    tmp.replace(dest)


def _kernel_read_record(root: Path, run_uid: str) -> dict[str, Any] | None:
    f = _kernel_runs_dir(root) / f"{run_uid}.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _kernel_delete_record(root: Path, run_uid: str) -> bool:
    """Remove a run's record JSON (and its build log) from disk. Used by the UI
    to clear finished/errored runs. Returns True if the JSON existed."""
    d = _kernel_runs_dir(root)
    existed = False
    for f in (d / f"{run_uid}.json", d / f"{run_uid}.json.tmp", d / f"{run_uid}.log"):
        try:
            if f.is_file():
                if f.suffix == ".json":
                    existed = True
                f.unlink()
        except OSError:
            pass
    return existed


def _sweep_stale_kernel_runs(roots: list[Path]) -> int:
    """Mark any ``launching``/``resolving`` run records as ``error`` across the
    given project roots. Called at server startup: a launch/prepare's worker
    thread can't survive a restart, so such records are definitionally stale."""
    swept = 0
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        d = _kernel_runs_dir(root)
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("state") in ("launching", "resolving"):
                rec["state"] = "error"
                rec["error"] = "launch interrupted by a server restart (stale)"
                try:
                    f.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
                    swept += 1
                except OSError:
                    pass
    return swept


def _kernel_list_records(root: Path, slug: str | None = None) -> list[dict[str, Any]]:
    d = _kernel_runs_dir(root)
    if not d.is_dir():
        return []
    recs: list[dict[str, Any]] = []
    for f in d.glob("*.json"):
        try:
            recs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    if slug:
        # Strict per-task scoping: a task shows ONLY its own runs (matched by the
        # Loom task slug). Older runs created before per-task scoping have no slug
        # and used to be shown under *every* task — that made separate kernel
        # tasks appear to "share" the same runs/leaderboards. They no longer leak
        # across tasks; such orphans simply don't appear under any task (the JSON
        # files remain on disk and can be cleaned up or re-attributed manually).
        scoped: list[dict[str, Any]] = []
        for r in recs:
            rslug = str(r.get("slug") or r.get("task_slug") or "").strip()
            if rslug == slug:
                scoped.append(r)
        recs = scoped
    recs.sort(key=lambda r: r.get("created_at", 0.0), reverse=True)
    return recs


def _kernel_helper_cmd(script_name: str) -> tuple[list[str] | None, str]:
    """Resolve how to invoke the bundled kernel helper, returning
    ``(base_cmd, error)``.

    The kernel stack is vendored in-tree under
    ``loom/kernel_hub/scaffold/agent_runner/``; the helper's own
    ``REPO_ROOT`` resolves to ``kernel_hub/`` so docker-compose and the
    kernel_evaluator service are found right beside it.
    """
    bundled = kernel_hub_dir() / "scaffold" / "agent_runner" / script_name
    if bundled.is_file():
        return [sys.executable, str(bundled)], ""
    return None, (
        f"kernel helper '{script_name}' not found at {bundled}; the vendored "
        f"kernel stack (loom/kernel_hub/) appears to be missing."
    )


# Short TTL cache for `service-status` so frequent polls (and concurrent
# browser tabs) don't each spawn a subprocess + network health-check.
_KERNEL_SERVICE_CACHE: dict[str, tuple[float, bool, dict[str, Any]]] = {}
_KERNEL_SERVICE_TTL = 6.0
_kernel_service_lock = threading.Lock()


def _kernel_service_status_cached(root: Path) -> tuple[bool, dict[str, Any]]:
    key = str(root)
    now = time.time()
    with _kernel_service_lock:
        hit = _KERNEL_SERVICE_CACHE.get(key)
        if hit and (now - hit[0]) < _KERNEL_SERVICE_TTL:
            return hit[1], hit[2]
    ok, data = _run_kernel_helper(root, ["service-status"], timeout=15)
    with _kernel_service_lock:
        _KERNEL_SERVICE_CACHE[key] = (now, ok, data)
    return ok, data


def _run_kernel_helper(
    root: Path, helper_args: list[str], timeout: int = 600
) -> tuple[bool, dict[str, Any]]:
    """Invoke rud_kernel (in-project script or pip module) and parse its JSON."""
    base, err = _kernel_helper_cmd("rud_kernel.py")
    if base is None:
        return False, {"ok": False, "error": err}
    try:
        proc = subprocess.run(
            [*base, *helper_args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, {"ok": False, "error": f"kernel helper timed out after {timeout}s"}
    out = (proc.stdout or "").strip()
    last = out.splitlines()[-1] if out else ""
    try:
        data = json.loads(last)
    except json.JSONDecodeError:
        return False, {
            "ok": False,
            "error": "kernel helper returned non-JSON",
            "stdout": out[-1000:],
            "stderr": (proc.stderr or "")[-1000:],
        }
    return bool(data.get("ok")), data


def _shape_to_str(shape: Any) -> str:
    return shape if isinstance(shape, str) else json.dumps(shape)


def _kernel_run_log_path(root: Path, run_uid: str) -> Path:
    return _kernel_runs_dir(root) / f"{run_uid}.log"


def _run_kernel_launch_streaming(
    root: Path, run_uid: str, helper_args: list[str], timeout: int = 2400
) -> tuple[bool, dict[str, Any]]:
    """Run the launch helper, streaming its progress (docker build, agent
    bring-up, …) to ``<run_uid>.log`` live so the web UI can tail it. The
    helper prints its final single-line JSON result to stdout (captured);
    everything else (the build log) goes to stderr → the log file."""
    base, err = _kernel_helper_cmd("rud_kernel.py")
    if base is None:
        return False, {"ok": False, "error": err}
    log_path = _kernel_run_log_path(root, run_uid)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = ""
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"$ {' '.join(base + helper_args)}\n\n")
            lf.flush()
            proc = subprocess.Popen(
                [*base, *helper_args],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=lf,
                text=True,
            )
            try:
                out, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
                out = (out or "") + f"\n[helper timed out after {timeout}s]"
    except OSError as exc:
        return False, {"ok": False, "error": str(exc)}
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n" + (out or ""))
    except OSError:
        pass
    data: dict[str, Any] | None = None
    for line in reversed((out or "").splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                data = json.loads(s)
                break
            except json.JSONDecodeError:
                continue
    if data is None:
        return False, {"ok": False, "error": "launch produced no result (see build log)"}
    return bool(data.get("ok")), data


def _kernel_propose_shape(root: Path, cfg: dict[str, Any]) -> Any:
    """Have the host ``claude`` CLI choose a representative benchmark shape for
    the kernel op, so the shape is agent-decided rather than a human input.

    Returns the shape (a dict) or ``None`` on any failure, in which case the
    caller omits ``--shape`` and the kernel helper falls back to the plugin's
    default template.
    """
    plugin = str(cfg.get("plugin", "")).strip()
    target = str(cfg.get("target", "")).strip()
    model = str(cfg.get("model", "")).strip()
    if not plugin:
        return None
    # Show the agent the plugin's expected shape keys (if known) so it returns a
    # shape the evaluator can actually use; a freshly-resolved plugin has none,
    # and the agent infers the keys from the operation instead.
    tpl = None
    try:
        ok, data = _run_kernel_helper(root, ["plugins"], timeout=30)
        if ok:
            tpl = (data.get("shape_templates") or {}).get(plugin)
    except Exception:  # noqa: BLE001
        tpl = None
    if tpl is not None:
        keys_hint = (
            "The shape is a JSON object with EXACTLY these keys (example values "
            "shown — keep the keys, pick realistic representative values for a "
            f"meaningful benchmark):\n{json.dumps(tpl)}"
        )
    else:
        keys_hint = (
            "Infer the correct shape keys for this operation yourself (e.g. "
            "batch, heads, seq_len, head_dim, m/n/k, page_size, dtype as "
            "appropriate) and pick realistic, representative values for a "
            "meaningful benchmark."
        )
    prompt = (
        "You are choosing ONE representative benchmark shape for a GPU kernel "
        f'optimization run. Operation/plugin: "{plugin}". Target backend: '
        f'"{target or "unspecified"}".\n\n{keys_hint}\n\n'
        "Reply with ONLY a single fenced ```json code block containing the shape "
        "object, and no other prose."
    )
    cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions"]
    # Only forward a Claude model; codex/gpt models aren't valid for `claude`.
    if model.startswith("claude-"):
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(
        r"(\{.*\})", text, re.DOTALL
    )
    if not m:
        return None
    try:
        shape = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return shape if isinstance(shape, dict) and shape else None


def _launch_kernel_run(root: Path, run_uid: str, cfg: dict[str, Any]) -> None:
    """Background worker: run the helper's launch and update the run record."""
    # Shape is no longer a human input. Use an explicit override when supplied,
    # otherwise let an agent propose a representative shape at launch. If that
    # also fails, omit --shape so the kernel helper falls back to the plugin's
    # default template.
    if cfg.get("shape"):
        cfg["shape_source"] = "override"
    else:
        proposed = _kernel_propose_shape(root, cfg)
        if proposed:
            cfg["shape"] = proposed
            cfg["shape_source"] = "agent"
        else:
            cfg["shape_source"] = "template"
    # Persist the resolved shape early so the UI reflects the agent's choice
    # while the (slow) build + launch runs.
    early = _kernel_read_record(root, run_uid) or {"id": run_uid}
    early["config"] = cfg
    _kernel_write_record(root, early)
    args = [
        "launch",
        "--plugin", str(cfg["plugin"]),
        "--target", str(cfg["target"]),
        "--model", str(cfg["model"]),
        "--n-agents", str(cfg.get("n_agents", 1)),
        "--starter-mode", str(cfg.get("starter_mode", "none")),
    ]
    if cfg.get("shape"):
        args += ["--shape", _shape_to_str(cfg["shape"])]
    if cfg.get("target_speedup") is not None:
        args += ["--target-speedup", str(cfg["target_speedup"])]
    if cfg.get("auto_terminate"):
        args += ["--auto-terminate", "--poll-interval", str(cfg.get("poll_interval", 60))]
    if cfg.get("build"):
        args += ["--build"]
    if cfg.get("build_mode"):
        args += ["--build-mode"]
    ok, data = _run_kernel_launch_streaming(root, run_uid, args, timeout=2400)
    rec = _kernel_read_record(root, run_uid) or {"id": run_uid}
    # Persist the resolved shape + how it was chosen so the UI can show it.
    rec["config"] = cfg
    if ok:
        rec.update({
            "state": "running",
            "run_id": data.get("run_id"),
            "task_slug": data.get("task_slug"),
            "containers": data.get("containers", []),
            "plugin": cfg.get("plugin"),
            "verified": cfg.get("plugin") not in _kernel_unverified_set(root),
            "launched_at": time.time(),
        })
    else:
        rec.update({
            "state": "error",
            "error": data.get("error", "launch failed"),
            "error_detail": {
                k: data[k] for k in ("stderr", "stdout", "stdout_tail", "service") if k in data
            },
        })
    _kernel_write_record(root, rec)


# --- Kernel Lab: verified state + interview-driven prepare ---

def _kernel_unverified_path(root: Path) -> Path:
    return root / ".RUD" / "kernel-plugins-unverified.json"


def _kernel_unverified_set(root: Path) -> set[str]:
    f = _kernel_unverified_path(root)
    if not f.is_file():
        return set()
    try:
        return set(json.loads(f.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def _kernel_set_unverified(root: Path, name: str, unverified: bool) -> None:
    s = _kernel_unverified_set(root)
    if unverified:
        s.add(name)
    else:
        s.discard(name)
    f = _kernel_unverified_path(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(s)))
    tmp.replace(f)


def _resolve_plugin_for(root: Path, source: str, timeout: int = 2400) -> tuple[str | None, bool, str]:
    """Run resolve_plugin (in-project script or pip module); return
    (plugin_name, created, output_tail)."""
    base, err = _kernel_helper_cmd("resolve_plugin.py")
    if base is None:
        return None, False, err
    try:
        proc = subprocess.run(
            [*base, "--source", source],
            cwd=str(root), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, False, "resolve_plugin timed out"
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    m = re.search(r"RESULT:\s*(CREATE|REUSE)\s+(\S+)", out)
    if not m:
        return None, False, out[-1500:]
    return m.group(2), (m.group(1) == "CREATE"), out[-1500:]


def _prepare_kernel_run(root: Path, prep_uid: str, spec: dict[str, Any]) -> None:
    """Background: resolve the plugin for the interview spec, mark a newly created
    plugin unverified, and leave a 'prepared' record the UI can launch from."""
    rec = _kernel_read_record(root, prep_uid) or {"id": prep_uid}
    source = str(spec.get("source", "")).strip()
    if not source:
        rec.update({"state": "error", "error": "interview spec has no source kernel"})
        _kernel_write_record(root, rec)
        return
    rec["state"] = "resolving"
    _kernel_write_record(root, rec)
    plugin, created, out = _resolve_plugin_for(root, source)
    if plugin is None:
        rec.update({"state": "error", "error": "plugin resolution failed", "error_detail": out})
        _kernel_write_record(root, rec)
        return
    if created:
        _kernel_set_unverified(root, plugin, True)
    rec.update({
        "state": "prepared",
        "kind": "prepare",
        "plugin": plugin,
        "plugin_created": created,
        "verified": plugin not in _kernel_unverified_set(root),
        "needs_build": created,
        "resolve_output": out,
        "prepared_at": time.time(),
    })
    _kernel_write_record(root, rec)


_KERNEL_INTERVIEW_SYS = """You are running a short technical interview inside "Kernel Lab" to collect everything needed to (a) define a kernel eval plugin for a GPU kernel and (b) launch an optimization run for it. Ask ONE focused question at a time and be concise. If the user gives a GitHub raw URL or a source link, use your tools to read it and INFER as much as possible (dims, dtype, operation) — only ask what you cannot infer.

Collect: source (a GitHub raw URL, a kernel name, or a clear description of the operation); target hardware (SM100 -> cutedsl/fp8, or H100 -> cuda with bf16/fp8 — this affects whether benchmarking is possible here); run params (target speedup [optional], number of agents, starter mode where "preset" means use the user's file as starting code).

Do NOT ask the user for the operation shape/dims. Infer a single representative shape yourself from the source/operation (operation-specific; for attention: heads, head_dim or latent+rope, page_size, KV length, query length Sq, batch, dtype) and include it in the spec — the evaluator benchmarks this shape and the user can override it later if needed.

When AND ONLY WHEN you have everything, reply with ONLY a fenced ```json code block (no other prose), shaped like:
{"done": true, "spec": {"source": "<url-or-name>", "target": "cutedsl", "shape": {"batch_size": 4, "num_heads": 128}, "dtype": "fp8", "model": "claude-sonnet-4-20250514", "n_agents": 3, "starter_mode": "preset", "target_speedup": null}}
Otherwise reply with your next question as plain text only."""


def _kernel_interview_turn(messages: list[dict[str, Any]], model: str = "") -> dict[str, Any]:
    """One interview turn via the logged-in host `claude` CLI. Returns either the
    next question ({done:false, assistant}) or a final spec ({done:true, spec})."""
    convo = "\n".join(
        f"{str(m.get('role', 'user')).capitalize()}: {m.get('content', '')}" for m in messages
    )
    prompt = (
        f"{_KERNEL_INTERVIEW_SYS}\n\nConversation so far:\n{convo}\n\n"
        "Produce your next turn (a single question, or the final json spec)."
    )
    cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "interview turn timed out"}
    text = (proc.stdout or "").strip()
    if not text:
        return {"ok": False, "error": "empty response from claude", "stderr": (proc.stderr or "")[-500:]}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(
        r"(\{\s*\"done\"\s*:\s*true.*\})", text, re.DOTALL
    )
    if m:
        try:
            obj = json.loads(m.group(1))
            if obj.get("done"):
                return {"ok": True, "done": True, "spec": obj.get("spec", obj)}
        except json.JSONDecodeError:
            pass
    return {"ok": True, "done": False, "assistant": text}


_REVIEW_DEFAULT_RULES = """- Correctness: logic bugs, edge cases, wrong APIs, off-by-one, error handling.
- Security: NO hardcoded secrets/tokens/keys; no injection; safe file/subprocess use.
- Hygiene: no leftover debug prints, commented-out code, stray TODOs, dead code.
- Tests: meaningful changes should add/keep tests; flag "claims tested" with no test.
- Consistency: matches the surrounding code style and the project's skills."""


def _run_worktree_review(
    wt: Path, rules: str, skills: str, model: str = ""
) -> dict[str, Any]:
    """Bugbot-style review: run the logged-in host ``claude -p`` over a
    worktree's diff against plain-English rules (falling back to the task's
    skills). Returns ``{ok, review}`` markdown."""
    diff = worktree_diff(wt)
    files = diff.get("files", [])
    if not files:
        return {"ok": True, "review": "✅ No changes to review in this worktree.", "files": 0}
    parts: list[str] = []
    total = 0
    for f in files:
        patch = f.get("patch") or ""
        if not patch:
            continue
        parts.append(patch)
        total += len(patch)
        if total > 60000:
            parts.append("\n... (diff truncated for review) ...\n")
            break
    diff_text = "\n".join(parts).strip() or "(no textual diff)"
    rules_block = (rules or "").strip() or (skills or "").strip() or "(use general best practices)"
    prompt = (
        "You are a strict senior code reviewer (think Bugbot). Review the DIFF "
        "for this change. Only flag real problems. For each finding, output a "
        "markdown bullet exactly like:\n"
        "  - `path:line` - **[severity]** what's wrong -> concrete fix\n"
        "Always cover: correctness/bugs, security (secrets, injection), leftover "
        "debug/TODO/dead code, missing or weak tests, and any RULES violations. "
        "If everything looks good, reply with exactly: `✅ No issues found.` "
        "Keep it concise.\n\n"
        f"RULES (plain-English, from the user / task skills):\n{rules_block}\n\n"
        f"DEFAULT CHECKLIST:\n{_REVIEW_DEFAULT_RULES}\n\n"
        f"DIFF:\n{diff_text}"
    )
    cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "review timed out"}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    text = (proc.stdout or "").strip()
    if not text:
        return {
            "ok": False,
            "error": "empty response from claude",
            "stderr": (proc.stderr or "")[-500:],
        }
    return {"ok": True, "review": text, "files": len(files)}


# --- Claude prompt builder --------------------------------------------------


def _build_claude_prompt(
    project_root: Path,
    slug: str,
    default_skills: Path | None = None,
) -> str:
    meta = read_meta(project_root, slug)
    if not meta:
        return ""
    td = task_root(project_root, slug)
    wt = task_worktree_path(project_root, slug)
    wt_line = f"Worktree (branch {meta.branch or '(unset)'}): {wt}" if wt else "Worktree: (none)"
    skills = ""
    skills_path = Path(meta.skills_path).expanduser() if meta.skills_path else None
    if skills_path is None or not skills_path.is_file():
        skills_path = default_skills if default_skills and default_skills.is_file() else bundled_skills_path()
    if skills_path.is_file():
        skills = skills_path.read_text(encoding="utf-8", errors="replace")[:12000]
    plan_path = td / PLAN
    if (meta.kind or "").strip().lower() == "aris":
        aris_skill_path = bundled_skills_path().parent / "aris" / "ARIS.md"
        aris_skill = ""
        if aris_skill_path.is_file():
            aris_skill = aris_skill_path.read_text(encoding="utf-8", errors="replace")[:20000]
        return f"""You are running an ARIS (Auto Research In Sleep) task in Loom -
an autonomous research / optimization loop driven by you.

Task directory:
{td}

{wt_line}

Experiment worktrees go under: {td / "work"}/

General goal (your research objective):
{meta.general_goal}

PLAN.md ledger (your single source of truth):
{plan_path}

=== ARIS methodology - follow this exactly ===
{aris_skill or "(ARIS skill missing)"}
=== end ARIS methodology ===

Domain skills (extra context for this base):
---
{skills or "(none)"}
---

Begin now: read {plan_path}; if it has no ledger yet, bootstrap it from the
General goal (Goal + Baseline + a ranked Idea backlog). Then start cycle 1 -
branch a worktree under work/ for the top idea, run a real experiment, and
record the result back into {plan_path}. Keep running cycles autonomously;
only pause for the human checkpoints listed in the methodology. PLAN.md is the
ONLY task-state file - do not create other status files.
"""
    return f"""You are running Loom's {agent_label(meta.agent)} pane for this task.

You start in this task's work directory (your git worktree is a subdirectory here - cd into it to touch code):
{td / "work"}

General goal:
{meta.general_goal}

{wt_line}

Default skills from:
{skills_path}

Default skills:
---
{skills or "(none)"}
---

RUD workflow:
1. Start from the General goal above and run a short deep-interview. Ask
   one high-leverage question at a time about scope, constraints,
   acceptance, tests, risks, non-goals, and available worktrees.
2. When the interview has enough information, write or overwrite
   {plan_path} with a concise executable plan:
   - Goal
   - Context / Decisions from the interview
   - Constraints / non-goals
   - Acceptance criteria
   - Next steps as a checkbox list
   - Progress Log / Result section
   Do not leave interview notes only in chat; the result of the interview
   must be captured DIRECTLY in {plan_path}.
3. After PLAN.md is solid, tell the user it is ready to run. The user can
   click RUD's "Run /goal" button (or type /goal) to execute PLAN.md.
4. While executing and when finished, keep writing useful progress,
   blockers, decisions, and final results back into {plan_path}. Remove
   obsolete/noisy details, but preserve unrelated prior sections.

Behavioural constraints:
- PLAN.md is the ONLY task-state file. Do not create INTERVIEW.md,
  TODO.md, PROGRESS.md, NOTES.md, or any other scattered status files in
  the task directory or the repo.
- Project-scoped scratch lives in the project's NOTES.md at .RUD/NOTES.md
  (handled by the user via the web UI), not inside the worktree.

Begin by reading {plan_path}, then either ask the first interview
question or, if PLAN.md is already detailed enough, acknowledge that it is
ready and wait for the user to run ``/goal``.
"""


def _task_pane_cwd(project_root: Path, slug: str, meta=None) -> Path:
    """Directory the agent pane launches in - and where Claude Code stores the
    session transcript (``~/.claude/projects/<encoded-cwd>/``).

    All tasks launch in the task's ``work/`` dir: a stable base that holds every
    git worktree (the agent cd's into the relevant one to run git / code), so
    the session-transcript location stays consistent no matter how many
    worktrees a task has. Falls back to the primary worktree / task dir only if
    ``work/`` can't be created.
    """
    td = task_root(project_root, slug)
    wd = td / "work"
    try:
        wd.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if wd.is_dir():
        return wd
    wt = task_worktree_path(project_root, slug)
    return wt if wt is not None else td


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
        default_skills: Path | None = None,
    ) -> dict[str, Any]:
        meta = read_meta(project_root, slug)
        if not meta:
            return {"ok": False, "error": "Task not found"}
        td = task_root(project_root, slug)
        if not td.is_dir():
            return {"ok": False, "error": "Task directory missing"}

        # Run the agent inside the worktree when we have one - that's where
        # the user will eventually want /goal (or codex's equivalent) to act.
        cwd = _task_pane_cwd(project_root, slug, meta)

        agent = normalize_agent(meta.agent)
        session_name = self._live_or_default_session(project_id, slug, agent)
        target = f"{session_name}:0.0"
        existing_files = {p.name for p in list_session_files(cwd, agent)}
        if self._tmux_session_exists(session_name):
            if resume_session_id:
                pane_command = self._pane_current_command(target)
                if not self._pane_is_idle_shell(pane_command):
                    return {
                        "ok": False,
                        "error": (
                            "The tmux pane is still running a command. Stop it before "
                            "resuming another session."
                        ),
                        "target": target,
                        "session": session_name,
                        "pane_command": pane_command,
                    }
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
                add_claude_session(project_root, slug, resume_session_id)
                threading.Thread(
                    target=self._watch_for_session_id,
                    args=(project_root, slug, cwd, agent, existing_files),
                    daemon=True,
                ).start()
                return {
                    "ok": True,
                    "target": target,
                    "session": session_name,
                    "cwd": str(cwd),
                    "agent": agent,
                    "resumed_session_id": resume_session_id,
                    "already_running": False,
                    "reused_tmux": True,
                    "prompt_pending": False,
                    "pane_command": pane_command,
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
                target=self._watch_for_session_id,
                args=(project_root, slug, cwd, agent, existing_files),
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

    def paste_prompt(
        self,
        project_root: Path,
        project_id: str,
        slug: str,
        *,
        default_skills: Path | None = None,
    ) -> dict[str, Any]:
        """Paste the task's deep-interview prompt into the running agent pane."""
        meta = read_meta(project_root, slug)
        if not meta:
            return {"ok": False, "error": "Task not found"}
        agent = normalize_agent(meta.agent)
        session_name = self._live_or_default_session(project_id, slug, agent)
        target = (meta.tmux_interview_target or "").strip() or f"{session_name}:0.0"
        if not self._tmux_session_exists(session_name):
            return {"ok": False, "error": "Start the agent pane first"}
        update_meta(project_root, slug, tmux_interview_target=target)
        return self._paste_prompt_to_target(project_root, slug, target, default_skills=default_skills)

    def stop(self, project_root: Path, project_id: str, slug: str) -> dict[str, Any]:
        meta = read_meta(project_root, slug)
        agent = normalize_agent(meta.agent) if meta else AGENT_CLAUDE
        session_name = self._live_or_default_session(project_id, slug, agent)
        stopped, msg = self._kill_tmux_session(session_name)
        # Clean up every other variant for this task: the other agent's session
        # (if the user flipped agents), the old "interview" pane name, and the
        # legacy "claudeloop-" brand from before the rename.
        for alias in _session_name_aliases(project_id, slug):
            if alias != session_name:
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

    def _live_or_default_session(
        self, project_id: str, slug: str, agent: str = AGENT_CLAUDE
    ) -> str:
        """Session name to operate on: an already-running pane for this task
        (current ``loom-`` or legacy ``claudeloop-``), otherwise the new
        ``loom-`` name for a fresh start. Keeps pre-rename panes fully working."""
        primary = _safe_claude_session_name(project_id, slug, agent)
        if self._tmux_session_exists(primary):
            return primary
        legacy = _legacy_claude_session_name(project_id, slug, agent)
        if legacy != primary and self._tmux_session_exists(legacy):
            return legacy
        return primary

    def _pane_current_command(self, target: str) -> str:
        try:
            r = subprocess.run(
                ["tmux", "display-message", "-p", "-t", target, "#{pane_current_command}"],
                capture_output=True,
                text=True,
                env=tmux_subprocess_env(),
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if r.returncode != 0:
            return ""
        return (r.stdout or "").strip()

    @staticmethod
    def _pane_is_idle_shell(command: str) -> bool:
        cmd = Path((command or "").strip()).name.lower()
        return cmd in {"", "bash", "dash", "fish", "sh", "tmux", "zsh"}

    def session_status(self, project_id: str, slug: str, agent: str = AGENT_CLAUDE) -> dict[str, Any]:
        session_name = self._live_or_default_session(project_id, slug, agent)
        target = f"{session_name}:0.0"
        tmux_alive = self._tmux_session_exists(session_name)
        pane_command = self._pane_current_command(target) if tmux_alive else ""
        return {
            "session": session_name,
            "target": target,
            "tmux_alive": tmux_alive,
            "pane_command": pane_command,
            "agent_running": tmux_alive and not self._pane_is_idle_shell(pane_command),
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
        default_skills: Path | None = None,
    ) -> None:
        # Give the CLI a short chance to paint its input prompt, but do not
        # wait 90s: if the readiness heuristic misses a newer Claude/Codex UI
        # the paste should still happen quickly.
        time.sleep(2)
        self._wait_for_claude_ready(target, timeout=12.0)
        result = self._paste_prompt_to_target(project_root, slug, target, default_skills=default_skills)
        if not result.get("ok"):
            print(
                f"[web] paste prompt failed slug={slug}: {result.get('error', 'unknown error')}",
                flush=True,
            )
        self._watch_for_session_id(project_root, slug, cwd, agent, existing_filenames)

    def _paste_prompt_to_target(
        self,
        project_root: Path,
        slug: str,
        target: str,
        *,
        default_skills: Path | None = None,
    ) -> dict[str, Any]:
        prompt = _build_claude_prompt(project_root, slug, default_skills=default_skills)
        if not prompt:
            return {"ok": False, "error": "empty prompt", "target": target}
        ok, err = send_pane_text(target, prompt, submit=True)
        if ok:
            # Some agent CLIs render bracketed paste but require one more Enter
            # after the paste block; for CLIs that already submitted, this is
            # harmless.
            time.sleep(0.1)
            send_pane_key(target, "Enter")
            return {
                "ok": True,
                "target": target,
                "prompt_chars": len(prompt),
                "has_skills": "Default skills:\n---\n(none)" not in prompt,
            }
        return {"ok": False, "error": err or "paste failed", "target": target}


# --- Per-task run monitor ---------------------------------------------------

_MONITOR_POLL_SECONDS = 4.0
_MONITOR_CAPTURE_LINES = 160
# After a stop is reported, ignore further stops for this long - a guard
# against the working indicator flickering off for a single poll mid-turn.
_MONITOR_FIRE_COOLDOWN = 10.0
# Only treat the agent as *really* stopped once the "working" indicator has been
# gone for this many consecutive polls (~12s). This filters the brief flickers
# and short mid-turn pauses that otherwise caused spurious "stopped" pings - we
# only notify OpenClaw when the agent has genuinely finished and is waiting for
# input (i.e. actually needs your attention).
_MONITOR_STOP_CONFIRM_POLLS = 3

# Interactive agent CLIs (Claude Code / Codex) show an interrupt hint while
# actively working. When it disappears, the agent has stopped and is waiting
# for input - that running -> stopped edge is what the monitor fires on.
_AGENT_WORKING_RE = re.compile(r"esc to interrupt", re.IGNORECASE)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _TaskMonitor:
    """Background poller that watches whether the task's agent pane is working.

    Edge-triggered on the *running -> stopped* transition: when the agent was
    actively working and then stops (waiting for input), it emits an OpenClaw
    event. If the pane is already idle when monitoring is switched on, nothing
    fires until the agent runs and then stops again.
    """

    def __init__(
        self,
        manager: "TaskMonitorManager",
        project_root: Path,
        project_id: str,
        slug: str,
        pattern: str = "",
    ) -> None:
        self.manager = manager
        self.project_root = project_root
        self.project_id = project_id
        self.slug = slug
        self.pattern = pattern  # retained for API/JSON compat; not used to match
        self._stop = threading.Event()
        self._was_working = False
        self._idle_polls = 0
        self._initialized = False
        self._last_fire_ts = 0.0
        self.last_fired = ""
        self.last_match = ""
        self.thread = threading.Thread(
            target=self._loop, name=f"loom-monitor-{slug}", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_alive(self) -> bool:
        return self.thread.is_alive() and not self._stop.is_set()

    def _current_target(self) -> str:
        meta = read_meta(self.project_root, self.slug)
        if meta is None:
            return ""
        return (getattr(meta, "tmux_interview_target", "") or "").strip()

    def _loop(self) -> None:
        if self._stop.wait(_MONITOR_POLL_SECONDS):
            return
        while not self._stop.is_set():
            try:
                target = self._current_target()
                if target:
                    ok, text = capture_pane(target, _MONITOR_CAPTURE_LINES)
                    if ok:
                        working = bool(_AGENT_WORKING_RE.search(text or ""))
                        if not self._initialized:
                            # Baseline only - never fire on the first read, so
                            # enabling on an already-idle pane stays silent.
                            self._was_working = working
                            self._idle_polls = 0 if working else 1
                            self._initialized = True
                        elif working:
                            self._was_working = True
                            self._idle_polls = 0
                        else:
                            # Not working: only count it as a real stop once the
                            # indicator has stayed gone for several consecutive
                            # polls, so a one-off flicker or short mid-turn pause
                            # doesn't fire a spurious notification.
                            self._idle_polls += 1
                            if self._was_working and self._idle_polls >= _MONITOR_STOP_CONFIRM_POLLS:
                                self._was_working = False
                                self._fire(text or "")
            except Exception as exc:  # noqa: BLE001
                print(f"[monitor] {self.slug} loop error: {exc}", flush=True)
            if self._stop.wait(_MONITOR_POLL_SECONDS):
                break

    @staticmethod
    def _tail_snippet(text: str) -> str:
        lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
        # Carry a generous chunk of the final output so OpenClaw can actually
        # summarize what the agent just did, not just the last couple of lines.
        return "\n".join(lines[-60:]).strip()[-5000:]

    def _fire(self, pane_text: str) -> None:
        now = time.time()
        if now - self._last_fire_ts < _MONITOR_FIRE_COOLDOWN:
            return
        self._last_fire_ts = now
        self.last_fired = _iso_now()
        self.last_match = "stopped"
        snippet = self._tail_snippet(pane_text)
        print(f"[monitor] {self.slug} agent stopped -> openclaw", flush=True)
        try:
            self.manager.openclaw.emit(
                "agent-stopped",
                instruction=(
                    f"Loom: the agent in task {self.slug} just stopped and is "
                    f"waiting for input. Its recent terminal output is in "
                    f"data.tail below — summarize for me what it just did / "
                    f"finished, then I can reply to this message to continue it."
                ),
                project_root=self.project_root,
                task_slug=self.slug,
                data={"event": "agent-stopped", "tail": snippet},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[monitor] {self.slug} emit error: {exc}", flush=True)
        try:
            write_task_monitor(
                self.project_root,
                self.slug,
                enabled=True,
                pattern=self.pattern,
                last_fired=self.last_fired,
                last_match=self.last_match,
            )
        except Exception:  # noqa: BLE001
            pass


class TaskMonitorManager:
    """Owns per-task monitor threads keyed by ``(project_id, slug)``."""

    def __init__(self, openclaw_client: OpenClawClient) -> None:
        self.openclaw = openclaw_client
        self._monitors: dict[tuple[str, str], _TaskMonitor] = {}
        self._lock = threading.Lock()

    def enable(
        self,
        project_root: Path,
        project_id: str,
        slug: str,
        pattern: str,
    ) -> dict[str, Any]:
        pattern = (pattern or "").strip() or DEFAULT_MONITOR_PATTERN
        key = (project_id, slug)
        with self._lock:
            existing = self._monitors.pop(key, None)
            if existing is not None:
                existing.stop()
            mon = _TaskMonitor(self, project_root, project_id, slug, pattern)
            self._monitors[key] = mon
        mon.start()
        cur = read_task_monitor(project_root, slug)
        write_task_monitor(project_root, slug, enabled=True, pattern=pattern)
        return {
            "enabled": True,
            "running": True,
            "pattern": pattern,
            "default_pattern": DEFAULT_MONITOR_PATTERN,
            "last_fired": mon.last_fired or cur.get("last_fired", ""),
            "last_match": mon.last_match or cur.get("last_match", ""),
        }

    def disable(self, project_root: Path, project_id: str, slug: str) -> dict[str, Any]:
        key = (project_id, slug)
        with self._lock:
            mon = self._monitors.pop(key, None)
        if mon is not None:
            mon.stop()
        cur = read_task_monitor(project_root, slug)
        write_task_monitor(
            project_root,
            slug,
            enabled=False,
            pattern=cur.get("pattern", ""),
        )
        return self.status(project_root, project_id, slug)

    def status(self, project_root: Path, project_id: str, slug: str) -> dict[str, Any]:
        key = (project_id, slug)
        with self._lock:
            mon = self._monitors.get(key)
        cfg = read_task_monitor(project_root, slug)
        # Lazily resume a persisted-on monitor that isn't running yet (e.g.
        # after a server restart) so the toggle survives restarts.
        if (mon is None or not mon.is_alive()) and cfg.get("enabled"):
            return self.enable(project_root, project_id, slug, cfg.get("pattern", ""))
        running = bool(mon and mon.is_alive())
        return {
            "enabled": running,
            "running": running,
            "pattern": (mon.pattern if mon else cfg.get("pattern", "")) or DEFAULT_MONITOR_PATTERN,
            "default_pattern": DEFAULT_MONITOR_PATTERN,
            "last_fired": (mon.last_fired if (mon and mon.last_fired) else cfg.get("last_fired", "")),
            "last_match": (mon.last_match if (mon and mon.last_match) else cfg.get("last_match", "")),
        }

    def resume_enabled(self, projects: list[tuple[str, Path]]) -> int:
        """Start monitors for every task whose monitor.json has enabled=true.

        Called once at startup so the per-task Notify toggle survives a server
        restart without the user re-opening each task. *projects* is a list of
        ``(project_id, project_root)`` pairs.
        """
        started = 0
        for project_id, root in projects:
            try:
                metas = list_tasks(root)
            except Exception:  # noqa: BLE001
                continue
            for meta in metas:
                try:
                    cfg = read_task_monitor(root, meta.slug)
                    if cfg.get("enabled"):
                        self.enable(root, project_id, meta.slug, cfg.get("pattern", ""))
                        started += 1
                except Exception:  # noqa: BLE001
                    continue
        return started


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
    monitor_manager: "TaskMonitorManager | None" = None,
) -> type[BaseHTTPRequestHandler]:
    static_root = web_static_dir().resolve()
    required_token = auth_token.strip()
    pr = project_registry
    launch_root_resolved = launch_root.resolve()
    multi_ws = multi_project_workspace
    monitor_manager = monitor_manager or TaskMonitorManager(openclaw_client)

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

        @staticmethod
        def _kill_pty(proc, master) -> None:
            """Detach the tmux attach client + close its PTY (terminal stream)."""
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                os.close(master)
            except Exception:
                pass

        def _resolve_scope(self, parsed) -> tuple[Path | None, str | None]:
            qs = parse_qs(parsed.query or "")
            qpid = (qs.get("project") or [""])[0].strip()
            hp = (self.headers.get("X-Loom-Project") or "").strip()
            pid = qpid or hp or pr.default_project_id
            if not pid:
                return None, None
            pth = pr.get_path(pid)
            if pth is None:
                return None, None
            return pth, pid

        def _bad_project(self) -> None:
            st, b, h = _json_bytes(
                {"error": "unknown or invalid project; pass ?project=<id> or header X-Loom-Project"},
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
            self.send_header("WWW-Authenticate", 'Basic realm="Loom"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return False

        def _claude_session_summary(self, project_id: str, slug: str, meta) -> dict[str, Any]:
            agent = normalize_agent(meta.agent)
            root = pr.get_path(project_id)
            # A session can live under the current pane cwd (work/) OR under any
            # worktree where a pane was historically launched. Scan them all so
            # Resume finds every session this task ever spawned.
            candidates: list[Path] = []
            for c in (
                _task_pane_cwd(root, slug, meta),
                *list_task_worktrees(root, slug),
                task_root(root, slug),
            ):
                if c and c not in candidates:
                    candidates.append(c)
            files_by_id: dict[str, dict[str, Any]] = {}
            for cwd in candidates:
                for p in list_session_files(cwd, agent):
                    sid = session_id_from_path(p, agent)
                    if not sid:
                        continue
                    try:
                        stat = p.stat()
                    except OSError:
                        continue
                    prev = files_by_id.get(sid)
                    if prev is None or stat.st_mtime >= prev.get("mtime", 0.0):
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
                "pane_command": live["pane_command"],
                "agent_running": live["agent_running"],
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
                # Never let the browser reuse a stale index.html - it
                # references the versioned app.css/app.js, so the entry
                # document must always be fresh.
                h.append(("Cache-Control", "no-store, must-revalidate"))
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
                # Assets are cache-busted via ?v=... in index.html; still tell
                # the browser to revalidate so edits show up without a hard refresh.
                h.append(("Cache-Control", "no-cache"))
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
                        "skillsBundledRelative": "loom/skills/charlie_skills.md",
                        "skillsOptions": _available_skill_options(sk, root),
                    }
                )
                self._send(st, b, h)
                return

            if path == "/api/projects":
                if multi_ws:
                    pr.prune_redundant_parent_projects(launch_root_resolved)
                cur_id = (parse_qs(parsed.query or "").get("project") or [""])[0].strip()
                hdr = (self.headers.get("X-Loom-Project") or "").strip()
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

            if path == "/api/tmux/stream":
                import select as _select

                qs = parse_qs(parsed.query or "")
                target = (qs.get("target") or [""])[0].strip()
                try:
                    cols = int((qs.get("cols") or ["80"])[0] or 80)
                    rows = int((qs.get("rows") or ["24"])[0] or 24)
                except ValueError:
                    cols, rows = 80, 24
                if not validate_tmux_target(target):
                    st, b, h = _json_bytes({"ok": False, "error": "invalid target"}, 400)
                    self._send(st, b, h)
                    return
                proc, master = open_pane_attach(target, cols, rows)
                if proc is None or master is None:
                    st, b, h = _json_bytes(
                        {"ok": False, "error": "could not attach to pane"}, 502
                    )
                    self._send(st, b, h)
                    return
                self.close_connection = True
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    self.end_headers()
                except OSError:
                    self._kill_pty(proc, master)
                    return
                conn = self.connection
                try:
                    while True:
                        if proc.poll() is not None:
                            try:
                                while True:
                                    data = os.read(master, 65536)
                                    if not data:
                                        break
                                    self.wfile.write(data)
                            except OSError:
                                pass
                            break
                        r, _, _ = _select.select([master, conn], [], [], 30)
                        if conn in r:
                            try:
                                probe = conn.recv(4096)
                            except OSError:
                                probe = b""
                            if not probe:
                                break  # client closed the stream
                        if master in r:
                            try:
                                data = os.read(master, 65536)
                            except OSError:
                                break
                            if not data:
                                break
                            self.wfile.write(data)
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    self._kill_pty(proc, master)
                return

            if path == "/api/tasks":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                st, b, h = _json_bytes({"tasks": [m.to_dict() for m in list_tasks(root)]})
                self._send(st, b, h)
                return

            m_ki = re.match(r"^/api/tasks/([^/]+)/kernel-interview$", path)
            if m_ki:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m_ki.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes(read_kernel_interview(root, slug))
                self._send(st, b, h)
                return

            if path == "/api/kernel/plugins":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                ok, data = _run_kernel_helper(root, ["plugins"], timeout=30)
                if ok:
                    data["unverified"] = sorted(_kernel_unverified_set(root))
                st, b, h = _json_bytes(data, 200 if ok else 502)
                self._send(st, b, h)
                return

            if path == "/api/kernel/service":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                ok, data = _kernel_service_status_cached(root)
                st, b, h = _json_bytes(data, 200 if ok else 502)
                self._send(st, b, h)
                return

            if path == "/api/kernel/runs":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                task = (parse_qs(parsed.query or "").get("task") or [""])[0].strip()
                st, b, h = _json_bytes({"runs": _kernel_list_records(root, task or None)})
                self._send(st, b, h)
                return

            m_klog = re.match(r"^/api/kernel/runs/([^/]+)/log$", path)
            if m_klog:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                run_uid = m_klog.group(1)
                if not _KERNEL_ID_RE.match(run_uid):
                    st, b, h = _json_bytes({"error": "invalid run id"}, 400)
                    self._send(st, b, h)
                    return
                lp = _kernel_run_log_path(root, run_uid)
                text = ""
                if lp.is_file():
                    try:
                        # tail (~24KB) so a long build log stays cheap to poll
                        text = lp.read_bytes()[-24000:].decode("utf-8", "replace")
                    except OSError:
                        text = ""
                st, b, h = _json_bytes({"log": text})
                self._send(st, b, h)
                return

            m_kleader = re.match(r"^/api/kernel/runs/([^/]+)/leaderboard$", path)
            if m_kleader:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                run_uid = m_kleader.group(1)
                if not _KERNEL_ID_RE.match(run_uid):
                    st, b, h = _json_bytes({"error": "invalid run id"}, 400)
                    self._send(st, b, h)
                    return
                rec = _kernel_read_record(root, run_uid)
                if rec is None:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                rid = rec.get("run_id")
                if not rid:
                    st, b, h = _json_bytes({"ok": False, "error": "run not started yet"})
                    self._send(st, b, h)
                    return
                ok, status = _run_kernel_helper(root, ["status", "--run-id", rid], timeout=30)
                st, b, h = _json_bytes(
                    status if ok else {"ok": False, "error": (status or {}).get("error", "status failed")},
                    200 if ok else 502,
                )
                self._send(st, b, h)
                return

            m_kbest = re.match(r"^/api/kernel/runs/([^/]+)/best-kernel$", path)
            if m_kbest:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                run_uid = m_kbest.group(1)
                if not _KERNEL_ID_RE.match(run_uid):
                    st, b, h = _json_bytes({"error": "invalid run id"}, 400)
                    self._send(st, b, h)
                    return
                rec = _kernel_read_record(root, run_uid)
                rid = (rec or {}).get("run_id")
                if not rid:
                    st, b, h = _json_bytes({"ok": False, "error": "run not started yet"})
                    self._send(st, b, h)
                    return
                ok, data = _run_kernel_helper(root, ["best-kernel", "--run-id", rid], timeout=30)
                st, b, h = _json_bytes(data, 200 if ok else 502)
                self._send(st, b, h)
                return

            m_ksrc = re.match(r"^/api/kernel/runs/([^/]+)/kernel/([^/]+)$", path)
            if m_ksrc:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                job_id = m_ksrc.group(2)
                if not _KERNEL_ID_RE.match(job_id):
                    st, b, h = _json_bytes({"error": "invalid job id"}, 400)
                    self._send(st, b, h)
                    return
                ok, data = _run_kernel_helper(root, ["kernel-source", "--job-id", job_id], timeout=30)
                st, b, h = _json_bytes(data, 200 if ok else 502)
                self._send(st, b, h)
                return

            m_krun = re.match(r"^/api/kernel/runs/([^/]+)$", path)
            if m_krun:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                run_uid = m_krun.group(1)
                if not _KERNEL_ID_RE.match(run_uid):
                    st, b, h = _json_bytes({"error": "invalid run id"}, 400)
                    self._send(st, b, h)
                    return
                rec = _kernel_read_record(root, run_uid)
                if rec is None:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                if rec.get("run_id") and rec.get("state") == "running":
                    ok, status = _run_kernel_helper(
                        root, ["status", "--run-id", rec["run_id"]], timeout=30
                    )
                    if ok:
                        rec["status"] = status
                st, b, h = _json_bytes(rec)
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

            m_diff = re.match(r"^/api/tasks/([^/]+)/diff$", path)
            if m_diff:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m_diff.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                meta = detect_and_persist_worktree(root, slug) or meta
                worktrees = task_worktree_diffs(root, slug)
                st, b, h = _json_bytes(
                    {
                        "slug": slug,
                        "worktrees": worktrees,
                    }
                )
                self._send(st, b, h)
                return

            m_mon_get = re.match(r"^/api/tasks/([^/]+)/monitor$", path)
            if m_mon_get:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_mon_get.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                st, b, h = _json_bytes(monitor_manager.status(root, project_id, slug))
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
                        "task_root": str(task_root(root, slug)),
                        "plan_path": str(task_root(root, slug) / PLAN),
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

            if path == "/api/kernel/runs":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                plugin = str(body.get("plugin", "")).strip()
                target = str(body.get("target", "")).strip()
                model = str(body.get("model", "")).strip()
                slug = str(body.get("slug", "")).strip()
                # Shape is optional: an explicit override when given, otherwise an
                # agent proposes one at launch (see _launch_kernel_run).
                shape = body.get("shape")
                if not plugin or not target or not model:
                    st, b, h = _json_bytes(
                        {"error": "plugin, target and model are required"}, 400
                    )
                    self._send(st, b, h)
                    return
                run_uid = uuid.uuid4().hex[:12]
                cfg = {
                    "plugin": plugin,
                    "target": target,
                    "model": model,
                    "shape": shape if shape else None,
                    "n_agents": int(body.get("n_agents", 1) or 1),
                    "starter_mode": str(body.get("starter_mode", "none") or "none"),
                    "target_speedup": body.get("target_speedup"),
                    "auto_terminate": bool(body.get("auto_terminate", False)),
                    "poll_interval": int(body.get("poll_interval", 60) or 60),
                    "build": bool(body.get("build", False)),
                    "build_mode": bool(body.get("build_mode", False)),
                }
                if cfg["build_mode"]:
                    # correctness-first: stop at the first correct kernel; ignore speed
                    cfg["auto_terminate"] = True
                    if cfg["target_speedup"] is None:
                        cfg["target_speedup"] = 0
                rec = {
                    "id": run_uid,
                    "state": "launching",
                    "config": cfg,
                    "run_id": None,
                    "slug": slug or None,
                    "task_slug": slug or None,
                    "containers": [],
                    "created_at": time.time(),
                }
                _kernel_write_record(root, rec)
                threading.Thread(
                    target=_launch_kernel_run, args=(root, run_uid, cfg), daemon=True
                ).start()
                st, b, h = _json_bytes({"ok": True, "id": run_uid, "state": "launching"}, 202)
                self._send(st, b, h)
                return

            m_kstop = re.match(r"^/api/kernel/runs/([^/]+)/stop$", path)
            if m_kstop:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                run_uid = m_kstop.group(1)
                if not _KERNEL_ID_RE.match(run_uid):
                    st, b, h = _json_bytes({"error": "invalid run id"}, 400)
                    self._send(st, b, h)
                    return
                rec = _kernel_read_record(root, run_uid)
                if rec is None:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                result: dict[str, Any] = {"ok": True}
                if rec.get("run_id"):
                    _ok, result = _run_kernel_helper(
                        root, ["stop", "--run-id", rec["run_id"]], timeout=600
                    )
                rec["state"] = "stopped"
                _kernel_write_record(root, rec)
                st, b, h = _json_bytes({"ok": True, "stop": result, "run": rec})
                self._send(st, b, h)
                return

            if path == "/api/kernel/interview":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                msgs = body.get("messages", [])
                if not isinstance(msgs, list):
                    st, b, h = _json_bytes({"error": "messages must be a list"}, 400)
                    self._send(st, b, h)
                    return
                result = _kernel_interview_turn(msgs, str(body.get("model", "")))
                st, b, h = _json_bytes(result, 200 if result.get("ok") else 502)
                self._send(st, b, h)
                return

            if path == "/api/kernel/prepare":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                spec = body.get("spec") or {}
                if not isinstance(spec, dict):
                    st, b, h = _json_bytes({"error": "spec must be an object"}, 400)
                    self._send(st, b, h)
                    return
                slug = str(body.get("slug", "")).strip()
                prep_uid = uuid.uuid4().hex[:12]
                rec = {"id": prep_uid, "state": "resolving", "kind": "prepare",
                       "spec": spec, "slug": slug or None, "task_slug": slug or None,
                       "created_at": time.time()}
                _kernel_write_record(root, rec)
                threading.Thread(
                    target=_prepare_kernel_run, args=(root, prep_uid, spec), daemon=True
                ).start()
                st, b, h = _json_bytes({"ok": True, "id": prep_uid, "state": "resolving"}, 202)
                self._send(st, b, h)
                return

            if path == "/api/kernel/plugins/verify":
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                name = str(body.get("name", "")).strip()
                if not name:
                    st, b, h = _json_bytes({"error": "name required"}, 400)
                    self._send(st, b, h)
                    return
                _kernel_set_unverified(root, name, False)
                st, b, h = _json_bytes({"ok": True, "name": name, "verified": True})
                self._send(st, b, h)
                return

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
                skills_path = default_skills.resolve() if default_skills.is_file() else bundled_skills_path().resolve()
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
                    interview_model=(
                        str(body.get("interview_model", "")).strip()
                        or agent_default_model(raw_agent or AGENT_CLAUDE)
                    ),
                    agent=raw_agent or AGENT_CLAUDE,
                    kind={"kernel": "kernel", "aris": "aris"}.get(
                        str(body.get("kind", "")).strip().lower(), "agent"
                    ),
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
                    instruction=f"Loom task created: {meta.slug}",
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

            if path == "/api/tmux/send-literal":
                target = str(body.get("target", "")).strip()
                text = body.get("text", "")
                if not isinstance(text, str):
                    st, b, h = _json_bytes({"ok": False, "error": "text must be string"}, 400)
                    self._send(st, b, h)
                    return
                ok, msg = send_pane_literal(target, text)
                st, b, h = (
                    _json_bytes({"ok": True})
                    if ok
                    else _json_bytes({"ok": False, "error": msg}, 400)
                )
                self._send(st, b, h)
                return

            if path == "/api/tmux/scroll":
                target = str(body.get("target", "")).strip()
                direction = str(body.get("dir", "up")).strip()
                try:
                    lines = int(body.get("lines", 3))
                except (TypeError, ValueError):
                    lines = 3
                if not validate_tmux_target(target):
                    st, b, h = _json_bytes({"ok": False, "error": "invalid target"}, 400)
                    self._send(st, b, h)
                    return
                ok, msg = scroll_pane(target, direction, lines)
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
                mode = str(body.get("mode", "existing")).strip().lower()
                repo_url = str(body.get("repo_url", "")).strip()
                if not raw_path:
                    st, b, h = _json_bytes({"error": "path required"}, 400)
                    self._send(st, b, h)
                    return
                if mode in ("empty", "clone"):
                    # These create files on disk, so confine them to the launch
                    # directory tree (the modal promises nothing is written
                    # outside it). Registering an *existing* folder is unrestricted.
                    try:
                        dest = Path(raw_path).expanduser().resolve()
                    except OSError:
                        st, b, h = _json_bytes({"error": "invalid path"}, 400)
                        self._send(st, b, h)
                        return
                    if not _path_within(dest, launch_root_resolved):
                        st, b, h = _json_bytes(
                            {"error": f"new folders must be inside {launch_root_resolved}"}, 400
                        )
                        self._send(st, b, h)
                        return
                    if mode == "empty":
                        try:
                            dest.mkdir(parents=True, exist_ok=True)
                        except OSError as exc:
                            st, b, h = _json_bytes({"error": f"could not create folder: {exc}"}, 400)
                            self._send(st, b, h)
                            return
                    else:  # clone
                        if not repo_url:
                            st, b, h = _json_bytes({"error": "repo URL is required to clone"}, 400)
                            self._send(st, b, h)
                            return
                        ok_clone, msg_clone = _git_clone(repo_url, dest)
                        if not ok_clone:
                            st, b, h = _json_bytes({"error": msg_clone or "git clone failed"}, 400)
                            self._send(st, b, h)
                            return
                    raw_path = str(dest)
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
                result = claude_registry.start(root, project_id, slug, default_skills=default_skills)
                print(
                    f"[web] start claude slug={slug} ok={bool(result.get('ok'))} "
                    f"session={result.get('session', '')} target={result.get('target', '')}",
                    flush=True,
                )
                openclaw_client.emit(
                    "claude-start",
                    instruction=f"Loom Claude pane started for task {slug}",
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

            m_paste = re.match(r"^/api/tasks/([^/]+)/(?:claude|interview)/paste-prompt$", path)
            if m_paste:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_paste.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                result = claude_registry.paste_prompt(
                    root,
                    project_id,
                    slug,
                    default_skills=default_skills,
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
                    instruction=f"Loom Claude pane stopped for task {slug}",
                    project_root=root,
                    task_slug=slug,
                    data=result,
                )
                st, b, h = _json_bytes(result)
                self._send(st, b, h)
                return

            m_mon_post = re.match(r"^/api/tasks/([^/]+)/monitor$", path)
            if m_mon_post:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_mon_post.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                pattern = str(body.get("pattern", "")).strip()
                if pattern:
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        st, b, h = _json_bytes({"error": f"invalid regex: {exc}"}, 400)
                        self._send(st, b, h)
                        return
                result = monitor_manager.enable(root, project_id, slug, pattern)
                print(
                    f"[web] monitor enabled slug={slug} pattern={result.get('pattern', '')!r}",
                    flush=True,
                )
                st, b, h = _json_bytes(result)
                self._send(st, b, h)
                return

            m_send = re.match(r"^/api/tasks/([^/]+)/claude/send$", path)
            if m_send:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_send.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                meta = read_meta(root, slug)
                if not meta:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                target = (meta.tmux_interview_target or "").strip()
                if not target:
                    st, b, h = _json_bytes(
                        {"ok": False, "error": "no active Claude pane for this task"}, 409
                    )
                    self._send(st, b, h)
                    return
                text = body.get("text", "")
                if not isinstance(text, str) or not text:
                    st, b, h = _json_bytes({"ok": False, "error": "text required"}, 400)
                    self._send(st, b, h)
                    return
                submit = bool(body.get("submit", True))
                ok, msg = send_pane_text(target, text, submit=submit)
                print(
                    f"[web] inbound claude/send slug={slug} ok={ok} chars={len(text)}",
                    flush=True,
                )
                st, b, h = (
                    _json_bytes({"ok": True, "target": target})
                    if ok
                    else _json_bytes({"ok": False, "error": msg}, 400)
                )
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
                    instruction=f"Loom worktree created for task {slug}",
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

            m_wt_merge = re.match(r"^/api/tasks/([^/]+)/worktree/merge$", path)
            if m_wt_merge:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_wt_merge.group(1)
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
                        {"error": "worktree is not registered with this task"}, 400
                    )
                    self._send(st, b, h)
                    return
                result = merge_worktree_to_base(wt)
                print(
                    f"[web] merge worktree slug={slug} path={wt} "
                    f"ok={result.get('ok')} {result.get('branch')}->{result.get('base')}",
                    flush=True,
                )
                st, b, h = _json_bytes(result, 200 if result.get("ok") else 400)
                self._send(st, b, h)
                return

            m_review = re.match(r"^/api/tasks/([^/]+)/review$", path)
            if m_review:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_review.group(1)
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
                        {"error": "worktree is not registered with this task"}, 400
                    )
                    self._send(st, b, h)
                    return
                skills_text = ""
                sp = Path(meta.skills_path).expanduser() if meta.skills_path else None
                if sp is None or not sp.is_file():
                    sp = default_skills if default_skills.is_file() else bundled_skills_path()
                if sp.is_file():
                    skills_text = sp.read_text(encoding="utf-8", errors="replace")[:8000]
                result = _run_worktree_review(
                    wt,
                    str(body.get("rules", "")),
                    skills_text,
                    model=meta.interview_model or "",
                )
                print(
                    f"[web] review worktree slug={slug} path={wt} ok={result.get('ok')}",
                    flush=True,
                )
                st, b, h = _json_bytes(result, 200 if result.get("ok") else 502)
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
                    instruction=f"Loom pushed worktree branches for task {slug}",
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
                    instruction=f"Loom Claude pane resumed for task {slug}",
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

            m_ki_put = re.match(r"^/api/tasks/([^/]+)/kernel-interview$", path)
            if m_ki_put:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                slug = m_ki_put.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                messages = body.get("messages", [])
                if not isinstance(messages, list):
                    st, b, h = _json_bytes({"error": "messages must be a list"}, 400)
                    self._send(st, b, h)
                    return
                spec = body.get("spec")
                if not write_kernel_interview(root, slug, messages, spec):
                    st, b, h = _json_bytes({"error": "failed to save kernel interview"}, 500)
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
                skills_in = body.get("skills_path")
                if title is None and goal is None and agent_in is None and skills_in is None:
                    st, b, h = _json_bytes(
                        {"error": "supply title and/or general_goal and/or agent and/or skills_path"},
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
                if skills_in is not None:
                    try:
                        cand = Path(str(skills_in)).expanduser().resolve()
                    except OSError as exc:
                        st, b, h = _json_bytes({"error": f"invalid skills_path: {exc}"}, 400)
                        self._send(st, b, h)
                        return
                    if not cand.is_file() or cand.suffix.lower() != ".md":
                        st, b, h = _json_bytes({"error": "skills_path must be a markdown file"}, 400)
                        self._send(st, b, h)
                        return
                    update_meta(root, slug, skills_path=str(cand))
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

            m_krun_del = re.match(r"^/api/kernel/runs/([^/]+)$", path)
            if m_krun_del:
                root, _pid = self._resolve_scope(parsed)
                if root is None:
                    self._bad_project()
                    return
                run_uid = m_krun_del.group(1)
                if not _KERNEL_ID_RE.match(run_uid):
                    st, b, h = _json_bytes({"error": "invalid run id"}, 400)
                    self._send(st, b, h)
                    return
                rec = _kernel_read_record(root, run_uid)
                if rec is None:
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                # A live run must be stopped before its record can be deleted, so
                # we never orphan running agent containers.
                if rec.get("state") in ("launching", "running", "resolving"):
                    st, b, h = _json_bytes(
                        {"error": "stop the run before deleting it"}, 409
                    )
                    self._send(st, b, h)
                    return
                _kernel_delete_record(root, run_uid)
                st, b, h = _json_bytes({"ok": True, "id": run_uid})
                self._send(st, b, h)
                return

            m_mon_del = re.match(r"^/api/tasks/([^/]+)/monitor$", path)
            if m_mon_del:
                root, project_id = self._resolve_scope(parsed)
                if root is None or project_id is None:
                    self._bad_project()
                    return
                slug = m_mon_del.group(1)
                if not _SLUG_RE.match(slug):
                    st, b, h = _json_bytes({"error": "invalid slug"}, 400)
                    self._send(st, b, h)
                    return
                if not read_meta(root, slug):
                    st, b, h = _json_bytes({"error": "not found"}, 404)
                    self._send(st, b, h)
                    return
                result = monitor_manager.disable(root, project_id, slug)
                print(f"[web] monitor disabled slug={slug}", flush=True)
                st, b, h = _json_bytes(result)
                self._send(st, b, h)
                return

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
                    instruction=f"Loom worktree removed for task {slug}",
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
    os.environ["LOOM_PROJECT_ROOT"] = str(project_root)
    web_project_registry = WebProjectRegistry()
    if multi_project_workspace:
        web_project_registry.prune_redundant_parent_projects(project_root)
    claude_registry = ClaudeRegistry()
    openclaw_client = OpenClawClient(openclaw_config)
    monitor_manager = TaskMonitorManager(openclaw_client)
    # A launch/prepare runs in a background thread that does NOT survive a server
    # restart, leaving its run record stuck at "launching"/"resolving" forever.
    # On startup, no launch can be in flight, so sweep any such records to error.
    _sweep_roots = {project_root}
    try:
        for _p in web_project_registry.list_projects():
            _pp = _p.get("path")
            if _pp:
                _sweep_roots.add(Path(_pp))
    except Exception:  # noqa: BLE001
        pass
    _swept = _sweep_stale_kernel_runs(list(_sweep_roots))
    if _swept:
        print(f"  Swept {_swept} stale kernel run(s) (launching/resolving -> error)", flush=True)
    # Resume per-task run monitors that were left enabled, so the Notify toggle
    # survives a server restart without re-opening each task.
    _monitor_projects: list[tuple[str, Path]] = []
    try:
        for _p in web_project_registry.list_projects():
            _pid, _pp = _p.get("id"), _p.get("path")
            if _pid and _pp:
                _monitor_projects.append((str(_pid), Path(_pp)))
    except Exception:  # noqa: BLE001
        pass
    _resumed = monitor_manager.resume_enabled(_monitor_projects)
    if _resumed:
        print(f"  Resumed {_resumed} enabled run-monitor(s)", flush=True)
    sk = default_skills if default_skills.is_file() else bundled_skills_path().resolve()
    handler = make_handler(
        web_project_registry,
        project_root,
        sk,
        claude_registry,
        openclaw_client,
        auth_token,
        multi_project_workspace=multi_project_workspace,
        monitor_manager=monitor_manager,
    )
    server = ThreadingHTTPServer((host, port), handler)
    rud_root = project_root / ".RUD"
    print("", flush=True)
    print("Loom", flush=True)
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
        instruction=f"Loom web started for project {project_root}",
        project_root=project_root,
        data={"url": f"http://{host}:{port}/", "taskRoot": str(rud_root)},
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
