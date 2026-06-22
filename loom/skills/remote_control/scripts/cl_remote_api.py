#!/usr/bin/env python3
"""Shared HTTP helpers for remote_control scripts: auth, project discovery, scoped paths."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = os.environ.get("LOOM_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
TOKEN = os.environ.get("LOOM_WEB_AUTH_TOKEN", "")

_cached_projects: dict | None = None
_cached_project_id: str | None = None


def _request_json(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    """GET/POST JSON to BASE+path (path must start with /)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            err = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            err = body or e.reason
        raise SystemExit(f"HTTP {e.code} {path}: {err}") from e


def fetch_projects() -> dict:
    """GET /api/projects (never pass ?project=)."""
    global _cached_projects
    if _cached_projects is None:
        _cached_projects = _request_json("/api/projects")
    return _cached_projects


def invalidate_project_cache() -> None:
    global _cached_projects, _cached_project_id
    _cached_projects = None
    _cached_project_id = None


def _project_rows(data: dict) -> list[dict]:
    pl = data.get("projects")
    if not isinstance(pl, list):
        return []
    return [p for p in pl if isinstance(p, dict) and str(p.get("id", "")).strip()]


def resolve_project_id() -> str:
    """Pick project id for ?project= on /api/project and /api/tasks.

    Order: LOOM_PROJECT_ID (must exist), LOOM_PROJECT_PATH (resolved path match),
    single registered project, else defaultProjectId / currentProjectId from server,
    else exit with instructions.
    """
    global _cached_project_id
    if _cached_project_id is not None:
        return _cached_project_id

    explicit = os.environ.get("LOOM_PROJECT_ID", "").strip()
    path_hint = os.environ.get("LOOM_PROJECT_PATH", "").strip()
    data = fetch_projects()
    projects = _project_rows(data)
    by_id = {str(p["id"]): p for p in projects}

    if explicit:
        if explicit not in by_id:
            choices = ", ".join(sorted(by_id)) or "(none)"
            raise SystemExit(
                f"LOOM_PROJECT_ID={explicit!r} is not registered. "
                f"GET {BASE}/api/projects ids: {choices}"
            )
        _cached_project_id = explicit
        return explicit

    if path_hint:
        try:
            want = Path(path_hint).expanduser().resolve()
        except OSError as exc:
            raise SystemExit(f"LOOM_PROJECT_PATH invalid: {exc}") from exc
        for p in projects:
            try:
                rp = Path(str(p.get("path", ""))).expanduser().resolve()
            except OSError:
                continue
            if rp == want:
                _cached_project_id = str(p["id"])
                return _cached_project_id
        raise SystemExit(
            f"LOOM_PROJECT_PATH={path_hint!r} did not match any entry in GET {BASE}/api/projects"
        )

    if not projects:
        raise SystemExit(
            f"No projects registered at {BASE}. Add a repo in the web UI or POST /api/projects."
        )

    if len(projects) == 1:
        _cached_project_id = str(projects[0]["id"])
        return _cached_project_id

    for key in ("currentProjectId", "defaultProjectId"):
        pid = str(data.get(key) or "").strip()
        if pid in by_id:
            _cached_project_id = pid
            return pid

    lines = "; ".join(f"{p['id']} ({p.get('name', '')})" for p in projects)
    raise SystemExit(
        "Multiple projects registered; set LOOM_PROJECT_ID or LOOM_PROJECT_PATH. "
        f"Available: {lines}"
    )


def with_project(path: str, project_id: str) -> str:
    """Append ?project= / &project= for task and project-scope APIs (matches web UI rules)."""
    if not project_id:
        return path
    if path.startswith("/api/projects"):
        return path
    if not (
        path.startswith("/api/tasks")
        or path == "/api/project"
        or path.startswith("/api/project?")
    ):
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}project={urllib.parse.quote(project_id, safe='')}"


def scoped(path: str) -> str:
    return with_project(path, resolve_project_id())


def get_json(path: str) -> dict:
    return _request_json(scoped(path))


def post_json(path: str, payload: dict | None) -> dict:
    return _request_json(scoped(path), method="POST", payload=payload)


def post_json_raw(path: str, payload: dict) -> dict:
    """POST without ?project= (global APIs such as /api/tmux/send-text)."""
    return _request_json(path, method="POST", payload=payload)


def get_json_raw(path: str) -> dict:
    """GET without ?project= (e.g. /api/projects, /api/tmux/capture?...)."""
    return _request_json(path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="loom remote_control helpers")
    ap.add_argument(
        "--print-project-id",
        action="store_true",
        help="Print resolved project id for shell (uses LOOM_* env)",
    )
    args = ap.parse_args()
    if args.print_project_id:
        print(resolve_project_id())
