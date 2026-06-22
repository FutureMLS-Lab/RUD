"""Persisted multi-project registry for `loom web`."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = Path.home() / ".loom" / "web-projects.json"


class WebProjectRegistry:
    """Thread-safe JSON registry of project roots (paths on disk)."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = (path or DEFAULT_REGISTRY_PATH).expanduser()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "version": 1,
            "projects": [],
            "defaultProjectId": "",
        }
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict) and isinstance(raw.get("projects"), list):
            self._data = {**self._data, **raw}

    def _save_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for p in self._data["projects"]:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("id", ""))
                raw_path = str(p.get("path", ""))
                try:
                    resolved = str(Path(raw_path).expanduser().resolve())
                except OSError:
                    resolved = raw_path
                out.append(
                    {
                        "id": pid,
                        "name": str(p.get("name") or Path(raw_path).name),
                        "path": resolved,
                        "defaultWorkDirs": list(p.get("defaultWorkDirs") or []),
                    }
                )
            return out

    def ensure_project(self, path: Path, *, name: str | None = None) -> str:
        """Register `path` if missing; return its id."""
        path = path.resolve()
        with self._lock:
            for p in self._data["projects"]:
                if not isinstance(p, dict):
                    continue
                try:
                    if Path(str(p.get("path", ""))).expanduser().resolve() == path:
                        if name:
                            p["name"] = name
                        self._save_unlocked()
                        return str(p["id"])
                except OSError:
                    continue
            pid = uuid.uuid4().hex[:12]
            self._data["projects"].append(
                {
                    "id": pid,
                    "name": name or path.name,
                    "path": str(path),
                    "defaultWorkDirs": [],
                }
            )
            if not self._data.get("defaultProjectId"):
                self._data["defaultProjectId"] = pid
            self._save_unlocked()
            return pid

    def add_by_path(self, raw_path: str) -> tuple[str | None, str | None]:
        """Returns (new_id, error_message)."""
        try:
            path = Path(raw_path).expanduser().resolve()
        except OSError:
            return None, "invalid path"
        if not path.is_dir():
            return None, "path is not a directory"
        with self._lock:
            for p in self._data["projects"]:
                if not isinstance(p, dict):
                    continue
                try:
                    if Path(str(p.get("path", ""))).expanduser().resolve() == path:
                        return None, "project already registered"
                except OSError:
                    continue
            pid = uuid.uuid4().hex[:12]
            self._data["projects"].append(
                {
                    "id": pid,
                    "name": path.name,
                    "path": str(path),
                    "defaultWorkDirs": [],
                }
            )
            if not self._data.get("defaultProjectId"):
                self._data["defaultProjectId"] = pid
            self._save_unlocked()
            return pid, None

    def prune_redundant_parent_projects(self, launch_root: Path) -> None:
        """Remove a project row that is exactly *launch_root* when another registered path is a strict subdirectory.

        Intended for **multi-project workspace** mode (``loom web --projects``): the launch directory
        is only a container for several git checkouts, so a registry row for the parent path is redundant
        once a child repo is registered. Callers must not invoke this in single-project mode, where the
        launch directory may legitimately be the only registered project root.
        """
        try:
            root = launch_root.expanduser().resolve()
        except OSError:
            return
        with self._lock:
            projects = [p for p in self._data["projects"] if isinstance(p, dict)]
            parent_ids: list[str] = []
            for p in projects:
                try:
                    rp = Path(str(p.get("path", ""))).expanduser().resolve()
                except OSError:
                    continue
                if rp == root:
                    pid = str(p.get("id", "")).strip()
                    if pid:
                        parent_ids.append(pid)
            if not parent_ids:
                return
            has_strict_child = False
            for p in projects:
                try:
                    rp = Path(str(p.get("path", ""))).expanduser().resolve()
                except OSError:
                    continue
                if rp == root:
                    continue
                try:
                    rp.relative_to(root)
                    has_strict_child = True
                    break
                except ValueError:
                    continue
            if not has_strict_child:
                return
            ids = set(parent_ids)
            self._data["projects"] = [
                p for p in self._data["projects"] if not (isinstance(p, dict) and str(p.get("id", "")) in ids)
            ]
            if str(self._data.get("defaultProjectId", "")) in ids:
                remaining = [p for p in self._data["projects"] if isinstance(p, dict)]
                self._data["defaultProjectId"] = str(remaining[0]["id"]) if remaining else ""
            self._save_unlocked()

    def move(self, pid: str, direction: str) -> tuple[bool, str]:
        """Move a project row in the persisted display order."""
        direction = direction.strip().lower()
        with self._lock:
            projects = self._data["projects"]
            idx = next(
                (i for i, p in enumerate(projects) if isinstance(p, dict) and str(p.get("id")) == pid),
                -1,
            )
            if idx < 0:
                return False, "project not found"
            if direction == "front":
                new_idx = 0
            elif direction == "back":
                new_idx = len(projects) - 1
            elif direction == "left":
                new_idx = max(0, idx - 1)
            elif direction == "right":
                new_idx = min(len(projects) - 1, idx + 1)
            else:
                return False, "invalid direction"
            if new_idx == idx:
                return True, ""
            item = projects.pop(idx)
            projects.insert(new_idx, item)
            self._save_unlocked()
            return True, ""

    def reorder(self, ids: list[str]) -> tuple[bool, str]:
        """Replace the persisted project display order with the provided ids."""
        ids = [str(x).strip() for x in ids if str(x).strip()]
        with self._lock:
            projects = [p for p in self._data["projects"] if isinstance(p, dict)]
            known = [str(p.get("id", "")).strip() for p in projects]
            if set(ids) != set(known) or len(ids) != len(known):
                return False, "ids must contain every registered project exactly once"
            by_id = {str(p.get("id", "")).strip(): p for p in projects}
            self._data["projects"] = [by_id[pid] for pid in ids]
            self._save_unlocked()
            return True, ""

    def remove(self, pid: str) -> tuple[bool, str]:
        with self._lock:
            projects = [p for p in self._data["projects"] if isinstance(p, dict)]
            if not projects:
                return False, "project not found"
            idx = next((i for i, p in enumerate(self._data["projects"]) if str(p.get("id")) == pid), -1)
            if idx < 0:
                return False, "project not found"
            self._data["projects"].pop(idx)
            if str(self._data.get("defaultProjectId", "")) == pid:
                remaining = [p for p in self._data["projects"] if isinstance(p, dict)]
                self._data["defaultProjectId"] = str(remaining[0]["id"]) if remaining else ""
            self._save_unlocked()
            return True, ""

    @property
    def persist_path(self) -> Path:
        return self._path

    def get_path(self, pid: str) -> Path | None:
        with self._lock:
            for p in self._data["projects"]:
                if not isinstance(p, dict):
                    continue
                if str(p.get("id")) != pid:
                    continue
                try:
                    path = Path(str(p.get("path", ""))).expanduser().resolve()
                except OSError:
                    return None
                if path.is_dir():
                    return path
                return None
        return None

    def get_default_work_dirs(self, pid: str) -> list[Path] | None:
        """Return explicit dirs for this project, or None to use server CLI defaults."""
        with self._lock:
            for p in self._data["projects"]:
                if not isinstance(p, dict) or str(p.get("id")) != pid:
                    continue
                raw = p.get("defaultWorkDirs") or []
                if not isinstance(raw, list):
                    return None
                out: list[Path] = []
                for item in raw:
                    if not str(item).strip():
                        continue
                    try:
                        q = Path(str(item)).expanduser().resolve()
                    except OSError:
                        continue
                    if q.is_dir():
                        out.append(q)
                return out if out else None
        return None

    @property
    def default_project_id(self) -> str:
        with self._lock:
            return str(self._data.get("defaultProjectId") or "")

    def set_default(self, pid: str) -> bool:
        with self._lock:
            if not any(str(p.get("id")) == pid for p in self._data["projects"] if isinstance(p, dict)):
                return False
            self._data["defaultProjectId"] = pid
            self._save_unlocked()
            return True
