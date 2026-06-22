"""`.RUD` task storage and helpers.

Layout per project root::

    <project>/.RUD/
        NOTES.md            # project-scoped scratchpad (one file per project)
        task-order.json
        <slug>/
            task.json       # task metadata
            PLAN.md         # the only per-task markdown file - Claude reads
                            # and rewrites it; the user edits it via the
                            # PLAN.md tab and the embedded view on Claude tab
            work/<repo>/    # auto-created git worktree (branch zhongzhu/<slug>)

There is no worker / evaluator / runner anymore - the user drives Claude
themselves via the tmux pane.  We do track which Claude session UUIDs each
task has spawned so we can resume a session after tmux dies.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loom.paths import bundled_skills_path

RUD_DIR = ".RUD"
WORK_SUBDIR = "work"

PLAN = "PLAN.md"
NOTES = "NOTES.md"
META = "task.json"
TASK_ORDER = "task-order.json"

# Only PLAN.md is editable through the per-task template API now.  NOTES.md
# lives at the project root and has its own dedicated endpoint.
ALLOWED_TEMPLATE_NAMES = frozenset({PLAN})

# Supported agent CLIs that can drive a task's tmux pane.
AGENT_CLAUDE = "claude"
AGENT_CODEX = "codex"
SUPPORTED_AGENTS = frozenset({AGENT_CLAUDE, AGENT_CODEX})


def normalize_agent(name: str | None) -> str:
    """Return a valid agent name, defaulting to ``claude``."""
    s = (name or "").strip().lower()
    return s if s in SUPPORTED_AGENTS else AGENT_CLAUDE


def agent_label(name: str) -> str:
    """Display label for an agent name (e.g. ``"Claude"`` / ``"Codex"``)."""
    return {AGENT_CLAUDE: "Claude", AGENT_CODEX: "Codex"}.get(
        normalize_agent(name), "Claude"
    )


def agent_default_model(name: str) -> str:
    """Default model string to pass to the CLI for a given agent."""
    return {
        AGENT_CLAUDE: "claude-opus-4-8",
        AGENT_CODEX: "",  # codex falls back to its own ~/.codex/config.toml default
    }.get(normalize_agent(name), "")


# Model strings that were a previous global default and are not selectable via
# any UI - so they are vestigial, never an intentional per-task choice. Treat
# them as "use the current default" so old tasks start/resume on the new model.
_LEGACY_DEFAULT_MODELS = {"claude-sonnet-4-6"}


def _upgrade_legacy_model(model: str, agent: str = AGENT_CLAUDE) -> str:
    m = (model or "").strip()
    if m in _LEGACY_DEFAULT_MODELS:
        return agent_default_model(agent) or m
    return m


def build_agent_command(
    agent: str,
    model: str = "",
    resume_session_id: str = "",
) -> list[str]:
    """Build the CLI argv that should be exec'd in the tmux pane for *agent*.

    - Claude:  ``claude --model M --dangerously-skip-permissions --effort max [--resume ID]``
    - Codex:   ``codex resume ID`` (resume) or ``codex`` (fresh), optionally with ``-c model=…``
    """
    agent = normalize_agent(agent)
    if agent == AGENT_CODEX:
        if resume_session_id:
            cmd: list[str] = ["codex", "resume", resume_session_id]
        else:
            cmd = ["codex"]
        if model.strip():
            cmd += ["-c", f"model={model.strip()}"]
        return cmd
    # Claude
    cmd = [
        "claude",
        "--model",
        model or agent_default_model(AGENT_CLAUDE),
        "--dangerously-skip-permissions",
        "--effort",
        "max",
    ]
    if resume_session_id:
        cmd += ["--resume", resume_session_id]
    return cmd

_TASK_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_DUBIOUS_OWNERSHIP_RE = re.compile(r"detected dubious ownership in repository at '([^']+)'")


def rud_root(project_root: Path) -> Path:
    return (project_root / RUD_DIR).resolve()


def task_root(project_root: Path, slug: str) -> Path:
    """``<project>/.RUD/<slug>/`` - task name becomes slug (e.g. xorl1)."""
    return (rud_root(project_root) / slug).resolve()


def list_task_worktrees(project_root: Path, slug: str) -> list[Path]:
    """Return every git work-tree directory under ``<task>/work/`` sorted by name.

    A child counts only if it looks like a git work tree (``.git`` exists
    as a file or directory, OR ``git rev-parse`` agrees), so leftover
    non-git folders won't confuse the UI.
    """
    work = task_root(project_root, slug) / WORK_SUBDIR
    if not work.is_dir():
        return []
    out: list[Path] = []
    try:
        children = sorted(work.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    for child in children:
        try:
            if not child.is_dir() or child.name.startswith("."):
                continue
        except OSError:
            continue
        dotgit = child / ".git"
        if dotgit.exists() or git_toplevel(child) is not None:
            out.append(child.resolve())
    return out


def task_worktree_path(project_root: Path, slug: str) -> Path | None:
    """Primary worktree (= first entry in ``list_task_worktrees``)."""
    wts = list_task_worktrees(project_root, slug)
    return wts[0] if wts else None


def _detect_branch_in(wt: Path) -> str:
    """Best-effort current branch for a worktree directory."""
    if not wt.is_dir():
        return ""
    ok, out, _ = _git(["branch", "--show-current"], wt, timeout=10)
    return out.strip() if ok else ""


def detect_and_persist_worktree(project_root: Path, slug: str) -> TaskMeta | None:
    """Sync ``meta.worktrees`` / ``meta.branches`` with what's on disk.

    Behaviour:
    - drops entries whose directory no longer exists;
    - keeps the existing list order so the *primary* (claude-pane cwd)
      doesn't shuffle around unexpectedly;
    - appends any newly discovered worktree directories in alphabetical
      order (this is what back-fills tasks created before the
      auto-worktree feature, or tasks where the user did
      ``git worktree add`` manually);
    - recomputes per-worktree branch names with ``git branch --show-current``
      when missing.
    """
    meta = read_meta(project_root, slug)
    if meta is None:
        return None

    on_disk = list_task_worktrees(project_root, slug)
    on_disk_paths = {str(p) for p in on_disk}

    ordered: list[str] = []
    branches: list[str] = []
    # Preserve meta's existing ordering, dropping disappearing paths.
    for i, p in enumerate(meta.worktrees):
        if p in on_disk_paths and p not in ordered:
            ordered.append(p)
            branches.append(meta.branches[i] if i < len(meta.branches) else "")
    # Append disk-only newcomers in alphabetical order.
    for p_obj in on_disk:
        s = str(p_obj)
        if s not in ordered:
            ordered.append(s)
            branches.append("")
    # Fill in missing branch labels with a fresh `git branch --show-current`.
    for i, p in enumerate(ordered):
        if not branches[i]:
            try:
                branches[i] = _detect_branch_in(Path(p))
            except OSError:
                branches[i] = ""

    primary_wt = ordered[0] if ordered else ""
    primary_br = branches[0] if branches else ""
    if (
        ordered == meta.worktrees
        and branches == meta.branches
        and primary_wt == meta.worktree_path
        and primary_br == meta.branch
    ):
        return meta
    return update_meta(
        project_root,
        slug,
        worktree_path=primary_wt,
        branch=primary_br,
        worktrees=ordered,
        branches=branches,
    ) or meta


def remove_task_worktree(project_root: Path, slug: str, worktree: Path) -> tuple[bool, str]:
    """Delete a single worktree from disk + git registry, then resync meta.

    Tries the clean ``git worktree remove`` path first, falls back to
    ``--force`` if Git complains (typical when the worktree has
    uncommitted changes or has been moved).  Returns ``(ok, message)``.
    """
    try:
        worktree = worktree.expanduser().resolve()
    except OSError as exc:
        return False, f"invalid path: {exc}"
    meta = read_meta(project_root, slug)
    if meta is None:
        return False, "task not found"
    if str(worktree) not in meta.worktrees:
        return False, "worktree is not registered with this task"
    # Run git from the worktree itself so it can locate its parent repo.
    if worktree.is_dir():
        ok, _out, err = _git(["worktree", "remove", str(worktree)], worktree)
        if not ok:
            ok, _out, err = _git(
                ["worktree", "remove", "--force", str(worktree)],
                worktree,
            )
        if not ok:
            # Worktree gone but registration lingers? Try a hard delete.
            try:
                shutil.rmtree(worktree)
            except OSError:
                return False, err or "git worktree remove failed"
    detect_and_persist_worktree(project_root, slug)
    return True, "worktree removed"


def slugify(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.ASCII)
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s[:80] or "task"


def ensure_unique_slug(project_root: Path, base: str) -> str:
    root = rud_root(project_root)
    if not (root / base).exists():
        return base
    n = 2
    while (root / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_default_skills(skills_path: Path) -> str:
    if not skills_path.is_file():
        return ""
    return skills_path.read_text(encoding="utf-8", errors="replace")


def package_templates_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "templates"


def copy_default_plan(dest_dir: Path, overwrite: bool = False) -> None:
    """Seed PLAN.md from the shipped template if missing."""
    src = package_templates_dir() / PLAN
    out = dest_dir / PLAN
    if out.exists() and not overwrite:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        out.write_text("", encoding="utf-8")


@dataclass
class TaskMeta:
    slug: str
    title: str
    general_goal: str
    created_at: str
    updated_at: str
    skills_path: str = ""
    interview_model: str = "claude-opus-4-8"
    tmux_interview_target: str = ""
    # Which CLI drives this task's tmux pane.  Defaults to claude so
    # legacy task.json files (no `agent` key) keep working unchanged.
    agent: str = AGENT_CLAUDE
    # Task kind: "agent" (Claude/Codex deep-interview task, default) or
    # "kernel" (Kernel Lab task — the task view renders the Kernel Lab UI).
    kind: str = "agent"
    # ``worktree_path`` and ``branch`` are the *primary* worktree (also
    # mirrored at ``worktrees[0]`` / ``branches[0]``); the Claude pane
    # always opens there.  Use the lists when iterating over all the
    # worktrees a task owns.
    worktree_path: str = ""
    branch: str = ""
    worktrees: list[str] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)
    # UUIDs of every Claude Code session this task has launched (most
    # recent last).  We use these to offer a "Resume" button when tmux is
    # killed.
    claude_session_ids: list[str] = field(default_factory=list)
    # Per-worktree fork point captured when the worktree was created:
    #   {"<resolved worktree path>": {"commit": "<sha>", "branch": "<source branch>"}}
    # The diff uses this as its base ("what changed since this worktree was
    # branched from its source"); worktrees without an entry fall back to
    # ``_detect_base_ref`` (origin/main, ...).
    worktree_bases: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "general_goal": self.general_goal,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "skills_path": self.skills_path,
            "interview_model": self.interview_model,
            "tmux_interview_target": self.tmux_interview_target,
            "agent": self.agent,
            "kind": self.kind,
            "worktree_path": self.worktree_path,
            "branch": self.branch,
            "worktrees": list(self.worktrees),
            "branches": list(self.branches),
            "worktree_bases": {k: dict(v) for k, v in self.worktree_bases.items()},
            "claude_session_ids": list(self.claude_session_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMeta:
        # Tolerate legacy fields (work_dirs, tmux_runner_target, ...) by
        # only reading what we still care about.
        raw_sessions = data.get("claude_session_ids") or []
        sessions: list[str] = []
        if isinstance(raw_sessions, list):
            seen: set[str] = set()
            for s in raw_sessions:
                sid = str(s).strip()
                if sid and sid not in seen:
                    seen.add(sid)
                    sessions.append(sid)
        raw_wts = data.get("worktrees")
        wts: list[str] = []
        if isinstance(raw_wts, list):
            for p in raw_wts:
                s = str(p).strip()
                if s and s not in wts:
                    wts.append(s)
        raw_brs = data.get("branches")
        brs: list[str] = []
        if isinstance(raw_brs, list):
            brs = [str(b) for b in raw_brs]
        raw_bases = data.get("worktree_bases")
        bases: dict[str, dict[str, str]] = {}
        if isinstance(raw_bases, dict):
            for k, v in raw_bases.items():
                if isinstance(v, dict):
                    bases[str(k)] = {
                        "commit": str(v.get("commit", "")),
                        "branch": str(v.get("branch", "")),
                    }
        # Pre-list-era task.json: lift the single worktree_path / branch
        # into the new list so the rest of the code can treat everything
        # uniformly.
        legacy_wt = str(data.get("worktree_path", "")).strip()
        legacy_br = str(data.get("branch", "")).strip()
        if not wts and legacy_wt:
            wts = [legacy_wt]
            brs = [legacy_br]
        # Make sure branches has the same length as worktrees (pad / trim).
        if len(brs) < len(wts):
            brs = brs + [""] * (len(wts) - len(brs))
        elif len(brs) > len(wts):
            brs = brs[: len(wts)]
        primary_wt = wts[0] if wts else legacy_wt
        primary_br = brs[0] if brs else legacy_br
        return cls(
            slug=str(data["slug"]),
            title=str(data.get("title", "")),
            general_goal=str(data.get("general_goal", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            skills_path=str(data.get("skills_path", "")),
            interview_model=_upgrade_legacy_model(
                str(data.get("interview_model", "claude-opus-4-8")),
                normalize_agent(data.get("agent")),
            ),
            tmux_interview_target=str(data.get("tmux_interview_target", "")),
            agent=normalize_agent(data.get("agent")),
            kind=str(data.get("kind", "agent") or "agent"),
            worktree_path=primary_wt,
            branch=primary_br,
            worktrees=wts,
            branches=brs,
            worktree_bases=bases,
            claude_session_ids=sessions,
        )


def _meta_path(project_root: Path, slug: str) -> Path:
    return task_root(project_root, slug) / META


def _task_order_path(project_root: Path) -> Path:
    return rud_root(project_root) / TASK_ORDER


def _read_task_order(project_root: Path) -> list[str]:
    path = _task_order_path(project_root)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if _TASK_SLUG_RE.match(str(x))]


def _write_task_order(project_root: Path, slugs: list[str]) -> None:
    root = rud_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    path = _task_order_path(project_root)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(slugs, indent=2), encoding="utf-8")
    tmp.replace(path)


def _insert_task_order_front(project_root: Path, slug: str) -> None:
    order = [s for s in _read_task_order(project_root) if s != slug]
    _write_task_order(project_root, [slug, *order])


def _remove_task_from_order(project_root: Path, slug: str) -> None:
    order = [s for s in _read_task_order(project_root) if s != slug]
    if order:
        _write_task_order(project_root, order)
        return
    try:
        _task_order_path(project_root).unlink()
    except (FileNotFoundError, OSError):
        pass


def write_meta(project_root: Path, meta: TaskMeta) -> None:
    path = _meta_path(project_root, meta.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)


def read_meta(project_root: Path, slug: str) -> TaskMeta | None:
    path = _meta_path(project_root, slug)
    if not path.is_file():
        return None
    try:
        return TaskMeta.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def create_task(
    project_root: Path,
    title: str,
    general_goal: str,
    skills_path: Path | None = None,
    interview_model: str = "claude-opus-4-8",
    agent: str = AGENT_CLAUDE,
    kind: str = "agent",
    *,
    auto_worktree: bool = True,
) -> TaskMeta:
    project_root = project_root.resolve()
    base = slugify(title)
    slug = ensure_unique_slug(project_root, base)
    root = task_root(project_root, slug)
    root.mkdir(parents=True, exist_ok=True)
    copy_default_plan(root, overwrite=False)
    sk = (skills_path or bundled_skills_path()).expanduser().resolve()
    if not sk.is_file():
        sk = bundled_skills_path().resolve()
    now = _now_iso()
    meta = TaskMeta(
        slug=slug,
        title=title.strip() or slug,
        general_goal=general_goal.strip(),
        created_at=now,
        updated_at=now,
        skills_path=str(sk),
        interview_model=interview_model,
        agent=normalize_agent(agent),
        kind=kind,
    )
    write_meta(project_root, meta)
    _insert_task_order_front(project_root, meta.slug)
    if auto_worktree:
        worktree, _branch, msg = prepare_task_worktree(project_root, slug)
        if worktree is None:
            # The web layer logs this; keep one print here too so anyone
            # running create_task() from a script sees why nothing landed
            # under <task>/work/.
            print(
                f"[loom] auto-worktree skipped for {slug!r}: {msg}",
                flush=True,
            )
        # Either way, sync meta with whatever is now on disk - this also
        # handles the case where prepare_task_worktree said "worktree
        # already exists" because of a previous attempt.
        detect_and_persist_worktree(project_root, slug)
    return read_meta(project_root, slug) or meta


def update_meta(
    project_root: Path,
    slug: str,
    *,
    skills_path: str | None = None,
    interview_model: str | None = None,
    tmux_interview_target: str | None = None,
    agent: str | None = None,
    worktree_path: str | None = None,
    branch: str | None = None,
    worktrees: list[str] | None = None,
    branches: list[str] | None = None,
) -> TaskMeta | None:
    meta = read_meta(project_root, slug)
    if not meta:
        return None
    if skills_path is not None:
        meta.skills_path = skills_path
    if interview_model is not None:
        meta.interview_model = interview_model
    if tmux_interview_target is not None:
        meta.tmux_interview_target = tmux_interview_target
    if agent is not None:
        meta.agent = normalize_agent(agent)
    if worktree_path is not None:
        meta.worktree_path = worktree_path
    if branch is not None:
        meta.branch = branch
    if worktrees is not None:
        meta.worktrees = list(worktrees)
    if branches is not None:
        meta.branches = list(branches)
    # Keep primary in sync with list[0] (if list is set), and trim branches
    # length to match worktrees length.
    if worktrees is not None and meta.worktrees:
        meta.worktree_path = meta.worktrees[0]
    if branches is not None and meta.branches:
        meta.branch = meta.branches[0]
    if len(meta.branches) < len(meta.worktrees):
        meta.branches = meta.branches + [""] * (len(meta.worktrees) - len(meta.branches))
    elif len(meta.branches) > len(meta.worktrees):
        meta.branches = meta.branches[: len(meta.worktrees)]
    if not meta.worktrees:
        meta.worktree_path = ""
        meta.branch = ""
    meta.updated_at = _now_iso()
    write_meta(project_root, meta)
    return meta


def rename_task_meta(
    project_root: Path,
    slug: str,
    *,
    title: str | None = None,
    general_goal: str | None = None,
) -> TaskMeta | None:
    """Update human-readable task metadata (title and/or goal).

    Note: slug never changes - it would imply moving the directory and
    invalidating session IDs / tmux session names. If you really want a
    new slug, delete the task and recreate it.
    """
    meta = read_meta(project_root, slug)
    if not meta:
        return None
    if title is not None:
        cleaned = title.strip()
        if cleaned:
            meta.title = cleaned
    if general_goal is not None:
        meta.general_goal = general_goal.strip()
    meta.updated_at = _now_iso()
    write_meta(project_root, meta)
    return meta


def add_claude_session(project_root: Path, slug: str, session_id: str) -> TaskMeta | None:
    """Append *session_id* to the task's history (dedup, keep order)."""
    sid = (session_id or "").strip()
    if not sid:
        return None
    meta = read_meta(project_root, slug)
    if not meta:
        return None
    if sid in meta.claude_session_ids:
        # Move to the end so "latest" stays last.
        meta.claude_session_ids = [s for s in meta.claude_session_ids if s != sid] + [sid]
    else:
        meta.claude_session_ids.append(sid)
    meta.updated_at = _now_iso()
    write_meta(project_root, meta)
    return meta


def list_task_slugs(project_root: Path) -> list[str]:
    root = rud_root(project_root)
    if not root.is_dir():
        return []
    raw_slugs: list[str] = []
    for p in root.iterdir():
        if p.is_dir() and (p / META).is_file():
            raw_slugs.append(p.name)
    metas_by_slug: dict[str, TaskMeta] = {}
    for s in raw_slugs:
        m = read_meta(project_root, s)
        if m:
            metas_by_slug[m.slug] = m
    metas = list(metas_by_slug.values())
    metas.sort(key=lambda m: m.updated_at, reverse=True)
    fallback = [m.slug for m in metas]
    ordered: list[str] = []
    for slug in _read_task_order(project_root):
        if slug in metas_by_slug and slug not in ordered:
            ordered.append(slug)
    return ordered + [slug for slug in fallback if slug not in ordered]


def list_tasks(project_root: Path) -> list[TaskMeta]:
    return [m for s in list_task_slugs(project_root) if (m := read_meta(project_root, s))]


def reorder_tasks(project_root: Path, slugs: list[str]) -> tuple[bool, str]:
    """Persist the display order for task slugs under this project."""
    slugs = [str(x).strip() for x in slugs if str(x).strip()]
    if any(not _TASK_SLUG_RE.match(s) for s in slugs):
        return False, "invalid slug"
    existing = list_task_slugs(project_root)
    if set(slugs) != set(existing) or len(slugs) != len(existing):
        return False, "slugs must contain every task exactly once"
    _write_task_order(project_root, slugs)
    return True, ""


def delete_task(project_root: Path, slug: str) -> tuple[bool, str]:
    """Delete ``<project>/.RUD/<slug>`` after verifying it stays under ``.RUD``."""
    if not _TASK_SLUG_RE.match(slug):
        return False, "invalid slug"
    root = rud_root(project_root)
    td = task_root(project_root, slug)
    try:
        td.relative_to(root)
    except ValueError:
        return False, "invalid task path"
    if not td.is_dir() or not (td / META).is_file():
        return False, "task not found"
    # Best-effort: unregister EVERY git worktree under <task>/work/ from its
    # source repo before deleting on disk, so the user's checkout doesn't keep
    # dangling `git worktree list` entries.  Runs git from inside each worktree
    # so it resolves the right parent repo - handles project-root repos AND
    # child-repo / container-project worktrees, and multiple worktrees per task.
    worktree_repos = _unregister_task_worktrees(project_root, slug)
    try:
        shutil.rmtree(td)
    except OSError as exc:
        if not _sudo_rmtree(td, root):
            return False, str(exc)
    # Safety net: prune any registration that survived (e.g. `worktree remove`
    # failed but the directory got deleted anyway).
    for common_dir in worktree_repos:
        _git(["--git-dir", common_dir, "worktree", "prune"], project_root.resolve())
    _remove_task_from_order(project_root, slug)
    return True, ""


def path_under_task(task_dir: Path, relative: str) -> Path | None:
    """Resolve *relative* under *task_dir*; return None if traversal escapes."""
    task_dir = task_dir.resolve()
    candidate = (task_dir / relative).resolve()
    try:
        candidate.relative_to(task_dir)
    except ValueError:
        return None
    return candidate


# --- Templates (per-task) ---------------------------------------------------


def read_template(project_root: Path, slug: str, name: str) -> str | None:
    if name not in ALLOWED_TEMPLATE_NAMES:
        return None
    td = task_root(project_root, slug)
    p = td / name
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def write_template(project_root: Path, slug: str, name: str, content: str) -> bool:
    if name not in ALLOWED_TEMPLATE_NAMES:
        return False
    td = task_root(project_root, slug)
    if not td.is_dir():
        return False
    path = td / name
    try:
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        return _sudo_write_text(path, content)
    return True


# --- Kernel Lab interview state (per task, on disk) -------------------------

KERNEL_INTERVIEW = "kernel_interview.json"

# The Kernel Lab interview is a stateless `claude -p` chat - the server keeps
# no live session.  To make the conversation survive page reloads / server
# restarts we persist the message list + the last proposed spec next to the
# task (one JSON file per task).


def read_kernel_interview(project_root: Path, slug: str) -> dict[str, Any]:
    """Return ``{"messages": [...], "spec": <obj|None>}`` for a task.

    Always returns a well-formed dict; missing/corrupt files yield an empty
    conversation so the UI can start fresh.
    """
    p = task_root(project_root, slug) / KERNEL_INTERVIEW
    if not p.is_file():
        return {"messages": [], "spec": None}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"messages": [], "spec": None}
    messages = data.get("messages")
    if not isinstance(messages, list):
        messages = []
    spec = data.get("spec")
    if not isinstance(spec, dict):
        spec = None
    return {"messages": messages, "spec": spec}


def write_kernel_interview(
    project_root: Path,
    slug: str,
    messages: list[Any],
    spec: Any = None,
) -> bool:
    td = task_root(project_root, slug)
    if not td.is_dir():
        return False
    payload = json.dumps(
        {
            "messages": messages if isinstance(messages, list) else [],
            "spec": spec if isinstance(spec, dict) else None,
            "updated_at": _now_iso(),
        },
        indent=2,
    )
    path = td / KERNEL_INTERVIEW
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
    except PermissionError:
        return _sudo_write_text(path, payload)
    except OSError:
        return False
    return True


# --- Per-task run monitor ---------------------------------------------------

TASK_MONITOR = "monitor.json"

# A sensible default: fire when the agent reports it is done, blocked, or
# waiting for input.  Case-insensitive.  Users can override with any regex.
DEFAULT_MONITOR_PATTERN = (
    r"(?i)\b(all done|task complete|completed|finished|blocked|"
    r"waiting for (?:your )?input|need(?:s)? (?:your )?input|awaiting)\b"
)


def read_task_monitor(project_root: Path, slug: str) -> dict[str, Any]:
    """Return the persisted monitor config for a task.

    Shape: ``{enabled, pattern, last_fired, last_match}``.  Missing/corrupt
    files yield a disabled monitor with an empty pattern.
    """
    base = {"enabled": False, "pattern": "", "last_fired": "", "last_match": ""}
    p = task_root(project_root, slug) / TASK_MONITOR
    if not p.is_file():
        return base
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return base
    if not isinstance(data, dict):
        return base
    return {
        "enabled": bool(data.get("enabled", False)),
        "pattern": str(data.get("pattern", "")),
        "last_fired": str(data.get("last_fired", "")),
        "last_match": str(data.get("last_match", "")),
    }


def write_task_monitor(
    project_root: Path,
    slug: str,
    *,
    enabled: bool,
    pattern: str,
    last_fired: str | None = None,
    last_match: str | None = None,
) -> bool:
    """Persist the monitor config; preserves last_fired/last_match if omitted."""
    td = task_root(project_root, slug)
    if not td.is_dir():
        return False
    cur = read_task_monitor(project_root, slug)
    payload = json.dumps(
        {
            "enabled": bool(enabled),
            "pattern": str(pattern or ""),
            "last_fired": cur["last_fired"] if last_fired is None else str(last_fired),
            "last_match": cur["last_match"] if last_match is None else str(last_match),
            "updated_at": _now_iso(),
        },
        indent=2,
    )
    path = td / TASK_MONITOR
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
    except PermissionError:
        return _sudo_write_text(path, payload)
    except OSError:
        return False
    return True


# --- Top-level task markdown files (for the read-only embed picker) ---------

# Maximum size we will ship inline in the task GET payload. Larger files are
# listed in the picker but their content is empty; the user can still see the
# file exists. (10 MB is generous for hand-written markdown.)
_MAX_TASK_MD_INLINE_BYTES = 10 * 1024 * 1024


_SKIP_MARKDOWN_SCAN_DIRS = frozenset(
    {
        ".cache",
        ".git",
        ".hg",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)


def _is_safe_md_relpath(name: str) -> bool:
    """Reject anything that could escape the task directory or isn't *.md."""
    if not name or name in (".", ".."):
        return False
    if "\\" in name or "\x00" in name:
        return False
    if not name.lower().endswith(".md"):
        return False
    parts = Path(name).parts
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


def list_task_markdown_files(project_root: Path, slug: str) -> list[str]:
    """Return relative paths of ``*.md`` files under the task root.

    PLAN.md is always listed first when it exists; other files follow
    case-insensitively sorted. Common generated/dependency directories are
    skipped so worktrees with large build outputs do not make the UI slow.
    """
    root = task_root(project_root, slug)
    if not root.is_dir():
        return []
    candidates: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in _SKIP_MARKDOWN_SCAN_DIRS
            ]
            base = Path(dirpath)
            for filename in filenames:
                if not filename.lower().endswith(".md"):
                    continue
                path = base / filename
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if _is_safe_md_relpath(rel):
                    candidates.append(rel)
    except OSError:
        return []
    candidates.sort(key=str.lower)
    if PLAN in candidates:
        candidates.remove(PLAN)
        candidates.insert(0, PLAN)
    return candidates


def read_task_markdown_file(
    project_root: Path,
    slug: str,
    name: str,
    *,
    max_bytes: int = _MAX_TASK_MD_INLINE_BYTES,
) -> str | None:
    """Read a ``*.md`` file under the task root.

    Returns ``None`` when *name* is unsafe, the file does not exist, or
    its size exceeds *max_bytes* (callers can pass a larger limit to
    force inclusion, e.g. when serving a single file on demand).
    """
    if not _is_safe_md_relpath(name):
        return None
    root = task_root(project_root, slug)
    if not root.is_dir():
        return None
    path = root / Path(name)
    try:
        # Double-check the resolved path is still inside the task root
        # (defends against symlink trickery).
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > max_bytes:
            return None
    except OSError:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# --- Notes (per-project) ----------------------------------------------------


def project_notes_path(project_root: Path) -> Path:
    return rud_root(project_root) / NOTES


def read_project_notes(project_root: Path) -> str:
    p = project_notes_path(project_root)
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def write_project_notes(project_root: Path, content: str) -> bool:
    root = rud_root(project_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    p = project_notes_path(project_root)
    try:
        p.write_text(content, encoding="utf-8")
    except PermissionError:
        return _sudo_write_text(p, content)
    return True


# --- Privileged fallbacks ---------------------------------------------------


def _sudo_write_text(path: Path, content: str) -> bool:
    try:
        result = subprocess.run(
            ["sudo", "-n", "sh", "-c", 'cat > "$1"', "sh", str(path)],
            input=content,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    uid = str(os.getuid())
    gid = str(os.getgid())
    subprocess.run(
        ["sudo", "-n", "chown", f"{uid}:{gid}", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    subprocess.run(
        ["sudo", "-n", "chmod", "u+rw", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return True


def _sudo_rmtree(path: Path, allowed_root: Path) -> bool:
    try:
        target = path.resolve()
        root = allowed_root.resolve()
        target.relative_to(root)
    except (OSError, ValueError):
        return False
    if target == root:
        return False
    try:
        result = subprocess.run(
            ["sudo", "-n", "rm", "-rf", "--", str(target)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


# --- Claude session lookup (~/.claude/projects/<encoded>/<uuid>.jsonl) ------


def claude_project_dir(cwd: Path) -> Path:
    """Encode *cwd* the way Claude Code's CLI does for ``~/.claude/projects``.

    Both ``/`` and ``.`` become ``-`` in the encoded directory name, so e.g.
    ``/home/u/proj/.RUD/foo/work/r`` becomes
    ``-home-u-proj--RUD-foo-work-r`` (the double dash represents ``/.``).
    """
    s = str(cwd.resolve())
    encoded = re.sub(r"[/.]", "-", s)
    return (Path.home() / ".claude" / "projects" / encoded).resolve()


def _list_claude_session_files(cwd: Path) -> list[Path]:
    """All ``<uuid>.jsonl`` session files for *cwd*, oldest first."""
    d = claude_project_dir(cwd)
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file() and p.suffix == ".jsonl"]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def _read_codex_session_meta(path: Path) -> dict[str, Any] | None:
    """First-line JSON of a codex rollout file (``{type:'session_meta', payload:{…}}``).

    None if the file is missing or unreadable.  This is what we use to
    match sessions to a worktree (via ``payload.cwd``) and to extract
    the session id (``payload.id``).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
    except OSError:
        return None
    if not first:
        return None
    try:
        d = json.loads(first)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    return d


def _list_codex_session_files(cwd: Path) -> list[Path]:
    """Codex rollout files whose ``payload.cwd`` matches *cwd*, oldest first.

    Codex stores sessions at ``~/.codex/sessions/YYYY/MM/DD/rollout-…jsonl``
    and records the working directory inside the file (no encoded path in
    the file name).  Filter on the recorded cwd so simultaneous tasks in
    different worktrees don't get mixed up.
    """
    base = (Path.home() / ".codex" / "sessions").resolve()
    if not base.is_dir():
        return []
    try:
        target = str(cwd.resolve())
    except OSError:
        return []
    out: list[Path] = []
    for p in base.rglob("rollout-*.jsonl"):
        meta = _read_codex_session_meta(p)
        if not meta:
            continue
        payload = meta.get("payload") or {}
        if isinstance(payload, dict) and str(payload.get("cwd") or "") == target:
            out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime)
    return out


def list_session_files(cwd: Path, agent: str = AGENT_CLAUDE) -> list[Path]:
    """Agent-aware session file lookup.

    - ``claude`` -> ``~/.claude/projects/<encoded-cwd>/<uuid>.jsonl``
    - ``codex``  -> ``~/.codex/sessions/**/rollout-*.jsonl`` matched by ``payload.cwd``
    """
    agent = normalize_agent(agent)
    if agent == AGENT_CODEX:
        return _list_codex_session_files(cwd)
    return _list_claude_session_files(cwd)


def session_id_from_path(path: Path, agent: str = AGENT_CLAUDE) -> str:
    """Extract a resumable session id from one session file.

    Claude: the file is named ``<uuid>.jsonl`` so the stem *is* the id.
    Codex:  the file is named ``rollout-<timestamp>-<uuid>.jsonl`` but
            the canonical id lives in ``payload.id`` of the first line.
    """
    agent = normalize_agent(agent)
    if agent == AGENT_CODEX:
        meta = _read_codex_session_meta(path)
        if meta:
            payload = meta.get("payload") or {}
            if isinstance(payload, dict):
                sid = str(payload.get("id") or "").strip()
                if sid:
                    return sid
        return ""
    return path.stem  # claude: <uuid>.jsonl


# --- Git / worktree helpers -------------------------------------------------


def _git(args: list[str], cwd: Path, timeout: int = 60) -> tuple[bool, str, str]:
    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    try:
        r = run()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, "", str(exc)
    if r.returncode != 0:
        safe = _dubious_ownership_safe_dir(cwd, r.stdout, r.stderr)
        if safe is not None and _mark_git_safe_directory(safe):
            try:
                r = run()
            except (OSError, subprocess.TimeoutExpired) as exc:
                return False, "", str(exc)
    return r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip()


def _git_diff(args: list[str], cwd: Path, timeout: int = 60) -> tuple[bool, str]:
    """Run a git diff-style command, returning raw (unstripped) stdout.

    ``git diff --no-index`` exits 1 when files differ, which is the normal
    "there is a diff" case, so we treat exit codes 0 and 1 as success. Unlike
    :func:`_git` we must not strip stdout - leading/trailing whitespace is
    significant in a unified patch.
    """
    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    try:
        r = run()
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    if r.returncode not in (0, 1):
        safe = _dubious_ownership_safe_dir(cwd, r.stdout, r.stderr)
        if safe is not None and _mark_git_safe_directory(safe):
            try:
                r = run()
            except (OSError, subprocess.TimeoutExpired):
                return False, ""
    return r.returncode in (0, 1), (r.stdout or "")


def _dubious_ownership_safe_dir(cwd: Path, stdout: str, stderr: str) -> Path | None:
    text = "\n".join(x for x in (stdout, stderr) if x)
    if "detected dubious ownership" not in text:
        return None
    m = _DUBIOUS_OWNERSHIP_RE.search(text)
    raw = m.group(1) if m else str(cwd)
    try:
        return Path(raw).expanduser().resolve()
    except OSError:
        return Path(raw).expanduser()


def _mark_git_safe_directory(path: Path) -> bool:
    safe_path = str(path)
    try:
        existing = subprocess.run(
            ["git", "config", "--global", "--get-all", "safe.directory"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if safe_path in (existing.stdout or "").splitlines():
            return True
        added = subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", safe_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return added.returncode == 0


def git_toplevel(path: Path) -> Path | None:
    path = path.resolve()
    ok, out, _ = _git(["rev-parse", "--show-toplevel"], path)
    if not ok or not out:
        return None
    return Path(out)


def _branch_name_for(slug: str) -> str:
    # Per charlie_skills.md the user wants branches under zhongzhu/<slug>.
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-") or "task"
    return f"zhongzhu/{cleaned[:80]}"


def direct_child_git_repos(parent: Path) -> list[Path]:
    """Immediate subdirectories of *parent* that are themselves git repo roots."""
    try:
        parent = parent.resolve()
    except OSError:
        return []
    out: list[Path] = []
    if not parent.is_dir():
        return out
    try:
        children = sorted(parent.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return out
    for child in children:
        try:
            if not child.is_dir() or child.name.startswith("."):
                continue
        except OSError:
            continue
        top = git_toplevel(child)
        if top is not None and top.resolve() == child.resolve():
            out.append(child)
    return out


def list_worktree_candidates(project_root: Path) -> list[dict[str, Any]]:
    """Git repos a task can branch a worktree from.

    - If *project_root* itself is a git repo, returns just that one with
      ``kind == "self"``.
    - Otherwise scans immediate subdirectories for git repos and returns
      each with ``kind == "child"``.  Useful when the registered project
      is a container directory holding several clones side-by-side (e.g.
      ``~/work/xorl/{xorl-internal,xorl-eval}``).
    """
    try:
        project_root = project_root.resolve()
    except OSError:
        return []
    git_root = git_toplevel(project_root)
    if git_root is not None:
        return [{
            "path": str(git_root),
            "name": git_root.name,
            "kind": "self",
        }]
    return [
        {"path": str(p), "name": p.name, "kind": "child"}
        for p in direct_child_git_repos(project_root)
    ]


def _record_worktree_base(
    project_root: Path, slug: str, wt: Path, commit: str, branch: str
) -> None:
    """Persist a worktree's fork point (base commit + source branch) into meta.

    Recorded at creation time so the diff can later compare against where the
    worktree was branched from, instead of guessing the trunk.
    """
    meta = read_meta(project_root, slug)
    if meta is None:
        return
    meta.worktree_bases[str(wt.resolve())] = {
        "commit": commit.strip(),
        "branch": branch.strip(),
    }
    write_meta(project_root, meta)


def prepare_task_worktree_from(
    project_root: Path,
    slug: str,
    source_repo: Path,
) -> tuple[Path | None, str, str]:
    """Create a worktree at ``<task>/work/<source-repo-name>`` branched from
    the HEAD of *source_repo*.

    Returns ``(worktree_path_or_None, branch_name, message)``.
    """
    project_root = project_root.resolve()
    try:
        source_repo = source_repo.expanduser().resolve()
    except OSError as exc:
        return None, "", f"invalid source path: {exc}"
    git_root = git_toplevel(source_repo)
    if git_root is None:
        return None, "", f"not a git repository: {source_repo}"
    branch = _branch_name_for(slug)
    td = task_root(project_root, slug)
    work_dir = td / WORK_SUBDIR / git_root.name
    work_dir.parent.mkdir(parents=True, exist_ok=True)

    if work_dir.is_dir():
        ok, _, _ = _git(["rev-parse", "--is-inside-work-tree"], work_dir)
        if ok:
            return work_dir.resolve(), branch, "worktree already exists"

    ok_head, head_out, head_err = _git(["rev-parse", "HEAD"], git_root)
    if not ok_head or not head_out:
        return None, branch, f"cannot read HEAD: {head_err or '(empty)'}"
    head_out = head_out.strip()
    # The source repo's current branch ("" if it's detached) - recorded as the
    # worktree's "created from" base alongside the exact fork commit.
    source_branch = _detect_branch_in(git_root)

    add_new = _git(
        ["worktree", "add", "-b", branch, str(work_dir), head_out],
        git_root,
    )
    if add_new[0]:
        _record_worktree_base(project_root, slug, work_dir, head_out, source_branch)
        return work_dir.resolve(), branch, "worktree created"
    reuse = _git(["worktree", "add", str(work_dir), branch], git_root)
    if reuse[0]:
        return work_dir.resolve(), branch, "worktree attached to existing branch"
    return None, branch, f"git worktree add failed: {add_new[2] or add_new[1]} | {reuse[2]}"


def prepare_task_worktree(
    project_root: Path,
    slug: str,
) -> tuple[Path | None, str, str]:
    """Auto-create from the project root itself (only works when it's a git repo).

    Returns ``(worktree_path_or_None, branch_name, message)``.  Container
    project roots (no ``.git`` of their own) return ``(None, branch,
    "project root is not a git repository; pick a candidate manually")``;
    the web UI exposes a manual picker for that case.
    """
    project_root = project_root.resolve()
    if git_toplevel(project_root) is None:
        return (
            None,
            _branch_name_for(slug),
            "project root is not a git repository; pick a candidate manually",
        )
    return prepare_task_worktree_from(project_root, slug, project_root)


def worktree_status(wt: Path) -> dict[str, Any] | None:
    """Return a ``git status``-derived snapshot of *wt*, or ``None`` on error.

    The shape is friendly for direct JSON serialisation; see the keys at
    the bottom of the function.  Uses ``git status --branch --porcelain``
    so we get the branch line (with ahead/behind) plus a per-file tally.
    """
    try:
        wt = wt.resolve()
    except OSError:
        return None
    if not wt.is_dir():
        return None
    ok, out, err = _git(["status", "--branch", "--porcelain"], wt, timeout=15)
    if not ok:
        return {
            "path": str(wt),
            "branch": "",
            "upstream": "",
            "has_remote": False,
            "ahead": 0,
            "behind": 0,
            "staged": 0,
            "unstaged": 0,
            "untracked": 0,
            "clean": False,
            "dirty_count": 0,
            "error": err or "git status failed",
        }
    branch = ""
    upstream = ""
    ahead = 0
    behind = 0
    staged = 0
    unstaged = 0
    untracked = 0
    for line in out.splitlines():
        if line.startswith("## "):
            rest = line[3:]
            if " [" in rest and rest.endswith("]"):
                head, brackets = rest.rsplit(" [", 1)
                rest = head
                for part in brackets.rstrip("]").split(","):
                    p = part.strip()
                    if p.startswith("ahead "):
                        try:
                            ahead = int(p[6:])
                        except ValueError:
                            pass
                    elif p.startswith("behind "):
                        try:
                            behind = int(p[7:])
                        except ValueError:
                            pass
            if "..." in rest:
                bp, up = rest.split("...", 1)
                branch = bp.strip()
                upstream = up.strip()
            else:
                branch = rest.strip()
        elif line.startswith("?? "):
            untracked += 1
        elif len(line) >= 2:
            s_char, w_char = line[0], line[1]
            if s_char not in (" ", "?"):
                staged += 1
            if w_char not in (" ", "?"):
                unstaged += 1
    dirty = staged + unstaged + untracked
    return {
        "path": str(wt),
        "branch": branch,
        "upstream": upstream,
        "has_remote": bool(upstream),
        "ahead": ahead,
        "behind": behind,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "clean": dirty == 0,
        "dirty_count": dirty,
    }


def _strip_ab_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _parse_patch_files(
    patch_text: str,
    scope: str,
    max_patch_bytes: int = 200_000,
) -> list[dict[str, Any]]:
    """Split a unified ``git diff`` blob into one entry per file.

    Each entry has ``path``, ``old_path`` (for renames), ``status``
    (added/deleted/modified/renamed), ``scope``, ``additions``,
    ``deletions``, ``binary`` and the per-file ``patch`` text.
    """
    if not patch_text.strip():
        return []
    files: list[dict[str, Any]] = []
    chunks = re.split(r"(?m)^(?=diff --git )", patch_text)
    for chunk in chunks:
        if not chunk.strip():
            continue
        lines = chunk.splitlines()
        header = lines[0] if lines else ""
        path = ""
        old_path = ""
        status = "modified"
        additions = 0
        deletions = 0
        is_binary = False
        for ln in lines:
            if ln.startswith("new file"):
                status = "added"
            elif ln.startswith("deleted file"):
                status = "deleted"
            elif ln.startswith("rename from "):
                old_path = ln[len("rename from "):].strip()
                status = "renamed"
            elif ln.startswith("rename to "):
                path = ln[len("rename to "):].strip()
                status = "renamed"
            elif ln.startswith("Binary files"):
                is_binary = True
            elif ln.startswith("+++ "):
                p = ln[4:].strip()
                if p != "/dev/null":
                    path = _strip_ab_prefix(p)
            elif ln.startswith("--- "):
                p = ln[4:].strip()
                if p != "/dev/null":
                    old_path = _strip_ab_prefix(p)
            elif ln.startswith("+") and not ln.startswith("+++"):
                additions += 1
            elif ln.startswith("-") and not ln.startswith("---"):
                deletions += 1
        if not path:
            path = old_path
        if not path:
            m = re.match(r"diff --git a/(.*?) b/(.*)$", header)
            if m:
                path = m.group(2)
        patch = chunk
        if len(patch) > max_patch_bytes:
            patch = patch[:max_patch_bytes] + "\n... (diff truncated) ...\n"
        files.append({
            "path": path or "(unknown)",
            "old_path": old_path if (status == "renamed" and old_path and old_path != path) else "",
            "status": status,
            "scope": scope,
            "additions": additions,
            "deletions": deletions,
            "binary": is_binary,
            "patch": patch,
        })
    return files


def _detect_base_ref(wt: Path) -> str:
    """Best-effort default base branch to compare committed work against."""
    for ref in ("origin/main", "origin/master", "main", "master"):
        ok, _, _ = _git(["rev-parse", "--verify", "--quiet", ref], wt, timeout=10)
        if ok:
            return ref
    return ""


def worktree_diff(
    wt: Path, max_files: int = 400, base_ref: str = "", base_label: str = ""
) -> dict[str, Any]:
    """Read-only diff snapshot for a single worktree.

    Combines two scopes into one per-file list:

    - ``uncommitted``: working tree + index vs ``HEAD`` (tracked changes via
      ``git diff HEAD``) plus untracked files (rendered as additions).
    - ``committed``: commits on this branch vs the base (``base...HEAD``).
      The base is the recorded fork point (``base_ref`` = the commit this
      worktree was branched from, ``base_label`` = its source branch) when
      known, else a detected trunk (origin/main, ...), so reviewers see what
      the task added since it forked.
    """
    try:
        wt = wt.resolve()
    except OSError as exc:
        return {"path": str(wt), "branch": "", "base": "", "files": [], "error": str(exc)}
    if not wt.is_dir():
        return {"path": str(wt), "branch": "", "base": "", "files": [], "error": "worktree directory missing"}

    _ok, branch, _ = _git(["branch", "--show-current"], wt, timeout=10)
    branch = branch.strip()

    # Prefer the recorded fork point (the commit this worktree was branched
    # from); verify it still resolves, else fall back to trunk detection.
    base = ""
    base_kind = "none"
    display_base = ""
    if base_ref:
        ok_b, _, _ = _git(["rev-parse", "--verify", "--quiet", base_ref], wt, timeout=10)
        if ok_b:
            base = base_ref
            base_kind = "fork"
            display_base = base_label or base_ref[:12]
    if not base:
        base = _detect_base_ref(wt)
        if base:
            base_kind = "detected"
            display_base = base

    files: list[dict[str, Any]] = []
    truncated = False

    # Committed changes vs base (skip if no base or base == HEAD tip).
    if base:
        ok_c, committed = _git_diff(
            ["diff", "--find-renames", f"{base}...HEAD"], wt, timeout=60
        )
        if ok_c:
            files.extend(_parse_patch_files(committed, "committed"))

    # Tracked uncommitted changes vs HEAD (staged + unstaged).
    ok_u, uncommitted = _git_diff(["diff", "--find-renames", "HEAD"], wt, timeout=60)
    if ok_u:
        files.extend(_parse_patch_files(uncommitted, "uncommitted"))

    # Untracked files: list each, render as an addition via --no-index.
    ok_s, status_out, _ = _git(
        ["status", "--porcelain", "--untracked-files=all"], wt, timeout=15
    )
    if ok_s:
        untracked_paths = [
            ln[3:].strip().strip('"')
            for ln in status_out.splitlines()
            if ln.startswith("?? ")
        ]
        for rel in untracked_paths[:80]:
            ok_d, patch = _git_diff(
                ["diff", "--no-index", "--", "/dev/null", rel], wt, timeout=30
            )
            if ok_d and patch.strip():
                files.extend(_parse_patch_files(patch, "uncommitted"))
            else:
                files.append({
                    "path": rel,
                    "old_path": "",
                    "status": "added",
                    "scope": "uncommitted",
                    "additions": 0,
                    "deletions": 0,
                    "binary": True,
                    "patch": "",
                })
        if len(untracked_paths) > 80:
            truncated = True

    if len(files) > max_files:
        files = files[:max_files]
        truncated = True

    return {
        "path": str(wt),
        "branch": branch,
        "base": display_base,
        "base_kind": base_kind,
        "files": files,
        "truncated": truncated,
    }


def task_worktree_diffs(project_root: Path, slug: str) -> list[dict[str, Any]]:
    """``worktree_diff`` for every worktree tracked by the task.

    Passes each worktree's recorded fork point (base commit + source branch)
    so the diff compares against where the worktree was created from.
    """
    meta = read_meta(project_root, slug)
    if meta is None:
        return []
    paths = list(meta.worktrees)
    if not paths:
        return []
    bases = meta.worktree_bases or {}

    def _base_for(p: str) -> tuple[str, str]:
        info = bases.get(p) or bases.get(str(Path(p).resolve())) or {}
        return str(info.get("commit", "")), str(info.get("branch", ""))

    if len(paths) == 1:
        commit, label = _base_for(paths[0])
        return [worktree_diff(Path(paths[0]), base_ref=commit, base_label=label)]

    def _one(p: str) -> dict[str, Any]:
        commit, label = _base_for(p)
        return worktree_diff(Path(p), base_ref=commit, base_label=label)

    max_workers = min(len(paths), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, paths))


def list_task_worktree_statuses(project_root: Path, slug: str) -> list[dict[str, Any]]:
    """Status snapshot for every worktree currently tracked by the task.

    Runs ``git status`` for every worktree concurrently. Each git call
    shells out and typically takes 100-500 ms on a large repo, so for
    tasks with multiple worktrees this is a near-linear speedup over the
    previous serial loop.
    """
    meta = read_meta(project_root, slug)
    if meta is None:
        return []
    paths = list(meta.worktrees)
    if not paths:
        return []

    def _safe_status(p_str: str) -> dict[str, Any]:
        s = worktree_status(Path(p_str))
        if s is not None:
            return s
        return {
            "path": p_str,
            "branch": "",
            "upstream": "",
            "has_remote": False,
            "ahead": 0,
            "behind": 0,
            "staged": 0,
            "unstaged": 0,
            "untracked": 0,
            "clean": False,
            "dirty_count": 0,
            "error": "worktree directory missing",
        }

    # Cap the pool so we don't fork unbounded git processes if a task
    # somehow ends up with dozens of worktrees - 8 is plenty for typical
    # multi-repo tasks (2-4 worktrees).
    if len(paths) == 1:
        return [_safe_status(paths[0])]
    max_workers = min(len(paths), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_safe_status, paths))


def push_worktree_branch(wt: Path) -> dict[str, Any]:
    """``git push -u origin <current-branch>`` from inside the worktree.

    Returns a dict with ``ok``, ``branch``, and a human-readable
    ``message``.  Doesn't touch task metadata - the caller can re-run
    ``worktree_status`` to refresh ahead/behind counts.
    """
    try:
        wt = wt.resolve()
    except OSError as exc:
        return {"ok": False, "error": str(exc), "branch": "", "message": str(exc)}
    if not wt.is_dir():
        return {"ok": False, "error": "worktree not found", "branch": "", "message": "worktree not found"}
    ok, branch, err = _git(["branch", "--show-current"], wt, timeout=10)
    branch = branch.strip()
    if not ok or not branch:
        msg = err or "no current branch"
        return {"ok": False, "error": msg, "branch": "", "message": msg}
    ok_push, out, err_push = _git(
        ["push", "-u", "origin", branch], wt, timeout=180
    )
    text = "\n".join(x for x in (out, err_push) if x).strip()
    return {
        "ok": ok_push,
        "branch": branch,
        "message": text or ("pushed" if ok_push else "push failed"),
        "error": "" if ok_push else (text or "push failed"),
    }


def merge_worktree_to_base(wt: Path) -> dict[str, Any]:
    """Merge a worktree's branch back into the source repo's checked-out branch.

    Runs ``git merge --no-ff <wt-branch>`` inside the *source* repo (the main
    working tree this worktree was created from). Safety:
    - refuses if the source working tree is dirty (won't clobber your work),
    - aborts the merge on conflict and reports the conflicting files,
    - does NOT push (the merge stays local, so it's easy to review or reset).
    """
    try:
        wt = wt.resolve()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    if not wt.is_dir():
        return {"ok": False, "error": "worktree not found"}
    ok, branch, _ = _git(["branch", "--show-current"], wt, timeout=10)
    branch = branch.strip()
    if not ok or not branch:
        return {"ok": False, "error": "worktree has no current branch"}
    ok, common, _ = _git(["rev-parse", "--git-common-dir"], wt, timeout=10)
    if not ok or not common:
        return {"ok": False, "error": "could not resolve the source repo"}
    cp = Path(common)
    if not cp.is_absolute():
        cp = wt / cp
    try:
        source = cp.resolve().parent
    except OSError:
        source = cp.parent
    if not source.is_dir():
        return {"ok": False, "error": f"source repo not found at {source}"}
    ok, base, _ = _git(["branch", "--show-current"], source, timeout=10)
    base = base.strip()
    if not base:
        return {
            "ok": False,
            "error": "source repo is in a detached HEAD - checkout a base branch first",
            "source": str(source),
        }
    if base == branch:
        return {
            "ok": False,
            "error": f"source repo is already on {branch}; nothing to merge",
            "base": base,
            "branch": branch,
        }
    _ok, dirty, _ = _git(["status", "--porcelain"], source, timeout=15)
    # Only block on uncommitted changes to *tracked* files (a merge could
    # clobber them). Untracked files - notably the project's own ``.RUD/``
    # task dir living inside a self-repo - must not block the merge; git's
    # own merge still aborts if an untracked file would be overwritten.
    tracked_dirty = [
        ln for ln in dirty.splitlines() if ln.strip() and not ln.startswith("??")
    ]
    if tracked_dirty:
        return {
            "ok": False,
            "base": base,
            "branch": branch,
            "source": str(source),
            "error": (
                f"source repo ({source}) has uncommitted changes to tracked "
                "files; commit or stash them before merging"
            ),
        }
    ok_m, out, err = _git(["merge", "--no-ff", branch], source, timeout=120)
    text = "\n".join(x for x in (out, err) if x).strip()
    if ok_m:
        return {
            "ok": True,
            "base": base,
            "branch": branch,
            "source": str(source),
            "message": text or f"merged {branch} into {base}",
        }
    # Conflict / failure: abort so the source tree is left clean.
    _ok2, conflicts, _ = _git(
        ["diff", "--name-only", "--diff-filter=U"], source, timeout=15
    )
    _git(["merge", "--abort"], source, timeout=30)
    return {
        "ok": False,
        "base": base,
        "branch": branch,
        "source": str(source),
        "error": (
            "merge hit conflicts (aborted, source left clean) - resolve them "
            "manually or ask the agent to merge"
        ),
        "conflicts": [c for c in conflicts.splitlines() if c.strip()],
        "detail": text,
    }


def _unregister_task_worktrees(project_root: Path, slug: str) -> set[str]:
    """``git worktree remove --force`` every worktree under the task.

    Returns the set of parent-repo git-common-dirs so the caller can run a
    final ``git worktree prune`` after the task directory is deleted.  Runs git
    from inside each worktree so it resolves the correct parent repo even when
    the worktree was created from a child repo (container projects) rather than
    the project root, and so multi-worktree tasks are fully cleaned up.
    """
    commons: set[str] = set()
    for wt in list_task_worktrees(project_root, slug):
        try:
            if not wt.is_dir():
                continue
        except OSError:
            continue
        ok, common, _ = _git(["rev-parse", "--git-common-dir"], wt)
        if ok and common:
            cp = Path(common)
            if not cp.is_absolute():
                cp = wt / cp
            try:
                commons.add(str(cp.resolve()))
            except OSError:
                commons.add(str(cp))
        # remove (then --force fallback) from inside the worktree itself
        ok_rm, _out, _err = _git(["worktree", "remove", str(wt)], wt)
        if not ok_rm:
            _git(["worktree", "remove", "--force", str(wt)], wt)
    return commons
