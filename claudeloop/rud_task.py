"""`.RUD` task layout: storage, paths, worktrees, default skills."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claudeloop.paths import bundled_skills_path

RUD_DIR = ".RUD"
WORK_SUBDIR = "work"
RUNS_SUBDIR = "runs"
WORK_ROOT_REPO_KEY = "work-root"

TASK_PROMPT = "TASK_PROMPT.md"
SUCCESS_CONDITION = "SUCCESS_CONDITION.md"
PLAN = "PLAN.md"
INTERVIEW = "INTERVIEW.md"
LEGACY_INTERVIEW = "interview.md"
META = "task.json"
TASK_ORDER = "task-order.json"

ALLOWED_TEMPLATE_NAMES = frozenset({TASK_PROMPT, SUCCESS_CONDITION, PLAN})

_REPO_SEG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_TASK_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_DUBIOUS_OWNERSHIP_RE = re.compile(r"detected dubious ownership in repository at '([^']+)'")


def rud_root(project_root: Path) -> Path:
    return (project_root / RUD_DIR).resolve()


def task_root(project_root: Path, slug: str) -> Path:
    """``<project>/.RUD/<slug>/`` — task name becomes slug (e.g. xorl1)."""
    return (rud_root(project_root) / slug).resolve()


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


def copy_default_templates(dest_dir: Path, overwrite: bool = False) -> None:
    src = package_templates_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in (TASK_PROMPT, SUCCESS_CONDITION, PLAN):
        out = dest_dir / name
        if out.exists() and not overwrite:
            continue
        p = src / name
        if p.is_file():
            out.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            out.write_text("", encoding="utf-8")


@dataclass
class TaskMeta:
    slug: str
    title: str
    general_goal: str
    created_at: str
    updated_at: str
    work_dirs: list[str] = field(default_factory=list)
    skills_path: str = ""
    interview_model: str = "claude-sonnet-4-6"
    interview_backend: str = "cli"
    tmux_interview_target: str = ""
    tmux_ask_target: str = ""
    tmux_runner_target: str = ""
    tmux_evaluator_target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "general_goal": self.general_goal,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "work_dirs": self.work_dirs,
            "skills_path": self.skills_path,
            "interview_model": self.interview_model,
            "interview_backend": self.interview_backend,
            "tmux_interview_target": self.tmux_interview_target,
            "tmux_ask_target": self.tmux_ask_target,
            "tmux_runner_target": self.tmux_runner_target,
            "tmux_evaluator_target": self.tmux_evaluator_target,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskMeta:
        ib = str(data.get("interview_backend", "cli")).lower()
        if ib not in ("cli", "sdk"):
            ib = "cli"
        return cls(
            slug=str(data["slug"]),
            title=str(data.get("title", "")),
            general_goal=str(data.get("general_goal", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            work_dirs=[str(x) for x in data.get("work_dirs", [])],
            skills_path=str(data.get("skills_path", "")),
            interview_model=str(data.get("interview_model", "claude-sonnet-4-6")),
            interview_backend=ib,
            tmux_interview_target=str(data.get("tmux_interview_target", "")),
            tmux_ask_target=str(data.get("tmux_ask_target", "")),
            tmux_runner_target=str(data.get("tmux_runner_target", "")),
            tmux_evaluator_target=str(data.get("tmux_evaluator_target", "")),
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
    except FileNotFoundError:
        pass
    except OSError:
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
    interview_model: str = "claude-sonnet-4-6",
    interview_backend: str = "cli",
    work_dirs: list[Path] | None = None,
    *,
    auto_worktrees: bool = True,
) -> TaskMeta:
    project_root = project_root.resolve()
    base = slugify(title)
    slug = ensure_unique_slug(project_root, base)
    root = task_root(project_root, slug)
    root.mkdir(parents=True, exist_ok=True)
    (root / WORK_SUBDIR).mkdir(exist_ok=True)
    (root / RUNS_SUBDIR).mkdir(exist_ok=True)
    copy_default_templates(root, overwrite=False)
    if not (root / INTERVIEW).exists():
        legacy = root / LEGACY_INTERVIEW
        if legacy.is_file():
            legacy.replace(root / INTERVIEW)
        else:
            (root / INTERVIEW).write_text("", encoding="utf-8")
    sk = (skills_path or bundled_skills_path()).expanduser().resolve()
    if not sk.is_file():
        sk = bundled_skills_path().resolve()
    sp = str(sk)
    wd: list[str] = []
    if work_dirs:
        for p in work_dirs:
            wd.append(str(p.expanduser().resolve()))
    ib = interview_backend.lower() if interview_backend.lower() in ("cli", "sdk") else "cli"
    now = _now_iso()
    meta = TaskMeta(
        slug=slug,
        title=title.strip() or slug,
        general_goal=general_goal.strip(),
        created_at=now,
        updated_at=now,
        work_dirs=wd,
        skills_path=sp,
        interview_model=interview_model,
        interview_backend=ib,
    )
    write_meta(project_root, meta)
    _insert_task_order_front(project_root, meta.slug)
    if auto_worktrees and wd:
        prepare_all_worktrees(project_root, slug)
    return meta


def update_meta(
    project_root: Path,
    slug: str,
    *,
    work_dirs: list[str] | None = None,
    skills_path: str | None = None,
    interview_model: str | None = None,
    interview_backend: str | None = None,
    tmux_interview_target: str | None = None,
    tmux_ask_target: str | None = None,
    tmux_runner_target: str | None = None,
    tmux_evaluator_target: str | None = None,
) -> TaskMeta | None:
    meta = read_meta(project_root, slug)
    if not meta:
        return None
    if work_dirs is not None:
        meta.work_dirs = work_dirs
    if skills_path is not None:
        meta.skills_path = skills_path
    if interview_model is not None:
        meta.interview_model = interview_model
    if interview_backend is not None and interview_backend.lower() in ("cli", "sdk"):
        meta.interview_backend = interview_backend.lower()
    if tmux_interview_target is not None:
        meta.tmux_interview_target = tmux_interview_target
    if tmux_ask_target is not None:
        meta.tmux_ask_target = tmux_ask_target
    if tmux_runner_target is not None:
        meta.tmux_runner_target = tmux_runner_target
    if tmux_evaluator_target is not None:
        meta.tmux_evaluator_target = tmux_evaluator_target
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
    """Delete ``<project>/.RUD/<slug>`` after verifying the target stays under ``.RUD``."""
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
    try:
        shutil.rmtree(td)
    except OSError as exc:
        if not _sudo_rmtree(td, root):
            return False, str(exc)
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
    """Remove a task directory with sudo after re-checking the safety boundary."""
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


def append_interview(project_root: Path, slug: str, role: str, text: str) -> None:
    td = task_root(project_root, slug)
    path = td / INTERVIEW
    legacy = td / LEGACY_INTERVIEW
    if not path.exists() and legacy.is_file():
        legacy.replace(path)
    block = f"\n## {role}\n\n{text.strip()}\n\n"
    if path.exists():
        path.write_text(path.read_text(encoding="utf-8") + block, encoding="utf-8")
    else:
        path.write_text(block.lstrip(), encoding="utf-8")


def read_interview(project_root: Path, slug: str) -> str:
    td = task_root(project_root, slug)
    p = td / INTERVIEW
    legacy = td / LEGACY_INTERVIEW
    if not p.is_file() and legacy.is_file():
        legacy.replace(p)
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


# --- git / worktree ---


def _git(args: list[str], cwd: Path) -> tuple[bool, str, str]:
    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
        )

    r = run()
    if r.returncode != 0:
        safe_dir = _dubious_ownership_safe_dir(cwd, r.stdout, r.stderr)
        if safe_dir is not None and _mark_git_safe_directory(safe_dir):
            r = run()
    return r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip()


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


def repo_name_for_work_dir(work_dir: Path) -> str:
    return work_dir.resolve().name


def parse_repo_key(repo_key: str) -> list[str] | None:
    """Return path segments for ``G1`` or ``G1/G2``; None if invalid."""
    if repo_key == WORK_ROOT_REPO_KEY:
        return [WORK_ROOT_REPO_KEY]
    if not repo_key or repo_key.startswith("/") or ".." in repo_key:
        return None
    parts = [p for p in repo_key.replace("\\", "/").split("/") if p]
    if not parts or len(parts) > 2:
        return None
    for p in parts:
        if not _REPO_SEG_RE.match(p):
            return None
    return parts


def work_path_for_repo_key(task_dir: Path, repo_key: str) -> Path | None:
    if repo_key == WORK_ROOT_REPO_KEY:
        return task_dir.resolve() / WORK_SUBDIR
    parts = parse_repo_key(repo_key)
    if not parts:
        return None
    d = task_dir.resolve() / WORK_SUBDIR
    for part in parts:
        d = d / part
    return d


def runs_dir_for_repo(task_dir: Path, repo_key: str) -> Path | None:
    if repo_key == WORK_ROOT_REPO_KEY:
        d = task_dir.resolve() / RUNS_SUBDIR / WORK_ROOT_REPO_KEY
        d.mkdir(parents=True, exist_ok=True)
        return d
    parts = parse_repo_key(repo_key)
    if not parts:
        return None
    d = task_dir.resolve() / RUNS_SUBDIR
    for part in parts:
        d = d / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def direct_child_git_repos(parent_git_root: Path) -> list[Path]:
    """Immediate subdirectories of *parent_git_root* that are separate git roots (max one level)."""
    parent_git_root = parent_git_root.resolve()
    out: list[Path] = []
    try:
        for child in sorted(parent_git_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            top = git_toplevel(child)
            if top and top.resolve() == child.resolve():
                out.append(child)
    except OSError:
        pass
    return out


def _branch_slug(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", label).strip("-")
    return (s[:80] or "wt").replace("/", "-")


def _git_worktree_add(
    git_root: Path,
    worktree_dest: Path,
    task_slug: str,
    branch_suffix: str,
) -> tuple[bool, str, Path | None]:
    """``git worktree add`` from *git_root* to *worktree_dest*."""
    git_root = git_root.resolve()
    worktree_dest = worktree_dest.resolve()
    if not git_toplevel(git_root):
        return False, f"Not a git repository: {git_root}", None
    if worktree_dest.is_dir():
        ok, _, _ = _git(["rev-parse", "--is-inside-work-tree"], worktree_dest)
        if ok:
            return True, "existing", worktree_dest
    worktree_dest.parent.mkdir(parents=True, exist_ok=True)
    ok_head, head_out, head_err = _git(["rev-parse", "HEAD"], git_root)
    if not ok_head:
        return False, f"Cannot read HEAD: {head_err}", None
    branch = f"claudeloop-rud/{_branch_slug(task_slug)}/{_branch_slug(branch_suffix)}"
    add_new = _git(
        ["worktree", "add", "-b", branch, str(worktree_dest), head_out],
        git_root,
    )
    if add_new[0]:
        return True, "created", worktree_dest
    reuse = _git(["worktree", "add", str(worktree_dest), branch], git_root)
    if reuse[0]:
        return True, "reused_branch", worktree_dest
    return False, f"git worktree add failed: {add_new[2] or add_new[1]} | {reuse[2]}", None


def prepare_worktrees_from_work_dir(
    task_dir: Path,
    work_dir: Path,
    slug: str,
) -> list[dict[str, Any]]:
    """
    Primary repo ``G1`` -> ``<task>/work/G1``.
    Nested repos (direct children of G1's git root only) -> ``<task>/work/G1/G2``, …
    """
    task_dir = task_dir.resolve()
    results: list[dict[str, Any]] = []
    git_root = git_toplevel(work_dir)
    if not git_root:
        child_repos = direct_child_git_repos(work_dir)
        if not child_repos:
            results.append(
                {
                    "work_dir": str(work_dir),
                    "repo_key": None,
                    "ok": False,
                    "reason": f"Not a git repository and no direct child git repositories: {work_dir}",
                    "worktree": None,
                }
            )
            return results
        for child_root in child_repos:
            repo_key = child_root.name
            dest = task_dir / WORK_SUBDIR / repo_key
            ok, reason, wt = _git_worktree_add(child_root, dest, slug, repo_key)
            results.append(
                {
                    "work_dir": str(child_root),
                    "repo_key": repo_key,
                    "ok": ok,
                    "reason": reason,
                    "worktree": str(wt) if wt else None,
                }
            )
        return results

    primary_name = git_root.name
    dest_primary = task_dir / WORK_SUBDIR / primary_name
    ok, reason, wt = _git_worktree_add(git_root, dest_primary, slug, primary_name)
    results.append(
        {
            "work_dir": str(work_dir),
            "repo_key": primary_name,
            "ok": ok,
            "reason": reason,
            "worktree": str(wt) if wt else None,
        }
    )
    if not ok:
        return results

    for child in direct_child_git_repos(git_root):
        nested_key = f"{primary_name}/{child.name}"
        dest_nested = task_dir / WORK_SUBDIR / primary_name / child.name
        child_root = git_toplevel(child)
        if not child_root:
            continue
        ok2, reason2, wt2 = _git_worktree_add(child_root, dest_nested, slug, nested_key)
        results.append(
            {
                "work_dir": str(child),
                "repo_key": nested_key,
                "ok": ok2,
                "reason": reason2,
                "worktree": str(wt2) if wt2 else None,
            }
        )
    return results


def prepare_all_worktrees(project_root: Path, slug: str) -> list[dict[str, Any]]:
    meta = read_meta(project_root, slug)
    if not meta:
        return []
    td = task_root(project_root, slug)
    all_results: list[dict[str, Any]] = []
    for wd_str in meta.work_dirs:
        all_results.extend(prepare_worktrees_from_work_dir(td, Path(wd_str), slug))
    return all_results


def prepare_selected_worktrees(project_root: Path, slug: str, work_dirs: list[Path]) -> list[dict[str, Any]]:
    """Prepare worktrees only for the explicit source repo/container directories."""
    td = task_root(project_root, slug)
    all_results: list[dict[str, Any]] = []
    for work_dir in work_dirs:
        all_results.extend(prepare_worktrees_from_work_dir(td, work_dir, slug))
    return all_results


def list_work_repo_keys(task_dir: Path) -> list[str]:
    """Repo keys under ``work/``: ``G1``, ``G1/G2``, … (at most one slash)."""
    work = task_dir.resolve() / WORK_SUBDIR
    if not work.is_dir():
        return []
    keys: list[str] = []
    for p in sorted(work.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        keys.append(p.name)
        try:
            for q in sorted(p.iterdir()):
                if not q.is_dir() or q.name.startswith("."):
                    continue
                top = git_toplevel(q)
                if top and top.resolve() == q.resolve():
                    keys.append(f"{p.name}/{q.name}")
        except OSError:
            pass
    keys.sort()
    return keys


def validate_repo_key(repo_key: str) -> bool:
    return parse_repo_key(repo_key) is not None


# Backwards-compatible name for tests
def prepare_worktree(
    task_dir: Path,
    work_dir: Path,
    slug: str,
) -> tuple[bool, str, Path | None]:
    rows = prepare_worktrees_from_work_dir(task_dir, work_dir, slug)
    if not rows:
        return False, "no result", None
    r0 = rows[0]
    return bool(r0["ok"]), str(r0["reason"]), Path(r0["worktree"]) if r0.get("worktree") else None
