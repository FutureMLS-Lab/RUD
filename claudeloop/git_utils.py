"""Git helper functions for claudeloop."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_short_hash(cwd: Path | None = None) -> str | None:
    """Get current HEAD short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=6", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def commit_all(message: str, cwd: Path | None = None) -> str | None:
    """Stage all changes and commit. Returns short hash or None if nothing to commit."""
    cwd = cwd or Path.cwd()

    # Stage all changes
    subprocess.run(["git", "add", "-A"], cwd=cwd, capture_output=True)

    # Check if there are staged changes
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=cwd,
        capture_output=True,
    )
    if status.returncode == 0:
        return None  # Nothing to commit

    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=cwd,
        capture_output=True,
    )
    return get_short_hash(cwd)


def is_git_repo(cwd: Path | None = None) -> bool:
    """Check if current directory is inside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.returncode == 0
    except Exception:
        return False
