"""Tmux helpers for the web UI (list sessions, capture panes, send input)."""

from __future__ import annotations

import re
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

# Session:window.pane — conservative allowlist (no shell metacharacters)
_TARGET_RE = re.compile(r"^[A-Za-z0-9_.@-]+:\d+\.\d+$")
_KEYS = {
    "Enter",
    "Up",
    "Down",
    "Left",
    "Right",
    "Escape",
    "C-c",
    "C-d",
    "Tab",
    "Backspace",
}


def tmux_subprocess_env() -> dict[str, str]:
    """Run tmux commands against the current user's default socket.

    ``claudeloop web`` is often launched from inside tmux or through ``su``.
    Inheriting ``TMUX`` can point tmux clients at another user's socket
    (for example /tmp/tmux-0), which fails with Permission denied.
    """
    env = os.environ.copy()
    env.pop("TMUX", None)
    env.pop("TMUX_PANE", None)
    return env


def tmux_available() -> bool:
    import shutil

    return shutil.which("tmux") is not None


def list_tmux_sessions() -> list[dict[str, str]]:
    """Return ``[{name, attached}, ...]`` (best-effort; empty if tmux missing)."""
    import shutil

    if not shutil.which("tmux"):
        return []
    try:
        r = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}\t#{session_attached}"],
            capture_output=True,
            text=True,
            env=tmux_subprocess_env(),
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    out: list[dict[str, str]] = []
    for line in (r.stdout or "").splitlines():
        parts = line.split("\t", 1)
        if not parts:
            continue
        name = parts[0].strip()
        if not name:
            continue
        attached = parts[1].strip() if len(parts) > 1 else ""
        out.append({"name": name, "attached": attached})
    return out


def list_tmux_panes(session: str) -> list[dict[str, str]]:
    """List panes in a session: ``[{id, title}, ...]`` where id is ``session:win.pane``."""
    import shutil

    if not shutil.which("tmux"):
        return []
    if not re.match(r"^[A-Za-z0-9_.@-]+$", session):
        return []
    try:
        r = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-t",
                session,
                "-F",
                "#{session_name}:#{window_index}.#{pane_index}\t#{pane_title}",
            ],
            capture_output=True,
            text=True,
            env=tmux_subprocess_env(),
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in (r.stdout or "").splitlines():
        parts = line.split("\t", 1)
        pid = parts[0].strip() if parts else ""
        title = parts[1].strip() if len(parts) > 1 else ""
        if pid:
            rows.append({"id": pid, "title": title})
    return rows


def validate_tmux_target(t: str) -> bool:
    s = t.strip()
    if not s:
        return True
    return bool(_TARGET_RE.match(s))


def resize_window_for_capture(target: str, columns: int = 240, rows: int = 64) -> None:
    """Best-effort resize so newly rendered terminal output has enough columns."""
    import shutil

    if not shutil.which("tmux"):
        return
    t = target.strip()
    if not _TARGET_RE.match(t):
        return
    window_target = t.rsplit(".", 1)[0]
    cols = max(120, min(columns, 360))
    height = max(32, min(rows, 120))
    try:
        subprocess.run(
            ["tmux", "resize-window", "-t", window_target, "-x", str(cols), "-y", str(height)],
            capture_output=True,
            text=True,
            env=tmux_subprocess_env(),
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def capture_pane(target: str, lines: int = 80) -> tuple[bool, str]:
    """``tmux capture-pane`` for *target* ``session:win.pane``."""
    import shutil

    if not shutil.which("tmux"):
        return False, "tmux not on PATH"
    t = target.strip()
    if not _TARGET_RE.match(t):
        return False, "invalid pane target (expected session:window.pane)"
    n = max(1, min(lines, 500))
    resize_window_for_capture(t)
    try:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", t, "-p", "-S", f"-{n}"],
            capture_output=True,
            text=True,
            env=tmux_subprocess_env(),
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "capture failed").strip()
    return True, r.stdout or ""


def send_pane_key(target: str, key: str) -> tuple[bool, str]:
    """Send a single safe tmux key to ``session:window.pane``."""
    import shutil

    if not shutil.which("tmux"):
        return False, "tmux not on PATH"
    t = target.strip()
    k = key.strip()
    if not _TARGET_RE.match(t):
        return False, "invalid pane target (expected session:window.pane)"
    if k not in _KEYS:
        return False, f"unsupported key: {k}"
    try:
        r = subprocess.run(
            ["tmux", "send-keys", "-t", t, k],
            capture_output=True,
            text=True,
            env=tmux_subprocess_env(),
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "send-key failed").strip()
    return True, ""


def send_pane_text(target: str, text: str, submit: bool = False) -> tuple[bool, str]:
    """Paste text into a tmux pane via buffer; optionally submit with Enter."""
    import shutil

    if not shutil.which("tmux"):
        return False, "tmux not on PATH"
    t = target.strip()
    if not _TARGET_RE.match(t):
        return False, "invalid pane target (expected session:window.pane)"
    if not isinstance(text, str):
        return False, "text must be a string"
    buffer_name = f"claudeloop-web-{uuid.uuid4().hex}"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(text)
            tmp_path = f.name
        load = subprocess.run(
            ["tmux", "load-buffer", "-b", buffer_name, tmp_path],
            capture_output=True,
            text=True,
            env=tmux_subprocess_env(),
            timeout=5,
        )
        if load.returncode != 0:
            return False, (load.stderr or load.stdout or "load-buffer failed").strip()
        paste = subprocess.run(
            ["tmux", "paste-buffer", "-b", buffer_name, "-p", "-d", "-t", t],
            capture_output=True,
            text=True,
            env=tmux_subprocess_env(),
            timeout=5,
        )
        if paste.returncode != 0:
            return False, (paste.stderr or paste.stdout or "paste-buffer failed").strip()
        if submit:
            return send_pane_key(t, "Enter")
        return True, ""
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
