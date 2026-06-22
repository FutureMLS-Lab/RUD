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
    "Tab",
    "BTab",
    "Backspace",
    "BSpace",
    "Space",
    "Home",
    "End",
    "DC",
    "IC",
    "PageUp",
    "PageDown",
}
# Also accept Ctrl-/Alt-<letter> combos and function keys. These are validated
# against a strict pattern and passed as a single argv (never via a shell), so
# they can't smuggle options or shell metacharacters.
_KEY_RE = re.compile(r"^(?:[CM]-[A-Za-z]|F[0-9]{1,2})$")


def tmux_subprocess_env() -> dict[str, str]:
    """Run tmux commands against the current user's default socket.

    ``loom web`` is often launched from inside tmux or through ``su``.
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
    if k not in _KEYS and not _KEY_RE.match(k):
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


def _exit_copy_mode_if_active(t: str, env: dict) -> None:
    """Leave tmux copy-mode if the pane is in it, so subsequent input reaches
    the running program. While scrolled up (copy-mode), tmux otherwise consumes
    every keystroke (arrows move the copy cursor, symbols/typing do nothing)."""
    try:
        chk = subprocess.run(
            ["tmux", "display-message", "-p", "-t", t, "#{pane_in_mode}"],
            capture_output=True, text=True, env=env, timeout=5,
        )
        if chk.returncode == 0 and chk.stdout.strip() == "1":
            subprocess.run(
                ["tmux", "send-keys", "-t", t, "-X", "cancel"],
                capture_output=True, text=True, env=env, timeout=5,
            )
    except (OSError, subprocess.TimeoutExpired):
        pass


def send_pane_literal(target: str, text: str) -> tuple[bool, str]:
    """Send literal text to a pane fast via ``tmux send-keys -l`` (no Enter).

    Used by the native-terminal keystroke forwarding so each typed character (or
    a short burst of them) reaches the pane immediately, without the heavier
    load-buffer/paste path that ``send_pane_text`` uses for big blocks.
    """
    import shutil

    if not shutil.which("tmux"):
        return False, "tmux not on PATH"
    t = target.strip()
    if not _TARGET_RE.match(t):
        return False, "invalid pane target (expected session:window.pane)"
    if not isinstance(text, str):
        return False, "text must be a string"
    if text == "":
        return True, ""
    if len(text) > 10000:
        return False, "text too long"
    env = tmux_subprocess_env()
    # If the user scrolled up (copy-mode), leave it so typing reaches the program.
    _exit_copy_mode_if_active(t, env)
    try:
        r = subprocess.run(
            ["tmux", "send-keys", "-t", t, "-l", "--", text],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "send-literal failed").strip()
    return True, ""


def open_pane_attach(target: str, cols: int = 80, rows: int = 24):
    """Open a PTY running ``tmux attach-session`` to *target*, sized cols x rows.

    Returns ``(proc, master_fd)`` on success or ``(None, None)`` on failure. The
    caller reads ``master_fd`` (the live terminal byte stream for xterm.js),
    then must ``proc.terminate()`` and ``os.close(master_fd)`` when done. Input
    is delivered separately via ``send_pane_literal`` / ``send_pane_key``.
    """
    import pty
    import struct
    import fcntl
    import termios
    import shutil

    if not shutil.which("tmux"):
        return None, None
    t = target.strip()
    if not _TARGET_RE.match(t):
        return None, None
    try:
        cols = max(20, min(500, int(cols)))
        rows = max(5, min(300, int(rows)))
    except (TypeError, ValueError):
        cols, rows = 80, 24
    env = tmux_subprocess_env()
    env["TERM"] = "xterm-256color"
    # If the session doesn't exist, don't spawn a doomed `tmux attach` - it would
    # just print "can't find session: ..." into the stream. Signal "not alive" so
    # the caller can show a friendly message instead.
    try:
        chk = subprocess.run(
            ["tmux", "has-session", "-t", t.split(":", 1)[0]],
            capture_output=True, text=True, env=env, timeout=5,
        )
        if chk.returncode != 0:
            return None, None
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    # Make the tmux window track the LATEST (this) client so the pane follows the
    # browser's xterm size: the PTY below is sized to cols x rows, and the window
    # resizes to match. We use "latest" (NOT "manual") on purpose — it always
    # re-follows when the browser refits, so the window can't get stuck at a
    # transient narrow size the way an explicit resize-window/manual pin could.
    try:
        subprocess.run(
            ["tmux", "set-option", "-t", t.split(":")[0], "window-size", "latest"],
            capture_output=True, text=True, env=env, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        master, slave = pty.openpty()
    except OSError:
        return None, None
    try:
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError:
        pass
    try:
        proc = subprocess.Popen(
            ["tmux", "attach-session", "-t", t],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            env=env,
        )
    except OSError:
        os.close(master)
        os.close(slave)
        return None, None
    os.close(slave)
    return proc, master


def scroll_pane(target: str, direction: str = "up", lines: int = 3) -> tuple[bool, str]:
    """Scroll a pane's tmux history via copy-mode (what ``Ctrl-b [`` does).

    Lets the web terminal's mouse/touchpad wheel browse scrollback instead of
    xterm converting the wheel into arrow keys for the full-screen app. Entering
    ``copy-mode -e`` is idempotent and auto-exits when the user scrolls back to
    the bottom, so live output resumes on its own.
    """
    import shutil

    if not shutil.which("tmux"):
        return False, "tmux not on PATH"
    t = target.strip()
    if not _TARGET_RE.match(t):
        return False, "invalid pane target (expected session:window.pane)"
    try:
        n = max(1, min(500, int(lines)))
    except (TypeError, ValueError):
        n = 3
    env = tmux_subprocess_env()
    if direction == "down":
        # scroll-down (send-keys -X) only works while the pane is in copy-mode.
        # If it isn't, we're already at the live bottom, so treat it as a
        # successful no-op instead of erroring out with a 400.
        try:
            chk = subprocess.run(
                ["tmux", "display-message", "-p", "-t", t, "#{pane_in_mode}"],
                capture_output=True, text=True, env=env, timeout=5,
            )
            if (chk.stdout or "").strip() != "1":
                return True, ""
        except (OSError, subprocess.TimeoutExpired):
            pass
        cmd = ["tmux", "send-keys", "-t", t, "-X", "-N", str(n), "scroll-down"]
    else:
        cmd = [
            "tmux", "copy-mode", "-e", "-t", t, ";",
            "send-keys", "-t", t, "-X", "-N", str(n), "scroll-up",
        ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    return (r.returncode == 0), (r.stderr or "").strip()


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
    # Leave copy-mode first so the paste lands in the program, not copy-mode.
    _exit_copy_mode_if_active(t, tmux_subprocess_env())
    buffer_name = f"loom-web-{uuid.uuid4().hex}"
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
