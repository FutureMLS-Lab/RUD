"""Non-blocking ESC key listener for pause functionality."""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty
from typing import Any


class KeyListener:
    """Listens for ESC key in a background thread to signal pause requests.

    Usage::

        with KeyListener() as listener:
            while doing_work:
                if listener.pause_requested.is_set():
                    handle_pause()
                    listener.reset()
    """

    def __init__(self) -> None:
        self.pause_requested = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_settings: list[Any] | None = None
        try:
            fd = sys.stdin.fileno()
            self._is_tty = os.isatty(fd) and os.tcgetpgrp(fd) == os.getpgrp()
        except (AttributeError, OSError, ValueError):
            self._is_tty = False

    def __enter__(self) -> KeyListener:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening for ESC key."""
        if not self._is_tty:
            return
        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._stop.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listener and restore terminal."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._restore_terminal()

    def reset(self) -> None:
        """Clear the pause_requested flag so the loop can continue."""
        self.pause_requested.clear()

    # ------------------------------------------------------------------
    # Terminal mode toggling (for interactive pause prompt)
    # ------------------------------------------------------------------

    def pause_listening(self) -> None:
        """Restore cooked terminal mode so ``input()`` works normally."""
        if not self._is_tty or self._old_settings is None:
            return
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)

    def resume_listening(self) -> None:
        """Re-enable cbreak mode for ESC detection."""
        if not self._is_tty:
            return
        tty.setcbreak(sys.stdin.fileno())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _restore_terminal(self) -> None:
        if self._old_settings is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings,
                )
            except (ValueError, termios.error):
                pass
            self._old_settings = None

    def _listen_loop(self) -> None:
        """Background thread: poll stdin for bare ESC key."""
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([fd], [], [], 0.1)
            except (ValueError, OSError):
                break
            if not readable:
                continue
            try:
                ch = os.read(fd, 1)
            except OSError:
                break
            if ch == b"\x1b":
                # Might be the start of an escape sequence (e.g. arrow keys).
                # Wait briefly — if more bytes arrive it's a sequence, not bare ESC.
                try:
                    more, _, _ = select.select([fd], [], [], 0.05)
                except (ValueError, OSError):
                    break
                if more:
                    # Drain remaining escape-sequence bytes.
                    try:
                        os.read(fd, 32)
                    except OSError:
                        pass
                else:
                    self.pause_requested.set()
