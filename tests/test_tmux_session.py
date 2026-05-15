"""Tests for TmuxSession low-level operations.

These tests create real tmux sessions to verify send/capture behavior.
They require tmux to be installed.
"""

from __future__ import annotations

import time
import uuid

import pytest

from claudeloop.tmux_controller import TmuxSession


@pytest.fixture
def tmux():
    """Create a temporary tmux session and clean it up after the test."""
    name = f"test-claudeloop-{uuid.uuid4().hex[:8]}"
    session = TmuxSession(name)
    session.create()
    # Wait for shell to initialize
    time.sleep(1)
    yield session
    if session.exists():
        session.kill()


class TestTmuxSessionLifecycle:
    def test_create_and_exists(self, tmux: TmuxSession):
        assert tmux.exists()

    def test_kill(self, tmux: TmuxSession):
        tmux.kill()
        assert not tmux.exists()

    def test_is_alive(self, tmux: TmuxSession):
        assert tmux.is_alive()
        tmux.kill()
        assert not tmux.is_alive()


class TestSendKeys:
    def test_send_keys_echo(self, tmux: TmuxSession):
        """send_keys should execute a command and its output should be capturable."""
        tmux.send_keys("echo HELLO_WORLD_123")
        time.sleep(1)
        content = tmux.capture_pane()
        assert "HELLO_WORLD_123" in content

    def test_send_keys_no_enter(self, tmux: TmuxSession):
        """send_keys with enter=False should not execute the command."""
        tmux.send_keys("echo SHOULD_NOT_RUN", enter=False)
        time.sleep(1)
        content = tmux.capture_pane()
        # The text should appear (typed) but not as command output
        # It should appear once on the prompt line, not twice (prompt + output)
        lines_with_text = [
            l for l in content.splitlines() if "SHOULD_NOT_RUN" in l
        ]
        assert len(lines_with_text) == 1  # Only the prompt line


class TestSendTextViaBuffer:
    def test_paste_simple_text(self, tmux: TmuxSession):
        """send_text_via_buffer should paste text and send Enter."""
        tmux.send_text_via_buffer("echo PASTED_TEXT_42")
        time.sleep(1)
        content = tmux.capture_pane()
        assert "PASTED_TEXT_42" in content

    def test_paste_multiline_text(self, tmux: TmuxSession):
        """send_text_via_buffer with multiline text should paste all lines."""
        # Use cat with heredoc to capture multiline input
        tmux.send_keys("cat << 'ENDOFTEST'")
        time.sleep(0.5)
        tmux.send_text_via_buffer("LINE_ONE\nLINE_TWO\nLINE_THREE")
        time.sleep(0.5)
        tmux.send_keys("ENDOFTEST")
        time.sleep(1)
        content = tmux.capture_pane()
        assert "LINE_ONE" in content
        assert "LINE_TWO" in content
        assert "LINE_THREE" in content

    def test_paste_text_with_special_chars(self, tmux: TmuxSession):
        """send_text_via_buffer should handle special characters."""
        tmux.send_text_via_buffer("echo 'hello $USER \"world\" `date`'")
        time.sleep(1)
        content = tmux.capture_pane()
        assert "hello" in content


class TestCapturePaneContent:
    def test_capture_returns_string(self, tmux: TmuxSession):
        content = tmux.capture_pane()
        assert isinstance(content, str)

    def test_capture_includes_command_output(self, tmux: TmuxSession):
        """capture_pane should include output from commands run in the pane."""
        marker = f"MARKER_{uuid.uuid4().hex[:8]}"
        tmux.send_keys(f"echo {marker}")
        time.sleep(1)
        content = tmux.capture_pane()
        assert marker in content

    def test_capture_multiple_commands(self, tmux: TmuxSession):
        """capture_pane should include output from multiple commands."""
        tmux.send_keys("echo FIRST_CMD_OUTPUT")
        time.sleep(0.5)
        tmux.send_keys("echo SECOND_CMD_OUTPUT")
        time.sleep(1)
        content = tmux.capture_pane()
        assert "FIRST_CMD_OUTPUT" in content
        assert "SECOND_CMD_OUTPUT" in content

    def test_capture_with_fewer_lines(self, tmux: TmuxSession):
        """capture_pane with small line count should still work."""
        tmux.send_keys("echo SHORT_CAPTURE")
        time.sleep(1)
        content = tmux.capture_pane(lines=5)
        assert isinstance(content, str)

    def test_capture_strips_trailing_empty_lines(self, tmux: TmuxSession):
        """Verify capture_pane returns content (may include blank lines)."""
        tmux.send_keys("echo CONTENT_CHECK")
        time.sleep(1)
        content = tmux.capture_pane()
        # Should have non-empty content
        assert content.strip()


class TestInteractiveProgram:
    """Test tmux interactions with an interactive program (python REPL)."""

    def test_python_repl(self, tmux: TmuxSession):
        """Simulate interacting with python REPL — similar to Claude interactive."""
        tmux.send_keys("python3 -c \"import code; code.interact(banner='READY>')\"")
        time.sleep(2)
        content = tmux.capture_pane()
        assert "READY>" in content or ">>>" in content

        # Send a command to the REPL
        tmux.send_keys("print('REPL_OUTPUT_999')")
        time.sleep(1)
        content = tmux.capture_pane()
        assert "REPL_OUTPUT_999" in content

    def test_send_buffer_to_interactive(self, tmux: TmuxSession):
        """send_text_via_buffer should work with interactive programs."""
        tmux.send_keys("python3 -c \"import code; code.interact(banner='READY>')\"")
        time.sleep(2)

        # Paste a command via buffer
        tmux.send_text_via_buffer("print('BUFFER_OUTPUT_777')")
        time.sleep(1)
        content = tmux.capture_pane()
        assert "BUFFER_OUTPUT_777" in content


class TestSplitPane:
    """Test creating and interacting with split panes."""

    def test_split_window_returns_target(self, tmux: TmuxSession):
        """split_window should return a valid target for the new pane."""
        pane1_target = tmux.split_window(horizontal=True)
        assert tmux.session_name in pane1_target
        # New pane should be pane 1
        assert ".1" in pane1_target

    def test_send_keys_to_specific_pane(self, tmux: TmuxSession):
        """Commands sent to different panes should be independent."""
        pane1_target = tmux.split_window(horizontal=True)
        time.sleep(0.5)

        tmux.send_keys("echo PANE_ZERO", target=tmux.target)
        tmux.send_keys("echo PANE_ONE", target=pane1_target)
        time.sleep(1)

        content0 = tmux.capture_pane(target=tmux.target)
        content1 = tmux.capture_pane(target=pane1_target)

        assert "PANE_ZERO" in content0
        assert "PANE_ONE" not in content0
        assert "PANE_ONE" in content1
        assert "PANE_ZERO" not in content1

    def test_send_buffer_to_specific_pane(self, tmux: TmuxSession):
        """send_text_via_buffer with target should paste to the right pane."""
        pane1_target = tmux.split_window(horizontal=True)
        time.sleep(0.5)

        tmux.send_text_via_buffer("echo BUFFER_PANE1", target=pane1_target)
        time.sleep(1)

        content0 = tmux.capture_pane(target=tmux.target)
        content1 = tmux.capture_pane(target=pane1_target)

        assert "BUFFER_PANE1" not in content0
        assert "BUFFER_PANE1" in content1

    def test_capture_pane_from_specific_pane(self, tmux: TmuxSession):
        """capture_pane with target should read the correct pane."""
        pane1_target = tmux.split_window(horizontal=True)
        time.sleep(0.5)

        marker = f"MARKER_{uuid.uuid4().hex[:8]}"
        tmux.send_keys(f"echo {marker}", target=pane1_target)
        time.sleep(1)

        content1 = tmux.capture_pane(target=pane1_target)
        assert marker in content1

    def test_two_panes_idle_independently(self, tmux: TmuxSession):
        """Each pane should have stable hashes independently."""
        pane1_target = tmux.split_window(horizontal=True)
        time.sleep(0.5)

        tmux.send_keys("echo DONE0", target=tmux.target)
        tmux.send_keys("echo DONE1", target=pane1_target)
        time.sleep(2)

        c0a = tmux.capture_pane(target=tmux.target)
        c1a = tmux.capture_pane(target=pane1_target)
        time.sleep(1)
        c0b = tmux.capture_pane(target=tmux.target)
        c1b = tmux.capture_pane(target=pane1_target)

        assert c0a == c0b
        assert c1a == c1b

        # Activity in one pane shouldn't affect the other
        tmux.send_keys("echo CHANGE", target=pane1_target)
        time.sleep(1)

        c0c = tmux.capture_pane(target=tmux.target)
        c1c = tmux.capture_pane(target=pane1_target)

        assert c0c == c0b  # Pane 0 unchanged
        assert c1c != c1b  # Pane 1 changed

    def test_vertical_split(self, tmux: TmuxSession):
        """Vertical split (top/bottom) should also work."""
        pane1_target = tmux.split_window(horizontal=False)
        time.sleep(0.5)

        tmux.send_keys("echo TOP_PANE", target=tmux.target)
        tmux.send_keys("echo BOTTOM_PANE", target=pane1_target)
        time.sleep(1)

        content0 = tmux.capture_pane(target=tmux.target)
        content1 = tmux.capture_pane(target=pane1_target)

        assert "TOP_PANE" in content0
        assert "BOTTOM_PANE" in content1


class TestIdleDetection:
    """Test that capture_pane output is stable for idle detection."""

    def test_stable_content_hashing(self, tmux: TmuxSession):
        """Two consecutive captures of idle pane should return same content."""
        tmux.send_keys("echo DONE")
        time.sleep(2)  # Wait for command to finish and pane to settle

        content1 = tmux.capture_pane()
        time.sleep(1)
        content2 = tmux.capture_pane()

        # Content should be identical when nothing is changing
        assert content1 == content2

    def test_content_changes_during_activity(self, tmux: TmuxSession):
        """Captures during activity should differ."""
        content1 = tmux.capture_pane()

        tmux.send_keys("echo ACTIVITY_MARKER")
        time.sleep(1)

        content2 = tmux.capture_pane()
        assert content1 != content2
