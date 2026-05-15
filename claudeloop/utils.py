"""Shared utilities for claudeloop."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging with a file handler.

    The file always captures DEBUG-level detail (tool inputs, outputs,
    raw event data, etc.).  Console output is handled separately by
    Rich in the runner.
    """
    logger = logging.getLogger("claudeloop")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "claudeloop.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def parse_json_response(text: str) -> dict | None:
    """Parse JSON from a Claude response, handling markdown fences."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def extract_bash_commands(markdown_text: str) -> list[str]:
    """Extract commands from ```bash fenced code blocks in markdown."""
    pattern = r"```bash\n(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    commands: list[str] = []
    for block in matches:
        for line in block.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line)
    return commands


def truncate(text: str, max_chars: int = 10000) -> str:
    """Truncate long text, keeping start and end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    removed = len(text) - max_chars
    return text[:half] + f"\n\n... [{removed} chars truncated] ...\n\n" + text[-half:]
