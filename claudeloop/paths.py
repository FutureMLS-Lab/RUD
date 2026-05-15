"""Paths to bundled resources (skills, web static)."""

from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent


def bundled_skills_path() -> Path:
    return _PKG / "skills" / "AK_skills.md"


def web_static_dir() -> Path:
    return _PKG / "web_static"
