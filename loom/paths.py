"""Paths to bundled resources (skills, web static)."""

from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent


def bundled_skills_path() -> Path:
    return _PKG / "skills" / "charlie_skills.md"


def web_static_dir() -> Path:
    return _PKG / "web_static"


def kernel_hub_dir() -> Path:
    """Vendored kernel stack bundled with Loom (scaffold + kernel_evaluator
    + docker-compose). ``scaffold/agent_runner/rud_kernel.py`` lives under here and
    its ``REPO_ROOT`` resolves to this directory."""
    return _PKG / "kernel_hub"
