"""Allow ``python -m loom`` (used by the web worker launcher)."""

from loom.cli import app

if __name__ == "__main__":
    app()
