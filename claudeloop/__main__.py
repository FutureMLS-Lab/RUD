"""Allow ``python -m claudeloop`` (used by the web worker launcher)."""

from claudeloop.cli import app

if __name__ == "__main__":
    app()
