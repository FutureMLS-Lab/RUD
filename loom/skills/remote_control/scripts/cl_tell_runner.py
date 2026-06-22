#!/usr/bin/env python3
"""Send a confirmed instruction to the current loom runner pane.

Project scope is resolved via GET /api/projects (see cl_remote_api.py / remote_control.md).
"""

from __future__ import annotations

import json
import sys
import urllib.parse

from cl_remote_api import get_json, post_json_raw


def latest_task_slug() -> str:
    tasks = get_json("/api/tasks").get("tasks", [])
    if not tasks:
        raise SystemExit("no loom tasks found for resolved project")
    return sorted(tasks, key=lambda t: t.get("updated_at", ""), reverse=True)[0]["slug"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: cl_tell_runner.py "message" [task-slug]')
    message = sys.argv[1]
    slug = sys.argv[2] if len(sys.argv) > 2 else latest_task_slug()
    detail = get_json(f"/api/tasks/{urllib.parse.quote(slug)}")
    target = detail["meta"].get("tmux_runner_target", "")
    if not target:
        raise SystemExit(f"task {slug!r} has no runner target")
    result = post_json_raw(
        "/api/tmux/send-text",
        {"target": target, "text": message, "submit": True},
    )
    print(json.dumps({"task": slug, "target": target, "result": result}, indent=2))


if __name__ == "__main__":
    main()
