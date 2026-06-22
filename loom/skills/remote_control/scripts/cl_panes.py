#!/usr/bin/env python3
"""Capture runner, evaluator, and interview panes for one loom task.

Project scope is resolved via GET /api/projects (see cl_remote_api.py / remote_control.md).
"""

from __future__ import annotations

import sys
import urllib.parse

from cl_remote_api import get_json, get_json_raw


def latest_task_slug() -> str:
    tasks = get_json("/api/tasks").get("tasks", [])
    if not tasks:
        raise SystemExit("no loom tasks found for resolved project")
    return sorted(tasks, key=lambda t: t.get("updated_at", ""), reverse=True)[0]["slug"]


def main() -> None:
    slug = sys.argv[1] if len(sys.argv) > 1 else latest_task_slug()
    detail = get_json(f"/api/tasks/{urllib.parse.quote(slug)}")
    meta = detail["meta"]
    for label, target in [
        ("runner", meta.get("tmux_runner_target", "")),
        ("evaluator", meta.get("tmux_evaluator_target", "")),
        ("interview", meta.get("tmux_interview_target", "")),
        ("ask", meta.get("tmux_ask_target", "")),
    ]:
        print(f"\n## {label}\n")
        if not target:
            print("(not running)")
            continue
        qs = urllib.parse.urlencode({"target": target, "lines": "180"})
        data = get_json_raw(f"/api/tmux/capture?{qs}")
        print(f"target={target}\n")
        print(data.get("text", "") if data.get("ok") else data.get("error", "capture failed"))


if __name__ == "__main__":
    main()
