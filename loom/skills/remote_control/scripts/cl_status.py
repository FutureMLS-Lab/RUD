#!/usr/bin/env python3
"""Print latest loom task, markdown summary, and worker status.

Project scope is resolved automatically via GET /api/projects unless you set:
  LOOM_PROJECT_ID       — registry id
  LOOM_PROJECT_PATH     — absolute path matching a registered project root
See remote_control.md.
"""

from __future__ import annotations

import urllib.parse

from cl_remote_api import get_json


def main() -> None:
    tasks = get_json("/api/tasks").get("tasks", [])
    if not tasks:
        raise SystemExit("no loom tasks found for resolved project")
    task = sorted(tasks, key=lambda t: t.get("updated_at", ""), reverse=True)[0]
    slug = task["slug"]
    detail = get_json(f"/api/tasks/{urllib.parse.quote(slug)}")
    templates = detail.get("templates", {})
    repos = detail.get("work_repos") or []

    print(f"task={slug}")
    print(f"title={detail.get('meta', {}).get('title', '')}")
    print(f"repos={', '.join(repos) if repos else '(none)'}")
    print("\n## PLAN.md first 2000 chars\n")
    print(templates.get("PLAN.md", "")[:2000])

    if not repos:
        return
    repo = repos[0]
    log = get_json(
        f"/api/tasks/{urllib.parse.quote(slug)}/worker/log?repo={urllib.parse.quote(repo)}"
    )
    status = log.get("status", {})
    session = status.get("session", {})
    print("\n## Worker\n")
    print(f"repo={repo}")
    print(f"running={status.get('running')} pid={status.get('pid')} returncode={status.get('returncode')}")
    print(
        "round=%s/%s completed=%s state=%s"
        % (
            session.get("current_round"),
            session.get("max_rounds"),
            session.get("completed_iteration"),
            session.get("status"),
        )
    )
    print("\n## Worker log tail\n")
    print(log.get("tail", ""))


if __name__ == "__main__":
    main()
