# claudeloop Remote Control

Use these instructions when you are an external assistant, OpenClaw agent, or remote controller interacting with a running `claudeloop web` instance.

## Mental Model

claudeloop owns the task loop.

- `claudeloop web` exposes a local HTTP API.
- The runner and evaluator are Claude Code panes inside a tmux session.
- The Python controller runs in the background and monitors the panes.
- The authoritative task state is `<project>/.RUD/<task>/PLAN.md`.
- Source-code changes happen inside `<project>/.RUD/<task>/work/<repo>`.

Remote control should be conservative. Prefer reading status first. Only send commands to runner/evaluator when the user explicitly asks or when the instruction is clearly safe.

## Network Setup

If OpenClaw runs on another host, connect it to claudeloop using SSH reverse forwarding from the claudeloop machine:

```bash
ssh -f -N \
  -i ~/.ssh/id_ed25519 \
  -L 18789:127.0.0.1:18789 \
  -R 8765:127.0.0.1:8765 \
  charles@34.102.85.57
```

On the OpenClaw host, claudeloop is then reachable at:

```text
http://127.0.0.1:8765/
```

OpenClaw can receive claudeloop lifecycle events on its gateway:

```text
http://127.0.0.1:18789/hooks/wake
```

For automatic follow-up work, configure claudeloop to post to:

```text
http://127.0.0.1:18789/hooks/agent
```

Use `/hooks/wake` for lightweight notification. Use `/hooks/agent` when OpenClaw should run an agent turn, inspect claudeloop, and optionally call back into `http://127.0.0.1:8765/api/...`.

## Safety Rules

1. Do not send arbitrary shell commands to runner/evaluator unless the user asked.
2. Do not stop a worker unless the user asked or the task is clearly unsafe.
3. Do not create worktrees or start new workers unless the user asked.
4. Do not write task state into the repo. Task status belongs in `.RUD/<task>/PLAN.md`.
5. Read before acting: inspect task metadata, pane output, and worker logs first.
6. Prefer high-level instructions to runner, not low-level micromanagement.
7. Avoid secrets in messages sent to panes or OpenClaw events.

## Base URL

When running from the OpenClaw host through the SSH reverse tunnel, use:

```text
http://127.0.0.1:8765
```

All examples below assume:

```bash
BASE="http://127.0.0.1:8765"
```

## Authentication

If `claudeloop web` was started with `--auth-token`, OpenClaw must know the same token. Set it before calling the API:

```bash
CLAUDELOOP_WEB_AUTH_TOKEN="the-token"
AUTH_HEADER="Authorization: Bearer $CLAUDELOOP_WEB_AUTH_TOKEN"
CURL_AUTH=(-H "$AUTH_HEADER")
```

Then use `${CURL_AUTH[@]}` on every curl request.

List registered project roots (always unscoped; call this first when automating):

```bash
curl -s "${CURL_AUTH[@]}" "$BASE/api/projects"
```

If auth is disabled, use `CURL_AUTH=()`.

The Python scripts in `scripts/` automatically read `CLAUDELOOP_WEB_AUTH_TOKEN` and send `Authorization: Bearer <token>`.

## Project selection (for `/api/project` and `/api/tasks`)

The web API needs a **registry project id** on every `/api/project` and `/api/tasks/...` request unless the server can infer a unique default (`?project=<id>` or header `X-ClaudeLoop-Project`).

### Automatic resolution (Python scripts + `cl_docs.sh`)

`scripts/cl_remote_api.py` is shared by `cl_status.py`, `cl_panes.py`, and `cl_tell_runner.py`. It calls `GET /api/projects` once and picks an id in this order:

1. **`CLAUDELOOP_PROJECT_ID`** — must match an `id` in the registry (set this when you need a specific repo).
2. **`CLAUDELOOP_PROJECT_PATH`** — absolute path of a registered project root (resolved and matched against each project’s `path`).
3. **Exactly one** registered project — use that id.
4. **Several projects** — use `currentProjectId` or `defaultProjectId` from `GET /api/projects` when that id is still registered.
5. Otherwise the helper **exits with an error** listing available ids (no silent wrong project).

Shell snippets can print the resolved id (same rules, same env):

```bash
SCRIPT_DIR=/path/to/claudeloop/skills/remote_control/scripts
export CLAUDELOOP_BASE_URL="http://127.0.0.1:8765"
PROJECT_ID="$(python3 "$SCRIPT_DIR/cl_remote_api.py" --print-project-id)"
curl -s "${CURL_AUTH[@]}" "$BASE/api/tasks?project=$PROJECT_ID"
```

`cl_docs.sh` runs the resolver automatically for task URLs.

### Bash helper for manual curl (`cl_api_scope`)

Define once per shell (after `BASE` and `CURL_AUTH`):

```bash
export CLAUDELOOP_PROJECT_ID=""  # optional; scripts auto-resolve if unset (see above)

cl_api_scope() {
  local p="$1"
  if [[ -z "${CLAUDELOOP_PROJECT_ID:-}" ]]; then
    printf '%s%s' "$BASE" "$p"
    return
  fi
  if [[ "$p" == *"?"* ]]; then
    printf '%s%s&project=%s' "$BASE" "$p" "$CLAUDELOOP_PROJECT_ID"
  else
    printf '%s%s?project=%s' "$BASE" "$p" "$CLAUDELOOP_PROJECT_ID"
  fi
}
```

Pin the session to the server default project when using **manual** curl and an empty `CLAUDELOOP_PROJECT_ID`:

```bash
export CLAUDELOOP_PROJECT_ID="$(
  curl -s "${CURL_AUTH[@]}" "$BASE/api/projects" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("defaultProjectId") or "")'
)"
```

When `CLAUDELOOP_PROJECT_ID` is empty, `cl_api_scope` does **not** append `project=` — use the Python helpers or set the id explicitly for multi-root servers.

Use `$(cl_api_scope "/api/…")` for every **project-scoped** path below: `/api/project`, `/api/tasks`, `/api/tasks/<slug>/…`. Examples use `$(cl_api_scope "/api/tasks")` instead of `"$BASE/api/tasks"`.

- **Query string:** `?project=<id>` (append `&project=<id>` if the path already has a `?`).
- **Header:** `X-ClaudeLoop-Project: <id>` (query wins if both are set).

Global endpoints such as `POST /api/tmux/send-text` do **not** take `project`. For `GET /api/tmux/sessions`, omit `project` to list every session, or pass `?project=<id>` to list only claudeloop sessions tied to that project id (same naming as the web UI).

**Registry API** (no `project=` on these paths)

- `GET /api/projects` — list projects, `defaultProjectId`, `currentProjectId`.
- `POST /api/projects` — body `{"path": "/absolute/dir"}` to register another root.
- `DELETE /api/projects/<id>` — remove from the list only (at least one project must remain).
- `POST /api/projects/<id>/activate` — set default for requests that omit `project`.

**Scripts:** `cl_status.py`, `cl_panes.py`, and `cl_tell_runner.py` resolve the project automatically via `cl_remote_api.py` (see above). `cl_docs.sh` calls `cl_remote_api.py --print-project-id` for the same behavior.

You may still set `CLAUDELOOP_PROJECT_ID` or `CLAUDELOOP_PROJECT_PATH` to force a specific repo when several are registered.

If a request fails, first check that the tunnel is still alive and that `claudeloop web` is running on the local machine.

## Standard Inspection Flow

When the user asks "what is happening?", follow this order:

1. List projects with `GET /api/projects` when multiple roots are registered; note `defaultProjectId` / `currentProjectId`.
2. List tasks with `GET /api/tasks` (include `?project=<id>` or `X-ClaudeLoop-Project` when required).
3. Pick the requested task, or the most recently updated relevant task.
4. Read the task detail with `GET /api/tasks/<task>`.
5. Read `PLAN.md`, `TASK_PROMPT.md`, and `SUCCESS_CONDITION.md` from the `templates` object.
6. Check `work_repos`; if there is an active repo, read worker status and logs.
7. Capture runner/evaluator panes if their tmux targets are present.
8. Summarize status without changing anything.

## Discover Tasks

List tasks (for **curl**, use `cl_api_scope` with `CLAUDELOOP_PROJECT_ID` set, or append `?project=$(python3 …/cl_remote_api.py --print-project-id)`; the `scripts/*.py` helpers add `project=` automatically):

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks")"
```

The response shape is:

```json
{
  "tasks": [
    {
      "slug": "qwen34b",
      "title": "qwen34B reproduction",
      "general_goal": "...",
      "created_at": "...",
      "updated_at": "...",
      "work_dirs": ["/home/charlie/CoQuant"],
      "tmux_interview_target": "",
      "tmux_ask_target": "",
      "tmux_runner_target": "claudeloop-qwen34b-CoQuant:0.0",
      "tmux_evaluator_target": "claudeloop-qwen34b-CoQuant:0.1"
    }
  ]
}
```

Use `slug` for all task-specific API calls. Use `updated_at` to identify the most recently touched task when the user does not specify one.

Current project root and bundled skills path (scoped like tasks):

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/project")" | python3 -m json.tool
```

Read one task:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>")"
```

Important fields:

- `meta`: full task metadata.
- `meta.tmux_interview_target`: interview pane target, if a deep-interview is running.
- `meta.tmux_ask_target`: on-demand Ask pane target, if running.
- `meta.tmux_runner_target`: runner pane target, for example `claudeloop-qwen34b-CoQuant:0.0`.
- `meta.tmux_evaluator_target`: evaluator pane target, for example `claudeloop-qwen34b-CoQuant:0.1`.
- `work_repos`: repo keys with prepared worktrees. Use these as the `repo` query/body value.
- `templates`: markdown task documents keyed by filename. The only
  per-task file in current builds is `PLAN.md` - older `INTERVIEW.md` /
  `TASK_PROMPT.md` / `SUCCESS_CONDITION.md` files are no longer written
  or returned.

## Read Task Markdown

Task detail returns the three main markdown files in `templates`.

Read the plan and progress log:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>")" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["templates"].get("PLAN.md",""))'
```

Read the worker prompt:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>")" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["templates"].get("TASK_PROMPT.md",""))'
```

Read the success condition:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>")" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["templates"].get("SUCCESS_CONDITION.md",""))'
```

Interpretation:

- `PLAN.md` is authoritative for goal, progress, status, blockers, and
  next steps. It is now the only per-task markdown file.

Do not create new status files (INTERVIEW.md, TASK_PROMPT.md,
SUCCESS_CONDITION.md, TODO.md, …). If asked to update task state, write
into `PLAN.md`.

## Get Repos And Pane Targets

Print prepared worktree repo keys:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>")" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("\n".join(d.get("work_repos", [])))'
```

Print tmux targets:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>")" \
  | python3 -c 'import json,sys; m=json.load(sys.stdin)["meta"]; print("interview=", m.get("tmux_interview_target","")); print("ask=", m.get("tmux_ask_target","")); print("runner=", m.get("tmux_runner_target","")); print("evaluator=", m.get("tmux_evaluator_target",""))'
```

If `work_repos` is empty, no worktree has been created yet. Do not create one unless the user asks.

If runner/evaluator targets are empty, the worker has not started or the metadata has not been bound yet.

## Check Loop Status

Read worker status and controller log:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>/worker/log?repo=<repo>")"
```

The response shape is:

```json
{
  "tail": "recent controller log output...",
  "status": {
    "running": true,
    "pid": 12345,
    "returncode": null,
    "session": {
      "status": "running",
      "completed_iteration": 3,
      "current_round": 4,
      "max_rounds": 200,
      "updated_at": "...",
      "tmux_session_name": "claudeloop-qwen34b-CoQuant",
      "tmux_evaluator_target": "claudeloop-qwen34b-CoQuant:0.1"
    }
  }
}
```

How to interpret it:

- `status.running=true`: controller process is currently alive.
- `status.pid`: controller process id on the claudeloop machine.
- `status.returncode`: process exit code if the controller has exited.
- `status.session.status`: controller's own loop status, usually `running`, `succeeded`, `failed`, or `unknown`.
- `status.session.current_round`: current loop round being worked on.
- `status.session.completed_iteration`: last completed loop iteration.
- `status.session.max_rounds`: configured maximum rounds.
- `tail`: recent controller log. Use this to find errors, stop reasons, or current decisions.

If `status.running=false` but `status.session.status=running`, the controller may have crashed or the web server restarted after the worker began. Report that distinction.

Quick one-line status summary:

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>/worker/log?repo=<repo>")" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); s=d.get("status",{}); ss=s.get("session",{}); print("running=%s pid=%s round=%s/%s completed=%s state=%s" % (s.get("running"), s.get("pid"), ss.get("current_round"), ss.get("max_rounds"), ss.get("completed_iteration"), ss.get("status")))'
```

## Read Tmux Panes

Capture runner:

```bash
curl -s "${CURL_AUTH[@]}" "$BASE/api/tmux/capture?target=<runner-target>&lines=160"
```

Capture evaluator:

```bash
curl -s "${CURL_AUTH[@]}" "$BASE/api/tmux/capture?target=<evaluator-target>&lines=160"
```

Capture interview:

```bash
curl -s "${CURL_AUTH[@]}" "$BASE/api/tmux/capture?target=<interview-target>&lines=160"
```

Use pane output to understand current state before sending anything.

## Read Worker Log

```bash
curl -s "${CURL_AUTH[@]}" "$(cl_api_scope "/api/tasks/<task>/worker/log?repo=<repo>")"
```

The response includes:

- `tail`: recent controller log output.
- `status.running`: whether the background controller process is still running.
- `status.pid`: controller process id, if known.
- `status.returncode`: return code if exited.
- `status.session.current_round`: current round.
- `status.session.max_rounds`: total configured rounds.
- `status.session.completed_iteration`: completed iterations.

## Send Text To Runner

Use this when the user wants you to nudge the worker.

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$BASE/api/tmux/send-text" \
  -H 'Content-Type: application/json' \
  -d '{
    "target": "<runner-target>",
    "text": "Please summarize current progress, update PLAN.md, and continue with the next safe step.",
    "submit": true
  }'
```

Good runner messages:

- "Please update PLAN.md with current progress and next steps."
- "Please pause after the current command finishes and report blockers."
- "Please rerun the tests from SUCCESS_CONDITION.md and summarize failures."
- "Please avoid changing files outside the worktree and task PLAN.md."

Bad runner messages:

- Vague: "do it"
- Secret-bearing: "use this token: ..."
- Dangerous without confirmation: "delete all generated data"
- Repo-state destructive: "reset hard and force push"

## Send Text To Evaluator

Use this when the evaluator needs to re-check success conditions or explain a failure.

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$BASE/api/tmux/send-text" \
  -H 'Content-Type: application/json' \
  -d '{
    "target": "<evaluator-target>",
    "text": "Please rerun the success-condition tests and return a concise pass/fail judgment.",
    "submit": true
  }'
```

## Send Keys

Supported keys include `Enter`, `Up`, `Down`, `Left`, `Right`, `Escape`, `C-c`, `C-d`, `Tab`, and `Backspace`.

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$BASE/api/tmux/send-key" \
  -H 'Content-Type: application/json' \
  -d '{
    "target": "<runner-target>",
    "key": "Enter"
  }'
```

Use `C-c` carefully. It interrupts the foreground process in that pane.

## Stop Worker

Only stop when the user asks or when continuing is unsafe.

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/worker/stop")" \
  -H 'Content-Type: application/json' \
  -d '{
    "repo": "<repo>"
  }'
```

This stops the background controller process and kills the associated tmux session.

## Start Workflows

Start interview:

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/interview/start")" \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Stop interview:

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/interview/stop")" \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Start Ask pane:

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/ask/start")" \
  -H 'Content-Type: application/json' \
  -d '{
    "repo": "<repo>"
  }'
```

Stop Ask pane:

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/ask/stop")" \
  -H 'Content-Type: application/json' \
  -d '{}'
```

The Ask pane starts Claude Opus 4.7 with `effort=max` in the task directory. Use it for question-answering about task files, worktrees, and run logs. Do not use it to modify files unless the user explicitly asks.

Create worktree:

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/worktrees")" \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Start worker:

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/worker/start")" \
  -H 'Content-Type: application/json' \
  -d '{
    "repo": "<repo>",
    "model": "claude-opus-4-8",
    "max_iters": 200
  }'
```

Only use these lifecycle actions after explicit user approval.

Push worktree changes to the task branch:

```bash
curl -s "${CURL_AUTH[@]}" -X POST "$(cl_api_scope "/api/tasks/<task>/worker/push")" \
  -H 'Content-Type: application/json' \
  -d '{
    "repo": "<repo>"
  }'
```

Only push after explicit user approval.

## Status Report Template

When reporting status to the user, include:

- Project: resolved id (`python3 …/cl_remote_api.py --print-project-id`) or explicit `CLAUDELOOP_PROJECT_ID` / `CLAUDELOOP_PROJECT_PATH`, plus task `<task>` / repo `<repo>`.
- Controller: running/stopped, pid, return code if known.
- Loop: current round, completed iteration, max rounds.
- Plan: latest progress/blockers from `PLAN.md`.
- Runner: what the runner pane appears to be doing.
- Evaluator: latest pass/fail signal, or empty if not started.
- Next action: one safe suggested next step.

Do not over-report raw logs. Quote only the most relevant lines.

## Example Code

These examples are safe building blocks for an OpenClaw agent or another external assistant.

Ready-to-run scripts are also included next to this document:

```text
claudeloop/skills/remote_control/scripts/cl_remote_api.py
claudeloop/skills/remote_control/scripts/cl_status.py
claudeloop/skills/remote_control/scripts/cl_docs.sh
claudeloop/skills/remote_control/scripts/cl_panes.py
claudeloop/skills/remote_control/scripts/cl_tell_runner.py
```

Basic usage:

```bash
export CLAUDELOOP_BASE_URL="http://127.0.0.1:8765"
export CLAUDELOOP_WEB_AUTH_TOKEN="optional-web-auth-token"
# Optional when several repos are registered:
# export CLAUDELOOP_PROJECT_ID="<id>"   # or: export CLAUDELOOP_PROJECT_PATH="/abs/path/to/repo"

python3 claudeloop/skills/remote_control/scripts/cl_status.py
bash claudeloop/skills/remote_control/scripts/cl_docs.sh <task-slug>
python3 claudeloop/skills/remote_control/scripts/cl_panes.py <task-slug>
python3 claudeloop/skills/remote_control/scripts/cl_tell_runner.py \
  "Please update PLAN.md with current progress and blockers." <task-slug>
```

Only run `cl_tell_runner.py` after the user explicitly asks OpenClaw to send an instruction.

All snippets below use the same pattern as the `scripts/` tools: optional `CLAUDELOOP_PROJECT_ID`, and `with_project()` on every `/api/project` and `/api/tasks` URL path.

### List Tasks

```python
#!/usr/bin/env python3
import json
import os
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8765"
TOKEN = os.environ.get("CLAUDELOOP_WEB_AUTH_TOKEN", "")
PROJECT_ID = os.environ.get("CLAUDELOOP_PROJECT_ID", "").strip()


def with_project(path: str) -> str:
    if not PROJECT_ID:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}project={urllib.parse.quote(PROJECT_ID)}"


def get_json(path: str) -> dict:
    req = urllib.request.Request(BASE + path)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


projects_payload = get_json("/api/projects")
print("projects:", len(projects_payload.get("projects", [])))
data = get_json(with_project("/api/tasks"))
for task in data.get("tasks", []):
    print(f"{task['slug']}\tupdated={task.get('updated_at','')}\ttitle={task.get('title','')}")
```

### Pick Latest Task And Read Markdown

```python
#!/usr/bin/env python3
import json
import os
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8765"
TOKEN = os.environ.get("CLAUDELOOP_WEB_AUTH_TOKEN", "")
PROJECT_ID = os.environ.get("CLAUDELOOP_PROJECT_ID", "").strip()


def with_project(path: str) -> str:
    if not PROJECT_ID:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}project={urllib.parse.quote(PROJECT_ID)}"


def get_json(path: str) -> dict:
    req = urllib.request.Request(BASE + path)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


tasks = get_json(with_project("/api/tasks")).get("tasks", [])
if not tasks:
    raise SystemExit("no claudeloop tasks found")

latest = sorted(tasks, key=lambda t: t.get("updated_at", ""), reverse=True)[0]
slug = latest["slug"]
detail = get_json(with_project(f"/api/tasks/{urllib.parse.quote(slug)}"))
templates = detail.get("templates", {})

print(f"# Task: {slug}")
print(f"Title: {detail['meta'].get('title','')}")
print("\n## PLAN.md\n")
print(templates.get("PLAN.md", ""))
print("\n## TASK_PROMPT.md\n")
print(templates.get("TASK_PROMPT.md", ""))
print("\n## SUCCESS_CONDITION.md\n")
print(templates.get("SUCCESS_CONDITION.md", ""))
```

### Get Loop Status

```python
#!/usr/bin/env python3
import json
import os
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8765"
TOKEN = os.environ.get("CLAUDELOOP_WEB_AUTH_TOKEN", "")
PROJECT_ID = os.environ.get("CLAUDELOOP_PROJECT_ID", "").strip()


def with_project(path: str) -> str:
    if not PROJECT_ID:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}project={urllib.parse.quote(PROJECT_ID)}"


def get_json(path: str) -> dict:
    req = urllib.request.Request(BASE + path)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


task = "qwen34b"
repo = "CoQuant"
path = f"/api/tasks/{urllib.parse.quote(task)}/worker/log?repo={urllib.parse.quote(repo)}"
data = get_json(with_project(path))
status = data.get("status", {})
session = status.get("session", {})

print(f"controller_running={status.get('running')}")
print(f"pid={status.get('pid')}")
print(f"returncode={status.get('returncode')}")
print(f"loop_state={session.get('status')}")
print(f"round={session.get('current_round')}/{session.get('max_rounds')}")
print(f"completed_iteration={session.get('completed_iteration')}")
print("\n## Controller log tail\n")
print(data.get("tail", ""))
```

### Capture Runner And Evaluator

```python
#!/usr/bin/env python3
import json
import os
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8765"
TOKEN = os.environ.get("CLAUDELOOP_WEB_AUTH_TOKEN", "")
PROJECT_ID = os.environ.get("CLAUDELOOP_PROJECT_ID", "").strip()


def with_project(path: str) -> str:
    if not PROJECT_ID:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}project={urllib.parse.quote(PROJECT_ID)}"


def get_text(path: str) -> str:
    req = urllib.request.Request(BASE + path)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode("utf-8", "replace")


def get_json(path: str) -> dict:
    return json.loads(get_text(path))


task = "qwen34b"
detail = get_json(with_project(f"/api/tasks/{urllib.parse.quote(task)}"))
meta = detail["meta"]

for label, target in [
    ("runner", meta.get("tmux_runner_target", "")),
    ("evaluator", meta.get("tmux_evaluator_target", "")),
    ("interview", meta.get("tmux_interview_target", "")),
]:
    if not target:
        print(f"\n## {label}: not running\n")
        continue
    output = get_text(
        "/api/tmux/capture?"
        + urllib.parse.urlencode({"target": target, "lines": "120"})
    )
    print(f"\n## {label}: {target}\n")
    print(output)
```

### Send A Safe Runner Instruction

Only do this when the user explicitly asks OpenClaw to control claudeloop.

```python
#!/usr/bin/env python3
import json
import os
import urllib.request

BASE = "http://127.0.0.1:8765"
TOKEN = os.environ.get("CLAUDELOOP_WEB_AUTH_TOKEN", "")


def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


runner_target = "claudeloop-qwen34b-CoQuant:0.0"
message = (
    "Please pause after the current command finishes, update PLAN.md with "
    "current progress/blockers/next steps, and avoid changing files outside "
    "the worktree."
)

print(post_json("/api/tmux/send-text", {
    "target": runner_target,
    "text": message,
    "submit": True,
}))
```

### OpenClaw Interaction Examples

Useful user-facing commands to OpenClaw:

- "Call `GET /api/projects`, then check claudeloop status for the latest task in that root. Read `PLAN.md`, worker status, and runner/evaluator panes. Do not send commands."
- "For project id `<id>`, task `qwen34b` repo `CoQuant`, tell the runner to update `PLAN.md` and report blockers."
- "Capture the runner pane for task `qwen34b` (project `<id>`) and summarize what command it is running."
- "Read `SUCCESS_CONDITION.md` and ask the evaluator to rerun the pass/fail check."
- "Stop the worker for task `qwen34b` repo `CoQuant` (project `<id>`)." Only do this after explicit confirmation.

Recommended OpenClaw control loop:

1. Call `GET /api/projects`; choose `defaultProjectId` or the path the user named; export `CLAUDELOOP_PROJECT_ID` for scripts or append `?project=` on task URLs.
2. Resolve the task slug and repo (`GET /api/tasks?project=...`).
3. Read task detail, `PLAN.md`, and loop status (same `project=` on each call).
4. Capture runner/evaluator panes.
5. Report what is happening.
6. Ask for confirmation before sending text, interrupting, stopping, creating worktrees, or pushing branches.

## Recommended OpenClaw Behavior

When claudeloop sends a `worker-start` event:

1. Acknowledge the worker started.
2. Confirm `projectRoot` / `projectId` from the event payload or `GET /api/projects`; read task metadata and worker log with the same `project=` scope.
3. Do not send a command unless asked.
4. If asked for status, summarize `PLAN.md`, recent runner output, and worker log.

When claudeloop sends a `worker-stop` event:

1. Report that the controller and tmux session stopped.
2. Read `PLAN.md` if available.
3. Summarize final status and likely next step.

When the user says "tell the runner ...":

1. Resolve project id (`GET /api/projects` or `CLAUDELOOP_PROJECT_ID`), then the active task and repo.
2. Read `meta.tmux_runner_target`.
3. Send the user's instruction with `submit=true`.
4. Capture the pane again after a short delay and report what changed.

When the user says "what is happening?":

1. Resolve project id if multiple roots exist; read task metadata (`GET /api/tasks?project=...` then task detail).
2. Read worker log.
3. Capture runner and evaluator panes.
4. Summarize new progress only.

## Response Style

Be concise and operational.

- Say which **project** (`CLAUDELOOP_PROJECT_ID` or path) and task/repo you inspected.
- Say whether controller is running.
- Mention the runner/evaluator targets if relevant.
- Summarize current progress from `PLAN.md` and logs.
- Ask before taking destructive or interrupting actions.
