#!/usr/bin/env bash
# Dump PLAN.md for a task. Older versions also dumped TASK_PROMPT.md,
# SUCCESS_CONDITION.md, and INTERVIEW.md, but PLAN.md is now the only
# per-task markdown file.
# Resolves ?project= via scripts/cl_remote_api.py (GET /api/projects), unless you export CLAUDELOOP_PROJECT_ID / CLAUDELOOP_PROJECT_PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${CLAUDELOOP_BASE_URL:-http://127.0.0.1:8765}"
export CLAUDELOOP_BASE_URL="$BASE"

AUTH_ARGS=()
if [[ -n "${CLAUDELOOP_WEB_AUTH_TOKEN:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${CLAUDELOOP_WEB_AUTH_TOKEN}")
fi

PROJECT_ID="$(python3 "$SCRIPT_DIR/cl_remote_api.py" --print-project-id)"
PROJECT_QS="?project=${PROJECT_ID}"

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  TASK="$(curl -fsS "${AUTH_ARGS[@]}" "$BASE/api/tasks${PROJECT_QS}" \
    | python3 -c 'import json,sys; tasks=json.load(sys.stdin).get("tasks", []); tasks=sorted(tasks, key=lambda t:t.get("updated_at",""), reverse=True); print(tasks[0]["slug"] if tasks else "")')"
fi

if [[ -z "$TASK" ]]; then
  echo "no claudeloop tasks found" >&2
  exit 1
fi

curl -fsS "${AUTH_ARGS[@]}" "$BASE/api/tasks/${TASK}${PROJECT_QS}" | python3 -c '
import json
import sys

task = sys.argv[1]
d = json.load(sys.stdin)
templates = d.get("templates", {})
print(f"# {task}")
print("\n## PLAN.md\n")
print(templates.get("PLAN.md", ""))
' "$TASK"
