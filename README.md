# RUD CLAUDELOOP

`claudeloop` is a local task console for running long Claude Code jobs safely inside git worktrees. The primary workflow is the web UI: create a task, run an interactive deep interview, create a task-scoped worktree, then launch a tmux worker/evaluator loop.

## Prerequisites

Install the package in editable mode:

```bash
cd /path/to/claudeloop
pip install -e .
```

Install and authenticate Claude Code before starting workers:

```bash
claude
# complete login/auth in the Claude Code CLI
```

Install `tmux` and make sure the project you run from is a git repository:

```bash
tmux -V
git rev-parse --show-toplevel
```

## Start The Web UI

Start `claudeloop web` from the project repository you want agents to work on. That directory becomes the project root.

```bash
cd /path/to/your/project
claudeloop web --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

### Multi-project hub

One `claudeloop web` process can manage several git project roots from the same browser UI. Registered paths are stored under `~/.claudeloop/web-projects.json`. The server **does not** auto-add your current working directory: use **Add** in the sidebar to register each repo (nothing is deleted when you **Remove**—that only drops the path from the list).

Task APIs (`/api/project`, `/api/tasks`, …) are scoped with `?project=<id>` or the `X-ClaudeLoop-Project` header. `GET /api/tmux/sessions?project=<id>` lists only tmux sessions for that project. The project id is in the JSON from `GET /api/projects`. Remote-control scripts honor `CLAUDELOOP_PROJECT_ID`; see `claudeloop/skills/remote_control/remote_control.md`.

Background mode:

```bash
claudeloop web --nohup --log-file .RUD/web.log
```

Useful options:

- `--project PATH`: use a project root other than the current directory.
- `--skills PATH`: default skills markdown for new tasks. The packaged default is `claudeloop/skills/AK_skills.md`.
- `--work-dir PATH`: optional default repo source for worktree creation. If omitted, the current project root is used.
- `--interview-backend cli|sdk`: defaults to `cli`.
- `--auth-token TOKEN`: require HTTP auth for the web UI and API. Browser username can be anything; password is the token. API clients can send `Authorization: Bearer TOKEN`.
- `--openclaw`: enable direct claudeloop -> OpenClaw gateway events.
- `--openclaw-url URL`: OpenClaw hooks URL, usually `http://127.0.0.1:18789/hooks/wake`.
- `--openclaw-token TOKEN`: OpenClaw hooks token, sent as `Authorization: Bearer TOKEN`.
- `--openclaw-header "Name: value"`: request header for the gateway; repeatable.
- `--openclaw-config PATH`: claudeloop OpenClaw JSON config.
- `--openclaw-debug`: print OpenClaw delivery status.

## OpenClaw Integration

Use OpenClaw only as a notification/control bridge:

- claudeloop sends events to OpenClaw with `POST /hooks/wake`.
- OpenClaw or another remote agent can call back to claudeloop through the web API.
- Detailed commands and examples live in `claudeloop/skills/remote_control/remote_control.md`.

On the OpenClaw host, enable Gateway HTTP hooks and keep the token:

```bash
SECRET="$(openssl rand -hex 32)"
printf '{ hooks: { enabled: true, token: "%s", path: "/hooks" } }\n' "$SECRET" \
  | openclaw config patch --stdin
openclaw gateway restart
echo "$SECRET"
```

From the claudeloop machine, keep the SSH tunnel open:

```bash
ssh -N \
  -i ~/.ssh/id_ed25519 \
  -L 18789:127.0.0.1:18789 \
  -R 8765:127.0.0.1:8765 \
  charles@34.102.85.57
```

Start claudeloop with OpenClaw notifications and optional web auth:

```bash
export OPENCLAW_HOOK_TOKEN="token-from-openclaw"
export CLAUDELOOP_WEB_AUTH_TOKEN="$(openssl rand -hex 24)"

claudeloop web \
  --auth-token "$CLAUDELOOP_WEB_AUTH_TOKEN" \
  --openclaw \
  --openclaw-url http://127.0.0.1:18789/hooks/wake \
  --openclaw-token "$OPENCLAW_HOOK_TOKEN" \
  --openclaw-debug
```

For remote-control scripts, task status checks, tmux pane capture, and safe runner commands, use:

```text
claudeloop/skills/remote_control/remote_control.md
```

## Web Workflow

### 1. Create Task

Click `Create Task` and enter:

- `Title`: used to create the task slug.
- `General goal`: a rough description of what you want.

This creates:

```text
<project>/.RUD/<task>/
├── TASK_PROMPT.md
├── SUCCESS_CONDITION.md
├── PLAN.md
├── INTERVIEW.md
├── work/
└── runs/
```

Creating a task does not start tmux and does not create a worktree.

### 2. Run Deep Interview

Open the `Interview` tab and click `Start deep-interview`.

This starts a dedicated Claude Code tmux session:

```text
claudeloop-interview-<task>
```

The interview pane receives an automatic prompt with `effort=max`. You can interact with it from the web UI: send text, press Enter, arrows, Esc, or Ctrl-C. The interview should refine and write:

- `<project>/.RUD/<task>/TASK_PROMPT.md`
- `<project>/.RUD/<task>/SUCCESS_CONDITION.md`
- `<project>/.RUD/<task>/PLAN.md`

Stop it with `Stop deep-interview`, which kills the interview tmux session.

### 3. Review And Edit Markdown

Use the `PLAN.md`, `TASK_PROMPT.md`, and `SUCCESS_CONDITION.md` tabs to edit task files. Each tab has a raw Markdown editor and live preview.

`PLAN.md` is the authoritative task state file. Workers are prompted to keep task status, decisions, next steps, and progress logs in this file, especially the `Progress Log` section. Source code changes happen in the worktree; task-management notes should not be scattered through the repo.

### 4. Create Worktree

Open the `Worker` tab and click `Create worktree`.

By default this creates:

```text
<project>/.RUD/<task>/work/<repo>
```

For nested git repos one level deep, it can also create:

```text
<project>/.RUD/<task>/work/<repo>/<nested-repo>
```

The worker will run from the selected worktree, not from the original repository checkout.

### 5. Start Worker

In the `Worker` tab:

1. Select a repo from the `Repo` dropdown.
2. Choose the model and max rounds.
3. Click `Start worker`.

The web UI starts:

```bash
python -m claudeloop tmux \
  --prompt <project>/.RUD/<task>/TASK_PROMPT.md \
  --success <project>/.RUD/<task>/SUCCESS_CONDITION.md \
  --plan <project>/.RUD/<task>/PLAN.md \
  --log-dir <project>/.RUD/<task>/runs/<repo>/agent_logs \
  --max-rounds 200 \
  --model claude-opus-4-6 \
  --dangerously-skip-permissions \
  --effort max \
  --no-commit \
  --session-name claudeloop-<task>-<repo>
```

The controller process runs in the background. The runner/evaluator panes live in tmux.

### 6. Watch Or Attach

The `Runner / Evaluator` tab previews both tmux panes.

To attach manually:

```bash
tmux attach -t claudeloop-<task>-<repo>
```

Inside tmux:

- Switch panes: `Ctrl-b` then arrow key.
- Detach without stopping: `Ctrl-b d`.

Logs:

```text
<project>/.RUD/<task>/runs/<repo>/worker.log
<project>/.RUD/<task>/runs/<repo>/agent_logs/
```

### 7. Stop Worker

Click `Stop task` in the `Worker` tab.

This stops the background controller process and kills the corresponding tmux session, including runner and evaluator panes.

## Tmux Controller Mode

`claudeloop tmux` is the main execution engine used by the web UI. It creates one tmux session with two Claude Code panes and one background Python controller process.

```text
tmux session: claudeloop-<task>-<repo>

pane 0: runner
  - interactive Claude Code
  - edits code in the worktree
  - updates the task PLAN.md

pane 1: evaluator
  - interactive Claude Code
  - runs checks from SUCCESS_CONDITION.md
  - reports whether the task is done

background controller process
  - polls tmux panes
  - detects idle state
  - asks evaluator to verify
  - sends feedback back to runner
  - exits on success, max rounds, error, or Stop task
```

Manual usage:

```bash
claudeloop tmux \
  --prompt TASK_PROMPT.md \
  --success SUCCESS_CONDITION.md \
  --plan PLAN.md \
  --model claude-opus-4-6 \
  --max-rounds 200 \
  --effort max \
  --dangerously-skip-permissions \
  --no-commit
```

Important options:

- `--prompt, -p`: task prompt path.
- `--success, -s`: success condition path.
- `--plan`: plan/progress path.
- `--max-rounds`, `--max-iters`, `-n`: maximum controller rounds.
- `--model, -m`: Claude model for runner and evaluator.
- `--session-name`: explicit tmux session name.
- `--log-dir`: logs and session state.
- `--resume`: reconnect to a previous tmux session using the log directory.
- `--effort, -e`: Claude Code effort level, including `max`.
- `--dangerously-skip-permissions`: skip Claude Code permission prompts.
- `--allowed-tools`: comma-separated tools to auto-approve when not skipping permissions.
- `--additional-prompt`: extra text appended to the task prompt.

## Directory Layout

For a project at `/path/to/project` and task `qwen34b`:

```text
/path/to/project/.RUD/qwen34b/
├── TASK_PROMPT.md
├── SUCCESS_CONDITION.md
├── PLAN.md
├── INTERVIEW.md
├── task.json
├── work/
│   └── project/
└── runs/
    └── project/
        ├── process.json
        ├── worker.log
        └── agent_logs/
```

The original repo stays separate from task work. The worker edits files under `work/<repo>`. The authoritative task state stays in `.RUD/<task>/PLAN.md`.

## Subprocess Mode

`claudeloop run` is the simpler subprocess mode. It spawns a fresh Claude Code subprocess per iteration and re-sends the prompt each time. Prefer the web UI and tmux mode for long interactive jobs.

```bash
claudeloop run \
  --prompt TASK_PROMPT.md \
  --success SUCCESS_CONDITION.md \
  --plan PLAN.md \
  --model claude-opus-4-6 \
  --max-iters 20 \
  --effort max \
  --dangerously-skip-permissions
```

## Template Init

For manual workflows outside `.RUD`, create starter files in the current directory:

```bash
claudeloop init
```

This writes:

- `TASK_PROMPT.md`
- `SUCCESS_CONDITION.md`
- `PLAN.md`
