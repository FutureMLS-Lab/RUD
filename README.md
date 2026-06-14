<p align="center">
  <img src="claudeloop/web_static/loom-logo.png" alt="Loom" width="200" />
</p>

<h1 align="center">Loom</h1>

<p align="center"><em>Repeat Until Done.</em></p>

**Loom** is a lightweight task console for [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
(and Codex). You give each task a goal, a deep‑interview pane refines it into a
`PLAN.md`, you spin up git worktrees on `zhongzhu/<slug>` branches and `/goal`
your way through the work, review the diff, and optionally get pinged whenever
the agent stops and needs you. *Repeat Until Done.*

There is no autonomous agent loop, no worker, and no evaluator — **you drive the
agent**; Loom just removes the bookkeeping: worktrees, `PLAN.md`, diffs,
sessions, project notes, and notifications.

## The Loom flow

```
General goal ─▶ Start Deep Interview ─▶ Run /goal ─▶ Write result ─▶ (repeat)
                      │                     │              │
                 writes PLAN.md       executes PLAN.md   appends results
```

These three steps are the **Loom flow** buttons in the Claude tab. You start the
agent, let the deep interview turn your goal into `PLAN.md`, run `/goal` against
it, and write progress/results back into `PLAN.md` — then iterate.

```
┌─ Project root (e.g. ~/work/xorl) ──────────────────────────────────┐
│ .RUD/                                                               │
│ ├── NOTES.md            ← project-scoped scratchpad (📓 Notes)      │
│ ├── task-order.json                                                 │
│ └── <slug>/                                                         │
│     ├── task.json       ← title, goal, agent, skills, worktrees,…   │
│     ├── PLAN.md         ← the deep interview writes this; you edit  │
│     ├── monitor.json    ← run-monitor on/off + last-fired (if used) │
│     └── work/                                                       │
│         ├── xorl-internal/   ← git worktree, branch zhongzhu/<slug> │
│         └── xorl-sglang/     ← second worktree, same branch name    │
└────────────────────────────────────────────────────────────────────┘
```

(The `.RUD/` directory name is a nod to the tagline, *Repeat Until Done*.)

## Install

```bash
git clone https://github.com/FutureMLS-Lab/Loom.git
cd Loom
pip install -e .

# Authenticate the agent CLI used by the pane (Claude Code shown here).
claude
```

Optional but needed for the live agent pane:

```bash
tmux -V       # tmux must be on PATH
git --version
```

This installs the `loom` command (the legacy `claudeloop` command remains as an
alias). The Python module is importable as `claudeloop`.

## Run

From any directory:

```bash
loom web --project /path/to/your/project --port 8765
# open http://127.0.0.1:8765/
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--project PATH` | Project root. Defaults to `$PWD`. Can be a git repo OR a container directory holding several git repos. |
| `--skills PATH` | Default skills markdown injected into every agent session. Defaults to `claudeloop/skills/AK_skills.md`. The per-task Skills picker only lists markdown under the skills directory. |
| `--projects` | Multi-project workspace: the launch dir is a container; the parent registry entry is pruned once children are registered. |
| `--auth-token TOKEN` | Require HTTP basic / bearer auth. Username is ignored; password = token. Also reads `CLAUDELOOP_WEB_AUTH_TOKEN`. |
| `--daemon` / `--nohup` | Re-spawn in the background and exit. Logs land in `<project>/.RUD/web.log`. |
| `--openclaw …` | Optional [OpenClaw](#run-monitor--openclaw) gateway events (used by the run monitor). Full flag list in `claudeloop/cli.py`. |

```bash
loom init    # writes a minimal PLAN.md + NOTES.md in $PWD
loom --help  # all commands
```

## Web UI

### 1. Register a project

Use **+ Add repo** in the top bar. Registrations are stored in
`~/.claudeloop/web-projects.json` (no files are written outside the registered
path). If the launch directory is a container, pick from the subfolder chips.

### 2. Create a task

**Create Task** asks for:

- **Task type** — Claude, Codex, or Kernel Lab (a dropdown).
- **Title** — becomes the slug (lowercased, dash-separated).
- **General goal** — a short description the deep interview turns into `PLAN.md`.

A `.RUD/<slug>/` directory is created with an empty `PLAN.md`. If the project
root is itself a git repo, **one worktree is auto-created** at
`.RUD/<slug>/work/<repo-name>/` on branch `zhongzhu/<slug>`. Container projects
(no `.git` of their own) skip auto-creation — you pick the source repo from the
Claude tab. The default model is `claude-opus-4-8` for Claude tasks.

Click the title or goal in the task header to rename / re-goal at any time.

### 3. The Claude (or Codex) tab

The main tab for an agent task contains:

- **Loom flow row** — `Start Deep Interview → Run /goal → Write result`, with
  Start / Stop Claude above it.
- **Info card** with:
  - **Agent** — Claude or Codex (switch while stopped).
  - **Skills** — which skills markdown to inject (only files under the skills
    directory are listed).
  - **Worktrees** — every git worktree owned by the task, with branch, live
    `git status` (clean / N modified / ↑3 ↓2), a per-row **Push**, **+ Add
    worktree** (candidate picker; already-added ones dimmed), **Push all**, and
    a `×` to remove (`git worktree remove`).
  - **tmux** — the live pane target + alive/down pill.
  - **Sessions** — every agent session UUID the pane has spawned (scanned from
    `~/.claude/projects/<encoded-cwd>/` or the Codex equivalent). **Resume**
    launches a fresh tmux pane with `--resume <uuid>`, even if the original
    tmux was killed.
  - **Monitor** — a Notify toggle (see [below](#run-monitor--openclaw)).
- **Live tmux pane** preview (polls every ~4s, auto-scrolls to the bottom) with
  a keycap row (↑ ↓ ← → Enter Esc Ctrl-C) next to the input.
- **Embedded read-only Markdown viewer** — defaults to `PLAN.md`, with a picker
  for any other top-level `*.md` in the task directory.

Starting the pane runs `tmux new-session` + the agent CLI in the **primary**
worktree (first in the worktrees list). **Start Deep Interview** then pastes a
prompt (general goal + selected skills) that interviews you and writes `PLAN.md`.

### 4. The Changes tab

A read-only, VSCode-style git diff for the task's worktree(s): a file list (with
add/remove counts and A/M/D/R status) on the left and a red/green unified diff on
the right. It shows both **uncommitted** changes (working tree vs `HEAD`,
including untracked files) and **committed** work (vs the detected base branch,
e.g. `origin/main`). Refreshes on tab open and via the Refresh button.

### 5. Run monitor + OpenClaw

Flip **Notify** on the Monitor row to have Loom watch the agent's tmux pane. It
edge-triggers on the **running → stopped** transition: when the agent was
working (the pane shows its “esc to interrupt” hint) and then stops to wait for
input, Loom emits an [OpenClaw](https://github.com/) event with the last lines
of the pane for context. Reply in OpenClaw and it is pushed back into the pane
(`POST /api/tasks/<slug>/claude/send`), the agent runs again, and you're pinged
on the next stop — repeat. If the pane is already idle when you enable Notify,
it stays silent until the agent actually runs and then stops.

Configure the gateway with the `--openclaw*` flags (URL, token, hook, etc.).

### 6. PLAN.md & Notes

- The embedded viewer is read-only; the agent writes `PLAN.md` during the
  interview and you edit/`/goal` it inside the pane.
- The **📓 Notes** button opens a fullscreen editor for
  `<project>/.RUD/NOTES.md` — one project-scoped scratchpad. **Cmd/Ctrl + S**
  saves.

### 7. Kernel Lab (optional)

Kernel-optimization tasks get a dedicated panel that drives the TKCC kernel
evaluator: a persistent interview to produce a kernel spec, worktree management,
and a build/run launcher with a live log. This is an advanced, optional task
type; regular Claude/Codex tasks don't need it.

## Storage layout

```
<project>/.RUD/
├── NOTES.md              # project-scoped scratch (📓 Notes button)
├── task-order.json
└── <slug>/
    ├── task.json         # title, goal, agent, skills, worktrees, sessions
    ├── PLAN.md
    ├── monitor.json      # run-monitor state (only if used)
    ├── kernel_interview.json  # only for Kernel Lab tasks
    └── work/
        └── <repo>/...    # git worktree, branch zhongzhu/<slug>
```

```
~/.claude/projects/<encoded-cwd>/
└── <session-uuid>.jsonl  # native Claude Code session transcripts

~/.claudeloop/
└── web-projects.json     # registered project paths
```

## HTTP API

Everything is plain JSON; scope with `?project=<id>` (or the
`X-ClaudeLoop-Project` header). When `--auth-token` is set, send it as
`Authorization: Bearer <token>` (or HTTP basic, password = token).

| Method | URL | Purpose |
|--------|-----|---------|
| `GET` | `/api/project` | Active project root, skills path, skills options |
| `GET` | `/api/projects` | List registered projects, default, launch root |
| `POST` | `/api/projects` `{path}` | Register a project root |
| `POST` | `/api/projects/<id>/activate` | Set the default project |
| `POST` | `/api/projects/reorder` `{ids}` | Persist order |
| `DELETE` | `/api/projects/<id>` | Drop from registry (files untouched) |
| `GET` / `PUT` | `/api/notes` | Read / save project `NOTES.md` |
| `GET` | `/api/tasks` | All tasks for the active project |
| `POST` | `/api/tasks` `{title, general_goal, agent?, kind?}` | Create task (auto-worktree if the root is a git repo) |
| `POST` | `/api/tasks/reorder` `{slugs}` | Persist order |
| `GET` | `/api/tasks/<slug>` | Meta + PLAN.md + scanned markdown + agent summary + worktree statuses |
| `PUT` | `/api/tasks/<slug>/meta` `{title?, general_goal?, agent?, skills_path?}` | Rename / re-goal / switch agent |
| `PUT` | `/api/tasks/<slug>/template` `{name, content}` | Write PLAN.md |
| `DELETE` | `/api/tasks/<slug>` | Delete task tree (also unregisters worktrees) |
| `GET` | `/api/tasks/<slug>/diff` | **Changes tab**: per-worktree uncommitted + committed diff |
| `GET`/`POST`/`DELETE` | `/api/tasks/<slug>/monitor` | Run-monitor status / enable / disable |
| `POST` | `/api/tasks/<slug>/claude/send` `{text, submit?}` | Push a message into the pane (used by OpenClaw replies) |
| `GET` | `/api/tasks/<slug>/worktree-candidates` | Repos you could base a worktree on |
| `POST` / `DELETE` | `/api/tasks/<slug>/worktree` | Create / remove a worktree |
| `POST` | `/api/tasks/<slug>/worktree/push` `{path}` | `git push -u origin <branch>` |
| `POST` | `/api/tasks/<slug>/worktrees/push-all` | Push every task worktree branch |
| `POST` | `/api/tasks/<slug>/claude/start` | Launch the agent pane in the primary worktree |
| `POST` | `/api/tasks/<slug>/claude/stop` | Kill the tmux pane (on-disk sessions stay resumable) |
| `POST` | `/api/tasks/<slug>/claude/paste-prompt` | Re-paste the deep-interview prompt |
| `POST` | `/api/tasks/<slug>/claude/resume` `{session_id}` | New tmux, `--resume <id>` |
| `GET` | `/api/tasks/<slug>/claude-sessions` | Tracked session UUIDs + on-disk transcripts |
| `GET` | `/api/tmux/capture?target=…&lines=N` | Pane scrollback |
| `POST` | `/api/tmux/send-text` / `send-key` | Type / send a key into a pane |

Kernel Lab adds `/api/kernel/*` endpoints. For backwards compatibility,
`/api/tasks/<slug>/interview/{start,stop,paste-prompt}` still resolves to the
agent-pane endpoints.

## History

Loom started life as **claudeloop** — a full autonomous agent loop with
runner / evaluator panes, an `Ask` pane, `TASK_PROMPT.md` +
`SUCCESS_CONDITION.md`, and auto-commit. That loop was removed on purpose: the
project became **RUD (Repeat Until Done)** and then **Loom**, a human-driven
console where you drive Claude Code / Codex and Loom keeps the worktrees,
plans, diffs, and notes tidy. The internal Python module is still named
`claudeloop`. See `git log` for the full history.
