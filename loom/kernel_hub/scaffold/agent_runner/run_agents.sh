#!/usr/bin/env bash
set -euo pipefail

# Default values
N_AGENTS=1
MODEL=""
PLUGIN=""
TARGET=""
SHAPE=""
TARGET_SPEEDUP=""
AUTO_TERMINATE=false
POLL_INTERVAL=60
STARTER_MODE="none"
PRESET_PATH=""
MAX_ITERATIONS=1
BUILD_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plugin)         PLUGIN="$2";         shift 2 ;;
        --target)         TARGET="$2";         shift 2 ;;
        --shape)          SHAPE="$2";          shift 2 ;;
        --n-agents)       N_AGENTS="$2";       shift 2 ;;
        --model)          MODEL="$2";          shift 2 ;;
        --target-speedup) TARGET_SPEEDUP="$2"; shift 2 ;;
        --auto-terminate) AUTO_TERMINATE=true; shift ;;
        --poll-interval)  POLL_INTERVAL="$2";  shift 2 ;;
        --starter-mode)   STARTER_MODE="$2";   shift 2 ;;
        --preset-path)    PRESET_PATH="$2";    shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --build-mode)     BUILD_MODE=true;     shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$PLUGIN" ] || [ -z "$TARGET" ] || [ -z "$SHAPE" ]; then
    echo "Usage: $0 --plugin PLUGIN --target TARGET --shape JSON --model MODEL [options]"
    echo ""
    echo "Required:"
    echo "  --plugin         Plugin name (torch.linear, torch.sdpa, torch.fp8_gemm, cuda.int4_matmul, fa3.paged_decode, sparse_attention.fwd)"
    echo "  --target         Target type (cuda, cutedsl, triton)"
    echo "  --shape          Shape parameters as JSON (plugin-specific)"
    echo "  --model          Model to use (claude-*, gpt-*, o1-*)"
    echo ""
    echo "Optional:"
    echo "  --n-agents       Number of agents (default: 1)"
    echo "  --max-iterations Max agent sessions per container before it exits (default: 1)"
    echo "  --target-speedup Target speedup threshold (required if --auto-terminate)"
    echo "  --auto-terminate Automatically stop agents when target speedup is achieved"
    echo "  --poll-interval  Seconds between speedup checks (default: 60, only with --auto-terminate)"
    echo "  --starter-mode   Starter code mode: best-similar, generic, preset, or none (default: none)"
    echo "                   best-similar: Query DB for best kernel matching function+scalars"
    echo "                   generic: Use a generic template for the plugin type"
    echo "                   preset: Mount reference implementation (requires --preset-path)"
    echo "                   none: No starter code"
    echo "  --preset-path    Path to reference implementation directory (required for --starter-mode preset)"
    echo ""
    echo "Examples:"
    echo "  # GEMM kernel"
    echo "  $0 --plugin torch.linear --target cuda \\"
    echo "    --shape '{\"m\": 1, \"n\": 4096, \"k\": 4096, \"dtype\": \"float16\"}' \\"
    echo "    --model claude-sonnet-4-20250514 --n-agents 3"
    echo ""
    echo "  # With preset reference implementation"
    echo "  $0 --plugin sparse_attention.fwd --target cutedsl \\"
    echo "    --shape '{\"batch\": 4, \"total_q\": 4096, ...}' \\"
    echo "    --model claude-sonnet-4-20250514 \\"
    echo "    --starter-mode preset --preset-path /path/to/MM-Sparse-Attention"
    exit 1
fi

if [ "$AUTO_TERMINATE" = true ] && [ -z "$TARGET_SPEEDUP" ]; then
    echo "Error: --auto-terminate requires --target-speedup to be set"
    exit 1
fi

if [ "$STARTER_MODE" = "preset" ] && [ -z "$PRESET_PATH" ]; then
    echo "Error: --starter-mode preset requires --preset-path to be set"
    exit 1
fi

if [ -n "$PRESET_PATH" ] && [ ! -d "$PRESET_PATH" ]; then
    echo "Error: --preset-path '$PRESET_PATH' does not exist or is not a directory"
    exit 1
fi

# Infer CLI from model name
if [[ "$MODEL" == claude-* ]]; then
    CLI="claude"
elif [[ "$MODEL" == gpt-* || "$MODEL" == o[0-9]* ]]; then
    CLI="codex"
else
    echo "Error: could not determine CLI for model '$MODEL'"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create the evaluation run via API and capture run_id and task_slug
echo "Creating evaluation run..."
INIT_ARGS=(--plugin "$PLUGIN" --target "$TARGET" --shape "$SHAPE")
if [ -n "$TARGET_SPEEDUP" ]; then
    INIT_ARGS+=(--target-speedup "$TARGET_SPEEDUP")
fi
readarray -t INIT_OUTPUT < <(python3 "$SCRIPT_DIR/init_run.py" "${INIT_ARGS[@]}")
RUN_ID="${INIT_OUTPUT[0]}"
TASK_SLUG="${INIT_OUTPUT[1]}"

if [ -z "$RUN_ID" ]; then
    echo "Error: Failed to create evaluation run"
    exit 1
fi

echo "Run: $RUN_ID  task: $TASK_SLUG  agents: $N_AGENTS  model: $MODEL  starter: $STARTER_MODE"

# Path setup - scaffold is parent of agent_runner
SCAFFOLD_DIR="$SCRIPT_DIR/.."
REPO_DIR="$SCAFFOLD_DIR/.."
SCAFFOLD_CONFIG="$SCAFFOLD_DIR/scaffold.yaml"
IMAGE="turbo-kernel-agent"

# Kernel file extension based on target
if [ "$TARGET" = "cutedsl" ] || [ "$TARGET" = "triton" ]; then
    KERNEL_FILE="kernel.py"
else
    KERNEL_FILE="kernel.cu"
fi

# Parse scaffold config
read -r KB_ENABLED KB_PATH KB_MOUNT_POINT ROLES_FILE ROLE_SET < <(python3 - "$SCAFFOLD_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

kb = config.get("knowledge_base", {})
kb_enabled = "true" if kb.get("enabled") else "false"
kb_path = kb.get("path", "") or "-"
kb_mount = kb.get("mount_point", "/kb")

ma = config.get("multiagent", {})
roles_file = ma.get("roles_file", "multiagent/roles.yaml")
role_set = ma.get("role_set", "")

print(f"{kb_enabled} {kb_path} {kb_mount} {roles_file} {role_set}")
PY
)

# Set up knowledge base if enabled
KB_DOCKER_MOUNT=()
if [ "$KB_ENABLED" = "true" ]; then
    KB_DIR="$SCAFFOLD_DIR/$KB_PATH"
    if [ ! -d "$KB_DIR/nvidia-docs" ]; then
        echo "Initializing knowledge base submodule..."
        git -C "$SCAFFOLD_DIR" submodule update --init "$KB_PATH"
    fi
    KB_DOCKER_MOUNT=(-v "$KB_DIR:$KB_MOUNT_POINT:ro")
    echo "Knowledge base: enabled ($KB_MOUNT_POINT)"
else
    echo "Knowledge base: disabled"
fi

docker build --network=host -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$REPO_DIR"

RUNS_DIR="$SCRIPT_DIR/runs"
mkdir -p "$RUNS_DIR"

INSTRUCTIONS_FILE="AGENTS.md"
if [ "$CLI" = "claude" ]; then
    INSTRUCTIONS_FILE="CLAUDE.md"
fi

# Load roles from YAML
ROLES_PATH="$SCAFFOLD_DIR/$ROLES_FILE"
readarray -t SPECIALIST_PROMPTS < <(python3 - "$ROLES_PATH" "$ROLE_SET" "$TASK_SLUG" "$KERNEL_FILE" "$INSTRUCTIONS_FILE" <<'PY'
import sys
import yaml

roles_path, role_set, task_slug, kernel_file, instructions_file = sys.argv[1:6]

with open(roles_path) as f:
    roles = yaml.safe_load(f)

if role_set and role_set in roles:
    for spec in roles[role_set]:
        prompt = spec["prompt"].strip()
        prompt = prompt.replace("$TASK_SLUG", task_slug)
        prompt = prompt.replace("$KERNEL_FILE", kernel_file)
        prompt = prompt.replace("$INSTRUCTIONS_FILE", instructions_file)
        print(prompt.replace("\n", " "))
PY
)

DEFAULT_PROMPT=$(python3 - "$ROLES_PATH" "$TASK_SLUG" "$KERNEL_FILE" "$INSTRUCTIONS_FILE" <<'PY'
import sys
import yaml

roles_path, task_slug, kernel_file, instructions_file = sys.argv[1:5]

with open(roles_path) as f:
    roles = yaml.safe_load(f)

prompt = roles.get("default_role", "").strip()
prompt = prompt.replace("$TASK_SLUG", task_slug)
prompt = prompt.replace("$KERNEL_FILE", kernel_file)
prompt = prompt.replace("$INSTRUCTIONS_FILE", instructions_file)
print(prompt.replace("\n", " "))
PY
)

CONTINUE_PROMPT=$(python3 - "$ROLES_PATH" "$TASK_SLUG" "$KERNEL_FILE" "$INSTRUCTIONS_FILE" <<'PY'
import sys
import yaml

roles_path, task_slug, kernel_file, instructions_file = sys.argv[1:5]

with open(roles_path) as f:
    roles = yaml.safe_load(f)

prompt = roles.get("continue_role", "").strip()
prompt = prompt.replace("$TASK_SLUG", task_slug)
prompt = prompt.replace("$KERNEL_FILE", kernel_file)
prompt = prompt.replace("$INSTRUCTIONS_FILE", instructions_file)
print(prompt.replace("\n", " "))
PY
)

for i in $(seq 1 "$N_AGENTS"); do
    CONTAINER_NAME="kernel-agent-${RUN_ID}-$i"
    if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
        echo "WARNING: $CONTAINER_NAME already exists — stopping and replacing it."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    fi
    WORKDIR="$RUNS_DIR/$RUN_ID/agent-$i"
    rm -rf "$WORKDIR"
    mkdir -p "$WORKDIR"

    # Build instruction file from scaffold. Upstream split the old INSTRUCTIONS.md
    # / eval/WORKFLOW.md into instructions/common/*.md (+ optional per-target
    # instructions/<target>/*.md); concatenate whatever exists.
    : > "$WORKDIR/$INSTRUCTIONS_FILE"
    for _instr in "$SCAFFOLD_DIR"/instructions/common/*.md "$SCAFFOLD_DIR"/instructions/"$TARGET"/*.md; do
        if [ -f "$_instr" ]; then
            cat "$_instr" >> "$WORKDIR/$INSTRUCTIONS_FILE"
        fi
    done
    if [ "$BUILD_MODE" = true ] && [ -f "$SCAFFOLD_DIR/eval/BUILD_MODE.md" ]; then
        cat "$SCAFFOLD_DIR/eval/BUILD_MODE.md" >> "$WORKDIR/$INSTRUCTIONS_FILE"
    fi
    if [ "$KB_ENABLED" = "true" ]; then
        cat "$SCAFFOLD_DIR/knowledge_base/REFERENCE.md" >> "$WORKDIR/$INSTRUCTIONS_FILE"
    fi

    if [ "$CLI" = "codex" ]; then
        cp -r "$SCRIPT_DIR/.agents" "$WORKDIR/"
        mkdir -p "$WORKDIR/.codex"
        cp "$SCRIPT_DIR/config.toml" "$WORKDIR/.codex/config.toml"
        sed -i "s/^model = .*/model = \"$MODEL\"/" "$WORKDIR/.codex/config.toml"
    else
        mkdir -p "$WORKDIR/.claude"
        cp "$SCRIPT_DIR/claude_settings.json" "$WORKDIR/.claude/settings.json"
    fi

    # Assign specialist role if available, otherwise default
    PROMPT="$DEFAULT_PROMPT"
    IDX=$((i - 1))
    if [ "$IDX" -lt "${#SPECIALIST_PROMPTS[@]}" ]; then
        PROMPT="${SPECIALIST_PROMPTS[$IDX]}"
    fi

    # Mint a per-agent API key via the admin endpoint
    AGENT_KEY=$(curl -sS -X POST \
        "${KERNEL_EVALUATOR_API:-http://localhost:${KERNEL_EVALUATOR_PORT:-8000}}/api-keys" \
        -H "X-API-Key: ${KERNEL_EVALUATOR_ADMIN_API_KEY:?must be set}" \
        -H 'Content-Type: application/json' \
        -d '{"role":"user"}' \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['api_key'])")
    if [ -z "$AGENT_KEY" ]; then
        echo "Error: Failed to mint API key for agent $i"
        exit 1
    fi
    AGENT_INDEX="$i"

    if [ "$CLI" = "codex" ]; then
        API_KEY_ENV=(-e "CODEX_API_KEY=${OPENAI_API_KEY}")
        EXEC_CMD='codex exec \
                --sandbox danger-full-access \
                --ephemeral \
                --skip-git-repo-check \
                "$CURRENT_PROMPT"'
    else
        API_KEY_ENV=(-e "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
        EXEC_CMD='claude -p "$CURRENT_PROMPT" \
                --dangerously-skip-permissions \
                --model '"$MODEL"
    fi

    # Preset mount for reference implementation
    PRESET_DOCKER_MOUNT=()
    PRESET_ENV=()
    if [ -n "$PRESET_PATH" ]; then
        PRESET_DOCKER_MOUNT=(-v "$(realpath "$PRESET_PATH")":/preset:ro)
        PRESET_ENV=(-e "BENCH_PRESET_PATH=/preset")
    fi

    docker run -dt \
        --name "$CONTAINER_NAME" \
        --network=host \
        --user "$(id -u):$(id -g)" \
        -e HOME=/workspace \
        "${API_KEY_ENV[@]}" \
        -e KERNEL_EVALUATOR_PORT="${KERNEL_EVALUATOR_PORT:-8000}" \
        -e BENCH_RUN_ID="$RUN_ID" \
        -e BENCH_AGENT_INDEX="$AGENT_INDEX" \
        -e KERNEL_EVALUATOR_API_KEY="$AGENT_KEY" \
        -e BENCH_STARTER_MODE="$STARTER_MODE" \
        "${PRESET_ENV[@]}" \
        -v "$WORKDIR":/workspace \
        "${KB_DOCKER_MOUNT[@]}" \
        "${PRESET_DOCKER_MOUNT[@]}" \
        "$IMAGE" \
        bash -c "for iter in \$(seq 1 $MAX_ITERATIONS); do
            if [ -f /workspace/$KERNEL_FILE ]; then
                CURRENT_PROMPT=\"$CONTINUE_PROMPT\"
            else
                CURRENT_PROMPT=\"$PROMPT\"
            fi
            echo \"[agent] starting iteration \$iter/$MAX_ITERATIONS\"
            $EXEC_CMD || true
            echo \"[agent] iteration \$iter/$MAX_ITERATIONS ended\"
            sleep 5
        done
        echo \"[agent] reached max iterations ($MAX_ITERATIONS), exiting\""

    echo "Agent $i started (model=$MODEL, run=$RUN_ID, agent_index=$AGENT_INDEX) -> $WORKDIR"
done

echo ""
echo "Run: $RUN_ID"
echo "Logs:    docker logs -f kernel-agent-${RUN_ID}-1"

if [ "$AUTO_TERMINATE" = true ]; then
    echo "Auto-terminate: enabled (target: ${TARGET_SPEEDUP}x, poll: ${POLL_INTERVAL}s)"
    nohup bash "$SCRIPT_DIR/wait_for_speedup.sh" "$RUN_ID" --interval "$POLL_INTERVAL" \
        > "$RUNS_DIR/$RUN_ID/wait_for_speedup.log" 2>&1 &
    WATCHER_PID=$!
    echo "Watcher PID: $WATCHER_PID (log: $RUNS_DIR/$RUN_ID/wait_for_speedup.log)"
else
    echo "Finish:  bash $SCRIPT_DIR/finish_run.sh $RUN_ID"
fi
