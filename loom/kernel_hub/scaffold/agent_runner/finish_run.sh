#!/usr/bin/env bash
# Postprocess best kernel, stop containers, and clean up.
# Usage: finish_run.sh <run_id>
set -euo pipefail

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ]; then
    echo "Usage: $0 <run_id>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_EVALUATOR_PORT="${KERNEL_EVALUATOR_PORT:-8000}"
API_URL="${KERNEL_EVALUATOR_API:-http://localhost:$KERNEL_EVALUATOR_PORT}"
API_KEY="${KERNEL_EVALUATOR_ADMIN_API_KEY:-admin1234}"

# JSON field extraction helper (replaces jq dependency)
json_get() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('$1','') if d.get('$1') is not None else '')"
}

# JSON-escape a file's contents for embedding in a JSON payload
json_escape_file() {
    python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" < "$1"
}

echo "Finishing run: $RUN_ID"

# Stop and remove containers
containers=$(docker ps -aq --filter "name=kernel-agent-${RUN_ID}-" 2>/dev/null || true)
if [ -n "$containers" ]; then
    docker rm -f $containers 2>/dev/null || true
    echo "Containers stopped."
else
    echo "No containers to stop."
fi

# ---------------------------------------------------------------------------
# Postprocess best kernel (TMA descriptors + Python registration)
# ---------------------------------------------------------------------------
echo "Postprocessing best kernel for run..."

# Get best kernel for this run from database
BEST_KERNEL_INFO=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/evaluation/runs/$RUN_ID/best-kernel" 2>/dev/null || echo "null")

if [ "$BEST_KERNEL_INFO" != "null" ] && [ -n "$BEST_KERNEL_INFO" ]; then
    KERNEL_ID=$(echo "$BEST_KERNEL_INFO" | json_get id)

    if [ -n "$KERNEL_ID" ]; then
        KERNEL_SOURCE=$(echo "$BEST_KERNEL_INFO" | json_get kernel_source)
        FUNCTION_NAME=$(echo "$BEST_KERNEL_INFO" | json_get function_name)
        SPEEDUP=$(echo "$BEST_KERNEL_INFO" | json_get speedup)

        # Extract shape from scalar_args
        read -r M N K < <(echo "$BEST_KERNEL_INFO" | python3 -c "
import json,sys
d = json.load(sys.stdin)
sa = d.get('scalar_args') or {}
print(sa.get('M',''), sa.get('N',''), sa.get('K',''))
")

        if [ -n "$M" ] && [ -n "$N" ] && [ -n "$K" ]; then
            SLUG="f-linear-${M}x${N}x${K}"
        else
            SLUG=$(echo "$FUNCTION_NAME" | tr '.' '-')
        fi

        echo "  Best kernel: id=$KERNEL_ID, speedup=${SPEEDUP}x, slug=$SLUG"

        # Create temp dir for postprocessing
        TEMP_DIR=$(mktemp -d)
        KERNEL_FILE="$TEMP_DIR/kernel.cu"
        echo "$KERNEL_SOURCE" > "$KERNEL_FILE"

        # Step 1: TMA postprocessing
        POSTPROCESSED_FILE="$TEMP_DIR/kernel_postprocessed.cu"
        echo "  Running TMA postprocessing..."
        if bash "$SCRIPT_DIR/postprocess_tma.sh" "$SLUG" \
            --kernel-file "$KERNEL_FILE" \
            --output-file "$POSTPROCESSED_FILE" >/dev/null 2>&1; then
            echo "  TMA postprocessing complete"
        else
            echo "  Warning: TMA postprocessing failed, using original source"
            cp "$KERNEL_FILE" "$POSTPROCESSED_FILE"
        fi

        # Step 2: Generate Python registration
        PYREG_FILE="$TEMP_DIR/kernel.py"
        echo "  Generating Python registration..."
        if bash "$SCRIPT_DIR/generate_python_registration.sh" "$SLUG" \
            --kernel-file "$POSTPROCESSED_FILE" \
            --output-file "$PYREG_FILE" >/dev/null 2>&1; then
            echo "  Python registration generation complete"
        else
            echo "  Warning: Python registration generation failed"
            touch "$PYREG_FILE"
        fi

        # Step 3: Update database with generated content
        echo "  Updating database..."
        POSTPROCESSED_CONTENT=$(json_escape_file "$POSTPROCESSED_FILE")
        PYREG_CONTENT=$(json_escape_file "$PYREG_FILE")

        if curl -X PATCH "$API_URL/kernels/$KERNEL_ID" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $API_KEY" \
            -d "{\"postprocessed_source\": $POSTPROCESSED_CONTENT, \"python_registration\": $PYREG_CONTENT}" \
            --fail --silent --show-error 2>/dev/null; then
            echo "  Database updated successfully"
        else
            echo "  Warning: Failed to update database"
        fi

        rm -rf "$TEMP_DIR"
    else
        echo "  No winning kernel (speedup <= 1.0)"
    fi
else
    echo "  No best kernel found (API unavailable or no kernels)"
fi

# Remove run working directory
RUN_DIR="$SCRIPT_DIR/runs/$RUN_ID"
if [ -d "$RUN_DIR" ]; then
    rm -rf "$RUN_DIR"
    echo "Working directory cleared: $RUN_DIR"
else
    echo "No working directory found for run."
fi

echo "Done."
