#!/usr/bin/env bash
# Poll for target speedup and finish the run when achieved.
# Usage: wait_for_speedup.sh <run_id> [--interval SECONDS]
set -euo pipefail

RUN_ID=""
POLL_INTERVAL=60

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) POLL_INTERVAL="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) RUN_ID="$1"; shift ;;
    esac
done

if [ -z "$RUN_ID" ]; then
    echo "Usage: $0 <run_id> [--interval SECONDS]"
    echo ""
    echo "Polls the evaluation service until a kernel meeting the target speedup is found,"
    echo "then calls finish_run.sh to stop agents and postprocess the kernel."
    echo ""
    echo "Options:"
    echo "  --interval  Poll interval in seconds (default: 60)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_EVALUATOR_PORT="${KERNEL_EVALUATOR_PORT:-8000}"
API_URL="${KERNEL_EVALUATOR_API:-http://localhost:$KERNEL_EVALUATOR_PORT}"
API_KEY="${KERNEL_EVALUATOR_ADMIN_API_KEY:-admin1234}"

# Get run info and extract target_speedup
echo "Fetching run contract for $RUN_ID..."
RUN_INFO=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/evaluation/runs/$RUN_ID" 2>/dev/null || echo "null")

if [ "$RUN_INFO" = "null" ] || [ -z "$RUN_INFO" ]; then
    echo "Error: Could not fetch run info for $RUN_ID"
    exit 1
fi

TARGET_SPEEDUP=$(echo "$RUN_INFO" | jq -r '.target_speedup // empty')

if [ -z "$TARGET_SPEEDUP" ]; then
    echo "Error: No target_speedup set for run $RUN_ID"
    echo "Set target_speedup when creating the run with --target-speedup"
    exit 1
fi

echo "Run: $RUN_ID"
echo "Target speedup: ${TARGET_SPEEDUP}x"
echo "Poll interval: ${POLL_INTERVAL}s"
echo ""

while true; do
    BEST_KERNEL=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/evaluation/runs/$RUN_ID/best-kernel" 2>/dev/null || echo "null")

    if [ "$BEST_KERNEL" != "null" ] && [ -n "$BEST_KERNEL" ]; then
        KERNEL_US=$(echo "$BEST_KERNEL" | jq -r '.kernel_us // empty')
        BASELINE_US=$(echo "$BEST_KERNEL" | jq -r '.baseline_us // empty')

        if [ -n "$KERNEL_US" ] && [ -n "$BASELINE_US" ]; then
            CURRENT_SPEEDUP=$(echo "scale=4; $BASELINE_US / $KERNEL_US" | bc)
            echo "[$(date '+%H:%M:%S')] Best speedup: ${CURRENT_SPEEDUP}x (target: ${TARGET_SPEEDUP}x)"

            # Compare using bc (1 if current >= target, 0 otherwise)
            if [ "$(echo "$CURRENT_SPEEDUP >= $TARGET_SPEEDUP" | bc)" -eq 1 ]; then
                echo ""
                echo "Target speedup achieved! Finishing run..."
                bash "$SCRIPT_DIR/finish_run.sh" "$RUN_ID"
                exit 0
            fi
        else
            echo "[$(date '+%H:%M:%S')] Waiting for first valid kernel..."
        fi
    else
        echo "[$(date '+%H:%M:%S')] No kernels yet..."
    fi

    sleep "$POLL_INTERVAL"
done
