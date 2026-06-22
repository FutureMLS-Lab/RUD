#!/usr/bin/env bash
# Generate a Python registration file from an optimized turbo-gemm kernel.
# Run this after run_agents.sh has produced a best kernel for a given slug.
#
# Usage:
#   bash run_agents_pyreg.sh <slug> --kernel-file path/to/kernel.cu [--output-file path/to/output.py]
#
# Examples:
#   bash agents/run_agents_pyreg.sh f-linear-512x3072x3072 --kernel-file eval_service/results/f-linear-512x3072x3072/best.cu
#   bash agents/run_agents_pyreg.sh f-linear-4096x3072x3072 --kernel-file my_kernel.cu

set -euo pipefail

SLUG="${1:-}"
shift 1 2>/dev/null || true

KERNEL_FILE=""
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel-file) KERNEL_FILE="$2"; shift 2 ;;
        --output-file) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."

if [ -z "$SLUG" ]; then
    echo "Usage: $0 <slug> --kernel-file path [--output-file path]"
    echo ""
    echo "Available results:"
    ls "$REPO_DIR/eval_service/results/" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    exit 1
fi

# Extract M, N, K from slug (e.g., f-linear-512x3072x3072 → 512 3072 3072)
SHAPE="${SLUG##*-}"
IFS='x' read -r M_VAL N_VAL K_VAL <<< "$SHAPE"

if [ -z "$KERNEL_FILE" ]; then
    echo "Error: --kernel-file is required"
    exit 1
fi
if [ ! -f "$KERNEL_FILE" ]; then
    echo "Error: kernel file not found: $KERNEL_FILE"
    exit 1
fi

BASELINE_FILE="$REPO_DIR/../replacements/candidates/gemm/kernels/mm_tk_abt.cu"

# Resolve output file (default: kernel file with .cu → .py)
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="${KERNEL_FILE%.cu}.py"
fi

# Find an existing shape-specific Python file to use as a template, from the
# bundled python_registration_templates/.
LOCAL_TEMPLATES_DIR="$SCRIPT_DIR/python_registration_templates"
EXAMPLE_PY_FILE=""
if [ -d "$LOCAL_TEMPLATES_DIR" ]; then
    for candidate in \
        "$LOCAL_TEMPLATES_DIR/mm_abt_"*"_${N_VAL}_${K_VAL}.py" \
        "$LOCAL_TEMPLATES_DIR/mm_abt_"*".py" \
        "$LOCAL_TEMPLATES_DIR/mm_tk_abt.py"
    do
        if [ -f "$candidate" ]; then
            EXAMPLE_PY_FILE="$candidate"
            break
        fi
    done
fi

echo "Generating Python registration file for $SLUG"
echo "  Shape:  M=$M_VAL, N=$N_VAL, K=$K_VAL"
echo "  Kernel: $KERNEL_FILE"
echo "  Output: $OUTPUT_FILE"
[ -n "$EXAMPLE_PY_FILE" ] && echo "  Template: $EXAMPLE_PY_FILE"
echo ""

# Build template section (optional but helpful)
EXAMPLE_SECTION=""
if [ -n "$EXAMPLE_PY_FILE" ]; then
    EXAMPLE_SECTION="Example Python registration file to use as a template ($(basename "$EXAMPLE_PY_FILE")):
\`\`\`python
$(cat "$EXAMPLE_PY_FILE")
\`\`\`

"
fi

PROMPT="Generate a Python registration file that registers an optimized CUDA kernel with PyTorch.

Shape: M=$M_VAL, N=$N_VAL, K=$K_VAL (bf16 ABt GEMM on H100).
The CUDA source file will be named mm_abt_${M_VAL}_${N_VAL}_${K_VAL}.cu and lives in the same directory as the Python file (replacements/candidates/gemm/).

New optimized kernel ($(basename "$KERNEL_FILE")):
\`\`\`cuda
$(cat "$KERNEL_FILE")
\`\`\`

Baseline kernel (shows the extern \"C\" interface):
\`\`\`cuda
$(cat "$BASELINE_FILE")
\`\`\`

${EXAMPLE_SECTION}Write a Python registration file following the template's structure, adapted for:
- Artifact name: mm_abt_${M_VAL}_${N_VAL}_${K_VAL}
- Function prefix: use the extern \"C\" function names from the new kernel
- Fixed shape: M=$M_VAL, N=$N_VAL, K=$K_VAL (exact equality in eligible())
- MODULE_NAME: candidates.gemm.mm_abt_${M_VAL}_${N_VAL}_${K_VAL}
- Only include TMA descriptor functions if they are defined in the new kernel

Write the complete Python file to: $OUTPUT_FILE"

mkdir -p "$(dirname "$OUTPUT_FILE")"

LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"
OUTPUT_LOG="$LOGS_DIR/pyreg-${SLUG}.log"

echo "Running agent (log: $OUTPUT_LOG)..."
claude -p "$PROMPT" \
    --dangerously-skip-permissions \
    --output-format stream-json \
    --verbose \
    2>&1 | tee "$OUTPUT_LOG"

echo ""
if [ -f "$OUTPUT_FILE" ]; then
    echo "Done: $OUTPUT_FILE"
else
    echo "Warning: output file was not created. Check log: $OUTPUT_LOG"
fi
