#!/usr/bin/env bash
# Append tkcc TMA descriptor adapter functions to a ThunderKittens GEMM kernel.
# These are required for plugging the kernel into tkcc (ThunderKittens compiler)
# when the kernel uses TMA — which is the only strategy that needs this adaptation.
#
# Usage:
#   bash agents/run_agents_tkcc_adapt.sh <slug> --kernel-file path/to/kernel.cu [--output-file path/to/output.cu]
#
# If --output-file is omitted, the adapter code is appended in-place to the kernel file.
#
# Examples:
#   bash agents/run_agents_tkcc_adapt.sh f-linear-512x3072x3072 --kernel-file eval_service/results/f-linear-512x3072x3072/best.cu
#   bash agents/run_agents_tkcc_adapt.sh f-linear-512x3072x3072 --kernel-file best.cu --output-file best_tkcc.cu

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

if [ -z "$KERNEL_FILE" ]; then
    echo "Error: --kernel-file is required"
    exit 1
fi
if [ ! -f "$KERNEL_FILE" ]; then
    echo "Error: kernel file not found: $KERNEL_FILE"
    exit 1
fi

# Extract M, N, K from slug (e.g., f-linear-512x3072x3072 → 512 3072 3072)
SHAPE="${SLUG##*-}"
IFS='x' read -r M_VAL N_VAL K_VAL <<< "$SHAPE"

# Default: append in-place to the kernel file
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="$KERNEL_FILE"
fi

echo "Generating tkcc TMA descriptor adapter for $SLUG"
echo "  Shape:  M=$M_VAL, N=$N_VAL, K=$K_VAL"
echo "  Kernel: $KERNEL_FILE"
echo "  Output: $OUTPUT_FILE"
echo ""

PROMPT="You are adapting a ThunderKittens CUDA GEMM kernel for use with tkcc (the ThunderKittens compiler).

Shape: M=$M_VAL, N=$N_VAL, K=$K_VAL (bf16 ABt GEMM on H100).

Kernel to adapt ($(basename "$KERNEL_FILE")):
\`\`\`cuda
$(cat "$KERNEL_FILE")
\`\`\`

Your task: append two tkcc TMA descriptor adapter functions to the output file.

These functions are only needed when the kernel uses TMA (Tensor Memory Accelerator) — i.e., when
it uses ThunderKittens \`gl<>\` layouts with \`tma::load_async\`. If the kernel does not use TMA,
write nothing and exit without modifying the output file.

The two functions to generate are:

1. \`gemm_${M_VAL}x${N_VAL}x${K_VAL}_num_tma_descriptors()\` — returns 2 (one for A, one for Bt).

2. \`gemm_${M_VAL}x${N_VAL}x${K_VAL}_describe_tma_descriptors(void *d_A, void *d_Bt, void *d_C, int M, int N, int K, void *out_meta)\` — calls \`_fill_tma_desc_meta_2d\` for A and Bt.

The signature of _fill_tma_desc_meta_2d is:
  _fill_tma_desc_meta_2d(dest, base_ptr_as_uint64, elem_bytes, tensor_rows, tensor_cols, tile_rows, tile_cols)

For A:  tensor shape is (M, K), tile is (BM_A, BK) where BM_A is the row dimension of the a_tile in the gl<> layout (i.e., kRowBlock / kConsumerWarpgroups, NOT kRowBlock itself).
For Bt: tensor shape is (N, K), tile is (BN, BK) where BN is kColBlock and BK is kRedBlock.
Element size is 2 (bf16).
Each descriptor metadata record is 96 bytes: descriptor 0 at offset 0*96, descriptor 1 at offset 1*96.

Determine BM_A, BN, BK by reading the #define and static constexpr values in the kernel source.

Append the generated code (nothing else — no commentary, no markdown fences, just the C++ code)
to the file: $OUTPUT_FILE"

mkdir -p "$(dirname "$OUTPUT_FILE")"

# If output file differs from input, copy the kernel first so we're appending to a full file
if [ "$OUTPUT_FILE" != "$KERNEL_FILE" ]; then
    cp "$KERNEL_FILE" "$OUTPUT_FILE"
fi

LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"
OUTPUT_LOG="$LOGS_DIR/tkcc-adapt-${SLUG}.log"

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
