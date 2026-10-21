#!/usr/bin/env bash
# Experiment 3.8: positional role of sequence order (docs/analysis/position/design_sequence_order.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="2026-10-15_sequence_order"
STAGES="${STAGES:-restore_check,spotcheck,behavior,restore,probe,fail20}"   # or "figures" (analysis + figures)
SEED=20261015
BATCH_SIZE=16                  # unpadded batches of equal-length prompts (8x8 grids, 256 visual tokens)
BATCH_SIZE_LARGE=4             # batches for the 20x20 grids (1600 visual tokens)
MAX_NEW_TOKENS=24
N_SPOT=20                      # spot-check samples against the reference generate()
N_PROBE=40                     # rolled inputs per probe dataset
T6_SOURCE="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/sequence_order"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "figures" ]]; then
  uv run --frozen python -m analysis_tools.position.sequence_order_analysis --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.sequence_order_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --t6-source "$T6_SOURCE" --stages "$STAGES" --seed "$SEED" \
  --batch-size "$BATCH_SIZE" --batch-size-large "$BATCH_SIZE_LARGE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --n-spot "$N_SPOT" --n-probe "$N_PROBE" \
  2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run.log"
