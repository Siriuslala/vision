#!/usr/bin/env bash
# Experiment 3.8 round 2: Gated DeltaNet mechanism of the row ordinal
# (docs/analysis/position/results_2026-10-16_sequence_order_r2.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-4}"        # never GPU 1 (reserved)
RUN_NAME="2026-10-16_sequence_order_r2"
STAGES="${STAGES:-spotcheck,gates,probe,intervene,restore2,transplant,think}"   # or "figures"
SEED=20261016
BATCH_SIZE=16                  # unpadded batches of equal-length prompts (8x8 grids, 256 visual tokens)
MAX_NEW_TOKENS=24              # direct-answer prompt
MAX_NEW_THINK=512              # think-then-answer prompt
N_SPOT=20                      # spot-check samples against the reference generate()
N_PROBE=32                     # 8x8 images in the probe stage (12 layouts each)
N_NATURAL=32                   # RefCOCO test images in the gate stage
N_THINK=32                     # images in the think-then-answer comparison
NATURAL_RUN="$WORK_DIR/analysis/position/2026-10-14_probe_refinement"   # RefCOCO images of experiment 3.4
ROUND1_RUN="$WORK_DIR/analysis/position/2026-10-15_sequence_order"
OUTPUT_DIR="$WORK_DIR/analysis/position/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/sequence_order_r2"

if [[ "$GPU_IDS" == *1* && "$GPU_IDS" != *[02-9]1* ]]; then echo "GPU 1 is reserved" >&2; exit 1; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "figures" ]]; then
  uv run --frozen python -m analysis_tools.position.sequence_order_r2_analysis --run "$OUTPUT_DIR" \
    --round1 "$ROUND1_RUN" --figures "$FIGURE_DIR"
  exit 0
fi
uv run --frozen python -m analysis_tools.position.sequence_order_r2_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --natural "$NATURAL_RUN" --stages "$STAGES" --seed "$SEED" \
  --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" --max-new-think "$MAX_NEW_THINK" \
  --n-spot "$N_SPOT" --n-probe "$N_PROBE" --n-natural "$N_NATURAL" --n-think "$N_THINK" \
  2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' | tee -a "$OUTPUT_DIR/run.log"
