#!/usr/bin/env bash
# Experiment 5.1: local re-read pilot (docs/analysis/reread/results_2026-10-16_reread_pilot.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-2}"                            # never GPU 1 (reserved)
RUN_NAME="2026-10-16_reread_pilot"
# comma list of: verify, det, loc, think, think_long, refcoco, babyvision; or "analysis" (CPU metrics and figures)
STAGES="${STAGES:-verify,det,loc,refcoco,babyvision,think}"
SEED=20261016
GRIDS="10 20"                             # synthetic grid sizes (n x n instances of 64 pixels)
DET_N=48                                  # images per (grid, change) unit of the detail task
LOC_N=32                                  # images per (task, grid) unit of the localization task
LOC_K=5                                   # localization window: k x k instance cells
THINK_N=16                                # images per task of the think-then-answer subset (largest grid)
REF_N=151                                 # RefCOCO val cases of the 1-5 % area stratum (all available)
REF_SKIP="${REF_SKIP:-0}"                     # second RefCOCO worker: first case index it processes
BATCH_SIZE=8                              # sequences per forward (equal length only, no padding)
IMAGE_CHUNK="${IMAGE_CHUNK:-8}"                            # images assembled per decoding round
MAX_NEW_TOKENS=128
MAX_THINK=1024
MAX_THINK_LONG=2048                       # think_long stage: budget check on THINK_LONG_N images per task
THINK_LONG_N=8
REFCOCO_CASES="$WORK_DIR/analysis/deepstack/2026-10-14_end_to_end/refcoco"   # experiment 4.2
BV_RUN="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"                # experiment 1.3
BV_ANNOTATION="$DATA_DIR/BabyVision_annotations/find_the_different_grid_v1.json"
OUTPUT_DIR="$WORK_DIR/analysis/reread/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/reread/figures/reread_pilot"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "analysis" ]]; then
  exec uv run --frozen python -m analysis_tools.reread.reread_pilot_analysis --run "$OUTPUT_DIR" \
    --figures "$FIGURE_DIR" --bv-run "$BV_RUN" --refcoco "$REFCOCO_CASES"
fi
exec uv run --frozen python -m analysis_tools.reread.reread_pilot_run --output "$OUTPUT_DIR" --model "$MODEL" \
  --stages "$STAGES" --seed "$SEED" --grids $GRIDS --det-n "$DET_N" --loc-n "$LOC_N" --loc-k "$LOC_K" \
  --think-n "$THINK_N" --ref-n "$REF_N" --ref-skip "$REF_SKIP" --refcoco "$REFCOCO_CASES" --bv-run "$BV_RUN" \
  --bv-annotation "$BV_ANNOTATION" --batch-size "$BATCH_SIZE" --image-chunk "$IMAGE_CHUNK" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-think "$MAX_THINK" --max-think-long "$MAX_THINK_LONG" --think-long-n "$THINK_LONG_N"
