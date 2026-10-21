#!/usr/bin/env bash
# Experiment 3.11: information flow of coordinate readout
# (docs/analysis/position/design_coordinate_flow.md, results_2026-10-20_coordinate_flow.md).
# Configure this section, then run this script without terminal parameters.
# STAGES (GPU, run in order in one process): gen spotcheck heads kv flow field; "cases" is CPU only.
# STAGE="analyze" or "figures" runs the CPU analysis / figure scripts.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="2"
MAX_MEMORY=""                     # e.g. "0:15GiB,1:9GiB" to split the model over the GPUs in GPU_IDS
RUN_DATE="2026-10-20"
STAGE="gpu"                       # cases | gpu | analyze | figures
GPU_STAGES="gen spotcheck kv flow heads field"   # every stage resumes from its records
SOURCE_DIR="$WORK_DIR/analysis/representation/2026-10-16_sees_right"   # 2.5e run: images, cases, answers
N_DISC=48                         # disc images (each asked in 5 answer formats)
N_GRID=40                         # T6 10x10 images (4 formats)
N_RC=48                           # RefCOCO images with a feasible content swap (2 formats)
N_HEADS=24                        # answers per condition in the head stage
N_KV_FULL=24                      # answers per full-set condition in the kv stage
N_KV=20                           # answers per order-variant condition
HEAD_CONDITIONS="disc_point t6g10_rowcol t6g10_point rc_point"
KV_CONDITIONS="disc_point disc_point_xyk disc_point_yxk disc_bbox disc_bbox_yxk t6g10_rowcol t6g10_colrow t6g10_point t6g10_point_yxk rc_point rc_point_yxk"
KV_FULL_CONDITIONS="disc_point t6g10_rowcol t6g10_point rc_point"   # full group set + decoder-only splice pairs
FLOW_CONDITIONS="disc_point t6g10_point rc_point"      # on the 2.5e answers
FIELD_CONDITIONS="t6g20_rowcol t6g10_point disc_point"  # on the 2.5e images
BATCH_SIZE=16                     # unpadded equal-length generation batches
HEAD_ROWS=48                      # rows per head-patching forward (row 0 = run B)
KV_ROWS=32                        # rows per KV-edit forward (row 0 = run B)
MAX_NEW_TOKENS=64
SPOTCHECK=20
SEED=20261020
OUTPUT_DIR="$WORK_DIR/analysis/position/${RUN_DATE}_coordinate_flow"
FIGURE_DIR="$ROOT_DIR/docs/analysis/position/figures/coordinate_flow"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$STAGE" in
  analyze)
    uv run --frozen python -m analysis_tools.position.coordinate_flow_analysis --run "$OUTPUT_DIR" --source "$SOURCE_DIR" --seed "$SEED"
    exit 0 ;;
  figures)
    uv run --frozen python -m analysis_tools.position.coordinate_flow_figures --run "$OUTPUT_DIR" --source "$SOURCE_DIR" --out "$FIGURE_DIR"
    exit 0 ;;
  cases) STAGES="cases" ;;
  gpu) STAGES="$GPU_STAGES" ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.position.coordinate_flow \
  --stages $STAGES --model "$MODEL" --output "$OUTPUT_DIR" --source "$SOURCE_DIR" \
  --n-disc "$N_DISC" --n-grid "$N_GRID" --n-rc "$N_RC" --head-conditions $HEAD_CONDITIONS --n-heads "$N_HEADS" --n-kv-full "$N_KV_FULL" --n-kv "$N_KV" \
  --kv-conditions $KV_CONDITIONS --kv-full-conditions $KV_FULL_CONDITIONS --flow-conditions $FLOW_CONDITIONS --field-conditions $FIELD_CONDITIONS \
  --batch-size "$BATCH_SIZE" --head-rows "$HEAD_ROWS" --kv-rows "$KV_ROWS" --max-new-tokens "$MAX_NEW_TOKENS" \
  --spotcheck "$SPOTCHECK" --seed "$SEED" --max-memory "$MAX_MEMORY"
