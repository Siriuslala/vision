#!/usr/bin/env bash
# Experiment 2.5e: verification of "sees right but answers wrong"
# (docs/analysis/representation/results_2026-10-16_sees_right.md). Configure this section, then run
# this script without terminal parameters.
# STAGE: cases (CPU) -> gen -> spotcheck -> readout -> causal (GPU) -> analyze -> figures (CPU).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="4"                       # never GPU 1 (reserved)
RUN_DATE="2026-10-16"
STAGE="cases"
CONDITIONS="disc_point t6g06_rowcol t6g06_point t6g10_rowcol t6g10_point t6g20_rowcol t6g20_point t6g20_rowcol_think t5g03_rowcol t5g03_point t5g06_rowcol t5g06_point t5g10_rowcol t5g10_point rc_point"
CAUSAL_CONDITIONS="disc_point t6g06_rowcol t6g10_point t6g20_rowcol t5g06_rowcol rc_point"
N_DISC=240                        # single red disc, diameters 32 / 64 pixels alternating
N_T6=160                          # T6 images per grid size (each asked in row/column and point)
N_THINK=96                        # 20x20 images also asked with the think-then-answer prompt
T6_GRIDS="6 10 20"
T5_GRIDS="3 6 10"
N_T5_CORRECT=60                   # T5 images per grid whose 1.3 answer was correct (all 1.3 content failures are added)
DECOMPOSITION_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
REFCOCO_CASES="$WORK_DIR/analysis/position/2026-10-14_coordinate_readout/refcoco/cases.json"
BATCH_SIZE=16                     # unpadded equal-length batches; halved for prompts of 800 tokens or more
MAX_NEW_TOKENS=64
MAX_NEW_THINK=1024
SPOTCHECK=20
N_CAUSAL=64                       # samples per causal condition (up to half answered wrong)
N_CAUSAL_LONG=40                  # the same for 20x20 grids (1600 visual tokens)
MAX_ROWS_LONG=6                   # patching rows per batch for 20x20 prompts
KV_CHUNK=4                        # KV-edit conditions per batch for 20x20 prompts
K_CELLS=2                         # grid displacement of the target (cells) for patching and KV edits
K_TOKENS=4                        # disc / RefCOCO displacement (merged tokens)
MAX_ROWS=24
SEED=20261016
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_sees_right"
FIGURE_DIR="$ROOT_DIR/docs/analysis/representation/figures/sees_right"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$STAGE" == "analyze" ]]; then
  uv run --frozen python -m analysis_tools.representation.sees_right_analysis --run "$OUTPUT_DIR" --seed "$SEED"
  exit 0
fi
if [[ "$STAGE" == "figures" ]]; then
  uv run --frozen python -m analysis_tools.representation.sees_right_figures --run "$OUTPUT_DIR" --out "$FIGURE_DIR" --seed "$SEED"
  exit 0
fi
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.representation.sees_right \
  --stage "$STAGE" --model "$MODEL" --output "$OUTPUT_DIR" --conditions $CONDITIONS \
  --causal-conditions $CAUSAL_CONDITIONS --n-disc "$N_DISC" --n-t6 "$N_T6" --n-think "$N_THINK" \
  --t6-grids $T6_GRIDS --t5-grids $T5_GRIDS --n-t5-correct "$N_T5_CORRECT" \
  --decomposition-dir "$DECOMPOSITION_DIR" --refcoco-cases "$REFCOCO_CASES" --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-new-think "$MAX_NEW_THINK" --spotcheck "$SPOTCHECK" \
  --n-causal "$N_CAUSAL" --k-cells "$K_CELLS" --k-tokens "$K_TOKENS" --max-rows "$MAX_ROWS" --max-rows-long "$MAX_ROWS_LONG" --kv-chunk "$KV_CHUNK" --n-causal-long "$N_CAUSAL_LONG" --seed "$SEED"
