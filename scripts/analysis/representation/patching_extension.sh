#!/usr/bin/env bash
# Experiments 2.5b (failure cases) and 2.5c (generation-phase patching), see
# docs/analysis/representation/design_causal.md.  Configure this section, then run this script
# without terminal parameters.
# STAGE: screen (decode answers / baseline prompts, choose the scored token) -> spotcheck (manual greedy
# decoder vs qwen_runtime.generate) -> scan (patching) -> detail (single answer-token patches on generation
# pairs) -> analyze (summaries and figures).
# Parallel processes must use different TAG values; analyze merges every tag in OUTPUT_DIR.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
MODEL="Qwen3.5-9B"
GPU_IDS="0"
RUN_DATE="2026-10-15"
STAGE="screen"
TAG="gen"
# 2.5c generation conditions: t6_rowcol_col t6_rowcol_row t6_point_x t6_point_y t5_rowcol_col t5_rowcol_row t5_rowcol_remove
# 2.5b conditions: t3_gap16_g06 t3_gap16_g10 t3_gap16_g20 t3_red_g06 t3_color20_g06 t3_color10_g06 t5_fail t3_fail
CONDITIONS="t6_rowcol_col t6_rowcol_row t6_point_x t5_rowcol_col t5_rowcol_row t5_rowcol_remove t5_fail"
PAIRS_PER_CONDITION=24          # 2.5c generation pairs per condition
YESNO_PAIRS=40                  # 2.5b revised-T3 pairs per condition
FAIL_GRIDS="3 6 10"             # experiment-1.3 grids for t5_fail / t3_fail
FAIL_MAX_PER_GRID=0             # 0 keeps every content failure
DECOMPOSITION_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"
BATCH_SIZE=8                    # 6x6 grids; halved for 10x10 and quartered for 20x20
SITES="$(seq -s ' ' 0 32)"      # generation pairs: decoder input (0) and every layer output
YESNO_SITES="0 2 4 6 7 8 10 11 12 14 15 16 18 20 24 28 31 32"
SANITY_SITES="1 16"
DETAIL_SITES="12 16 19 20 22 24 28 31"
MAX_NEW_TOKENS=48
SPOTCHECK=20
N_BOOT=2000
SEED=0
OUTPUT_DIR="$WORK_DIR/analysis/representation/${RUN_DATE}_patching_extension"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# shellcheck disable=SC2086
uv run --frozen python -m analysis_tools.representation.patching_extension \
  --stage "$STAGE" --model "$MODEL" --output "$OUTPUT_DIR" --tag "$TAG" --conditions $CONDITIONS \
  --pairs-per-condition "$PAIRS_PER_CONDITION" --yesno-pairs "$YESNO_PAIRS" --fail-grids $FAIL_GRIDS \
  --fail-max-per-grid "$FAIL_MAX_PER_GRID" --decomposition-dir "$DECOMPOSITION_DIR" --batch-size "$BATCH_SIZE" \
  --sites $SITES --yesno-sites $YESNO_SITES --sanity-sites $SANITY_SITES --detail-sites $DETAIL_SITES --max-new-tokens "$MAX_NEW_TOKENS" \
  --spotcheck "$SPOTCHECK" --n-boot "$N_BOOT" --seed "$SEED"
