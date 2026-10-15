#!/usr/bin/env bash
# Supplementary conditions of experiment 1.3 (T3 prompt revision, non-square grids, fixed canvas, ring).
# Configure below, then run:
#   bash scripts/analysis/behavior/decomposition_supplement.sh
# COMMAND: build | run | spotcheck | score | analyze
# PART:    t3dev | t3full | layouts | ring   (analyze ignores PART)
# "run" resumes from existing results; SHARD="i/k" runs every k-th pending case starting at i.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="build"
PART="t3dev"
GPU_IDS="1"
SHARD="0/1"
LIMIT=""                                  # empty = all pending cases
MODEL="Qwen3.5-9B"
OUTPUT_DIR="$WORK_DIR/analysis/behavior/2026-10-14_decomposition_supplement"
FORMAL_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"   # read only

# Case construction.
SEED=0
DEV_REPS=4                                # t3dev: replicates 0..3 of every unit ...
DEV_LEVELS="2 4"                          # ... at these magnitude levels
T3_SELECTED="C5_majority"                 # chosen on t3dev (analysis/tables.md); used by t3full and the layouts
LAYOUT_T3_CANDIDATES="C0_original C5_majority"
LAYOUTS="g4x8 g8x4 g10x20 g20x10 f6c128 f12c64 f24c32 g20c32"
SEARCH_PER_CHANGE=96                      # images per search change (gap 16 px, rotation 45 deg)
RED_PER_LAYOUT=192                        # T6 images per layout or ring
COUNT_PER_LAYOUT=48                       # T0 images per layout or ring
RINGS="n8 n12 n16 n24 n8_r128 n8_r256"
RING_FULL="n8 n12 n16 n24"                # rings that also receive T5 and T0 (others: T6 only)

# Decoding.
MAX_NEW_TOKENS=256
BATCH_SIZE=8
SPOTCHECK_N=20

ARGS=(--part "$PART" --output "$OUTPUT_DIR" --formal "$FORMAL_DIR" --seed "$SEED" --dev-reps "$DEV_REPS"
  --dev-levels $DEV_LEVELS --layouts $LAYOUTS --layout-t3-candidates $LAYOUT_T3_CANDIDATES
  --search-per-change "$SEARCH_PER_CHANGE" --red-per-layout "$RED_PER_LAYOUT" --count-per-layout "$COUNT_PER_LAYOUT"
  --rings $RINGS --ring-full $RING_FULL --model "$MODEL" --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "$BATCH_SIZE" --shard "$SHARD" --spotcheck-n "$SPOTCHECK_N")
[[ -n "$T3_SELECTED" ]] && ARGS+=(--t3-selected "$T3_SELECTED")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")

case "$COMMAND" in
  build|score)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition_conditions "$COMMAND" "${ARGS[@]}" ;;
  run|spotcheck)
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec uv run --frozen python -m analysis_tools.behavior.decomposition_conditions \
      "$COMMAND" "${ARGS[@]}" ;;
  analyze)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition_conditions_analysis \
      --output "$OUTPUT_DIR" --formal "$FORMAL_DIR" ;;
  *) echo "Unknown COMMAND: $COMMAND" >&2; exit 2 ;;
esac
