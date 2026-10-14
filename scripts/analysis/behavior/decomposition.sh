#!/usr/bin/env bash
# Experiment 1.3 (find-the-difference decomposition). Configure below, then run:
#   bash scripts/analysis/behavior/decomposition.sh
# COMMAND: build | run | score | analyze | supplement | reasoning | babyvision_annotate | babyvision_build
# "run" resumes from existing per-case results; a changed configuration raises an error.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="babyvision_build"
STAGE="main"                      # main | explicit | thinking | babyvision (for COMMAND=run)
LIMIT=""                          # empty = all pending cases of the stage
MODEL="Qwen3.5-9B"
GPU_IDS="2"
OUTPUT_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"

# Stimuli and case construction (build).
SEED=0
IMAGES_PER_UNIT=48                # images per change type x magnitude x grid size
COUNT_IMAGES_PER_GRID=16          # T0 images per grid size
ALT_PERIOD=5                      # every 5th image also receives the alternative wording
REASONING_GRIDS="6 20"
EXPLICIT_LEVELS="1 3"
EXPLICIT_PER_CELL=6
THINKING_LEVELS="3"
THINKING_PER_CELL=2

# Decoding budgets (run).
MAX_NEW_TOKENS_BINARY=8
MAX_NEW_TOKENS_LOCATE=256
MAX_NEW_TOKENS_EXPLICIT=2048
MAX_NEW_TOKENS_THINKING=8192

# BabyVision correspondence (babyvision_build).
BABYVISION_ANNOTATION="$DATA_DIR/BabyVision_annotations/find_the_different_grid_v1.json"

COMMON=(--output "$OUTPUT_DIR" --seed "$SEED" --images-per-unit "$IMAGES_PER_UNIT"
  --count-images-per-grid "$COUNT_IMAGES_PER_GRID" --alt-period "$ALT_PERIOD"
  --reasoning-grids $REASONING_GRIDS --explicit-levels $EXPLICIT_LEVELS --explicit-per-cell "$EXPLICIT_PER_CELL"
  --thinking-levels $THINKING_LEVELS --thinking-per-cell "$THINKING_PER_CELL"
  --model "$MODEL" --max-new-tokens-binary "$MAX_NEW_TOKENS_BINARY" --max-new-tokens-locate "$MAX_NEW_TOKENS_LOCATE"
  --max-new-tokens-explicit "$MAX_NEW_TOKENS_EXPLICIT" --max-new-tokens-thinking "$MAX_NEW_TOKENS_THINKING")

case "$COMMAND" in
  build|score)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition "$COMMAND" "${COMMON[@]}" ;;
  run)
    ARGS=(--stage "$STAGE")
    [[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec uv run --frozen python -m analysis_tools.behavior.decomposition run \
      "${COMMON[@]}" "${ARGS[@]}" ;;
  analyze)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition_analysis --output "$OUTPUT_DIR" ;;
  supplement)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition_supplement --output "$OUTPUT_DIR" ;;
  reasoning)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition_reasoning --output "$OUTPUT_DIR" ;;
  babyvision_annotate|babyvision_build)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition_babyvision "${COMMAND#babyvision_}" --output "$OUTPUT_DIR" \
      --babyvision "$DATA_DIR/BabyVision" --annotation "$BABYVISION_ANNOTATION" --seed "$SEED" ;;
  *) echo "Unknown COMMAND: $COMMAND" >&2; exit 2 ;;
esac
