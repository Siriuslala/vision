#!/usr/bin/env bash
# Configure below, then run: bash scripts/analysis/behavior/comparison_grounding.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="7"
OUTPUT_DIR="$WORK_DIR/grounding/comparison_grounding_v1"
NATURAL_COUNT=8
MAX_NEW_TOKENS=512
SCORE_GEOMETRY_ONLY=false

: "${REFCOCO_DIR:?Set REFCOCO_DIR in .env.}"
ARGS=()
case "$SCORE_GEOMETRY_ONLY" in
  true) ARGS+=(--score-geometry-only) ;;
  false) ;;
  *) echo 'SCORE_GEOMETRY_ONLY must be true or false.' >&2; exit 2 ;;
esac
exec uv run --frozen python -m analysis_tools.behavior.comparison_grounding \
  --model "$MODEL" --gpu-ids "$GPU_IDS" --output "$OUTPUT_DIR" \
  --refcoco "$REFCOCO_DIR" --natural-count "$NATURAL_COUNT" --max-new-tokens "$MAX_NEW_TOKENS" "${ARGS[@]}"
