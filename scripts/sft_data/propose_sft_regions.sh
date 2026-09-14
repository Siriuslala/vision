#!/usr/bin/env bash
# Run: bash scripts/sft_data/propose_sft_regions.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
DATASET="cosyn" # cosyn or llava
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/CoSyn-400K"
OUTPUT="$WORK_DIR/sft_annotations/cosyn.jsonl"
MODEL="Qwen3.5-9B" # Use Qwen3.5-4B when GPU resources are limited.
GPU_ID="5"
LIMIT=2
MAX_NEW_TOKENS=1536
# =======================================================
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi
exec uv run --frozen python -m data.sft.tools.teacher --dataset "$DATASET" --source-root "$SOURCE_ROOT" \
  --output "$OUTPUT" --model "$MODEL" --cache-root "$HF_HUB_CACHE" --gpu "$GPU_ID" \
  --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"
