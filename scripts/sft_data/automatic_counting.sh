#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/pixmo-count"
PROPOSAL_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/tmp/pixmo_sam_proposals"
OUTPUT_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/tmp/pixmo_automatic_trial"
MODEL="Qwen3.5-9B"
GPU="4"
LIMIT=5
MAX_NEW_TOKENS=2048
CACHE_ROOT="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
# =======================================================
if (( $# > 0 )); then echo "Configure this script instead of passing arguments." >&2; exit 2; fi
exec uv run --frozen python -m data.sft.tools.automatic_counting \
  --source-root "$SOURCE_ROOT" --proposal-root "$PROPOSAL_ROOT" --output-root "$OUTPUT_ROOT" \
  --data-root "${DATA_DIR_PUBLIC:-$DATA_DIR}" --cache-root "$CACHE_ROOT" --model "$MODEL" --gpu "$GPU" \
  --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"
