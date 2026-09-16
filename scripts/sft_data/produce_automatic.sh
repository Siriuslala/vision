#!/usr/bin/env bash
# Run: bash scripts/sft_data/produce_automatic.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
# ==================== Configuration ====================
DATASET="llava" # llava, cosyn, thyme, visual_cot
SOURCE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/LLaVA-OneVision-Data"
SUBSETS=("TabMWP(MathV360K)" "Super-CLEVR(MathV360K)") # Empty () for Thyme/Visual-CoT.
EXAMPLE_ROOT="${DATA_DIR_PUBLIC:-$DATA_DIR}/tmp/sft_automatic"
OUTPUT_ROOT="$EXAMPLE_ROOT/$DATASET" # Must not already exist.
MODEL="Qwen3.5-9B" # Qwen3.5-4B is also supported.
GPU_ID="4"
LIMIT=8
SCAN_LIMIT=120
PER_SHARD_LIMIT=5 # Thyme only; zero disables the cap.
MAX_NEW_TOKENS=2048
MAX_PIXELS=1048576
WINDOW_METHOD="catalog"
SKIP_IMAGES=0 # Use a different small pilot without rereading the first images.
MAX_RETRIES=1 # At most one regeneration after a failed automatic check.
SOURCE_IDS=() # Optional exact source IDs for repeatable targeted checks.
# =======================================================
if (( $# > 0 )); then echo "Edit the script configuration instead of passing arguments." >&2; exit 2; fi
exec uv run --frozen python -m data.sft.tools.automatic \
  --dataset "$DATASET" --source-root "$SOURCE_ROOT" --subsets "${SUBSETS[@]}" \
  --output-root "$OUTPUT_ROOT" --data-root "$EXAMPLE_ROOT" \
  --model "$MODEL" --cache-root "$HF_HUB_CACHE" --gpu "$GPU_ID" \
  --limit "$LIMIT" --scan-limit "$SCAN_LIMIT" --per-shard-limit "$PER_SHARD_LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-pixels "$MAX_PIXELS" \
  --window-method "$WINDOW_METHOD" --skip-images "$SKIP_IMAGES" --max-retries "$MAX_RETRIES" \
  --source-ids "${SOURCE_IDS[@]}"
