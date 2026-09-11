#!/usr/bin/env bash
# Edit the configuration section below, then run this command:
# Run from the repository root after creating .env from .env_example:
# bash scripts/evaluation.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/load_env.sh"
if (( $# > 0 )); then
  echo "Edit the configuration at the top of this script, then run it without arguments." >&2
  exit 2
fi

# ==================== Configuration ====================
MODEL="Qwen3.5-9B"
GPU_IDS="3"                    # Choose available GPUs; multi-GPU example: "0,3".
BENCH="babyvision"
CASE_IDS=()                    # An empty array selects all 388 cases; subset example: (445 462).
OUTPUT_FILE="$WORK_DIR/qwen35_9b_full_4096.jsonl"
SPLIT="train"                  # Official split name; used as a test benchmark here.
RANDOM_SEED=0
BATCHSIZE=1
MAX_NEW_TOKENS=4096             # Maximum new tokens; generation stops early at EOS.
THINKING="off"                 # Qwen3.5: off/on/default; consider an initial budget of 8192 for on.
MAX_PIXELS=""                  # An empty string preserves the model processor defaults.
DTYPE="bfloat16"
ATTN_IMPLEMENTATION="sdpa"                 # Seed for Python, NumPy, and PyTorch (CPU/CUDA).
RESUME=false                   # Set true to resume the same output; other settings must match.
DRY_RUN=false                  # true: Validate cases and prompts without loading model weights.
DATA_ROOT="$DATA_DIR"
CACHE_ROOT="$HF_HUB_CACHE"
# ==================================================

ARGS=(
  --bench "$BENCH" --split "$SPLIT"
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT"
  --batchsize "$BATCHSIZE" --gpu_ids "$GPU_IDS"
  --max-new-tokens "$MAX_NEW_TOKENS" --thinking "$THINKING"
  --dtype "$DTYPE" --attn-implementation "$ATTN_IMPLEMENTATION" --random-seed "$RANDOM_SEED"
)
[[ -z "$MAX_PIXELS" ]] || ARGS+=(--max-pixels "$MAX_PIXELS")
[[ "$RESUME" == false ]] || ARGS+=(--resume)
[[ "$DRY_RUN" == false ]] || ARGS+=(--dry-run)
if (( ${#CASE_IDS[@]} > 0 )); then ARGS+=(--case_ids "${CASE_IDS[@]}"); fi
exec uv run --frozen python -m eva.evaluation --model "$MODEL" --output "$OUTPUT_FILE" "${ARGS[@]}"
