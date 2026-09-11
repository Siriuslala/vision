#!/usr/bin/env bash
# Edit the configuration section below, then run this command:
# Run from the repository root after creating .env from .env_example:
# bash scripts/smoke_models.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/load_env.sh"
if (( $# > 0 )); then
  echo "Edit the configuration at the top of this script, then run it without arguments." >&2
  exit 2
fi

# ==================== Configuration ====================
MODELS=(
  "Qwen3.5-4B"
  "Qwen3.5-9B"
  "Qwen3-VL-4B-Instruct"
  "Qwen3-VL-8B-Instruct"
)
CASE_IDS=(445 462)
OUTPUT_DIR="$WORK_DIR/smoke_all_4096"
BENCH="babyvision"
SPLIT="train"                  # Official split name; used as a test benchmark here.
DATA_ROOT="$DATA_DIR"
CACHE_ROOT="$HF_HUB_CACHE"
BATCHSIZE=1
GPU_IDS="3"                    # Choose available GPUs; multi-GPU example: "0,3".
MAX_NEW_TOKENS=4096             # Maximum new tokens; generation stops early at EOS.
THINKING="off"                 # Qwen3.5: off/on/default; consider an initial budget of 8192 for on.
MAX_PIXELS=""                  # An empty string preserves the model processor defaults.
DTYPE="bfloat16"
ATTN_IMPLEMENTATION="sdpa"
RANDOM_SEED=0                 # Seed for Python, NumPy, and PyTorch (CPU/CUDA).
RESUME=false                   # Set true to resume the same output; other settings must match.
DRY_RUN=false                  # true: Validate cases and prompts without loading model weights.
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
if (( ${#CASE_IDS[@]} == 0 || ${#MODELS[@]} == 0 )); then
  echo "MODELS and CASE_IDS must not be empty." >&2
  exit 2
fi
# Load models sequentially with the same cases and generation settings.
for MODEL in "${MODELS[@]}"; do
  uv run --frozen python -m eva.simple_test \
    --model "$MODEL" --case_ids "${CASE_IDS[@]}" \
    --output "$OUTPUT_DIR/$MODEL.jsonl" "${ARGS[@]}"
done
