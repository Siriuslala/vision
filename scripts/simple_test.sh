#!/usr/bin/env bash
# Edit the configuration section below, then run this command:
# Run from the repository root after creating .env from .env_example:
# bash scripts/simple_test.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/load_env.sh"

# ==================== Configuration ====================
MODEL="Qwen3.5-4B"
# Each image is paired with the prompt at the same array index.
IMAGE_PATHS=(
  "$DATA_DIR/BabyVision/train/image/445.jpg"
  "$DATA_DIR/BabyVision/train/image/462.jpg"
)
PROMPTS=(
  "One tiger in this 7-by-7 grid differs from the others. Give its row and column as (row,column)."
  "Which of the four options A, B, C, D matches the pattern at the top? Explain your choice."
)
OUTPUT_FILE="$WORK_DIR/qwen35_4b_images_4096.jsonl"
CACHE_ROOT="$HF_HUB_CACHE"
BATCHSIZE=2
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
  --cache-root "$CACHE_ROOT"
  --image_path "${IMAGE_PATHS[@]}" --prompt "${PROMPTS[@]}"
  --batchsize "$BATCHSIZE" --gpu_ids "$GPU_IDS"
  --max-new-tokens "$MAX_NEW_TOKENS" --thinking "$THINKING"
  --dtype "$DTYPE" --attn-implementation "$ATTN_IMPLEMENTATION" --random-seed "$RANDOM_SEED"
)
[[ -z "$MAX_PIXELS" ]] || ARGS+=(--max-pixels "$MAX_PIXELS")
[[ "$RESUME" == false ]] || ARGS+=(--resume)
[[ "$DRY_RUN" == false ]] || ARGS+=(--dry-run)
if (( ${#IMAGE_PATHS[@]} == 0 || ${#IMAGE_PATHS[@]} != ${#PROMPTS[@]} )); then
  echo "IMAGE_PATHS and PROMPTS must be nonempty arrays with equal lengths." >&2
  exit 2
fi
exec uv run --frozen python -m eva.simple_test --model "$MODEL" --output "$OUTPUT_FILE" "${ARGS[@]}"
