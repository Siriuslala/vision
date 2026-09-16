#!/usr/bin/env bash
# Edit the configuration section below, then run this command:
# Run from the repository root after creating .env from .env_example:
# bash scripts/eva/evaluation.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"
if (( $# > 0 )); then
  echo "Edit the configuration at the top of this script, then run it without arguments." >&2
  exit 2
fi

source "$ROOT_DIR/scripts/decoding_presets.sh"

# ==================== Configuration ====================
MODEL="Qwen3.5-9B"
GPU_IDS="0"                    # Choose available GPUs; multi-GPU example: "0,3".
GPU_TAG="a40"
GPU_TAG="a800"
BENCH="babyvision"
CASE_IDS=()                    # An empty array selects all 388 cases; subset example: (445 462).
SPLIT="train"                  # Official split name; used as a test benchmark here.
BATCHSIZE=4
RANDOM_SEED=42
THINKING="off"                 # Qwen3.5: on/off/default; Qwen3-VL: match checkpoint or use default.
MAX_NEW_TOKENS=32768          # Maximum new tokens; generation stops early at EOS.
MAX_PIXELS=""                  # An empty string preserves the model processor defaults.
DTYPE="bfloat16"
ATTN_IMPLEMENTATION="flash_attention_2"  # FlashAttention 2; alternatives: sdpa, eager.
# Sampling settings are selected from MODEL and THINKING in decoding_presets.sh.
RESUME=false                   # Set true to resume the same output; other settings must match.
DRY_RUN=false                  # true: Validate cases and prompts without loading model weights.
DATA_ROOT="$DATA_DIR"
CACHE_ROOT="$HF_HUB_CACHE"
OUTPUT_DIR="$WORK_DIR/eval"
select_decoding_preset
OUTPUT_MODEL="${MODEL_SHORT,,}"
OUTPUT_MODEL="${OUTPUT_MODEL//./}"
OUTPUT_PREFIX="${OUTPUT_MODEL//-/_}_${BENCH}"
# Keep the filename derived from the settings above; full provenance is in metadata.
OUTPUT_TAG="thinking${THINKING}-gpu${GPU_TAG}-random_seed${RANDOM_SEED}-do_sample${DO_SAMPLE}"
OUTPUT_TAG+="-temperature${TEMPERATURE}-top_p${TOP_P}-top_k${TOP_K}-min_p${MIN_P}"
OUTPUT_TAG+="-repetition_penalty${REPETITION_PENALTY}"
OUTPUT_FILE="$OUTPUT_DIR/${OUTPUT_PREFIX}-${OUTPUT_TAG}.jsonl"
# ==================================================

ARGS=(
  --temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P"
  --min-p "$MIN_P" --repetition-penalty "$REPETITION_PENALTY"
  --bench "$BENCH" --split "$SPLIT"
  --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT"
  --batchsize "$BATCHSIZE" --gpu_ids "$GPU_IDS"
  --max-new-tokens "$MAX_NEW_TOKENS" --thinking "$THINKING"
  --dtype "$DTYPE" --attn-implementation "$ATTN_IMPLEMENTATION" --random-seed "$RANDOM_SEED"==
)
case "$DO_SAMPLE" in
  true) ARGS+=(--do-sample) ;;
  false) ;;
  *) echo "DO_SAMPLE must be true or false." >&2; exit 2 ;;
esac
[[ -z "$MAX_PIXELS" ]] || ARGS+=(--max-pixels "$MAX_PIXELS")
[[ "$RESUME" == false ]] || ARGS+=(--resume)
[[ "$DRY_RUN" == false ]] || ARGS+=(--dry-run)
if (( ${#CASE_IDS[@]} > 0 )); then ARGS+=(--case_ids "${CASE_IDS[@]}"); fi
exec uv run --frozen python -m eva.evaluation --model "$MODEL" --output "$OUTPUT_FILE" "${ARGS[@]}"
