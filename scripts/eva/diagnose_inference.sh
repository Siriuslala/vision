#!/usr/bin/env bash
# Edit configuration, then run from the repository root:
# bash scripts/eva/diagnose_inference.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/load_env.sh"

source "$ROOT_DIR/scripts/decoding_presets.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="6"  # Select an idle GPU; avoid concurrent evaluation on the same GPU.
CASE_IDS=(445 462)
MAX_NEW_TOKENS=64
BATCHSIZE=1
THINKING="on"
RANDOM_SEED=42
ATTN_IMPLEMENTATION="flash_attention_2"
select_decoding_preset
OUTPUT_DIR="$WORK_DIR/diagnostics/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_DIR"
uv run --frozen python -m eva.evaluation \
  --model "$MODEL" --bench babyvision --case-ids "${CASE_IDS[@]}" \
  --gpu-ids "$GPU_IDS" --batchsize "$BATCHSIZE" --thinking "$THINKING" \
  --max-new-tokens "$MAX_NEW_TOKENS" --random-seed "$RANDOM_SEED" \
  --attn-implementation "$ATTN_IMPLEMENTATION" --do-sample \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --output "$OUTPUT_DIR/predictions.jsonl" 2>&1 | tee "$OUTPUT_DIR/run.log"
