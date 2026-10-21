#!/usr/bin/env bash
# Experiment 4.4: DeepStack follow-ups (blur sweep, content-free injection, non-coordinate control,
# representation readability). Configure below, then run:
#   bash scripts/analysis/deepstack/deepstack_todo.sh
# COMMAND: build | verify | run | spotcheck | represent | score | analyze
# run resumes from existing per-case results; a changed configuration raises an error.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="${COMMAND:-run}"                     # environment overrides allow queued runs
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"        # Qwen3-VL-8B-Instruct | Qwen3-VL-4B-Instruct | Qwen3.5-9B (represent only)
CONDITION="${CONDITION:-blank}"               # full | off | mean | blur2 | blur4 | blur8 | blur16 | blank | blankflip
TASKS="${TASKS:-T5 T6 RefCOCO}"               # part A: T5 T6 RefCOCO; part B: POPE TextVQA BabyVision
GPU_IDS="${GPU_IDS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="$WORK_DIR/analysis/deepstack/2026-10-20_todo"
ABLATION_DIR="$WORK_DIR/analysis/deepstack/2026-10-15_ablation_light"   # 4.3 cases, runs and mean vectors
MAX_NEW_TOKENS_BINARY=8
MAX_NEW_TOKENS_LOCATE=256
N_BOOT=1000
SPOTCHECK_CONDITIONS="${SPOTCHECK_CONDITIONS:-blank blankflip blur8}"
SPOTCHECK_N=20
SEED=0
FIGURE_DIR="$ROOT_DIR/docs/analysis/deepstack/figures/todo"
PROBE_DEVICE="${PROBE_DEVICE:-cpu}"            # cuda:0 speeds up the ridge eigendecompositions of part C
SKIP_REPRESENTATION="${SKIP_REPRESENTATION:-0}"   # 1 = reuse analysis/representation.json (behaviour tables and figures only)

PY=(uv run --frozen python -m analysis_tools.deepstack.deepstack_todo)
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
case "$COMMAND" in
  build)
    exec "${PY[@]}" build --output "$OUTPUT_DIR" --ablation "$ABLATION_DIR" --data-dir "$DATA_DIR" --seed "$SEED" ;;
  verify)
    exec "${PY[@]}" verify --output "$OUTPUT_DIR" --ablation "$ABLATION_DIR" --model "$MODEL" ;;
  run)
    exec "${PY[@]}" run --output "$OUTPUT_DIR" --ablation "$ABLATION_DIR" --model "$MODEL" --condition "$CONDITION" \
      --tasks $TASKS --batch-size "$BATCH_SIZE" --max-new-tokens-binary "$MAX_NEW_TOKENS_BINARY" \
      --max-new-tokens-locate "$MAX_NEW_TOKENS_LOCATE" ;;
  spotcheck)
    exec "${PY[@]}" spotcheck --output "$OUTPUT_DIR" --ablation "$ABLATION_DIR" --model "$MODEL" \
      --conditions $SPOTCHECK_CONDITIONS --spotcheck-n "$SPOTCHECK_N" --seed "$SEED" ;;
  represent)
    exec "${PY[@]}" represent --output "$OUTPUT_DIR" --model "$MODEL" --seed "$SEED" ;;
  score)
    exec "${PY[@]}" score --output "$OUTPUT_DIR" ;;
  analyze)
    PROBE_DEVICE="$PROBE_DEVICE" exec uv run --frozen python -m analysis_tools.deepstack.deepstack_todo_analysis --output "$OUTPUT_DIR" \
      --ablation "$ABLATION_DIR" --figures "$FIGURE_DIR" --n-boot "$N_BOOT" --seed "$SEED" \
      $( [ "$SKIP_REPRESENTATION" = 1 ] && echo --skip-representation ) ;;
  *) echo "Unknown COMMAND=$COMMAND" >&2; exit 1 ;;
esac
