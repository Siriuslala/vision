#!/usr/bin/env bash
# Experiment 4.5: DeepStack ablations on natural fine-grained tasks (TextVQA small text, DocVQA, GQA small
# objects, CUB attributes, MMVP), with evidence-region replacement. Configure below, then run:
#   bash scripts/analysis/deepstack/deepstack_natural.sh
# COMMAND: build | run | score | analyze
# run resumes from existing per-case results; a changed configuration raises an error.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="${COMMAND:-run}"                     # environment overrides allow queued runs
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"        # Qwen3-VL-8B-Instruct | Qwen3-VL-4B-Instruct | Qwen3.5-9B (full only)
# Space-separated list, run in order inside one queue: full off mean blank blankflip blur2 blur8 blank_in blank_out blank_ctrl
CONDITIONS="${CONDITIONS:-full off mean blank blankflip blur2 blur8 blank_in blank_out blank_ctrl}"
TASKS="${TASKS:-textvqa docvqa gqa cub mmvp}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPLIT="${SPLIT:-}"                           # e.g. "12,7": split the model over the GPUs in GPU_IDS with these GiB caps
SEED=0
DOCVQA_N=200                                  # DocVQA items in every condition except Full (Full runs all 400)
N_BOOT=2000
OUTPUT_DIR="$WORK_DIR/analysis/deepstack/2026-10-21_natural"
ABLATION_DIR="$WORK_DIR/analysis/deepstack/2026-10-15_ablation_light"   # 4.3 reference means for the mean condition
TODO_DIR="$WORK_DIR/analysis/deepstack/2026-10-20_todo"                  # 4.4 TextVQA/POPE/RefCOCO runs (reused)
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv | head -1)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/deepstack/figures/natural"

PY=(uv run --frozen python -m analysis_tools.deepstack.deepstack_natural)
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
case "$COMMAND" in
  build)
    exec "${PY[@]}" build --output "$OUTPUT_DIR" --data-dir "$DATA_DIR" --data-public "$DATA_DIR_PUBLIC" \
      --mmvp-csv "$MMVP_CSV" --seed "$SEED" ;;
  run)
    for condition in $CONDITIONS; do
      "${PY[@]}" run --output "$OUTPUT_DIR" --ablation "$ABLATION_DIR" --model "$MODEL" --condition "$condition" \
        --tasks $TASKS --batch-size "$BATCH_SIZE" --docvqa-n "$DOCVQA_N" $( [ -n "$SPLIT" ] && echo --split "$SPLIT" )
    done ;;
  score)
    exec "${PY[@]}" score --output "$OUTPUT_DIR" ;;
  analyze)
    exec uv run --frozen python -m analysis_tools.deepstack.deepstack_natural_analysis --output "$OUTPUT_DIR" \
      --todo "$TODO_DIR" --figures "$FIGURE_DIR" --n-boot "$N_BOOT" --seed "$SEED" ;;
  *) echo "Unknown COMMAND=$COMMAND" >&2; exit 1 ;;
esac
