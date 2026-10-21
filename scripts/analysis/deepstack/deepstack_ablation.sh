#!/usr/bin/env bash
# Experiment 4.3 (light): DeepStack ablation inside Qwen3-VL. Configure below, then run:
#   bash scripts/analysis/deepstack/deepstack_ablation.sh
# COMMAND: build | means | verify | run | spotcheck | score | analyze
# run resumes from existing per-case results; a changed configuration raises an error.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="${COMMAND:-run}"                     # environment overrides allow queued runs
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"        # Qwen3-VL-8B-Instruct | Qwen3-VL-4B-Instruct
CONDITION="${CONDITION:-full}"                # full | off | blur | mean | off1 | off2 | off3
TASKS="${TASKS:-T1 T2 T5 T6 RefCOCO}"         # subset of the built cases to run for this model
GPU_IDS="${GPU_IDS:-2}"
OUTPUT_DIR="$WORK_DIR/analysis/deepstack/2026-10-15_ablation_light"
DECOMP_SOURCE="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"   # formal 1.3 stimuli and cases
REFCOCO_SOURCE="$WORK_DIR/analysis/deepstack/2026-10-14_end_to_end/refcoco"   # 4.2 area-stratified sample
END_TO_END="$WORK_DIR/analysis/deepstack/2026-10-14_end_to_end"              # 4.2 results for reference lines

BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS_BINARY=8
MAX_NEW_TOKENS_LOCATE=256
N_REF_SYNTHETIC=250                # mean condition: synthetic reference images disjoint from the evaluated cases
N_REF_NATURAL=250                  # mean condition: RefCOCO testA images disjoint from the evaluated sample
SPOTCHECK_CONDITIONS="full off blur mean"
SPOTCHECK_N=20
N_BOOT=1000
SEED=0
FIGURE_DIR="$ROOT_DIR/docs/analysis/deepstack/figures/ablation_light"

PY=(uv run --frozen python -m analysis_tools.deepstack.deepstack_ablation)
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
case "$COMMAND" in
  build)
    exec "${PY[@]}" build --output "$OUTPUT_DIR" --decomp-source "$DECOMP_SOURCE" --refcoco-source "$REFCOCO_SOURCE" \
      --refcoco-dir "$REFCOCO_DIR" --n-ref-synthetic "$N_REF_SYNTHETIC" --n-ref-natural "$N_REF_NATURAL" --seed "$SEED" ;;
  means|verify)
    exec "${PY[@]}" "$COMMAND" --output "$OUTPUT_DIR" --model "$MODEL" ;;
  run)
    exec "${PY[@]}" run --output "$OUTPUT_DIR" --model "$MODEL" --condition "$CONDITION" --tasks $TASKS \
      --batch-size "$BATCH_SIZE" --max-new-tokens-binary "$MAX_NEW_TOKENS_BINARY" \
      --max-new-tokens-locate "$MAX_NEW_TOKENS_LOCATE" ;;
  spotcheck)
    exec "${PY[@]}" spotcheck --output "$OUTPUT_DIR" --model "$MODEL" --conditions $SPOTCHECK_CONDITIONS \
      --spotcheck-n "$SPOTCHECK_N" --seed "$SEED" --max-new-tokens-binary "$MAX_NEW_TOKENS_BINARY" \
      --max-new-tokens-locate "$MAX_NEW_TOKENS_LOCATE" ;;
  score)
    exec "${PY[@]}" score --output "$OUTPUT_DIR" ;;
  analyze)
    exec uv run --frozen python -m analysis_tools.deepstack.deepstack_ablation_analysis --output "$OUTPUT_DIR" \
      --end-to-end "$END_TO_END" --figures "$FIGURE_DIR" --n-boot "$N_BOOT" --seed "$SEED" ;;
  *) echo "Unknown COMMAND=$COMMAND" >&2; exit 1 ;;
esac
