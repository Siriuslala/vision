#!/usr/bin/env bash
# Experiment 1.6: ViT PE scaling on grids, BabyVision and RefCOCO, plus T0 counting with image variation
# (Qwen3.5-9B, thinking off; docs/analysis/behavior/results_2026-10-16_pe_scaling_t0.md).
# Configure below, then run:
#   bash scripts/analysis/behavior/pe_scaling_t0.sh
# COMMAND: build | run | spotcheck | analyze
# "run" resumes from existing results; SHARD="i/k" processes every k-th pending case (BabyVision: batch) from i.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

COMMAND="${COMMAND:-build}"
GPU_IDS="${GPU_IDS:-5}"                                # never GPU 1 (reserved)
STAGES="${STAGES:-grid grid_explicit t0 t0_explicit refcoco}"   # any of: grid grid_explicit t0 t0_explicit refcoco babyvision
CONDITIONS="${CONDITIONS:-x1 x2 x2.76}"                   # PE scale factors 1, 2 and 2.756
SHARD="${SHARD:-0/1}"
PARTS="${PARTS:-grid t0 refcoco babyvision}"   # analyze: parts to (re)score
MODEL="Qwen3.5-9B"
ATTN="${ATTN:-sdpa}"                                # flash_attention_2 for babyvision (as in the reference run)
OUTPUT_DIR="$WORK_DIR/analysis/behavior/2026-10-16_pe_scaling_t0"
FORMAL_DIR="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"     # read only
T0_DATASET="$DATA_DIR/stimuli_t0_count_variation_v1"
BV_REFERENCE="$WORK_DIR/eval/qwen35_9b_babyvision-thinkingoff-gpua800-random_seed42-do_sampletrue-temperature1.0-top_p1.0-top_k40-min_p0.0-repetition_penalty1.0.jsonl"
FIGURE_DIR="$ROOT_DIR/docs/analysis/behavior/figures/pe_scaling_t0"

# Build.
SEED=0
PER_COMBO=12                               # T4/T5 cases per (task, format, grid, kind, level)
CONTROL_PER_COMBO=4                        # T1/T2 cases per (task, kind, level, variant)
T0_SEED=20261016
T0_PER_VARIATION=32                        # images per layout and variation (V1-V3); V0 has one per template
T0_EXPLICIT_PER_LAYOUT=12                  # think-then-answer T0 cases (V3 images) per layout
REFCOCO_PER_STRATUM=100

# Decoding.
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_SIZE_EXPLICIT="${BATCH_SIZE_EXPLICIT:-16}"
MAX_NEW_TOKENS_BINARY=8
MAX_NEW_TOKENS_LOCATE=256
MAX_NEW_TOKENS_EXPLICIT=2048
BV_BATCH_SIZE=8
BV_SEEDS="${BV_SEEDS:-42}"
BV_MAX_NEW_TOKENS=32768                    # sampling settings of the reference thinking-off run
BV_TEMPERATURE=1.0
BV_TOP_K=40
BV_TOP_P=1.0

[[ ",$GPU_IDS," == *",1,"* ]] && { echo "GPU 1 is reserved; choose another GPU." >&2; exit 2; }
PY=(uv run --frozen python -m analysis_tools.behavior.pe_scaling_t0)
COMMON=(--output "$OUTPUT_DIR" --formal "$FORMAL_DIR" --t0-dataset "$T0_DATASET" --seed "$SEED"
  --t0-explicit-per-layout "$T0_EXPLICIT_PER_LAYOUT")

case "$COMMAND" in
  build)
    "${PY[@]}" build-grid "${COMMON[@]}" --per-combo "$PER_COMBO" --control-per-combo "$CONTROL_PER_COMBO"
    [[ -f "$T0_DATASET/manifest.json" ]] || "${PY[@]}" build-t0 "${COMMON[@]}" --t0-per-variation "$T0_PER_VARIATION" \
      --seed "$T0_SEED"
    "${PY[@]}" build-refcoco "${COMMON[@]}" --refcoco "$REFCOCO_DIR" --refcoco-per-stratum "$REFCOCO_PER_STRATUM"
    "${PY[@]}" build-babyvision "${COMMON[@]}" --data-root "$DATA_DIR" --babyvision-reference "$BV_REFERENCE" ;;
  run)
    mkdir -p "$OUTPUT_DIR/logs"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "${PY[@]}" run "${COMMON[@]}" --model "$MODEL" --attn "$ATTN" --stages $STAGES --conditions $CONDITIONS \
      --shard "$SHARD" --batch-size "$BATCH_SIZE" --batch-size-explicit "$BATCH_SIZE_EXPLICIT" \
      --max-new-tokens-binary "$MAX_NEW_TOKENS_BINARY" --max-new-tokens-locate "$MAX_NEW_TOKENS_LOCATE" \
      --max-new-tokens-explicit "$MAX_NEW_TOKENS_EXPLICIT" --data-root "$DATA_DIR" --bv-batch-size "$BV_BATCH_SIZE" \
      --bv-seeds $BV_SEEDS --bv-max-new-tokens "$BV_MAX_NEW_TOKENS" --bv-temperature "$BV_TEMPERATURE" \
      --bv-top-k "$BV_TOP_K" --bv-top-p "$BV_TOP_P" \
      2>&1 | grep --line-buffered -v 'causal_conv1d\|Loading weights' \
      | tee -a "$OUTPUT_DIR/logs/run_${STAGES// /-}_${SHARD/\//of}.log" ;;
  spotcheck)
    "${PY[@]}" spotcheck "${COMMON[@]}" ;;
  analyze)
    uv run --frozen python -m analysis_tools.behavior.pe_scaling_t0_analysis --output "$OUTPUT_DIR" \
      --formal "$FORMAL_DIR" --t0-dataset "$T0_DATASET" --data-root "$DATA_DIR" --figures "$FIGURE_DIR" \
      --t0-explicit-per-layout "$T0_EXPLICIT_PER_LAYOUT" --babyvision-reference "$BV_REFERENCE" --parts $PARTS ;;
  *) echo "Unknown COMMAND: $COMMAND" >&2; exit 2 ;;
esac
