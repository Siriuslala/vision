#!/usr/bin/env bash
# Experiment 4.2 (DeepStack end-to-end comparison). Configure below, then run:
#   bash scripts/analysis/deepstack/deepstack_comparison.sh
# COMMAND: babyvision | bv_merge | bv_extension | decomp_prepare | decomp_run | t3c5_run | spotcheck | batch_validate | decomp_score | decomp_analyze |
#          refcoco_build | refcoco_run | refcoco_score | analyze | reasoning
# Run commands resume from existing per-case results; a changed configuration raises an error.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"
source "$ROOT_DIR/scripts/decoding_presets.sh"

COMMAND="${COMMAND:-decomp_run}"      # environment overrides allow queued runs
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"   # Qwen3-VL-8B-Instruct | Qwen3.5-9B | Qwen3-VL-4B-Instruct | Qwen3.5-4B
GPU_IDS="${GPU_IDS:-1}"
OUTPUT_DIR="$WORK_DIR/analysis/deepstack/2026-10-14_end_to_end"
DECOMP_SOURCE="$WORK_DIR/analysis/behavior/2026-10-13_decomposition"   # formal 1.3 run (stimuli, cases, Qwen3.5-9B)
T3_SUPPLEMENT="$WORK_DIR/analysis/behavior/2026-10-14_decomposition_supplement"   # revised T3 wording C5

# Experiment 1.3 sub-tasks and RefCOCO (greedy, thinking off).
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-}"                         # empty = all pending cases
MAX_NEW_TOKENS_BINARY=8
MAX_NEW_TOKENS_LOCATE=256
SPOTCHECK_KIND="${SPOTCHECK_KIND:-decomp}"           # decomp | refcoco
SPOTCHECK_N=20
VALIDATE_PER_CELL=8                # batch_validate: cases per task x format x grid against the 1.3 batch-size-1 run
SEED=0
REFCOCO_PER_STRATUM=300

# BabyVision (official per-model thinking-off / Instruct sampling preset, as the existing Qwen3.5-9B run).
BV_THINKING="off"
BV_SEED=42
BV_ATTN="flash_attention_2"
BV_REFERENCE="$WORK_DIR/eval/qwen35_9b_babyvision-thinkingoff-gpua800-random_seed42-do_sampletrue-temperature1.0-top_p1.0-top_k40-min_p0.0-repetition_penalty1.0.jsonl"
BV_ORDER_BY_REFERENCE=true         # batch items of similar Qwen3.5-9B response length together (throughput only)
BV_MAX_NEW_TOKENS_QWEN35=32768     # same budget as the existing Qwen3.5-9B run
BV_BATCHSIZE_QWEN35=8
BV_MAX_NEW_TOKENS_QWEN3VL=4096     # without presence_penalty, Qwen3-VL loops until the budget on hard items
BV_BATCHSIZE_QWEN3VL="${BV_BATCHSIZE_QWEN3VL:-16}"
BV_TAIL_FROM="${BV_TAIL_FROM:-}"   # non-empty: second worker on length ranks BV_TAIL_FROM.. (shortest first), separate file
BV_EXTENSION_N=32                  # bv_extension: truncated Qwen3-VL items re-answered with a larger budget
BV_EXTENSION_MAX_NEW_TOKENS=16384
BV_RANK_RANGE="${BV_RANK_RANGE:-}" # non-empty "start stop": extra worker on length ranks [start, stop), separate file

# BabyVision reasoning coding (design_decomposition.md 6.3).
REASONING_STEP="${REASONING_STEP:-all}"             # code | export | summarize | all (export never overwrites an existing sample)
BABYVISION_ANNOTATION="$DATA_DIR/BabyVision_annotations/find_the_different_grid_v1.json"
MANUAL_WRONG_PER_SUBTYPE=2         # per model and non-FtD subtype; Find the different is coded in full
MANUAL_RIGHT_PER_SUBTYPE=1

DECOMP_DIR="$OUTPUT_DIR/decomposition"
MODEL_SLUG="${MODEL,,}"; MODEL_SLUG="${MODEL_SLUG//./}"; MODEL_SLUG="${MODEL_SLUG/-instruct/}"; MODEL_SLUG="${MODEL_SLUG//-/_}"
PY=(uv run --frozen python -m analysis_tools.deepstack.deepstack_comparison)
COMMON=(--model "$MODEL" --batch-size "$BATCH_SIZE" --seed "$SEED"
  --max-new-tokens-binary "$MAX_NEW_TOKENS_BINARY" --max-new-tokens-locate "$MAX_NEW_TOKENS_LOCATE")

case "$COMMAND" in
  babyvision)
    THINKING="$BV_THINKING"; select_decoding_preset
    BV_MAX_NEW_TOKENS="$BV_MAX_NEW_TOKENS_QWEN35"; BV_BATCHSIZE="$BV_BATCHSIZE_QWEN35"
    if [[ "$MODEL" == Qwen3-VL-* ]]; then BV_MAX_NEW_TOKENS="$BV_MAX_NEW_TOKENS_QWEN3VL"; BV_BATCHSIZE="$BV_BATCHSIZE_QWEN3VL"; fi
    ARGS=(--temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P"
      --repetition-penalty "$REPETITION_PENALTY" --bench babyvision --split train
      --data-root "$DATA_DIR" --cache-root "$HF_HUB_CACHE" --batchsize "$BV_BATCHSIZE" --gpu_ids "$GPU_IDS"
      --max-new-tokens "$BV_MAX_NEW_TOKENS" --thinking "$THINKING" --dtype bfloat16
      --attn-implementation "$BV_ATTN" --random-seed "$BV_SEED")
    [[ "$DO_SAMPLE" == true ]] && ARGS+=(--do-sample)
    if [[ -n "$BV_RANK_RANGE" ]]; then
      ARGS+=(--case_ids $("${PY[@]}" bv_order --output "$OUTPUT_DIR" --source "$BV_REFERENCE" --rank-range $BV_RANK_RANGE))
    elif [[ -n "$BV_TAIL_FROM" ]]; then
      ARGS+=(--case_ids $("${PY[@]}" bv_order --output "$OUTPUT_DIR" --source "$BV_REFERENCE" --tail-from "$BV_TAIL_FROM"))
    elif [[ "$BV_ORDER_BY_REFERENCE" == true ]]; then
      ARGS+=(--case_ids $("${PY[@]}" bv_order --output "$OUTPUT_DIR" --source "$BV_REFERENCE"))
    fi
    OUT="$OUTPUT_DIR/babyvision/${MODEL_SLUG}_babyvision-thinking${THINKING}-${DECODING_PRESET}-max${BV_MAX_NEW_TOKENS}-seed${BV_SEED}.jsonl"
    [[ -n "$BV_TAIL_FROM" ]] && OUT="$OUTPUT_DIR/babyvision/parts/$(basename "$OUT" .jsonl).tail${BV_TAIL_FROM}.jsonl"
    [[ -n "$BV_RANK_RANGE" ]] && OUT="$OUTPUT_DIR/babyvision/parts/$(basename "$OUT" .jsonl).ranks${BV_RANK_RANGE// /-}.jsonl"
    [[ -f "$OUT" ]] && ARGS+=(--resume)
    exec uv run --frozen python -m eva.evaluation --model "$MODEL" --output "$OUT" "${ARGS[@]}" ;;
  bv_extension)
    THINKING="$BV_THINKING"; select_decoding_preset
    RUN="$OUTPUT_DIR/babyvision/${MODEL_SLUG}_babyvision-thinking${THINKING}-${DECODING_PRESET}-max${BV_MAX_NEW_TOKENS_QWEN3VL}-seed${BV_SEED}.jsonl"
    EXT_RANGE=(); EXT_SUFFIX=""
    if [[ -n "$BV_RANK_RANGE" ]]; then EXT_RANGE=(--rank-range $BV_RANK_RANGE); EXT_SUFFIX=".ids${BV_RANK_RANGE// /-}"; fi
    OUT="$OUTPUT_DIR/babyvision/extension/$(basename "$RUN" .jsonl).ext${BV_EXTENSION_MAX_NEW_TOKENS}${EXT_SUFFIX}.jsonl"
    ARGS=(--temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P"
      --repetition-penalty "$REPETITION_PENALTY" --bench babyvision --split train
      --data-root "$DATA_DIR" --cache-root "$HF_HUB_CACHE" --batchsize "$BV_BATCHSIZE_QWEN3VL" --gpu_ids "$GPU_IDS"
      --max-new-tokens "$BV_EXTENSION_MAX_NEW_TOKENS" --thinking "$THINKING" --dtype bfloat16
      --attn-implementation "$BV_ATTN" --random-seed "$BV_SEED" --do-sample
      --case_ids $("${PY[@]}" bv_truncated --output "$OUTPUT_DIR" --source "$RUN" --spotcheck-n "$BV_EXTENSION_N" --seed "$SEED" "${EXT_RANGE[@]}"))
    [[ -f "$OUT" ]] && ARGS+=(--resume)
    mkdir -p "$(dirname "$OUT")"
    exec uv run --frozen python -m eva.evaluation --model "$MODEL" --output "$OUT" "${ARGS[@]}" ;;
  bv_merge)
    THINKING="$BV_THINKING"; select_decoding_preset
    BV_MAX_NEW_TOKENS="$BV_MAX_NEW_TOKENS_QWEN35"; [[ "$MODEL" == Qwen3-VL-* ]] && BV_MAX_NEW_TOKENS="$BV_MAX_NEW_TOKENS_QWEN3VL"
    exec "${PY[@]}" bv_merge --output "$OUTPUT_DIR" \
      --source "$OUTPUT_DIR/babyvision/${MODEL_SLUG}_babyvision-thinking${THINKING}-${DECODING_PRESET}-max${BV_MAX_NEW_TOKENS}-seed${BV_SEED}.jsonl" ;;
  decomp_prepare)
    exec "${PY[@]}" decomp_prepare --source "$DECOMP_SOURCE" --output "$DECOMP_DIR/$MODEL_SLUG" ;;
  decomp_run)
    ARGS=(); [[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec "${PY[@]}" decomp_run --output "$DECOMP_DIR/$MODEL_SLUG" "${COMMON[@]}" "${ARGS[@]}" ;;
  t3c5_run)
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec "${PY[@]}" t3c5_run --output "$DECOMP_DIR/$MODEL_SLUG" --source "$DECOMP_SOURCE" \
      --supplement "$T3_SUPPLEMENT" "${COMMON[@]}" ;;
  spotcheck)
    TARGET="$DECOMP_DIR/$MODEL_SLUG"; [[ "$SPOTCHECK_KIND" == refcoco ]] && TARGET="$OUTPUT_DIR/refcoco"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec "${PY[@]}" spotcheck --output "$TARGET" "${COMMON[@]}" \
      --spotcheck-kind "$SPOTCHECK_KIND" --spotcheck-n "$SPOTCHECK_N" ;;
  batch_validate)
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec "${PY[@]}" batch_validate --source "$DECOMP_SOURCE" \
      --output "$OUTPUT_DIR/checks" "${COMMON[@]}" --per-cell "$VALIDATE_PER_CELL" ;;
  decomp_score)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition score --output "$DECOMP_DIR/$MODEL_SLUG" ;;
  decomp_analyze)
    exec uv run --frozen python -m analysis_tools.behavior.decomposition_analysis --output "$DECOMP_DIR/$MODEL_SLUG" ;;
  refcoco_build)
    : "${REFCOCO_DIR:?Set REFCOCO_DIR in .env.}"
    exec "${PY[@]}" refcoco_build --refcoco "$REFCOCO_DIR" --output "$OUTPUT_DIR/refcoco" \
      --per-stratum "$REFCOCO_PER_STRATUM" --seed "$SEED" ;;
  refcoco_run)
    CUDA_VISIBLE_DEVICES="$GPU_IDS" exec "${PY[@]}" refcoco_run --output "$OUTPUT_DIR/refcoco" "${COMMON[@]}" ;;
  refcoco_score)
    exec "${PY[@]}" refcoco_score --output "$OUTPUT_DIR/refcoco" ;;
  analyze)
    exec uv run --frozen python -m analysis_tools.deepstack.deepstack_comparison_analysis --output "$OUTPUT_DIR" \
      --decomp-source "$DECOMP_SOURCE" --legacy-babyvision "$WORK_DIR/eval" --t3-supplement "$T3_SUPPLEMENT" ;;
  reasoning)
    exec uv run --frozen python -m analysis_tools.deepstack.deepstack_reasoning "$REASONING_STEP" --output "$OUTPUT_DIR" \
      --legacy-babyvision "$WORK_DIR/eval" --annotation "$BABYVISION_ANNOTATION" --data-root "$DATA_DIR" \
      --wrong-per-subtype "$MANUAL_WRONG_PER_SUBTYPE" --right-per-subtype "$MANUAL_RIGHT_PER_SUBTYPE" ;;
  *) echo "Unknown COMMAND: $COMMAND" >&2; exit 2 ;;
esac
