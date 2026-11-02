#!/usr/bin/env bash
# Experiment 5.3: cache-only re-read without re-encoding (docs/analysis/reread/results_2026-10-21_cache_only.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="2026-10-21_cache_only"
# comma list of: verify, lean_<bench>, native_<bench>, timing; or "analysis" (CPU statistics and figures).
# <bench>: textvqa, gqa, mmvp, pope, syn_color, syn_gap, syn_loc (native: natural benchmarks only)
STAGES="${STAGES:-verify,lean_textvqa,lean_gqa,lean_mmvp,lean_pope,lean_syn_color,lean_syn_gap,lean_syn_loc,timing}"
SEED=20261020                         # same seed as experiment 5.2, hence the same questions
N_NATURAL=200
N_SYNTH=48
GRID_N=20
LOC_K=5
GQA_MAX_AREA=0.02
MAX_PIXELS=$((512 * 512))             # natural-image input budget, as in 5.2
MAX_CALL=40                           # lean protocol: tokens for the forced bbox call
MAX_NEW=32                            # lean protocol: tokens after the prefilled "Final answer:"
NATIVE_MAX_CALL=160                   # native protocol (5.2 settings)
NATIVE_MAX_CALLS=3
NATIVE_MAX_NEW=256
CHUNK=8
TOKEN_BUDGET="${TOKEN_BUDGET:-24000}"
MAX_BATCH=16
CONDS="${CONDS:-}"                    # optional explicit condition list
MAX_MEMORY="${MAX_MEMORY:-}"          # optional per-GPU weight budgets in GiB (e.g. "14 6") to split layers over GPU_IDS
TIMING_N=20
TIMING_BENCHES="textvqa gqa mmvp pope syn_loc"
VCOT_GQA="$DATA_DIR_PUBLIC/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv | head -1)"
OUTPUT_DIR="$WORK_DIR/analysis/reread/$RUN_NAME"
REF_5_2="$WORK_DIR/analysis/reread/2026-10-20_cached_reread"
FIGURE_DIR="$ROOT_DIR/docs/analysis/reread/figures/cache_only"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "analysis" ]]; then
  exec uv run --frozen python -m analysis_tools.reread.cache_only_analysis --run "$OUTPUT_DIR" --ref "$REF_5_2" \
    --figures "$FIGURE_DIR"
fi
exec uv run --frozen python -m analysis_tools.reread.cache_only_run --output "$OUTPUT_DIR" --model "$MODEL" \
  --stages "$STAGES" --seed "$SEED" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" \
  --n-natural "$N_NATURAL" --n-synth "$N_SYNTH" --grid-n "$GRID_N" --loc-k "$LOC_K" --gqa-max-area "$GQA_MAX_AREA" \
  --max-pixels "$MAX_PIXELS" --max-call "$MAX_CALL" --max-new "$MAX_NEW" --native-max-call "$NATIVE_MAX_CALL" \
  --native-max-calls "$NATIVE_MAX_CALLS" --native-max-new "$NATIVE_MAX_NEW" --chunk "$CHUNK" \
  --token-budget "$TOKEN_BUDGET" --max-batch "$MAX_BATCH" --timing-n "$TIMING_N" --timing-benches $TIMING_BENCHES \
  ${CONDS:+--conds $CONDS} ${MAX_MEMORY:+--max-memory $MAX_MEMORY}
