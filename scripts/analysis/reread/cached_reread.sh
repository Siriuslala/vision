#!/usr/bin/env bash
# Experiment 5.2: native zoom tool vs cached re-read (docs/analysis/reread/design_cached_reread.md).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-2}"
RUN_NAME="2026-10-20_cached_reread"
# comma list of: verify, textvqa, gqa, mmvp, pope, babyvision, syn_color, syn_gap, syn_loc, timing;
# or "analysis" (CPU metrics and figures)
STAGES="${STAGES:-verify,syn_loc,syn_color,syn_gap,gqa,textvqa,mmvp,pope,babyvision,timing}"
SEED=20261020
N_NATURAL=200                         # questions per natural benchmark (MMVP: first 200 = 100 pairs)
N_BV=163                              # BabyVision Fine-grained Discrimination (all 163)
N_SYNTH=48                            # images per synthetic task (20x20 grid)
GRID_N=20
LOC_K=5                               # oracle window of syn_loc: k x k instance cells containing the target
GQA_MAX_AREA=0.02                     # GQA (Visual-CoT val) evidence box area / image area upper bound
MAX_PIXELS=$((512 * 512))             # natural-image input budget (TextVQA, GQA, MMVP, POPE)
BV_MAX_PIXELS=$((1024 * 1024))        # BabyVision input budget (native size for almost all images)
MAX_CALL=160                          # budget of the first tool-call turn
MAX_CALLS=3                           # tool calls allowed per question (native multi-round protocol)
MAX_NEW=256                           # answer budget
MAX_NEW_BV=256
CHUNK=16                              # cases whose features are held at once
TOKEN_BUDGET="${TOKEN_BUDGET:-24000}"                   # padded prompt tokens per decoding batch
MAX_BATCH=16
MAX_MEMORY="${MAX_MEMORY:-}"          # optional per-GPU weight budgets in GiB (e.g. "14 6") to split layers over GPU_IDS
CONDS="${CONDS:-}"                    # optional explicit condition list (supplementary runs, e.g. "h9r o_h9r")
TIMING_N=20                           # batch-size-1 timing cases per timing benchmark
TIMING_BENCHES="textvqa gqa syn_loc"
VCOT_GQA="$DATA_DIR_PUBLIC/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv | head -1)"
OUTPUT_DIR="$WORK_DIR/analysis/reread/$RUN_NAME"
FIGURE_DIR="$ROOT_DIR/docs/analysis/reread/figures/cached_reread"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"
if [[ "$STAGES" == "analysis" ]]; then
  exec uv run --frozen python -m analysis_tools.reread.cached_reread_analysis --run "$OUTPUT_DIR" --figures "$FIGURE_DIR"
fi
exec uv run --frozen python -m analysis_tools.reread.cached_reread_run --output "$OUTPUT_DIR" --model "$MODEL" \
  --stages "$STAGES" --seed "$SEED" --data-dir "$DATA_DIR" --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" \
  --n-natural "$N_NATURAL" --n-bv "$N_BV" --n-synth "$N_SYNTH" --grid-n "$GRID_N" --loc-k "$LOC_K" \
  --gqa-max-area "$GQA_MAX_AREA" --max-pixels "$MAX_PIXELS" --bv-max-pixels "$BV_MAX_PIXELS" \
  --max-call "$MAX_CALL" --max-calls "$MAX_CALLS" --max-new "$MAX_NEW" --max-new-bv "$MAX_NEW_BV" --chunk "$CHUNK" \
  --token-budget "$TOKEN_BUDGET" --max-batch "$MAX_BATCH" --timing-n "$TIMING_N" --timing-benches $TIMING_BENCHES \
  ${MAX_MEMORY:+--max-memory $MAX_MEMORY} ${CONDS:+--conds $CONDS}
