#!/usr/bin/env bash
# Experiment 7.2: interpretability-located fine-tuning of the coordinate readout
# (docs/analysis/application/results_2026-11-05_located_finetune.md; prediction in
#  docs/analysis/position/results_2026-11-05_readout_prior_localization.md).
#
# Usage: edit the parameters below, then run, e.g.
#   STAGES=link bash scripts/analysis/application/coord_locate.sh                         # CPU
#   STAGES=bvprep bash scripts/analysis/application/coord_locate.sh                       # CPU
#   STAGES=fixprep bash scripts/analysis/application/coord_locate.sh                      # CPU (format-repair data)
#   CONDITION=<cond> STAGES=train,digits,coord,general,babyvision GPU_IDS=3 bash ...      # one condition
#   STAGES=analysis bash scripts/analysis/application/coord_locate.sh                     # CPU
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/load_env.sh"

MODEL="Qwen3.5-9B"
GPU_IDS="${GPU_IDS:-}"                     # an idle GPU claimed through the claim helper; required for GPU stages
RUN_NAME="2026-11-05_located_finetune"
STAGES="${STAGES:-link}"
CONDITION="${CONDITION:-base}"
# Training (3.13 settings: rank 8, alpha 32, lr 1e-4, 16 samples per step; conditions may override epochs / lr)
RANK=8
ALPHA=32
LR=1e-4
EPOCHS=1
STEP_SAMPLES=16
MICRO_TOKENS=4096
# Evaluation
TOKEN_BUDGET=16000
MAX_NEW_TOKENS=64
GENERAL_N=200
GENERAL_MAX_NEW=96
BATCH=24
BV_BATCH=8
BV_MAX_NEW=1024
EVAL_POSITIONS="${EVAL_POSITIONS:-}"        # digits stage only: restrict the adapter to question / prefix tokens

OUTPUT_DIR="$WORK_DIR/analysis/application/$RUN_NAME"
FINETUNE_RUN="$WORK_DIR/analysis/position/2026-11-03_targeted_finetune_r2"
ROUND1_DIR="$WORK_DIR/analysis/position/2026-10-15_coordinate_value"
FEATURE_DIR="$DATA_DIR/scratch/analysis/application/$RUN_NAME"
BV_ANNOTATION="$DATA_DIR/BabyVision_annotations/find_the_different_grid_v1.json"
VCOT_GQA="${DATA_DIR_PUBLIC:-$DATA_DIR}/Visual-CoT/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
MMVP_CSV="$(ls "$HF_HUB_CACHE"/datasets--MMVP--MMVP/snapshots/*/Questions.csv 2>/dev/null | head -1 || true)"
FIGURE_DIR="$ROOT_DIR/docs/analysis/application/figures/located_finetune"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

if [[ "$STAGES" == "analysis" ]]; then
  # Small dense solves: one BLAS thread is much faster than many on the shared, busy CPU.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  uv run --frozen python -m analysis_tools.application.coord_locate_analysis \
    --run "$OUTPUT_DIR" --features "$FEATURE_DIR" --figures "$FIGURE_DIR"
  exit 0
fi
if [[ "$STAGES" != "link" && "$STAGES" != "bvprep" && "$STAGES" != "fixprep" ]]; then
  [[ -n "$GPU_IDS" ]] || { echo "GPU_IDS must be set to a claimed idle GPU" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
uv run --frozen python -m analysis_tools.application.coord_locate_run \
  --model "$MODEL" --output "$OUTPUT_DIR" --stages "$STAGES" --condition "$CONDITION" --finetune-run "$FINETUNE_RUN" \
  --round1 "$ROUND1_DIR" --features "$FEATURE_DIR" --data-dir "$DATA_DIR" --bv-annotation "$BV_ANNOTATION" \
  --vcot-gqa "$VCOT_GQA" --mmvp-csv "$MMVP_CSV" --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --epochs "$EPOCHS" \
  --step-samples "$STEP_SAMPLES" --micro-tokens "$MICRO_TOKENS" --token-budget "$TOKEN_BUDGET" \
  --max-new-tokens "$MAX_NEW_TOKENS" --general-n "$GENERAL_N" --general-max-new "$GENERAL_MAX_NEW" --batch "$BATCH" \
  --bv-batch "$BV_BATCH" --bv-max-new "$BV_MAX_NEW" ${EVAL_POSITIONS:+--eval-positions "$EVAL_POSITIONS"} \
  2>&1 | grep --line-buffered -v 'causal_conv1d' | tee -a "$OUTPUT_DIR/run_${CONDITION}.log"
